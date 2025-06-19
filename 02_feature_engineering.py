import pandas as pd
import numpy as np
import os
import logging
from collections import defaultdict
from tqdm import tqdm # Import tqdm for progress bar

# Setup basic logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# --- ELO Calculation Functions ---
K_FACTOR = 32 # Standard K-factor for ELO
DEFAULT_ELO = 1500

def calculate_expected_score(rating1, rating2):
    """Calculate the expected score (win probability) for player 1."""
    return 1.0 / (1.0 + 10.0 ** ((rating2 - rating1) / 400.0))

def update_elo(winner_elo, loser_elo):
    """Update ELO ratings based on match outcome."""
    expected_winner = calculate_expected_score(winner_elo, loser_elo)
    expected_loser = 1 - expected_winner
    new_winner_elo = winner_elo + K_FACTOR * (1 - expected_winner)
    new_loser_elo = loser_elo + K_FACTOR * (0 - expected_loser)
    return new_winner_elo, new_loser_elo
# --- End ELO Functions ---

def load_merged_data(input_path='data/merged_atp_data.csv'):
    """Loads the merged ATP data."""
    if not os.path.exists(input_path):
        logging.error(f"Input file not found: {input_path}")
        return None
    try:
        logging.info(f"Loading merged data from: {input_path}")
        df = pd.read_csv(input_path, low_memory=False, parse_dates=['tourney_date'])
        logging.info(f"Data loaded successfully. Shape: {df.shape}")
        return df
    except Exception as e:
        logging.error(f"Error loading data from {input_path}: {e}")
        return None

def handle_missing_values(df):
    """Handles missing values in the DataFrame."""
    initial_shape = df.shape
    logging.info(f"Starting missing value handling. Initial shape: {initial_shape}")
    
    # Drop columns with excessive NaNs
    cols_to_drop = ['winner_seed', 'loser_seed', 'winner_entry', 'loser_entry',
                    # Drop detailed match stats as we won't use them for rolling features now
                    'w_ace', 'w_df', 'w_svpt', 'w_1stIn', 'w_1stWon', 'w_2ndWon', 'w_SvGms', 'w_bpSaved', 'w_bpFaced',
                    'l_ace', 'l_df', 'l_svpt', 'l_1stIn', 'l_1stWon', 'l_2ndWon', 'l_SvGms', 'l_bpSaved', 'l_bpFaced']
    cols_exist = [col for col in cols_to_drop if col in df.columns]
    df.drop(columns=cols_exist, inplace=True)
    logging.info(f"Dropped columns: {cols_exist}")

    # Drop rows with NaNs in critical columns
    critical_cols = ['surface', 'winner_rank', 'loser_rank', 'winner_age', 'loser_age', 'loser_hand', 'winner_ht', 'loser_ht', 'minutes']
    critical_cols_exist = [col for col in critical_cols if col in df.columns]
    rows_before_drop = len(df)
    df.dropna(subset=critical_cols_exist, inplace=True)
    rows_after_drop = len(df)
    logging.info(f"Dropped {rows_before_drop - rows_after_drop} rows with NaNs in critical columns: {critical_cols_exist}.")
    logging.info(f"Shape after row drops: {df.shape}")

    # Impute rank points (only remaining column needing imputation likely)
    impute_zero_cols = ['winner_rank_points', 'loser_rank_points']
    for col in impute_zero_cols:
        if col in df.columns:
            if df[col].isnull().any():
                logging.info(f"Imputing NaNs in '{col}' with 0")
                df[col] = df[col].fillna(0)

    final_shape = df.shape
    logging.info(f"Missing value handling complete. Final shape: {final_shape}")
    if df.isnull().sum().sum() > 0:
        logging.warning(f"Unexpected NaNs remain: {df.isnull().sum()[df.isnull().sum() > 0]}")
    else:
        logging.info("No remaining NaNs found.")
    return df

# --- Helper function for rolling WIN stats ONLY ---
def calculate_rolling_stats(df, windows):
    """Calculates rolling win averages for players."""
    logging.info("Calculating rolling WIN statistics...")
    df['match_id'] = df['tourney_date'].dt.strftime('%Y%m%d') + '-' + df['match_num'].astype(str)
    df['w_win'] = 1
    df['l_win'] = 0

    # Create Long Format for wins
    w_cols = {'match_id': 'match_id', 'tourney_date': 'tourney_date', 'winner_id': 'player_id', 'w_win': 'win'}
    l_cols = {'match_id': 'match_id', 'tourney_date': 'tourney_date', 'loser_id': 'player_id', 'l_win': 'win'}
    df_w = df[list(w_cols.keys())].rename(columns=w_cols)
    df_l = df[list(l_cols.keys())].rename(columns=l_cols)
    df_long = pd.concat([df_w, df_l], ignore_index=True)
    df_long.sort_values(by=['player_id', 'tourney_date', 'match_id'], inplace=True)
    logging.info(f"Long format created for wins. Shape: {df_long.shape}")

    # Add this after creating match_id
    duplicate_match_ids = df['match_id'].duplicated().sum()
    if duplicate_match_ids > 0:
        logging.warning(f"Found {duplicate_match_ids} duplicate match_ids!")
        # Make them unique by adding an extra identifier
        df['match_id'] = df['match_id'] + '-' + df.index.astype(str)

    # Calculate Rolling Win Averages
    rolling_results = {} 
    grouped = df_long.groupby('player_id')
    for window in tqdm(windows, desc="Calculating Rolling Win Windows"):
        col_name = f'rolling_win_last_{window}'
        rolling_mean = grouped['win'].rolling(window=window, min_periods=1).mean().shift(1)
        rolling_results[col_name] = rolling_mean.reset_index(level=0, drop=True)

    df_rolling_stats = pd.DataFrame(rolling_results, index=df_long.index)
    df_long = pd.concat([df_long, df_rolling_stats], axis=1)
    logging.info("Rolling win averages calculated.")

    # Merge Rolling Stats back
    df_merged = pd.merge(df, df_long[df_long.index.isin(df_w.index)][['match_id'] + list(df_rolling_stats.columns)], on='match_id', how='left')
    rename_w = {col: f'winner_{col}' for col in df_rolling_stats.columns}
    df_merged.rename(columns=rename_w, inplace=True)
    df_merged = pd.merge(df_merged, df_long[df_long.index.isin(df_l.index)][['match_id'] + list(df_rolling_stats.columns)], on='match_id', how='left')
    rename_l = {col: f'loser_{col}' for col in df_rolling_stats.columns}
    df_merged.rename(columns=rename_l, inplace=True)

    rolling_cols = list(rename_w.values()) + list(rename_l.values())
    for col in rolling_cols:
        if col in df_merged.columns:
            df_merged[col].fillna(0, inplace=True)
            
    df_merged.drop(columns=['match_id', 'w_win', 'l_win'], inplace=True, errors='ignore')
    logging.info("Rolling win stats merged back.")
    return df_merged

def calculate_h2h_stats(df):
    """Calculates rolling head-to-head (H2H) statistics to avoid data leakage."""
    logging.info("Calculating rolling H2H statistics...")
    # Use defaultdict for convenient handling of new player pairs
    h2h_stats = defaultdict(lambda: defaultdict(int))
    h2h_surface_stats = defaultdict(lambda: defaultdict(lambda: defaultdict(int)))

    # Lists to store pre-match H2H stats for each match
    w_h2h_wins, l_h2h_wins, h2h_matches = [], [], []
    w_surface_h2h_wins, l_surface_h2h_wins, surface_h2h_matches = [], [], []

    # Data must be sorted chronologically before this function
    for _, row in tqdm(df.iterrows(), total=df.shape[0], desc="Calculating H2H Stats"):
        winner_id, loser_id, surface = row['winner_id'], row['loser_id'], row['surface']
        
        # Use sorted IDs for a consistent key regardless of who is winner/loser
        p1, p2 = sorted((winner_id, loser_id))

        # --- Overall H2H ---
        p1_wins = h2h_stats[(p1, p2)][p1]
        p2_wins = h2h_stats[(p1, p2)][p2]
        
        # Append pre-match stats
        h2h_matches.append(p1_wins + p2_wins)
        if winner_id == p1:
            w_h2h_wins.append(p1_wins)
            l_h2h_wins.append(p2_wins)
        else: # winner_id == p2
            w_h2h_wins.append(p2_wins)
            l_h2h_wins.append(p1_wins)
            
        # Update stats for the next match
        h2h_stats[(p1, p2)][winner_id] += 1

        # --- Surface-specific H2H ---
        if pd.notna(surface):
            p1_surface_wins = h2h_surface_stats[(p1, p2)][surface][p1]
            p2_surface_wins = h2h_surface_stats[(p1, p2)][surface][p2]
            
            surface_h2h_matches.append(p1_surface_wins + p2_surface_wins)
            if winner_id == p1:
                w_surface_h2h_wins.append(p1_surface_wins)
                l_surface_h2h_wins.append(p2_surface_wins)
            else: # winner_id == p2
                w_surface_h2h_wins.append(p2_surface_wins)
                l_surface_h2h_wins.append(p1_surface_wins)

            # Update stats for the next match on this surface
            h2h_surface_stats[(p1, p2)][surface][winner_id] += 1
        else:
            # If surface is NaN, append NaN for surface stats
            w_surface_h2h_wins.append(np.nan)
            l_surface_h2h_wins.append(np.nan)
            surface_h2h_matches.append(np.nan)

    # Assign calculated stats back to the DataFrame
    df['winner_h2h_wins'] = w_h2h_wins
    df['loser_h2h_wins'] = l_h2h_wins
    df['h2h_matches'] = h2h_matches
    df['winner_surface_h2h_wins'] = w_surface_h2h_wins
    df['loser_surface_h2h_wins'] = l_surface_h2h_wins
    df['surface_h2h_matches'] = surface_h2h_matches

    # Calculate win percentages, handling division by zero for pairs with no prior matches
    df['winner_h2h_win_perc'] = (df['winner_h2h_wins'] / df['h2h_matches']).fillna(0)
    df['loser_h2h_win_perc'] = (df['loser_h2h_wins'] / df['h2h_matches']).fillna(0)
    df['winner_surface_h2h_win_perc'] = (df['winner_surface_h2h_wins'] / df['surface_h2h_matches']).fillna(0)
    df['loser_surface_h2h_win_perc'] = (df['loser_surface_h2h_wins'] / df['surface_h2h_matches']).fillna(0)

    # Clean up intermediate columns we don't need for restructuring
    df.drop(columns=['winner_h2h_wins', 'loser_h2h_wins', 'h2h_matches', 
                     'winner_surface_h2h_wins', 'loser_surface_h2h_wins', 'surface_h2h_matches'],
            inplace=True)
            
    logging.info("H2H statistics calculated and added to DataFrame.")
    return df

def engineer_features(df):
    """Engineers features: ELO, Surface ELO, Rolling Win %."""
    logging.info(f"Starting feature engineering (Simplified). Input shape: {df.shape}")

    # --- 0. Chronological Sort & Index --- 
    df.sort_values(by=['tourney_date', 'match_num'], inplace=True)
    df.reset_index(inplace=True) # Keep index for linking
    logging.info("Data sorted by 'tourney_date' and 'match_num'.")

    # --- 1. Calculate Rolling Win Stats --- 
    rolling_windows = [5, 25, 100] 
    df = calculate_rolling_stats(df, rolling_windows)
    
    # --- 2. Calculate H2H stats ---
    df = calculate_h2h_stats(df)
    
    # --- 3. Calculate ELO & Surface ELO --- 
    logging.info("Calculating pre-match ELO and Surface ELO ratings...")
    player_elos = defaultdict(lambda: DEFAULT_ELO)
    player_surface_elos = defaultdict(lambda: defaultdict(lambda: DEFAULT_ELO))
    winner_prematch_elos = []
    loser_prematch_elos = []
    winner_prematch_surface_elos = []
    loser_prematch_surface_elos = []
    surfaces = df['surface'].dropna().unique()

    for index, row in tqdm(df.iterrows(), total=df.shape[0], desc="Calculating ELOs"):
        winner_id = row['winner_id']
        loser_id = row['loser_id']
        surface = row['surface']
        winner_elo_before = player_elos[winner_id]
        loser_elo_before = player_elos[loser_id]
        winner_prematch_elos.append(winner_elo_before)
        loser_prematch_elos.append(loser_elo_before)
        new_winner_elo, new_loser_elo = update_elo(winner_elo_before, loser_elo_before)
        player_elos[winner_id] = new_winner_elo
        player_elos[loser_id] = new_loser_elo
        if pd.notna(surface) and surface in surfaces:
            winner_surface_elo_before = player_surface_elos[winner_id][surface]
            loser_surface_elo_before = player_surface_elos[loser_id][surface]
            winner_prematch_surface_elos.append(winner_surface_elo_before)
            loser_prematch_surface_elos.append(loser_surface_elo_before)
            new_winner_surface_elo, new_loser_surface_elo = update_elo(winner_surface_elo_before, loser_surface_elo_before)
            player_surface_elos[winner_id][surface] = new_winner_surface_elo
            player_surface_elos[loser_id][surface] = new_loser_surface_elo
        else:
             winner_prematch_surface_elos.append(np.nan)
             loser_prematch_surface_elos.append(np.nan)

    df['winner_prematch_elo'] = pd.Series(winner_prematch_elos, index=df.index)
    df['loser_prematch_elo'] = pd.Series(loser_prematch_elos, index=df.index)
    df['winner_prematch_surface_elo'] = pd.Series(winner_prematch_surface_elos, index=df.index)
    df['loser_prematch_surface_elo'] = pd.Series(loser_prematch_surface_elos, index=df.index)
    logging.info("Pre-match ELO calculations complete.")

    # --- 4. Restructure Data --- 
    logging.info("Restructuring data...")
    df_p1 = pd.DataFrame()
    df_p2 = pd.DataFrame()
    swap_indices = np.random.choice([True, False], size=len(df), p=[0.5, 0.5])

    # Define columns to transfer (Reduced set)
    player_cols_base = {
        'id': ('winner_id', 'loser_id'), 'name': ('winner_name', 'loser_name'),
        'hand': ('winner_hand', 'loser_hand'), 'ht': ('winner_ht', 'loser_ht'),
        'ioc': ('winner_ioc', 'loser_ioc'), 'age': ('winner_age', 'loser_age'),
        'rank': ('winner_rank', 'loser_rank'), 'rank_points': ('winner_rank_points', 'loser_rank_points'),
        'prematch_elo': ('winner_prematch_elo', 'loser_prematch_elo'),
        'prematch_surface_elo': ('winner_prematch_surface_elo', 'loser_prematch_surface_elo'),
        'h2h_win_perc': ('winner_h2h_win_perc', 'loser_h2h_win_perc'),
        'surface_h2h_win_perc': ('winner_surface_h2h_win_perc', 'loser_surface_h2h_win_perc')
    }
    # Add ONLY rolling WIN stat columns 
    for window in rolling_windows:
         col_name_base = f'rolling_win_last_{window}'
         player_cols_base[col_name_base] = (f'winner_{col_name_base}', f'loser_{col_name_base}')
              
    player_cols = player_cols_base 

    for key, (win_col, lose_col) in player_cols.items():
        if win_col in df.columns and lose_col in df.columns:
            df_p1[f'p1_{key}'] = np.where(swap_indices, df[lose_col], df[win_col])
            df_p2[f'p2_{key}'] = np.where(swap_indices, df[win_col], df[lose_col])
        else:
            logging.warning(f"Skipping restructuring for key '{key}' as cols not found.")
            df_p1[f'p1_{key}'] = np.nan 
            df_p2[f'p2_{key}'] = np.nan
            
    common_cols = ['tourney_id', 'tourney_name', 'surface', 'draw_size', 'tourney_level', 'tourney_date', 'match_num', 'score', 'best_of', 'round', 'minutes']
    common_cols_exist = [col for col in common_cols if col in df.columns]
    df_feat = pd.concat([df[['index'] + common_cols_exist], df_p1, df_p2], axis=1)
    df_feat['RESULT'] = np.where(swap_indices, 0, 1)
    logging.info(f"Restructuring complete. Shape: {df_feat.shape}")

    # --- 5. Calculate Difference Features (Reduced set) ---
    logging.info("Calculating difference features...")
    diff_pairs = {
        'ATP_RANK_DIFF': ('p1_rank', 'p2_rank'), 'ATP_POINTS_DIFF': ('p1_rank_points', 'p2_rank_points'),
        'AGE_DIFF': ('p1_age', 'p2_age'), 'HEIGHT_DIFF': ('p1_ht', 'p2_ht'),
        'ELO_DIFF': ('p1_prematch_elo', 'p2_prematch_elo'), 
        'ELO_SURFACE_DIFF': ('p1_prematch_surface_elo', 'p2_prematch_surface_elo'),
        'H2H_WIN_PERC_DIFF': ('p1_h2h_win_perc', 'p2_h2h_win_perc'),
        'SURFACE_H2H_WIN_PERC_DIFF': ('p1_surface_h2h_win_perc', 'p2_surface_h2h_win_perc')
    }
    # Add rolling WIN differences
    for window in rolling_windows:
        p1_col = f'p1_rolling_win_last_{window}'
        p2_col = f'p2_rolling_win_last_{window}'
        diff_name = f'WIN_LAST_{window}_DIFF'
        diff_pairs[diff_name] = (p1_col, p2_col)

    calculated_diffs = []
    for diff_name, (col1, col2) in diff_pairs.items():
         cols_to_check = [col1, col2]
         valid_cols = True
         for col in cols_to_check:
             if col not in df_feat.columns:
                 logging.warning(f"Column {col} not found for diff {diff_name}. Skipping.")
                 df_feat[diff_name] = np.nan 
                 valid_cols = False
                 break
             df_feat[col] = pd.to_numeric(df_feat[col], errors='coerce')
             
         if valid_cols:
            df_feat[diff_name] = df_feat[col1] - df_feat[col2]
            calculated_diffs.append(diff_name)
            if df_feat[diff_name].isnull().any():
                logging.warning(f"NaNs found in {diff_name} after calculation. Imputing with 0.")
                df_feat[diff_name].fillna(0, inplace=True)
                
    logging.info(f"Calculated difference features: {calculated_diffs}")

    # --- 6. Placeholders (None remaining) --- 
    logging.info("No remaining complex features to calculate in this version.")

    # Keep index column for linking
    logging.info("Feature engineering complete.")
    return df_feat

def select_and_encode_features(df):
    """Selects relevant features and encodes categorical ones."""
    logging.info(f"Starting feature selection and encoding. Input shape: {df.shape}")
    
    features_to_keep = ['RESULT', 'index']
    metadata_cols = ['tourney_level', 'surface', 'round', 'best_of', 'draw_size']
    # Exclude IOC, keep Hand
    identity_cols = ['p1_id', 'p1_name', 'p1_hand', 'p2_id', 'p2_name', 'p2_hand'] 
    
    # Add calculated difference features (Base, ELO, Rolling Win)
    diff_cols = [col for col in df.columns if '_DIFF' in col and not df[col].isnull().all()]
    logging.info(f"Adding {len(diff_cols)} calculated difference features.")

    # No remaining placeholders to keep track of
    placeholder_diff_cols = []
    
    all_cols_to_keep = features_to_keep + metadata_cols + identity_cols + diff_cols + placeholder_diff_cols
    columns_exist = [col for col in all_cols_to_keep if col in df.columns]
    if 'year' in df.columns: columns_exist.append('year')
    if 'month' in df.columns: columns_exist.append('month')
    columns_exist = list(dict.fromkeys(columns_exist)) 
    df_selected = df[columns_exist].copy()
    logging.info(f"Selected {len(columns_exist)} features.")
    
    # Encode Categorical Features
    cat_features = ['surface', 'round', 'tourney_level', 'p1_hand', 'p2_hand']
    cat_features_exist = [col for col in cat_features if col in df_selected.columns]
    for col in cat_features_exist:
        if df_selected[col].isna().sum() > 0:
            logging.warning(f"Found {df_selected[col].isna().sum()} NaN values in category '{col}'. Filling with 'Unknown'.")
            df_selected = df_selected.assign(**{col: df_selected[col].fillna('Unknown')})
    
    logging.info(f"One-hot encoding categorical features: {cat_features_exist}")
    index_before_encode = df_selected['index']
    df_to_encode = df_selected.drop(columns=['index']) # Drop index before dummy
    df_encoded_part = pd.get_dummies(df_to_encode, columns=cat_features_exist, dummy_na=False, drop_first=True)
    df_encoded = pd.concat([index_before_encode.reset_index(drop=True), df_encoded_part.reset_index(drop=True)], axis=1)
    logging.info(f"Encoding complete. Final shape: {df_encoded.shape}")
    
    # Add Year/Month if not present (using the original df passed into the function)
    if 'tourney_date' in df.columns and 'year' not in df_encoded.columns:
         logging.info("Adding year and month features from original tourney_date.")
         date_df = df[['index', 'tourney_date']].copy()
         date_df['year'] = pd.to_datetime(date_df['tourney_date']).dt.year
         date_df['month'] = pd.to_datetime(date_df['tourney_date']).dt.month
         df_encoded = pd.merge(df_encoded, date_df[['index', 'year', 'month']], on='index', how='left')

    # Remove Placeholder Columns (Should be none now, but keep for safety)
    cols_before_drop = df_encoded.shape[1]
    df_encoded.dropna(axis=1, how='all', inplace=True)
    cols_after_drop = df_encoded.shape[1]
    if cols_before_drop > cols_after_drop:
        logging.info(f"Dropped {cols_before_drop - cols_after_drop} columns containing only NaNs.")
    
    # Final NaN check
    nan_count = df_encoded.isna().sum().sum()
    if nan_count > 0:
        nan_cols = df_encoded.columns[df_encoded.isna().any()].tolist()
        logging.error(f"Unexpected NaNs remain after final processing! Count: {nan_count}. Columns: {nan_cols}")
    else:
        logging.info("Final feature set contains no NaNs.")
        
    return df_encoded

def save_features(df, output_path='data/features_atp_data.csv'):
    """Saves the engineered features DataFrame to a CSV file."""
    if df.empty:
        logging.warning("Feature DataFrame is empty. Not saving.")
        return

    output_dir = os.path.dirname(output_path)
    if output_dir and not os.path.exists(output_dir):
        os.makedirs(output_dir)
        logging.info(f"Created directory: {output_dir}")

    try:
        # Save index=False because we kept 'index' as a column
        df.to_csv(output_path, index=False) 
        logging.info(f"Engineered features saved to: {output_path}")
    except Exception as e:
        logging.error(f"Error saving features to {output_path}: {e}")

if __name__ == "__main__":
    logging.info("Starting ATP Feature Engineering Script")

    merged_df = load_merged_data()

    if merged_df is not None:
        df_cleaned = handle_missing_values(merged_df)
        df_engineered = engineer_features(df_cleaned)
        df_final = select_and_encode_features(df_engineered)
        save_features(df_final)
    else:
        logging.warning("Pipeline halted as merged data could not be loaded.")

    logging.info("Script finished.") 