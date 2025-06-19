import pandas as pd
import numpy as np
import joblib
import os
import logging
from datetime import datetime

# Setup basic logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# Global cache for merged data to avoid reloading
MERGED_DATA_CACHE = None

def load_merged_data_once(data_path='data/merged_atp_data.csv'):
    """Loads the merged data file once and caches it."""
    global MERGED_DATA_CACHE
    if MERGED_DATA_CACHE is not None:
        return MERGED_DATA_CACHE
    
    if not os.path.exists(data_path):
        logging.error(f"Merged data file not found: {data_path}")
        return None
    try:
        logging.info(f"Loading merged data from: {data_path} for player info lookup.")
        # Select only necessary columns to reduce memory usage
        cols_needed = ['tourney_date', 'winner_id', 'winner_name', 'winner_hand', 'winner_ht', 'winner_ioc', 'winner_age', 'winner_rank', 'winner_rank_points', 
                       'loser_id', 'loser_name', 'loser_hand', 'loser_ht', 'loser_ioc', 'loser_age', 'loser_rank', 'loser_rank_points']
        df = pd.read_csv(data_path, usecols=cols_needed, low_memory=False, parse_dates=['tourney_date'])
        df.sort_values(by='tourney_date', ascending=False, inplace=True) # Sort once
        MERGED_DATA_CACHE = df
        logging.info("Merged data loaded and cached.")
        return MERGED_DATA_CACHE
    except Exception as e:
        logging.error(f"Error loading merged data from {data_path}: {e}")
        return None

def load_model_and_feature_info(model_choice='XGBoost', model_dir='results/'):
    """Loads the specified trained model and extracts expected feature names."""
    model = None
    expected_features = None
    
    # Map choice to filename
    model_files = {
        "Decision Tree": "decision_tree_model.joblib",
        "Random Forest": "random_forest_model.joblib",
        "XGBoost": "xgboost_model.joblib"
    }
    
    if model_choice not in model_files:
        logging.error(f"Invalid model choice: '{model_choice}'. Choose from {list(model_files.keys())}")
        return None, None
        
    model_filename = model_files[model_choice]
    model_path = os.path.join(model_dir, model_filename)

    logging.info(f"Attempting to load model: {model_path}")
    if not os.path.exists(model_path):
        logging.error(f"Model file not found: {model_path}")
        return None, None
        
    try:
        model = joblib.load(model_path)
        logging.info(f"Model '{model_choice}' loaded successfully from: {model_path}")
        if hasattr(model, 'feature_names_in_'):
            expected_features = model.feature_names_in_.tolist()
            logging.info(f"Retrieved {len(expected_features)} expected feature names from model.")
        else:
            logging.error("Could not retrieve feature names from the loaded model!")
            return None, None 
    except Exception as e:
        logging.error(f"Error loading model from {model_path}: {e}")
        return None, None
        
    if model is None or expected_features is None:
        logging.error("Failed to load model or extract feature names.")
        return None, None
        
    return model, expected_features

def get_player_latest_info_from_merged(player_name, merged_df):
    """Gets the most recent available stats for a player from the MERGED dataset."""
    # Assumes merged_df is pre-sorted by date descending
    player_match = merged_df[(merged_df['winner_name'] == player_name) | (merged_df['loser_name'] == player_name)].iloc[0:1]
    
    if player_match.empty:
        logging.warning(f"No data found for player: {player_name} in merged data.")
        return None

    latest_match = player_match.iloc[0]
    info = {}
    if latest_match['winner_name'] == player_name:
        prefix_orig = 'winner_'
    else:
        prefix_orig = 'loser_'
        
    # Get info directly from original columns
    info['id'] = latest_match[f'{prefix_orig}id']
    info['name'] = latest_match[f'{prefix_orig}name']
    info['hand'] = latest_match[f'{prefix_orig}hand']
    info['ht'] = latest_match[f'{prefix_orig}ht']
    info['ioc'] = latest_match[f'{prefix_orig}ioc']
    info['age'] = latest_match[f'{prefix_orig}age']
    info['rank'] = latest_match[f'{prefix_orig}rank']
    info['rank_points'] = latest_match[f'{prefix_orig}rank_points']
    info['latest_match_date'] = latest_match['tourney_date']

    # Handle potential NaNs in retrieved info (important for calculations)
    numeric_cols = ['ht', 'age', 'rank', 'rank_points']
    for col in numeric_cols:
        if pd.isna(info[col]):
             logging.warning(f"Latest data for {player_name} has NaN for '{col}'. Using default/median if possible.")
             # Use a simple default or median from training data (median preferred but harder here)
             if col == 'ht': info[col] = 185 # Approx median
             elif col == 'age': info[col] = 25 # Approx median
             elif col == 'rank': info[col] = 500 # Penalize missing rank
             elif col == 'rank_points': info[col] = 0 
             
    if pd.isna(info['hand']): info['hand'] = 'R' # Default to Right
    if pd.isna(info['ioc']): info['ioc'] = 'UNK' # Unknown country

    logging.info(f"Found latest info for {player_name} from match date: {info['latest_match_date'].strftime('%Y-%m-%d') if pd.notna(info['latest_match_date']) else 'N/A'}")
    return info

def create_prediction_input(p1_info, p2_info, match_context, expected_features):
    """Creates a single-row DataFrame for prediction with the correct features."""
    input_data = {}

    # 1. Add match context features (like year, month, best_of, draw_size)
    now = datetime.now()
    input_data['year'] = match_context.get('year', now.year)
    input_data['month'] = match_context.get('month', now.month)
    input_data['best_of'] = match_context.get('best_of', 3)
    input_data['draw_size'] = match_context.get('draw_size', 32)
    input_data['minutes'] = np.nan # Imputed later if needed

    # 2. Calculate difference features
    input_data['ATP_RANK_DIFF'] = p1_info['rank'] - p2_info['rank']
    input_data['ATP_POINTS_DIFF'] = p1_info['rank_points'] - p2_info['rank_points']
    input_data['AGE_DIFF'] = p1_info['age'] - p2_info['age'] 
    input_data['HEIGHT_DIFF'] = p1_info['ht'] - p2_info['ht']

    # 3. Create DataFrame and handle categorical features
    pred_df = pd.DataFrame([input_data])
    pred_df['surface'] = match_context.get('surface', 'Hard')
    pred_df['round'] = match_context.get('round', 'R32')
    pred_df['tourney_level'] = match_context.get('tourney_level', 'A')
    pred_df['p1_hand'] = p1_info['hand']
    pred_df['p2_hand'] = p2_info['hand']
    pred_df['p1_ioc'] = p1_info['ioc']
    pred_df['p2_ioc'] = p2_info['ioc']
    
    cat_features = ['surface', 'round', 'tourney_level', 'p1_hand', 'p2_hand', 'p1_ioc', 'p2_ioc']
    # Ensure all categorical columns exist before encoding
    for col in cat_features:
        if col not in pred_df.columns:
             pred_df[col] = np.nan # Add if missing, will use default logic
             logging.warning(f"Categorical column '{col}' was missing, added with NaN.")
             
    # Handle potential NaNs in categorical columns before encoding
    pred_df[cat_features] = pred_df[cat_features].fillna({
        'surface': 'Hard', 'round': 'R32', 'tourney_level': 'A',
        'p1_hand': 'R', 'p2_hand': 'R', 'p1_ioc': 'UNK', 'p2_ioc': 'UNK'
    })
             
    pred_encoded = pd.get_dummies(pred_df, columns=cat_features, dummy_na=False, drop_first=True)

    # Align columns with the model's expected features
    final_pred_input = pd.DataFrame(columns=expected_features)
    # Use concat instead of direct assignment to handle different dtypes better
    final_pred_input = pd.concat([final_pred_input, pred_encoded], axis=0)
    
    # Fill missing columns and NaNs
    for col in expected_features:
        if col not in final_pred_input.columns:
            final_pred_input[col] = 0
        if final_pred_input[col].isnull().any():
            if col == 'minutes':
                logging.warning("Imputing missing 'minutes' for prediction with 97 (example median).")
                final_pred_input[col] = final_pred_input[col].fillna(97)
            else:
                logging.warning(f"Imputing missing numeric feature '{col}' with 0 for prediction.")
                final_pred_input[col] = final_pred_input[col].fillna(0)

    # Ensure column order and select only expected features
    final_pred_input = final_pred_input.reindex(columns=expected_features, fill_value=0)

    if final_pred_input.isnull().sum().sum() > 0:
        logging.error("NaNs detected in the final prediction input vector after reindexing!")
        logging.error(f"NaNs in columns: {final_pred_input.columns[final_pred_input.isnull().any()].tolist()}")
        return None

    return final_pred_input

def predict_match(player1_name, player2_name, model, merged_df, expected_features, match_context={}):
    """Predicts the outcome of a match between two players using merged data for info."""
    logging.info(f"--- Predicting Match: {player1_name} vs {player2_name} ---")
    logging.info(f"Match Context: {match_context}")

    # 1. Get Player Info from Merged Data
    p1_info = get_player_latest_info_from_merged(player1_name, merged_df)
    p2_info = get_player_latest_info_from_merged(player2_name, merged_df)

    # Improved check for missing player info
    if p1_info is None:
        print(f"\nError: Could not find recent data for player: {player1_name}")
        logging.error(f"Could not retrieve information for player: {player1_name}")
        return None
    if p2_info is None:
        print(f"\nError: Could not find recent data for player: {player2_name}")
        logging.error(f"Could not retrieve information for player: {player2_name}")
        return None

    # 2. Create Input Vector
    prediction_input = create_prediction_input(p1_info, p2_info, match_context, expected_features)

    if prediction_input is None:
        logging.error("Failed to create prediction input vector.")
        return None
        
    # 3. Predict Probability
    try:
        prediction_input = prediction_input.astype(np.float32)
        probability_p1_wins = model.predict_proba(prediction_input)[0][1]
        logging.info(f"Prediction successful. Probability {player1_name} wins: {probability_p1_wins:.4f}")
    except Exception as e:
        logging.error(f"Error during model prediction: {e}")
        logging.error(f"Input data columns ({prediction_input.shape[1]}): {prediction_input.columns.tolist()}")
        logging.error(f"Input data shape: {prediction_input.shape}")
        logging.error(f"Input data dtypes: {prediction_input.dtypes}")
        return None

    # 4. Format Output (Add Rank Info)
    p1_rank = p1_info.get('rank', 'N/A')
    p2_rank = p2_info.get('rank', 'N/A')
    print(f"\nPrediction for {player1_name} (Rank: {p1_rank}) vs {player2_name} (Rank: {p2_rank}):")
    print(f" Context: Surface={match_context.get('surface', 'Default')}, Round={match_context.get('round', 'Default')}, Best of={match_context.get('best_of', 'Default')}")
    print(f" -> {player1_name} has a {probability_p1_wins*100:.1f}% chance of winning.")
    print(f" -> {player2_name} has a {(1-probability_p1_wins)*100:.1f}% chance of winning.")
    return probability_p1_wins


if __name__ == "__main__":
    logging.info("--- Starting ATP Match Prediction Script ---")
    
    # --- Configuration ---
    # Choose model: 'Decision Tree', 'Random Forest', 'XGBoost'
    MODEL_TO_USE = "XGBoost" 
    # --- End Configuration ---
    
    # Load chosen model and expected features
    model, expected_feature_list = load_model_and_feature_info(model_choice=MODEL_TO_USE)
    # Load merged data separately for player info lookup
    merged_data = load_merged_data_once()

    if model and merged_data is not None and expected_feature_list:
        logging.info(f"--- Using model: {MODEL_TO_USE} ---")
        # --- Example Predictions ---
        logging.info("Running example predictions...")
        
        predict_match(
            player1_name="Novak Djokovic", 
            player2_name="Rafael Nadal", 
            model=model, 
            merged_df=merged_data, 
            expected_features=expected_feature_list,
            match_context={'surface': 'Clay', 'round': 'F', 'best_of': 5, 'tourney_level': 'G'}
        )
        
        predict_match(
            player1_name="Carlos Alcaraz", 
            player2_name="Jannik Sinner", 
            model=model, 
            merged_df=merged_data, 
            expected_features=expected_feature_list,
            match_context={'surface': 'Clay', 'round': 'F', 'best_of': 5, 'tourney_level': 'G'}
        )
        
        predict_match(
            player1_name="Jack Draper",
            player2_name="Thanasi Kokkinakis", 
            model=model, 
            merged_df=merged_data, 
            expected_features=expected_feature_list,
            match_context={'surface': 'Hard', 'best_of': 5, 'tourney_level': 'G'}
        )
        
    else:
        logging.info("Could not load model, merged data, or expected features. Cannot run predictions.")

    logging.info("--- Prediction Script Finished ---") 