import pandas as pd
import numpy as np
import os
import logging
import joblib
import matplotlib.pyplot as plt
import seaborn as sns
from collections import defaultdict
from tqdm import tqdm
from sklearn.metrics import accuracy_score

# Assuming ELO functions are defined here or imported (copy from 02_feature_engineering.py)
# --- ELO Calculation Functions ---
K_FACTOR = 32
DEFAULT_ELO = 1500
def calculate_expected_score(rating1, rating2):
    return 1.0 / (1.0 + 10.0 ** ((rating2 - rating1) / 400.0))
def update_elo(winner_elo, loser_elo):
    expected_winner = calculate_expected_score(winner_elo, loser_elo)
    new_winner_elo = winner_elo + K_FACTOR * (1 - expected_winner)
    new_loser_elo = loser_elo + K_FACTOR * (0 - expected_winner)
    return new_winner_elo, new_loser_elo
# --- End ELO Functions ---

from sklearn.model_selection import learning_curve, train_test_split, StratifiedKFold # For learning curve
# Assume prepare_data_for_modeling is defined or imported (copy/adapt from 03_model_training.py)
# We need it to get X and y for learning curve

# Setup basic logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# --- Global Settings ---
RESULTS_DIR = "results/"
DATA_DIR = "data/"
MERGED_DATA_PATH = os.path.join(DATA_DIR, "merged_atp_data.csv")
FEATURES_DATA_PATH = os.path.join(DATA_DIR, "features_atp_data.csv")
MODEL_PATH = os.path.join(RESULTS_DIR, "xgboost_model.joblib") # Default to best model

# Ensure results directory exists
if not os.path.exists(RESULTS_DIR):
    os.makedirs(RESULTS_DIR)
    logging.info(f"Created directory: {RESULTS_DIR}")

# --- Visualization Functions ---

def plot_elo_progression(merged_df, top_players, output_path):
    """Plots ELO rating progression for specified players."""
    logging.info(f"Plotting ELO progression for: {top_players}")
    if merged_df is None:
        logging.error("Merged data not loaded for ELO plotting.")
        return

    # Recalculate ELO history (simplified version focusing only on these players)
    player_elos = defaultdict(lambda: DEFAULT_ELO)
    elo_history = defaultdict(list) # player_id -> [(date, elo)]
    
    df_sorted = merged_df.sort_values(by='tourney_date').copy()
    df_sorted['tourney_date'] = pd.to_datetime(df_sorted['tourney_date'], errors='coerce')
    df_sorted.dropna(subset=['tourney_date', 'winner_id', 'loser_id'], inplace=True)

    target_player_ids = df_sorted[(df_sorted['winner_name'].isin(top_players)) | 
                                 (df_sorted['loser_name'].isin(top_players))]['winner_id'].unique().tolist() + \
                        df_sorted[(df_sorted['winner_name'].isin(top_players)) | 
                                 (df_sorted['loser_name'].isin(top_players))]['loser_id'].unique().tolist()
    target_player_ids = list(set(target_player_ids)) # Get unique IDs
    player_id_to_name = pd.concat([df_sorted[['winner_id', 'winner_name']].rename(columns={'winner_id':'id', 'winner_name':'name'}), 
                                   df_sorted[['loser_id', 'loser_name']].rename(columns={'loser_id':'id', 'loser_name':'name'})]).drop_duplicates('id').set_index('id')['name'].to_dict()
    
    target_player_ids = [pid for pid in target_player_ids if player_id_to_name.get(pid) in top_players] # Filter IDs matching names
    logging.info(f"Target player IDs found: {target_player_ids}")

    if not target_player_ids:
        logging.error("None of the specified top players found in the data.")
        return

    # Initialize history
    min_date = df_sorted['tourney_date'].min()
    for pid in target_player_ids:
        elo_history[pid].append((min_date - pd.Timedelta(days=1), DEFAULT_ELO)) # Starting ELO

    logging.info("Calculating ELO history...")
    for _, row in tqdm(df_sorted.iterrows(), total=df_sorted.shape[0], desc="Calculating ELO History"):
        winner_id = row['winner_id']
        loser_id = row['loser_id']
        match_date = row['tourney_date']

        winner_elo_before = player_elos[winner_id]
        loser_elo_before = player_elos[loser_id]
        
        new_winner_elo, new_loser_elo = update_elo(winner_elo_before, loser_elo_before)
        player_elos[winner_id] = new_winner_elo
        player_elos[loser_id] = new_loser_elo
        
        # Record history if target player
        if winner_id in target_player_ids:
            elo_history[winner_id].append((match_date, new_winner_elo))
        if loser_id in target_player_ids:
            elo_history[loser_id].append((match_date, new_loser_elo))

    logging.info("Plotting...")
    plt.figure(figsize=(12, 7))
    for pid in target_player_ids:
        history = sorted(elo_history[pid])
        dates = [item[0] for item in history]
        elos = [item[1] for item in history]
        player_name = player_id_to_name.get(pid, f"ID {pid}")
        plt.plot(dates, elos, label=player_name)

    plt.title("ELO Rating Progression for Top Players")
    plt.xlabel("Date")
    plt.ylabel("ELO Rating")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(output_path)
    plt.close()
    logging.info(f"ELO progression plot saved to {output_path}")

def plot_correlation_matrix(features_df, numeric_feature_list, output_path):
    """Plots the correlation matrix heatmap for selected numeric features."""
    logging.info("Plotting feature correlation matrix...")
    if features_df is None:
        logging.error("Feature data not loaded for correlation plotting.")
        return
    if not numeric_feature_list:
         logging.error("Numeric feature list is empty.")
         return

    # Select only the numeric features used in the final model
    df_numeric = features_df[numeric_feature_list].copy()
    
    # Handle potential remaining NaNs (shouldn't be any after script 03 prep)
    if df_numeric.isnull().sum().sum() > 0:
        logging.warning("NaNs found in numeric features before correlation. Imputing with median.")
        df_numeric.fillna(df_numeric.median(), inplace=True)

    corr = df_numeric.corr()
    
    plt.figure(figsize=(15, 12))
    sns.heatmap(corr, cmap='coolwarm', annot=False, fmt=".2f", linewidths=.5)
    plt.title("Correlation Matrix of Numeric Features")
    plt.tight_layout()
    plt.savefig(output_path)
    plt.close()
    logging.info(f"Correlation matrix plot saved to {output_path}")

def plot_learning_curves(estimator, title, X, y, cv=None, n_jobs=-1, train_sizes=np.linspace(0.1, 1.0, 5), output_path=None):
    """Generates and saves learning curves for a given estimator."""
    logging.info(f"Generating learning curves for {title}...")
    plt.figure(figsize=(10, 6))
    plt.title(title)
    plt.xlabel("Training examples")
    plt.ylabel("Score (e.g., ROC AUC)")

    try:
        train_sizes, train_scores, test_scores = learning_curve(
            estimator, X, y, cv=cv, n_jobs=n_jobs, train_sizes=train_sizes, scoring='roc_auc' # Use ROC AUC score
        )
        train_scores_mean = np.mean(train_scores, axis=1)
        train_scores_std = np.std(train_scores, axis=1)
        test_scores_mean = np.mean(test_scores, axis=1)
        test_scores_std = np.std(test_scores, axis=1)

        plt.grid()
        plt.fill_between(train_sizes, train_scores_mean - train_scores_std,
                         train_scores_mean + train_scores_std, alpha=0.1, color="r")
        plt.fill_between(train_sizes, test_scores_mean - test_scores_std,
                         test_scores_mean + test_scores_std, alpha=0.1, color="g")
        plt.plot(train_sizes, train_scores_mean, 'o-', color="r", label="Training score")
        plt.plot(train_sizes, test_scores_mean, 'o-', color="g", label="Cross-validation score")
        plt.legend(loc="best")
        plt.ylim(0.5, 1.01) # Adjust ylim based on expected scores
        plt.tight_layout()
        if output_path:
            plt.savefig(output_path)
            logging.info(f"Learning curve plot saved to {output_path}")
        else:
            plt.show()
        plt.close()
    except Exception as e:
        logging.error(f"Could not generate learning curve for {title}: {e}")

def plot_feature_importances(model, feature_names, title, output_path, top_n=20):
     """Plots feature importances for tree-based models."""
     logging.info(f"Plotting feature importances for {title}...")
     if not hasattr(model, 'feature_importances_'):
         logging.warning(f"Model {title} does not have feature_importances_ attribute.")
         return
         
     importances = model.feature_importances_
     indices = np.argsort(importances)[::-1]
     num_features = len(feature_names)
     plot_n = min(top_n, num_features)

     plt.figure(figsize=(10, max(6, plot_n * 0.3))) # Adjust height based on number of features
     plt.barh(range(plot_n), importances[indices][:plot_n][::-1], align='center') # Use barh for horizontal
     plt.yticks(range(plot_n), np.array(feature_names)[indices][:plot_n][::-1])
     plt.title(f'{title} - Top {plot_n} Feature Importances')
     plt.xlabel('Importance')
     plt.tight_layout()
     plt.savefig(output_path)
     plt.close()
     logging.info(f"Feature importance plot saved to {output_path}")
     
def plot_performance_by_surface(test_indices, y_true, y_pred, merged_data, output_path):
    """Plots model accuracy grouped by surface using indices to link to merged data."""
    logging.info("Plotting performance by surface...")
    if merged_data is None or merged_data.empty:
        logging.error("Merged data not loaded or empty for surface performance plot.")
        return
    if 'surface' not in merged_data.columns:
        logging.error("'surface' column not found in merged data.")
        return
    # Check for the linking column ('index' which originated from reset_index in script 02)
    if 'index' not in merged_data.columns:
         logging.error("'index' column needed for merging not found in merged_data.")
         return

    try:
        # Create a DataFrame from test results
        results_df = pd.DataFrame({
            'index': test_indices, 
            'y_true': y_true, 
            'y_pred': y_pred
        })
        
        # Select only necessary columns from merged_data for the merge
        surface_lookup = merged_data[['index', 'surface']].copy()
        
        # Merge results with surface lookup on the shared 'index'
        plot_data = pd.merge(results_df, surface_lookup, on='index', how='left')
        
        if plot_data['surface'].isnull().any():
            logging.warning(f"Could not find surface for {plot_data['surface'].isnull().sum()} test instances after merge.")
            plot_data.dropna(subset=['surface'], inplace=True) # Drop rows where merge failed
            
        if plot_data.empty:
             logging.error("No data left after merging surface information.")
             return

        surface_accuracy = plot_data.groupby('surface').apply(
            lambda x: accuracy_score(x['y_true'], x['y_pred']), 
            include_groups=False
        )
        
        plt.figure(figsize=(8, 5))
        surface_accuracy.sort_values().plot(kind='bar')
        plt.title("Model Accuracy by Surface (Test Set)")
        plt.xlabel("Surface")
        plt.ylabel("Accuracy")
        plt.xticks(rotation=45)
        plt.grid(axis='y', linestyle='--')
        plt.tight_layout()
        plt.savefig(output_path)
        plt.close()
        logging.info(f"Surface performance plot saved to {output_path}")
    except Exception as e:
        logging.error(f"Failed to plot performance by surface: {e}")
        logging.error(f"Columns available in plot_data: {plot_data.columns.tolist() if 'plot_data' in locals() else 'N/A'}")

# --- Main Execution --- 
if __name__ == "__main__":
    logging.info("--- Starting Visualization Script ---")

    # --- Load Data and Model ---
    logging.info("Loading data and model...")
    merged_df = None
    try:
        logging.info(f"Loading merged data from: {MERGED_DATA_PATH}")
        # Load necessary columns. Script 01 doesn't explicitly save index, 
        # so we load all and reset index to match features_df's index source.
        temp_merged_df = pd.read_csv(MERGED_DATA_PATH, low_memory=False, parse_dates=['tourney_date'])
        # IMPORTANT: Reset index here to create the 'index' column that aligns 
        # with the 'index' created in script 02 after its reset_index call.
        temp_merged_df.reset_index(inplace=True)
        
        cols_needed = ['index', 'tourney_date', 'winner_id', 'winner_name', 'loser_id', 'loser_name', 'surface'] 
        merged_df = temp_merged_df[[col for col in cols_needed if col in temp_merged_df.columns]]
        logging.info("Merged data loaded and processed for linking.")
    except Exception as e:
         logging.error(f"Failed to load or process merged data: {e}")
         # merged_df remains None
         
    features_df_raw = pd.read_csv(FEATURES_DATA_PATH, low_memory=False) if os.path.exists(FEATURES_DATA_PATH) else None
    model = joblib.load(MODEL_PATH) if os.path.exists(MODEL_PATH) else None
    
    # --- Generate Plots requiring features_df and model ---
    plots_generated_count = 0
    if features_df_raw is None or model is None:
        logging.error("Feature data or model file not found. Cannot generate all visualizations that require them.")
    else:
        features_df = features_df_raw.copy() 
        target_col = 'RESULT'
        numeric_features = [] 
        valid_data_for_plots = False
        
        # Ensure 'index' column exists from features file for linking
        if 'index' not in features_df.columns:
             logging.error("'index' column expected but not found in features data. Cannot link for surface plot.")
        else:
            if hasattr(model, 'feature_names_in_'):
                numeric_features = model.feature_names_in_.tolist()
                logging.info(f"Using {len(numeric_features)} features from loaded model for plots.")
                
                features_present = [f for f in numeric_features if f in features_df.columns]
                missing_features = [f for f in numeric_features if f not in features_df.columns]
                
                if missing_features:
                     logging.error(f"Model expects features not present in data: {missing_features}. Skipping dependent plots.")
                else:
                    # Prepare X (numeric only) and y (target) for splitting
                    X = features_df[features_present].copy()
                    y = features_df[target_col]
                    # Keep track of original indices from features_df
                    original_indices = features_df['index'] 
                    
                    if X.isnull().sum().sum() > 0:
                        logging.warning(f"Imputing {X.isnull().sum().sum()} NaNs found before plotting.")
                        numeric_cols_in_X = X.select_dtypes(include=np.number).columns
                        imputed_values = X[numeric_cols_in_X].fillna(X[numeric_cols_in_X].median())
                        X[numeric_cols_in_X] = imputed_values 
                        if X.isnull().sum().sum() > 0:
                             logging.error("NaNs still present after imputation.")
                        else:
                            valid_data_for_plots = True
                    else:
                         valid_data_for_plots = True 

                    if valid_data_for_plots:
                        # Split data - Stratify on y 
                        # Pass original_indices to maintain the link
                        _, X_test_vis, _, y_test_vis, _, test_indices = train_test_split(
                            X, y, original_indices, test_size=0.2, random_state=42, stratify=y
                        )
                        
                        y_pred_test_vis = model.predict(X_test_vis)

                        # --- Generate Plots ---
                        plot_correlation_matrix(X, numeric_features, os.path.join(RESULTS_DIR, "feature_correlation_heatmap.png"))
                        plots_generated_count += 1
                        cv_vis = StratifiedKFold(n_splits=3, shuffle=True, random_state=42) # Use Stratified for learning curve vis
                        plot_learning_curves(model, f"Learning Curve ({type(model).__name__})", 
                                             X, y, cv=cv_vis, n_jobs=-1, 
                                             output_path=os.path.join(RESULTS_DIR, "learning_curve.png"))
                        plots_generated_count += 1
                        plot_feature_importances(model, numeric_features, f"Feature Importances ({type(model).__name__})",
                                                 os.path.join(RESULTS_DIR, "final_model_feature_importance.png"))
                        plots_generated_count += 1                                             
                        # Pass original indices from the test split, true labels, predictions, and merged_df
                        if merged_df is not None:
                            plot_performance_by_surface(test_indices.tolist(), y_test_vis.values, y_pred_test_vis, merged_df, 
                                                        os.path.join(RESULTS_DIR, "performance_by_surface.png"))
                            plots_generated_count += 1  
                        else:
                            logging.warning("Skipping surface performance plot as merged_df failed to load.")                                              
                    else:
                         logging.warning("Skipping plots requiring valid X matrix due to missing features or imputation failure.")
            else:
                 logging.error("Cannot proceed with model-dependent plots without feature names from the model.")

    # --- Generate Plots requiring only merged_df ---
    if merged_df is not None:
        # Need different columns for ELO plot function
        try:
            logging.info(f"Reloading merged data with necessary ELO columns...")
            cols_needed_elo = ['tourney_date', 'winner_id', 'winner_name', 'loser_id', 'loser_name']
            merged_df_elo = pd.read_csv(MERGED_DATA_PATH, usecols=cols_needed_elo, low_memory=False, parse_dates=['tourney_date'])
            plot_elo_progression(merged_df_elo, 
                                 top_players=["Novak Djokovic", "Rafael Nadal", "Roger Federer", "Andy Murray"], 
                                 output_path=os.path.join(RESULTS_DIR, "elo_progression.png"))
            plots_generated_count += 1
        except Exception as e:
             logging.error(f"Failed to load data or plot ELO progression: {e}")                             
    else:
         logging.warning("Skipping ELO progression plot as merged data failed to load initially.")

    # --- Textual Case Study Reminder --- 
    logging.info("--- Prediction Case Study --- (See 04_model_prediction.py output and feature importance plot)")
    
    logging.info(f"--- Visualization Script Finished ({plots_generated_count} plots attempted) ---") 