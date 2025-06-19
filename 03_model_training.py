import pandas as pd
import numpy as np
import os
import logging
import joblib # For saving models
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, cross_validate, TimeSeriesSplit, RandomizedSearchCV
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from sklearn.metrics import (accuracy_score, precision_score, recall_score, f1_score,
                           roc_auc_score, confusion_matrix, RocCurveDisplay, PrecisionRecallDisplay)
from sklearn.preprocessing import LabelEncoder # If needed for any remaining categoricals
from scipy.stats import randint, uniform 

# Setup basic logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def load_feature_data(input_path='data/features_atp_data.csv'):
    """Loads the engineered feature data."""
    if not os.path.exists(input_path):
        logging.error(f"Feature file not found: {input_path}")
        return None
    try:
        logging.info(f"Loading feature data from: {input_path}")
        df = pd.read_csv(input_path, low_memory=False)
        logging.info(f"Feature data loaded successfully. Shape: {df.shape}")
        # Explicitly handle potential infinities that might arise from diffs
        df.replace([np.inf, -np.inf], np.nan, inplace=True)
        # Note: We expect NaNs in the placeholder columns. Handle them before training.
        return df
    except Exception as e:
        logging.error(f"Error loading data from {input_path}: {e}")
        return None

def prepare_data_for_modeling(df, test_split_year=2023):
    """Prepares data for modeling: selects features, handles NaNs, performs temporal split."""
    logging.info("Preparing data for modeling with temporal split...")
    
    # Ensure year column exists
    if 'year' not in df.columns:
        logging.error("'year' column is required for temporal split but not found.")
        return None, None, None, None, None

    target_col = 'RESULT'
    # Exclude IDs, names, etc. Also exclude year/month as they are used for splitting/context
    cols_to_exclude = [target_col, 'p1_id', 'p2_id', 'p1_name', 'p2_name',
                       'tourney_id', 'tourney_name', 'score', 'match_num', 'year', 'month'] 
    
    # Exclude index if present
    if 'index' in df.columns:
        cols_to_exclude.append('index')
        
    # Exclude any remaining all-NaN columns (placeholders)
    placeholder_cols = [col for col in df.columns if df[col].isnull().all()]
    if placeholder_cols:
        logging.warning(f"Excluding placeholder/all-NaN columns: {placeholder_cols}")
        cols_to_exclude.extend(placeholder_cols)
        
    initial_features = [col for col in df.columns if col not in cols_to_exclude]
    logging.info(f"Target variable: {target_col}")
    logging.info(f"Initial number of potential features: {len(initial_features)}")

    X_initial = df[initial_features]
    y = df[target_col]

    # Select only numeric features
    numeric_features = X_initial.select_dtypes(include=np.number).columns.tolist()
    X = X_initial[numeric_features].copy() 
    logging.info(f"Number of numeric features selected for modeling: {len(numeric_features)}")

    # Handle NaNs 
    if X.isnull().sum().sum() > 0:
        logging.warning(f"Found {X.isnull().sum().sum()} NaNs in numeric feature set. Imputing with median.")
        numeric_cols_in_X = X.select_dtypes(include=np.number).columns
        medians = X[numeric_cols_in_X].median()
        X[numeric_cols_in_X] = X[numeric_cols_in_X].fillna(medians)
        if X.isnull().sum().sum() > 0:
            logging.error("NaNs still present after attempting median imputation!")
            return None, None, None, None, None
    else:
        logging.info("No NaNs found in the final numeric feature set.")

    # --- Temporal Split --- 
    logging.info(f"Performing temporal split: Training data < {test_split_year}, Test data >= {test_split_year}")
    train_mask = df['year'] < test_split_year
    test_mask = df['year'] >= test_split_year

    X_train = X[train_mask]
    X_test = X[test_mask]
    y_train = y[train_mask]
    y_test = y[test_mask]
    
    if X_train.empty or X_test.empty:
        logging.error(f"Train or test set is empty after temporal split on year {test_split_year}. Adjust split year.")
        return None, None, None, None, None

    logging.info(f"Train set shape: {X_train.shape}, Test set shape: {X_test.shape}")
    min_train_year = df.loc[train_mask, 'year'].min() if train_mask.any() else 'N/A'
    max_train_year = df.loc[train_mask, 'year'].max() if train_mask.any() else 'N/A'
    min_test_year = df.loc[test_mask, 'year'].min() if test_mask.any() else 'N/A'
    max_test_year = df.loc[test_mask, 'year'].max() if test_mask.any() else 'N/A'
    logging.info(f"Training period: {min_train_year}-{max_train_year}")
    logging.info(f"Test period: {min_test_year}-{max_test_year}")

    return X_train, X_test, y_train, y_test, numeric_features

def train_evaluate_single_model(model, name, X_train, y_train, X_test, y_test, cv, results):
    """Helper to train and evaluate a single model instance."""
    logging.info(f"--- Evaluating Model: {name} ---")
    final_model = None
    try:
        # CV first
        logging.info("Performing cross-validation...")
        cv_scoring = ['accuracy', 'precision_macro', 'recall_macro', 'f1_macro', 'roc_auc']
        # Add return_train_score=True for overfitting check
        cv_results = cross_validate(model, X_train, y_train, cv=cv, scoring=cv_scoring, n_jobs=-1, return_train_score=True) 
        results[name] = {f"cv_{metric}": np.mean(cv_results[f'test_{metric}']) for metric in cv_scoring}
        results[name]['cv_train_roc_auc'] = np.mean(cv_results['train_roc_auc']) # Store train score
        logging.info(f"CV results (mean): {results[name]}")
        
        # Train final model
        logging.info("Training final model on full training data...")
        model.fit(X_train, y_train)
        final_model = model
        logging.info("Training complete.")

        # Evaluate on test set
        logging.info("Evaluating on temporal test set...")
        y_pred = final_model.predict(X_test)
        y_proba = final_model.predict_proba(X_test)[:, 1]
        test_metrics = {
            'accuracy': accuracy_score(y_test, y_pred),
            'precision': precision_score(y_test, y_pred, average='macro'),
            'recall': recall_score(y_test, y_pred, average='macro'),
            'f1': f1_score(y_test, y_pred, average='macro'),
            'roc_auc': roc_auc_score(y_test, y_proba)
        }
        results[name].update({f"test_{k}": v for k, v in test_metrics.items()})
        logging.info(f"Test set performance: {test_metrics}")

    except Exception as e:
        logging.error(f"Error during training/evaluation for {name}: {e}")
        results[name] = results.get(name, {}) # Ensure key exists
        results[name].update({f"test_{k}": np.nan for k in ['accuracy', 'precision', 'recall', 'f1', 'roc_auc']})
        
    return final_model # Return trained model or None

def train_and_evaluate_tuned(X_train, X_test, y_train, y_test, features):
    """Demonstrates under/overfitting, tunes all models, evaluates all."""
    logging.info("Starting model training with underfit/overfit examples and light tuning...")
    results = {}
    trained_models = {}
    output_dir = "results/"
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
        logging.info(f"Created directory: {output_dir}")

    # Define CV strategy (TimeSeriesSplit)
    N_SPLITS_CV = 5 
    tscv = TimeSeriesSplit(n_splits=N_SPLITS_CV)
    logging.info(f"Using TimeSeriesSplit with {N_SPLITS_CV} splits for CV and Tuning.")
    
    # --- 1. Underfitting Examples --- 
    underfit_dt = DecisionTreeClassifier(max_depth=2, random_state=42)
    train_evaluate_single_model(underfit_dt, "Decision Tree (Underfit, max_depth=2)", 
                                X_train, y_train, X_test, y_test, tscv, results)
                                
    underfit_rf = RandomForestClassifier(n_estimators=5, max_depth=3, random_state=42, n_jobs=-1)
    train_evaluate_single_model(underfit_rf, "Random Forest (Underfit, n_estimators=5, max_depth=3)",
                                X_train, y_train, X_test, y_test, tscv, results)
                                
    underfit_xgb = XGBClassifier(n_estimators=5, max_depth=2, learning_rate=0.5, random_state=42, use_label_encoder=False, eval_metric='logloss', n_jobs=-1)
    train_evaluate_single_model(underfit_xgb, "XGBoost (Underfit, n_estimators=5, max_depth=2)",
                                X_train, y_train, X_test, y_test, tscv, results)

    # --- 2. Overfitting Examples --- 
    overfit_dt = DecisionTreeClassifier(max_depth=None, min_samples_leaf=1, random_state=42)
    train_evaluate_single_model(overfit_dt, "Decision Tree (Overfit, max_depth=None)", 
                                X_train, y_train, X_test, y_test, tscv, results)
    if "Decision Tree (Overfit, max_depth=None)" in results and 'cv_train_roc_auc' in results["Decision Tree (Overfit, max_depth=None)"]:
        overfit_train_auc = results["Decision Tree (Overfit, max_depth=None)"]['cv_train_roc_auc']
        overfit_cv_auc = results["Decision Tree (Overfit, max_depth=None)"]['cv_roc_auc']
        logging.warning(f"Overfit Example: Train AUC={overfit_train_auc:.4f}, CV AUC={overfit_cv_auc:.4f}. Large gap suggests overfitting.")

    # --- 3. Default Simplified Models (Optional - can be removed if only tuned needed) ---
    # rf_default = RandomForestClassifier(random_state=42, n_estimators=50, n_jobs=-1)
    # xgb_default = XGBClassifier(random_state=42, n_estimators=50, use_label_encoder=False, eval_metric='logloss', n_jobs=-1)
    # train_evaluate_single_model(rf_default, "Random Forest (Default Simplified)", 
    #                             X_train, y_train, X_test, y_test, tscv, results)
    # train_evaluate_single_model(xgb_default, "XGBoost (Default Simplified)", 
    #                             X_train, y_train, X_test, y_test, tscv, results)
                                
    # --- 4. LIGHT Hyperparameter Tuning (DT, RF & XGBoost) ---
    # Define Base models for tuning search
    dt_tune_base = DecisionTreeClassifier(random_state=42)
    rf_tune_base = RandomForestClassifier(random_state=42, n_jobs=-1)
    xgb_tune_base = XGBClassifier(random_state=42, use_label_encoder=False, eval_metric='logloss', n_jobs=-1)
    
    # Define Parameter Distributions
    dt_param_dist_light = {
        'criterion': ['gini', 'entropy'],
        'max_depth': [5, 10, 15, 20, None], # None means no limit
        'min_samples_split': randint(2, 20),
        'min_samples_leaf': randint(1, 20)
    }
    rf_param_dist_light = {
        'n_estimators': randint(50, 150), 
        'max_depth': [10, 20, 30, None], 
        'min_samples_leaf': [1, 3, 5] 
    }
    xgb_param_dist_light = {
        'n_estimators': randint(50, 150), 
        'max_depth': [3, 5, 7],
        'learning_rate': uniform(0.05, 0.2), 
        'reg_alpha': [0, 0.01, 0.1],
        'reg_lambda': [0.5, 1.0, 5.0]
    }
    N_ITER_SEARCH = 20 # Number of iterations for RandomizedSearch

    # Models to tune
    tuned_models_dict = {
        "Decision Tree (Tuned)": dt_tune_base,
        "Random Forest (Tuned)": rf_tune_base,
        "XGBoost (Tuned)": xgb_tune_base
    }
    tuning_params = {
        "Decision Tree (Tuned)": dt_param_dist_light,
        "Random Forest (Tuned)": rf_param_dist_light, 
        "XGBoost (Tuned)": xgb_param_dist_light
    }
                       
    for name, model_instance in tuned_models_dict.items():
         logging.info(f"--- Tuning Model: {name} ---")
         random_search = RandomizedSearchCV(
             model_instance, 
             param_distributions=tuning_params[name], 
             n_iter=N_ITER_SEARCH, 
             cv=tscv, 
             scoring='roc_auc', 
             n_jobs=-1, 
             random_state=42,
             verbose=1 
         )
         try:
             random_search.fit(X_train, y_train)
             logging.info(f"Best parameters found for {name}: {random_search.best_params_}")
             logging.info(f"Best CV ROC AUC score during tuning for {name}: {random_search.best_score_:.4f}")
             best_model = random_search.best_estimator_
             # Evaluate this best model on the test set using the helper function
             trained_models[name] = train_evaluate_single_model(best_model, name, X_train, y_train, X_test, y_test, tscv, results)
         except Exception as e:
             logging.error(f"Randomized Search failed for {name}: {e}.")
             results[name] = results.get(name, {}) # Ensure key exists
             results[name].update({f"test_{k}": np.nan for k in ['accuracy', 'precision', 'recall', 'f1', 'roc_auc']})
             trained_models[name] = None

    # --- Save Visualizations for the Best Tuned Model (e.g., XGBoost Tuned) --- 
    # Find best model based on test roc_auc from the results dict
    best_model_name = None
    best_roc_auc = -1
    # Consider only tuned models for the 'best' designation
    tuned_model_names = [name for name in results if "(Tuned)" in name and not np.isnan(results[name].get('test_roc_auc', -1))]
    
    if tuned_model_names: # Check if any tuned models ran successfully
        for name in tuned_model_names:
            if results[name]['test_roc_auc'] > best_roc_auc:
                best_roc_auc = results[name]['test_roc_auc']
                best_model_name = name
    else: # Fallback if tuning failed for all
         # Choose best from defaults if available
         default_model_names = [name for name in results if "(Default Simplified)" in name and not np.isnan(results[name].get('test_roc_auc', -1))]
         if default_model_names:
              for name in default_model_names:
                   if results[name]['test_roc_auc'] > best_roc_auc:
                       best_roc_auc = results[name]['test_roc_auc']
                       best_model_name = name # e.g., "XGBoost (Default Simplified)" 
         # Add further fallbacks if needed (e.g., to overfit/underfit DT)
    
    if best_model_name and best_model_name in trained_models and trained_models[best_model_name] is not None:
         final_model_to_plot = trained_models[best_model_name] 
         logging.info(f"Generating visualizations for best model: {best_model_name} (Test ROC AUC: {best_roc_auc:.4f})...")
         y_pred = final_model_to_plot.predict(X_test)
         y_proba = final_model_to_plot.predict_proba(X_test)[:, 1]
         
         fig, ax = plt.subplots(figsize=(6, 5))
         cm = confusion_matrix(y_test, y_pred)
         sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax)
         ax.set_title(f'{best_model_name} - Confusion Matrix (Temporal Test)')
         ax.set_xlabel('Predicted Label')
         ax.set_ylabel('True Label')
         plt.tight_layout()
         # Clean filename
         clean_name = best_model_name.lower().replace(' ', '_').replace('(', '').replace(')','')
         plt.savefig(os.path.join(output_dir, f"{clean_name}_confusion_matrix_temporal.png"))
         plt.close(fig)

         fig, ax = plt.subplots(figsize=(6, 5))
         RocCurveDisplay.from_estimator(final_model_to_plot, X_test, y_test, ax=ax)
         ax.set_title(f'{best_model_name} - ROC Curve (Temporal Test)')
         plt.tight_layout()
         plt.savefig(os.path.join(output_dir, f"{clean_name}_roc_curve_temporal.png"))
         plt.close(fig)

         fig, ax = plt.subplots(figsize=(6, 5))
         PrecisionRecallDisplay.from_estimator(final_model_to_plot, X_test, y_test, ax=ax)
         ax.set_title(f'{best_model_name} - Precision-Recall Curve (Temporal Test)')
         plt.tight_layout()
         plt.savefig(os.path.join(output_dir, f"{clean_name}_precision_recall_curve_temporal.png"))
         plt.close(fig)

         if hasattr(final_model_to_plot, 'feature_importances_'):
             importances = final_model_to_plot.feature_importances_
             indices = np.argsort(importances)[::-1]
             plot_n = min(20, len(features)) 
             fig, ax = plt.subplots(figsize=(10, 8))
             ax.bar(range(plot_n), importances[indices][:plot_n], align='center')
             ax.set_xticks(range(plot_n))
             ax.set_xticklabels(np.array(features)[indices][:plot_n], rotation=90)
             ax.set_title(f'{best_model_name} - Top {plot_n} Feature Importances')
             ax.set_xlabel('Importance')
             ax.set_ylabel('Importance')
             plt.tight_layout()
             plt.savefig(os.path.join(output_dir, f"{clean_name}_feature_importance.png"))
             plt.close(fig)
             logging.info(f"Visualizations saved for {best_model_name}.")
         else:
             logging.warning(f"Feature importance plot not generated for {best_model_name}.")
             
         # Save the best model separately
         joblib.dump(final_model_to_plot, os.path.join(output_dir, "best_tuned_model.joblib"))
         logging.info(f"Saved best model ({best_model_name}) to results/best_tuned_model.joblib")
    else:
        logging.warning("Could not generate visualizations as no suitable best model was found or trained successfully.")
        
    logging.info("Model training, tuning, and evaluation complete.")
    return results, trained_models # Return dict containing all trained models

if __name__ == "__main__":
    logging.info("Starting ATP Model Training Script")

    feature_df = load_feature_data()

    if feature_df is not None:
        X_train, X_test, y_train, y_test, feature_list = prepare_data_for_modeling(feature_df, test_split_year=2023)
        
        if X_train is not None and X_test is not None:
            if X_train.isnull().sum().sum() == 0 and X_test.isnull().sum().sum() == 0:
                # Call the function that includes tuning examples
                training_results, models = train_and_evaluate_tuned(X_train, X_test, y_train, y_test, feature_list)
                
                logging.info("--- Final Results Summary (Temporal Split) ---")
                results_df = pd.DataFrame(training_results).T
                results_df['Tuned'] = results_df.index.str.contains('\(Tuned\)')
                logging.info(f"\n{results_df.to_string()}")
                results_df.to_csv(os.path.join("results/", "model_performance_summary_temporal_tuned.csv"))
                logging.info("Saved temporal results summary to results/model_performance_summary_temporal_tuned.csv")
            else:
                 logging.error("NaNs still present in data after preparation. Halting.")
        else:
             logging.error("Data preparation failed, likely due to split issue. Halting.")
    else:
        logging.warning("Pipeline halted as feature data could not be loaded.")

    logging.info("Script finished.") 