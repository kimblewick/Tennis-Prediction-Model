# Tennis Match Prediction Model

## Overview

The Tennis Prediction Model is a machine learning pipeline that predicts ATP Men's Singles match outcomes using historical data from 1985 to 2024. The project implements sophisticated feature engineering including ELO ratings, head-to-head statistics, and rolling performance metrics to achieve competitive prediction accuracy.

The model uses a temporal train-test split to avoid data leakage and employs multiple algorithms (Decision Trees, Random Forest, XGBoost) with the XGBoost model performing best at approximately 70% accuracy on test data.

## Key Features

- **Advanced Feature Engineering**: Implements ELO rating system, head-to-head statistics, rolling win percentages, and surface-specific metrics
- **Temporal Data Splitting**: Uses chronological splits (pre-2023 for training, 2023+ for testing) to simulate real-world prediction scenarios
- **Multiple Model Comparison**: Trains and evaluates Decision Trees, Random Forest, and XGBoost models with hyperparameter tuning
- **Comprehensive Pipeline**: Step-by-step modular approach from data cleaning to prediction with clear separation of concerns
- **Real Tournament Predictions**: Successfully applied to predict 2025 Australian Open matches
- **Visualization Suite**: Includes ELO progression plots, feature importance analysis, and performance metrics visualization

## Technologies Used

- **Python 3.x**: Core programming language
- **Data Processing**: Pandas, NumPy
- **Machine Learning**: Scikit-learn, XGBoost
- **Visualization**: Matplotlib, Seaborn
- **Model Persistence**: Joblib
- **Progress Tracking**: tqdm

## Project Structure

```
├── 01_data_cleaning_merging.py    # Data loading and initial cleaning
├── 02_feature_engineering.py     # ELO ratings, H2H stats, rolling metrics
├── 03_model_training.py          # Model training and evaluation
├── 04_model_prediction.py        # Match prediction interface
├── 05_visualizations.py          # Analysis and visualization tools
├── data/
│   ├── all/                      # Raw ATP match data (1985-2024)
│   ├── atp_players.csv          # Player information
│   ├── merged_atp_data.csv      # Cleaned and merged data
│   └── features_atp_data.csv    # Engineered features dataset
└── results/                     # Trained models and outputs
```

## Features Implemented

### ELO Rating System
- Dynamic player ratings updated after each match
- K-factor of 32 for standard ELO calculations
- Starting rating of 1500 for all players

### Head-to-Head Statistics
- Overall H2H win/loss records between players
- Surface-specific H2H performance
- Rolling H2H percentages to avoid data leakage

### Rolling Performance Metrics
- Win percentages over last 5, 10, and 20 matches
- Surface-specific rolling statistics
- Temporal ordering to prevent lookahead bias

## Results/Achievements

- **~70% Test Accuracy**: XGBoost model achieves approximately 70% accuracy on temporal test set (2023-2024 data)
- **Real Tournament Application**: Successfully generated predictions for 2025 Australian Open
- **Model Comparison**: Comprehensive evaluation of multiple algorithms with cross-validation
- **Feature Importance Analysis**: Identified key predictive features (rankings, ELO ratings, H2H stats)
- **Overfitting Prevention**: Temporal splits and proper validation prevent data leakage

## How to Run

### Prerequisites

1. **Python 3.7+** installed on your system
2. **Virtual environment** (recommended)

### Installation

1. **Clone the repository:**
   ```bash
   git clone <your-repo-url>
   cd "Tennis Prediction Model"
   ```

2. **Create and activate virtual environment:**
   ```bash
   python -m venv venv
   
   # On macOS/Linux:
   source venv/bin/activate
   
   # On Windows:
   venv\Scripts\activate
   ```

3. **Install required packages:**
   ```bash
   pip install -r requirements.txt
   ```

### Running the Pipeline

**Step 1: Data Cleaning and Merging**
```bash
python 01_data_cleaning_merging.py
```
- Loads ATP match data from `data/all/` directory
- Cleans and merges data from 1985 onwards
- Outputs: `data/merged_atp_data.csv`

**Step 2: Feature Engineering**
```bash
python 02_feature_engineering.py
```
- Calculates ELO ratings, H2H statistics, rolling metrics
- Handles missing values and creates model-ready features
- Outputs: `data/features_atp_data.csv`

**Step 3: Model Training**
```bash
python 03_model_training.py
```
- Trains multiple models with temporal cross-validation
- Performs hyperparameter tuning
- Outputs: Trained models in `results/` directory

**Step 4: Make Predictions**
```bash
python 04_model_prediction.py
```
- Interactive script for predicting match outcomes
- Uses best performing model (XGBoost by default)
- Requires player names and match context

**Step 5: Generate Visualizations**
```bash
python 05_visualizations.py
```
- Creates ELO progression plots for top players
- Generates feature importance and correlation matrices
- Outputs: Visualization files in `results/` directory

### Quick Example

To predict a match between two players:

```python
from 04_model_prediction import predict_match, load_model_and_feature_info, load_merged_data_once

# Load model and data
model, features = load_model_and_feature_info('XGBoost')
merged_data = load_merged_data_once()

# Define match context
match_context = {
    'surface': 'Hard',
    'tourney_level': 'G',  # Grand Slam
    'round': 'F',          # Final
    'best_of': 5
}

# Make prediction
result = predict_match('Novak Djokovic', 'Carlos Alcaraz', model, merged_data, features, match_context)
```

## Data Requirements

- **ATP Match Data**: Historical match results from 1985-2024 in CSV format
- **Player Information**: Player IDs, names, and biographical data
- **Match Context**: Tournament information, surface types, rounds

The data should be placed in the `data/all/` directory following the naming convention `atp_matches_YYYY.csv`.

## Model Performance

- **Training Accuracy**: ~75% (with cross-validation)
- **Test Accuracy**: ~70% (on 2023-2024 data)
- **Best Model**: XGBoost with tuned hyperparameters
- **Key Features**: ATP ranking difference, ELO rating difference, H2H records

## Future Improvements

- Incorporate serve/return statistics when available
- Add player injury and form data
- Implement ensemble methods
- Extend to women's tennis (WTA) predictions
- Add live betting odds as features

## Contributing

This project is part of a portfolio demonstration. For suggestions or improvements, please open an issue or submit a pull request.

## License

This project is for educational and portfolio purposes.