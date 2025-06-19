import pandas as pd
import os
import glob
import logging

# Setup basic logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def load_data(data_path='data/all/', start_year=1985):
    """Loads ATP match CSV files from the specified path, starting from start_year."""
    logging.info(f"Searching for ATP match data in: {os.path.abspath(data_path)}")
    all_files = glob.glob(os.path.join(data_path, "atp_matches_*.csv"))

    if not all_files:
        logging.warning(f"No CSV files found matching 'atp_matches_*.csv' in {os.path.abspath(data_path)}")
        return []

    logging.info(f"Found {len(all_files)} files initially.")
    all_dfs = []
    files_loaded_count = 0
    for f in all_files:
        try:
            # Extract year from filename
            filename = os.path.basename(f)
            year_str = filename.split('_')[-1].split('.')[0]
            year = int(year_str)
            
            if year >= start_year:
                df = pd.read_csv(f, low_memory=False)
                all_dfs.append(df)
                logging.info(f"Successfully loaded: {f} ({len(df)} rows)")
                files_loaded_count += 1
            else:
                 logging.debug(f"Skipping file (year {year} < {start_year}): {f}")
                 
        except (ValueError, IndexError) as e:
             logging.warning(f"Could not parse year from filename '{f}'. Skipping. Error: {e}")
        except Exception as e:
            logging.error(f"Error loading file {f}: {e}")
            
    logging.info(f"Loaded {files_loaded_count} files from year {start_year} onwards.")
    logging.info(f"Successfully loaded {len(all_dfs)} dataframes.")
    return all_dfs

def merge_data(dataframes):
    """Merges a list of DataFrames into a single DataFrame."""
    if not dataframes:
        logging.warning("No dataframes provided for merging.")
        return pd.DataFrame() # Return empty DataFrame

    logging.info(f"Merging {len(dataframes)} dataframes...")
    merged_df = pd.concat(dataframes, ignore_index=True)
    logging.info(f"Merging complete. Merged DataFrame shape: {merged_df.shape}")
    return merged_df

def clean_data(df):
    """Performs initial data cleaning on the merged DataFrame."""
    if df.empty:
        logging.warning("DataFrame is empty, skipping cleaning.")
        return df

    logging.info("Starting initial data cleaning...")
    # --- Add specific cleaning steps here ---

    # Example: Convert 'tourney_date' to datetime
    if 'tourney_date' in df.columns:
        try:
            df['tourney_date'] = pd.to_datetime(df['tourney_date'], format='%Y%m%d', errors='coerce')
            logging.info("'tourney_date' converted to datetime.")
            # Log rows where conversion failed (NaT values)
            nat_count = df['tourney_date'].isna().sum()
            if nat_count > 0:
                logging.warning(f"{nat_count} rows had invalid 'tourney_date' format and were set to NaT.")
        except Exception as e:
            logging.error(f"Error converting 'tourney_date': {e}")

    # Example: Handle missing values (Illustrative - adjust strategy based on actual data)
    initial_rows = len(df)
    logging.info(f"Missing values before handling:\n{df.isnull().sum()}")
    # Example: Drop rows missing crucial identifiers if necessary
    # df.dropna(subset=['winner_id', 'loser_id'], inplace=True)
    # logging.info(f"Dropped {initial_rows - len(df)} rows with missing winner/loser IDs.")

    # Placeholder for more cleaning...
    logging.info("Data cleaning placeholder complete. Add more steps as needed.")

    # --- End cleaning steps ---

    logging.info(f"DataFrame shape after cleaning: {df.shape}")
    return df

def save_data(df, output_path='data/merged_atp_data.csv'):
    """Saves the DataFrame to a CSV file."""
    if df.empty:
        logging.warning("DataFrame is empty. Not saving.")
        return

    output_dir = os.path.dirname(output_path)
    if output_dir and not os.path.exists(output_dir):
        os.makedirs(output_dir)
        logging.info(f"Created directory: {output_dir}")

    try:
        df.to_csv(output_path, index=False)
        logging.info(f"Cleaned and merged data saved to: {output_path}")
    except Exception as e:
        logging.error(f"Error saving data to {output_path}: {e}")


if __name__ == "__main__":
    logging.info("Starting ATP Data Cleaning and Merging Script")

    # Use the default path ('data/all/') defined in the function
    dataframes = load_data()
    merged_dataframe = merge_data(dataframes)

    if not merged_dataframe.empty:
        logging.info("First 5 rows of merged data:")
        print(merged_dataframe.head()) # Use print for direct output in script
        logging.info("Columns and data types:")
        merged_dataframe.info(verbose=True) # verbose=True for more details

        cleaned_dataframe = clean_data(merged_dataframe)
        save_data(cleaned_dataframe, output_path='data/merged_atp_data.csv')
    else:
        logging.warning("Pipeline halted as no data could be loaded or merged.")

    logging.info("Script finished.") 