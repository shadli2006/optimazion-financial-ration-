import pandas as pd
import logging
from config.user_config import USER_CONFIG

def load_data(file_path):
    try:
        df = pd.read_csv(file_path)
        logging.info(f"Loaded {len(df)} rows.")
        return df
    except Exception as e:
        logging.error(f"Load error: {e}")
        return None

def filter_banks(df):
    if USER_CONFIG['SELECTED_BANKS']:
        return df[df['Nama_Bank'].isin(USER_CONFIG['SELECTED_BANKS'])]
    return df
