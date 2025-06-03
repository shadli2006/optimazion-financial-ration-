import pandas as pd
import logging

def preprocess_data(df):
    try:
        percent_cols = ['ROA', 'ROE', 'NIM', 'NPL', 'CAR']
        for col in percent_cols:
            if col in df.columns and df[col].dtype == 'object':
                df[col] = df[col].str.replace('%', '', regex=False).astype(float) / 100

        if 'Harga_Saham' in df.columns:
            df['Harga_Saham'] = pd.to_numeric(df['Harga_Saham'], errors='coerce')
            df['Harga_Saham'] = df.groupby('Nama_Bank')['Harga_Saham'].ffill().bfill()

        df['Harga_Saham_Next'] = df.groupby('Nama_Bank')['Harga_Saham'].shift(-1)
        df = df.dropna(subset=['Harga_Saham_Next'])

        if 'Periode' in df.columns:
            df['Periode'] = df['Periode'].str.replace('Q1', '03-31').str.replace('Q2', '06-30')
            df['Periode'] = df['Periode'].str.replace('Q3', '09-30').str.replace('Q4', '12-31')
            df['Periode'] = pd.to_datetime(df['Periode'] + '-2024', format='%m-%d-%Y', errors='coerce')
        return df
    except Exception as e:
        logging.error(f"Preprocessing error: {e}")
        return None
