### 3. src/preprocessing.py
python
import pandas as pd
import numpy as np
from config import PROCESSED_DATA_DIR

def preprocess_data(df):
    """Preprocessing data keuangan"""
    # Validasi kolom
    required_cols = ['Nama_Bank', 'Periode', 'ROA', 'ROE', 'NIM', 'NPL', 'Harga_Saham']
    if not all(col in df.columns for col in required_cols):
        missing = [col for col in required_cols if col not in df.columns]
        print(f"Kolom wajib tidak ditemukan: {missing}")
        return None
    
    # Konversi tipe data
    numeric_cols = ['ROA', 'ROE', 'NIM', 'NPL', 'Harga_Saham']
    for col in numeric_cols:
        df[col] = pd.to_numeric(df[col], errors='coerce')
    
    # Handle missing values
    initial_count = len(df)
    df = df.dropna(subset=numeric_cols)
    
    # Buat target variable
    df['Harga_Saham_Next'] = df.groupby('Nama_Bank')['Harga_Saham'].shift(-1)
    df = df.dropna(subset=['Harga_Saham_Next'])
    
    # Hitung return historis
    df['Return_Historis'] = df.groupby('Nama_Bank')['Harga_Saham'].pct_change()
    
    # Urutkan data
    df = df.sort_values(by=['Nama_Bank', 'Periode'])
    
    print(f"Data setelah preprocessing: {len(df)} baris (Dihapus {initial_count - len(df)})")
    
    # Simpan data yang telah diproses
    processed_path = os.path.join(PROCESSED_DATA_DIR, 'processed_data.csv')
    df.to_csv(processed_path, index=False)
    print(f"Data yang telah diproses disimpan di: {processed_path}")
    
    return df
