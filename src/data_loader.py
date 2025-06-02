### 2. src/data_loader.py
python
import os
import pandas as pd
import requests
from io import StringIO
from config import RAW_DATA_DIR

def load_from_github(repo_url, file_path):
    """Muat data dari repository GitHub"""
    try:
        raw_url = repo_url.replace('github.com', 'raw.githubusercontent.com').replace('/blob/', '/')
        if '/tree/' in raw_url:
            raw_url = raw_url.replace('/tree/', '/')
        
        full_url = f"{raw_url.rstrip('/')}/{file_path.lstrip('/')}"
        print(f"Mengambil data dari: {full_url}")
        
        response = requests.get(full_url)
        response.raise_for_status()
        
        if file_path.endswith('.csv'):
            return pd.read_csv(StringIO(response.text))
        else:
            return pd.read_excel(StringIO(response.content))
    except Exception as e:
        print(f"Error GitHub: {str(e)}")
        return None

def load_data(source):
    """Muat data dari berbagai sumber"""
    try:
        # Jika sumber adalah path file lokal
        if os.path.isfile(source):
            print(f"Membaca file lokal: {source}")
            if source.endswith('.csv'):
                return pd.read_csv(source)
            elif source.endswith(('.xls', '.xlsx')):
                return pd.read_excel(source)
        
        # Jika sumber adalah URL GitHub
        elif "github.com" in source:
            parts = source.split('/blob/') if '/blob/' in source else source.split('/tree/')
            repo_url = parts[0]
            file_path = parts[1] if len(parts) > 1 else None
            return load_from_github(repo_url, file_path)
        
        # Jika sumber adalah URL umum
        elif source.startswith('http'):
            print(f"Membaca data dari URL: {source}")
            if source.endswith('.csv'):
                return pd.read_csv(source)
            else:
                return pd.read_excel(source)
        
        # Jika sumber adalah nama file dalam direktori data
        elif os.path.isfile(os.path.join(RAW_DATA_DIR, source)):
            file_path = os.path.join(RAW_DATA_DIR, source)
            return load_data(file_path)
        
        else:
            print("Sumber data tidak valid")
            return None
    except Exception as e:
        print(f"Error memuat data: {str(e)}")
        return None
