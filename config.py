### 1. config.py
python
import os

# Konfigurasi direktori
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(BASE_DIR, 'data')
RAW_DATA_DIR = os.path.join(DATA_DIR, 'raw')
PROCESSED_DATA_DIR = os.path.join(DATA_DIR, 'processed')
MODELS_DIR = os.path.join(BASE_DIR, 'models')
RESULTS_DIR = os.path.join(BASE_DIR, 'results')
FIGURES_DIR = os.path.join(RESULTS_DIR, 'figures')
TABLES_DIR = os.path.join(RESULTS_DIR, 'tables')

# Konfigurasi model
MODEL_PARAMS = {
    'test_size': 0.2,
    'random_state': 42,
    'n_estimators': 200,
    'max_depth': 10
}

# Konfigurasi optimasi
OPTIMIZATION_PARAMS = {
    'n_gen': 50,
    'pop_size': 100,
    'health_thresholds': {
        'ROA': 1.5,
        'NPL': 3.0
    }
}

# Buat direktori jika belum ada
for dir_path in [RAW_DATA_DIR, PROCESSED_DATA_DIR, MODELS_DIR, FIGURES_DIR, TABLES_DIR]:
    os.makedirs(dir_path, exist_ok=True)
