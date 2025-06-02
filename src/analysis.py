### 4. src/analysis.py
python
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from config import FIGURES_DIR

def financial_analysis(df):
    """Analisis rasio keuangan dan korelasi"""
    # Analisis korelasi
    corr_matrix = df[['ROA', 'ROE', 'NIM', 'NPL', 'Harga_Saham']].corr()
    
    plt.figure(figsize=(10, 8))
    sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', fmt=".2f")
    plt.title('Korelasi Antar Variabel')
    
    corr_path = os.path.join(FIGURES_DIR, 'financial_correlation.png')
    plt.savefig(corr_path)
    plt.close()
    
    return corr_path

def calculate_health_score(row, thresholds):
    """Hitung skor kesehatan bank"""
    score = 0
    if row['ROA'] >= thresholds['ROA']: score += 1
    if row['NPL'] <= thresholds['NPL']: score += 1
    return score
