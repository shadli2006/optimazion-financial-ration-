### 7. src/main.py
python
import os
import argparse
import pandas as pd
from config import RAW_DATA_DIR, FIGURES_DIR, TABLES_DIR, OPTIMIZATION_PARAMS
from .data_loader import load_data
from .preprocessing import preprocess_data
from .analysis import financial_analysis, calculate_health_score
from .modeling import build_prediction_model
from .optimization import (
    prepare_optimization_data,
    markowitz_optimization,
    genetic_algorithm_optimization
)

def main():
    parser = argparse.ArgumentParser(description='Analisis Prediksi Saham dan Optimasi Portofolio Bank')
    parser.add_argument('--data', type=str, required=True, 
                        help='Path/URL ke file data atau nama file di data/raw')
    args = parser.parse_args()
    
    print("="*50)
    print("ANALISIS PREDIKSI HARGA SAHAM DAN OPTIMASI PORTOFOLIO")
    print("="*50)
    
    # 1. Load data
    print("\n[1/5] Memuat data...")
    df = load_data(args.data)
    if df is None:
        print("Gagal memuat data. Keluar dari program.")
        return
    
    # 2. Preprocessing data
    print("\n[2/5] Preprocessing data...")
    df = preprocess_data(df)
    if df is None:
        print("Gagal memproses data. Keluar dari program.")
        return
    
    # 3. Analisis keuangan
    print("\n[3/5] Melakukan analisis keuangan...")
    corr_path = financial_analysis(df)
    print(f"  - Grafik korelasi disimpan di: {corr_path}")
    
    # 4. Bangun model prediksi
    print("\n[4/5] Membangun model prediksi harga saham...")
    model, scaler, mse, r2, pred_path, importance_path = build_prediction_model(df)
    print(f"  - Evaluasi Model: MSE={mse:.4f}, R²={r2:.4f}")
    print(f"  - Grafik prediksi disimpan di: {pred_path}")
    print(f"  - Grafik feature importance disimpan di: {importance_path}")
    
    # 5. Optimasi portofolio
    print("\n[5/5] Melakukan optimasi portofolio...")
    latest_data, returns_data = prepare_optimization_data(
        df, model, scaler, OPTIMIZATION_PARAMS['health_thresholds']
    )
    
    # Matriks kovariansi
    cov_matrix = returns_data.cov()
    
    # Optimasi Markowitz
    markowitz_weights, markowitz_path = markowitz_optimization(
        latest_data.set_index('Nama_Bank')['Expected_Return'],
        cov_matrix,
        latest_data['Nama_Bank'].unique()
    )
    
    # Optimasi Algoritma Genetik
    ga_weights, ga_path = genetic_algorithm_optimization(
        latest_data['Expected_Return'].values,
        latest_data['Health_Score'].values,
        cov_matrix.values,
        latest_data['Nama_Bank'].unique()
    )
    
    # Gabungkan hasil
    results_df = latest_data[['Nama_Bank', 'ROA', 'ROE', 'NIM', 'NPL', 'Health_Score', 'Harga_Saham', 'Prediksi_Harga', 'Expected_Return']]
    results_df['Markowitz_Weight'] = results_df['Nama_Bank'].map(markowitz_weights)
    results_df['GA_Weight'] = ga_weights
    
    # Simpan hasil
    results_path = os.path.join(TABLES_DIR, 'portfolio_results.csv')
    results_df.to_csv(results_path, index=False)
    
    print("\n" + "="*50)
    print("HASIL AKHIR")
    print("="*50)
    print(f"  - Hasil Markowitz disimpan di: {markowitz_path}")
    print(f"  - Hasil Algoritma Genetik disimpan di: {ga_path}")
    print(f"  - Tabel hasil lengkap disimpan di: {results_path}")
    
    # Tampilkan rekomendasi portofolio
    print("\nRekomendasi Portofolio:")
    print(results_df[['Nama_Bank', 'Markowitz_Weight', 'GA_Weight']].sort_values('Markowitz_Weight', ascending=False))

if __name__ == "__main__":
    main()
