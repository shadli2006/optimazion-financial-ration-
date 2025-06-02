### 9. README.md
markdown
# Analisis Prediksi Harga Saham dan Optimasi Portofolio Bank

Repositori ini berisi implementasi model AI untuk memprediksi harga saham bank berdasarkan rasio keuangan dan melakukan optimasi portofolio.

## Struktur Repositori

financial-ratio-ai-prediction/
├── data/               # Data mentah dan yang telah diproses
├── models/             # Model machine learning
├── results/            # Hasil analisis
├── src/                # Kode sumber
│   ├── data_loader.py
│   ├── preprocessing.py
│   ├── analysis.py
│   ├── modeling.py
│   ├── optimization.py
│   └── main.py
├── config.py           # Konfigurasi
├── requirements.txt    # Dependensi
└── README.md           # Dokumentasi


## Instalasi
1. Clone repositori:
bash
git clone https://github.com/shadli2006/financial-ratio-ai-prediction.git
cd financial-ratio-ai-prediction


2. Instal dependensi:
bash
pip install -r requirements.txt


## Penggunaan
1. Tempatkan data di `data/raw/` atau gunakan URL/data online
2. Jalankan program:
bash
python src/main.py --data data/raw/data_keuangan.csv


Atau untuk data online:
bash
python src/main.py --data https://github.com/shadli2006/financial-ratio-ai-prediction/raw/main/data/raw/data_keuangan.csv


## Output
- Data yang telah diproses: `data/processed/processed_data.csv`
- Model prediksi: `models/stock_prediction_model.pkl`
- Hasil analisis: 
  - Grafik: `results/figures/`
  - Tabel: `results/tables/portfolio_results.csv`

## Parameter Konfigurasi
Ubah `config.py` untuk menyesuaikan:
- Parameter model (n_estimators, max_depth)
- Parameter optimasi (jumlah generasi, ukuran populasi)
- Ambang batas kesehatan bank (ROA, NPL)


### Cara Menjalankan Program:
1. Letakkan file data di data/raw/ (misal: data_keuangan.csv)
2. Install dependensi: pip install -r requirements.txt
3. Jalankan program:
bash
python src/main.py --data data_keuangan.csv


Atau untuk data dari GitHub:
bash
python src/main.py --data https://github.com/shadli2006/financial-ratio-ai-prediction/raw/main/data/raw/data_keuangan.csv


Struktur ini memberikan beberapa keunggulan:
1. *Modular* - Kode terpisah berdasarkan fungsi
2. *Reproduktif* - Konfigurasi terpusat
3. *Terorganisir* - Direktori jelas untuk setiap jenis aset
4. *Skalabel* - Mudah menambahkan fitur baru
5. *Dokumentasi* - README lengkap untuk pengguna# optimazion-financial-ration-
