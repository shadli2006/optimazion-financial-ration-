import matplotlib.pyplot as plt
import seaborn as sns
from config.user_config import USER_CONFIG
from .visualization import currency_formatter, percent_formatter

def analyze_ratio_impact(df):
    ratios_to_show = [r for r in USER_CONFIG['PREDICTION_FEATURES'] if r in df.columns and r != 'Harga_Saham']
    if not ratios_to_show:
        print("Tidak ada rasio untuk divisualisasikan.")
        return

    n_cols = 2
    n_rows = (len(ratios_to_show) + 1) // n_cols
    plt.figure(figsize=(16, 6 * n_rows))

    for i, ratio in enumerate(ratios_to_show):
        plt.subplot(n_rows, n_cols, i + 1)
        sns.regplot(x=ratio, y='Prediksi_Harga', data=df,
                    scatter_kws={'s': 60, 'alpha': 0.6},
                    line_kws={'color': 'red', 'linestyle': '--'})
        corr = df[[ratio, 'Prediksi_Harga']].corr().iloc[0, 1]
        plt.title(f'{ratio} vs Prediksi Harga (Korelasi: {corr:.2f})')
        plt.xlabel(ratio)
        plt.ylabel('Prediksi Harga')
        plt.grid(True)
        plt.gca().xaxis.set_major_formatter(percent_formatter)
        plt.gca().yaxis.set_major_formatter(currency_formatter)

    plt.tight_layout()
    plt.show()
