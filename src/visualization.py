import matplotlib.pyplot as plt
import matplotlib.dates as mdates

def currency_formatter(x, pos):
    return f'Rp{x:,.0f}'

def percent_formatter(x, pos):
    return f'{x:.1%}'

def plot_predictions(df):
    plt.figure(figsize=(14, 8))
    banks = df['Nama_Bank'].unique()

    for i, bank in enumerate(banks):
        sub = df[df['Nama_Bank'] == bank].sort_values('Periode')
        plt.subplot(len(banks), 1, i + 1)
        plt.plot(sub['Periode'], sub['Harga_Saham'], 'o-', label='Aktual')
        plt.plot(sub['Periode'], sub['Prediksi_Harga'], 's--', label='Prediksi')
        plt.title(f'{bank} - Harga Saham')
        plt.ylabel('Harga')
        plt.legend()
        plt.grid(True)
        plt.gca().xaxis.set_major_formatter(mdates.DateFormatter('%b %Y'))
        plt.gca().yaxis.set_major_formatter(currency_formatter)

    plt.tight_layout()
    plt.show()
