import numpy as np
import pandas as pd
from scipy.optimize import minimize
import matplotlib.pyplot as plt
from config.user_config import USER_CONFIG
from .visualization import currency_formatter

def calculate_health_score(row):
    score = 0
    for ratio in USER_CONFIG['HEALTH_RATIOS']:
        if ratio in row and ratio in USER_CONFIG['HEALTH_THRESHOLDS']:
            threshold = USER_CONFIG['HEALTH_THRESHOLDS'][ratio]
            if ratio in ['ROA', 'ROE', 'CAR', 'NIM']:
                if row[ratio] >= threshold:
                    score += 1
            elif ratio in ['NPL', 'DAR', 'DER']:
                if row[ratio] <= threshold:
                    score += 1
    return score

def optimize_portfolio(df):
    df['Health_Score'] = df.apply(calculate_health_score, axis=1)
    df['Expected_Return'] = (df['Prediksi_Harga'] - df['Harga_Saham']) / df['Harga_Saham']
    latest = df.groupby('Nama_Bank').last().reset_index()
    cov_matrix = df.pivot_table(index='Periode', columns='Nama_Bank', values='Harga_Saham').cov().values
    returns = latest['Expected_Return'].values
    n_assets = len(returns)

    def port_perf(weights):
        port_return = np.dot(weights, returns)
        port_vol = np.sqrt(np.dot(weights.T, np.dot(cov_matrix, weights)))
        return -port_return / port_vol

    constraints = {'type': 'eq', 'fun': lambda w: np.sum(w) - 1}
    bounds = tuple((0, 1) for _ in range(n_assets))
    init_guess = [1/n_assets] * n_assets

    result = minimize(port_perf, init_guess, method='SLSQP', bounds=bounds, constraints=constraints)
    if not result.success:
        print("Optimasi gagal:", result.message)
        return

    latest['Markowitz_Weight'] = result.x
    latest = latest.sort_values(by=['Health_Score', 'Expected_Return'], ascending=[False, False])
    top = latest.head(USER_CONFIG['MAX_BANKS_IN_PORTFOLIO']).copy()
    top['Markowitz_Weight'] = top['Markowitz_Weight'] / top['Markowitz_Weight'].sum()

    def recommend(row):
        threshold = len(USER_CONFIG['HEALTH_RATIOS'])
        return 'Beli' if row['Health_Score'] >= 0.75 * threshold else 'Tahan' if row['Health_Score'] >= 0.5 * threshold else 'Jual'

    top['Recommendation'] = top.apply(recommend, axis=1)

    plt.figure(figsize=(7, 7))
    plt.pie(top['Markowitz_Weight'], labels=top['Nama_Bank'], autopct='%1.1f%%', startangle=140)
    plt.title('Alokasi Portofolio Optimal')
    plt.axis('equal')
    plt.show()

    print(top[['Nama_Bank', 'Markowitz_Weight', 'Health_Score', 'Expected_Return', 'Recommendation']])
    return top
