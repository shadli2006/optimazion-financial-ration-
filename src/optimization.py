### 6. src/optimization.py
python
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pypfopt import EfficientFrontier, risk_models
from deap import base, creator, tools, algorithms
import random
from config import FIGURES_DIR, TABLES_DIR, OPTIMIZATION_PARAMS

def markowitz_optimization(expected_returns, cov_matrix, bank_names):
    """Optimasi portofolio menggunakan Model Markowitz"""
    ef = EfficientFrontier(expected_returns, cov_matrix)
    weights = ef.max_sharpe()
    cleaned_weights = ef.clean_weights()
    
    # Konversi ke Series
    weights_series = pd.Series(cleaned_weights)
    
    # Visualisasi
    plt.figure(figsize=(12, 6))
    weights_series.sort_values().plot(kind='barh')
    plt.title('Komposisi Portofolio Optimal (Markowitz)')
    plt.xlabel('Bobot')
    
    markowitz_path = os.path.join(FIGURES_DIR, 'markowitz_portfolio.png')
    plt.savefig(markowitz_path)
    plt.close()
    
    return weights_series, markowitz_path

def genetic_algorithm_optimization(expected_returns, health_scores, cov_matrix, bank_names):
    """Optimasi portofolio menggunakan Algoritma Genetik"""
    n_assets = len(expected_returns)
    
    # Setup DEAP
    creator.create("FitnessMulti", base.Fitness, weights=(1.0, -1.0, 1.0))
    creator.create("Individual", list, fitness=creator.FitnessMulti)
    
    toolbox = base.Toolbox()
    toolbox.register("attr_float", random.uniform, 0, 1)
    toolbox.register("individual", tools.initRepeat, creator.Individual, toolbox.attr_float, n_assets)
    toolbox.register("population", tools.initRepeat, list, toolbox.individual)
    
    def evaluate(individual):
        weights = np.array(individual)
        weights /= np.sum(weights)
        
        # 1. Return ekspektasi
        port_return = np.dot(weights, expected_returns)
        
        # 2. Risiko (volatilitas)
        port_volatility = np.sqrt(np.dot(weights.T, np.dot(cov_matrix, weights)))
        
        # 3. Skor kesehatan
        health_score = np.dot(weights, health_scores)
        
        return port_return, port_volatility, health_score
    
    toolbox.register("evaluate", evaluate)
    toolbox.register("mate", tools.cxBlend, alpha=0.5)
    toolbox.register("mutate", tools.mutGaussian, mu=0, sigma=0.1, indpb=0.2)
    toolbox.register("select", tools.selNSGA2)
    
    # Jalankan algoritma
    population = toolbox.population(n=OPTIMIZATION_PARAMS['pop_size'])
    algorithms.eaMuPlusLambda(
        population, toolbox,
        mu=OPTIMIZATION_PARAMS['pop_size'], 
        lambda_=OPTIMIZATION_PARAMS['pop_size']*2,
        cxpb=0.7, 
        mutpb=0.3, 
        ngen=OPTIMIZATION_PARAMS['n_gen'],
        stats=None, 
        halloffame=None, 
        verbose=False
    )
    
    # Ambil solusi terbaik
    best_ind = tools.selBest(population, k=1)[0]
    weights = np.array(best_ind)
    weights /= np.sum(weights)
    
    # Visualisasi
    plt.figure(figsize=(12, 6))
    sorted_idx = np.argsort(weights)
    plt.barh(np.array(bank_names)[sorted_idx], weights[sorted_idx])
    plt.title('Komposisi Portofolio Optimal (Algoritma Genetik)')
    plt.xlabel('Bobot')
    
    ga_path = os.path.join(FIGURES_DIR, 'ga_portfolio.png')
    plt.savefig(ga_path)
    plt.close()
    
    return weights, ga_path

def prepare_optimization_data(df, model, scaler, thresholds):
    """Siapkan data untuk optimasi portofolio"""
    # Hitung skor kesehatan
    df['Health_Score'] = df.apply(
        lambda row: calculate_health_score(row, thresholds), 
        axis=1
    )
    
    # Ambil data terbaru
    latest_data = df.groupby('Nama_Bank').last().reset_index()
    
    # Hitung return ekspektasi
    X = latest_data[['ROA', 'ROE', 'NIM', 'NPL', 'Harga_Saham']]
    X_scaled = scaler.transform(X)
    latest_data['Prediksi_Harga'] = model.predict(X_scaled)
    latest_data['Expected_Return'] = (latest_data['Prediksi_Harga'] - latest_data['Harga_Saham']) / latest_data['Harga_Saham']
    
    # Siapkan data return historis
    returns_data = df.pivot_table(
        index='Periode', 
        columns='Nama_Bank', 
        values='Return_Historis',
        aggfunc='mean'
    ).fillna(0)
    
    return latest_data, returns_data


