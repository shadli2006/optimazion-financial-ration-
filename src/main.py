import logging
from data_loader import load_data, filter_banks
from preprocessing import preprocess_data
from modeling import build_prediction_model
from analysis import analyze_ratio_impact
from optimization import optimize_portfolio
from visualization import plot_predictions

logging.basicConfig(level=logging.INFO)

def main():
    df = load_data("data/data_saham_bbca_bnga_2024.csv")
    if df is None:
        return
    df = preprocess_data(df)
    df = filter_banks(df)
    model, df = build_prediction_model(df)
    if model is None:
        return
    plot_predictions(df)
    analyze_ratio_impact(df)
    optimize_portfolio(df)

if __name__ == "__main__":
    main()
