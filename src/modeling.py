### 5. src/modeling.py
python
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import joblib
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.preprocessing import StandardScaler
from config import MODELS_DIR, FIGURES_DIR, MODEL_PARAMS

def build_prediction_model(df):
    """Bangun dan latih model prediksi harga saham"""
    X = df[['ROA', 'ROE', 'NIM', 'NPL', 'Harga_Saham']]
    y = df['Harga_Saham_Next']
    
    # Normalisasi data
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        X_scaled, y, 
        test_size=MODEL_PARAMS['test_size'], 
        random_state=MODEL_PARAMS['random_state']
    )
    
    # Bangun model
    model = RandomForestRegressor(
        n_estimators=MODEL_PARAMS['n_estimators'],
        max_depth=MODEL_PARAMS['max_depth'],
        random_state=MODEL_PARAMS['random_state']
    )
    model.fit(X_train, y_train)
    
    # Evaluasi model
    y_pred = model.predict(X_test)
    mse = mean_squared_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)
    
    # Visualisasi hasil prediksi
    plt.figure(figsize=(10, 6))
    plt.scatter(y_test, y_pred, alpha=0.5)
    plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'k--')
    plt.xlabel('Harga Aktual')
    plt.ylabel('Harga Prediksi')
    plt.title('Prediksi vs Aktual Harga Saham')
    
    pred_path = os.path.join(FIGURES_DIR, 'prediction_vs_actual.png')
    plt.savefig(pred_path)
    plt.close()
    
    # Feature importance
    feature_importances = pd.Series(model.feature_importances_, index=X.columns)
    plt.figure(figsize=(10, 6))
    feature_importances.sort_values().plot(kind='barh')
    plt.title('Feature Importance untuk Prediksi Harga Saham')
    
    importance_path = os.path.join(FIGURES_DIR, 'feature_importance.png')
    plt.savefig(importance_path)
    plt.close()
    
    # Simpan model dan scaler
    model_path = os.path.join(MODELS_DIR, 'stock_prediction_model.pkl')
    scaler_path = os.path.join(MODELS_DIR, 'scaler.pkl')
    
    joblib.dump(model, model_path)
    joblib.dump(scaler, scaler_path)
    
    return model, scaler, mse, r2, pred_path, importance_path

