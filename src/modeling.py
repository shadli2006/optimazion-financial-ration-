from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score
import logging
from config.user_config import USER_CONFIG

def build_prediction_model(df):
    try:
        features = [f for f in USER_CONFIG['PREDICTION_FEATURES'] if f in df.columns]
        features.append('Harga_Saham')
        X = df[features]
        y = df['Harga_Saham_Next']

        X_train, X_test, y_train, y_test = train_test_split(
            X, y,
            test_size=USER_CONFIG['MODEL_PARAMS']['test_size'],
            random_state=USER_CONFIG['MODEL_PARAMS']['random_state']
        )

        model = RandomForestRegressor(
            n_estimators=USER_CONFIG['MODEL_PARAMS']['n_estimators'],
            max_depth=USER_CONFIG['MODEL_PARAMS']['max_depth'],
            random_state=USER_CONFIG['MODEL_PARAMS']['random_state']
        )
        model.fit(X_train, y_train)
        df['Prediksi_Harga'] = model.predict(X)

        mse = mean_squared_error(y_test, model.predict(X_test))
        r2 = r2_score(y_test, model.predict(X_test))

        logging.info(f"Model MSE: {mse:.4f}, R²: {r2:.4f}")
        return model, df
    except Exception as e:
        logging.error(f"Model error: {e}")
        return None, df
