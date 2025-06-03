USER_CONFIG = {
    'PREDICTION_FEATURES': ['ROA', 'ROE', 'NIM', 'NPL', 'CAR', 'EPS'],
    'SELECTED_BANKS': ['BBCA', 'BNGA', 'BMRI', 'BRI', 'MANDIRI'],
    'HEALTH_RATIOS': ['ROA', 'NPL', 'CAR'],
    'HEALTH_THRESHOLDS': {
        'ROA': 0.015,
        'NPL': 0.03,
        'CAR': 0.12
    },
    'MODEL_PARAMS': {
        'test_size': 0.2,
        'random_state': 42,
        'n_estimators': 200,
        'max_depth': 10
    },
    'MAX_BANKS_IN_PORTFOLIO': 4
}
