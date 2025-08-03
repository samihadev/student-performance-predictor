from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor, AdaBoostRegressor
from sklearn.metrics import mean_squared_error, r2_score
import joblib
from config import Config  # Import Config here


def train_and_evaluate_models(X_train, X_test, y_train, y_test):
    models = {
        'Random Forest': RandomForestRegressor(n_estimators=100, random_state=42),
        'Gradient Boosting': GradientBoostingRegressor(n_estimators=100, random_state=42),
        'AdaBoost': AdaBoostRegressor(n_estimators=100, random_state=42)
    }

    results = {}

    for name, model in models.items():
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)

        results[name] = {
            'model': model,
            'mse': mean_squared_error(y_test, y_pred),
            'rmse': mean_squared_error(y_test, y_pred, squared=False),
            'r2': r2_score(y_test, y_pred),
            'y_test': y_test,
            'y_pred': y_pred
        }

    # Save best model using Config.MODEL_FILE
    best_model_name = max(results, key=lambda k: results[k]['r2'])
    best_model = results[best_model_name]['model']
    joblib.dump(best_model, Config.MODEL_FILE)

    return results