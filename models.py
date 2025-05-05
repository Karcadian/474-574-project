from sklearn.linear_model import LinearRegression
from sklearn.svm import SVR
from sklearn.ensemble import RandomForestRegressor
from xgboost import XGBRegressor
from sklearn.metrics import mean_squared_error, r2_score

def train_and_evaluate(model, X_train, y_train, X_val, y_val, X_test, y_test):
    model.fit(X_train, y_train)
    y_val_pred = model.predict(X_val)
    y_test_pred = model.predict(X_test)

    results = {
        "model": model.__class__.__name__,
        "val_mse": mean_squared_error(y_val, y_val_pred),
        "val_r2": r2_score(y_val, y_val_pred),
        "test_mse": mean_squared_error(y_test, y_test_pred),
        "test_r2": r2_score(y_test, y_test_pred),
        "val_pred": y_val_pred,
        "test_pred": y_test_pred
    }
    return results

def get_models():
    return {
        "Linear Regression": LinearRegression(),
        "SVR (Linear Kernel)": SVR(kernel="linear", C=1, epsilon=0.1),
        "Random Forest": RandomForestRegressor(n_estimators=100, random_state=1),
        "XGBoost": XGBRegressor(n_estimators=100, learning_rate=0.1, random_state=1)
    }