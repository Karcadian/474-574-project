import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.svm import SVR
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.preprocessing import StandardScaler

csv_filename = "gym_members_exercise_tracking.csv"

data = pd.read_csv(csv_filename)

columns_to_drop = [
    "Member_ID",
    "Session_ID",
    "Workout_Frequency",     # not useful for per-session
    "Workout_Type",          # categorical
    "Experience_Level",      # subjective, hard to quantify
    "Water_Intake_Liters",   # not directly tied to calories burned
    "Resting_Heart_Rate"     # not helpful for exercise relation
]

# dropping columns
existing_drop = [col for col in columns_to_drop if col in data.columns]
data.drop(columns=existing_drop, inplace=True)

# dropping rows with faulty data
data.dropna(inplace=True)

# turn Male/Female into numeric
X = pd.get_dummies(data.drop(columns=["Calories_Burned"]), drop_first=True)
y = data["Calories_Burned"]

# normalize feature matrix for SVR
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# 70% training, 15% testing, 15% validation
# first split the testing into 15%
X_train_val, X_test, y_train_val, y_test = train_test_split(
    X_scaled, y, test_size=0.15, random_state=1
)

# 0.1765 * 0.85 = 0.15
X_train, X_val, y_train, y_val = train_test_split(
    X_train_val, y_train_val, test_size=0.1765, random_state=1
)

# train models
lin_reg = LinearRegression()
lin_reg.fit(X_train, y_train)

svr_reg = SVR(kernel="linear")
svr_reg.fit(X_train, y_train)

y_pred_lin_val = lin_reg.predict(X_val)
y_pred_svr_val = svr_reg.predict(X_val)

# calculate mse and r^2
mse_lin_val = mean_squared_error(y_val, y_pred_lin_val)
r2_lin_val = r2_score(y_val, y_pred_lin_val)

mse_svr_val = mean_squared_error(y_val, y_pred_svr_val)
r2_svr_val = r2_score(y_val, y_pred_svr_val)

print("=== Validation Results ===")
print("Linear Regression - MSE:", mse_lin_val, "R²:", r2_lin_val)
print("SVR (Linear)      - MSE:", mse_svr_val, "R²:", r2_svr_val)

y_pred_lin_test = lin_reg.predict(X_test)
y_pred_svr_test = svr_reg.predict(X_test)

mse_lin_test = mean_squared_error(y_test, y_pred_lin_test)
r2_lin_test = r2_score(y_test, y_pred_lin_test)

mse_svr_test = mean_squared_error(y_test, y_pred_svr_test)
r2_svr_test = r2_score(y_test, y_pred_svr_test)

print("\n=== Test Results ===")
print("Linear Regression - MSE:", mse_lin_test, "R^2:", r2_lin_test)
print("SVR (Linear)      - MSE:", mse_svr_test, "R^2:", r2_svr_test)

plt.figure(figsize=(12, 5))

plt.subplot(1, 2, 1)
plt.scatter(y_val, y_pred_lin_val, alpha=0.6, label="Linear Regression")
plt.scatter(y_val, y_pred_svr_val, alpha=0.6, label="SVR (Linear)", color="red")
plt.plot([y_val.min(), y_val.max()], [y_val.min(), y_val.max()], "k--", lw=2)
plt.xlabel("Actual Calories Burned")
plt.ylabel("Predicted Calories Burned")
plt.title("Validation: Predicted vs Actual")
plt.legend()
plt.grid(True)

plt.subplot(1, 2, 2)
plt.scatter(y_test, y_pred_lin_test, alpha=0.6, label="Linear Regression")
plt.scatter(y_test, y_pred_svr_test, alpha=0.6, label="SVR (Linear)", color="red")
plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], "k--", lw=2)
plt.xlabel("Actual Calories Burned")
plt.ylabel("Predicted Calories Burned")
plt.title("Test: Predicted vs Actual")
plt.legend()
plt.grid(True)

plt.tight_layout()
plt.show()
