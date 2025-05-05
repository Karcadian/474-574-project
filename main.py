import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from models import get_models, train_and_evaluate
from plot_utils import plot_combined, plot_single

# take out unnecessary columns
data = pd.read_csv("gym_members_exercise_tracking.csv")
drop_cols = ["Member_ID", "Session_ID", "Workout_Frequency", "Workout_Type",
             "Experience_Level", "Water_Intake_Liters", "Resting_Heart_Rate"]
data.drop(columns=[col for col in drop_cols if col in data.columns], inplace=True)
data.dropna(inplace=True)

# one-hot encoding
X = pd.get_dummies(data.drop(columns=["Calories_Burned"]), drop_first=True)
y = data["Calories_Burned"]

# making mean = 0 and standard deviation = 1
scaler = StandardScaler() #
X_scaled = scaler.fit_transform(X)

# split training and validation sets -- 85% used for training + validation, 15% for testing
X_train_val, X_test, y_train_val, y_test = train_test_split(X_scaled, y, test_size=0.15, random_state=1)
X_train, X_val, y_train, y_val = train_test_split(X_train_val, y_train_val, test_size=0.1765, random_state=1)

models = get_models()
results = {}
val_preds = {}
test_preds = {}

for name, model in models.items():
    result = train_and_evaluate(model, X_train, y_train, X_val, y_val, X_test, y_test)
    results[name] = result
    val_preds[name] = result["val_pred"]
    test_preds[name] = result["test_pred"]

    print(f"\n{name}")
    print(f"Validation MSE: {result['val_mse']:.2f}, R^2: {result['val_r2']:.4f}")
    print(f"Test MSE: {result['test_mse']:.2f}, R^2: {result['test_r2']:.4f}")

plot_combined(y_val, val_preds, "Validation Set: All Models", save_path="plots/combined_validation.png")
plot_combined(y_test, test_preds, "Test Set: All Models", save_path="plots/combined_test.png")

for name in models.keys():
    filename_base = name.lower().replace(" ", "_").replace("(", "").replace(")", "")
    plot_single(y_val, val_preds[name], "Validation Set", name, save_path=f"plots/{filename_base}_val.png")
    plot_single(y_test, test_preds[name], "Test Set", name, save_path=f"plots/{filename_base}_test.png")