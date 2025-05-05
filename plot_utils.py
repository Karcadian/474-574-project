import os
import matplotlib.pyplot as plt

def plot_combined(y_true, preds_dict, title, save_path=None):
    plt.figure(figsize=(8, 6))
    for name, y_pred in preds_dict.items():
        plt.scatter(y_true, y_pred, alpha=0.4, label=name)
    plt.plot([y_true.min(), y_true.max()], [y_true.min(), y_true.max()], 'k--')
    plt.xlabel("Actual Calories Burned")
    plt.ylabel("Predicted Calories Burned")
    plt.title(title)
    plt.legend()
    plt.grid(True)
    plt.tight_layout()

    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path)

    plt.show()

def plot_single(y_true, y_pred, title, model_name, save_path=None):
    plt.figure(figsize=(6, 6))
    plt.scatter(y_true, y_pred, alpha=0.6, color='tab:blue')
    plt.plot([y_true.min(), y_true.max()], [y_true.min(), y_true.max()], 'k--')
    plt.xlabel("Actual Calories Burned")
    plt.ylabel("Predicted Calories Burned")
    plt.title(f"{model_name} – {title}")
    plt.grid(True)
    plt.tight_layout()

    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path)

    plt.show()