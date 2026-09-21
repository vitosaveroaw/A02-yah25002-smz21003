from sklearn.datasets import fetch_california_housing
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import numpy as np
import os
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

# Load California Housing dataset
housing = fetch_california_housing(as_frame=True)

# Features + target as a single DataFrame
df = housing.frame

# Quick check
print(df.head())
print(df.shape)

# Define features (X) and target (y)
X = df.drop('MedHouseVal', axis=1)
y = df['MedHouseVal']

# Split the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Check the shape of the training and testing sets
print(f"X_train shape: {X_train.shape}")
print(f"X_test shape: {X_test.shape}")
print(f"y_train shape: {y_train.shape}")
print(f"y_test shape: {y_test.shape}")

# Initialize MLPRegressor with early stopping and custom hyperparameters
mlp_regressor = MLPRegressor(
    hidden_layer_sizes=(10, 5), # Custom hyperparameter: two hidden layers with 10 and 5 neurons
    activation='relu',
    solver='adam',
    early_stopping=True,          # Enable early stopping
    max_iter=1000,                # Maximum number of iterations for the solver to converge
    batch_size=100,                # Custom hyperparameter: Use mini-batches of size 100
    random_state=42
)

# Train the model
mlp_regressor.fit(X_train, y_train)
print("MLPRegressor model trained successfully with early stopping.")
 
# Create predictions on the training set
y_train_pred = mlp_regressor.predict(X_train)

# Make sure the figures folder exists
os.makedirs('figures', exist_ok=True)

#Plot actual vs.predicted valuesfor the training set
plt.figure(figsize=(6, 6))
plt.scatter(y_train, y_train_pred, alpha=0.3, s=10)
plt.plot(
    [y_train.min(), y_train.max()], 
    [y_train.min(), y_train.max()], "r--",
    linewidth=2,
) 
plt.xlabel ("Actual median house value")
plt.ylabel("Predicted median house value")
plt.title("Actual vs. Predicted Train Set")
plt.grid(True)
plt.tight_layout()
plt.savefig("figures/train_actual_vs_pred.png", dpi=150)
plt.close()

print("Saved figures/train_actual_vs_pred.png")

# Create predictions on the test set
y_test_pred = mlp_regressor.predict(X_test)

plt.figure(figsize=(6, 6))
plt.scatter(y_test, y_test_pred, alpha=0.3, s=10)
plt.plot(
    [y_test.min(), y_test.max()],
    [y_test.min(), y_test.max()],
    "r--",
    linewidth=2,
)
plt.xlabel("Actual median house value")
plt.ylabel("Predicted median house value")
plt.title("Actual vs. Predicted Test Set")
plt.grid(True)
plt.tight_layout()
plt.savefig("figures/test_actual_vs_pred.png", dpi=150)
plt.close()

print("Saved figures/test_actual_vs_pred.png")

# Metrics Evaluation
# --- Evaluate Training Set Performance ---
print("\n--- Training Set Metrics ---")
mae_train = mean_absolute_error(y_train, y_train_pred)
mse_train = mean_squared_error(y_train, y_train_pred)
rmse_train = np.sqrt(mse_train)
r2_train = r2_score(y_train, y_train_pred)

print(f"Mean Absolute Error (MAE): {mae_train:.4f}")
print(f"Mean Squared Error (MSE): {mse_train:.4f}")
print(f"Root Mean Squared Error (RMSE): {rmse_train:.4f}")
print(f"R-squared (R2 Score): {r2_train:.4f}")

# --- Evaluate Test Set Performance ---
print("\n--- Test Set Metrics ---")
mae_test = mean_absolute_error(y_test, y_test_pred)
mse_test = mean_squared_error(y_test, y_test_pred)
rmse_test = np.sqrt(mse_test)
r2_test = r2_score(y_test, y_test_pred)

print(f"Mean Absolute Error (MAE): {mae_test:.4f}")
print(f"Mean Squared Error (MSE): {mse_test:.4f}")
print(f"Root Mean Squared Error (RMSE): {rmse_test:.4f}")
print(f"R-squared (R2 Score): {r2_test:.4f}")

# Save metrics to a text file
# Define the output directory for metrics
metrics_output_dir = 'metrics'
os.makedirs(metrics_output_dir, exist_ok=True)

# Prepare the metrics string to be saved
metrics_content = f"""
--- Training Set Metrics ---
Mean Absolute Error (MAE): {mae_train:.4f}
Mean Squared Error (MSE): {mse_train:.4f}
Root Mean Squared Error (RMSE): {rmse_train:.4f}
R-squared (R2 Score): {r2_train:.4f}

--- Test Set Metrics ---
Mean Absolute Error (MAE): {mae_test:.4f}
Mean Squared Error (MSE): {mse_test:.4f}
Root Mean Squared Error (RMSE): {rmse_test:.4f}
R-squared (R2 Score): {r2_test:.4f}
"""

# Define the file path
metrics_file_path = os.path.join(metrics_output_dir, 'regression_metrics.txt')

# Write the metrics to the file
with open(metrics_file_path, 'w') as f:
    f.write(metrics_content)

print(f"Regression metrics saved successfully to '{metrics_file_path}'")

# Residuals
# Residual plot for the test set
residuals = y_test - y_test_pred
plt.figure(figsize=(6, 6))
plt.scatter(y_test_pred, residuals, alpha=0.3, s=10, color="green")
plt.grid(True)
plt.axhline(y=0, color="r", linestyle="--", linewidth=2)
plt.xlabel("Predicted median house value")
plt.ylabel("Residuals(Actual - Predicted)")
plt.title("Residuals plot for Test Set")
plt.grid(True)
plt.tight_layout()
plt.savefig("figures/test_residuals.png", dpi=150)
plt.close()
print("Saved figures/test_residuals.png")