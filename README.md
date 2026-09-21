# A02 Assignment - MLPRegressor
Ping pong assignment <br>
Python version: 3.14.5

## Description
This project trains California Housing dataset using an `MLPRegressor` to predict median house value from features such as median income, house age, and location. The script trains the model twice, raw features and standardized (scaled) features, to compare how feature scaling affects performance. Evaluation metrics and diagnostic plots are generated for both runs.

## How to run the code:
Clone the repository by running this in your terminal:
```
git clone https://github.com/vitosaveroaw/A02-yah25002-sm21003
```
Go to the cloned repository
```
cd A02-yah25002-sm21003
```
Install the requirements by running this in your terminal
```
pip install -r requirements.txt
```
Run the python script in your terminal
```
python src/a02.py
```

## Output
Running the script creates two folders called 'metrics' and 'figures' and prints results to the console.

### Console output
- Preview of the dataset (`df.head()`) and its shape.
- Shapes of the train/test splits.
- Confirmation messages after training and after each file is saved.
- Regression metrics (MAE, MSE, RMSE, R²) for training and test sets both for raw features and scaled features.

### [`figures`](figures) folder
| File | Description |
| :--- | :--- |
| [medhouseval_distribution.png](figures/medhouseval_distribution.png) | Distribution of the target variable (`MedHouseVal`) |
| [train_actual_vs_pred.png](figures/train_actual_vs_pred.png) | Actual vs predicted values plot (raw) |
| [test_actual_vs_pred.png](figures/test_actual_vs_pred.png) | Actual vs predicted values plot (raw) |
| [test_residuals.png](figures/test_residuals.png) | Test residual plot (raw) |
| [loss_curve.png](figures/loss_curve.png) | Training loss vs iterations (raw) |
| [scaled_train_actual_vs_pred.png](figures/scaled_train_actual_vs_pred.png) | Actual vs. predicted values plot (scaled features) |
| [scaled_test_actual_vs_pred.png](figures/scaled_test_actual_vs_pred.png) | Actual vs. predicted values plot (scaled features) |
| [scaled_test_residuals.png](figures/scaled_test_residuals.png) | Test residual plot (scaled features) |
| [scaled_loss_curve.png](figures/scaled_loss_curve.png) | Training loss vs. iterations (scaled features) |

### [`metrics`](metrics) folder
| File | Description |
| :--- | :--- |
| [metrics_performance.txt](metrics/metrics_performance.txt) | A text file that have both raw and scaled metric performance on training and test set |

## Collaborators
- **Vitosavero Wibisono** / yah25002
- **Sana Lulat** / smz21003