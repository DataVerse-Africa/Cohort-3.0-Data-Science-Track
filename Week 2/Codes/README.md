# Week 2 — From-Scratch ML (NumPy Only)

This pack contains compact, **teachable** implementations of classic ML models and metrics written from scratch:

- `linear_regression.py` — Normal Equation + Gradient Descent
- `logistic_regression.py` — Binary logistic regression (GD)
- `decision_tree.py` — CART-style classifier & regressor (Gini / variance)
- `random_forest.py` — Bagging over our trees (classifier & regressor)
- `svm.py` — Linear SVM classifier via SGD on hinge loss (no kernels)
- `metrics.py` — MAE, MSE, RMSE, R², Accuracy, Precision, Recall, F1
- `utils.py` — train/test split, standardize, simple synthetic data generators

## How to use

1. Open **Week2_From_Scratch_Demo.ipynb** and run cells.
2. Or import the modules in your own notebooks:

```python
from src.linear_regression import LinearRegressionGD
from src.metrics import rmse, r2
```