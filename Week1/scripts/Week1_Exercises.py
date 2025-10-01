"""
Week 1 — Exercises (Separate Script)
Complete the TODOs. Use the provided synthetic dataset by default:
    /mnt/data/week1_synthetic_agri_yield.csv
"""

from dataclasses import dataclass
from typing import List, Tuple, Dict, Any

import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.linear_model import LinearRegression

DATA_PATH = "/mnt/data/week1_synthetic_agri_yield.csv"

@dataclass
class EvalResult:
    model_name: str
    MAE: float
    RMSE: float
    R2: float

def load_data(path: str = DATA_PATH) -> pd.DataFrame:
    """Load the dataset as a DataFrame."""
    return pd.read_csv(path)

def split_xy(df: pd.DataFrame, target: str) -> Tuple[pd.DataFrame, pd.Series]:
    X = df.drop(columns=[target])
    y = df[target]
    return X, y

# === TODO 1: implement a simple EDA summary ================================
def eda_summary(df: pd.DataFrame) -> pd.DataFrame:
    """
    Return a tidy EDA table with:
      - column
      - dtype
      - missing_pct
      - nunique
    """
    # TODO: compute and return the table described above
    rows = []
    for col in df.columns:
        dtype = df[col].dtype
        missing_pct = float(df[col].isna().mean() * 100.0)
        nunique = int(df[col].nunique(dropna=True))
        rows.append({"column": col, "dtype": str(dtype), "missing_pct": round(missing_pct, 2), "nunique": nunique})
    return pd.DataFrame(rows).sort_values("missing_pct", ascending=False).reset_index(drop=True)

# === TODO 2: build a ColumnTransformer for numeric & categorical ===========
def build_preprocess(X: pd.DataFrame) -> ColumnTransformer:
    """
    Return a ColumnTransformer that:
      - median-imputes + standardizes numeric columns
      - most_frequent-imputes + one-hot encodes categorical columns
    """
    num_cols = X.select_dtypes(include=['int64','float64']).columns.tolist()
    cat_cols = X.select_dtypes(include=['object']).columns.tolist()

    numeric_pipe = Pipeline([('imputer', SimpleImputer(strategy='median')), ('scaler', StandardScaler())])
    categorical_pipe = Pipeline([('imputer', SimpleImputer(strategy='most_frequent')), ('ohe', OneHotEncoder(handle_unknown='ignore'))])
    pre = ColumnTransformer([('num', numeric_pipe, num_cols), ('cat', categorical_pipe, cat_cols)])
    return pre

# === TODO 3: train a baseline Linear Regression and report metrics =========
def train_and_eval_baseline(df: pd.DataFrame, target: str = "yield_t_ha") -> EvalResult:
    """
    Split, build pipeline (preprocess + LinearRegression), fit, and evaluate.
    Return EvalResult(MAE, RMSE, R2).
    """
    X, y = split_xy(df, target)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    preprocess = build_preprocess(X_train)
    pipe = Pipeline([('preprocess', preprocess), ('model', LinearRegression())])
    pipe.fit(X_train, y_train)
    preds = pipe.predict(X_test)
    mae = mean_absolute_error(y_test, preds)
    rmse = mean_squared_error(y_test, preds, squared=False)
    r2 = r2_score(y_test, preds)
    return EvalResult("LinearRegression", float(mae), float(rmse), float(r2))

if __name__ == "__main__":
    df = load_data()
    print("Head:"); print(df.head())
    print("\nEDA summary:"); print(eda_summary(df))
    res = train_and_eval_baseline(df)
    print(f"\nBaseline -> MAE: {res.MAE:.3f}, RMSE: {res.RMSE:.3f}, R2: {res.R2:.3f}")
