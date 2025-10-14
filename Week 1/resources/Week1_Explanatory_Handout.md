
# Week 1 Explanatory Handout — Vision, EDA & Baseline ML (Detailed Notes)
_DataVerse Africa Internship Cohort 3.0 · Data Science · 2025-09-28_

This document expands the lecture material with **concepts, examples, code patterns, pitfalls, and references**. It is designed for self‑study and as a companion during live coding.

---

## 1. Framing the ML Problem (decision first)
- **Stakeholders**: who uses predictions? What decision changes?
- **Target**: continuous outcome → regression (e.g., yield in t/ha).
- **Unit of analysis**: field/season/region.
- **Metric bound to action**: MAE often reads in t/ha, useful for agronomic decisions.
- **Scope & assumptions**: what data is known at prediction time?

> **Rule of thumb**: If your metric doesn’t tell someone what to do differently, pick a better metric.

---

## 2. Supervised vs Unsupervised (quick map)
- **Supervised**: inputs + labels → predict `y`. Tasks: regression, classification.
- **Unsupervised**: inputs only → patterns (clustering, dimensionality reduction).
- This week’s focus is **regression** baselines.

---

## 3. End‑to‑end tabular workflow (the “leak‑safe” pattern)
1. **Split** data into train/test *before* data‑dependent transforms.  
2. **Preprocess** numerics (impute, scale) & categoricals (impute, one‑hot).  
3. **Model**: start with a transparent baseline (Linear Regression) and a sturdy non‑linear model (Random Forest).  
4. **Evaluate** with MAE/RMSE/R²; interpret **in business units**.  
5. **Save** the pipeline; log results; iterate.

### 3.1 Why Pipelines & ColumnTransformer?
- Encapsulate all preprocessing so that **fit** happens on **train only**; **transform** applies to test/production the same way.
- Compose different transforms for numeric vs categorical columns without manual leakage.

**Starter code** (edit column names as needed):
```python
from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

X = df.drop(columns=['yield_t_ha'])
y = df['yield_t_ha']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

num_cols = X.select_dtypes(include=['int64','float64']).columns.tolist()
cat_cols = X.select_dtypes(include=['object']).columns.tolist()

preprocess = ColumnTransformer([
    ('num', Pipeline([('imp', SimpleImputer(strategy='median')), ('sc', StandardScaler())]), num_cols),
    ('cat', Pipeline([('imp', SimpleImputer(strategy='most_frequent')), ('ohe', OneHotEncoder(handle_unknown='ignore'))]), cat_cols)
])

models = {
    'LinearRegression': LinearRegression(),
    'RandomForest': RandomForestRegressor(n_estimators=300, random_state=42)
}

for model_name, est in models.items():
    pipe = Pipeline([('prep', preprocess), ('model', est)])
    pipe.fit(X_train, y_train)
    preds = pipe.predict(X_test)
    mae = mean_absolute_error(y_test, preds)
    rmse = mean_squared_error(y_test, preds, squared=False)
    r2 = r2_score(y_test, preds)
    print("{}: MAE={:.3f} RMSE={:.3f} R^2={:.3f}".format(model_name, mae, rmse, r2))
```

---

## 4. Metrics you’ll use (intuition + math)
- **MAE**: average absolute error; robust and in target units.
- **RMSE**: square‑root of MSE; highlights large errors (outliers matter).
- **R²**: fraction of variance explained; okay for sanity, not for action.
- **Choosing**: If decisions hinge on cost proportional to absolute error, use **MAE**; if big mistakes hurt more, monitor **RMSE** too.

---

## 5. EDA essentials (time‑boxed)
- Inspect `df.info()`, `.describe()`, and **missingness %**.
- Plot the **target**; suspect outliers or truncation.
- Bivariate sanity plots (1–2): rainfall vs yield (scatter), region vs yield (boxplot).
- Keep a **question list**: “What might be predictive?” “What will be available at inference?”

---

## 6. Preprocessing patterns (copy/paste)

### 6.1 Missing data
- Numerics → **median** imputation; Categoricals → **most frequent**.
- Always impute **inside** the pipeline to avoid leakage.

### 6.2 Scaling & encoding
- Standardize numerics (0‑mean/1‑std) for linear models & metrics stability.
- `OneHotEncoder(handle_unknown='ignore')` to avoid runtime errors on new categories.

### 6.3 Feature engineering (domain‑aware)
- pH optimum around 6.5 → `(soil_ph-6.5)**2`.
- Heat stress above 27°C → `max(0, temperature_c-27)`.
- Interaction: `rainfall_mm * irrigated`.

> Guardrail: If a feature won’t be known at prediction time, **don’t** use it.

---

## 7. Baseline model families (pros/cons)

### 7.1 Linear Regression (OLS)
- Simple, interpretable coefficients; fast.
- Sensitive to multicollinearity/outliers; regularize if needed (Ridge/Lasso).

### 7.2 Tree ensembles (Random Forest)
- Capture non‑linearities and interactions; low tuning burden.
- Larger models; less interpretable (but stable as baselines).

---

## 8. Leakage & evaluation discipline
- **Leakage**: information from test leaking into training via imputation/scaling/encoding/feature construction computed on the full dataset.
- Fix by nesting *all* transforms in a Pipeline and fitting on **train** only.
- Keep the **split and metric fixed** while iterating; change one thing at a time.

---

## 9. Reproducibility checklist
- Set `random_state` for models and splits.
- Record: dataset version/hash, features, parameters, **scores**, and notes.
- Save the fitted **pipeline** (e.g., `joblib.dump`).

---

## 10. Agriculture data sources (for later weeks)
- **FAOSTAT** (national annual yields, production).  
- **World Bank WDI** (cereal yield indicator).  
- **NASA POWER** weather; **CHIRPS** rainfall (Africa coverage).  
- **Digital Earth Africa** registry; **Nigeria Open Data** portals.

---

## 11. Responsible & ethical AI (starter checklist)
- Fairness across **regions/varieties**: compare MAE by subgroup.
- Privacy and **data sovereignty** (align with AU Data Policy).
- Human oversight where stakes are high; fallback plan for outages.
- Document assumptions/limits in the Research Brief.

---

## 12. FAQ
- **Do I need scaling for Random Forests?** Not strictly, but scalers won’t hurt inside a ColumnTransformer; they matter for linear models and distance‑based methods.
- **Why does MAE improve but RMSE worsen?** Fewer medium errors but some larger mistakes; check outliers.
- **When to move beyond baseline?** When the baseline + metric is stable and you’ve logged a few honest iterations.

---

## 13. Glossary
- **Leakage**: using information in training that would not be available at inference.  
- **Pipeline**: a chain of transforms ending with a predictor; fit/transform separation.  
- **ColumnTransformer**: apply different transforms to different column sets.  
- **MAE/RMSE/R²**: common regression metrics.

---

## 14. Further reading (links)
- scikit‑learn: Pipelines & ColumnTransformer; Linear/Tree models; Metrics.  
- pandas & NumPy: user guides and quickstarts.  
- Data sources: FAOSTAT, WDI, NASA POWER, CHIRPS.  
- Responsible AI: AU Data Policy; Microsoft & Google practices.

> For clickable links, use the class LMS or the “Resources & Links” slide; the instructor will post the URLs in chat.

