
# Week 1 — Vision & Foundation (Lecture Script / Facilitator Notes)
_DataVerse Africa Internship Cohort 3.0 · Data Science · 2025-09-28_

> **Purpose**: A complete teaching script for Week‑1, with time breakdowns, talk tracks, live‑coding prompts, checks for understanding, and activities.
> **Deliverables this week**: Baseline regression notebook + research brief + short team presentation.

---

## At‑a‑glance schedule (8:00 PM WAT)
- **Mon (Strategist, 60–90 min):** Vision, impact use‑cases in African agriculture/health/finance; Responsible AI guardrails.
- **Tue (120 min):** ML workflow + live coding (EDA → split → preprocess → baseline models → metrics).
- **Thu (120 min):** Feature engineering + leakage prevention workshop; baseline iteration.
- **Sat (90 min):** Team presentations “Baseline ML Model & Data Exploration”.

---

## Teaching setup & assets
- **Slides**: _Week1_Vision_and_Foundation_Slides.pptx_ (provided).
- **Notebook**: _Week1_Baseline_Regression_Agri_Yield.ipynb_.
- **Dataset**: _week1_synthetic_agri_yield.csv_ (no internet required).
- **Exercises**: _Week1_Exercises.py_ (three TODOs).
- **Handout**: _Week1_Explanatory_Handout.md_ (share with cohort).

> **Pro tip**: Print the “rubric” slide and bring a timer for Saturday.

---

## MONDAY (Strategist) · 60–90 min

### Learning goals
1. See concrete, **Africa‑centric** ML use‑cases (agriculture first).
2. Identify **stakeholders**, **decisions**, and **metrics**.
3. Introduce **Responsible AI** checklists you’ll revisit weekly.

### 0–10 min · Cold‑open
**Hook**: “If we could forecast field‑level yield ±0.3 t/ha, what decisions would change next season?”  
**Poll**: Who’s worked with ag, health, or finance data? What was hard?

### 10–30 min · Use‑cases (high‑level)
- **Smartphone plant disease detection** (cassava, PlantVillage Nuru) → fast triage in the field.
- **Mechanization on demand** (Hello Tractor) → match tractor supply to rainfall windows.
- **Weather‑driven advisory** (NASA POWER / CHIRPS) → fill in station data gaps.
- **Healthcare logistics / neonatal triage** → breadth beyond ag.

**Talking prompts**
- What *data* powers each? (images, tabular, weather, transactions)
- What’s the **decision**? Who acts and when? What’s the **SLA**?

### 30–45 min · Responsible AI guardrails
- Fairness across **regions/varieties**; monitor subgroup error.
- Privacy, consent, and **data sovereignty** (AU Data Policy).
- Human‑in‑the‑loop for critical decisions; fallback plans.
- Distribution shift: season, region, market changes.

**Mini‑activity (Think–Pair–Share, 8 min)**  
Pick one use‑case. List *two* risks and *one* mitigation you can test by Week‑3.

### 45–75 min · From idea → minimally viable model (MVM)
- Frame the **target**, **unit of analysis**, **features**, **metric** (MAE is actionable).
- Collect → Explore → **Split** → Preprocess → Baseline → Evaluate → Iterate.
- Homework: skim the Handout sections 1–3; preview Tuesday notebook.

**Exit ticket (3 prompts)**  
1) Our actionable metric is … 2) The first baseline we’ll try is … 3) A risk we’ll monitor is …

---

## TUESDAY (Intro + Live Coding) · 120 min

### Learning goals
- Distinguish **supervised** vs **unsupervised** learning; focus on **regression**.  
- Run an **end‑to‑end tabular workflow** with scikit‑learn.  
- Avoid **leakage** using `ColumnTransformer` + `Pipeline`.

### 0–15 min · Concepts
- Supervised (targets known): regression/classification; Unsupervised: clustering, DR.  
- Metrics: **MAE**, **RMSE**, **R²** — when to use which.  
- Baseline first: a transparent model (Linear Regression) and a strong non‑linear baseline (Random Forest).

### 15–90 min · Live‑coding path (use the provided notebook)
1) **Load** the dataset → `df.info()`, `describe()`, missingness %.  
2) **EDA**: target histogram; 1–2 bivariate plots (rainfall vs yield; region boxplot).  
3) **Split before transforms** → `train_test_split`.  
4) **Preprocessing**  
   - Numeric: `SimpleImputer(strategy='median')` → `StandardScaler()`  
   - Categorical: `SimpleImputer(strategy='most_frequent')` → `OneHotEncoder(handle_unknown='ignore')`  
   - Combine with **`ColumnTransformer`**; wrap with **`Pipeline`**.  
5) **Modeling**: `LinearRegression()` and `RandomForestRegressor(n_estimators=300, random_state=42)`  
6) **Evaluate**: print **MAE/RMSE/R²**; discuss what’s “good enough” for the decision.  
7) **Save baseline**: persist best pipeline with `joblib.dump`.

**Checks for understanding (ask aloud)**
- Why do we **split before** imputing/scaling/encoding?  
- Why does MAE read in the same **units** as the target?  
- What does `handle_unknown='ignore'` prevent at inference time?

### 90–120 min · Stretch goals
- Add a domain feature: `(soil_ph-6.5)**2` or `max(0, temp-27)` inside the numeric pipeline.  
- Compare MAE again; record results in a simple experiment log.

---

## THURSDAY (Advanced Workshop) · 120 min

### Focus
Feature engineering + robust baselines; guard against leakage in *every* transform.

### 0–20 min · Mini‑lecture
- Feature types: polynomials, thresholds, interactions, group means (with care).  
- Inference stability: prefer features available at **prediction time**.  
- Experiment discipline: fixed split/metric, one change at a time.

### 20–75 min · Guided exercise (breakouts)
Each team:  
1) Add 1–2 domain features inside the pipeline.  
2) Refit models; report **ΔMAE** vs baseline.  
3) Note where errors are largest (region/variety subgroup).

### 75–110 min · Leakage detective
- Demo a **bad** pipeline (fit imputer on full data) → inflated scores.  
- Fix: move every data‑dependent op into the Pipeline; re‑evaluate.

### 110–120 min · Share‑outs
Each team shares a 1‑slide “experiment summary” (what changed, ΔMAE, next step).

---

## SATURDAY (Presentations) · 90 min

### Format per team (≤8 min + 2 Q&A)
1) Problem framing & stakeholders.  
2) Data exploration (1–2 plots).  
3) Preprocessing & model diagram (ColumnTransformer → Model).  
4) Scores + lessons (MAE/RMSE/R²).  
5) Risks & next steps (fairness, more data, Week‑2 plan).

### Rubric (10 pts)
- Framing (2) · EDA (2) · Leak‑safe pipeline (2) · Metrics interpretation (2) · Clarity/story (2)

---

## Appendix A — Live‑coding snippets (pasteable)

### Baseline preprocessing + models

```python
from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

target = 'yield_t_ha'
X = df.drop(columns=[target])
y = df[target]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

num_cols = X.select_dtypes(include=['int64','float64']).columns.tolist()
cat_cols = X.select_dtypes(include=['object']).columns.tolist()

preprocess = ColumnTransformer([
    ('num', Pipeline([('imp', SimpleImputer(strategy='median')), ('sc', StandardScaler())]), num_cols),
    ('cat', Pipeline([('imp', SimpleImputer(strategy='most_frequent')), ('ohe', OneHotEncoder(handle_unknown='ignore'))]), cat_cols),
])

linreg = Pipeline([('prep', preprocess), ('model', LinearRegression())])
rf = Pipeline([('prep', preprocess), ('model', RandomForestRegressor(n_estimators=300, random_state=42))])
```

### Evaluation helper

```python
def evaluate(model, X_train, y_train, X_test, y_test, name):
    model.fit(X_train, y_train)
    preds = model.predict(X_test)
    mae = mean_absolute_error(y_test, preds)
    rmse = mean_squared_error(y_test, preds, squared=False)
    r2 = r2_score(y_test, preds)
    print("{}: MAE={:.3f} RMSE={:.3f} R^2={:.3f}".format(name, mae, rmse, r2))
```

### Feature engineering inside the pipeline

```python
from sklearn.preprocessing import FunctionTransformer
import numpy as np

def add_domain_features(X):
    X = X.copy()
    if 'soil_ph' in X.columns:
        X['ph_dist_sq'] = (X['soil_ph'] - 6.5)**2
    if 'temperature_c' in X.columns:
        X['temp_over27'] = np.maximum(0, X['temperature_c'] - 27.0)
    if set(['rainfall_mm','irrigated']).issubset(X.columns):
        X['rain_x_irrig'] = X['rainfall_mm'] * X['irrigated']
    return X

domain = FunctionTransformer(add_domain_features, feature_names_out='one-to-one')
preprocess_plus = ColumnTransformer([
    ('num', Pipeline([('imp', SimpleImputer(strategy='median')), ('sc', StandardScaler())]), num_cols),
    ('cat', Pipeline([('imp', SimpleImputer(strategy='most_frequent')), ('ohe', OneHotEncoder(handle_unknown='ignore'))]), cat_cols)
])
rf_plus = Pipeline([('domain', domain), ('prep', preprocess_plus), ('model', RandomForestRegressor(random_state=42))])
```

---

## Appendix B — Whiteboard sketches (ASCII)

**ML workflow**
```
Data → EDA → [Split] → Preprocess (num|cat) → Baseline model → Metrics → Iterate
```

**Leak‑safe pipeline**
```
X_train ---- fit(imputer, scaler, ohe) ----> model.fit
X_test  ---- transform only ----------------> model.predict
```

**Metric intuition**
```
MAE ~ average absolute error in t/ha  |  RMSE penalizes large errors  |  R² variance explained
```

---

## Appendix C — Classroom prompts

- “Which feature do you expect to matter most and why?”  
- “If MAE improved but RMSE worsened, what changed in the error distribution?”  
- “What would break if we imputed *before* train/test split?”

