# Week 1 — Vision & Foundation (Facilitator Pack)

**Cohort:** DataVerse Africa Internship Cohort 3.0 — Data Science  
**Track:** Data Science (Weeks 1–3 = ML Fundamentals)  
**Timezone:** 8:00PM WAT (see cohort calendar)

---

## Learning objectives (end of Week 1)
1. Explain ML workflow end‑to‑end and distinguish **supervised vs. unsupervised** learning.
2. Perform **EDA** and set up a **leak‑safe preprocessing pipeline** (scikit‑learn `ColumnTransformer` + `Pipeline`).
3. Train and evaluate a **baseline regression** model with **MAE/RMSE/R²**.
4. Present a short **Baseline Model & Data Exploration** talk.

---

## Day‑by‑day plan (from curriculum)
**Mon (Strategist):** ML in African *Agriculture / Healthcare / Finance* — use‑cases, impact, and risks.  
**Tue:** Intro to ML workflow; supervised vs. unsupervised; live coding of scikit‑learn basics.  
**Thu (Advanced):** Data preprocessing, feature engineering, and baseline training.  
**Sat:** Team presentations “Baseline ML Model & Data Exploration”.  
**Self‑study:** NumPy, pandas, EDA practice.  
**Deliverable:** Baseline regression notebook + research brief.

---

## Monday (Strategist) — 60–90 min
**Slide outline (10 slides):**
1. *Why ML for African agriculture?* Yield gaps, climate variability, input optimization.
2. *Use‑case 1: Crop disease detection on smartphone* (e.g., cassava, PlantVillage Nuru).
3. *Use‑case 2: Mechanization on‑demand* (e.g., Hello Tractor demand forecasting).
4. *Use‑case 3: Weather & advisory services* (NASA POWER + CHIRPS for rainfall/temperature).
5. *Healthcare case:* AI‑assisted neonatal diagnostics (e.g., Ubenwa) and supply‑chain (Zipline drones).
6. *Finance case:* Alternative‑data **credit scoring** for financial inclusion.
7. Risks: data leakage, distribution shift, bias across regions/varieties; privacy & consent.
8. Responsible AI checklist: fairness, transparency, accountability, security.
9. *From idea to MVP:* scope, data, metrics, baseline first.
10. Q&A: “What would a **minimally viable model** look like for our project?”

**Talking points (examples):**
- Smartphone vision for cassava disease detection can help extension officers scale diagnosis quickly.  
- Tractor demand forecasting aligns scarce equipment with peak windows after rainfall.  
- Weather reanalysis/remote‑sensing (POWER/CHIRPS) fill gaps where station data are sparse.  
- In health and finance, ML augments scarce expertise: triage, logistics, risk scoring.  
- Every deployment must include *guardrails* (testing on held‑out regions, error bands, human‑in‑the‑loop).

---

## Tuesday — Live coding (90–120 min)
**Goal:** Teach the *mechanics* of scikit‑learn on tabular data.

**Demo agenda:**
- Notebook sections: load → EDA → split → preprocess (`SimpleImputer`, `StandardScaler`, `OneHotEncoder`) → fit → evaluate.
- Two models: `LinearRegression` (transparent baseline) and `RandomForestRegressor` (stronger non‑linear baseline).
- Metrics: **MAE** (actionable), **RMSE**, **R²**.

**Practice prompts:**
- What happens to MAE if we drop `pests_index`?  
- Compare OHE vs. dropping `seed_variety` entirely.  
- Add an interaction feature (e.g., `rainfall_mm * irrigated`) in the numeric pipeline.

---

## Thursday — Advanced workshop (120 min)
**Focus:** Feature engineering + baseline improvement.
- Create domain features: rainfall squared, temp thresholds, pH distance from 6.5, region/variety group means.
- Guard against leakage: compute all transforms **inside** the pipeline only on training folds.
- Prepare a clean **experiment log** (parameters, scores, notes).

**Breakout exercise (45 min):**
- Each team designs a pipeline with 1–2 domain features and reports MAE change vs. baseline.

---

## Saturday — Presentations (90 min)
**Format per team (≤ 8 min + 2 min Q&A):**
1. Problem framing: target, unit of analysis, who uses it.
2. Data exploration: key distributions, missingness, 1–2 plots.
3. Preprocessing & model: (diagram) ColumnTransformer → model.
4. Scores and lessons: MAE/RMSE/R², error analysis.
5. Risks/next steps: fairness, additional data, Week‑2 plan.

**Rubric (10 pts):**
- Correct framing (2), solid EDA (2), leak‑safe pipeline (2), metrics interpretation (2), clarity/story (2).

---

## Instructor checklist
- [ ] Open the **Week1_Baseline_Regression_Agri_Yield.ipynb** and run cells ahead of class.
- [ ] Share the **synthetic dataset** and **Exercises.py**.
- [ ] Remind teams to copy and complete **Research_Brief_Template.md**.
- [ ] Emphasize *baseline first* and *metrics tied to action*.

---

## Appendix: Slide headlines (ready to copy)
- **Slide:** “What is our *decision*?” (forecast window, action owner, SLA)
- **Slide:** “Anatomy of an ML workflow” (collect → EDA → preprocess → baseline → evaluate → iterate)
- **Slide:** “Pipelines prevent leakage” (fit/transform on train only; transform on test)
