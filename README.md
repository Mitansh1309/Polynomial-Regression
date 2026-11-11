# ML Assignment 1 — Polynomial Regression

**Mitansh Shringi · BT2024046**

Polynomial regression models for two personalised datasets:

| Problem | Inputs | Chosen model | Terms kept | CV MSE | CV R² |
|---|---|---|---|---|---|
| var1 — steam-turbine Net Power Score | x1…x6 | degree **5**, Lasso (α = 0.005) | 158 / 462 | 0.358 | 0.966 |
| var2 — thermal anomaly score | x1, x2, x3 | degree **8**, Lasso (α = 0.001) | 155 / 165 | 0.271 | 0.994 |

CV = 5-fold × 3 repeats on the training set. Every model is an ordinary polynomial
in the raw inputs; see `report/` for the full write-up.

## Layout

```
data/                  BT2024046_{train,test}_var{1,2}.csv
src/polybasis.py       total-degree polynomial basis (Legendre coordinates) + monomial conversion
src/models.py          PolyRegressor: OLS / Ridge / Lasso / Relaxed-Lasso on that basis
src/train.py           grid search over degree × estimator × penalty, model selection, final fit
src/predict.py         inference -> predictions/BT2024046_pred_var{1,2}.csv
src/make_figures.py    figures used in the report
src/export_equation.py prints the fitted polynomial in monomial form
results/               CV tables (cv_var*.csv) and summaries (summary_var*.json)
models/                fitted coefficients (model_var*.npz)
predictions/           submission files
```

## Reproduce

```bash
pip install -r requirements.txt
bash run_all.sh            # full grid search (~5 min) + predictions + figures
# or, reusing the saved CV tables:
python src/train.py --var 1 --reuse-cv && python src/predict.py --var 1
python src/train.py --var 2 --reuse-cv && python src/predict.py --var 2
```

## Method in one paragraph

Features are polynomial terms of total degree ≤ d, built from products of Legendre
polynomials P_a(x1)·P_b(x2)·… with a+b+… ≤ d. Because the inputs lie in [-1, 1],
this spans exactly the same space as the monomials x1^a·x2^b·… but is far better
conditioned at high degree. For each degree we fit OLS, Ridge, Lasso and a relaxed
Lasso, score them with repeated K-fold CV, re-weight validation errors to match the
test set's share of boundary (±1) points, and pick the lowest degree whose error is
within one standard error of the best.
