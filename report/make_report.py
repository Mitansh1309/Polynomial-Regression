"""Builds report/BT2024046_report.pdf from results/, figures/ and the data.

Usage: python report/make_report.py --repo-url https://github.com/<user>/<repo>
"""
import argparse, json, os, sys
import numpy as np, pandas as pd
from sklearn.model_selection import KFold
from sklearn.preprocessing import PolynomialFeatures
from reportlab.lib.pagesizes import A4
from reportlab.lib.units import cm
from reportlab.lib import colors
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.platypus import (SimpleDocTemplate, Paragraph, Spacer, Image, Table,
                                TableStyle, KeepTogether)

sys.path.insert(0, "src")
from models import PolyRegressor
from polybasis import LegendrePolyFeatures
from train import n_clipped

ROLL, NAME, DATA = "BT2024046", "Mitansh Shringi", "data"
ap = argparse.ArgumentParser(); ap.add_argument("--repo-url", required=True)
ap.add_argument("--out", default=f"report/{ROLL}_report.pdf")
args = ap.parse_args()

# ----------------------------------------------------------------- numbers
def load(v):
    tr = pd.read_csv(f"{DATA}/{ROLL}_train_var{v}.csv"); te = pd.read_csv(f"{DATA}/{ROLL}_test_var{v}.csv")
    return tr.drop(columns="y").values, tr["y"].values, te.values

S = {v: json.load(open(f"results/summary_var{v}.json")) for v in (1, 2)}
CV = {v: pd.read_csv(f"results/cv_var{v}.csv") for v in (1, 2)}
D = {v: load(v) for v in (1, 2)}


def cv_mse(X, y, cfg, feats=None):
    e = []
    for tr, va in KFold(5, shuffle=True, random_state=0).split(X):
        m = PolyRegressor(**cfg, features=feats).fit(X[tr], y[tr])
        e.append(np.mean((m.predict(X[va]) - y[va]) ** 2))
    return float(np.mean(e))


def cond(X, d):
    a = np.linalg.cond(PolynomialFeatures(d).fit_transform(X))
    b = np.linalg.cond(LegendrePolyFeatures(d).fit_transform(X))
    return a, b

ablation = {}
for v in (1, 2):
    X, y, _ = D[v]; p = X.shape[1]
    ablation[v] = [cv_mse(X, y, S[v]["chosen"], [i for i in range(p) if i != j]) for j in range(p)]

cond1 = cond(D[1][0], 5); cond2 = cond(D[2][0], 8); cond20 = cond(D[2][0], 20)

# stability: chosen model vs runner-ups on the test set
def rms(a, b): return float(np.sqrt(np.mean((a - b) ** 2)))
stab = {}
for v, alts in [(1, [dict(degree=5, method="relaxed", alpha=0.02), dict(degree=6, method="lasso", alpha=0.005)]),
                (2, [dict(degree=9, method="lasso", alpha=0.001), dict(degree=8, method="ols", alpha=0.0)])]:
    X, y, T = D[v]
    p0 = PolyRegressor(**S[v]["chosen"]).fit(X, y).predict(T)
    stab[v] = [(a, rms(p0, PolyRegressor(**a).fit(X, y).predict(T))) for a in alts]

# ----------------------------------------------------------------- styles
ss = getSampleStyleSheet()
INK = colors.HexColor("#0b0b0b"); INK2 = colors.HexColor("#52514e"); RULE = colors.HexColor("#d9d8d4")
ACC = colors.HexColor("#2a78d6")
body = ParagraphStyle("b", parent=ss["Normal"], fontName="Helvetica", fontSize=9.2, leading=12.4,
                      textColor=INK, spaceAfter=4, alignment=4)
h1 = ParagraphStyle("h1", parent=body, fontName="Helvetica-Bold", fontSize=11.5, leading=14,
                    spaceBefore=8, spaceAfter=4, alignment=0)
h2 = ParagraphStyle("h2", parent=h1, fontSize=10, spaceBefore=5, spaceAfter=2)
cap = ParagraphStyle("cap", parent=body, fontSize=8, leading=10, textColor=INK2, alignment=0, spaceAfter=6)
small = ParagraphStyle("s", parent=body, fontSize=8.2, leading=10.5, alignment=0)
title = ParagraphStyle("t", parent=body, fontName="Helvetica-Bold", fontSize=16, leading=20, alignment=0)
bullet = ParagraphStyle("bl", parent=body, leftIndent=12, bulletIndent=2, spaceAfter=2)

P = lambda t, s=body: Paragraph(t, s)
B = lambda t: Paragraph(t, bullet, bulletText="•")


def table(rows, widths, hl_row=None):
    t = Table([[Paragraph(str(c), small) for c in r] for r in rows], colWidths=widths, hAlign="LEFT")
    st = [("LINEABOVE", (0, 0), (-1, 0), 0.8, INK), ("LINEBELOW", (0, 0), (-1, 0), 0.5, INK2),
          ("LINEBELOW", (0, -1), (-1, -1), 0.8, INK), ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
          ("TOPPADDING", (0, 0), (-1, -1), 1.5), ("BOTTOMPADDING", (0, 0), (-1, -1), 1.5)]
    if hl_row is not None:
        st.append(("BACKGROUND", (0, hl_row), (-1, hl_row), colors.HexColor("#e8f0fb")))
    t.setStyle(TableStyle(st)); return t


def fig(path, w_cm, caption):
    from reportlab.lib.utils import ImageReader
    iw, ih = ImageReader(path).getSize(); w = w_cm * cm
    return KeepTogether([Image(path, width=w, height=w * ih / iw), P(caption, cap)])


def best_per_degree(v, degs):
    c = CV[v]; rows = [["Deg.", "# terms", "OLS CV MSE", "Best estimator", "Best CV MSE", "Test-weighted"]]
    hl = None
    for i, d in enumerate(degs, 1):
        q = c[c.degree == d]; b = q.loc[q.cv_mse.idxmin()]; o = q[q.method == "ols"].cv_mse
        nt = PolyRegressor(d).fit(D[v][0][:3], D[v][1][:3]).basis_.n_terms
        name = {"ols": "OLS", "ridge": "Ridge", "lasso": "Lasso", "relaxed": "Relaxed Lasso"}[b.method]
        a = "" if b.method == "ols" else f" (α={b.alpha:g})"
        rows.append([d, nt, f"{o.iloc[0]:.3f}" if len(o) else "n/a (p ≥ n)", name + a,
                     f"{b.cv_mse:.3f}", f"{b.cv_wmse:.3f}"])
        if d == S[v]["chosen"]["degree"]: hl = i
    return rows, hl

# ----------------------------------------------------------------- content
s1, s2 = S[1], S[2]
st = []
st += [P("Polynomial Regression — Assignment 1 Report", title), Spacer(1, 3),
       P(f"<b>{NAME}</b> · Roll No. <b>{ROLL}</b> · Code: <link href='{args.repo_url}' color='#2a78d6'>"
         f"<u>{args.repo_url}</u></link>", body), Spacer(1, 4)]

st += [table([
    ["Problem", "Model submitted", "Terms used", "CV MSE (± s.e.)", "CV R<super>2</super>"],
    ["var1 — Net Power Score (x1…x6)", f"Total degree <b>{s1['chosen']['degree']}</b>, Lasso α={s1['chosen']['alpha']:g}",
     f"{s1['n_terms']} of 462", f"{s1['cv_mse']:.3f} ± {s1['cv_mse_se']:.3f}", f"{s1['cv_r2']:.4f}"],
    ["var2 — Thermal Anomaly Score (x1…x3)", f"Total degree <b>{s2['chosen']['degree']}</b>, Lasso α={s2['chosen']['alpha']:g}",
     f"{s2['n_terms']} of 165", f"{s2['cv_mse']:.3f} ± {s2['cv_mse_se']:.3f}", f"{s2['cv_r2']:.4f}"],
], [5.0*cm, 4.6*cm, 2.0*cm, 2.8*cm, 1.6*cm]),
    P("CV = 5-fold cross-validation repeated 3 times on the 1000 training rows (15 fits per configuration). "
      "R<super>2</super> is 1 − CV MSE / Var(y).", cap)]

# 1. data
X1, y1, T1 = D[1]; X2, y2, T2 = D[2]
fr = lambda X: 100 * np.mean(np.abs(X) == 1)
st += [P("1. Data and what it implies for modelling", h1),
       P(f"Each problem has 1000 training and 1000 test rows, with no missing values or duplicates. All inputs lie in "
         f"[−1, 1]. Var(y) is {y1.var():.2f} for var1 and {y2.var():.2f} for var2. Two observations shaped the method:"),
       B(f"<b>Inputs are clipped to ±1.</b> {fr(X1):.0f}% of var1 and {fr(X2):.0f}% of var2 training values are exactly "
         f"±1. Out-of-fold residuals of a good var1 model are about the same whether 0, 1, 2 or 3 coordinates are clipped "
         f"(MSE ≈ 0.34–0.37). This indicates y was generated from the clipped values, so no special handling is needed."),
       B(f"<b>The var1 test set is shifted toward the corners of the cube.</b> {fr(T1):.0f}% of var1 test values are "
         f"±1 (vs {fr(X1):.0f}% in training). {100*np.mean(n_clipped(T1)>=3):.0f}% of test rows have ≥3 clipped "
         f"coordinates, against {100*np.mean(n_clipped(X1)>=3):.0f}% of training rows "
         f"(Fig. 1). The edges of [−1, 1]<super>6</super> are where high-degree polynomials oscillate most, so validation "
         f"error is also reported <i>re-weighted to the test distribution</i>. var2 shows almost no such shift."),
       fig("figures/fig_clipping.png", 15.5,
           "Fig. 1 — Share of rows by number of coordinates exactly at ±1. var1's test set is concentrated near the "
           "corners; var2's matches its training set.")]

# 2. method
st += [P("2. Method", h1),
       P("2.1 Polynomial features in a well-conditioned basis", h2),
       P("A polynomial of total degree d contains every term x1<super>a</super>x2<super>b</super>… with a+b+… ≤ d. "
         "For k inputs that is C(k+d, d) terms: 462 for var1 at d=5, 165 for var2 at d=8, and 1771 for var2 at d=20. "
         "Raw monomials become nearly collinear at high degree (x<super>18</super> and x<super>20</super> look almost "
         "the same on [−1, 1]). So I build the same space from products of Legendre polynomials "
         "P<sub>a</sub>(x1)·P<sub>b</sub>(x2)·… with a+b+… ≤ d. Each P<sub>a</sub> is a degree-a polynomial and the set is "
         "a basis of the same space. The fitted function is therefore still an ordinary polynomial of degree d, and "
         "<font face='Courier'>export_equation.py</font> converts it back to monomial coefficients. Only the numerics "
         f"change: on the training data the design-matrix condition number drops from {cond1[0]:.0f} to {cond1[1]:.0f} "
         f"(var1, d=5), from {cond2[0]:.1e} to {cond2[1]:.0f} (var2, d=8) and from {cond20[0]:.1e} to {cond20[1]:.1e} "
         "(var2, d=20). This makes the high-degree end of the search reliable."),
       P("2.2 Estimators", h2),
       P("For each degree, four ways of fitting the coefficients are compared. All of them produce a polynomial of that "
         "degree. <b>OLS</b> is plain least squares. <b>Ridge</b> adds an L2 penalty α‖β‖² (intercept not penalised). "
         "<b>Lasso</b> adds an L1 penalty α‖β‖<sub>1</sub>, which sets many coefficients exactly to zero and so selects "
         "which terms of the degree-d polynomial to keep. <b>Relaxed Lasso</b> uses the Lasso only to choose terms, then "
         "refits them by OLS to remove the shrinkage bias. Penalties are what make degrees with as many terms as rows "
         "fittable at all, so they decide how far up the degree ladder the search can go."),
       P("2.3 Model selection", h2),
       B("<b>Search space:</b> var1 degrees 1–8 (degree 8 = 3003 terms, already 3× the sample size); var2 degrees 1–20, "
         "the full range the problem statement allows. Ridge uses 6–7 values of α at every degree. Lasso and relaxed "
         "Lasso use 4–6 values of α up to degree 7 (var1) or 12 (var2); above that the curves were already rising and "
         "the cost was high. OLS is skipped once the number of terms exceeds the fold's training rows."),
       B("<b>Score:</b> mean MSE over 5-fold × 3-repeat CV, with the same fixed splits for every configuration. "
         "There is also a test-weighted variant: each validation error is multiplied by "
         "w(k) = P<sub>test</sub>(k)/P<sub>train</sub>(k), where k is the number of clipped coordinates (capped at 10)."),
       B("<b>Rule:</b> take the configuration with the lowest test-weighted CV MSE. Among all configurations within one "
         "standard error of it, choose the <i>lowest degree</i>, then the best score at that degree. The final model is "
         "refit on all 1000 training rows.")]

# 3. var1
r1, hl1 = best_per_degree(1, range(1, 9))
st += [P("3. Problem var1 — six turbine settings", h1),
       fig("figures/fig_degree.png", 16.5,
           "Fig. 2 — Training and CV error against degree (log scale). Training error keeps falling; CV error has a "
           "clear minimum. Plain OLS overfits one or two degrees earlier than penalised fits. var2's Lasso curve stops at "
           "d=12, where the search for it ended."),
       table(r1, [1.2*cm, 1.5*cm, 2.4*cm, 4.5*cm, 2.2*cm, 2.4*cm], hl1),
       P("Table 1 — var1: best configuration at each degree (CV MSE). Highlighted row = submitted model.", cap),
       P(f"Plain OLS is best at degree 4 (CV MSE {CV[1].query('degree==4 and method==\"ols\"').cv_mse.iloc[0]:.3f}). "
         f"At degree 5 OLS has 462 coefficients for 800 training rows per fold and its error rises to "
         f"{CV[1].query('degree==5 and method==\"ols\"').cv_mse.iloc[0]:.2f}. Fig. 3 shows that the L1 penalty recovers "
         f"degree 5 and more than halves the error, while the L2 penalty helps much less. Ridge keeps all 462 terms and "
         f"only shrinks them. The Lasso keeps about a third of the terms, which suggests the true degree-5 polynomial is "
         f"sparse. Degrees 6 and 7 with Lasso are no better ({CV[1].query('degree==6').cv_mse.min():.3f}, "
         f"{CV[1].query('degree==7').cv_mse.min():.3f}): the extra terms are not needed. Degree 3 or 4 is clearly too "
         f"simple ({CV[1].query('degree==3').cv_mse.min():.2f}, {CV[1].query('degree==4').cv_mse.min():.2f}). "
         f"So the submitted model is <b>degree 5, Lasso α={s1['chosen']['alpha']:g}, with {s1['n_terms']} non-zero "
         f"terms</b>."),
       fig("figures/fig_alpha_var1.png", 7.6,
           "Fig. 3 — var1 at degree 5: CV MSE against penalty strength for each estimator."),
       P("<b>Are all six inputs needed?</b> Removing one input at a time and refitting the chosen configuration gives "
         "CV MSE of " + ", ".join(f"x{j+1}: {e:.2f}" for j, e in enumerate(ablation[1])) +
         f" (vs {s1['cv_mse']:.2f} with all six). Every input is needed: dropping any one multiplies the error by "
         f"{min(ablation[1])/s1['cv_mse']:.0f}–{max(ablation[1])/s1['cv_mse']:.0f}×. The model therefore uses all six "
         "settings."),
       P(f"<b>Noise floor.</b> Training MSE is {s1['train_mse']:.2f} and CV MSE is {s1['cv_mse']:.2f}. Out-of-fold "
         "error is flat across clipping groups and across degrees 5–7, so about 0.35 appears to be irreducible noise "
         "rather than missing structure.")]

# 4. var2
r2, hl2 = best_per_degree(2, [4, 5, 6, 7, 8, 9, 10, 12, 14, 16, 20])
_bd = CV[2].groupby("degree").cv_mse.min()
_d89, _d7 = abs(_bd[8] - _bd[9]), _bd[7] - _bd[8]
st += [P("4. Problem var2 — 3-D thermal anomaly field", h1),
       P("var2 has only three inputs, so a high degree is affordable (165 terms at d=8, 1771 at d=20). The right panel "
         "of Fig. 2 shows a textbook bias–variance curve. CV error drops by two orders of magnitude up to d≈8, and OLS "
         "then explodes (d=13: MSE 26; d≥15: no unique solution). Penalties keep the high-degree fits under control but "
         "never beat degree 8. At d=20 the best Ridge fit has 4–5× the error of d=8, so the full degree budget is "
         "clearly not needed."),
       table(r2, [1.2*cm, 1.5*cm, 2.4*cm, 4.5*cm, 2.2*cm, 2.4*cm], hl2),
       P("Table 2 — var2: best configuration at selected degrees (CV MSE). Highlighted row = submitted model.", cap),
       P(f"Degrees 8 and 9 tie (difference {_d89:.4f}, under 1/{int(s2['cv_mse_se']/max(_d89,1e-9))} of one standard "
         f"error), and degree 7 is {_d7:.2f} worse. "
         f"Under the one-standard-error rule <b>degree 8</b> is chosen. Unlike var1, the var2 polynomial is "
         f"<i>dense</i>: the Lasso keeps {s2['n_terms']} of 165 terms, and OLS, Ridge and Lasso at d=8 are within "
         f"0.005 of each other. A light Lasso (α={s2['chosen']['alpha']:g}) has a marginally lower test-weighted error, "
         f"so it is used. All three inputs matter: removing one gives CV MSE " +
         ", ".join(f"x{j+1}: {e:.1f}" for j, e in enumerate(ablation[2])) + f" (vs {s2['cv_mse']:.2f}).")]

# 5. diagnostics
st += [P("5. Final checks", h1),
       fig("figures/fig_oof.png", 15.0,
           "Fig. 4 — Out-of-fold predictions of the submitted models against the true y (training set)."),
       B("<b>Residuals</b> scatter evenly around the diagonal across the whole range of y, including the extremes "
         "(Fig. 4). There is no curvature, which would have been a sign of underfitting."),
       B("<b>Robustness on the test inputs.</b> RMS difference between the submitted predictions and those of the "
         "runner-up models: " + "; ".join(
             f"var{v}: " + ", ".join(f"d={a['degree']} {a['method']}: {r:.2f}" for a, r in stab[v]) for v in (1, 2)) +
         ". All are below the noise standard deviation (≈0.5–0.6), so the predictions do not depend on the exact pick. "
         "For var1 I also checked that predictions in each clipping group (k = 0…6) stay within the training y range "
         "of that group. They do, so the model does not blow up at the corners even though the test set is "
         "corner-heavy."),
       B("<b>Generalisation estimate.</b> Expected test MSE ≈ "
         f"{s1['cv_wmse']:.2f} (var1, test-weighted) and {s2['cv_wmse']:.2f} (var2). Expected test R<super>2</super> ≈ "
         f"{1 - s1['cv_wmse']/y1.var():.3f} and {1 - s2['cv_wmse']/y2.var():.3f}."),
       P("6. Reproducibility", h1),
       P(f"Repository: <link href='{args.repo_url}' color='#2a78d6'><u>{args.repo_url}</u></link>. "
         "<font face='Courier'>bash run_all.sh</font> runs the full grid search (≈5 min), refits the final models, "
         f"writes <font face='Courier'>predictions/{ROLL}_pred_var1.csv</font> and "
         f"<font face='Courier'>{ROLL}_pred_var2.csv</font>, and regenerates every figure. All CV results are saved in "
         "<font face='Courier'>results/cv_var*.csv</font>. All randomness uses fixed seeds. Dependencies: "
         "numpy, pandas, scikit-learn (Lasso solver only), matplotlib.")]


def footer(c, d):
    c.saveState(); c.setFont("Helvetica", 7.5); c.setFillColor(INK2)
    c.drawString(1.8*cm, 1.1*cm, f"{NAME} · {ROLL} · Polynomial Regression")
    c.drawRightString(A4[0] - 1.8*cm, 1.1*cm, f"{d.page}"); c.restoreState()


os.makedirs(os.path.dirname(args.out), exist_ok=True)
doc = SimpleDocTemplate(args.out, pagesize=A4, leftMargin=1.8*cm, rightMargin=1.8*cm, topMargin=1.5*cm,
                        bottomMargin=1.7*cm, title=f"{ROLL} — Polynomial Regression Report", author=NAME)
doc.build(st, onFirstPage=footer, onLaterPages=footer)
print("wrote", args.out)
