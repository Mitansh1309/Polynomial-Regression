"""Figures used in the report (reads results/cv_var*.csv, writes figures/*.png)."""
import json, os
import numpy as np, pandas as pd
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.model_selection import KFold

from models import PolyRegressor
from train import n_clipped

ROLL, DATA = "BT2024046", "data"
C1, C2, C3 = "#2a78d6", "#eb6834", "#1baf7a"   # categorical slots 1-3, fixed order
INK, INK2, GRID = "#0b0b0b", "#52514e", "#e4e3df"
plt.rcParams.update({
    "font.size": 9, "axes.edgecolor": INK2, "axes.labelcolor": INK, "xtick.color": INK2,
    "ytick.color": INK2, "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.6,
    "axes.spines.top": False, "axes.spines.right": False, "legend.frameon": False,
    "lines.linewidth": 2, "lines.markersize": 5, "savefig.dpi": 200, "savefig.bbox": "tight",
})
os.makedirs("figures", exist_ok=True)


def load(v):
    tr = pd.read_csv(f"{DATA}/{ROLL}_train_var{v}.csv")
    te = pd.read_csv(f"{DATA}/{ROLL}_test_var{v}.csv")
    return tr.drop(columns="y").values, tr["y"].values, te.values


def train_mse_ols(X, y, degrees):
    out = []
    for d in degrees:
        m = PolyRegressor(d, "ols").fit(X, y)
        out.append(np.mean((m.predict(X) - y) ** 2) if m.basis_.n_terms < len(X) else np.nan)
    return np.array(out)


# --- Fig 1: CV error vs degree ------------------------------------------------
C4 = "#eda100"  # categorical slot 4
fig, axes = plt.subplots(1, 2, figsize=(7.2, 3.0))
for ax, v in zip(axes, [1, 2]):
    cv = pd.read_csv(f"results/cv_var{v}.csv"); s = json.load(open(f"results/summary_var{v}.json"))
    X, y, _ = load(v)
    degs = sorted(cv.degree.unique())
    per = lambda meths: cv[cv.method.isin(meths)].groupby("degree").cv_mse.min().reindex(degs)
    trn = train_mse_ols(X, y, degs)
    ax.plot(degs, trn, "-^", color=C3, label="Train MSE (OLS)")
    ax.plot(degs, per(["ols"]), "-o", color=C1, label="CV MSE: OLS")
    ax.plot(degs, per(["ridge"]), "-s", color=C2, label="CV MSE: best Ridge")
    ax.plot(degs, per(["lasso", "relaxed"]), "-D", color=C4, markersize=4,
            label="CV MSE: best Lasso / relaxed")
    d = s["chosen"]["degree"]
    ax.axvline(d, color=INK2, lw=1, ls="--")
    ax.text(d + 0.15, 0.97, f"chosen d={d}", transform=ax.get_xaxis_transform(),
            va="top", ha="left", color=INK, fontsize=8)
    ax.set_yscale("log"); ax.set_xlabel("Polynomial degree"); ax.set_ylabel("MSE (log)")
    ax.set_title(f"var{v}", color=INK, fontsize=10, loc="left")
    ax.set_xticks(degs if len(degs) < 10 else degs[::2])
    if v == 2:
        ax.set_ylim(0.05, 60)
h, l = axes[0].get_legend_handles_labels()
fig.legend(h, l, loc="lower center", ncol=4, fontsize=7.5, bbox_to_anchor=(0.5, -0.02))
fig.tight_layout(rect=(0, 0.08, 1, 1)); fig.savefig("figures/fig_degree.png"); plt.close(fig)

# --- Fig 2: boundary (clipping) distribution, train vs test --------------------
fig, axes = plt.subplots(1, 2, figsize=(7.2, 2.4))
for ax, v in zip(axes, [1, 2]):
    X, y, T = load(v)
    k = np.arange(X.shape[1] + 1)
    a = np.bincount(n_clipped(X), minlength=len(k)) / len(X)
    b = np.bincount(n_clipped(T), minlength=len(k)) / len(T)
    ax.bar(k - 0.2, a, 0.38, color=C1, label="train")
    ax.bar(k + 0.2, b, 0.38, color=C2, label="test")
    ax.set_xticks(k); ax.set_xlabel("# coordinates exactly at ±1"); ax.set_ylabel("fraction of rows")
    ax.set_title(f"var{v}", color=INK, fontsize=10, loc="left")
axes[0].legend(fontsize=8)
fig.tight_layout(); fig.savefig("figures/fig_clipping.png"); plt.close(fig)

# --- Fig 3: var1 regularisation at the chosen degree --------------------------
cv = pd.read_csv("results/cv_var1.csv"); d = json.load(open("results/summary_var1.json"))["chosen"]["degree"]
fig, ax = plt.subplots(figsize=(3.6, 2.6))
sub = cv[cv.degree == d]
for meth, col, mk, lab in [("ridge", C1, "o", "Ridge"), ("lasso", C2, "s", "Lasso"),
                           ("relaxed", C3, "^", "Relaxed Lasso")]:
    q = sub[sub.method == meth].sort_values("alpha")
    ax.plot(q.alpha, q.cv_mse, "-" + mk, color=col, label=lab)
o = sub[sub.method == "ols"].cv_mse
if len(o):
    ax.axhline(float(o.iloc[0]), color=INK2, ls=":", lw=1.2, label="OLS")
ax.set_xscale("log"); ax.set_yscale("log"); ax.set_xlabel("penalty α"); ax.set_ylabel("CV MSE (log)")
ax.set_title(f"var1, degree {d}", color=INK, fontsize=10, loc="left"); ax.legend(fontsize=7.5)
fig.tight_layout(); fig.savefig("figures/fig_alpha_var1.png"); plt.close(fig)

# --- Fig 4: out-of-fold predictions for the chosen models ---------------------
fig, axes = plt.subplots(1, 2, figsize=(7.2, 3.0))
for ax, v in zip(axes, [1, 2]):
    X, y, _ = load(v); s = json.load(open(f"results/summary_var{v}.json"))
    oof = np.empty_like(y)
    for tr, va in KFold(5, shuffle=True, random_state=0).split(X):
        oof[va] = PolyRegressor(**s["chosen"]).fit(X[tr], y[tr]).predict(X[va])
    mse = np.mean((oof - y) ** 2); r2 = 1 - mse / y.var()
    lo, hi = min(y.min(), oof.min()), max(y.max(), oof.max())
    ax.plot([lo, hi], [lo, hi], color=INK2, lw=1, ls="--")
    ax.scatter(y, oof, s=8, color=C1, alpha=0.55, edgecolors="none")
    ax.set_xlabel("true y"); ax.set_ylabel("out-of-fold prediction")
    ax.set_title(f"var{v}: MSE {mse:.3f}, R² {r2:.4f}", color=INK, fontsize=10, loc="left")
    ax.set_aspect("equal")
fig.tight_layout(); fig.savefig("figures/fig_oof.png"); plt.close(fig)
print("figures written")
