"""Model selection + final training for both problems.

Usage:  python src/train.py --data data --roll BT2024046 --var 1
Outputs: results/cv_var{V}.csv, models/model_var{V}.npz, results/summary_var{V}.json
"""
import argparse, json, os, time
import numpy as np, pandas as pd
from sklearn.model_selection import RepeatedKFold

from models import PolyRegressor

# Search grid per problem. Degrees above these are clearly worse in the
# exploratory sweep (see report); Lasso is skipped where the basis is huge.
GRID = {
    1: dict(degrees=range(1, 9), max_lasso_degree=7,
            ridge=[1e-3, 1e-2, 1e-1, 1, 3, 10, 30],
            lasso=[1e-3, 3e-3, 5e-3, 1e-2, 2e-2, 3e-2]),
    2: dict(degrees=range(1, 21), max_lasso_degree=12,
            ridge=[1e-4, 1e-3, 1e-2, 1e-1, 1, 10],
            lasso=[3e-4, 1e-3, 3e-3, 1e-2]),
}
N_SPLITS, N_REPEATS, SEED = 5, 3, 42


def n_clipped(X):
    return (np.abs(X) >= 1.0 - 1e-12).sum(1)


def shift_weights(Xtr, Xte, cap=10.0):
    """Importance weights w(k)=P_test(k)/P_train(k), k = #coordinates at +-1.

    The test sets contain more boundary/corner points than the training sets,
    so we also report a validation MSE re-weighted to the test distribution.
    """
    ktr, kte = n_clipped(Xtr), n_clipped(Xte)
    w = np.ones(len(Xtr))
    for k in np.unique(ktr):
        p_tr = np.mean(ktr == k); p_te = np.mean(kte == k)
        w[ktr == k] = min(p_te / p_tr, cap)
    return w / w.mean()


def candidates(var):
    g = GRID[var]
    for d in g["degrees"]:
        yield dict(degree=d, method="ols", alpha=0.0)
        for a in g["ridge"]:
            yield dict(degree=d, method="ridge", alpha=a)
        if d <= g["max_lasso_degree"]:
            for a in g["lasso"]:
                yield dict(degree=d, method="lasso", alpha=a)
                yield dict(degree=d, method="relaxed", alpha=a)


def evaluate(cfg, X, y, w, splits):
    mse, wmse, nterms = [], [], []
    for tr, va in splits:
        m = PolyRegressor(**cfg).fit(X[tr], y[tr])
        e = (m.predict(X[va]) - y[va]) ** 2
        mse.append(e.mean()); wmse.append(np.average(e, weights=w[va]))
        nterms.append(len(m.support_))
    return dict(cfg, n_terms=float(np.mean(nterms)),
                cv_mse=float(np.mean(mse)), cv_mse_se=float(np.std(mse) / np.sqrt(len(mse))),
                cv_wmse=float(np.mean(wmse)))


def select(cv):
    """One-standard-error rule on the test-weighted CV MSE.

    1. find the configuration with the lowest weighted CV MSE;
    2. keep every configuration within one standard error of it;
    3. among those, take the lowest polynomial degree (simplest model),
       and within that degree the lowest weighted CV MSE.
    """
    b = cv.loc[cv["cv_wmse"].idxmin()]
    ok = cv[cv["cv_wmse"] <= b.cv_wmse + b.cv_mse_se]
    ok = ok[ok["degree"] == ok["degree"].min()]
    return ok.loc[ok["cv_wmse"].idxmin()]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="data"); ap.add_argument("--roll", default="BT2024046")
    ap.add_argument("--var", type=int, required=True)
    ap.add_argument("--reuse-cv", action="store_true",
                    help="skip the grid search and reuse results/cv_var{V}.csv")
    args = ap.parse_args()
    os.makedirs("results", exist_ok=True); os.makedirs("models", exist_ok=True)

    tr = pd.read_csv(f"{args.data}/{args.roll}_train_var{args.var}.csv")
    te = pd.read_csv(f"{args.data}/{args.roll}_test_var{args.var}.csv")
    X, y = tr.drop(columns="y").values, tr["y"].values
    w = shift_weights(X, te.values)
    splits = list(RepeatedKFold(n_splits=N_SPLITS, n_repeats=N_REPEATS, random_state=SEED).split(X))

    rows, t0 = [], time.time()
    for cfg in ([] if args.reuse_cv else candidates(args.var)):
        p_full = PolyRegressor(cfg["degree"]).fit(X[:5], y[:5]).basis_.n_terms
        if cfg["method"] == "ols" and p_full > 0.8 * len(X) * (N_SPLITS - 1) / N_SPLITS:
            continue  # under-determined; OLS meaningless
        rows.append(evaluate(cfg, X, y, w, splits))
        r = rows[-1]
        print(f"[{time.time()-t0:6.0f}s] d={r['degree']:2d} {r['method']:7s} a={r['alpha']:<7g} "
              f"terms={r['n_terms']:6.0f} mse={r['cv_mse']:.4f} wmse={r['cv_wmse']:.4f}", flush=True)
    if args.reuse_cv:
        cv = pd.read_csv(f"results/cv_var{args.var}.csv")
    else:
        cv = pd.DataFrame(rows)
        cv.to_csv(f"results/cv_var{args.var}.csv", index=False)

    best = select(cv)
    cfg = dict(degree=int(best.degree), method=best.method, alpha=float(best.alpha))
    model = PolyRegressor(**cfg).fit(X, y)
    model.save(f"models/model_var{args.var}.npz")

    resid = model.predict(X) - y
    summary = dict(var=args.var, chosen=cfg, n_terms=int(len(model.support_)),
                   cv_mse=float(best.cv_mse), cv_mse_se=float(best.cv_mse_se),
                   cv_wmse=float(best.cv_wmse),
                   cv_r2=float(1 - best.cv_mse / y.var()),
                   train_mse=float(np.mean(resid ** 2)),
                   train_r2=float(1 - np.mean(resid ** 2) / y.var()))
    json.dump(summary, open(f"results/summary_var{args.var}.json", "w"), indent=2)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
