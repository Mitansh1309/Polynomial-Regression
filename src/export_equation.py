"""Print a fitted model as an explicit polynomial in the raw inputs.

Usage: python src/export_equation.py --var 1 [--top 20]
"""
import argparse
import numpy as np
from models import PolyRegressor
from polybasis import to_monomial


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--var", type=int, required=True); ap.add_argument("--top", type=int, default=20)
    a = ap.parse_args()
    m = PolyRegressor.load(f"models/model_var{a.var}.npz")
    mono = to_monomial(m.basis_.exps_, m.coef_)
    names = [f"x{j+1}" for j in m.basis_.features_]
    items = sorted(mono.items(), key=lambda kv: -abs(kv[1]))
    nz = [kv for kv in items if abs(kv[1]) > 1e-10]
    print(f"var{a.var}: degree {m.degree}, {len(nz)} non-zero monomial coefficients "
          f"(max total degree present: {max(sum(k) for k, _ in nz)})")
    for k, c in nz[: a.top]:
        term = "·".join(f"{n}^{p}" if p > 1 else n for n, p in zip(names, k) if p) or "1"
        print(f"  {c:+12.5f}  {term}")


if __name__ == "__main__":
    main()
