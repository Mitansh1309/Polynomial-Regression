"""Inference: load a trained model and write the submission CSV.

Usage:  python src/predict.py --data data --roll BT2024046 --var 1
Writes: predictions/<ROLL>_pred_var<V>.csv  (single column `y`, same row order as test)
"""
import argparse, os
import numpy as np, pandas as pd
from models import PolyRegressor


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="data"); ap.add_argument("--roll", default="BT2024046")
    ap.add_argument("--var", type=int, required=True)
    args = ap.parse_args()
    te = pd.read_csv(f"{args.data}/{args.roll}_test_var{args.var}.csv")
    model = PolyRegressor.load(f"models/model_var{args.var}.npz")
    pred = model.predict(te.values)
    assert np.isfinite(pred).all() and len(pred) == len(te)
    os.makedirs("predictions", exist_ok=True)
    out = f"predictions/{args.roll}_pred_var{args.var}.csv"
    pd.DataFrame({"y": pred}).to_csv(out, index=False)
    print(f"wrote {out}  ({len(pred)} rows, mean={pred.mean():.3f}, std={pred.std():.3f})")


if __name__ == "__main__":
    main()
