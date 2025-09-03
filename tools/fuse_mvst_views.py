import argparse, pandas as pd
from functools import reduce


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--views", nargs="+", required=True,
                    help="List of per-view CSV files (e.g., 16/..csv 32/..csv ...)")
    ap.add_argument("--out", required=True, help="Output CSV path")
    args = ap.parse_args()

    dfs = [pd.read_csv(p) for p in args.views]
    keys = ["filepath","patient_id","mode","label"]
    fused = reduce(lambda l, r: pd.merge(l, r, on=keys, how="inner"), dfs)
    fused.to_csv(args.out, index=False)
    print("[MVST] fused:", args.out, fused.shape)


if __name__ == "__main__":
    main()
