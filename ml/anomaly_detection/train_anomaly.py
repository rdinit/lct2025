import argparse
import pandas as pd
from anomaly_model import ADConfig, train_isolation_forest
from convert_data import merge_physiological_data

def main():
    parser = argparse.ArgumentParser(description="Train IsolationForest anomaly detector")
    parser.add_argument("--input_csv", required=True, help="Path to data directory with physiological data structure")
    parser.add_argument("--out_dir", required=True, help="Where to save model_if.pkl and meta.json")
    parser.add_argument("--n_estimators", type=int, default=300)
    parser.add_argument("--max_samples", default="auto")
    parser.add_argument("--contamination", default="auto")
    parser.add_argument("--max_features", type=float, default=1.0)
    parser.add_argument("--val_size", type=float, default=0.2)
    parser.add_argument("--random_state", type=int, default=42)
    parser.add_argument("--n_jobs", type=int, default=None)
    args = parser.parse_args()

    df = merge_physiological_data(args.input_csv)
    cfg = ADConfig(
        n_estimators=args.n_estimators,
        max_samples=args.max_samples,
        contamination=args.contamination,
        max_features=args.max_features,
        val_size=args.val_size,
        random_state=args.random_state,
        n_jobs=args.n_jobs,
    )
    train_isolation_forest(df, cfg, out_dir=args.out_dir)
    print(f"Saved IsolationForest artifacts to {args.out_dir}")

if __name__ == "__main__":
    main()