import argparse
import pandas as pd
from ts_model import TSConfig, train_multivariate_forecaster
from convert_data import merge_physiological_data

def main():
    parser = argparse.ArgumentParser(description="Train multi-target CatBoost forecaster")
    parser.add_argument("--input_csv", required=True, help="Path to data directory with physiological data structure")
    parser.add_argument("--out_dir", required=True, help="Directory to save model artifacts")
    parser.add_argument("--task_type", default="CPU", choices=["CPU", "GPU"], help="CatBoost task type")
    parser.add_argument("--iterations", type=int, default=3000)
    parser.add_argument("--learning_rate", type=float, default=0.03)
    parser.add_argument("--depth", type=int, default=6)
    parser.add_argument("--val_size", type=float, default=0.2)
    parser.add_argument("--verbose", type=int, default=200)
    args = parser.parse_args()

    df = merge_physiological_data(args.input_csv)
    cfg = TSConfig(
        task_type=args.task_type,
        iterations=args.iterations,
        learning_rate=args.learning_rate,
        depth=args.depth,
        val_size=args.val_size,
        verbose=args.verbose,
    )
    train_multivariate_forecaster(df, cfg, out_dir=args.out_dir)
    print(f"Saved model to {args.out_dir}")

if __name__ == "__main__":
    main()