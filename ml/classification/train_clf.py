import argparse
import pandas as pd
from ts_model_clf import CLFConfig, train_classifier, train_classifier_cv
from convert_data import merge_physiological_data

def main():
    parser = argparse.ArgumentParser(description="Train CatBoost classification model for time series")
    parser.add_argument("--input_csv", required=True, help="Path to CSV file with physiological data")
    parser.add_argument("--out_dir", required=True, help="Directory to save model artifacts")
    parser.add_argument("--task_type", default="CPU", choices=["CPU", "GPU"], help="CatBoost task type")
    parser.add_argument("--iterations", type=int, default=1000)
    parser.add_argument("--learning_rate", type=float, default=0.03)
    parser.add_argument("--depth", type=int, default=6)
    parser.add_argument("--val_size", type=float, default=0.2)
    parser.add_argument("--n_splits", type=int, default=5)
    parser.add_argument("--verbose", type=int, default=200)
    parser.add_argument("--cv_only", action="store_true", help="Only run cross-validation, don't save final model")
    args = parser.parse_args()

    df = merge_physiological_data(args.input_csv)
    
    cfg = CLFConfig(
        task_type=args.task_type,
        iterations=args.iterations,
        learning_rate=args.learning_rate,
        depth=args.depth,
        val_size=args.val_size,
        n_splits=args.n_splits,
        verbose=args.verbose,
    )
    
    if args.cv_only:
        print("Running cross-validation...")
        results = train_classifier_cv(df, cfg)
        print(f"Cross-validation completed. Mean F1: {results.get('fold_f1', [])}")
    else:
        print("Running cross-validation...")
        cv_results = train_classifier_cv(df, cfg)
        print(f"Cross-validation completed. Mean F1: {cv_results.get('fold_f1', [])}")
        
        print("Training final model...")
        model, le, meta = train_classifier(df, cfg, out_dir=args.out_dir)
        print(f"Final model saved to {args.out_dir}")

if __name__ == "__main__":
    main()