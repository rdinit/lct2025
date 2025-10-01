import argparse
import pandas as pd
from ml.anomaly_detection.anomaly_model_unet import ADConfig, train_unet_autoencoder
from convert_data import merge_physiological_data

def main():
    parser = argparse.ArgumentParser(description="Train UNet autoencoder anomaly detector")
    parser.add_argument("--input_csv", required=True, help="Path to data directory with physiological data structure")
    parser.add_argument("--out_dir", required=True, help="Where to save UNet model and meta.json")
    
    parser.add_argument("--seq_len", type=int, default=5000, help="Sequence length for UNet")
    parser.add_argument("--threshold_method", default="percentile", help="Threshold computation method")
    parser.add_argument("--threshold_percentile", type=float, default=95.0, help="Percentile for threshold")
    parser.add_argument("--n_folds", type=int, default=5, help="Number of cross-validation folds")
    parser.add_argument("--n_epochs", type=int, default=50, help="Number of training epochs")
    parser.add_argument("--final_epochs", type=int, default=36, help="Number of final model training epochs")
    parser.add_argument("--batch_size", type=int, default=64, help="Training batch size")
    parser.add_argument("--learning_rate", type=float, default=1e-4, help="Learning rate")
    parser.add_argument("--weight_decay", type=float, default=1e-3, help="Weight decay")
    parser.add_argument("--patience", type=int, default=5, help="Early stopping patience")
    
    parser.add_argument("--base_ch", type=int, default=16, help="Base number of channels")
    parser.add_argument("--depth", type=int, default=3, help="UNet depth")
    parser.add_argument("--dropout", type=float, default=0.1, help="Dropout rate")
    
    parser.add_argument("--use_kalman", action="store_true", help="Apply Kalman filtering")
    parser.add_argument("--kalman_Q", type=float, default=1e-5, help="Kalman process noise")
    parser.add_argument("--kalman_R", type=float, default=1e-2, help="Kalman observation noise")
    parser.add_argument("--use_denoising", action="store_true", default=True, help="Use denoising autoencoder")
    
    parser.add_argument("--random_state", type=int, default=228, help="Random seed")
    parser.add_argument("--device", default="cuda", help="Device to use (cuda/cpu)")
    
    args = parser.parse_args()

    print("Loading and merging physiological data...")
    df = merge_physiological_data(args.input_csv)
    print(f"Loaded {len(df)} samples")
    
    cfg = ADConfig(
        seq_len=args.seq_len,
        threshold_method=args.threshold_method,
        threshold_percentile=args.threshold_percentile,
        n_folds=args.n_folds,
        n_epochs=args.n_epochs,
        final_epochs=args.final_epochs,
        batch_size=args.batch_size,
        learning_rate=args.learning_rate,
        weight_decay=args.weight_decay,
        patience=args.patience,
        base_ch=args.base_ch,
        depth=args.depth,
        dropout=args.dropout,
        use_kalman=args.use_kalman,
        kalman_Q=args.kalman_Q,
        kalman_R=args.kalman_R,
        use_denoising=args.use_denoising,
        random_state=args.random_state,
        device=args.device,
    )
    
    print("Training UNet autoencoder...")
    model, meta = train_unet_autoencoder(df, cfg, out_dir=args.out_dir)
    print(f"Saved UNet autoencoder artifacts to {args.out_dir}")
    print(f"OOF MAE: {meta['train_stats']['oof_mae_mean']:.6f} ± {meta['train_stats']['oof_mae_std']:.6f}")
    print(f"Threshold: {meta['train_stats']['threshold']:.6f}")
    print(f"Anomaly rate: {meta['train_stats']['oof_anomaly_rate']:.4f}")

if __name__ == "__main__":
    main()