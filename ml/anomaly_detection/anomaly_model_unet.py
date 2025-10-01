from __future__ import annotations
import os, json, pickle
from dataclasses import dataclass, asdict
from typing import List, Tuple, Dict
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import GroupKFold
from transformers import get_cosine_schedule_with_warmup
from tqdm import tqdm
import gc
from scipy.interpolate import CubicSpline

from __future__ import annotations
import os, json, pickle
from dataclasses import dataclass, asdict
from typing import List, Tuple, Dict
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import GroupKFold
from transformers import get_cosine_schedule_with_warmup
from tqdm import tqdm
import gc
from scipy.interpolate import CubicSpline

@dataclass
class ADConfig:
    time_col: str = "timestamp"
    group_col: str = "group_id"
    seq_col: str = "sequence_id"
    target_cols: Tuple[str, str] = ("bpm", "uterus")
    seq_len: int = 5000
    threshold_method: str = "percentile"
    threshold_percentile: float = 95.0
    kalman_Q: float = 1e-5
    kalman_R: float = 1e-2
    n_folds: int = 5
    n_epochs: int = 50
    final_epochs: int = 36
    batch_size: int = 64
    learning_rate: float = 1e-4
    weight_decay: float = 1e-3
    patience: int = 5
    base_ch: int = 16
    depth: int = 3
    dropout: float = 0.1
    max_norm: float = 1.0
    num_warmup_steps_ratio: float = 0.05
    use_denoising: bool = True
    use_kalman: bool = True
    random_state: int = 228
    device: str = "cuda"

def _to_seconds_timestamp(s: pd.Series) -> pd.Series:
    if pd.api.types.is_numeric_dtype(s):
        return s.astype(float)
    dt = pd.to_datetime(s, errors="coerce")
    return dt.astype("int64") / 1e9

def kalman_1d(z: np.ndarray, Q: float = 1e-5, R: float = 1e-2, x0: float | None = None, P0: float = 1.0) -> np.ndarray:
    n = len(z)
    x_hat = np.empty(n)
    x_est = z[0] if x0 is None else x0
    P = P0
    for k in range(n):
        x_pred = x_est
        P = P + Q
        K = P / (P + R)
        x_est = x_pred + K * (z[k] - x_pred)
        P = (1.0 - K) * P
        x_hat[k] = x_est
    return x_hat

def apply_kalman_to_sequences(df: pd.DataFrame, sensor_cols: list[str] = ['bpm', 'uterus'], Q: float = 1e-5, R: float = 1e-2) -> pd.DataFrame:
    out = df.copy()
    for col in sensor_cols:
        if col in out.columns:
            out[col] = out[col].apply(lambda seq: kalman_1d(np.asarray(seq, dtype=float), Q=Q, R=R))
    return out

def fast_seq_agg(df: pd.DataFrame) -> pd.DataFrame:
    sc = ['key', 'sequence_id', 'group_id', 'target']
    seq_cols = [c for c in df.columns if c not in sc + ['timestamp']]
    static_cols = [c for c in sc if c in df.columns]

    df = df.sort_values(['sequence_id', 'timestamp']).reset_index(drop=True)
    seq_id_codes, _ = pd.factorize(df['sequence_id'])
    _, seq_start_idxs = np.unique(seq_id_codes, return_index=True)

    res = {'sequence_id': df['sequence_id'].values[seq_start_idxs]}
    
    for c in static_cols:
        res[c] = df[c].values[seq_start_idxs]

    for c in seq_cols:
        res[c] = np.split(df[c].values, seq_start_idxs[1:])

    res_df = pd.DataFrame(res)
    return res_df

def scale_per_fold(df: pd.DataFrame, cols=['bpm', 'uterus'], n_folds=5, eps=1e-8) -> pd.DataFrame:
    df_scaled = df.copy()
    for f in range(n_folds):
        train_mask = df['fold'] != f
        val_mask = df['fold'] == f
        for c in cols:
            all_vals = []
            for idx in df[train_mask].index:
                all_vals.extend(df.loc[idx, c])
            all_vals = np.asarray(all_vals, dtype=float)
            mean = all_vals.mean() if all_vals.size > 0 else 0.0
            std = all_vals.std() if all_vals.size > 0 else 1.0
            std = max(std, eps)
            for idx in df[val_mask].index:
                seq = np.asarray(df.loc[idx, c], dtype=float)
                df_scaled.at[idx, c] = (seq - mean) / std
    return df_scaled

# Augmentation functions
def jitter(x, sigma=0.05):
    return x + np.random.normal(loc=0., scale=sigma, size=x.shape)

def scaling_aug(x, sigma=0.1):
    x = np.expand_dims(x, axis=0)
    factor = np.random.normal(loc=1., scale=sigma, size=(x.shape[0], x.shape[2]))
    output = np.multiply(x, factor[:, np.newaxis, :])
    return np.squeeze(output, axis=0)

def magnitude_warp(x, sigma=0.2, knot=4):
    x = np.expand_dims(x, axis=0)
    orig_steps = np.arange(x.shape[1])
    random_warps = np.random.normal(loc=1.0, scale=sigma, size=(x.shape[0], knot+2, x.shape[2]))
    warp_steps = (np.ones((x.shape[2],1))*(np.linspace(0, x.shape[1]-1., num=knot+2))).T
    ret = np.zeros_like(x)
    for i, pat in enumerate(x):
        warper = np.array([CubicSpline(warp_steps[:,dim], random_warps[i,:,dim])(orig_steps) for dim in range(x.shape[2])]).T
        ret[i] = pat * warper
    return np.squeeze(ret, axis=0)

def time_warp(x, sigma=0.2, knot=4):
    x = np.expand_dims(x, axis=0)
    orig_steps = np.arange(x.shape[1])
    random_warps = np.random.normal(loc=1.0, scale=sigma, size=(x.shape[0], knot+2, x.shape[2]))
    warp_steps = (np.ones((x.shape[2],1))*(np.linspace(0, x.shape[1]-1., num=knot+2))).T
    ret = np.zeros_like(x)
    for i, pat in enumerate(x):
        for dim in range(x.shape[2]):
            time_warp_ = CubicSpline(warp_steps[:,dim], warp_steps[:,dim] * random_warps[i,:,dim])(orig_steps)
            scale = (x.shape[1]-1)/time_warp_[-1]
            ret[i,:,dim] = np.interp(orig_steps, np.clip(scale*time_warp_, 0, x.shape[1]-1), pat[:,dim]).T
    return np.squeeze(ret, axis=0)

def window_slice(x, reduce_ratio=0.9):
    x = np.expand_dims(x, axis=0)
    target_len = np.ceil(reduce_ratio*x.shape[1]).astype(int)
    if target_len >= x.shape[1]:
        return np.squeeze(x, axis=0)
    starts = np.random.randint(low=0, high=x.shape[1]-target_len, size=(x.shape[0])).astype(int)
    ends = (target_len + starts).astype(int)
    ret = np.zeros_like(x)
    for i, pat in enumerate(x):
        for dim in range(x.shape[2]):
            ret[i,:,dim] = np.interp(np.linspace(0, target_len, num=x.shape[1]), np.arange(target_len), pat[starts[i]:ends[i],dim]).T
    return np.squeeze(ret, axis=0)

def window_warp(x, window_ratio=0.1, scales=[0.5, 2.]):
    x = np.expand_dims(x, axis=0)
    warp_scales = np.random.choice(scales, x.shape[0])
    warp_size = np.ceil(window_ratio*x.shape[1]).astype(int)
    window_steps = np.arange(warp_size)
    window_starts = np.random.randint(low=1, high=x.shape[1]-warp_size-1, size=(x.shape[0])).astype(int)
    window_ends = (window_starts + warp_size).astype(int)
    ret = np.zeros_like(x)
    for i, pat in enumerate(x):
        for dim in range(x.shape[2]):
            start_seg = pat[:window_starts[i],dim]
            window_seg = np.interp(np.linspace(0, warp_size-1, num=int(warp_size*warp_scales[i])), window_steps, pat[window_starts[i]:window_ends[i],dim])
            end_seg = pat[window_ends[i]:,dim]
            warped = np.concatenate((start_seg, window_seg, end_seg))
            ret[i,:,dim] = np.interp(np.arange(x.shape[1]), np.linspace(0, x.shape[1]-1., num=warped.size), warped).T
    return np.squeeze(ret, axis=0)

class TSDataset(Dataset):
    def __init__(self, df, seq_len=5000, augment=False, denoise=True):
        self.df = df.reset_index(drop=True)
        self.seq_len = seq_len
        self.augment = augment
        self.denoise = denoise

        self.augmentations = [
            ('jitter', lambda x: jitter(x, sigma=0.05), 0.7),
            ('time_warp', lambda x: time_warp(x, sigma=0.2, knot=4), 0.5),
            ('scaling', lambda x: scaling_aug(x, sigma=0.1), 0.4),
            ('magnitude_warp', lambda x: magnitude_warp(x, sigma=0.2, knot=4), 0.3),
            ('window_slice', lambda x: window_slice(x, reduce_ratio=0.9), 0.3),
            ('window_warp', lambda x: window_warp(x, window_ratio=0.1, scales=[0.5, 2.]), 0.3)
        ]

    def __len__(self):
        return len(self.df)

    def maybe_augment(self, sequence):
        if not self.augment:
            return sequence
        seq_aug = sequence.copy()
        for _, aug_func, prob in self.augmentations:
            if np.random.random() < prob:
                seq_aug = aug_func(seq_aug)
        return seq_aug

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        bpm_seq = np.asarray(row['bpm'], dtype=float)
        uterus_seq = np.asarray(row['uterus'], dtype=float)

        min_len = min(len(bpm_seq), len(uterus_seq))
        bpm_seq = bpm_seq[:min_len]
        uterus_seq = uterus_seq[:min_len]

        target = np.stack([bpm_seq, uterus_seq], axis=1)

        if self.augment and self.denoise:
            x_in = self.maybe_augment(target)
        else:
            x_in = target if not self.augment else self.maybe_augment(target)

        if x_in.shape[0] > self.seq_len:
            x_in = x_in[-self.seq_len:]
            target = target[-self.seq_len:]
        if x_in.shape[0] < self.seq_len:
            pad_len = self.seq_len - x_in.shape[0]
            x_in = np.concatenate([np.zeros((pad_len, 2)), x_in], axis=0)
            target = np.concatenate([np.zeros((pad_len, 2)), target], axis=0)

        return {
            'input': torch.tensor(x_in, dtype=torch.float32),
            'target': torch.tensor(target, dtype=torch.float32),
            'idx': idx
        }

class ConvBlock1D(nn.Module):
    def __init__(self, in_ch, out_ch, dropout=0.0):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv1d(in_ch, out_ch, kernel_size=3, padding=1),
            nn.BatchNorm1d(out_ch),
            nn.ReLU(inplace=True),
            nn.Conv1d(out_ch, out_ch, kernel_size=3, padding=1),
            nn.BatchNorm1d(out_ch),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout) if dropout > 0 else nn.Identity(),
        )

    def forward(self, x):
        return self.conv(x)

class UNet1D(nn.Module):
    def __init__(self, in_channels=2, base_ch=16, depth=3, dropout=0.1, out_channels=2):
        super().__init__()
        chs = [base_ch * (2**i) for i in range(depth)]
        self.enc_blocks = nn.ModuleList()
        self.pools = nn.ModuleList()
        prev = in_channels
        for c in chs:
            self.enc_blocks.append(ConvBlock1D(prev, c, dropout=dropout))
            self.pools.append(nn.MaxPool1d(kernel_size=2))
            prev = c

        self.bottleneck = ConvBlock1D(prev, prev*2, dropout=dropout)

        self.upconvs = nn.ModuleList()
        self.dec_blocks = nn.ModuleList()
        dec_chs = list(reversed(chs))
        prev_dec = prev*2
        for c in dec_chs:
            self.upconvs.append(nn.ConvTranspose1d(prev_dec, c, kernel_size=2, stride=2))
            self.dec_blocks.append(ConvBlock1D(prev_dec, c, dropout=dropout))
            prev_dec = c

        self.out_conv = nn.Conv1d(prev_dec, out_channels, kernel_size=1)

    def forward(self, x):
        x = x.permute(0, 2, 1)
        skips = []
        for enc, pool in zip(self.enc_blocks, self.pools):
            x = enc(x)
            skips.append(x)
            x = pool(x)
        x = self.bottleneck(x)
        for up, dec, skip in zip(self.upconvs, self.dec_blocks, reversed(skips)):
            x = up(x)
            if x.size(-1) != skip.size(-1):
                diff = skip.size(-1) - x.size(-1)
                if diff > 0:
                    x = nn.functional.pad(x, (0, diff))
                else:
                    x = x[..., :skip.size(-1)]
            x = torch.cat([skip, x], dim=1)
            x = dec(x)
        x = self.out_conv(x)
        x = x.permute(0, 2, 1)
        return x

def per_sample_recon_error(pred, target, reduction='mean'):
    err = torch.abs(pred - target) if reduction == 'mae' else (pred - target)**2
    if reduction in ('mean', 'mae', 'mse'):
        return err.mean(dim=(1,2))
    elif reduction == 'sum':
        return err.sum(dim=(1,2))
    else:
        return err.view(err.size(0), -1).mean(dim=1)

def prepare_data(df: pd.DataFrame, cfg: ADConfig) -> pd.DataFrame:
    data = df.copy()
    data[cfg.time_col] = _to_seconds_timestamp(data[cfg.time_col])
    data = data.sort_values(['group_id', 'sequence_id', cfg.time_col]).reset_index(drop=True)
    
    sequence_lengths = data.groupby(['group_id', 'sequence_id']).size().reset_index(name='length')
    THOLD = 100
    gs_to_drop = sequence_lengths[sequence_lengths['length'] < THOLD][['group_id', 'sequence_id']]
    gs_to_drop['key'] = gs_to_drop['group_id'] + ':' + gs_to_drop['sequence_id']
    data['key'] = data['group_id'] + ':' + data['sequence_id']
    data = data[~data['key'].isin(gs_to_drop['key'])]
    
    data = fast_seq_agg(data)
    
    from sklearn.model_selection import StratifiedGroupKFold
    sgkf = StratifiedGroupKFold(n_splits=cfg.n_folds, shuffle=True, random_state=cfg.random_state)
    data['fold'] = -1
    
    data['target'] = data['target'].map({'regular': 0, 'hypoxia': 1})
    
    for fold, (train_idx, val_idx) in enumerate(sgkf.split(data, data['target'], data['group_id'])):
        data.loc[val_idx, 'fold'] = fold
    
    data = scale_per_fold(data, cols=['bpm', 'uterus'], n_folds=cfg.n_folds)
    
    if cfg.use_kalman:
        data = apply_kalman_to_sequences(data, sensor_cols=['bpm', 'uterus'], Q=cfg.kalman_Q, R=cfg.kalman_R)
    
    return data

def compute_threshold(errors: np.ndarray, method='percentile', p=95):
    if method == 'percentile':
        return np.percentile(errors, p)
    raise ValueError('unknown method')

def train_unet_autoencoder(df: pd.DataFrame, cfg: ADConfig, out_dir: str | None = None) -> Tuple[UNet1D, Dict]:
    device = torch.device(cfg.device if torch.cuda.is_available() else 'cpu')
    
    train_seq = prepare_data(df, cfg)
    
    gkf = GroupKFold(n_splits=cfg.n_folds)
    groups = train_seq['group_id'].values
    oof_errors = np.zeros(len(train_seq), dtype=np.float32)

    for fold, (train_idx, val_idx) in enumerate(gkf.split(train_seq, groups=groups)):
        print(f'Fold {fold+1}/{cfg.n_folds}')
        
        train_subset = train_seq.iloc[train_idx].reset_index(drop=True)
        val_subset = train_seq.iloc[val_idx].reset_index(drop=True)

        train_dataset = TSDataset(df=train_subset, seq_len=cfg.seq_len, augment=True, denoise=cfg.use_denoising)
        val_dataset = TSDataset(df=val_subset, seq_len=cfg.seq_len, augment=False, denoise=False)

        train_loader = DataLoader(train_dataset, batch_size=cfg.batch_size, shuffle=True, num_workers=2)
        val_loader = DataLoader(val_dataset, batch_size=cfg.batch_size, shuffle=False, num_workers=2)

        model = UNet1D(in_channels=2, base_ch=cfg.base_ch, depth=cfg.depth, dropout=cfg.dropout, out_channels=2).to(device)
        optimizer = torch.optim.AdamW(model.parameters(), lr=cfg.learning_rate, weight_decay=cfg.weight_decay)
        criterion = nn.L1Loss()

        num_training_steps = cfg.n_epochs * len(train_loader)
        num_warmup_steps = int(cfg.num_warmup_steps_ratio * num_training_steps)
        scheduler = get_cosine_schedule_with_warmup(optimizer, num_warmup_steps, num_training_steps)

        best_val_mae = float('inf')
        patience_counter = 0

        for epoch in range(cfg.n_epochs):
            model.train()
            for batch in tqdm(train_loader, desc=f'Epoch {epoch+1}', leave=False):
                optimizer.zero_grad()
                x = batch['input'].to(device)
                y = batch['target'].to(device)
                out = model(x)
                loss = criterion(out, y)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=cfg.max_norm)
                optimizer.step()
                scheduler.step()

            model.eval()
            val_mae = 0.0
            val_samples = 0
            with torch.no_grad():
                for batch in val_loader:
                    x = batch['input'].to(device)
                    y = batch['target'].to(device)
                    out = model(x)
                    mae = torch.mean(torch.abs(out - y)).item()
                    bsz = x.size(0)
                    val_mae += mae * bsz
                    val_samples += bsz
            
            val_mae /= val_samples
            print(f'Epoch {epoch+1}: val_mae={val_mae:.6f}')

            if val_mae < best_val_mae:
                best_val_mae = val_mae
                patience_counter = 0
            else:
                patience_counter += 1

            if patience_counter >= cfg.patience:
                print('Early stopping')
                break

        model.eval()
        with torch.no_grad():
            for batch in val_loader:
                x = batch['input'].to(device)
                y = batch['target'].to(device)
                idx = batch['idx'].numpy()
                out = model(x)
                err = per_sample_recon_error(out, y, reduction='mean').cpu().numpy()
                global_indices = val_subset.index.values[idx]
                oof_errors[global_indices] = err

        del model, optimizer, scheduler
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    dataset = TSDataset(df=train_seq, seq_len=cfg.seq_len, augment=True, denoise=cfg.use_denoising)
    loader = DataLoader(dataset, batch_size=cfg.batch_size, shuffle=True, num_workers=2)

    final_model = UNet1D(in_channels=2, base_ch=cfg.base_ch, depth=cfg.depth, dropout=cfg.dropout, out_channels=2).to(device)
    optimizer = torch.optim.AdamW(final_model.parameters(), lr=cfg.learning_rate, weight_decay=cfg.weight_decay)
    criterion = nn.L1Loss()

    num_training_steps = cfg.final_epochs * len(loader)
    num_warmup_steps = int(cfg.num_warmup_steps_ratio * num_training_steps)
    scheduler = get_cosine_schedule_with_warmup(optimizer, num_warmup_steps, num_training_steps)

    for epoch in range(cfg.final_epochs):
        final_model.train()
        for batch in tqdm(loader, desc=f'Final epoch {epoch+1}', leave=False):
            optimizer.zero_grad()
            x = batch['input'].to(device)
            y = batch['target'].to(device)
            out = final_model(x)
            loss = criterion(out, y)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(final_model.parameters(), max_norm=cfg.max_norm)
            optimizer.step()
            scheduler.step()

    threshold = compute_threshold(oof_errors, method=cfg.threshold_method, p=cfg.threshold_percentile)
    
    train_stats = {
        'oof_mae_mean': float(np.mean(oof_errors)),
        'oof_mae_std': float(np.std(oof_errors)),
        'threshold': float(threshold),
        'oof_anomaly_rate': float((oof_errors > threshold).mean())
    }
    
    scaler_stats = {}
    for col in ['bpm', 'uterus']:
        vals = []
        for seq in train_seq[col].values:
            vals.extend(seq)
        vals = np.asarray(vals, dtype=float)
        mean = float(vals.mean()) if vals.size else 0.0
        std = float(vals.std()) if vals.size else 1.0
        std = max(std, 1e-8)
        scaler_stats[col] = {'mean': mean, 'std': std}

    meta = {
        "config": asdict(cfg),
        "train_stats": train_stats,
        "scaler_stats": scaler_stats
    }

    if out_dir is not None:
        os.makedirs(out_dir, exist_ok=True)
        torch.save(final_model.state_dict(), os.path.join(out_dir, "ae_unet1d_final_unet_all_data.pt"))
        np.save(os.path.join(out_dir, "ae_unet1d_oof_recon_errors.npy"), oof_errors)
        with open(os.path.join(out_dir, "ae_unet1d_oof_info.json"), "w", encoding="utf-8") as f:
            json.dump(train_stats, f, ensure_ascii=False, indent=2)
        with open(os.path.join(out_dir, "meta.json"), "w", encoding="utf-8") as f:
            json.dump(meta, f, ensure_ascii=False, indent=2)

    return final_model, meta

def load_unet_artifacts(model_dir: str) -> Tuple[UNet1D, Dict]:
    with open(os.path.join(model_dir, "meta.json"), "r", encoding="utf-8") as f:
        meta = json.load(f)
    
    cfg_dict = meta["config"]
    device = torch.device(cfg_dict.get("device", "cuda") if torch.cuda.is_available() else "cpu")
    
    model = UNet1D(
        in_channels=2,
        base_ch=cfg_dict.get("base_ch", 16),
        depth=cfg_dict.get("depth", 3),
        dropout=cfg_dict.get("dropout", 0.1),
        out_channels=2
    )
    
    state_dict = torch.load(os.path.join(model_dir, "ae_unet1d_final_unet_all_data.pt"), map_location=device)
    model.load_state_dict(state_dict)
    model.to(device)
    model.eval()
    
    return model, meta