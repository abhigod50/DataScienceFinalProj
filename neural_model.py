"""
Neural-network ensemble member for market-making signal generation.
====================================================================
Primary:  Bidirectional LSTM + temporal attention (PyTorch) — captures 4-hour
          sequential context that tree models cannot see.
Fallback: Sklearn MLP (two-hidden-layer dense net) — used automatically when
          PyTorch is unavailable (e.g. Windows c10.dll init failure).

Both backends expose the same interface:
    train_neural_model(feat_df, feature_cols) -> (model, metrics)
    save_neural_model(model, metrics, model_dir)
    load_neural_model(model_dir, n_features)  -> (model, meta)
    inference_neural(model, meta, feat_df, feature_cols) -> (dir_proba, vol_pred)
"""

import json
import pickle
from pathlib import Path
from typing import Optional, Tuple

import numpy as np
import pandas as pd
from sklearn.neural_network import MLPClassifier, MLPRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import roc_auc_score, mean_absolute_error, r2_score

from runtime_backends import select_runtime_backends

# ── PyTorch (optional) ─────────────────────────────────────────────────────────
try:
    import torch
    import torch.nn as nn
    from torch.utils.data import Dataset, DataLoader
    # Validate that torch actually loads (Windows DLL init can fail silently)
    _ = torch.zeros(1)
    TORCH_AVAILABLE = True
except Exception:
    TORCH_AVAILABLE = False

BACKEND = "pytorch" if TORCH_AVAILABLE else "sklearn_mlp"

# ── Hyper-parameters ───────────────────────────────────────────────────────────
# PyTorch LSTM
SEQ_LENGTH     = 48
HIDDEN_SIZE    = 256
NUM_LAYERS     = 2
DROPOUT        = 0.30
PT_BATCH_SIZE  = 512
PT_EPOCHS      = 60
PT_LR          = 3e-4
PT_WEIGHT_DECAY= 1e-4
PT_PATIENCE    = 12
VOL_LOSS_SCALE = 10.0
NN_BLEND_FAST_WEIGHT = 0.35
NN_BLEND_MID_WEIGHT = 0.65

# CPU fallback tuning: keep the real LSTM path, but shrink it enough to make
# retrains practical on hosts where the installed PyTorch build cannot use CUDA.
PT_CPU_HIDDEN_SIZE = 128
PT_CPU_NUM_LAYERS  = 1
PT_CPU_DROPOUT     = 0.10
PT_CPU_BATCH_SIZE  = 2048
PT_CPU_EPOCHS      = 12
PT_CPU_PATIENCE    = 4

# Sklearn MLP
MLP_HIDDEN     = (128, 64)  # smaller -> less overfitting
MLP_ALPHA      = 1e-2       # stronger L2 regularisation
MLP_BATCH      = 512
MLP_LR         = 3e-4
MLP_MAX_ITER   = 150
MLP_PATIENCE   = 15

_DEVICE: Optional[object] = None
_CUDA_PROBED = False
_CUDA_FALLBACK_REASON = ""


def _combine_direction_heads(
    fast_prob: np.ndarray | list[float],
    mid_prob: np.ndarray | list[float],
    *,
    fast_weight: float = NN_BLEND_FAST_WEIGHT,
    mid_weight: float = NN_BLEND_MID_WEIGHT,
) -> np.ndarray:
    fast = np.asarray(fast_prob, dtype=float)
    mid = np.asarray(mid_prob, dtype=float)
    total = max(float(fast_weight + mid_weight), 1e-9)
    blend = (float(fast_weight) * fast + float(mid_weight) * mid) / total
    return np.clip(blend, 0.0, 1.0)


def _probe_cuda_device() -> bool:
    """Return True only when CUDA is usable for recurrent ops on this machine."""
    global _CUDA_PROBED, _CUDA_FALLBACK_REASON
    if _CUDA_PROBED:
        return _CUDA_FALLBACK_REASON == ""
    _CUDA_PROBED = True

    selection = select_runtime_backends("training")
    torch_info = selection.get("libraries", {}).get("torch", {})
    if torch_info.get("selected") == "gpu":
        _CUDA_FALLBACK_REASON = ""
        return True

    _CUDA_FALLBACK_REASON = str(torch_info.get("reason", "") or "cpu_selected")
    return False


def _device() -> object:
    global _DEVICE
    if _DEVICE is None:
        if TORCH_AVAILABLE and _probe_cuda_device():
            _DEVICE = torch.device("cuda")
        else:
            if TORCH_AVAILABLE and _CUDA_FALLBACK_REASON:
                print(f"[neural] CUDA probe failed, falling back to CPU: {_CUDA_FALLBACK_REASON}")
            _DEVICE = torch.device("cpu")
    return _DEVICE


# ══════════════════════════════════════════════════════════════════════════════
#  PYTORCH LSTM BACKEND
# ══════════════════════════════════════════════════════════════════════════════
if TORCH_AVAILABLE:

    class _TSDataset(Dataset):
        def __init__(self, seqs, y_dir, y_vol):
            self.seqs  = torch.from_numpy(seqs).float()
            self.y_dir = torch.from_numpy(y_dir).float()
            self.y_vol = torch.from_numpy(y_vol).float()

        def __len__(self):
            return len(self.seqs)

        def __getitem__(self, idx):
            return self.seqs[idx], self.y_dir[idx], self.y_vol[idx]

    class MultiTaskLSTM(nn.Module):
        """Bidirectional LSTM → temporal attention → direction + volatility heads."""

        def __init__(self, input_size, hidden=HIDDEN_SIZE, layers=NUM_LAYERS, drop=DROPOUT):
            super().__init__()
            self.norm = nn.LayerNorm(input_size)
            self.lstm = nn.LSTM(input_size, hidden, layers, batch_first=True,
                                bidirectional=True, dropout=drop if layers > 1 else 0.0)
            H2 = hidden * 2
            self.attn = nn.Linear(H2, 1, bias=False)
            self.drop = nn.Dropout(drop)
            self.dir_head = nn.Sequential(
                nn.Linear(H2, 64), nn.GELU(), nn.Dropout(drop * 0.5), nn.Linear(64, 1))
            self.vol_head = nn.Sequential(
                nn.Linear(H2, 64), nn.GELU(), nn.Dropout(drop * 0.5),
                nn.Linear(64, 1), nn.Softplus())

        def forward(self, x):
            x = self.norm(x)
            out, _ = self.lstm(x)
            w = torch.softmax(self.attn(out), dim=1)
            ctx = self.drop((w * out).sum(dim=1))
            return self.dir_head(ctx).squeeze(-1), self.vol_head(ctx).squeeze(-1)

    class _TSTriDataset(Dataset):
        def __init__(self, seqs, y_fast, y_mid, y_vol):
            self.seqs = torch.from_numpy(seqs).float()
            self.y_fast = torch.from_numpy(y_fast).float()
            self.y_mid = torch.from_numpy(y_mid).float()
            self.y_vol = torch.from_numpy(y_vol).float()

        def __len__(self):
            return len(self.seqs)

        def __getitem__(self, idx):
            return self.seqs[idx], self.y_fast[idx], self.y_mid[idx], self.y_vol[idx]

    class TriHeadLSTM(nn.Module):
        """Shared trunk with fast/mid direction heads plus volatility head."""

        def __init__(self, input_size, hidden=HIDDEN_SIZE, layers=NUM_LAYERS, drop=DROPOUT):
            super().__init__()
            self.norm = nn.LayerNorm(input_size)
            self.lstm = nn.LSTM(input_size, hidden, layers, batch_first=True,
                                bidirectional=True, dropout=drop if layers > 1 else 0.0)
            h2 = hidden * 2
            self.attn = nn.Linear(h2, 1, bias=False)
            self.drop = nn.Dropout(drop)
            self.fast_head = nn.Sequential(
                nn.Linear(h2, 64), nn.GELU(), nn.Dropout(drop * 0.5), nn.Linear(64, 1))
            self.mid_head = nn.Sequential(
                nn.Linear(h2, 64), nn.GELU(), nn.Dropout(drop * 0.5), nn.Linear(64, 1))
            self.vol_head = nn.Sequential(
                nn.Linear(h2, 64), nn.GELU(), nn.Dropout(drop * 0.5),
                nn.Linear(64, 1), nn.Softplus())

        def forward(self, x):
            x = self.norm(x)
            out, _ = self.lstm(x)
            w = torch.softmax(self.attn(out), dim=1)
            ctx = self.drop((w * out).sum(dim=1))
            return (
                self.fast_head(ctx).squeeze(-1),
                self.mid_head(ctx).squeeze(-1),
                self.vol_head(ctx).squeeze(-1),
            )


def _build_seqs(X, y_dir, y_vol, seq_len=SEQ_LENGTH):
    n, F = X.shape
    row_b, col_b = X.strides
    seqs = np.lib.stride_tricks.as_strided(
        X, shape=(n - seq_len, seq_len, F), strides=(row_b, row_b, col_b),
    ).copy().astype(np.float32)
    return seqs, y_dir[seq_len - 1: n - 1].astype(np.float32), y_vol[seq_len - 1: n - 1].astype(np.float32)


def _build_multihead_seqs(X, y_fast, y_mid, y_vol, seq_len=SEQ_LENGTH):
    n, F = X.shape
    row_b, col_b = X.strides
    seqs = np.lib.stride_tricks.as_strided(
        X, shape=(n - seq_len, seq_len, F), strides=(row_b, row_b, col_b),
    ).copy().astype(np.float32)
    start = seq_len - 1
    stop = n - 1
    return (
        seqs,
        y_fast[start:stop].astype(np.float32),
        y_mid[start:stop].astype(np.float32),
        y_vol[start:stop].astype(np.float32),
    )


def _train_pytorch(feat_df, feature_cols):
    dev = _device()
    use_cpu_profile = (str(dev) == "cpu")
    hidden_size = PT_CPU_HIDDEN_SIZE if use_cpu_profile else HIDDEN_SIZE
    num_layers = PT_CPU_NUM_LAYERS if use_cpu_profile else NUM_LAYERS
    dropout = PT_CPU_DROPOUT if use_cpu_profile else DROPOUT
    batch_size = PT_CPU_BATCH_SIZE if use_cpu_profile else PT_BATCH_SIZE
    epochs = PT_CPU_EPOCHS if use_cpu_profile else PT_EPOCHS
    patience = PT_CPU_PATIENCE if use_cpu_profile else PT_PATIENCE
    print(f"\n{'=' * 60}")
    print(f"TRAINING NEURAL LSTM  (device={dev})")
    if use_cpu_profile:
        print(f"  CPU profile: hidden={hidden_size} layers={num_layers} batch={batch_size} epochs={epochs}")
    print(f"{'=' * 60}")

    X     = feat_df[feature_cols].values.astype(np.float32)
    y_dir = feat_df["direction"].values.astype(np.float32)
    y_vol = feat_df["future_volatility"].values.astype(np.float32)

    n = len(X); t1 = int(n * 0.70)
    mu = X[:t1].mean(0); sig = X[:t1].std(0) + 1e-8
    Xn = np.clip((X - mu) / sig, -10.0, 10.0)

    seqs, ld, lv = _build_seqs(Xn, y_dir, y_vol)
    ns = len(seqs); s1 = int(ns * 0.70); s2 = int(ns * 0.85)
    print(f"  Sequences: train={s1}  val={s2-s1}  test={ns-s2}")

    use_pin = (str(dev) == "cuda")
    def _dl(a, b, c, shuffle):
        ds = _TSDataset(a, b, c)
        return DataLoader(ds, batch_size=batch_size, shuffle=shuffle, pin_memory=use_pin, num_workers=0)

    tr_dl = _dl(seqs[:s1], ld[:s1], lv[:s1], True)
    va_dl = _dl(seqs[s1:s2], ld[s1:s2], lv[s1:s2], False)
    te_dl = _dl(seqs[s2:], ld[s2:], lv[s2:], False)

    model = MultiTaskLSTM(len(feature_cols), hidden=hidden_size, layers=num_layers, drop=dropout).to(dev)
    opt   = torch.optim.AdamW(model.parameters(), lr=PT_LR, weight_decay=PT_WEIGHT_DECAY)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs, eta_min=PT_LR * 0.05)
    bce = nn.BCEWithLogitsLoss(); hub = nn.HuberLoss(delta=0.002)
    use_amp = (str(dev) == "cuda")
    scaler  = torch.amp.GradScaler("cuda", enabled=use_amp)

    best_val, patience_cnt, best_state = float("inf"), 0, None
    for epoch in range(epochs):
        model.train()
        tr_losses = []
        for Xb, ydb, yvb in tr_dl:
            Xb, ydb, yvb = Xb.to(dev), ydb.to(dev), yvb.to(dev)
            opt.zero_grad(set_to_none=True)
            with torch.amp.autocast("cuda", enabled=use_amp):
                dl, vl = model(Xb)
                loss = bce(dl, ydb) + VOL_LOSS_SCALE * hub(vl, yvb)
            scaler.scale(loss).backward()
            scaler.unscale_(opt); nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            scaler.step(opt); scaler.update()
            tr_losses.append(loss.item())

        model.eval()
        val_losses = []
        with torch.no_grad():
            for Xb, ydb, yvb in va_dl:
                Xb, ydb, yvb = Xb.to(dev), ydb.to(dev), yvb.to(dev)
                with torch.amp.autocast("cuda", enabled=use_amp):
                    dl, vl = model(Xb)
                    val_losses.append((bce(dl, ydb) + VOL_LOSS_SCALE * hub(vl, yvb)).item())
        val_loss = float(np.mean(val_losses)); sched.step()

        if epoch % 10 == 0 or epoch == epochs - 1:
            print(f"  Epoch {epoch:3d}  train={np.mean(tr_losses):.4f}  val={val_loss:.4f}")

        if val_loss < best_val:
            best_val, patience_cnt = val_loss, 0
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
        else:
            patience_cnt += 1
            if patience_cnt >= patience:
                print(f"  Early stop at epoch {epoch}"); break

    model.load_state_dict(best_state)
    model.eval()
    dp, vp, yt, yv = [], [], [], []
    with torch.no_grad():
        for Xb, ydb, yvb in te_dl:
            Xb = Xb.to(dev)
            with torch.amp.autocast("cuda", enabled=use_amp):
                dl, vl = model(Xb)
            dp.extend(torch.sigmoid(dl).cpu().float().numpy())
            vp.extend(vl.cpu().float().numpy())
            yt.extend(ydb.numpy()); yv.extend(yvb.numpy())

    auc = roc_auc_score(yt, dp); vmae = mean_absolute_error(yv, vp); vr2 = r2_score(yv, vp)
    print(f"\n  LSTM Test  AUC={auc:.4f}  Vol MAE={vmae:.6f}  R²={vr2:.4f}")

    return (
        {"backend": "pytorch", "model": model, "norm_mean": mu.tolist(), "norm_std": sig.tolist()},
        {"backend": "pytorch", "auc": float(auc), "vol_mae": float(vmae), "vol_r2": float(vr2),
         "seq_length": SEQ_LENGTH, "hidden_size": hidden_size, "num_layers": num_layers,
         "dropout": dropout, "device": str(dev), "epochs": epochs,
         "norm_mean": mu.tolist(), "norm_std": sig.tolist()},
    )


# ══════════════════════════════════════════════════════════════════════════════
#  SKLEARN MLP BACKEND
# ══════════════════════════════════════════════════════════════════════════════

def _train_pytorch_multitask(feat_df, feature_cols):
    dev = _device()
    use_cpu_profile = (str(dev) == "cpu")
    hidden_size = PT_CPU_HIDDEN_SIZE if use_cpu_profile else HIDDEN_SIZE
    num_layers = PT_CPU_NUM_LAYERS if use_cpu_profile else NUM_LAYERS
    dropout = PT_CPU_DROPOUT if use_cpu_profile else DROPOUT
    batch_size = PT_CPU_BATCH_SIZE if use_cpu_profile else PT_BATCH_SIZE
    epochs = PT_CPU_EPOCHS if use_cpu_profile else PT_EPOCHS
    patience = PT_CPU_PATIENCE if use_cpu_profile else PT_PATIENCE
    print(f"\n{'=' * 60}")
    print(f"TRAINING NEURAL LSTM  (device={dev}, tri-head)")
    if use_cpu_profile:
        print(f"  CPU profile: hidden={hidden_size} layers={num_layers} batch={batch_size} epochs={epochs}")
    print(f"{'=' * 60}")

    X = feat_df[feature_cols].values.astype(np.float32)
    y_fast = feat_df["direction_1"].values.astype(np.float32)
    y_mid = feat_df["direction_3"].values.astype(np.float32)
    y_vol = feat_df["future_volatility"].values.astype(np.float32)

    n = len(X); t1 = int(n * 0.70)
    mu = X[:t1].mean(0); sig = X[:t1].std(0) + 1e-8
    Xn = np.clip((X - mu) / sig, -10.0, 10.0)

    seqs, yf, ym, yv = _build_multihead_seqs(Xn, y_fast, y_mid, y_vol)
    ns = len(seqs); s1 = int(ns * 0.70); s2 = int(ns * 0.85)
    print(f"  Sequences: train={s1}  val={s2-s1}  test={ns-s2}")

    use_pin = (str(dev) == "cuda")
    def _dl(a, b, c, d, shuffle):
        ds = _TSTriDataset(a, b, c, d)
        return DataLoader(ds, batch_size=batch_size, shuffle=shuffle, pin_memory=use_pin, num_workers=0)

    tr_dl = _dl(seqs[:s1], yf[:s1], ym[:s1], yv[:s1], True)
    va_dl = _dl(seqs[s1:s2], yf[s1:s2], ym[s1:s2], yv[s1:s2], False)
    te_dl = _dl(seqs[s2:], yf[s2:], ym[s2:], yv[s2:], False)

    model = TriHeadLSTM(len(feature_cols), hidden=hidden_size, layers=num_layers, drop=dropout).to(dev)
    opt = torch.optim.AdamW(model.parameters(), lr=PT_LR, weight_decay=PT_WEIGHT_DECAY)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs, eta_min=PT_LR * 0.05)
    bce = nn.BCEWithLogitsLoss(); hub = nn.HuberLoss(delta=0.002)
    use_amp = (str(dev) == "cuda")
    scaler = torch.amp.GradScaler("cuda", enabled=use_amp)

    best_val, patience_cnt, best_state = float("inf"), 0, None
    for epoch in range(epochs):
        model.train()
        tr_losses = []
        for Xb, yfb, ymb, yvb in tr_dl:
            Xb, yfb, ymb, yvb = Xb.to(dev), yfb.to(dev), ymb.to(dev), yvb.to(dev)
            opt.zero_grad(set_to_none=True)
            with torch.amp.autocast("cuda", enabled=use_amp):
                fl, ml, vl = model(Xb)
                loss = bce(fl, yfb) + (1.15 * bce(ml, ymb)) + VOL_LOSS_SCALE * hub(vl, yvb)
            scaler.scale(loss).backward()
            scaler.unscale_(opt); nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            scaler.step(opt); scaler.update()
            tr_losses.append(loss.item())

        model.eval()
        val_losses = []
        with torch.no_grad():
            for Xb, yfb, ymb, yvb in va_dl:
                Xb, yfb, ymb, yvb = Xb.to(dev), yfb.to(dev), ymb.to(dev), yvb.to(dev)
                with torch.amp.autocast("cuda", enabled=use_amp):
                    fl, ml, vl = model(Xb)
                    val_losses.append((bce(fl, yfb) + (1.15 * bce(ml, ymb)) + VOL_LOSS_SCALE * hub(vl, yvb)).item())
        val_loss = float(np.mean(val_losses)); sched.step()

        if epoch % 10 == 0 or epoch == epochs - 1:
            print(f"  Epoch {epoch:3d}  train={np.mean(tr_losses):.4f}  val={val_loss:.4f}")

        if val_loss < best_val:
            best_val, patience_cnt = val_loss, 0
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
        else:
            patience_cnt += 1
            if patience_cnt >= patience:
                print(f"  Early stop at epoch {epoch}")
                break

    model.load_state_dict(best_state)
    model.eval()
    fast_prob, mid_prob, vol_out, y_fast_true, y_mid_true, y_vol_true = [], [], [], [], [], []
    with torch.no_grad():
        for Xb, yfb, ymb, yvb in te_dl:
            Xb = Xb.to(dev)
            with torch.amp.autocast("cuda", enabled=use_amp):
                fl, ml, vl = model(Xb)
            fast_prob.extend(torch.sigmoid(fl).cpu().float().numpy())
            mid_prob.extend(torch.sigmoid(ml).cpu().float().numpy())
            vol_out.extend(vl.cpu().float().numpy())
            y_fast_true.extend(yfb.numpy()); y_mid_true.extend(ymb.numpy()); y_vol_true.extend(yvb.numpy())

    fast_prob = np.asarray(fast_prob, dtype=float)
    mid_prob = np.asarray(mid_prob, dtype=float)
    blend_prob = _combine_direction_heads(fast_prob, mid_prob)
    vol_out = np.asarray(vol_out, dtype=float)
    auc_fast = roc_auc_score(y_fast_true, fast_prob)
    auc_mid = roc_auc_score(y_mid_true, mid_prob)
    auc_blend = roc_auc_score(y_mid_true, blend_prob)
    vmae = mean_absolute_error(y_vol_true, vol_out)
    vr2 = r2_score(y_vol_true, vol_out)
    print(
        f"\n  LSTM Test  fast_auc={auc_fast:.4f}  mid_auc={auc_mid:.4f}  "
        f"blend_auc={auc_blend:.4f}  Vol MAE={vmae:.6f}  RÂ²={vr2:.4f}"
    )

    return (
        {
            "backend": "pytorch",
            "model": model,
            "norm_mean": mu.tolist(),
            "norm_std": sig.tolist(),
            "model_version": 2,
            "direction_blend_weights": [NN_BLEND_FAST_WEIGHT, NN_BLEND_MID_WEIGHT],
        },
        {
            "backend": "pytorch",
            "auc": float(auc_blend),
            "auc_fast": float(auc_fast),
            "auc_mid": float(auc_mid),
            "vol_mae": float(vmae),
            "vol_r2": float(vr2),
            "seq_length": SEQ_LENGTH,
            "hidden_size": hidden_size,
            "num_layers": num_layers,
            "dropout": dropout,
            "device": str(dev),
            "epochs": epochs,
            "model_version": 2,
            "task_heads": ["direction_1", "direction_3", "future_volatility"],
            "direction_blend_weights": [NN_BLEND_FAST_WEIGHT, NN_BLEND_MID_WEIGHT],
            "norm_mean": mu.tolist(),
            "norm_std": sig.tolist(),
        },
    )


def _fit_chronological(mlp, X_tr, y_tr, X_va, y_va, is_classifier: bool,
                       patience: int = 15, verbose: bool = False) -> None:
    """Train *mlp* via partial_fit, one full-pass epoch at a time.

    Validation uses the chronologically-last slice (X_va / y_va) so no future
    data leaks into the early-stopping signal.  Best weights are restored.
    """
    from sklearn.metrics import log_loss

    classes = np.array([0, 1]) if is_classifier else None
    best_val = float("inf")
    patience_cnt = 0
    best_coefs: Optional[list] = None
    best_icpts: Optional[list] = None

    for epoch in range(MLP_MAX_ITER):
        if is_classifier:
            mlp.partial_fit(X_tr, y_tr, classes=classes)
            p = mlp.predict_proba(X_va)
            val_score = log_loss(y_va, p)
        else:
            mlp.partial_fit(X_tr, y_tr)
            p = mlp.predict(X_va)
            val_score = float(np.mean(np.abs(p - y_va)))

        if verbose and (epoch % 10 == 0 or epoch == MLP_MAX_ITER - 1):
            print(f"    epoch {epoch:3d}  val={val_score:.6f}")

        if val_score < best_val:
            best_val, patience_cnt = val_score, 0
            best_coefs = [c.copy() for c in mlp.coefs_]
            best_icpts = [b.copy() for b in mlp.intercepts_]
        else:
            patience_cnt += 1
            if patience_cnt >= patience:
                if verbose:
                    print(f"    Early stop epoch {epoch}  best_val={best_val:.6f}")
                break

    # Restore best weights
    if best_coefs is not None:
        mlp.coefs_ = best_coefs
        mlp.intercepts_ = best_icpts


class _SklearnNNEnsemble:
    """Pair of MLP models (direction + volatility) with internal StandardScaler."""

    def __init__(self):
        self.scaler   = StandardScaler()
        self.vol_mean = 0.0   # fitted in .fit(); used to unscale vol predictions
        self.vol_std  = 1.0
        # Iteration driven by _fit_chronological via partial_fit; max_iter unused
        self.dir_mlp  = MLPClassifier(
            hidden_layer_sizes=MLP_HIDDEN, activation="relu",
            alpha=MLP_ALPHA, batch_size=MLP_BATCH, learning_rate_init=MLP_LR,
            max_iter=MLP_MAX_ITER, random_state=42, verbose=False,
        )
        self.dir_fast_mlp = MLPClassifier(
            hidden_layer_sizes=MLP_HIDDEN, activation="relu",
            alpha=MLP_ALPHA, batch_size=MLP_BATCH, learning_rate_init=MLP_LR,
            max_iter=MLP_MAX_ITER, random_state=43, verbose=False,
        )
        self.dir_mid_mlp = MLPClassifier(
            hidden_layer_sizes=MLP_HIDDEN, activation="relu",
            alpha=MLP_ALPHA, batch_size=MLP_BATCH, learning_rate_init=MLP_LR,
            max_iter=MLP_MAX_ITER, random_state=44, verbose=False,
        )
        self.vol_mlp  = MLPRegressor(
            hidden_layer_sizes=MLP_HIDDEN, activation="relu",
            alpha=MLP_ALPHA, batch_size=MLP_BATCH, learning_rate_init=MLP_LR,
            max_iter=MLP_MAX_ITER, random_state=42, verbose=False,
        )

    def fit(self, X_train, y_dir_train, y_vol_train, X_val, y_dir_val, y_vol_val):
        self.scaler.fit(X_train)
        Xn  = self.scaler.transform(X_train)
        Xnv = self.scaler.transform(X_val)

        # Normalise vol target so both heads see similar gradient scales
        self.vol_mean = float(y_vol_train.mean())
        self.vol_std  = float(y_vol_train.std() + 1e-8)
        yv_n  = (y_vol_train - self.vol_mean) / self.vol_std
        yvv_n = (y_vol_val   - self.vol_mean) / self.vol_std

        print("  Training direction MLP (chronological early stop)...")
        _fit_chronological(self.dir_mlp, Xn, y_dir_train, Xnv, y_dir_val,
                           is_classifier=True, patience=MLP_PATIENCE, verbose=True)

        print("  Training volatility MLP (chronological early stop)...")
        _fit_chronological(self.vol_mlp, Xn, yv_n, Xnv, yvv_n,
                           is_classifier=False, patience=MLP_PATIENCE, verbose=True)

    def predict_dir_proba(self, X) -> np.ndarray:
        return self.dir_mlp.predict_proba(self.scaler.transform(X))[:, 1]

    def predict_vol(self, X) -> np.ndarray:
        raw = self.vol_mlp.predict(self.scaler.transform(X))
        return np.maximum(raw * self.vol_std + self.vol_mean, 0.0)


class _SklearnNNEnsembleMulti:
    """Tier-2 fallback: fast/mid direction heads plus a volatility head."""

    def __init__(self):
        self.scaler = StandardScaler()
        self.vol_mean = 0.0
        self.vol_std = 1.0
        self.dir_fast_mlp = MLPClassifier(
            hidden_layer_sizes=MLP_HIDDEN, activation="relu",
            alpha=MLP_ALPHA, batch_size=MLP_BATCH, learning_rate_init=MLP_LR,
            max_iter=MLP_MAX_ITER, random_state=43, verbose=False,
        )
        self.dir_mid_mlp = MLPClassifier(
            hidden_layer_sizes=MLP_HIDDEN, activation="relu",
            alpha=MLP_ALPHA, batch_size=MLP_BATCH, learning_rate_init=MLP_LR,
            max_iter=MLP_MAX_ITER, random_state=44, verbose=False,
        )
        self.vol_mlp = MLPRegressor(
            hidden_layer_sizes=MLP_HIDDEN, activation="relu",
            alpha=MLP_ALPHA, batch_size=MLP_BATCH, learning_rate_init=MLP_LR,
            max_iter=MLP_MAX_ITER, random_state=42, verbose=False,
        )

    def fit(self, X_train, y_fast_train, y_mid_train, y_vol_train, X_val, y_fast_val, y_mid_val, y_vol_val):
        self.scaler.fit(X_train)
        Xn = self.scaler.transform(X_train)
        Xnv = self.scaler.transform(X_val)

        self.vol_mean = float(y_vol_train.mean())
        self.vol_std = float(y_vol_train.std() + 1e-8)
        yv_n = (y_vol_train - self.vol_mean) / self.vol_std
        yvv_n = (y_vol_val - self.vol_mean) / self.vol_std

        print("  Training fast direction MLP (chronological early stop)...")
        _fit_chronological(self.dir_fast_mlp, Xn, y_fast_train, Xnv, y_fast_val,
                           is_classifier=True, patience=MLP_PATIENCE, verbose=True)
        print("  Training mid direction MLP (chronological early stop)...")
        _fit_chronological(self.dir_mid_mlp, Xn, y_mid_train, Xnv, y_mid_val,
                           is_classifier=True, patience=MLP_PATIENCE, verbose=True)
        print("  Training volatility MLP (chronological early stop)...")
        _fit_chronological(self.vol_mlp, Xn, yv_n, Xnv, yvv_n,
                           is_classifier=False, patience=MLP_PATIENCE, verbose=True)

    def predict_dir_components(self, X) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        Xs = self.scaler.transform(X)
        fast = self.dir_fast_mlp.predict_proba(Xs)[:, 1]
        mid = self.dir_mid_mlp.predict_proba(Xs)[:, 1]
        return fast, mid, _combine_direction_heads(fast, mid)

    def predict_dir_proba(self, X) -> np.ndarray:
        _, _, blend = self.predict_dir_components(X)
        return blend

    def predict_vol(self, X) -> np.ndarray:
        raw = self.vol_mlp.predict(self.scaler.transform(X))
        return np.maximum(raw * self.vol_std + self.vol_mean, 0.0)


def _train_sklearn(feat_df, feature_cols):
    print(f"\n{'=' * 60}")
    print("TRAINING SKLEARN MLP  (PyTorch unavailable — using dense-net fallback)")
    print(f"  hidden={MLP_HIDDEN}  alpha={MLP_ALPHA}  max_iter={MLP_MAX_ITER}")
    print(f"{'=' * 60}")

    X     = feat_df[feature_cols].values
    y_dir = feat_df["direction"].values
    y_vol = feat_df["future_volatility"].values

    n  = len(X); t1 = int(n * 0.70); t2 = int(n * 0.85)
    X_train, y_dir_train, y_vol_train = X[:t1],    y_dir[:t1],    y_vol[:t1]
    X_val,   y_dir_val,   y_vol_val   = X[t1:t2],  y_dir[t1:t2],  y_vol[t1:t2]
    X_test,  y_dir_test,  y_vol_test  = X[t2:],    y_dir[t2:],    y_vol[t2:]
    print(f"  Rows: train={t1}  val={t2-t1}  test={n-t2}")

    model = _SklearnNNEnsemble()
    model.fit(X_train, y_dir_train, y_vol_train, X_val, y_dir_val, y_vol_val)

    dp   = model.predict_dir_proba(X_test)
    vp   = model.predict_vol(X_test)
    auc  = roc_auc_score(y_dir_test, dp)
    vmae = mean_absolute_error(y_vol_test, vp)
    vr2  = r2_score(y_vol_test, vp)
    print(f"\n  MLP Test  AUC={auc:.4f}  Vol MAE={vmae:.6f}  R²={vr2:.4f}")

    return (
        {"backend": "sklearn_mlp", "model": model},
        {"backend": "sklearn_mlp", "auc": float(auc), "vol_mae": float(vmae), "vol_r2": float(vr2)},
    )


def _train_sklearn_multitask(feat_df, feature_cols):
    print(f"\n{'=' * 60}")
    print("TRAINING SKLEARN MLP  (tri-head fallback)")
    print(f"  hidden={MLP_HIDDEN}  alpha={MLP_ALPHA}  max_iter={MLP_MAX_ITER}")
    print(f"{'=' * 60}")

    X = feat_df[feature_cols].values
    y_fast = feat_df["direction_1"].values
    y_mid = feat_df["direction_3"].values
    y_vol = feat_df["future_volatility"].values

    n = len(X); t1 = int(n * 0.70); t2 = int(n * 0.85)
    X_train, y_fast_train, y_mid_train, y_vol_train = X[:t1], y_fast[:t1], y_mid[:t1], y_vol[:t1]
    X_val, y_fast_val, y_mid_val, y_vol_val = X[t1:t2], y_fast[t1:t2], y_mid[t1:t2], y_vol[t1:t2]
    X_test, y_fast_test, y_mid_test, y_vol_test = X[t2:], y_fast[t2:], y_mid[t2:], y_vol[t2:]
    print(f"  Rows: train={t1}  val={t2-t1}  test={n-t2}")

    model = _SklearnNNEnsembleMulti()
    model.fit(X_train, y_fast_train, y_mid_train, y_vol_train, X_val, y_fast_val, y_mid_val, y_vol_val)

    fast_prob, mid_prob, blend_prob = model.predict_dir_components(X_test)
    vp = model.predict_vol(X_test)
    auc_fast = roc_auc_score(y_fast_test, fast_prob)
    auc_mid = roc_auc_score(y_mid_test, mid_prob)
    auc = roc_auc_score(y_mid_test, blend_prob)
    vmae = mean_absolute_error(y_vol_test, vp)
    vr2 = r2_score(y_vol_test, vp)
    print(f"\n  MLP Test  fast_auc={auc_fast:.4f}  mid_auc={auc_mid:.4f}  blend_auc={auc:.4f}  Vol MAE={vmae:.6f}  RÂ²={vr2:.4f}")

    return (
        {"backend": "sklearn_mlp", "model": model},
        {
            "backend": "sklearn_mlp",
            "auc": float(auc),
            "auc_fast": float(auc_fast),
            "auc_mid": float(auc_mid),
            "vol_mae": float(vmae),
            "vol_r2": float(vr2),
            "model_version": 2,
            "task_heads": ["direction_1", "direction_3", "future_volatility"],
            "direction_blend_weights": [NN_BLEND_FAST_WEIGHT, NN_BLEND_MID_WEIGHT],
        },
    )


# ══════════════════════════════════════════════════════════════════════════════
#  PUBLIC INTERFACE
# ══════════════════════════════════════════════════════════════════════════════
def train_neural_model(feat_df, feature_cols) -> Tuple:
    """Train neural model (LSTM if PyTorch available, MLP otherwise)."""
    if TORCH_AVAILABLE:
        return _train_pytorch_multitask(feat_df, feature_cols)
    return _train_sklearn_multitask(feat_df, feature_cols)


def save_neural_model(model_bundle, metrics: dict, model_dir: Path) -> None:
    """Save model bundle (backend-agnostic) to model_dir."""
    if model_bundle is None:
        return
    model_dir.mkdir(parents=True, exist_ok=True)
    backend = model_bundle.get("backend", "unknown")

    if backend == "pytorch" and TORCH_AVAILABLE:
        torch.save(model_bundle["model"].state_dict(), model_dir / "neural_model.pt")
        nm = metrics.copy()
        nm["norm_mean"] = model_bundle["norm_mean"]
        nm["norm_std"]  = model_bundle["norm_std"]
        with open(model_dir / "neural_model_meta.json", "w") as f:
            json.dump(nm, f, indent=2)
    elif backend == "sklearn_mlp":
        with open(model_dir / "neural_model_sklearn.pkl", "wb") as f:
            pickle.dump(model_bundle["model"], f)
        with open(model_dir / "neural_model_meta.json", "w") as f:
            json.dump(metrics, f, indent=2)

    print(f"  Neural model ({backend}) saved -> {model_dir}")


def load_neural_model(model_dir: Path, n_features: int) -> Tuple:
    """Load neural model.  Returns (bundle, meta) or (None, {})."""
    meta_path = model_dir / "neural_model_meta.json"
    if not meta_path.exists():
        return None, {}
    with open(meta_path) as f:
        meta = json.load(f)

    backend = meta.get("backend", "pytorch")

    if backend == "pytorch" and TORCH_AVAILABLE:
        pt_path = model_dir / "neural_model.pt"
        if not pt_path.exists():
            return None, {}
        try:
            model_version = int(meta.get("model_version", 1) or 1)
            model_cls = TriHeadLSTM if model_version >= 2 else MultiTaskLSTM
            model = model_cls(
                n_features,
                hidden=meta.get("hidden_size", HIDDEN_SIZE),
                layers=meta.get("num_layers", NUM_LAYERS),
                drop=meta.get("dropout", DROPOUT),
            )
            dev = _device()
            model.load_state_dict(torch.load(pt_path, map_location=dev, weights_only=True))
            model.to(dev).eval()
            return {
                "backend": "pytorch",
                "model": model,
                "norm_mean": meta["norm_mean"],
                "norm_std": meta["norm_std"],
                "model_version": model_version,
                "direction_blend_weights": meta.get("direction_blend_weights", [NN_BLEND_FAST_WEIGHT, NN_BLEND_MID_WEIGHT]),
            }, meta
        except Exception as e:
            print(f"  [neural] PyTorch load failed: {e}")
            return None, {}

    elif backend == "sklearn_mlp":
        pkl_path = model_dir / "neural_model_sklearn.pkl"
        if not pkl_path.exists():
            return None, {}
        try:
            with open(pkl_path, "rb") as f:
                model = pickle.load(f)
            return {"backend": "sklearn_mlp", "model": model}, meta
        except Exception as e:
            print(f"  [neural] sklearn MLP load failed: {e}")
            return None, {}

    return None, {}


def inference_neural_components_series(model_bundle, meta: dict, feat_df, feature_cols) -> dict[str, np.ndarray]:
    """Predict fast/mid/blend probabilities plus volatility for each row."""
    n_rows = len(feat_df)
    nan_arr = np.full(n_rows, np.nan, dtype=float)
    out = {
        "fast": nan_arr.copy(),
        "mid": nan_arr.copy(),
        "blend": nan_arr.copy(),
        "vol": nan_arr.copy(),
    }

    if model_bundle is None or not meta or n_rows == 0:
        return out

    backend = model_bundle.get("backend", "unknown")
    try:
        if backend == "pytorch" and TORCH_AVAILABLE:
            seq_len = int(meta.get("seq_length", SEQ_LENGTH))
            mu = np.array(meta["norm_mean"], dtype=np.float32)
            sig = np.array(meta["norm_std"], dtype=np.float32)
            X = feat_df[feature_cols].values.astype(np.float32)
            if len(X) < seq_len:
                return out

            X_df = pd.DataFrame(X, columns=feature_cols).replace([np.inf, -np.inf], np.nan).ffill()
            X = X_df.values.astype(np.float32)
            if np.isnan(X[-seq_len:]).any():
                return out

            sequences = []
            valid_idx = []
            for end_idx in range(seq_len - 1, len(X)):
                seq = X[end_idx - seq_len + 1:end_idx + 1]
                if np.isnan(seq).any():
                    continue
                sequences.append(seq)
                valid_idx.append(end_idx)
            if not sequences:
                return out

            Xn = np.clip((np.asarray(sequences, dtype=np.float32) - mu) / sig, -10.0, 10.0)
            dev = _device()
            Xt = torch.from_numpy(Xn).to(dev)
            use_amp = (str(dev) == "cuda")
            with torch.no_grad(), torch.amp.autocast("cuda", enabled=use_amp):
                outputs = model_bundle["model"](Xt)
            valid = np.asarray(valid_idx, dtype=int)
            model_version = int(model_bundle.get("model_version", meta.get("model_version", 1)) or 1)
            if model_version >= 2:
                fast_logits, mid_logits, vol_vals = outputs
                fast_vals = torch.sigmoid(fast_logits).cpu().float().numpy()
                mid_vals = torch.sigmoid(mid_logits).cpu().float().numpy()
                out["fast"][valid] = fast_vals
                out["mid"][valid] = mid_vals
                out["blend"][valid] = _combine_direction_heads(
                    fast_vals,
                    mid_vals,
                    fast_weight=float((model_bundle.get("direction_blend_weights") or meta.get("direction_blend_weights") or [NN_BLEND_FAST_WEIGHT, NN_BLEND_MID_WEIGHT])[0]),
                    mid_weight=float((model_bundle.get("direction_blend_weights") or meta.get("direction_blend_weights") or [NN_BLEND_FAST_WEIGHT, NN_BLEND_MID_WEIGHT])[1]),
                )
                out["vol"][valid] = vol_vals.cpu().float().numpy()
            else:
                dir_logits, vol_vals = outputs
                blend_vals = torch.sigmoid(dir_logits).cpu().float().numpy()
                out["fast"][valid] = blend_vals
                out["mid"][valid] = blend_vals
                out["blend"][valid] = blend_vals
                out["vol"][valid] = vol_vals.cpu().float().numpy()
            return out

        if backend == "sklearn_mlp":
            model = model_bundle["model"]
            X = feat_df[feature_cols].replace([np.inf, -np.inf], np.nan)
            valid_mask = ~X.isnull().any(axis=1)
            if not valid_mask.any():
                return out
            X_valid = X.loc[valid_mask].values
            if hasattr(model, "predict_dir_components"):
                fast_vals, mid_vals, blend_vals = model.predict_dir_components(X_valid)
            else:
                blend_vals = model.predict_dir_proba(X_valid)
                fast_vals = blend_vals
                mid_vals = blend_vals
            out["fast"][valid_mask.values] = fast_vals
            out["mid"][valid_mask.values] = mid_vals
            out["blend"][valid_mask.values] = blend_vals
            out["vol"][valid_mask.values] = model.predict_vol(X_valid)
            return out

    except Exception as e:
        print(f"  [neural] Series inference error: {e}")

    return out


def inference_neural_series(model_bundle, meta: dict, feat_df, feature_cols) -> Tuple[np.ndarray, np.ndarray]:
    """Predict directional probability and volatility for each row.

    Returns arrays shaped like ``feat_df`` with NaN for rows that cannot be
    scored (for example, the warm-up window of a sequence model).
    """
    components = inference_neural_components_series(model_bundle, meta, feat_df, feature_cols)
    return components["blend"], components["vol"]


def inference_neural(model_bundle, meta: dict, feat_df, feature_cols) -> Optional[Tuple[float, float]]:
    """
    Predict (dir_proba, vol_pred) from latest data.
    Returns None on failure or if model unavailable.
    """
    dir_series, vol_series = inference_neural_series(model_bundle, meta, feat_df, feature_cols)
    if len(dir_series) == 0 or not np.isfinite(dir_series[-1]) or not np.isfinite(vol_series[-1]):
        return None
    return float(dir_series[-1]), float(vol_series[-1])
