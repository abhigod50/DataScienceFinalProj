"""
Optimization research: test specific improvements in isolation.
Each experiment measures a single change against the current baseline.
"""

import json
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow.feather as pf
from sklearn.metrics import accuracy_score, roc_auc_score, mean_absolute_error, r2_score
import xgboost as xgb
import lightgbm as lgb

from config import DATA_DIR, TRAIN_RATIO, VAL_RATIO, XGB_DIR_PARAMS, XGB_VOL_FALLBACK_PARAMS
from features import compute_features, compute_labels, get_feature_columns

def load_and_prepare():
    df = pf.read_feather(str(DATA_DIR / 'ETH_USDT-5m.feather'), memory_map=False)
    btc_df = pf.read_feather(str(DATA_DIR / 'BTC_USDT-5m.feather'), memory_map=False)
    sol_df = pf.read_feather(str(DATA_DIR / 'SOL_USDT-5m.feather'), memory_map=False)
    return df, btc_df, sol_df

def split(feat_df, feature_cols):
    n = len(feat_df)
    t1, t2 = int(n * TRAIN_RATIO), int(n * (TRAIN_RATIO + VAL_RATIO))
    tr, va, te = feat_df.iloc[:t1], feat_df.iloc[t1:t2], feat_df.iloc[t2:]
    return (tr[feature_cols].values, va[feature_cols].values, te[feature_cols].values,
            tr['direction'].values, va['direction'].values, te['direction'].values,
            tr['future_volatility'].values, va['future_volatility'].values, te['future_volatility'].values)

def eval_dir(X_tr, y_tr, X_va, y_va, X_te, y_te, params, label):
    p = params.copy()
    n_est = p.pop('n_estimators', 800)
    early = p.pop('early_stopping_rounds', 60)
    m = xgb.XGBClassifier(n_estimators=n_est, early_stopping_rounds=early, **p)
    m.fit(X_tr, y_tr, eval_set=[(X_va, y_va)], verbose=0)
    proba = m.predict_proba(X_te)[:, 1]
    auc = roc_auc_score(y_te, proba)
    acc = accuracy_score(y_te, (proba > 0.5).astype(int))
    print(f'  {label:>35}: AUC={auc:.4f}  Acc={acc:.4f}')
    return auc, acc, m

def eval_vol(X_tr, y_tr, X_va, y_va, X_te, y_te, params, label):
    p = params.copy()
    n_est = p.pop('n_estimators', 600)
    early = p.pop('early_stopping_rounds', 50)
    m = xgb.XGBRegressor(n_estimators=n_est, early_stopping_rounds=early, **p)
    m.fit(X_tr, y_tr, eval_set=[(X_va, y_va)], verbose=0)
    pred = m.predict(X_te)
    mae = mean_absolute_error(y_te, pred)
    r2 = r2_score(y_te, pred)
    corr = np.corrcoef(pred, y_te)[0,1]
    print(f'  {label:>35}: MAE={mae:.6f}  R2={r2:.4f}  Corr={corr:.4f}')
    return mae, r2, corr

# === CURRENT BASELINE PARAMS (from shared config) ===
BASE_DIR_PARAMS = {k: v for k, v in XGB_DIR_PARAMS.items()}
BASE_DIR_PARAMS["verbosity"] = 0  # quiet for experiments

BASE_VOL_PARAMS = {k: v for k, v in XGB_VOL_FALLBACK_PARAMS.items()}
BASE_VOL_PARAMS["verbosity"] = 0

df, btc_df, sol_df = load_and_prepare()
feat_df = compute_features(df, btc_df)
feat_df = compute_labels(feat_df, horizon=6)
feature_cols_base = get_feature_columns(feat_df)
feat_df = feat_df.dropna(subset=feature_cols_base + ['direction', 'future_volatility']).reset_index(drop=True)

X_tr, X_va, X_te, y_dir_tr, y_dir_va, y_dir_te, y_vol_tr, y_vol_va, y_vol_te = split(feat_df, feature_cols_base)

results = {
    'generated_at': datetime.now(timezone.utc).isoformat(),
    'baseline': {},
    'comparisons': [],
    'headline_findings': [],
}

# ════════════════════════════════════════════════════════════════════
print('=' * 70)
print('EXPERIMENT 1: CURRENT BASELINE')
print('=' * 70)
base_auc, base_acc, _ = eval_dir(X_tr, y_dir_tr, X_va, y_dir_va, X_te, y_dir_te, BASE_DIR_PARAMS, 'XGB Direction Baseline')
base_mae, base_r2, base_corr = eval_vol(X_tr, y_vol_tr, X_va, y_vol_va, X_te, y_vol_te, BASE_VOL_PARAMS, 'XGB Volatility Baseline')
results['baseline'] = {
    'label': 'baseline',
    'feature_count': len(feature_cols_base),
    'direction_auc': round(float(base_auc), 4),
    'direction_accuracy': round(float(base_acc), 4),
    'volatility_mae': round(float(base_mae), 6),
    'volatility_r2': round(float(base_r2), 4),
    'volatility_corr': round(float(base_corr), 4),
}

# ════════════════════════════════════════════════════════════════════
print('\n' + '=' * 70)
print('EXPERIMENT 2: REMOVE REDUNDANT FEATURES (>0.95 corr)')
print('=' * 70)
# Remove one from each highly-correlated pair
remove_feats = {'roc_12', 'close_1h_return', 'close_15m_return', 'vol_regime',
                'roc_6', 'natr_14', 'log_return', 'macd_signal'}
pruned_cols = [c for c in feature_cols_base if c not in remove_feats]
print(f'  Features: {len(feature_cols_base)} -> {len(pruned_cols)}')

X_tr_p, X_va_p, X_te_p = feat_df.iloc[:int(len(feat_df)*0.7)][pruned_cols].values, \
    feat_df.iloc[int(len(feat_df)*0.7):int(len(feat_df)*0.85)][pruned_cols].values, \
    feat_df.iloc[int(len(feat_df)*0.85):][pruned_cols].values

pruned_auc, pruned_acc, _ = eval_dir(X_tr_p, y_dir_tr, X_va_p, y_dir_va, X_te_p, y_dir_te, BASE_DIR_PARAMS, 'XGB Direction Pruned')
pruned_mae, pruned_r2, pruned_corr = eval_vol(X_tr_p, y_vol_tr, X_va_p, y_vol_va, X_te_p, y_vol_te, BASE_VOL_PARAMS, 'XGB Volatility Pruned')
results['comparisons'].append({
    'group': 'feature_set',
    'label': 'pruned_features',
    'feature_count': len(pruned_cols),
    'direction_auc': round(float(pruned_auc), 4),
    'direction_accuracy': round(float(pruned_acc), 4),
    'volatility_mae': round(float(pruned_mae), 6),
    'volatility_r2': round(float(pruned_r2), 4),
    'volatility_corr': round(float(pruned_corr), 4),
})

# ════════════════════════════════════════════════════════════════════
print('\n' + '=' * 70)
print('EXPERIMENT 3: ADD SOL CROSS-ASSET FEATURES')
print('=' * 70)
feat_df2 = compute_features(df, btc_df)
sol_close = sol_df.set_index('date')['close'].reindex(df['date']).values
sol_s = pd.Series(sol_close, index=df.index)
feat_df2['sol_return_1'] = sol_s.pct_change(1)
feat_df2['sol_return_6'] = sol_s.pct_change(6)
feat_df2['sol_eth_corr_24'] = feat_df2['returns_1'].rolling(24).corr(feat_df2['sol_return_1'])
feat_df2['sol_return_lag1'] = feat_df2['sol_return_1'].shift(1)
feat_df2['sol_return_lag3'] = feat_df2['sol_return_1'].shift(3)
# SOL-BTC spread as macro risk signal
btc_close = btc_df.set_index('date')['close'].reindex(df['date']).values
btc_s = pd.Series(btc_close, index=df.index)
feat_df2['sol_btc_ratio'] = sol_s / btc_s
feat_df2['sol_btc_ratio_delta'] = feat_df2['sol_btc_ratio'].pct_change(6)

feat_df2 = compute_labels(feat_df2, horizon=6)
feature_cols_sol = get_feature_columns(feat_df2)
feat_df2 = feat_df2.dropna(subset=feature_cols_sol + ['direction', 'future_volatility']).reset_index(drop=True)
X_tr_s, X_va_s, X_te_s, ydt, ydv, ydte, yvt, yvv, yvte = split(feat_df2, feature_cols_sol)
print(f'  Features: {len(feature_cols_sol)} (added {len(feature_cols_sol)-len(feature_cols_base)} SOL features)')
sol_auc, sol_acc, _ = eval_dir(X_tr_s, ydt, X_va_s, ydv, X_te_s, ydte, BASE_DIR_PARAMS, 'XGB Direction +SOL')
sol_mae, sol_r2, sol_corr = eval_vol(X_tr_s, yvt, X_va_s, yvv, X_te_s, yvte, BASE_VOL_PARAMS, 'XGB Volatility +SOL')
results['comparisons'].append({
    'group': 'feature_set',
    'label': 'sol_cross_asset',
    'feature_count': len(feature_cols_sol),
    'direction_auc': round(float(sol_auc), 4),
    'direction_accuracy': round(float(sol_acc), 4),
    'volatility_mae': round(float(sol_mae), 6),
    'volatility_r2': round(float(sol_r2), 4),
    'volatility_corr': round(float(sol_corr), 4),
})

# ════════════════════════════════════════════════════════════════════
print('\n' + '=' * 70)
print('EXPERIMENT 4: LIGHTGBM vs XGBOOST')
print('=' * 70)
lgb_dir_params = {
    'device': 'gpu', 'boosting_type': 'gbdt', 'objective': 'binary',
    'metric': 'auc', 'n_estimators': 1500, 'max_depth': 5,
    'learning_rate': 0.01, 'subsample': 0.75, 'colsample_bytree': 0.6,
    'min_child_samples': 20, 'reg_alpha': 0.5, 'reg_lambda': 2.0,
    'verbose': -1, 'n_jobs': -1,
}
lgb_vol_params = {
    'device': 'gpu', 'boosting_type': 'gbdt', 'objective': 'regression',
    'metric': 'mae', 'n_estimators': 1200, 'max_depth': 4,
    'learning_rate': 0.01, 'subsample': 0.75, 'colsample_bytree': 0.6,
    'min_child_samples': 25, 'reg_alpha': 0.5, 'reg_lambda': 1.0,
    'verbose': -1, 'n_jobs': -1,
}

lgb_dir = lgb.LGBMClassifier(**lgb_dir_params)
lgb_dir.fit(X_tr, y_dir_tr, eval_set=[(X_va, y_dir_va)],
            callbacks=[lgb.early_stopping(80, verbose=False)])
proba = lgb_dir.predict_proba(X_te)[:, 1]
auc = roc_auc_score(y_dir_te, proba)
acc = accuracy_score(y_dir_te, (proba > 0.5).astype(int))
print(f'  {"LightGBM Direction":>35}: AUC={auc:.4f}  Acc={acc:.4f}')

lgb_vol = lgb.LGBMRegressor(**lgb_vol_params)
lgb_vol.fit(X_tr, y_vol_tr, eval_set=[(X_va, y_vol_va)],
            callbacks=[lgb.early_stopping(50, verbose=False)])
vpred = lgb_vol.predict(X_te)
mae = mean_absolute_error(y_vol_te, vpred)
r2 = r2_score(y_vol_te, vpred)
corr = np.corrcoef(vpred, y_vol_te)[0,1]
print(f'  {"LightGBM Volatility":>35}: MAE={mae:.6f}  R2={r2:.4f}  Corr={corr:.4f}')
results['comparisons'].append({
    'group': 'model_family',
    'label': 'lightgbm',
    'feature_count': len(feature_cols_base),
    'direction_auc': round(float(auc), 4),
    'direction_accuracy': round(float(acc), 4),
    'volatility_mae': round(float(mae), 6),
    'volatility_r2': round(float(r2), 4),
    'volatility_corr': round(float(corr), 4),
})

# ════════════════════════════════════════════════════════════════════
print('\n' + '=' * 70)
print('EXPERIMENT 5: DIFFERENT PREDICTION HORIZONS')
print('=' * 70)
horizon_results = []
for horizon in [3, 6, 12]:
    feat_h = compute_features(df, btc_df)
    feat_h = compute_labels(feat_h, horizon=horizon)
    fc = get_feature_columns(feat_h)
    feat_h = feat_h.dropna(subset=fc + ['direction', 'future_volatility']).reset_index(drop=True)
    Xtr, Xva, Xte, ytr, yva, yte, _, _, _ = split(feat_h, fc)
    p = BASE_DIR_PARAMS.copy()
    p['n_estimators'] = 500
    h_auc, h_acc, _ = eval_dir(Xtr, ytr, Xva, yva, Xte, yte, p, f'horizon={horizon} ({horizon*5}min)')
    horizon_results.append({
        'group': 'horizon',
        'label': f'horizon_{horizon}',
        'horizon_candles': horizon,
        'feature_count': len(fc),
        'direction_auc': round(float(h_auc), 4),
        'direction_accuracy': round(float(h_acc), 4),
    })
results['comparisons'].extend(horizon_results)

# ════════════════════════════════════════════════════════════════════
print('\n' + '=' * 70)
print('EXPERIMENT 6: 3-CLASS DIRECTION (UP/FLAT/DOWN)')
print('=' * 70)
threshold = 0.001  # 0.1% move = meaningful direction
feat_3c = feat_df.copy()
feat_3c['direction_3c'] = 1  # flat
feat_3c.loc[feat_3c['future_return'] > threshold, 'direction_3c'] = 2   # up
feat_3c.loc[feat_3c['future_return'] < -threshold, 'direction_3c'] = 0  # down

y3_tr = feat_3c.iloc[:int(len(feat_3c)*0.7)]['direction_3c'].values
y3_va = feat_3c.iloc[int(len(feat_3c)*0.7):int(len(feat_3c)*0.85)]['direction_3c'].values
y3_te = feat_3c.iloc[int(len(feat_3c)*0.85):]['direction_3c'].values
print(f'  Class distribution: down={np.mean(y3_te==0):.3f} flat={np.mean(y3_te==1):.3f} up={np.mean(y3_te==2):.3f}')

p3 = BASE_DIR_PARAMS.copy()
p3['objective'] = 'multi:softprob'
p3['num_class'] = 3
p3['eval_metric'] = 'mlogloss'
m3 = xgb.XGBClassifier(**{k:v for k,v in p3.items() if k not in ('n_estimators', 'early_stopping_rounds')},
                        n_estimators=800, early_stopping_rounds=60)
m3.fit(X_tr, y3_tr, eval_set=[(X_va, y3_va)], verbose=0)
pred3 = m3.predict(X_te)
acc3 = accuracy_score(y3_te, pred3)
# Convert to binary for comparison
pred3_binary = (pred3 == 2).astype(int)
y3_binary = (y3_te == 2).astype(int)
auc3 = roc_auc_score(y3_binary, m3.predict_proba(X_te)[:, 2])
print(f'  {"3-Class Model":>35}: 3way-Acc={acc3:.4f}  Up-AUC={auc3:.4f}')
results['comparisons'].append({
    'group': 'target_definition',
    'label': 'three_class_direction',
    'feature_count': len(feature_cols_base),
    'direction_auc': round(float(auc3), 4),
    'direction_accuracy': round(float(acc3), 4),
    'three_class_accuracy': round(float(acc3), 4),
    'up_auc': round(float(auc3), 4),
})

# ════════════════════════════════════════════════════════════════════
print('\n' + '=' * 70)
print('EXPERIMENT 7: ENSEMBLE (XGB + LGB AVERAGE)')
print('=' * 70)
xgb_proba = xgb.XGBClassifier(**{k:v for k,v in BASE_DIR_PARAMS.items() if k not in ('n_estimators', 'early_stopping_rounds')},
                               n_estimators=1500, early_stopping_rounds=80)
xgb_proba.fit(X_tr, y_dir_tr, eval_set=[(X_va, y_dir_va)], verbose=0)
xgb_p = xgb_proba.predict_proba(X_te)[:, 1]
lgb_p = lgb_dir.predict_proba(X_te)[:, 1]
ensemble_p = 0.5 * xgb_p + 0.5 * lgb_p
ens_auc = roc_auc_score(y_dir_te, ensemble_p)
ens_acc = accuracy_score(y_dir_te, (ensemble_p > 0.5).astype(int))
print(f'  {"XGB+LGB Ensemble Direction":>35}: AUC={ens_auc:.4f}  Acc={ens_acc:.4f}')

# Vol ensemble
xgb_vp = xgb.XGBRegressor(**{k:v for k,v in BASE_VOL_PARAMS.items() if k not in ('n_estimators', 'early_stopping_rounds')},
                           n_estimators=1200, early_stopping_rounds=50)
xgb_vp.fit(X_tr, y_vol_tr, eval_set=[(X_va, y_vol_va)], verbose=0)
xvp = xgb_vp.predict(X_te)
lvp = lgb_vol.predict(X_te)
evp = 0.5 * xvp + 0.5 * lvp
emae = mean_absolute_error(y_vol_te, evp)
er2 = r2_score(y_vol_te, evp)
ecorr = np.corrcoef(evp, y_vol_te)[0,1]
print(f'  {"XGB+LGB Ensemble Volatility":>35}: MAE={emae:.6f}  R2={er2:.4f}  Corr={ecorr:.4f}')
results['comparisons'].append({
    'group': 'ensemble',
    'label': 'xgb_lgb_ensemble',
    'feature_count': len(feature_cols_base),
    'direction_auc': round(float(ens_auc), 4),
    'direction_accuracy': round(float(ens_acc), 4),
    'volatility_mae': round(float(emae), 6),
    'volatility_r2': round(float(er2), 4),
    'volatility_corr': round(float(ecorr), 4),
})

# ════════════════════════════════════════════════════════════════════
print('\n' + '=' * 70)
print('EXPERIMENT 8: QUANTILE VOLATILITY (better tail risk)')
print('=' * 70)
# Train a model for the 75th percentile of volatility instead of mean
from sklearn.metrics import mean_pinball_loss
p_qr = BASE_VOL_PARAMS.copy()
p_qr['objective'] = 'reg:quantileerror'
p_qr['quantile_alpha'] = 0.75
m_qr = xgb.XGBRegressor(**{k:v for k,v in p_qr.items() if k not in ('n_estimators', 'early_stopping_rounds')},
                         n_estimators=800, early_stopping_rounds=50)
m_qr.fit(X_tr, y_vol_tr, eval_set=[(X_va, y_vol_va)], verbose=0)
qr_pred = m_qr.predict(X_te)
qr_mae = mean_absolute_error(y_vol_te, qr_pred)
qr_coverage = np.mean(y_vol_te <= qr_pred)
qr_pinball = mean_pinball_loss(y_vol_te, qr_pred, alpha=0.75)
print(f'  {"Quantile(75) Volatility":>35}: MAE={qr_mae:.6f}  Coverage={qr_coverage:.4f}')
print(f'  {"(baseline coverage)":>35}: {np.mean(y_vol_te <= xvp):.4f}')
results['comparisons'].append({
    'group': 'volatility_objective',
    'label': 'quantile_volatility_q75',
    'feature_count': len(feature_cols_base),
    'volatility_mae': round(float(qr_mae), 6),
    'volatility_pinball_loss': round(float(qr_pinball), 6),
    'volatility_coverage': round(float(qr_coverage), 4),
    'baseline_coverage': round(float(np.mean(y_vol_te <= xvp)), 4),
})

print('\n' + '=' * 70)
print('SUMMARY OF FINDINGS')
print('=' * 70)

direction_candidates = [results['baseline']] + [
    row for row in results['comparisons'] if row.get('direction_auc') is not None
]
volatility_candidates = [results['baseline']] + [
    row for row in results['comparisons'] if row.get('volatility_r2') is not None
]
best_direction = max(direction_candidates, key=lambda row: row.get('direction_auc', float('-inf')))
best_volatility = max(volatility_candidates, key=lambda row: row.get('volatility_r2', float('-inf')))
results['headline_findings'] = [
    f"Best direction AUC: {best_direction['label']} ({best_direction['direction_auc']:.4f})",
    f"Best volatility R2: {best_volatility['label']} ({best_volatility['volatility_r2']:.4f})",
    f"Quantile coverage at q=0.75: {qr_coverage:.4f}",
]

for finding in results['headline_findings']:
    print(f'  - {finding}')

output_path = Path(__file__).resolve().parent / 'experiments_results.json'
output_path.write_text(json.dumps(results, indent=2), encoding='utf-8')
print(f'  Saved experiment summary to: {output_path}')
