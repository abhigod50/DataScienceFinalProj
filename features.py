"""
ML Feature Engineering Pipeline for Market Making
===================================================
Computes ~60 features across multiple timeframes from OHLCV data.
Designed for predicting short-term price direction and volatility
to optimize market-making spread parameters.
"""

import numpy as np
import pandas as pd
import talib
from typing import Optional


def _normalize_utc_ns(series: pd.Series) -> pd.Series:
    return pd.to_datetime(series, utc=True, errors="coerce").astype("datetime64[ns, UTC]")


def _merge_higher_timeframe_features(feat: pd.DataFrame) -> pd.DataFrame:
    """Build and merge 15m / 1h features from 5m OHLCV via resampling."""
    if "date" not in feat.columns:
        return feat

    base = feat.copy()
    base["date"] = _normalize_utc_ns(base["date"])
    base = base.dropna(subset=["date"]).sort_values("date")
    if base.empty:
        return feat

    ohlcv = base[["date", "open", "high", "low", "close", "volume"]].set_index("date")

    def _resample_ohlcv(rule: str) -> pd.DataFrame:
        out = ohlcv.resample(rule).agg(
            {
                "open": "first",
                "high": "max",
                "low": "min",
                "close": "last",
                "volume": "sum",
            }
        )
        out = out.dropna(subset=["open", "high", "low", "close"])
        if out.empty:
            return out

        c = out["close"]
        h = out["high"]
        l = out["low"]
        v = out["volume"]
        out["ret_1"] = c.pct_change(1)
        out["ret_3"] = c.pct_change(3)
        out["rsi_14"] = talib.RSI(c, timeperiod=14)
        out["atr_pct"] = talib.ATR(h, l, c, timeperiod=14) / c
        out["ema_21"] = talib.EMA(c, timeperiod=21)
        out["close_vs_ema_21"] = (c - out["ema_21"]) / out["ema_21"]
        out["realized_vol_12"] = np.log(c / c.shift(1)).rolling(12).std()
        vol_sma = v.rolling(20).mean().replace(0, np.nan)
        out["volume_ratio"] = (v / vol_sma).replace([np.inf, -np.inf], np.nan)
        return out

    tf15 = _resample_ohlcv("15min")
    tf60 = _resample_ohlcv("1h")

    rename_15 = {
        "ret_1": "mtf_15m_ret_1",
        "ret_3": "mtf_15m_ret_3",
        "rsi_14": "mtf_15m_rsi_14",
        "atr_pct": "mtf_15m_atr_pct",
        "close_vs_ema_21": "mtf_15m_close_vs_ema_21",
        "realized_vol_12": "mtf_15m_realized_vol_12",
        "volume_ratio": "mtf_15m_volume_ratio",
    }
    rename_60 = {
        "ret_1": "mtf_1h_ret_1",
        "ret_3": "mtf_1h_ret_3",
        "rsi_14": "mtf_1h_rsi_14",
        "atr_pct": "mtf_1h_atr_pct",
        "close_vs_ema_21": "mtf_1h_close_vs_ema_21",
        "realized_vol_12": "mtf_1h_realized_vol_12",
        "volume_ratio": "mtf_1h_volume_ratio",
    }

    if not tf15.empty:
        tf15 = tf15.rename(columns=rename_15)
        keep_15 = [c for c in rename_15.values() if c in tf15.columns]
        base = pd.merge_asof(
            base,
            tf15[keep_15].sort_index(),
            left_on="date",
            right_index=True,
            direction="backward",
        )

    if not tf60.empty:
        tf60 = tf60.rename(columns=rename_60)
        keep_60 = [c for c in rename_60.values() if c in tf60.columns]
        base = pd.merge_asof(
            base,
            tf60[keep_60].sort_index(),
            left_on="date",
            right_index=True,
            direction="backward",
        )

    for col in [
        "mtf_15m_ret_1",
        "mtf_15m_ret_3",
        "mtf_15m_rsi_14",
        "mtf_15m_atr_pct",
        "mtf_15m_close_vs_ema_21",
        "mtf_15m_realized_vol_12",
        "mtf_15m_volume_ratio",
        "mtf_1h_ret_1",
        "mtf_1h_ret_3",
        "mtf_1h_rsi_14",
        "mtf_1h_atr_pct",
        "mtf_1h_close_vs_ema_21",
        "mtf_1h_realized_vol_12",
        "mtf_1h_volume_ratio",
    ]:
        if col in base.columns:
            base[col] = base[col].replace([np.inf, -np.inf], np.nan).ffill().fillna(0.0)

    return base


def _compute_vpin_proxy(
    volume: pd.Series,
    buy_volume_pct: pd.Series,
    *,
    bucket_size_mult: float = 8.0,
    ema_span: int = 24,
) -> tuple[pd.Series, pd.Series]:
    """Approximate VPIN from candle-level buy/sell volume proxies.

    We do not have trade-by-trade classification, so the bar's close position
    inside the candle range is used as a buy-volume proxy. Volume is packed into
    equal-volume buckets and each completed bucket contributes an imbalance
    score |V_buy - V_sell| / V_total. The rolling EMA of those bucket scores is
    the usable VPIN feature for short-horizon toxicity prediction.
    """
    vol = pd.to_numeric(volume, errors="coerce").fillna(0.0).clip(lower=0.0).to_numpy(dtype=float)
    buy_pct = pd.to_numeric(buy_volume_pct, errors="coerce").fillna(0.5).clip(0.0, 1.0).to_numpy(dtype=float)
    nonzero = vol[vol > 0]
    if len(nonzero) == 0:
        zeros = pd.Series(np.zeros(len(vol), dtype=float), index=volume.index)
        return zeros, zeros

    bucket_volume = max(float(np.nanmedian(nonzero)) * bucket_size_mult, 1.0)
    bucket_scores = np.full(len(vol), np.nan, dtype=float)
    current_total = 0.0
    current_buy = 0.0
    current_sell = 0.0
    last_score = 0.0

    for idx in range(len(vol)):
        remaining_total = vol[idx]
        if remaining_total <= 0:
            bucket_scores[idx] = last_score
            continue
        remaining_buy = remaining_total * buy_pct[idx]
        remaining_sell = remaining_total - remaining_buy

        while remaining_total > 1e-12:
            take = min(bucket_volume - current_total, remaining_total)
            ratio = take / remaining_total if remaining_total > 0 else 0.0
            buy_take = remaining_buy * ratio
            sell_take = remaining_sell * ratio
            current_total += take
            current_buy += buy_take
            current_sell += sell_take
            remaining_total -= take
            remaining_buy -= buy_take
            remaining_sell -= sell_take

            if current_total >= bucket_volume - 1e-9:
                denom = max(current_buy + current_sell, 1e-9)
                last_score = abs(current_buy - current_sell) / denom
                current_total = 0.0
                current_buy = 0.0
                current_sell = 0.0
        bucket_scores[idx] = last_score

    bucket_series = pd.Series(bucket_scores, index=volume.index).ffill().fillna(0.0)
    return bucket_series, bucket_series.ewm(span=ema_span, adjust=False).mean().fillna(0.0)


def compute_features(
    df: pd.DataFrame,
    btc_df: Optional[pd.DataFrame] = None,
    sol_df: Optional[pd.DataFrame] = None,
    execution_df: Optional[pd.DataFrame] = None,
    orderbook_df: Optional[pd.DataFrame] = None,
    binance_df: Optional[pd.DataFrame] = None,
) -> pd.DataFrame:
    """
    Compute full feature set from a single-pair OHLCV DataFrame.

    Parameters
    ----------
    df : pd.DataFrame with columns [date, open, high, low, close, volume]
    btc_df : optional BTC OHLCV for cross-asset features
    sol_df : optional SOL OHLCV for cross-asset features
    execution_df : optional execution-learning features from Hummingbot DB
    orderbook_df : optional L1/L2 order book features from live snapshots

    Returns
    -------
    pd.DataFrame with original columns + computed features (NaN rows dropped)
    """
    feat = df.copy()
    if "date" in feat.columns:
        feat["date"] = _normalize_utc_ns(feat["date"])
    c, h, l, o, v = feat["close"], feat["high"], feat["low"], feat["open"], feat["volume"]

    # ── Price action ────────────────────────────────────────────────
    feat["returns_1"] = c.pct_change(1)
    feat["returns_3"] = c.pct_change(3)
    feat["returns_6"] = c.pct_change(6)
    feat["returns_12"] = c.pct_change(12)
    feat["log_return"] = np.log(c / c.shift(1))

    # Body and wick ratios (proxy for order-flow pressure)
    candle_range = h - l
    safe_range = candle_range.replace(0, np.nan)
    feat["body_ratio"] = ((c - o).abs() / safe_range).fillna(0)
    feat["upper_wick_ratio"] = ((h - pd.concat([c, o], axis=1).max(axis=1)) / safe_range).fillna(0)
    feat["lower_wick_ratio"] = ((pd.concat([c, o], axis=1).min(axis=1) - l) / safe_range).fillna(0)

    # ── Trend indicators ────────────────────────────────────────────
    for period in [8, 21, 50, 200]:
        feat[f"ema_{period}"] = talib.EMA(c, timeperiod=period)
        feat[f"close_vs_ema_{period}"] = (c - feat[f"ema_{period}"]) / feat[f"ema_{period}"]

    feat["ema_cross_8_21"] = feat["ema_8"] - feat["ema_21"]
    feat["ema_cross_21_50"] = feat["ema_21"] - feat["ema_50"]

    # MACD
    feat["macd"], feat["macd_signal"], feat["macd_hist"] = talib.MACD(c, 12, 26, 9)

    # ADX - trend strength
    feat["adx"] = talib.ADX(h, l, c, timeperiod=14)
    feat["plus_di"] = talib.PLUS_DI(h, l, c, timeperiod=14)
    feat["minus_di"] = talib.MINUS_DI(h, l, c, timeperiod=14)
    feat["di_diff"] = feat["plus_di"] - feat["minus_di"]

    # ── Momentum / oscillators ──────────────────────────────────────
    feat["rsi_14"] = talib.RSI(c, timeperiod=14)
    feat["rsi_6"] = talib.RSI(c, timeperiod=6)
    feat["stoch_k"], feat["stoch_d"] = talib.STOCH(h, l, c, 14, 3, 0, 3, 0)
    feat["williams_r"] = talib.WILLR(h, l, c, timeperiod=14)
    feat["cci"] = talib.CCI(h, l, c, timeperiod=14)
    feat["mfi"] = talib.MFI(h, l, c, v, timeperiod=14)
    feat["roc_6"] = talib.ROC(c, timeperiod=6)
    feat["roc_12"] = talib.ROC(c, timeperiod=12)

    # ── Volatility ──────────────────────────────────────────────────
    feat["atr_14"] = talib.ATR(h, l, c, timeperiod=14)
    feat["atr_pct"] = feat["atr_14"] / c  # normalized ATR
    feat["natr_14"] = talib.NATR(h, l, c, timeperiod=14)

    # Bollinger Bands
    feat["bb_upper"], feat["bb_mid"], feat["bb_lower"] = talib.BBANDS(c, 20, 2, 2)
    bb_width = feat["bb_upper"] - feat["bb_lower"]
    bb_width = bb_width.replace(0, np.nan)
    feat["bb_width_pct"] = bb_width / feat["bb_mid"]
    feat["bb_position"] = ((c - feat["bb_lower"]) / bb_width).fillna(0.5)

    # Realized volatility (rolling std of log returns)
    feat["realized_vol_12"] = feat["log_return"].rolling(12).std()
    feat["realized_vol_24"] = feat["log_return"].rolling(24).std()
    feat["realized_vol_72"] = feat["log_return"].rolling(72).std()
    feat["vol_ratio_12_72"] = feat["realized_vol_12"] / feat["realized_vol_72"]

    # ── Volume features ─────────────────────────────────────────────
    feat["volume_sma_20"] = v.rolling(20).mean()
    feat["volume_ratio"] = v / feat["volume_sma_20"]
    feat["obv"] = talib.OBV(c, v)
    feat["obv_slope"] = feat["obv"].pct_change(6)

    # Buy/sell volume proxy (using close position in bar)
    close_position = ((c - l) / safe_range).fillna(0.5)  # 0.5 = neutral when range is 0
    feat["buy_volume_pct"] = close_position
    feat["buy_volume_pct_sma"] = feat["buy_volume_pct"].rolling(12).mean()
    feat["vpin_bucket_imbalance"] , feat["vpin_ema_24"] = _compute_vpin_proxy(v, feat["buy_volume_pct"])
    feat["vpin_ema_96"] = feat["vpin_bucket_imbalance"].ewm(span=96, adjust=False).mean().fillna(0.0)
    feat["vpin_pressure_delta"] = (feat["vpin_ema_24"] - feat["vpin_ema_96"]).fillna(0.0)

    # VWAP proxy (rolling)
    typical_price = (h + l + c) / 3
    cumulative_tp_vol = (typical_price * v).rolling(24).sum()
    cumulative_vol = v.rolling(24).sum()
    cumulative_vol = cumulative_vol.replace(0, np.nan)
    feat["vwap_24"] = cumulative_tp_vol / cumulative_vol
    feat["close_vs_vwap"] = ((c - feat["vwap_24"]) / feat["vwap_24"]).fillna(0)

    # ── Microstructure proxies ──────────────────────────────────────
    # Amihud illiquidity (|return| / dollar volume) - higher = less liquid
    dollar_vol = c * v
    dollar_vol = dollar_vol.replace(0, np.nan)
    feat["amihud"] = (feat["returns_1"].abs() / dollar_vol).fillna(0)
    feat["amihud_sma"] = feat["amihud"].rolling(24).mean()

    # Kyle's lambda proxy (price impact per unit volume)
    feat["kyle_lambda"] = (feat["returns_1"].abs().rolling(12).sum() / v.rolling(12).sum().replace(0, np.nan)).fillna(0)

    # ── Multi-timeframe features (coarse 15m/1h proxies from 5m bars) ──────
    feat["close_15m_return"] = c.pct_change(3)  # 3 x 5m = 15m
    feat["close_1h_return"] = c.pct_change(12)  # 12 x 5m = 1h
    feat["high_1h"] = h.rolling(12).max()
    feat["low_1h"] = l.rolling(12).min()
    feat["range_1h_pct"] = (feat["high_1h"] - feat["low_1h"]) / c

    # True higher-timeframe features from resampled bars.
    feat = _merge_higher_timeframe_features(feat)

    # ── Mean-reversion & momentum signals ───────────────────────────
    # Z-score of returns (mean-reversion detector)
    ret_mean = feat["returns_1"].rolling(48).mean()
    ret_std = feat["returns_1"].rolling(48).std().replace(0, np.nan)
    feat["return_zscore"] = ((feat["returns_1"] - ret_mean) / ret_std).fillna(0)

    # Price distance from recent high/low (support/resistance proxy)
    feat["dist_from_12h_high"] = (c - h.rolling(144).max()) / c
    feat["dist_from_12h_low"] = (c - l.rolling(144).min()) / c

    # Consecutive up/down candles
    up_candle = (c > o).astype(int)
    feat["consecutive_up"] = up_candle.groupby((up_candle != up_candle.shift()).cumsum()).cumcount() * up_candle
    feat["consecutive_down"] = (1 - up_candle).groupby(((1 - up_candle) != (1 - up_candle).shift()).cumsum()).cumcount() * (1 - up_candle)

    # Volume-weighted momentum
    feat["vw_momentum_6"] = ((c.pct_change(1) * v).rolling(6).sum() / v.rolling(6).sum().replace(0, np.nan)).fillna(0)
    feat["vw_momentum_12"] = ((c.pct_change(1) * v).rolling(12).sum() / v.rolling(12).sum().replace(0, np.nan)).fillna(0)

    # Volatility regime (ratio of short to long vol)
    feat["vol_regime"] = (feat["realized_vol_12"] / feat["realized_vol_72"].replace(0, np.nan)).fillna(1.0)

    # RSI divergence (price making new high but RSI not)
    feat["rsi_momentum"] = feat["rsi_14"].diff(6)

    # ── Risk-adjusted momentum ───────────────────────────────────────
    # Rolling Sharpe (24 candles = 2h): captures sustained risk-adjusted momentum
    _ret = feat["log_return"]
    _roll_mean_24 = _ret.rolling(24).mean()
    _roll_std_24 = _ret.rolling(24).std().replace(0, np.nan)
    feat["rolling_sharpe_24"] = (_roll_mean_24 / _roll_std_24).fillna(0)

    # Rolling Sortino (24 candles): only penalises downside deviation
    _downside_ret = _ret.clip(upper=0)
    _roll_down_std_24 = _downside_ret.rolling(24).std().replace(0, np.nan)
    feat["rolling_sortino_24"] = (_roll_mean_24 / _roll_down_std_24).fillna(0)

    # Hurst-exponent proxy via variance-ratio (>0.5 = trending, <0.5 = mean-reverting)
    # VR(k) = Var(k-period returns) / (k * Var(1-period returns))
    _var1 = _ret.rolling(24).var().replace(0, np.nan)
    _var4 = _ret.rolling(4).sum().rolling(21).var().replace(0, np.nan)
    feat["variance_ratio_4"] = (_var4 / (4 * _var1)).fillna(1.0)

    # ── Cross-asset features (BTC as leader) ────────────────────────
    if btc_df is not None:
        btc_df = btc_df.copy()
        btc_df["date"] = _normalize_utc_ns(btc_df["date"])
        btc = btc_df.set_index("date")["close"].reindex(feat["date"]).ffill().values
        btc_s = pd.Series(btc, index=df.index)
        feat["btc_return_1"] = btc_s.pct_change(1, fill_method=None)
        feat["btc_return_6"] = btc_s.pct_change(6, fill_method=None)
        feat["btc_corr_24"] = feat["returns_1"].rolling(24).corr(feat["btc_return_1"])
        # BTC leads: lagged BTC returns as features
        feat["btc_return_lag1"] = feat["btc_return_1"].shift(1)
        feat["btc_return_lag3"] = feat["btc_return_1"].shift(3)

    # ── Cross-exchange: Binance as leading indicator for KuCoin ─────
    # Binance has 10-100× the liquidity of KuCoin; its prices lead KuCoin.
    # The Coinbase/Binance US premium is a strong short-horizon mean-reversion signal.
    if binance_df is not None:
        try:
            binance_df = binance_df.copy()
            binance_df["date"] = _normalize_utc_ns(binance_df["date"])
            bnc = binance_df.set_index("date")["close"].reindex(feat["date"]).ffill().values
            bnc_s = pd.Series(bnc, index=df.index)
            bnc_safe = bnc_s.replace(0, np.nan)
            # Premium: positive → KuCoin at premium → tends to revert down
            feat["coinbase_premium_pct"] = (c / bnc_safe - 1).fillna(0)
            _prem_mean = feat["coinbase_premium_pct"].rolling(144).mean()
            _prem_std = feat["coinbase_premium_pct"].rolling(144).std().replace(0, np.nan)
            feat["coinbase_premium_zscore"] = ((feat["coinbase_premium_pct"] - _prem_mean) / _prem_std).fillna(0)
            # Binance lagged returns are a leading indicator
            bnc_ret = bnc_s.pct_change(1, fill_method=None)
            feat["coinbase_return_1"] = bnc_ret
            feat["coinbase_return_lag1"] = bnc_ret.shift(1)
            feat["coinbase_return_lag2"] = bnc_ret.shift(2)
            # Signed cross-exchange momentum: are both exchanges moving the same way?
            feat["coinbase_eth_co_move"] = (feat["returns_1"] * feat["coinbase_return_1"]).rolling(12).mean().fillna(0)
        except Exception:
            pass

    # Guarantee cross-exchange columns exist even when reference feed is unavailable.
    for col in [
        "coinbase_premium_pct",
        "coinbase_premium_zscore",
        "coinbase_return_1",
        "coinbase_return_lag1",
        "coinbase_return_lag2",
        "coinbase_eth_co_move",
    ]:
        if col not in feat.columns:
            feat[col] = 0.0

    # ── Cross-asset features (SOL as alt-coin sentiment proxy) ──────
    if sol_df is not None:
        sol_df = sol_df.copy()
        sol_df["date"] = _normalize_utc_ns(sol_df["date"])
        sol = sol_df.set_index("date")["close"].reindex(feat["date"]).ffill().values
        sol_s = pd.Series(sol, index=df.index)
        feat["sol_return_1"] = sol_s.pct_change(1, fill_method=None)
        feat["sol_return_6"] = sol_s.pct_change(6, fill_method=None)
        feat["sol_corr_24"] = feat["returns_1"].rolling(24).corr(feat["sol_return_1"])
        # SOL leads: lagged SOL returns as features
        feat["sol_return_lag1"] = feat["sol_return_1"].shift(1)
        feat["sol_return_lag3"] = feat["sol_return_1"].shift(3)
        # ETH-SOL relative strength: positive = ETH outperforming SOL
        feat["eth_sol_relative_6"] = feat["returns_6"] - feat["sol_return_6"]

    # ── Time features ───────────────────────────────────────────────
    if "date" in feat.columns:
        dt = _normalize_utc_ns(feat["date"])
        feat["hour"] = dt.dt.hour
        feat["hour_sin"] = np.sin(2 * np.pi * feat["hour"] / 24)
        feat["hour_cos"] = np.cos(2 * np.pi * feat["hour"] / 24)
        feat["day_of_week"] = dt.dt.dayofweek
        feat["is_weekend"] = (feat["day_of_week"] >= 5).astype(int)

    # Merge live execution-learning features from the Hummingbot paper-trading DB.
    if execution_df is not None and not execution_df.empty and "date" in feat.columns:
        exec_feat = execution_df.copy()
        exec_feat["date"] = _normalize_utc_ns(exec_feat["date"])
        exec_feat = exec_feat.dropna(subset=["date"]).sort_values("date").reset_index(drop=True)
        feat["date"] = _normalize_utc_ns(feat["date"])
        feat = pd.merge_asof(
            feat.sort_values("date").reset_index(drop=True),
            exec_feat,
            on="date",
            direction="backward",
        )
        exec_cols = [col for col in exec_feat.columns if col != "date"]
        for col in exec_cols:
            feat[col] = feat[col].fillna(0.0)

    # Merge L1/L2 order book features from live snapshots.
    if orderbook_df is not None and not orderbook_df.empty and "date" in feat.columns:
        ob_feat = orderbook_df.copy()
        ob_feat["date"] = _normalize_utc_ns(ob_feat["date"])
        ob_feat = ob_feat.dropna(subset=["date"]).sort_values("date").reset_index(drop=True)
        feat["date"] = _normalize_utc_ns(feat["date"])
        feat = pd.merge_asof(
            feat.sort_values("date").reset_index(drop=True),
            ob_feat,
            on="date",
            direction="backward",
        )
        ob_cols = [col for col in ob_feat.columns if col != "date"]
        for col in ob_cols:
            feat[col] = feat[col].fillna(0.0)

    if "ob_weighted_mid_offset_bps" in feat.columns:
        microprice = pd.to_numeric(feat["ob_weighted_mid_offset_bps"], errors="coerce").replace([np.inf, -np.inf], np.nan).fillna(0.0)
        for lag in range(1, 6):
            feat[f"microprice_offset_lag{lag}"] = microprice.shift(lag).fillna(0.0)
        feat["microprice_offset_ema_3"] = microprice.ewm(span=3, adjust=False).mean().fillna(0.0)
        feat["microprice_offset_slope_3"] = microprice.diff().rolling(3).mean().fillna(0.0)
    else:
        for lag in range(1, 6):
            feat[f"microprice_offset_lag{lag}"] = 0.0
        feat["microprice_offset_ema_3"] = 0.0
        feat["microprice_offset_slope_3"] = 0.0

    # Drop intermediate EMA/BB columns that are prices (keep ratios)
    drop_cols = [c for c in feat.columns if c.startswith("ema_") and not c.startswith("ema_cross")
                 and "vs" not in c]
    drop_cols += ["bb_upper", "bb_mid", "bb_lower", "volume_sma_20", "obv",
                  "vwap_24", "high_1h", "low_1h"]
    feat.drop(columns=[c for c in drop_cols if c in feat.columns], inplace=True)

    return feat


def compute_labels(df: pd.DataFrame, horizon: int = 6, add_multi_horizon: bool = True) -> pd.DataFrame:
    """
    Compute prediction targets for market-making ML.

    Parameters
    ----------
    df : DataFrame with 'close', 'high', 'low' columns
    horizon : primary horizon in candles (6 × 5m = 30 min)
    add_multi_horizon : if True, also compute 5-min (h=1) and 15-min (h=3) labels.
        Short-horizon labels directly capture adverse selection risk — the key
        quantity a market maker needs to predict.

    Returns
    -------
    DataFrame with added columns:
        - direction / future_return at the primary horizon
        - direction_1 / future_return_1 (5-min, h=1)
        - direction_3 / future_return_3 (15-min, h=3)
        - future_volatility, future_max_up, future_max_down at primary horizon
    """
    out = df.copy()
    c = out["close"]

    out["future_return"] = c.shift(-horizon) / c - 1
    out["direction"] = (out["future_return"] > 0).astype(int)

    # Rolling max/min over forward window
    future_high = out["high"].shift(-1).rolling(horizon).max().shift(-(horizon - 1))
    future_low = out["low"].shift(-1).rolling(horizon).min().shift(-(horizon - 1))

    out["future_volatility"] = (future_high - future_low) / c
    out["future_max_up"] = (future_high - c) / c
    out["future_max_down"] = (c - future_low) / c

    if add_multi_horizon:
        # 5-min (1-candle) labels — direct adverse selection: "will price hurt a
        # fill placed right now within the next 5 minutes?"
        out["future_return_1"] = c.shift(-1) / c - 1
        out["direction_1"] = (out["future_return_1"] > 0).astype(int)
        # 15-min (3-candle) labels — medium-horizon context for spread skew
        out["future_return_3"] = c.shift(-3) / c - 1
        out["direction_3"] = (out["future_return_3"] > 0).astype(int)

    return out


def get_feature_columns(df: pd.DataFrame) -> list:
    """Return the list of feature column names (excluding targets and metadata)."""
    exclude = {
        "date", "open", "high", "low", "close", "volume",
        # Primary-horizon labels
        "direction", "future_return", "future_volatility",
        "future_max_up", "future_max_down",
        # Multi-horizon labels (added by compute_labels with add_multi_horizon=True)
        "direction_1", "future_return_1",
        "direction_3", "future_return_3",
        # Time metadata (encoded as sin/cos but raw not useful as float)
        "hour", "day_of_week",
    }
    return [c for c in df.columns if c not in exclude]
