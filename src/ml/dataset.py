"""
Dataset construction for the ML learning core (daily horizon).

Anti-lookahead contract (enforced by tests/test_ml_dataset.py):
  * every feature at date T is computed from data with timestamps <= T
  * the label at date T uses ONLY open[T+1] and open[T+1+HORIZON] — the
    executable next-open entry/exit — net of round-trip costs
  * therefore a training window whose last feature date is D has labels
    reaching to D+1+HORIZON; walk-forward windows must purge accordingly

Feature groups (all trailing): momentum, volatility, mean-reversion,
volume, cross-asset (vs BTC), statistical, calendar. Sentiment is excluded
(no 36-month history exists yet — see ML_PLAN.md §4).
"""

import logging
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

HORIZON = 5                 # prediction target: 5 trading days
DEAD_ZONE = 0.01            # |net| <= 1% -> FLAT (noise, not signal)
ROUND_TRIP_COST = 0.003     # 0.1% fee + 0.05% slip, both sides (baseline)

LABEL_DOWN, LABEL_FLAT, LABEL_UP = 0, 1, 2
LABEL_NAMES = {LABEL_DOWN: 'DOWN', LABEL_FLAT: 'FLAT', LABEL_UP: 'UP'}

FEATURE_GROUPS = {
    'momentum': ['ret_1', 'ret_3', 'ret_5', 'ret_10', 'ret_20', 'ret_60',
                 'ret_90'],
    'volatility': ['rv_10', 'rv_20', 'rv_60', 'atr_pct', 'dd_from_high_90',
                   'vol_of_vol'],
    'mean_reversion': ['rsi_14', 'dist_sma50', 'dist_sma200', 'boll_pos'],
    'volume': ['vol_z20', 'vol_trend', 'obv_slope'],
    'cross_asset': ['btc_ret_5', 'btc_ret_20', 'btc_rv_20',
                    'rel_btc_5', 'rel_btc_20'],
    'statistical': ['skew_20', 'kurt_20', 'autocorr_5', 'up_streak',
                    'hurst_50'],
    'calendar': ['day_of_week', 'month'],
}
FEATURE_COLUMNS = [f for group in FEATURE_GROUPS.values() for f in group]

# ── Evidence tiers: "stable is not the same as meaningful" ──────────────────
# Each feature has a characteristic timescale (roughly, its decorrelation
# horizon in days). The number of INDEPENDENT observations backing a feature
# over a data span is ~ span_days / timescale. A feature can be stably
# important across every retrain and still rest on 3 samples (e.g. `month`
# over 3 years) — importance stability tells you the model is consistent,
# the evidence count tells you whether the pattern deserves belief.
FEATURE_TIMESCALE_DAYS = {
    'ret_1': 1, 'ret_3': 3, 'ret_5': 5, 'ret_10': 10, 'ret_20': 20,
    'ret_60': 60, 'ret_90': 90,
    'rv_10': 10, 'rv_20': 20, 'rv_60': 60, 'atr_pct': 14,
    'dd_from_high_90': 90, 'vol_of_vol': 30,
    'rsi_14': 14, 'dist_sma50': 50, 'dist_sma200': 200, 'boll_pos': 20,
    'vol_z20': 20, 'vol_trend': 20, 'obv_slope': 20,
    'btc_ret_5': 5, 'btc_ret_20': 20, 'btc_rv_20': 20,
    'rel_btc_5': 5, 'rel_btc_20': 20,
    'skew_20': 20, 'kurt_20': 20, 'autocorr_5': 20, 'up_streak': 5,
    'hurst_50': 50,
    'day_of_week': 7,        # each weekday recurs weekly...
    'month': 365,            # ...but each month recurs once a YEAR
}


def evidence_count(feature: str, span_days: int) -> int:
    """~how many independent observations back this feature over span_days."""
    timescale = FEATURE_TIMESCALE_DAYS.get(feature, 20)
    return max(int(span_days / timescale), 0)


def evidence_tier(feature: str, span_days: int) -> str:
    """'well-supported' (>=100 obs) / 'moderate' (20-99) /
    'thin — treat as noise' (<20)."""
    n = evidence_count(feature, span_days)
    if n >= 100:
        return 'well-supported'
    if n >= 20:
        return 'moderate'
    return 'thin — treat as noise'


def _rolling_slope(s: pd.Series, window: int) -> pd.Series:
    x = np.arange(window, dtype=float)
    x_center = x - x.mean()
    denom = (x_center ** 2).sum()

    def slope(w):
        return float(np.dot(x_center, w) / denom)

    return s.rolling(window).apply(slope, raw=True)


def _hurst(prices: np.ndarray) -> float:
    """Rescaled-range Hurst exponent on a trailing window (compact port of
    AdvancedFeatureExtractor._calculate_hurst_exponent)."""
    if len(prices) < 20 or np.any(prices <= 0):
        return 0.5
    rets = np.diff(np.log(prices))
    if rets.std() == 0:
        return 0.5
    lags = range(2, min(len(rets) // 2, 20))
    tau = []
    for lag in lags:
        diffs = rets[lag:] - rets[:-lag]
        tau.append(np.sqrt(np.mean(diffs ** 2)))
    tau = np.asarray(tau)
    if np.any(tau <= 0):
        return 0.5
    slope = np.polyfit(np.log(list(lags)), np.log(tau), 1)[0]
    return float(np.clip(slope, 0.0, 1.0))


def build_features(df: pd.DataFrame,
                   btc_df: Optional[pd.DataFrame] = None) -> pd.DataFrame:
    """Feature matrix for one symbol's daily OHLCV frame. Every value at
    index T derives from rows <= T only (rolling/shift operations)."""
    from src.utils.indicators import RSI, ATR

    out = pd.DataFrame(index=df.index)
    close, high, low = df['close'], df['high'], df['low']
    volume = df['volume']
    rets = close.pct_change()

    # momentum
    for n in (1, 3, 5, 10, 20, 60, 90):
        out[f'ret_{n}'] = close.pct_change(n)

    # volatility
    for n in (10, 20, 60):
        out[f'rv_{n}'] = rets.rolling(n).std()
    out['atr_pct'] = pd.Series(ATR(high, low, close, 14),
                               index=df.index) / close
    out['dd_from_high_90'] = close / close.rolling(90).max() - 1
    out['vol_of_vol'] = out['rv_10'].rolling(20).std()

    # mean reversion
    out['rsi_14'] = pd.Series(RSI(close, 14), index=df.index)
    out['dist_sma50'] = close / close.rolling(50).mean() - 1
    out['dist_sma200'] = close / close.rolling(200).mean() - 1
    mid = close.rolling(20).mean()
    band = close.rolling(20).std(ddof=0)
    out['boll_pos'] = (close - mid) / (2 * band).replace(0, np.nan)

    # volume
    v_ma, v_sd = volume.rolling(20).mean(), volume.rolling(20).std()
    out['vol_z20'] = (volume - v_ma) / v_sd.replace(0, np.nan)
    out['vol_trend'] = volume.rolling(5).mean() / v_ma - 1
    obv = (np.sign(rets.fillna(0)) * volume).cumsum()
    out['obv_slope'] = _rolling_slope(obv, 20) / v_ma.replace(0, np.nan)

    # cross-asset (market factor + relative strength vs BTC)
    if btc_df is not None:
        btc_close = btc_df['close'].reindex(df.index).ffill()
        btc_rets = btc_close.pct_change()
        out['btc_ret_5'] = btc_close.pct_change(5)
        out['btc_ret_20'] = btc_close.pct_change(20)
        out['btc_rv_20'] = btc_rets.rolling(20).std()
        out['rel_btc_5'] = out['ret_5'] - out['btc_ret_5']
        out['rel_btc_20'] = out['ret_20'] - out['btc_ret_20']
    else:
        for col in FEATURE_GROUPS['cross_asset']:
            out[col] = 0.0

    # statistical
    out['skew_20'] = rets.rolling(20).skew()
    out['kurt_20'] = rets.rolling(20).kurt()
    out['autocorr_5'] = rets.rolling(20).apply(
        lambda w: pd.Series(w).autocorr(lag=5) if len(w) == 20 else np.nan,
        raw=False)
    up = (rets > 0).astype(int)
    out['up_streak'] = up.groupby((up != up.shift()).cumsum()).cumsum() * up
    out['hurst_50'] = close.rolling(50).apply(_hurst, raw=True)

    # calendar
    out['day_of_week'] = df.index.dayofweek
    out['month'] = df.index.month

    return out[FEATURE_COLUMNS]


def build_labels(df: pd.DataFrame, horizon: int = HORIZON,
                 cost: float = ROUND_TRIP_COST,
                 dead_zone: float = DEAD_ZONE) -> pd.DataFrame:
    """Executable after-fee labels. label[T] compares open[T+1] (earliest
    possible fill after deciding at T's close) to open[T+1+horizon]."""
    entry = df['open'].shift(-1)
    exit_ = df['open'].shift(-(1 + horizon))
    net = (exit_ / entry - 1) - cost

    label = pd.Series(LABEL_FLAT, index=df.index, dtype=int)
    label[net > dead_zone] = LABEL_UP
    label[net < -dead_zone] = LABEL_DOWN
    label[net.isna()] = -1        # unlabeled tail (future unknown)

    return pd.DataFrame({'label': label, 'net_fwd_return': net})


def assemble_panel(frames: Dict[str, pd.DataFrame]
                   ) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Pooled (symbol-agnostic) panel across symbols.

    Returns (X, meta) where meta has columns [date, symbol, label,
    net_fwd_return] aligned row-wise with X. Rows with NaN features or
    unlabeled tails are dropped."""
    btc_df = frames.get('BTC/USDT')
    x_parts, meta_parts = [], []
    for symbol, df in frames.items():
        feats = build_features(df, btc_df=btc_df)
        labels = build_labels(df)
        meta = pd.DataFrame({
            'date': df.index,
            'symbol': symbol,
            'label': labels['label'].values,
            'net_fwd_return': labels['net_fwd_return'].values,
        })
        x_parts.append(feats.reset_index(drop=True))
        meta_parts.append(meta.reset_index(drop=True))

    X = pd.concat(x_parts, ignore_index=True)
    meta = pd.concat(meta_parts, ignore_index=True)

    valid = X.notna().all(axis=1) & (meta['label'] >= 0)
    X, meta = X[valid].reset_index(drop=True), meta[valid].reset_index(drop=True)
    logger.info(f"panel: {len(X)} rows, {X.shape[1]} features, "
                f"labels UP {(meta['label'] == LABEL_UP).mean():.0%} / "
                f"FLAT {(meta['label'] == LABEL_FLAT).mean():.0%} / "
                f"DOWN {(meta['label'] == LABEL_DOWN).mean():.0%}")
    return X, meta
