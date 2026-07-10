"""
Technical indicators for AI Crypto Trading System.

Pure numpy/pandas implementations with TA-Lib-compatible signatures for every
function this codebase uses. Modules that want TA-Lib do:

    try:
        import talib
    except ImportError:
        from src.utils import indicators as talib

If the real TA-Lib C library is installed it wins; otherwise these fallbacks
keep the system fully functional (macOS dev boxes, slim Docker images).

All functions accept numpy arrays or pandas Series and return numpy arrays
padded with NaN over warmup periods, matching TA-Lib conventions.
"""

import numpy as np
import pandas as pd
from typing import Tuple

__all__ = [
    'SMA', 'EMA', 'RSI', 'MACD', 'ATR', 'BBANDS', 'STOCH', 'WILLR', 'ROC',
    'OBV', 'AD', 'ADOSC', 'ADX', 'PLUS_DI', 'MINUS_DI', 'SAR', 'AROON',
    'CDLDOJI', 'CDLHAMMER', 'CDLSHOOTINGSTAR', 'CDLENGULFING',
    'TechnicalIndicators',
]


def _series(x) -> pd.Series:
    if isinstance(x, pd.Series):
        return x.astype(float).reset_index(drop=True)
    return pd.Series(np.asarray(x, dtype=float))


def _wilder(s: pd.Series, period: int) -> pd.Series:
    """Wilder's smoothing (used by RSI, ATR, ADX). Equivalent to EMA with
    alpha = 1/period, seeded with the SMA of the first `period` values."""
    values = s.to_numpy(dtype=float)
    out = np.full_like(values, np.nan)
    if len(values) < period:
        return pd.Series(out)
    # find first window with no NaN
    start = 0
    while start + period <= len(values) and np.isnan(values[start:start + period]).any():
        start += 1
    if start + period > len(values):
        return pd.Series(out)
    out[start + period - 1] = values[start:start + period].mean()
    alpha = 1.0 / period
    for i in range(start + period, len(values)):
        out[i] = out[i - 1] + alpha * (values[i] - out[i - 1])
    return pd.Series(out)


# ── Overlap / momentum ──────────────────────────────────────────────────────

def SMA(close, timeperiod: int = 30) -> np.ndarray:
    return _series(close).rolling(timeperiod).mean().to_numpy()


def EMA(close, timeperiod: int = 30) -> np.ndarray:
    s = _series(close)
    sma_seed = s.rolling(timeperiod).mean()
    ema = s.ewm(span=timeperiod, adjust=False).mean()
    # TA-Lib seeds the EMA with an SMA: blend by replacing warmup with NaN
    ema[:timeperiod - 1] = np.nan
    ema.iloc[timeperiod - 1:timeperiod] = sma_seed.iloc[timeperiod - 1:timeperiod]
    return ema.to_numpy()


def RSI(close, timeperiod: int = 14) -> np.ndarray:
    s = _series(close)
    delta = s.diff()
    gain = delta.clip(lower=0)
    loss = (-delta).clip(lower=0)
    avg_gain = _wilder(gain.fillna(0), timeperiod)
    avg_loss = _wilder(loss.fillna(0), timeperiod)
    rs = avg_gain / avg_loss.replace(0, np.nan)
    rsi = 100 - (100 / (1 + rs))
    rsi[avg_loss == 0] = 100.0
    rsi[:timeperiod] = np.nan
    return rsi.to_numpy()


def MACD(close, fastperiod: int = 12, slowperiod: int = 26,
         signalperiod: int = 9) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    s = _series(close)
    fast = s.ewm(span=fastperiod, adjust=False).mean()
    slow = s.ewm(span=slowperiod, adjust=False).mean()
    macd = fast - slow
    signal = macd.ewm(span=signalperiod, adjust=False).mean()
    hist = macd - signal
    warmup = slowperiod + signalperiod - 2
    for arr in (macd, signal, hist):
        arr[:warmup] = np.nan
    return macd.to_numpy(), signal.to_numpy(), hist.to_numpy()


def ROC(close, timeperiod: int = 10) -> np.ndarray:
    s = _series(close)
    return (100 * (s / s.shift(timeperiod) - 1)).to_numpy()


def WILLR(high, low, close, timeperiod: int = 14) -> np.ndarray:
    h, l, c = _series(high), _series(low), _series(close)
    hh = h.rolling(timeperiod).max()
    ll = l.rolling(timeperiod).min()
    return (-100 * (hh - c) / (hh - ll).replace(0, np.nan)).to_numpy()


def STOCH(high, low, close, fastk_period: int = 5, slowk_period: int = 3,
          slowk_matype: int = 0, slowd_period: int = 3,
          slowd_matype: int = 0) -> Tuple[np.ndarray, np.ndarray]:
    h, l, c = _series(high), _series(low), _series(close)
    ll = l.rolling(fastk_period).min()
    hh = h.rolling(fastk_period).max()
    fastk = 100 * (c - ll) / (hh - ll).replace(0, np.nan)
    slowk = fastk.rolling(slowk_period).mean()
    slowd = slowk.rolling(slowd_period).mean()
    return slowk.to_numpy(), slowd.to_numpy()


# ── Volatility ──────────────────────────────────────────────────────────────

def _true_range(high, low, close) -> pd.Series:
    h, l, c = _series(high), _series(low), _series(close)
    prev_close = c.shift(1)
    return pd.concat([h - l, (h - prev_close).abs(), (l - prev_close).abs()],
                     axis=1).max(axis=1)


def ATR(high, low, close, timeperiod: int = 14) -> np.ndarray:
    tr = _true_range(high, low, close)
    return _wilder(tr.fillna(tr), timeperiod).to_numpy()


def BBANDS(close, timeperiod: int = 5, nbdevup: float = 2.0,
           nbdevdn: float = 2.0, matype: int = 0
           ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    s = _series(close)
    mid = s.rolling(timeperiod).mean()
    std = s.rolling(timeperiod).std(ddof=0)
    return ((mid + nbdevup * std).to_numpy(), mid.to_numpy(),
            (mid - nbdevdn * std).to_numpy())


# ── Volume ──────────────────────────────────────────────────────────────────

def OBV(close, volume) -> np.ndarray:
    c, v = _series(close), _series(volume)
    direction = np.sign(c.diff()).fillna(0)
    return (direction * v).cumsum().to_numpy()


def AD(high, low, close, volume) -> np.ndarray:
    h, l, c, v = _series(high), _series(low), _series(close), _series(volume)
    rng = (h - l).replace(0, np.nan)
    mfm = ((c - l) - (h - c)) / rng
    return (mfm.fillna(0) * v).cumsum().to_numpy()


def ADOSC(high, low, close, volume, fastperiod: int = 3,
          slowperiod: int = 10) -> np.ndarray:
    ad_line = pd.Series(AD(high, low, close, volume))
    fast = ad_line.ewm(span=fastperiod, adjust=False).mean()
    slow = ad_line.ewm(span=slowperiod, adjust=False).mean()
    return (fast - slow).to_numpy()


# ── Trend strength / direction ──────────────────────────────────────────────

def _directional_movement(high, low):
    h, l = _series(high), _series(low)
    up = h.diff()
    down = -l.diff()
    plus_dm = pd.Series(np.where((up > down) & (up > 0), up, 0.0))
    minus_dm = pd.Series(np.where((down > up) & (down > 0), down, 0.0))
    return plus_dm, minus_dm


def PLUS_DI(high, low, close, timeperiod: int = 14) -> np.ndarray:
    plus_dm, _ = _directional_movement(high, low)
    atr = pd.Series(ATR(high, low, close, timeperiod))
    return (100 * _wilder(plus_dm, timeperiod) / atr.replace(0, np.nan)).to_numpy()


def MINUS_DI(high, low, close, timeperiod: int = 14) -> np.ndarray:
    _, minus_dm = _directional_movement(high, low)
    atr = pd.Series(ATR(high, low, close, timeperiod))
    return (100 * _wilder(minus_dm, timeperiod) / atr.replace(0, np.nan)).to_numpy()


def ADX(high, low, close, timeperiod: int = 14) -> np.ndarray:
    plus_di = pd.Series(PLUS_DI(high, low, close, timeperiod))
    minus_di = pd.Series(MINUS_DI(high, low, close, timeperiod))
    dx = 100 * (plus_di - minus_di).abs() / (plus_di + minus_di).replace(0, np.nan)
    return _wilder(dx.fillna(0), timeperiod).to_numpy()


def AROON(high, low, timeperiod: int = 14) -> Tuple[np.ndarray, np.ndarray]:
    """Returns (aroondown, aroonup) — TA-Lib argument order."""
    h, l = _series(high), _series(low)
    # bars since the rolling max/min (0 = current bar)
    up = h.rolling(timeperiod + 1).apply(
        lambda w: 100 * (timeperiod - (timeperiod - np.argmax(w))) / timeperiod
        if not np.isnan(w).any() else np.nan, raw=True)
    down = l.rolling(timeperiod + 1).apply(
        lambda w: 100 * (timeperiod - (timeperiod - np.argmin(w))) / timeperiod
        if not np.isnan(w).any() else np.nan, raw=True)
    return down.to_numpy(), up.to_numpy()


def SAR(high, low, acceleration: float = 0.02, maximum: float = 0.2) -> np.ndarray:
    """Parabolic SAR (Wilder)."""
    h = _series(high).to_numpy()
    l = _series(low).to_numpy()
    n = len(h)
    sar = np.full(n, np.nan)
    if n < 2:
        return sar
    uptrend = h[1] + l[1] >= h[0] + l[0]
    ep = h[1] if uptrend else l[1]     # extreme point
    sar[1] = l[0] if uptrend else h[0]
    af = acceleration
    for i in range(2, n):
        sar[i] = sar[i - 1] + af * (ep - sar[i - 1])
        if uptrend:
            sar[i] = min(sar[i], l[i - 1], l[i - 2])
            if l[i] < sar[i]:          # reversal
                uptrend, sar[i], ep, af = False, ep, l[i], acceleration
            elif h[i] > ep:
                ep, af = h[i], min(af + acceleration, maximum)
        else:
            sar[i] = max(sar[i], h[i - 1], h[i - 2])
            if h[i] > sar[i]:          # reversal
                uptrend, sar[i], ep, af = True, ep, h[i], acceleration
            elif l[i] < ep:
                ep, af = l[i], min(af + acceleration, maximum)
    return sar


# ── Candlestick patterns (0 or ±100, TA-Lib convention) ─────────────────────

def _candle_parts(open_, high, low, close):
    o, h, l, c = _series(open_), _series(high), _series(low), _series(close)
    body = (c - o).abs()
    rng = (h - l).replace(0, np.nan)
    upper = h - pd.concat([o, c], axis=1).max(axis=1)
    lower = pd.concat([o, c], axis=1).min(axis=1) - l
    return o, h, l, c, body, rng, upper, lower


def CDLDOJI(open_, high, low, close) -> np.ndarray:
    o, h, l, c, body, rng, _, _ = _candle_parts(open_, high, low, close)
    return np.where((body / rng) < 0.1, 100, 0).astype(int)


def CDLHAMMER(open_, high, low, close) -> np.ndarray:
    o, h, l, c, body, rng, upper, lower = _candle_parts(open_, high, low, close)
    cond = (lower >= 2 * body) & (upper <= body) & ((body / rng) > 0.05)
    return np.where(cond.fillna(False), 100, 0).astype(int)


def CDLSHOOTINGSTAR(open_, high, low, close) -> np.ndarray:
    o, h, l, c, body, rng, upper, lower = _candle_parts(open_, high, low, close)
    cond = (upper >= 2 * body) & (lower <= body) & ((body / rng) > 0.05)
    return np.where(cond.fillna(False), -100, 0).astype(int)


def CDLENGULFING(open_, high, low, close) -> np.ndarray:
    o, h, l, c, *_ = _candle_parts(open_, high, low, close)
    prev_o, prev_c = o.shift(1), c.shift(1)
    bull = (prev_c < prev_o) & (c > o) & (c >= prev_o) & (o <= prev_c)
    bear = (prev_c > prev_o) & (c < o) & (c <= prev_o) & (o >= prev_c)
    out = np.zeros(len(o), dtype=int)
    out[bull.fillna(False).to_numpy()] = 100
    out[bear.fillna(False).to_numpy()] = -100
    return out


# ── DataFrame-level convenience API (used by strategies) ────────────────────

class TechnicalIndicators:
    """DataFrame-oriented indicator helpers over OHLCV columns
    (open, high, low, close, volume)."""

    @staticmethod
    def sma(df: pd.DataFrame, period: int, column: str = 'close') -> pd.Series:
        return df[column].rolling(period).mean()

    @staticmethod
    def ema(df: pd.DataFrame, period: int, column: str = 'close') -> pd.Series:
        return pd.Series(EMA(df[column], period), index=df.index)

    @staticmethod
    def rsi(df: pd.DataFrame, period: int = 14, column: str = 'close') -> pd.Series:
        return pd.Series(RSI(df[column], period), index=df.index)

    @staticmethod
    def macd(df: pd.DataFrame, fast: int = 12, slow: int = 26,
             signal: int = 9, column: str = 'close') -> pd.DataFrame:
        m, s, h = MACD(df[column], fast, slow, signal)
        return pd.DataFrame({'macd': m, 'signal': s, 'hist': h}, index=df.index)

    @staticmethod
    def atr(df: pd.DataFrame, period: int = 14) -> pd.Series:
        return pd.Series(ATR(df['high'], df['low'], df['close'], period),
                         index=df.index)

    @staticmethod
    def bollinger(df: pd.DataFrame, period: int = 20,
                  num_std: float = 2.0) -> pd.DataFrame:
        u, m, l = BBANDS(df['close'], period, num_std, num_std)
        return pd.DataFrame({'upper': u, 'middle': m, 'lower': l}, index=df.index)

    @staticmethod
    def adx(df: pd.DataFrame, period: int = 14) -> pd.Series:
        return pd.Series(ADX(df['high'], df['low'], df['close'], period),
                         index=df.index)

    @staticmethod
    def volatility(df: pd.DataFrame, period: int = 20) -> pd.Series:
        return df['close'].pct_change().rolling(period).std()
