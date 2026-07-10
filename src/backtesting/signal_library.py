"""
Published-strategy library: classic technical rules implemented EXACTLY as
documented — no parameter tweaking, no fitting. Each entry cites its source
and uses the canonical published parameters.

Every builder: daily OHLCV frame -> SignalArrays (long-only spot; decisions
at close T, execution next open through the identical Phase 2 simulator).
Entries/exits are level- or event-based exactly as the rule is written.

The library size (N) is counted and reported: testing N strategies
guarantees some look good by luck, and the study driver corrects for that
(Bonferroni + White's Reality Check). Adding a strategy here INCREASES the
correction burden — that is by design.
"""

import numpy as np
import pandas as pd

from src.backtesting.walkforward import SignalArrays
from src.utils import indicators as ta

RAIL_SIZE = 0.10


def _arrays(df, entry, exit_, stop_loss=None, size=RAIL_SIZE):
    n = len(df)
    return SignalArrays(
        entry=np.nan_to_num(entry).astype(bool),
        exit_=np.nan_to_num(exit_).astype(bool),
        confidence=np.where(np.nan_to_num(entry).astype(bool), 0.99, 0.0),
        size=np.full(n, size),
        stop_loss=stop_loss if stop_loss is not None else np.full(n, np.nan),
        take_profit=np.full(n, np.nan),
    )


def _cci(df, period=20):
    tp = (df['high'] + df['low'] + df['close']) / 3
    sma = tp.rolling(period).mean()
    mad = tp.rolling(period).apply(lambda w: np.abs(w - w.mean()).mean(),
                                   raw=True)
    return ((tp - sma) / (0.015 * mad)).to_numpy()


# ── Builders (name -> rule as documented) ────────────────────────────────────

def sma_cross_50_200(df, size=RAIL_SIZE):
    """Golden cross: long while SMA50 > SMA200. (Classic; e.g. Edwards & Magee)"""
    s50 = df['close'].rolling(50).mean()
    s200 = df['close'].rolling(200).mean()
    state = (s50 > s200).to_numpy()
    return _arrays(df, state, ~state, size=size)


def sma_cross_20_50(df, size=RAIL_SIZE):
    """Golden cross fast variant: long while SMA20 > SMA50."""
    s20 = df['close'].rolling(20).mean()
    s50 = df['close'].rolling(50).mean()
    state = (s20 > s50).to_numpy()
    return _arrays(df, state, ~state, size=size)


def faber_sma200(df, size=RAIL_SIZE):
    """Faber (2007) timing model, daily proxy: long while close > SMA200."""
    state = (df['close'] > df['close'].rolling(200).mean()).to_numpy()
    return _arrays(df, state, ~state, size=size)


def price_above_sma50(df, size=RAIL_SIZE):
    """Long while close > SMA50 (ubiquitous trend filter)."""
    state = (df['close'] > df['close'].rolling(50).mean()).to_numpy()
    return _arrays(df, state, ~state, size=size)


def ema_cross_12_26(df, size=RAIL_SIZE):
    """Long while EMA12 > EMA26 (Appel's MACD components as a cross system)."""
    e12 = pd.Series(ta.EMA(df['close'], 12), index=df.index)
    e26 = pd.Series(ta.EMA(df['close'], 26), index=df.index)
    state = (e12 > e26).to_numpy()
    return _arrays(df, state, ~state, size=size)


def macd_signal_cross(df, size=RAIL_SIZE):
    """Appel: long while MACD(12,26) > signal(9)."""
    macd, sig, _ = ta.MACD(df['close'])
    state = np.nan_to_num(macd > sig)
    return _arrays(df, state, ~state, size=size)


def macd_zero(df, size=RAIL_SIZE):
    """Long while MACD line > 0 (zero-line rule)."""
    macd, _, _ = ta.MACD(df['close'])
    state = np.nan_to_num(macd > 0)
    return _arrays(df, state, ~state, size=size)


def rsi14_reversion(df, size=RAIL_SIZE):
    """Wilder (1978): enter RSI(14) < 30 (oversold); exit RSI > 50."""
    rsi = ta.RSI(df['close'], 14)
    return _arrays(df, rsi < 30, rsi > 50, size=size)


def connors_rsi2(df, size=RAIL_SIZE):
    """Connors & Alvarez (2009): enter RSI(2) < 10 with close > SMA200;
    exit when close > SMA5."""
    rsi2 = ta.RSI(df['close'], 2)
    sma200 = df['close'].rolling(200).mean()
    sma5 = df['close'].rolling(5).mean()
    entry = (rsi2 < 10) & (df['close'] > sma200).to_numpy()
    exit_ = (df['close'] > sma5).to_numpy()
    return _arrays(df, entry, exit_, size=size)


def bollinger_reversion(df, size=RAIL_SIZE):
    """Bollinger (2001): enter close < lower band(20,2); exit at middle band."""
    upper, mid, lower = ta.BBANDS(df['close'], 20, 2, 2)
    close = df['close'].to_numpy()
    return _arrays(df, close < lower, close >= mid, size=size)


def bollinger_breakout(df, size=RAIL_SIZE):
    """Band breakout usage: enter close > upper band(20,2); exit < middle."""
    upper, mid, lower = ta.BBANDS(df['close'], 20, 2, 2)
    close = df['close'].to_numpy()
    return _arrays(df, close > upper, close < mid, size=size)


def _donchian(df, entry_n, exit_n, size):
    """Turtle rules (Faith, 'Way of the Turtle'): enter on entry_n-day-high
    breakout; exit on exit_n-day low; initial stop 2N (N = 20-day ATR)."""
    close = df['close']
    hi = close.rolling(entry_n).max().shift(1)
    lo = close.rolling(exit_n).min().shift(1)
    atr = ta.ATR(df['high'], df['low'], df['close'], 20)
    entry = (close > hi).to_numpy()
    exit_ = (close < lo).to_numpy()
    stop = close.to_numpy() - 2 * atr
    return _arrays(df, entry, exit_, stop_loss=stop, size=size)


def turtle_s1(df, size=RAIL_SIZE):
    return _donchian(df, 20, 10, size)


def turtle_s2(df, size=RAIL_SIZE):
    return _donchian(df, 55, 20, size)


def ichimoku_standard(df, size=RAIL_SIZE):
    """Hosoda's Ichimoku (9/26/52): long while close above the cloud and
    tenkan > kijun; exit when close falls below the cloud."""
    high, low = df['high'], df['low']
    tenkan = (high.rolling(9).max() + low.rolling(9).min()) / 2
    kijun = (high.rolling(26).max() + low.rolling(26).min()) / 2
    senkou_a = ((tenkan + kijun) / 2).shift(26)
    senkou_b = ((high.rolling(52).max() + low.rolling(52).min()) / 2).shift(26)
    cloud_top = pd.concat([senkou_a, senkou_b], axis=1).max(axis=1)
    cloud_bot = pd.concat([senkou_a, senkou_b], axis=1).min(axis=1)
    entry = ((df['close'] > cloud_top) & (tenkan > kijun)).to_numpy()
    exit_ = (df['close'] < cloud_bot).to_numpy()
    return _arrays(df, entry, exit_, size=size)


def adx_di_system(df, size=RAIL_SIZE):
    """Wilder (1978) directional system: long while +DI > -DI with ADX > 25;
    exit on the DI cross down."""
    plus = ta.PLUS_DI(df['high'], df['low'], df['close'], 14)
    minus = ta.MINUS_DI(df['high'], df['low'], df['close'], 14)
    adx = ta.ADX(df['high'], df['low'], df['close'], 14)
    entry = np.nan_to_num((plus > minus) & (adx > 25))
    exit_ = np.nan_to_num(plus < minus)
    return _arrays(df, entry, exit_, size=size)


def roc20_momentum(df, size=RAIL_SIZE):
    """Rate-of-change momentum: long while ROC(20) > 0."""
    roc = ta.ROC(df['close'], 20)
    state = np.nan_to_num(roc > 0)
    return _arrays(df, state, ~state.astype(bool), size=size)


def obv_trend(df, size=RAIL_SIZE):
    """Granville's OBV: long while OBV above its 20-day average."""
    obv = pd.Series(ta.OBV(df['close'], df['volume']), index=df.index)
    state = (obv > obv.rolling(20).mean()).to_numpy()
    return _arrays(df, state, ~state, size=size)


def stochastic_oversold(df, size=RAIL_SIZE):
    """Lane's stochastic (14,3,3): enter on %K crossing above %D while
    %K < 20; exit when %K > 80."""
    k, d = ta.STOCH(df['high'], df['low'], df['close'],
                    fastk_period=14, slowk_period=3, slowd_period=3)
    k_prev, d_prev = np.roll(k, 1), np.roll(d, 1)
    k_prev[0] = d_prev[0] = np.nan
    entry = np.nan_to_num((k_prev <= d_prev) & (k > d) & (k < 20))
    exit_ = np.nan_to_num(k > 80)
    return _arrays(df, entry, exit_, size=size)


def williams_r(df, size=RAIL_SIZE):
    """Williams %R(14): enter < -80 (oversold); exit > -20."""
    wr = ta.WILLR(df['high'], df['low'], df['close'], 14)
    return _arrays(df, np.nan_to_num(wr < -80), np.nan_to_num(wr > -20),
                   size=size)


def cci_lambert(df, size=RAIL_SIZE):
    """Lambert (1980) original CCI usage: long while CCI(20) > +100."""
    cci = _cci(df, 20)
    return _arrays(df, np.nan_to_num(cci > 100), np.nan_to_num(cci <= 100),
                   size=size)


def cci_reversion(df, size=RAIL_SIZE):
    """CCI mean-reversion usage: enter CCI(20) < -100; exit CCI > 0."""
    cci = _cci(df, 20)
    return _arrays(df, np.nan_to_num(cci < -100), np.nan_to_num(cci > 0),
                   size=size)


def psar_trend(df, size=RAIL_SIZE):
    """Wilder's Parabolic SAR (0.02/0.2): long while close > SAR."""
    sar = ta.SAR(df['high'], df['low'])
    state = np.nan_to_num(df['close'].to_numpy() > sar)
    return _arrays(df, state, ~state.astype(bool), size=size)


def keltner_breakout(df, size=RAIL_SIZE):
    """Keltner channel (EMA20 ± 2×ATR10, Chester Keltner as modernized by
    Linda Raschke): enter close > upper channel; exit close < EMA20."""
    ema20 = pd.Series(ta.EMA(df['close'], 20), index=df.index)
    atr10 = pd.Series(ta.ATR(df['high'], df['low'], df['close'], 10),
                      index=df.index)
    upper = ema20 + 2 * atr10
    entry = (df['close'] > upper).to_numpy()
    exit_ = (df['close'] < ema20).to_numpy()
    return _arrays(df, entry, exit_, size=size)


def high_52w_momentum(df, size=RAIL_SIZE):
    """George & Hwang (2004) 52-week-high momentum, daily adaptation:
    enter when close within 5% of the 252-day high; exit below 90% of it."""
    hi252 = df['close'].rolling(252).max()
    entry = (df['close'] >= 0.95 * hi252).to_numpy()
    exit_ = (df['close'] < 0.90 * hi252).to_numpy()
    return _arrays(df, entry, exit_, size=size)


def aroon_25(df, size=RAIL_SIZE):
    """Chande's Aroon(25): enter AroonUp > 70 with AroonDown < 30;
    exit when AroonUp < 50."""
    down, up = ta.AROON(df['high'], df['low'], 25)
    entry = np.nan_to_num((up > 70) & (down < 30))
    exit_ = np.nan_to_num(up < 50)
    return _arrays(df, entry, exit_, size=size)


def _tsmom(lookback):
    def build(df, size=RAIL_SIZE):
        from scripts.run_daily_momentum import tsmom_signals
        return tsmom_signals(df, lookback, size)
    build.__doc__ = (f"Moskowitz-Ooi-Pedersen (2012) time-series momentum: "
                     f"long while trailing {lookback}d return > 0, weekly.")
    return build


# ── The library (N counted for the multiple-testing correction) ─────────────

LIBRARY = {
    'sma_cross_50_200': sma_cross_50_200,
    'sma_cross_20_50': sma_cross_20_50,
    'faber_sma200': faber_sma200,
    'price_above_sma50': price_above_sma50,
    'ema_cross_12_26': ema_cross_12_26,
    'macd_signal_cross': macd_signal_cross,
    'macd_zero': macd_zero,
    'rsi14_reversion': rsi14_reversion,
    'connors_rsi2': connors_rsi2,
    'bollinger_reversion': bollinger_reversion,
    'bollinger_breakout': bollinger_breakout,
    'turtle_s1_20_10': turtle_s1,
    'turtle_s2_55_20': turtle_s2,
    'ichimoku_standard': ichimoku_standard,
    'adx_di_system': adx_di_system,
    'roc20_momentum': roc20_momentum,
    'obv_trend': obv_trend,
    'stochastic_oversold': stochastic_oversold,
    'williams_r': williams_r,
    'cci_lambert_trend': cci_lambert,
    'cci_reversion': cci_reversion,
    'psar_trend': psar_trend,
    'keltner_breakout': keltner_breakout,
    'high_52w_momentum': high_52w_momentum,
    'aroon_25': aroon_25,
    'tsmom_20d': _tsmom(20),
    'tsmom_40d': _tsmom(40),
    'tsmom_60d': _tsmom(60),
    'tsmom_90d': _tsmom(90),
}

N_STRATEGIES = len(LIBRARY)


# ── Noise control: random strategies through the identical pipeline ─────────

def noise_strategy(seed: int):
    """A random strategy with trade frequency matched to the library's
    typical cadence (P(enter)=5%/day when flat, P(exit)=8%/day when long
    -> ~15-30 round trips/year, ~2-week holds). Pure luck by construction."""
    def build(df, size=RAIL_SIZE):
        rng = np.random.default_rng(seed)
        n = len(df)
        entry = rng.random(n) < 0.05
        exit_ = rng.random(n) < 0.08
        return _arrays(df, entry, exit_, size=size)
    return build
