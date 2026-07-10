"""
Multi-timeframe (Elder triple-screen) strategy family — Part B of the
Fast Lab plan (docs/FASTLAB_PLAN.md). Long-only spot, zero tuning.

Source: Alexander Elder, "Trading for a Living" (1993), triple screen:
  Screen 1 (tide): higher-timeframe trend — EMA-26 slope rising, or
                   MACD-histogram rising (both documented by Elder).
  Screen 2 (wave): oscillator pullback against the tide on the trading TF —
                   Force Index(2) < 0, or Stochastic %K < 30.
  Screen 3 (ride): trailing buy-stop above the prior bar's high; initial
                   protective stop below the prior bar's low.
  Exit: the tide flips, or the oscillator reaches overbought.

Higher TF here: 4h (resampled from stored 1h bars). Entry TFs: 1m and 5m.
HTF gate values become available on the LTF ONLY after the 4h bar closes —
enforced by shifting the HTF stamp by the bar duration before projection
(lookahead-tested in tests/test_mtf_library.py).

Variant count N = 2 gates x 2 oscillators x 2 entry TFs = 8. Every variant
is counted toward the multiple-testing correction.
"""

import numpy as np
import pandas as pd

from src.backtesting.walkforward import SignalArrays
from src.utils import indicators as ta

RAIL_SIZE = 0.10


def resample_4h(h1: pd.DataFrame) -> pd.DataFrame:
    return pd.DataFrame({
        'open': h1['open'].resample('4h').first(),
        'high': h1['high'].resample('4h').max(),
        'low': h1['low'].resample('4h').min(),
        'close': h1['close'].resample('4h').last(),
        'volume': h1['volume'].resample('4h').sum(),
    }).dropna()


def project_htf(series: pd.Series, ltf_index: pd.DatetimeIndex,
                htf: str = '4h') -> pd.Series:
    """Project an HTF indicator onto an LTF index with availability delay:
    the value of the HTF bar stamped at open-time O becomes known only at
    O + duration (when the bar closes)."""
    duration = pd.Timedelta(htf)
    shifted = series.copy()
    shifted.index = shifted.index + duration
    return shifted.reindex(ltf_index, method='ffill')


# ── Screen 1 gates (on 4h) ───────────────────────────────────────────────────

def gate_ema26_slope(h4: pd.DataFrame) -> pd.Series:
    ema = pd.Series(ta.EMA(h4['close'], 26), index=h4.index)
    return ema > ema.shift(1)


def gate_macdh_rising(h4: pd.DataFrame) -> pd.Series:
    _, _, hist = ta.MACD(h4['close'])
    hist = pd.Series(hist, index=h4.index)
    return hist > hist.shift(1)


GATES = {'ema26_slope_4h': gate_ema26_slope,
         'macdh_rising_4h': gate_macdh_rising}


# ── Screen 2 oscillators (on the entry TF) ───────────────────────────────────

def osc_force_index2(df: pd.DataFrame):
    """Elder's Force Index smoothed with a 2-period EMA."""
    fi = df['volume'] * df['close'].diff()
    fi2 = fi.ewm(span=2, adjust=False).mean()
    return (fi2 < 0).to_numpy(), (fi2 > 0).to_numpy()   # pullback, overbought


def osc_stochastic(df: pd.DataFrame):
    k, _ = ta.STOCH(df['high'], df['low'], df['close'],
                    fastk_period=14, slowk_period=3, slowd_period=3)
    return np.nan_to_num(k < 30), np.nan_to_num(k > 70)


OSCILLATORS = {'force_index2': osc_force_index2,
               'stochastic': osc_stochastic}


# ── Triple screen assembly ───────────────────────────────────────────────────

def triple_screen(ltf: pd.DataFrame, h1: pd.DataFrame, gate_name: str,
                  osc_name: str, size: float = RAIL_SIZE) -> SignalArrays:
    h4 = resample_4h(h1)
    gate = project_htf(GATES[gate_name](h4).astype(float), ltf.index)
    gate = np.nan_to_num(gate.to_numpy()).astype(bool)

    pullback, overbought = OSCILLATORS[osc_name](ltf)
    pullback_prev = np.roll(pullback, 1)
    pullback_prev[0] = False

    prev_high = ltf['high'].shift(1).to_numpy()
    prev_low = ltf['low'].shift(1).to_numpy()
    close = ltf['close'].to_numpy()

    # Screen 3: prior bar showed the pullback; this bar takes out its high,
    # with the 4h tide up.
    entry = gate & pullback_prev & np.nan_to_num(close > prev_high)
    # Exit: tide flips or oscillator overbought.
    exit_ = (~gate) | overbought.astype(bool)

    n = len(ltf)
    return SignalArrays(
        entry=entry.astype(bool),
        exit_=exit_.astype(bool),
        confidence=np.where(entry, 0.99, 0.0),
        size=np.full(n, size),
        stop_loss=prev_low,                 # Elder: stop below prior bar low
        take_profit=np.full(n, np.nan),
    )


def build_variants():
    """The counted variant set: (name, entry_tf, builder)."""
    variants = []
    for gate_name in GATES:
        for osc_name in OSCILLATORS:
            for entry_tf in ('1m', '5m'):
                name = f"triple_{gate_name.split('_')[0]}_{osc_name}_{entry_tf}"

                def make(g=gate_name, o=osc_name):
                    def build(ltf, h1, size=RAIL_SIZE):
                        return triple_screen(ltf, h1, g, o, size)
                    return build
                variants.append((name, entry_tf, make()))
    return variants


N_VARIANTS = len(build_variants())      # 8, counted for the correction
