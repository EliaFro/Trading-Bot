"""MTF (triple screen) family: HTF projection must not leak the future,
and every variant must produce sane signals."""

import numpy as np
import pandas as pd
import pytest

from src.backtesting import mtf_library as mtf
from tests.test_ml_dataset import synthetic_daily


def synthetic_intraday(n_days=30, freq='5min', seed=5):
    n = int(n_days * 24 * 60 / int(freq.replace('min', '')))
    rng = np.random.default_rng(seed)
    rets = rng.normal(0, 0.002, n)
    close = 100 * np.exp(np.cumsum(rets))
    open_ = np.concatenate([[100.0], close[:-1]])
    high = np.maximum(open_, close) * (1 + np.abs(rng.normal(0, 0.0008, n)))
    low = np.minimum(open_, close) * (1 - np.abs(rng.normal(0, 0.0008, n)))
    volume = rng.uniform(10, 100, n)
    return pd.DataFrame({'open': open_, 'high': high, 'low': low,
                         'close': close, 'volume': volume},
                        index=pd.date_range('2025-01-01', periods=n, freq=freq))


def to_h1(ltf):
    return pd.DataFrame({
        'open': ltf['open'].resample('1h').first(),
        'high': ltf['high'].resample('1h').max(),
        'low': ltf['low'].resample('1h').min(),
        'close': ltf['close'].resample('1h').last(),
        'volume': ltf['volume'].resample('1h').sum(),
    }).dropna()


def test_htf_projection_has_availability_delay():
    """The 4h bar stamped at open-time O must NOT influence LTF bars before
    O + 4h (the bar hasn't closed yet)."""
    ltf = synthetic_intraday(10, '5min')
    h4 = mtf.resample_4h(to_h1(ltf))
    marker = pd.Series(np.arange(len(h4), dtype=float), index=h4.index)
    projected = mtf.project_htf(marker, ltf.index)

    for i, stamp in enumerate(h4.index[:5]):
        closes_at = stamp + pd.Timedelta('4h')
        before = projected[ltf.index < closes_at]
        # the value i must never appear before its bar closes
        assert not (before == float(i)).any(), \
            f"HTF bar {i} leaked before its close time"


def test_htf_projection_truncation_invariance():
    """Gate value at LTF time T unchanged when future data is removed."""
    ltf = synthetic_intraday(20, '5min')
    h1 = to_h1(ltf)
    full = mtf.triple_screen(ltf, h1, 'ema26_slope_4h', 'stochastic')

    t_idx = len(ltf) - 500
    ltf_cut = ltf.iloc[:t_idx + 1]
    h1_cut = to_h1(ltf_cut)
    trunc = mtf.triple_screen(ltf_cut, h1_cut, 'ema26_slope_4h', 'stochastic')

    assert full.entry[t_idx] == trunc.entry[-1]
    assert full.exit_[t_idx] == trunc.exit_[-1]


def test_all_variants_fire_and_are_sane():
    ltf = synthetic_intraday(40, '5min')
    h1 = to_h1(ltf)
    variants = mtf.build_variants()
    assert len(variants) == mtf.N_VARIANTS == 8
    for name, entry_tf, builder in variants:
        sig = builder(ltf, h1)
        assert sig.entry.dtype == bool
        assert sig.entry.sum() > 0, f"{name}: never fires on 40 days"
        assert sig.exit_.sum() > 0
        # stops are prior-bar lows: strictly below the entry bar's close
        fired = np.where(sig.entry)[0]
        closes = ltf['close'].to_numpy()
        ok = sig.stop_loss[fired] < closes[fired]
        assert ok.mean() > 0.95, f"{name}: stops not below price"
