"""Indicator fallback correctness (TA-Lib-compatible pure-pandas versions)."""

import numpy as np
import pandas as pd

from src.utils import indicators as ta


def _closes(n=100, seed=1):
    rng = np.random.default_rng(seed)
    return 100 * np.exp(np.cumsum(rng.normal(0, 0.01, n)))


def test_sma_matches_pandas():
    closes = _closes()
    out = ta.SMA(closes, 10)
    expected = pd.Series(closes).rolling(10).mean().to_numpy()
    np.testing.assert_allclose(out, expected, equal_nan=True)


def test_rsi_bounds_and_warmup():
    closes = _closes(200)
    rsi = ta.RSI(closes, 14)
    assert np.isnan(rsi[:14]).all()
    valid = rsi[~np.isnan(rsi)]
    assert len(valid) > 150
    assert (valid >= 0).all() and (valid <= 100).all()


def test_rsi_direction():
    up = np.linspace(100, 200, 60)          # monotonic rally -> RSI ~100
    down = np.linspace(200, 100, 60)        # monotonic sell-off -> RSI ~0
    assert ta.RSI(up, 14)[-1] > 90
    assert ta.RSI(down, 14)[-1] < 10


def test_macd_shapes_and_hist():
    closes = _closes(150)
    macd, signal, hist = ta.MACD(closes)
    assert macd.shape == signal.shape == hist.shape == closes.shape
    valid = ~np.isnan(hist)
    np.testing.assert_allclose(hist[valid], (macd - signal)[valid], rtol=1e-9)


def test_atr_positive():
    closes = _closes(120)
    high = closes * 1.01
    low = closes * 0.99
    atr = ta.ATR(high, low, closes, 14)
    valid = atr[~np.isnan(atr)]
    assert (valid > 0).all()


def test_bbands_ordering():
    closes = _closes(80)
    upper, mid, lower = ta.BBANDS(closes, 20)
    valid = ~np.isnan(mid)
    assert (upper[valid] >= mid[valid]).all()
    assert (mid[valid] >= lower[valid]).all()


def test_obv_accumulates_on_rally():
    closes = np.linspace(100, 120, 50)
    volume = np.full(50, 10.0)
    obv = ta.OBV(closes, volume)
    assert obv[-1] > obv[10]


def test_adx_in_range():
    closes = _closes(200, seed=3)
    adx = ta.ADX(closes * 1.01, closes * 0.99, closes, 14)
    valid = adx[~np.isnan(adx)]
    assert len(valid) > 0
    assert (valid >= 0).all() and (valid <= 100).all()


def test_engulfing_detects_bullish():
    # bar 1: red candle; bar 2: green candle engulfing it
    open_ = np.array([100.0, 97.0])
    close = np.array([98.0, 101.0])
    high = np.array([100.5, 101.5])
    low = np.array([97.5, 96.5])
    out = ta.CDLENGULFING(open_, high, low, close)
    assert out[1] == 100
