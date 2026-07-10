"""Fast Lab dataset (1m): anti-lookahead enforcement + label economics."""

import numpy as np
import pandas as pd
import pytest

from src.ml import fast_dataset as fd


def synthetic_1m(n=3000, seed=9):
    rng = np.random.default_rng(seed)
    rets = rng.normal(0, 0.0008, n)
    close = 100 * np.exp(np.cumsum(rets))
    open_ = np.concatenate([[100.0], close[:-1]])
    high = np.maximum(open_, close) * (1 + np.abs(rng.normal(0, 3e-4, n)))
    low = np.minimum(open_, close) * (1 - np.abs(rng.normal(0, 3e-4, n)))
    volume = rng.uniform(1, 10, n)
    return pd.DataFrame({'open': open_, 'high': high, 'low': low,
                         'close': close, 'volume': volume},
                        index=pd.date_range('2025-03-01', periods=n, freq='1min'))


def test_channels_truncation_invariant():
    df = synthetic_1m()
    full = fd.build_channels(df)
    for t in (1500, 2500):
        trunc = fd.build_channels(df.iloc[:t + 1])
        pd.testing.assert_series_equal(full.iloc[t], trunc.iloc[-1],
                                       check_names=False, atol=1e-10)


def test_labels_are_net_of_full_costs():
    df = synthetic_1m(n=200)
    rt = 0.0031
    labels = fd.build_labels(df, round_trip=rt)
    t = 50
    expected = (df['open'].iloc[t + 31] / df['open'].iloc[t + 1] - 1) - rt
    assert abs(labels['net_fwd_return'].iloc[t] - expected) < 1e-12


def test_costs_suppress_up_labels():
    """The fee wall must appear inside the target: subtracting the long-side
    round trip shifts the label distribution AGAINST UP — 'not worth buying'
    dominates, and UP is strictly rarer than it would be without costs."""
    df = synthetic_1m(n=3000)
    with_costs = fd.build_labels(df, round_trip=0.0031)
    without = fd.build_labels(df, round_trip=0.0)
    v_c = with_costs[with_costs['label'] >= 0]
    v_0 = without[without['label'] >= 0]

    up_with = (v_c['label'] == fd.LABEL_UP).mean()
    up_without = (v_0['label'] == fd.LABEL_UP).mean()
    assert up_with < up_without, "costs must reduce UP labels"
    assert up_with < 0.40, "UP must be a minority class at 1m cost levels"
    # and non-UP (FLAT + DOWN = 'don't buy') dominates
    assert up_with < 0.5


def test_labels_do_not_peek_beyond_horizon():
    df = synthetic_1m(n=200)
    before = fd.build_labels(df, 0.0031).iloc[40]
    tampered = df.copy()
    tampered.iloc[80:, tampered.columns.get_loc('open')] *= 3.0   # 40+31 < 80
    after = fd.build_labels(tampered, 0.0031).iloc[40]
    assert before['label'] == after['label']


def test_panel_sequences_align_with_meta():
    frames = {'A/USDT': synthetic_1m(seed=1), 'B/USDT': synthetic_1m(seed=2)}
    X_tab, X_seq, meta = fd.assemble_fast_panel(frames, {'A/USDT': 0.0001,
                                                         'B/USDT': 0.0002})
    assert len(X_tab) == len(X_seq) == len(meta)
    assert X_seq.shape[1:] == (fd.SEQ_LEN, len(fd.SEQ_CHANNELS))
    assert X_tab.shape[1] == len(fd.TAB_COLUMNS)
    assert meta['date'].is_monotonic_increasing
    assert np.isfinite(X_tab).all() and np.isfinite(X_seq).all()
    # the last channel row of each sequence equals the tabular row's shared cols
    np.testing.assert_allclose(X_seq[:, -1, :],
                               X_tab[:, :len(fd.SEQ_CHANNELS)], atol=1e-6)
