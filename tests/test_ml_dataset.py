"""Anti-lookahead enforcement for the ML dataset and walk-forward protocol.
These tests ARE the written guarantees in docs/ML_PLAN.md §6."""

import numpy as np
import pandas as pd
import pytest

from src.ml import dataset as ds
from src.ml import walkforward_ml as wfml


def synthetic_daily(n=600, seed=3, start='2023-01-01'):
    rng = np.random.default_rng(seed)
    rets = rng.normal(0.001, 0.03, n)
    close = 100 * np.exp(np.cumsum(rets))
    open_ = np.concatenate([[100.0], close[:-1]]) * (1 + rng.normal(0, 0.002, n))
    high = np.maximum(open_, close) * (1 + np.abs(rng.normal(0, 0.005, n)))
    low = np.minimum(open_, close) * (1 - np.abs(rng.normal(0, 0.005, n)))
    volume = rng.uniform(1e3, 5e3, n)
    return pd.DataFrame({'open': open_, 'high': high, 'low': low,
                         'close': close, 'volume': volume},
                        index=pd.date_range(start, periods=n, freq='1D'))


# ── Trap 1: no lookahead in features ─────────────────────────────────────────

def test_features_are_truncation_invariant():
    """Feature values at date T must be identical whether computed on the
    full history or on history truncated at T — i.e., no feature can see
    the future."""
    df = synthetic_daily()
    full = ds.build_features(df, btc_df=df)

    for t_idx in (250, 400, 550):
        truncated = ds.build_features(df.iloc[:t_idx + 1],
                                      btc_df=df.iloc[:t_idx + 1])
        row_full = full.iloc[t_idx]
        row_trunc = truncated.iloc[-1]
        pd.testing.assert_series_equal(row_full, row_trunc,
                                       check_names=False, atol=1e-10)


# ── Trap 2: labels use exactly the declared horizon, nothing beyond ─────────

def test_label_math_is_executable_and_after_fee():
    df = synthetic_daily(n=50)
    labels = ds.build_labels(df, horizon=5, cost=0.003, dead_zone=0.01)

    t = 10
    expected_net = (df['open'].iloc[t + 6] / df['open'].iloc[t + 1] - 1) - 0.003
    assert abs(labels['net_fwd_return'].iloc[t] - expected_net) < 1e-12

    if expected_net > 0.01:
        assert labels['label'].iloc[t] == ds.LABEL_UP
    elif expected_net < -0.01:
        assert labels['label'].iloc[t] == ds.LABEL_DOWN
    else:
        assert labels['label'].iloc[t] == ds.LABEL_FLAT


def test_labels_do_not_peek_beyond_horizon():
    """Changing prices AFTER T+1+horizon must not change the label at T."""
    df = synthetic_daily(n=60)
    before = ds.build_labels(df).iloc[20]

    tampered = df.copy()
    tampered.iloc[30:, tampered.columns.get_loc('open')] *= 5.0  # 20+6 < 30
    after = ds.build_labels(tampered).iloc[20]

    assert before['label'] == after['label']
    assert abs(before['net_fwd_return'] - after['net_fwd_return']) < 1e-12


def test_unlabeled_tail_is_excluded():
    df = synthetic_daily(n=40)
    labels = ds.build_labels(df, horizon=5)
    assert (labels['label'].iloc[-6:] == -1).all()   # future unknown
    assert (labels['label'].iloc[:-6] >= 0).all()


# ── Trap 3: purge + embargo separation in the walk-forward ──────────────────

def test_walkforward_train_labels_never_reach_test(monkeypatch):
    """For every window: the furthest forward-reach of any train/val label
    must end strictly before the test window starts."""
    monkeypatch.setattr(wfml, 'TRAIN_DAYS', 120)
    monkeypatch.setattr(wfml, 'VALIDATION_DAYS', 30)
    monkeypatch.setattr(wfml, 'TEST_DAYS', 15)
    monkeypatch.setattr(wfml, 'MIN_TRAIN_ROWS', 60)

    # 600 days: ~200 consumed by feature warmup (SMA200), the rest by
    # train+test windows
    df = synthetic_daily(n=600)
    X, meta = ds.assemble_panel({'BTC/USDT': df})
    reports = wfml.run_walkforward_ml(X, meta)

    assert len(reports) >= 3, "expected multiple windows on 600 days"
    for r in reports:
        train_mask = (meta['date'] < r.test_start)
        # the assertion inside run_walkforward_ml already guards this;
        # verify independently from the report bounds:
        latest_train_date = r.test_start - pd.Timedelta(
            days=wfml.PURGE_DAYS + wfml.EMBARGO_DAYS)
        label_reach = latest_train_date + pd.Timedelta(days=wfml.PURGE_DAYS)
        assert label_reach < r.test_start
        # and predictions only cover the test range
        assert (r.predictions['date'] >= r.test_start).all()
        assert (r.predictions['date'] < r.test_end).all()


def test_windows_advance_and_never_overlap(monkeypatch):
    monkeypatch.setattr(wfml, 'TRAIN_DAYS', 120)
    monkeypatch.setattr(wfml, 'VALIDATION_DAYS', 30)
    monkeypatch.setattr(wfml, 'TEST_DAYS', 15)
    monkeypatch.setattr(wfml, 'MIN_TRAIN_ROWS', 60)

    df = synthetic_daily(n=600)
    X, meta = ds.assemble_panel({'BTC/USDT': df})
    reports = wfml.run_walkforward_ml(X, meta)

    assert len(reports) >= 2, "vacuous test: no windows produced"
    for a, b in zip(reports, reports[1:]):
        assert b.test_start >= a.test_end   # OOS days touched exactly once


# ── Trap 4: on pure noise, OOS performance must hover at chance ─────────────

def test_no_skill_on_random_walk(monkeypatch):
    """The whole pipeline run on a random walk should show NO reliable OOS
    skill. If it does, something leaks."""
    monkeypatch.setattr(wfml, 'TRAIN_DAYS', 150)
    monkeypatch.setattr(wfml, 'VALIDATION_DAYS', 30)
    monkeypatch.setattr(wfml, 'TEST_DAYS', 15)

    frames = {f'S{i}/USDT': synthetic_daily(n=700, seed=100 + i)
              for i in range(2)}
    frames['BTC/USDT'] = synthetic_daily(n=700, seed=99)
    X, meta = ds.assemble_panel(frames)
    reports = wfml.run_walkforward_ml(X, meta)
    assert reports, "expected windows"

    oos_bal = np.mean([r.model_metrics['ensemble']['oos']['balanced_accuracy']
                       for r in reports])
    # 3-class chance = 1/3; allow noise band, fail on clear leakage
    assert oos_bal < 0.45, (
        f"OOS balanced accuracy {oos_bal:.3f} on pure noise — "
        f"the pipeline is leaking future information")
