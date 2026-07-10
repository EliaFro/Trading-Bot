"""MLCore live loop: champion/challenger guard, retrain logging,
prediction storage, and evidence tiers."""

import numpy as np
import pandas as pd
import pytest

from src.ml import live as ml_live
from src.ml.dataset import evidence_count, evidence_tier
from tests.test_ml_dataset import synthetic_daily


@pytest.fixture
def core(tmp_db, tmp_path, monkeypatch):
    monkeypatch.setattr(ml_live, 'BUNDLE_DIR', tmp_path / 'ml_core')
    # shrink windows so synthetic data trains fast
    from src.ml import walkforward_ml as wfml
    monkeypatch.setattr(wfml, 'TRAIN_DAYS', 200)
    monkeypatch.setattr(wfml, 'VALIDATION_DAYS', 40)
    monkeypatch.setattr(wfml, 'MIN_TRAIN_ROWS', 60)
    return ml_live.MLCore(tmp_db, min_improvement=0.02)


@pytest.fixture
def frames():
    return {
        'BTC/USDT': synthetic_daily(n=600, seed=11),
        'ETH/USDT': synthetic_daily(n=600, seed=12),
    }


# ── Evidence tiers: the "stable ≠ meaningful" lesson, codified ──────────────

def test_evidence_tiers_distinguish_month_from_fast_features():
    span = 1090   # 36 months
    assert evidence_tier('month', span) == 'thin — treat as noise'
    assert evidence_count('month', span) <= 3

    assert evidence_tier('ret_1', span) == 'well-supported'
    assert evidence_count('ret_1', span) >= 1000

    assert evidence_tier('btc_rv_20', span) == 'moderate'
    # slow lookbacks are honestly thin too — 3 years is only ~12
    # independent 90-day observations
    assert evidence_tier('ret_90', span) == 'thin — treat as noise'


# ── Champion/challenger guard ───────────────────────────────────────────────

def test_initial_retrain_creates_champion(core, frames):
    record = core.retrain(frames)
    assert record['decision'] == 'INITIAL'
    assert core.bundle is not None
    assert core.status()['active']
    # persisted: a fresh core on the same dir loads the champion
    core2 = ml_live.MLCore(core.db, min_improvement=0.02)
    assert core2.bundle is not None
    assert core2.bundle['version'] == core.bundle['version']


def test_guard_keeps_old_when_no_real_improvement(core, frames):
    core.retrain(frames)
    old_version = core.bundle['version']

    # same data -> challenger ~equal to champion -> cannot clear the +2% bar
    record = core.retrain(frames)
    assert record['decision'] == 'KEPT_OLD'
    assert core.bundle['version'] == old_version
    assert 'guard held' in record['reason']


def test_guard_replaces_when_improvement_clears_margin(core, frames):
    core.retrain(frames)
    old_version = core.bundle['version']

    # force the guard open: any positive challenger beats margin -1
    core.min_improvement = -1.0
    record = core.retrain(frames)
    assert record['decision'] == 'REPLACED'
    assert core.bundle['version'] != old_version


def test_every_decision_is_logged_with_numbers(core, frames):
    core.retrain(frames)
    core.retrain(frames)
    log = pd.read_sql_query("SELECT * FROM ml_retrain_log ORDER BY id",
                            core.db.engine)
    assert len(log) == 2
    assert list(log['decision']) == ['INITIAL', 'KEPT_OLD']
    row = log.iloc[1]
    assert row['old_val_f1'] > 0 and row['new_val_f1'] > 0
    assert row['new_is_bal_acc'] >= row['new_val_bal_acc'] - 0.2
    assert row['reason'] and row['feature_importance']


# ── Prediction path ─────────────────────────────────────────────────────────

def test_predict_and_store(core, frames):
    core.retrain(frames)
    predictions = core.predict(frames)
    assert set(predictions) == set(frames)
    for p in predictions.values():
        assert p['pred'] in ('UP', 'FLAT', 'DOWN')
        assert 0 <= p['p_up'] <= 1 and 0 <= p['p_down'] <= 1

    core.store_predictions(predictions, executed={'BTC/USDT': True})
    stored = pd.read_sql_query("SELECT * FROM ml_predictions", core.db.engine)
    assert len(stored) == len(frames)
    assert stored[stored['symbol'] == 'BTC/USDT']['executed'].iloc[0] == 1
