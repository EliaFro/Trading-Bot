"""
Walk-forward training and evaluation for the ML learning core.

Protocol (docs/ML_PLAN.md §5):
    rolling 400-day train  ->  purge last (HORIZON+1) train days
    ->  2-day embargo  ->  20-day test window, touched once  ->  roll 20d

Models: RandomForest, GradientBoosting, LogisticRegression — pre-registered
conservative hyperparameters, identical in every window. (Deviation from the
plan's optuna-per-window option, in the strict direction: with ~1,100 pooled
training samples per window, per-window hyperparameter search mostly fits
noise, so it is disabled for Stage 1. The trainer's TimeSeriesSplit/optuna
path remains available for Stage 2 experiments.)

Ensemble: soft vote weighted by each model's macro-F1 on a validation slice
carved from the END of the training window (last 60 train days). Reported
out-of-sample windows are never used for any selection decision.
"""

import logging
from dataclasses import dataclass, field
from datetime import timedelta
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import balanced_accuracy_score, f1_score
from sklearn.preprocessing import StandardScaler

from src.ml.dataset import (FEATURE_COLUMNS, FEATURE_GROUPS, HORIZON,
                            LABEL_DOWN, LABEL_UP)

logger = logging.getLogger(__name__)

TRAIN_DAYS = 400
PURGE_DAYS = HORIZON + 1        # labels at train-end reach HORIZON+1 forward
EMBARGO_DAYS = 2
TEST_DAYS = 20
VALIDATION_DAYS = 60            # tail of train window, for ensemble weights
MIN_TRAIN_ROWS = 300            # skip windows with fewer pooled samples

# Pre-registered hyperparameters — identical in every window, never tuned
# on anything the reported OOS touches.
MODEL_SPECS = {
    'logreg': lambda: LogisticRegression(
        C=1.0, class_weight='balanced', max_iter=2000),
    'random_forest': lambda: RandomForestClassifier(
        n_estimators=400, max_depth=6, min_samples_leaf=20,
        class_weight='balanced_subsample', n_jobs=-1, random_state=42),
    'gradient_boosting': lambda: GradientBoostingClassifier(
        n_estimators=200, learning_rate=0.05, max_depth=3,
        subsample=0.8, random_state=42),
}


@dataclass
class WindowReport:
    window: int
    train_start: pd.Timestamp
    test_start: pd.Timestamp
    test_end: pd.Timestamp
    n_train: int
    n_test: int
    model_metrics: Dict[str, Dict] = field(default_factory=dict)
    ensemble_weights: Dict[str, float] = field(default_factory=dict)
    feature_importance: Optional[pd.Series] = None
    # per-row test predictions (aligned with meta rows for this window)
    predictions: Optional[pd.DataFrame] = None


def _metrics(y_true, y_pred) -> Dict[str, float]:
    return {
        'accuracy': float((y_true == y_pred).mean()),
        'balanced_accuracy': float(balanced_accuracy_score(y_true, y_pred)),
        'macro_f1': float(f1_score(y_true, y_pred, average='macro')),
    }


def train_window(X_train, y_train, X_val, y_val) -> Tuple[Dict, Dict, object]:
    """Fit all models on the (purged) train slice; weight them by validation
    macro-F1. Returns (fitted models dict, weights dict, fitted scaler)."""
    scaler = StandardScaler().fit(X_train)
    Xt, Xv = scaler.transform(X_train), scaler.transform(X_val)

    models, weights = {}, {}
    for name, make in MODEL_SPECS.items():
        model = make()
        model.fit(Xt, y_train)
        val_pred = model.predict(Xv)
        f1 = f1_score(y_val, val_pred, average='macro')
        models[name] = model
        # weight = edge over the 3-class chance level; floor at 0
        weights[name] = max(f1 - 1 / 3, 0.0)

    total = sum(weights.values())
    if total <= 0:
        weights = {k: 1 / len(models) for k in models}
    else:
        weights = {k: v / total for k, v in weights.items()}
    return models, weights, scaler


def ensemble_proba(models: Dict, weights: Dict, scaler, X) -> np.ndarray:
    Xs = scaler.transform(X)
    proba = None
    for name, model in models.items():
        p = model.predict_proba(Xs) * weights[name]
        proba = p if proba is None else proba + p
    return proba


def tree_feature_importance(models: Dict, weights: Dict) -> pd.Series:
    """Weighted tree importances (LogReg excluded — different scale)."""
    total, imp = 0.0, np.zeros(len(FEATURE_COLUMNS))
    for name in ('random_forest', 'gradient_boosting'):
        model = models.get(name)
        if model is not None and hasattr(model, 'feature_importances_'):
            w = max(weights.get(name, 0.0), 1e-6)
            imp += model.feature_importances_ * w
            total += w
    return pd.Series(imp / total if total else imp, index=FEATURE_COLUMNS)


def run_walkforward_ml(X: pd.DataFrame, meta: pd.DataFrame
                       ) -> List[WindowReport]:
    """Full walk-forward pass over the pooled panel. Every window trains on
    strictly-prior data (purged + embargoed) and predicts its test days."""
    dates = pd.DatetimeIndex(sorted(meta['date'].unique()))
    first_test = dates[0] + timedelta(days=TRAIN_DAYS + PURGE_DAYS + EMBARGO_DAYS)

    reports: List[WindowReport] = []
    window_no = 0
    test_start = first_test
    while test_start + timedelta(days=TEST_DAYS) <= dates[-1]:
        window_no += 1
        test_end = test_start + timedelta(days=TEST_DAYS)
        train_cutoff = test_start - timedelta(days=PURGE_DAYS + EMBARGO_DAYS)
        train_start = train_cutoff - timedelta(days=TRAIN_DAYS)
        val_start = train_cutoff - timedelta(days=VALIDATION_DAYS)

        is_train = (meta['date'] >= train_start) & (meta['date'] < val_start)
        is_val = (meta['date'] >= val_start) & (meta['date'] < train_cutoff)
        is_test = (meta['date'] >= test_start) & (meta['date'] < test_end)

        # hard anti-leak assertion: no train/val label horizon may reach test
        latest_label_reach = (meta.loc[is_train | is_val, 'date'].max()
                              + timedelta(days=PURGE_DAYS))
        assert latest_label_reach < test_start, "purge/embargo violated"

        if is_test.sum() == 0 or is_train.sum() < MIN_TRAIN_ROWS:
            test_start += timedelta(days=TEST_DAYS)
            continue

        X_tr, y_tr = X[is_train], meta.loc[is_train, 'label']
        X_va, y_va = X[is_val], meta.loc[is_val, 'label']
        X_te, y_te = X[is_test], meta.loc[is_test, 'label']

        models, weights, scaler = train_window(X_tr, y_tr, X_va, y_va)

        report = WindowReport(
            window=window_no, train_start=train_start,
            test_start=test_start, test_end=test_end,
            n_train=int(is_train.sum()), n_test=int(is_test.sum()),
            ensemble_weights=weights,
            feature_importance=tree_feature_importance(models, weights))

        # per-model IS vs OOS (the overfitting gauge's raw material)
        Xt_s = scaler.transform(X_tr)
        Xe_s = scaler.transform(X_te)
        for name, model in models.items():
            report.model_metrics[name] = {
                'is': _metrics(y_tr.values, model.predict(Xt_s)),
                'oos': _metrics(y_te.values, model.predict(Xe_s)),
            }

        proba_te = ensemble_proba(models, weights, scaler, X_te)
        pred_te = proba_te.argmax(axis=1)
        proba_tr = ensemble_proba(models, weights, scaler, X_tr)
        report.model_metrics['ensemble'] = {
            'is': _metrics(y_tr.values, proba_tr.argmax(axis=1)),
            'oos': _metrics(y_te.values, pred_te),
        }

        report.predictions = pd.DataFrame({
            'date': meta.loc[is_test, 'date'].values,
            'symbol': meta.loc[is_test, 'symbol'].values,
            'label': y_te.values,
            'pred': pred_te,
            'p_down': proba_te[:, LABEL_DOWN],
            'p_up': proba_te[:, LABEL_UP],
        })

        oos = report.model_metrics['ensemble']['oos']
        gap = (report.model_metrics['ensemble']['is']['balanced_accuracy']
               - oos['balanced_accuracy'])
        logger.info(
            f"w{window_no:02d} {test_start:%Y-%m-%d} | ens OOS bal-acc "
            f"{oos['balanced_accuracy']:.3f} f1 {oos['macro_f1']:.3f} "
            f"| IS-OOS gap {gap:+.3f} | weights "
            + " ".join(f"{k[:2]}={v:.2f}" for k, v in weights.items()))

        reports.append(report)
        test_start += timedelta(days=TEST_DAYS)

    return reports


def stitch_predictions(reports: List[WindowReport]) -> pd.DataFrame:
    """All OOS predictions, one row per (date, symbol), chronological."""
    frames = [r.predictions for r in reports if r.predictions is not None]
    if not frames:
        return pd.DataFrame()
    out = pd.concat(frames, ignore_index=True)
    return out.sort_values(['date', 'symbol']).reset_index(drop=True)


def importance_stability(reports: List[WindowReport]) -> Dict:
    """Mean/std importance per feature + rank stability across retrains."""
    series = [r.feature_importance for r in reports
              if r.feature_importance is not None]
    if len(series) < 2:
        return {}
    matrix = pd.concat(series, axis=1)
    mean_imp = matrix.mean(axis=1).sort_values(ascending=False)
    std_imp = matrix.std(axis=1)

    from scipy.stats import spearmanr
    rank_corrs = [float(spearmanr(matrix.iloc[:, i], matrix.iloc[:, i + 1]).statistic)
                  for i in range(matrix.shape[1] - 1)]

    group_share = {}
    for group, cols in FEATURE_GROUPS.items():
        group_share[group] = float(mean_imp[cols].sum())

    return {
        'mean_importance': mean_imp,
        'std_importance': std_imp,
        'rank_stability_mean': float(np.mean(rank_corrs)),
        'rank_stability_min': float(np.min(rank_corrs)),
        'group_share': dict(sorted(group_share.items(),
                                   key=lambda kv: -kv[1])),
    }
