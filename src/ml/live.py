"""
MLCore — the live paper-trading ML loop (Stage 2).

Champion/challenger with the keep-old-unless-better guard:
  * weekly retrain builds a CHALLENGER on the latest 400 labeled days
  * champion and challenger are scored on the SAME validation slice
    (the most recent 60 labeled days — data the challenger never fit on
    for weighting, and the champion has never seen at all)
  * the challenger replaces the champion only if its validation macro-F1
    improves on the champion's by > models.min_improvement (2%)
  * every decision — kept or replaced — is logged to ml_retrain_log with
    the numbers and reasoning, so the guard is auditable over the weeks

Labels for the last HORIZON+1 days do not exist yet (the future hasn't
happened); assemble_panel drops them, so live training can never peek.
"""

import json
import logging
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, Optional

import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import balanced_accuracy_score, f1_score

from src.ml import walkforward_ml as wfml
from src.ml.dataset import (LABEL_NAMES, LABEL_UP, LABEL_DOWN,
                            assemble_panel)

logger = logging.getLogger(__name__)

BUNDLE_DIR = Path('./models/ml_core')


def build_daily_frames(db, symbols) -> Dict[str, pd.DataFrame]:
    """Daily UTC bars per symbol, resampled from stored 1h candles.
    Only completed days are included (the last, possibly-partial day is
    dropped) so features never see an unfinished bar."""
    out = {}
    today = pd.Timestamp.now(tz=timezone.utc).normalize().tz_localize(None)
    for symbol in symbols:
        h1 = db.get_ohlcv_data(symbol, '1h')
        if h1.empty:
            continue
        h1 = h1.set_index('timestamp').sort_index()
        d = pd.DataFrame({
            'open': h1['open'].resample('1D').first(),
            'high': h1['high'].resample('1D').max(),
            'low': h1['low'].resample('1D').min(),
            'close': h1['close'].resample('1D').last(),
            'volume': h1['volume'].resample('1D').sum(),
        }).dropna()
        out[symbol] = d[d.index < today]        # completed days only
    return out


class MLCore:
    """Owns the champion model bundle and the retrain/predict lifecycle."""

    def __init__(self, db, min_improvement: float = 0.02):
        self.db = db
        self.min_improvement = min_improvement
        self.bundle: Optional[Dict] = None
        BUNDLE_DIR.mkdir(parents=True, exist_ok=True)
        self._load_champion()

    # ── Persistence ──────────────────────────────────────────────────────

    def _champion_path(self) -> Path:
        return BUNDLE_DIR / 'champion.joblib'

    def _load_champion(self):
        path = self._champion_path()
        if path.exists():
            try:
                self.bundle = joblib.load(path)
                logger.info(f"ML champion loaded: version "
                            f"{self.bundle['version']} "
                            f"(val F1 {self.bundle['val_f1']:.3f})")
            except Exception as e:
                logger.error(f"Champion load failed: {e}")

    def _save_champion(self):
        joblib.dump(self.bundle, self._champion_path())

    # ── Training ─────────────────────────────────────────────────────────

    def _fit_challenger(self, frames: Dict[str, pd.DataFrame]) -> Dict:
        """Train a challenger on the latest labeled window."""
        X, meta = assemble_panel(frames)
        dates = pd.DatetimeIndex(sorted(meta['date'].unique()))
        train_start = dates[-1] - pd.Timedelta(days=wfml.TRAIN_DAYS)
        val_start = dates[-1] - pd.Timedelta(days=wfml.VALIDATION_DAYS)

        is_fit = (meta['date'] >= train_start) & (meta['date'] < val_start)
        is_val = meta['date'] >= val_start
        if is_fit.sum() < wfml.MIN_TRAIN_ROWS or is_val.sum() < 30:
            raise RuntimeError(
                f"Not enough labeled data to retrain "
                f"(fit {int(is_fit.sum())}, val {int(is_val.sum())})")

        X_fit, y_fit = X[is_fit], meta.loc[is_fit, 'label']
        X_val, y_val = X[is_val], meta.loc[is_val, 'label']

        models, weights, scaler = wfml.train_window(X_fit, y_fit, X_val, y_val)

        proba_val = wfml.ensemble_proba(models, weights, scaler, X_val)
        pred_val = proba_val.argmax(axis=1)
        proba_fit = wfml.ensemble_proba(models, weights, scaler, X_fit)

        importance = wfml.tree_feature_importance(models, weights)
        return {
            'version': datetime.now(timezone.utc).strftime('%Y%m%d%H%M%S'),
            'trained_at': datetime.now(timezone.utc).isoformat(),
            'models': models, 'weights': weights, 'scaler': scaler,
            'val_f1': float(f1_score(y_val, pred_val, average='macro')),
            'val_bal_acc': float(balanced_accuracy_score(y_val, pred_val)),
            'is_bal_acc': float(balanced_accuracy_score(
                y_fit, proba_fit.argmax(axis=1))),
            'n_train': int(is_fit.sum()),
            'feature_importance': importance.sort_values(
                ascending=False).round(5).to_dict(),
            '_val_data': (X_val, y_val),   # transient, for champion scoring
        }

    def _score_on(self, bundle: Dict, X_val, y_val) -> float:
        proba = wfml.ensemble_proba(bundle['models'], bundle['weights'],
                                    bundle['scaler'], X_val)
        return float(f1_score(y_val, proba.argmax(axis=1), average='macro'))

    def retrain(self, frames: Dict[str, pd.DataFrame]) -> Dict:
        """Champion/challenger retrain. Returns the logged decision record."""
        challenger = self._fit_challenger(frames)
        X_val, y_val = challenger.pop('_val_data')

        if self.bundle is None:
            decision, reason = 'INITIAL', 'no champion existed'
            old_f1 = None
            self.bundle = challenger
            self._save_champion()
        else:
            old_f1 = self._score_on(self.bundle, X_val, y_val)
            needed = old_f1 * (1 + self.min_improvement)
            if challenger['val_f1'] > needed:
                decision = 'REPLACED'
                reason = (f"challenger F1 {challenger['val_f1']:.3f} > "
                          f"champion {old_f1:.3f} x (1+{self.min_improvement:.0%})"
                          f" = {needed:.3f}, on the same validation slice")
                self.bundle = challenger
                self._save_champion()
            else:
                decision = 'KEPT_OLD'
                reason = (f"challenger F1 {challenger['val_f1']:.3f} did not "
                          f"beat champion {old_f1:.3f} by the required "
                          f"{self.min_improvement:.0%} margin — keeping the "
                          f"champion (guard held)")

        record = {
            'timestamp': int(datetime.now(timezone.utc).timestamp()),
            'old_version': self.bundle['version'] if decision == 'KEPT_OLD'
            else (None if decision == 'INITIAL' else 'superseded'),
            'new_version': challenger['version'],
            'decision': decision, 'reason': reason,
            'old_val_f1': old_f1, 'new_val_f1': challenger['val_f1'],
            'new_is_bal_acc': challenger['is_bal_acc'],
            'new_val_bal_acc': challenger['val_bal_acc'],
            'n_train': challenger['n_train'],
            'feature_importance': json.dumps(dict(list(
                challenger['feature_importance'].items())[:20])),
        }
        self._log_retrain(record)
        logger.info(f"ML retrain: {decision} — {reason}")
        return record

    def _log_retrain(self, r: Dict):
        try:
            with self.db.get_connection() as conn:
                conn.execute("""
                    INSERT INTO ml_retrain_log
                    (timestamp, old_version, new_version, decision, reason,
                     old_val_f1, new_val_f1, new_is_bal_acc, new_val_bal_acc,
                     n_train, feature_importance)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """, (r['timestamp'], r['old_version'], r['new_version'],
                      r['decision'], r['reason'], r['old_val_f1'],
                      r['new_val_f1'], r['new_is_bal_acc'],
                      r['new_val_bal_acc'], r['n_train'],
                      r['feature_importance']))
                conn.commit()
        except Exception as e:
            logger.error(f"Retrain log failed: {e}")

    def last_retrain_time(self) -> Optional[datetime]:
        try:
            from sqlalchemy import text
            with self.db.engine.connect() as conn:
                row = conn.execute(text(
                    "SELECT MAX(timestamp) FROM ml_retrain_log")).fetchone()
            if row and row[0]:
                return datetime.fromtimestamp(row[0], tz=timezone.utc)
        except Exception:
            pass
        return None

    # ── Prediction ───────────────────────────────────────────────────────

    def predict(self, frames: Dict[str, pd.DataFrame]) -> Dict[str, Dict]:
        """Today's prediction per symbol from the champion. Features use only
        completed daily bars (the caller passes frames of closed days)."""
        if self.bundle is None:
            return {}
        from src.ml.dataset import build_features, FEATURE_COLUMNS

        out = {}
        btc_df = frames.get('BTC/USDT')
        for symbol, df in frames.items():
            feats = build_features(df, btc_df=btc_df).iloc[[-1]]
            if feats.isna().any(axis=1).iloc[0]:
                logger.warning(f"ML predict: NaN features for {symbol}, skipping")
                continue
            proba = wfml.ensemble_proba(
                self.bundle['models'], self.bundle['weights'],
                self.bundle['scaler'], feats[FEATURE_COLUMNS])[0]
            pred = int(np.argmax(proba))
            out[symbol] = {
                'pred': LABEL_NAMES[pred],
                'p_up': float(proba[LABEL_UP]),
                'p_down': float(proba[LABEL_DOWN]),
                'model_version': self.bundle['version'],
                'as_of': str(df.index[-1].date()),
            }
        return out

    def store_predictions(self, predictions: Dict[str, Dict],
                          executed: Dict[str, bool] = None):
        executed = executed or {}
        try:
            with self.db.get_connection() as conn:
                for symbol, p in predictions.items():
                    conn.execute("""
                        INSERT INTO ml_predictions
                        (timestamp, symbol, pred, p_up, p_down,
                         model_version, executed)
                        VALUES (?, ?, ?, ?, ?, ?, ?)
                    """, (int(datetime.now(timezone.utc).timestamp()), symbol,
                          p['pred'], p['p_up'], p['p_down'],
                          p['model_version'],
                          1 if executed.get(symbol) else 0))
                conn.commit()
        except Exception as e:
            logger.error(f"Prediction store failed: {e}")

    def status(self) -> Dict:
        if self.bundle is None:
            return {'active': False}
        return {
            'active': True,
            'version': self.bundle['version'],
            'trained_at': self.bundle['trained_at'],
            'val_f1': self.bundle['val_f1'],
            'val_bal_acc': self.bundle['val_bal_acc'],
            'is_bal_acc': self.bundle['is_bal_acc'],
            'is_oos_gap': self.bundle['is_bal_acc'] - self.bundle['val_bal_acc'],
        }
