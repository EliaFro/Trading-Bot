#!/usr/bin/env python3
"""
⚡ Fast Lab bot — the 1m learning accelerator (docs/FASTLAB_PLAN.md).

PAPER ONLY. Banner verdict from day one: intraday trading loses to fees
(Phase 2 + Part B: gross edge/trade ~0.01% vs 0.31% cost). This process
exists to watch ML learning dynamics at ~500x the daily lab's sample rate,
with the fee wall visible live in the dashboard's cost decomposition.

Isolation: own database (data/fastlab.db) for trades/equity/predictions/
retrains; OHLCV is READ from the main DB (WAL mode), which the daily bot
keeps fresh — this process fetches nothing but tickers (1 request/min).

Resource caps (pre-registered): decisions every 60s (ms of CPU), tree
retrain at most every 24h, deep models offline-only pending the Part C
verdict. The kill date for any strategy-search role: 2026-08-07.
"""

import asyncio
import json
import logging
import sys
import time
from datetime import datetime, timezone
from logging.handlers import RotatingFileHandler
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parent.parent))

from dotenv import load_dotenv
load_dotenv()

Path('logs').mkdir(exist_ok=True)
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[logging.StreamHandler(sys.stdout),
              RotatingFileHandler('logs/fastlab.log',
                                  maxBytes=20 * 1024 * 1024, backupCount=3)])
logger = logging.getLogger('fastlab')

import numpy as np
import pandas as pd
from sqlalchemy import text

from src.utils.config import Config
from src.utils.database import DatabaseManager
from src.utils.monitoring import MetricsCollector
from src.utils.notifier import Notifier
from src.trading.engine import TradingEngine
from src.ml import fast_dataset as fdset
from src.ml.walkforward_ml import MODEL_SPECS

SYMBOLS = ['BTC/USDT', 'ETH/USDT', 'SOL/USDT']
SPREADS = {s: json.loads(Path('docs/spread_measurements.json').read_text())
           ['symbols'][s]['used_spread'] for s in SYMBOLS}
DECISION_INTERVAL = 60           # seconds
RETRAIN_INTERVAL_H = 24          # pre-registered cap
TRAIN_DAYS = 60
VAL_DAYS = 5
BUNDLE = Path('./models/fastlab/champion.joblib')


class NoopModels:
    def generate_signals(self, **kwargs):
        return []

    def get_active_models(self):
        return ['fastlab_ml']


class FastML:
    """Trees-only 1m champion with the same keep-old-unless-better guard,
    logging to the Fast Lab DB's ml_retrain_log."""

    def __init__(self, main_db, lab_db, min_improvement=0.02):
        self.main_db = main_db
        self.lab_db = lab_db
        self.min_improvement = min_improvement
        self.bundle = None
        BUNDLE.parent.mkdir(parents=True, exist_ok=True)
        if BUNDLE.exists():
            import joblib
            self.bundle = joblib.load(BUNDLE)
            logger.info(f"fastlab champion loaded: {self.bundle['version']}")

    def _panel(self, days):
        frames = {}
        for symbol in SYMBOLS:
            df = self.main_db.get_ohlcv_data(symbol, '1m',
                                             limit=days * 1440 + 600)
            frames[symbol] = df.set_index('timestamp').sort_index()[
                ['open', 'high', 'low', 'close', 'volume']]
        return fdset.assemble_fast_panel(frames, SPREADS)

    def retrain(self):
        from sklearn.metrics import balanced_accuracy_score, f1_score
        from sklearn.preprocessing import StandardScaler
        import joblib

        X_tab, _, meta = self._panel(TRAIN_DAYS + 2)
        dates = meta['date']
        val_start = dates.max() - pd.Timedelta(days=VAL_DAYS)
        is_fit = dates < val_start
        is_val = ~is_fit
        y = meta['label'].to_numpy()
        if is_fit.sum() < 5000:
            raise RuntimeError("not enough 1m history to retrain")

        scaler = StandardScaler().fit(X_tab[is_fit.to_numpy()])
        Xf = scaler.transform(X_tab[is_fit.to_numpy()])
        Xv = scaler.transform(X_tab[is_val.to_numpy()])
        yf, yv = y[is_fit.to_numpy()], y[is_val.to_numpy()]

        models, val_f1 = {}, {}
        for name, make in MODEL_SPECS.items():
            m = make()
            m.fit(Xf, yf)
            val_f1[name] = f1_score(yv, m.predict(Xv), average='macro')
            models[name] = m
        weights = {m: max(f - 1 / 3, 0) for m, f in val_f1.items()}
        tot = sum(weights.values())
        weights = ({m: w / tot for m, w in weights.items()} if tot
                   else {m: 1 / 3 for m in models})

        proba_v = sum(models[m].predict_proba(Xv) * w
                      for m, w in weights.items())
        proba_f = sum(models[m].predict_proba(Xf[:20000]) * w
                      for m, w in weights.items())
        challenger = {
            'version': datetime.now(timezone.utc).strftime('%Y%m%d%H%M%S'),
            'models': models, 'weights': weights, 'scaler': scaler,
            'val_f1': float(f1_score(yv, proba_v.argmax(1), average='macro')),
            'val_bal_acc': float(balanced_accuracy_score(yv, proba_v.argmax(1))),
            'is_bal_acc': float(balanced_accuracy_score(
                yf[:20000], proba_f.argmax(1))),
            'n_train': int(is_fit.sum()),
        }

        if self.bundle is None:
            decision, reason, old_f1 = 'INITIAL', 'no champion existed', None
            self.bundle = challenger
            joblib.dump(challenger, BUNDLE)
        else:
            old_proba = sum(
                self.bundle['models'][m].predict_proba(
                    self.bundle['scaler'].transform(X_tab[is_val.to_numpy()])) * w
                for m, w in self.bundle['weights'].items())
            old_f1 = float(f1_score(yv, old_proba.argmax(1), average='macro'))
            if challenger['val_f1'] > old_f1 * (1 + self.min_improvement):
                decision = 'REPLACED'
                reason = (f"challenger F1 {challenger['val_f1']:.3f} > champion "
                          f"{old_f1:.3f} x1.02 on the same validation slice")
                self.bundle = challenger
                joblib.dump(challenger, BUNDLE)
            else:
                decision = 'KEPT_OLD'
                reason = (f"challenger F1 {challenger['val_f1']:.3f} vs champion "
                          f"{old_f1:.3f}: below the 2% margin — guard held")

        with self.lab_db.get_connection() as conn:
            conn.execute("""
                INSERT INTO ml_retrain_log
                (timestamp, old_version, new_version, decision, reason,
                 old_val_f1, new_val_f1, new_is_bal_acc, new_val_bal_acc,
                 n_train, feature_importance)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, (int(time.time()), None, challenger['version'], decision,
                  reason, old_f1, challenger['val_f1'],
                  challenger['is_bal_acc'], challenger['val_bal_acc'],
                  challenger['n_train'], '{}'))
            conn.commit()
        logger.info(f"fastlab retrain: {decision} — {reason}")
        return decision

    def predict(self):
        if self.bundle is None:
            return {}
        X_tab, _, meta = self._panel(2)
        out = {}
        for symbol in SYMBOLS:
            rows = np.where((meta['symbol'] == symbol).to_numpy())[0]
            if len(rows) == 0:
                continue
            i = rows[-1]
            proba = sum(
                self.bundle['models'][m].predict_proba(
                    self.bundle['scaler'].transform(X_tab[i:i + 1])) * w
                for m, w in self.bundle['weights'].items())[0]
            out[symbol] = {'pred': ['DOWN', 'FLAT', 'UP'][int(proba.argmax())],
                           'p_up': float(proba[fdset.LABEL_UP]),
                           'p_down': float(proba[fdset.LABEL_DOWN]),
                           'model_version': self.bundle['version'],
                           'as_of': str(meta['date'].iloc[i])}
        return out


async def main():
    config = Config.load()
    config.trading['timeframes'] = []      # fetch nothing; main bot feeds OHLCV
    config.trading['symbols'] = SYMBOLS
    config.trading['initial_capital'] = 10_000.0
    # live paper fills include the half-spread on each side: worst measured
    # symbol (SOL, 0.02%) applied to ALL symbols — conservative by design
    config.execution['slippage_rate'] = 0.0005 + max(SPREADS.values()) / 2

    main_db = DatabaseManager('./data/trading_system.db')
    lab_db = DatabaseManager('./data/fastlab.db')
    notifier = Notifier(db=lab_db)
    engine = TradingEngine(config, NoopModels(), lab_db, MetricsCollector(),
                           notifier=notifier)
    ml = FastML(main_db, lab_db)

    notifier.alert('system', 'info', 'Fast Lab started',
                   'paper-only learning accelerator; kill date 2026-08-07',
                   dedupe_key=f'fastlab_start_{datetime.now():%Y%m%d%H%M}')

    from src.trading import kill_rule
    last_retrain = 0.0
    last_kill_check = 0.0
    while True:
        try:
            # pre-registered kill rule: checked daily, engages automatically
            if time.time() - last_kill_check > 86400 or last_kill_check == 0:
                kill_rule.check_and_engage(lab_db, notifier)
                last_kill_check = time.time()

            # daily-capped retrain
            if time.time() - last_retrain > RETRAIN_INTERVAL_H * 3600 \
                    or ml.bundle is None:
                decision = await asyncio.to_thread(ml.retrain)
                last_retrain = time.time()
                notifier.alert('ml', 'info', f'Fast Lab retrain: {decision}',
                               f'champion {ml.bundle["version"]}',
                               dedupe_key=f'fastlab_retrain_{ml.bundle["version"]}')

            await engine._refresh_prices()
            predictions = await asyncio.to_thread(ml.predict)
            executed = {}
            for symbol, p in predictions.items():
                action = 'BUY' if p['pred'] == 'UP' else 'SELL'
                engine._handle_signal({
                    'symbol': symbol, 'action': action, 'size': 0.10,
                    'confidence': 0.99, 'stop_loss': None, 'take_profit': None,
                    'metadata': {'strategy': 'fastlab_ml',
                                 'p_up': p['p_up'],
                                 'model_version': p['model_version']}})
                executed[symbol] = (
                    (action == 'BUY') == (symbol in engine.positions))
            if predictions:
                with lab_db.get_connection() as conn:
                    for symbol, p in predictions.items():
                        conn.execute("""
                            INSERT INTO ml_predictions
                            (timestamp, symbol, pred, p_up, p_down,
                             model_version, executed)
                            VALUES (?, ?, ?, ?, ?, ?, ?)
                        """, (int(time.time()), symbol, p['pred'], p['p_up'],
                              p['p_down'], p['model_version'],
                              1 if executed.get(symbol) else 0))
                    conn.commit()

            engine._process_pending_orders()
            engine._check_stops()
            engine._record_equity()
            engine._persist_cash()

        except Exception as e:
            logger.exception(f"fastlab loop error: {e}")
        await asyncio.sleep(DECISION_INTERVAL)


if __name__ == '__main__':
    asyncio.run(main())
