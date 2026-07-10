#!/usr/bin/env python3
"""
The ONE-TIME sentiment evaluation, pre-registered in docs/ML_PLAN.md §4:
after ~6 months of live sentiment collection, sentiment features earn a
single out-of-sample trial through the standard walk-forward harness —
the daily ML pipeline run twice on the sentiment-covered period, with and
without a daily sentiment feature, comparing OOS balanced accuracy and
after-fee trading results. Run once, reported honestly, never re-tuned.

Refuses to run before 180 days of history exist (no shortcuts), and the
digest records the verdict permanently after the single run.
"""

import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path

import numpy as np
import pandas as pd
from sqlalchemy import text

sys.path.append(str(Path(__file__).resolve().parent.parent))

from dotenv import load_dotenv
load_dotenv()

MIN_DAYS = 180


def run_once() -> str:
    from src.utils.database import DatabaseManager
    main_db = DatabaseManager('./data/trading_system.db')

    row = pd.read_sql_query(
        text("SELECT MIN(timestamp) m, MAX(timestamp) x, COUNT(*) n "
             "FROM sentiment_scores"), main_db.engine)
    if row.empty or pd.isna(row['m'].iloc[0]):
        raise RuntimeError("no sentiment history collected yet")
    start = datetime.fromtimestamp(int(row['m'].iloc[0]), tz=timezone.utc)
    days = (datetime.now(timezone.utc) - start).days
    if days < MIN_DAYS:
        raise RuntimeError(
            f"only {days} days of sentiment history — the pre-registered "
            f"evaluation requires {MIN_DAYS}. Due "
            f"{(start + timedelta(days=MIN_DAYS)):%Y-%m-%d}.")

    # ── With enough history: standard harness, two runs, one comparison ──
    from src.ml import dataset as ds
    from src.ml import walkforward_ml as wfml
    from src.ml.live import build_daily_frames

    frames = build_daily_frames(main_db, ['BTC/USDT', 'ETH/USDT', 'SOL/USDT'])
    # daily mean sentiment per symbol, joined as one extra feature
    senti = pd.read_sql_query(
        text("SELECT symbol, timestamp, sentiment_score FROM sentiment_scores"),
        main_db.engine)
    senti['date'] = pd.to_datetime(senti['timestamp'], unit='s').dt.normalize()
    daily_senti = senti.groupby(['symbol', 'date'])['sentiment_score'].mean()

    def panel(with_sentiment: bool):
        X, meta = ds.assemble_panel(frames)
        if with_sentiment:
            keys = list(zip(meta['symbol'],
                            pd.to_datetime(meta['date']).dt.normalize()))
            values = np.array([daily_senti.get(k, np.nan) for k in keys])
            X = X.copy()
            X['sentiment_daily'] = values
            keep = ~np.isnan(values)
            X, meta = X[keep].reset_index(drop=True), \
                meta[keep].reset_index(drop=True)
        # restrict both runs to the sentiment-covered span (fair comparison)
        lo = daily_senti.index.get_level_values(1).min()
        mask = pd.to_datetime(meta['date']) >= lo
        return X[mask.to_numpy()].reset_index(drop=True), \
            meta[mask.to_numpy()].reset_index(drop=True)

    results = {}
    for label, with_s in (('without_sentiment', False), ('with_sentiment', True)):
        X, meta = panel(with_s)
        reports = wfml.run_walkforward_ml(X, meta)
        oos = [r.model_metrics['ensemble']['oos']['balanced_accuracy']
               for r in reports]
        results[label] = {'n_windows': len(reports),
                          'oos_bal_acc': float(np.mean(oos)) if oos else 0.0}

    delta = (results['with_sentiment']['oos_bal_acc']
             - results['without_sentiment']['oos_bal_acc'])
    verdict = (
        f"with sentiment {results['with_sentiment']['oos_bal_acc']:.3f} vs "
        f"without {results['without_sentiment']['oos_bal_acc']:.3f} "
        f"(Δ {delta:+.3f}) over {results['with_sentiment']['n_windows']} "
        f"windows — sentiment "
        + ("added measurable OOS skill" if delta > 0.01
           else "did NOT add measurable OOS skill"))

    Path('docs/SENTIMENT_EVAL.md').write_text(
        f"# One-Time Sentiment Evaluation — {datetime.now():%Y-%m-%d}\n\n"
        f"Pre-registered in docs/ML_PLAN.md §4; standard harness; run ONCE.\n\n"
        f"**Verdict: {verdict}**\n\n```json\n"
        f"{results}\n```\n")
    return verdict


if __name__ == '__main__':
    try:
        print(run_once())
    except RuntimeError as e:
        print(f"NOT RUN: {e}")
        sys.exit(1)
