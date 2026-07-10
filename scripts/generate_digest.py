#!/usr/bin/env python3
"""
Monthly Evidence Digest — one honest page covering both labs.

Generated on the 1st of each month by the playbook companion (and on
demand: `python scripts/generate_digest.py`). Markdown lands in
docs/digests/, is viewable on the 📘 dashboard tab, and a summary goes to
Telegram with the file path.

Also owns the SENTIMENT COUNTDOWN: sentiment history collection began at
launch; at the 6-month mark the one-time out-of-sample evaluation
(scripts/run_sentiment_eval.py) runs ONCE through the standard harness and
its verdict is folded into that month's digest.
"""

import json
import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path

import numpy as np
import pandas as pd
from sqlalchemy import text

sys.path.append(str(Path(__file__).resolve().parent.parent))

from dotenv import load_dotenv
load_dotenv()

from src.utils.database import DatabaseManager
from src.ml.dataset import evidence_tier

DIGEST_DIR = Path('docs/digests')
SENTIMENT_EVAL_DAYS = 180


def _q(db, sql, params=None):
    try:
        return pd.read_sql_query(text(sql), db.engine, params=params or {})
    except Exception:
        return pd.DataFrame()


def _operating_criteria(main_db) -> list:
    """The five OPERATING.md §3 banner criteria, current vs required,
    measured on live paper ML decisions only."""
    rows = []
    trades = _q(main_db, "SELECT pnl, pnl_percentage, entry_time FROM trades "
                         "WHERE status='CLOSED' AND strategy='ml_core'")
    n = len(trades)
    rows.append(('≥100 completed live ML trades', f'{n}', n >= 100))

    if n >= 5:
        gw = trades.loc[trades['pnl'] > 0, 'pnl'].sum()
        gl = -trades.loc[trades['pnl'] <= 0, 'pnl'].sum()
        pf = gw / gl if gl > 0 else float('inf')
        rows.append(('profit factor > 1.15 after fees', f'{pf:.2f}', pf > 1.15))
    else:
        rows.append(('profit factor > 1.15 after fees', 'n/a (too few trades)', False))

    equity = _q(main_db, "SELECT total_equity FROM performance_tracking "
                         "WHERE mode='paper' ORDER BY timestamp")
    if len(equity) > 10:
        eq = equity['total_equity']
        dd = float(((eq - eq.cummax()) / eq.cummax()).min())
        rows.append(('max drawdown < 15%', f'{dd:.1%}', abs(dd) < 0.15))
    else:
        rows.append(('max drawdown < 15%', 'n/a', False))

    rows.append(('beats TSMOM-60d on identical live period',
                 'n/a (needs the full live record)', False))
    rows.append(('t-stat ≥ 2.0 + val acc ≥ chance+5pts, <10% thin-feature '
                 'importance', 'n/a (accrues with retrains)', False))
    return rows


def _gauge_trend(db, since: datetime) -> str:
    log = _q(db, "SELECT timestamp, new_is_bal_acc, new_val_bal_acc, decision "
                 "FROM ml_retrain_log WHERE timestamp >= :ts ORDER BY timestamp",
             {'ts': int(since.timestamp())})
    if log.empty:
        return "no retrains this period"
    gaps = log['new_is_bal_acc'] - log['new_val_bal_acc']
    trend = ('narrowing' if len(gaps) > 1 and gaps.iloc[-1] < gaps.iloc[0]
             else 'widening' if len(gaps) > 1 and gaps.iloc[-1] > gaps.iloc[0]
             else 'flat')
    verdict = ('memorized noise' if gaps.iloc[-1] > 0.15
               else 'watch' if gaps.iloc[-1] > 0.05 else 'healthy')
    return (f"{len(log)} retrain(s); IS−OOS gap {gaps.iloc[0]:+.3f} → "
            f"{gaps.iloc[-1]:+.3f} ({trend}); latest verdict: **{verdict}**")


def _tier_migration(db) -> str:
    """Evidence tiers move as the data span grows; report any changes since
    the last digest snapshot."""
    row = _q(db, "SELECT MIN(timestamp) m FROM ohlcv WHERE symbol='BTC/USDT' "
                 "AND timeframe='1h'")
    if row.empty or pd.isna(row['m'].iloc[0]):
        return "no data"
    span_days = int((datetime.now(timezone.utc).timestamp()
                     - row['m'].iloc[0]) / 86400)

    from src.ml.dataset import FEATURE_TIMESCALE_DAYS
    current = {f: evidence_tier(f, span_days) for f in FEATURE_TIMESCALE_DAYS}
    snap_path = DIGEST_DIR / '.tier_snapshot.json'
    previous = json.loads(snap_path.read_text()) if snap_path.exists() else {}
    snap_path.parent.mkdir(parents=True, exist_ok=True)
    snap_path.write_text(json.dumps(current))

    changed = [f"`{f}`: {previous[f]} → **{current[f]}**"
               for f in current if f in previous and previous[f] != current[f]]
    thin = [f for f, t in current.items() if t.startswith('thin')]
    out = f"data span {span_days}d; {len(thin)} features still thin-evidence"
    if changed:
        out += "; **tier changes:** " + ", ".join(changed)
    else:
        out += "; no tier changes this period"
    return out


def _retrain_summary(db, since, label) -> str:
    log = _q(db, "SELECT decision, reason FROM ml_retrain_log "
                 "WHERE timestamp >= :ts", {'ts': int(since.timestamp())})
    if log.empty:
        return f"{label}: no retrains"
    kept = (log['decision'] == 'KEPT_OLD').sum()
    replaced = log[log['decision'] == 'REPLACED']
    out = (f"{label}: {len(log)} retrain(s) — {kept} kept, "
           f"{len(replaced)} replaced, "
           f"{(log['decision'] == 'INITIAL').sum()} initial")
    for _, r in replaced.iterrows():
        out += f"\n  - REPLACED: {r['reason']}"
    return out


def _equity_line(db, mode, label) -> str:
    eq = _q(db, "SELECT timestamp, total_equity, benchmark_price FROM "
                "performance_tracking WHERE mode=:m ORDER BY timestamp",
            {'m': mode})
    if len(eq) < 2:
        return f"{label}: no equity history"
    ret = eq['total_equity'].iloc[-1] / eq['total_equity'].iloc[0] - 1
    bench = eq.dropna(subset=['benchmark_price'])
    btc = (bench['benchmark_price'].iloc[-1] / bench['benchmark_price'].iloc[0]
           - 1) if len(bench) > 1 else None
    out = (f"{label} (PAPER): ${eq['total_equity'].iloc[-1]:,.2f} "
           f"({ret:+.2%} since start)")
    if btc is not None:
        out += f" vs hold-BTC {btc:+.2%} over the same span"
    return out


def _sentiment_countdown(main_db) -> str:
    row = _q(main_db, "SELECT MIN(timestamp) m FROM sentiment_scores")
    if row.empty or pd.isna(row['m'].iloc[0]):
        return "sentiment collection has not started"
    start = datetime.fromtimestamp(int(row['m'].iloc[0]), tz=timezone.utc)
    eval_date = start + timedelta(days=SENTIMENT_EVAL_DAYS)
    remaining = (eval_date - datetime.now(timezone.utc)).days

    marker = DIGEST_DIR / '.sentiment_eval_done.json'
    if marker.exists():
        result = json.loads(marker.read_text())
        return (f"one-time evaluation COMPLETED {result.get('date', '?')}: "
                f"{result.get('verdict', 'see docs/SENTIMENT_EVAL.md')}")
    if remaining > 0:
        return (f"collection began {start:%Y-%m-%d}; one-time OOS evaluation "
                f"due **{eval_date:%Y-%m-%d}** ({remaining} days remaining)")
    # due: run the one-time evaluation through the standard harness
    try:
        from scripts.run_sentiment_eval import run_once
        verdict = run_once()
        marker.write_text(json.dumps(
            {'date': datetime.now(timezone.utc).date().isoformat(),
             'verdict': verdict}))
        return f"one-time evaluation RAN THIS MONTH: {verdict}"
    except Exception as e:
        return (f"evaluation DUE but failed to run ({e}) — run manually: "
                f"python scripts/run_sentiment_eval.py")


def generate_digest(notifier=None) -> Path:
    now = datetime.now(timezone.utc)
    month_ago = now - timedelta(days=31)
    DIGEST_DIR.mkdir(parents=True, exist_ok=True)

    main_db = DatabaseManager('./data/trading_system.db')
    lab_db = (DatabaseManager('./data/fastlab.db')
              if Path('data/fastlab.db').exists() else None)

    from src.trading.kill_rule import closure_record, KILL_DATE
    closure = closure_record()
    if closure:
        kill_line = (f"**CLOSED** since {closure.get('closed_at', '?')[:10]} "
                     f"per the pre-registered rule — observation only")
    else:
        kill_line = (f"{max((KILL_DATE - now).days, 0)} days until the "
                     f"2026-08-07 self-executing deadline")

    criteria = _operating_criteria(main_db)
    criteria_md = "\n".join(
        f"| {name} | {value} | {'✅' if ok else '✗'} |"
        for name, value, ok in criteria)

    body = f"""# Evidence Digest — {now:%Y-%m}

*Auto-generated {now:%Y-%m-%d %H:%M} UTC. Paper money throughout — nothing here is investment performance.*

## Banner statuses
- **🧪 Daily ML Lab:** "ML has not demonstrated an edge over simple momentum — status: learning" (unchanged; changes only via the OPERATING.md §3 bar)
- **⚡ Fast Lab:** learning accelerator, never a profit path; kill clock: {kill_line}
- **📘 Playbook:** the only real-money plan — manual DCA + 200-day rule

## The banner bar (OPERATING.md §3), current vs required
| criterion | current | met |
|---|---|---|
{criteria_md}

## Overfitting gauge (this month)
- Daily lab: {_gauge_trend(main_db, month_ago)}
- Fast lab: {_gauge_trend(lab_db, month_ago) if lab_db else 'n/a'}

## Evidence-tier migration
{_tier_migration(main_db)}

## Retrain guard
{_retrain_summary(main_db, month_ago, 'Daily lab')}
{_retrain_summary(lab_db, month_ago, 'Fast lab') if lab_db else ''}

## Paper equity (clearly labeled: PAPER)
- {_equity_line(main_db, 'paper', 'Daily ML lab')}
- {_equity_line(lab_db, 'paper', 'Fast lab') if lab_db else 'Fast lab: n/a'}

## Sentiment evaluation countdown
{_sentiment_countdown(main_db)}

---
*Standing verdicts: intraday dead (fees) · published TA indistinguishable from luck after correction · ML skill real but ~25× below the cost floor · full evidence: PHASE2_RESULTS.md, SIGNAL_LIBRARY_RESULTS.md, ML_RESULTS.md, FASTLAB_RESULTS.md.*
"""
    path = DIGEST_DIR / f"{now:%Y-%m}.md"
    path.write_text(body)

    if notifier:
        met = sum(1 for _, _, ok in criteria if ok)
        notifier.alert(
            'digest', 'info', f'Monthly evidence digest {now:%Y-%m}',
            f'Banner bar: {met}/5 criteria met. Kill clock: {kill_line}. '
            f'Full digest: docs/digests/{now:%Y-%m}.md (also on the 📘 tab).',
            dedupe_key=f'digest_{now:%Y-%m}')
    return path


if __name__ == '__main__':
    print(f"digest -> {generate_digest()}")
