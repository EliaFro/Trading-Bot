"""
Self-executing kill rule for the Fast Lab's strategy-search role.

Pre-registered (docs/FASTLAB_PLAN.md Part E, OPERATING.md §3b, locked
2026-07-10): if by 2026-08-07 no Fast Lab configuration has cleared the
multiple-testing-corrected significance bar AFTER full costs, strategy
search at the 1m/5m horizon closes PERMANENTLY.

Design properties (each tested in tests/test_kill_rule.py):
  * The deadline is a hard-coded constant. There is no config flag, no
    environment variable, and no function parameter that bypasses the
    lockout — re-enabling requires editing this file, i.e. a deliberate,
    reviewable code change.
  * The lockout persists in TWO places (a sentinel file and the Fast Lab
    database) and engages if EITHER exists, so it survives restarts,
    database resets, and partial file loss.
  * The evaluation is mechanical: it reads the offline study artifacts
    (Bonferroni survivor lists) and the live paper record, and applies the
    pre-registered criteria. No human judgment in the loop.
  * The closure alert fires exactly once.

The observation role is NOT locked: the existing champion may keep
retraining and predicting (that is the instrument the lab keeps). What
locks is the SEARCH: the study scripts that evaluate new strategy
configurations at this horizon.
"""

import json
import logging
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, Optional

logger = logging.getLogger(__name__)

# ── The pre-registered rule. Editing these constants IS the only override. ──
KILL_DATE = datetime(2026, 8, 7, 0, 0, tzinfo=timezone.utc)
SENTINEL = Path('./models/fastlab/SEARCH_CLOSED')
LOCK_KEY = 'fastlab_search_closed'

# Offline artifacts consulted by the mechanical evaluation
OFFLINE_ARTIFACTS = ('docs/fastlab_partB_metrics.json',
                     'docs/fastlab_partC_metrics.json')

# Live-record bar (the OPERATING.md criteria, mechanically):
# profit factor > 1.15 after full costs AND per-trade t-stat clearing a
# Bonferroni-style bar across everything tested at this horizon
# (8 MTF variants + 3 ML trading families + 1 live config = 12 tests).
LIVE_PF_BAR = 1.15
LIVE_T_BAR_P = 0.05 / 12
LIVE_MIN_TRADES = 100


class SearchClosedError(RuntimeError):
    """Raised when a strategy-search entry point runs after closure."""


def is_closed(lab_db=None) -> bool:
    """Locked if EITHER persistence layer says so."""
    if SENTINEL.exists():
        return True
    if lab_db is not None:
        try:
            from sqlalchemy import text
            with lab_db.engine.connect() as conn:
                row = conn.execute(
                    text("SELECT config_value FROM system_config "
                         "WHERE config_key = :k"), {'k': LOCK_KEY}).fetchone()
            return row is not None
        except Exception as e:
            logger.error(f"kill-rule DB check failed: {e}")
    return False


def assert_search_allowed(lab_db=None):
    """Guard for strategy-search entry points. No bypass parameter exists."""
    if is_closed(lab_db):
        raise SearchClosedError(
            "Strategy search at the 1m/5m horizon is PERMANENTLY CLOSED per "
            "the pre-registered rule of 2026-08-07 (docs/FASTLAB_PLAN.md "
            "Part E). The Fast Lab runs as an observation instrument only. "
            "Re-enabling requires editing src/trading/kill_rule.py — a "
            "deliberate code change, by design.")


def evaluate_criteria(lab_db) -> Dict:
    """Mechanical evaluation: did ANYTHING clear the corrected bar after
    full costs? Returns the evidence dict used in the closure record."""
    from scipy import stats
    import numpy as np

    evidence = {'evaluated_at': datetime.now(timezone.utc).isoformat(),
                'offline': {}, 'live': {}, 'anything_passed': False}

    for path in OFFLINE_ARTIFACTS:
        try:
            data = json.loads(Path(path).read_text())
            survivors = data.get('bonferroni_significant', [])
            evidence['offline'][path] = {
                'bonferroni_survivors': survivors,
                'reality_check_p': data.get('reality_check', {}).get('p')}
            if survivors:
                evidence['anything_passed'] = True
        except FileNotFoundError:
            evidence['offline'][path] = 'missing'

    try:
        import pandas as pd
        trades = pd.read_sql_query(
            "SELECT pnl, pnl_percentage FROM trades WHERE status='CLOSED'",
            lab_db.engine)
        n = len(trades)
        evidence['live']['n_trades'] = int(n)
        if n >= LIVE_MIN_TRADES:
            gw = trades.loc[trades['pnl'] > 0, 'pnl'].sum()
            gl = -trades.loc[trades['pnl'] <= 0, 'pnl'].sum()
            pf = float(gw / gl) if gl > 0 else float('inf')
            nets = trades['pnl_percentage'].dropna().to_numpy()
            t = float(nets.mean() / nets.std(ddof=1) * np.sqrt(len(nets))) \
                if nets.std(ddof=1) > 0 else 0.0
            p = float(stats.t.sf(t, df=len(nets) - 1))
            evidence['live'].update({'profit_factor': pf, 't_stat': t,
                                     'p_one_sided': p})
            if pf > LIVE_PF_BAR and p < LIVE_T_BAR_P:
                evidence['anything_passed'] = True
        else:
            evidence['live']['note'] = (
                f'fewer than {LIVE_MIN_TRADES} closed live trades — '
                f'cannot pass on live record alone')
    except Exception as e:
        evidence['live']['error'] = str(e)

    return evidence


def check_and_engage(lab_db, notifier=None,
                     now: Optional[datetime] = None) -> Optional[Dict]:
    """Call daily. On/after the deadline, evaluates once and — if nothing
    passed — engages the permanent lockout. Returns the closure record when
    the lockout engages on this call, else None."""
    now = now or datetime.now(timezone.utc)
    if now < KILL_DATE or is_closed(lab_db):
        return None

    evidence = evaluate_criteria(lab_db)
    if evidence['anything_passed']:
        # Pre-registered rule only closes on failure; a genuine pass is
        # loudly reported and the deadline check stops without locking.
        logger.critical("KILL RULE: a configuration PASSED the corrected "
                        "bar — closure NOT engaged. Review immediately.")
        if notifier:
            notifier.alert('kill_rule', 'critical',
                           'Fast Lab: a config PASSED the corrected bar',
                           json.dumps(evidence['live'], default=str)[:500],
                           dedupe_key='kill_rule_pass')
        return None

    record = {
        'closed_at': now.isoformat(),
        'rule': 'pre-registered 2026-08-07, docs/FASTLAB_PLAN.md Part E',
        'evidence': evidence,
    }

    # Engage: both persistence layers
    SENTINEL.parent.mkdir(parents=True, exist_ok=True)
    SENTINEL.write_text(json.dumps(record, indent=2, default=str))
    try:
        from sqlalchemy import text
        with lab_db.engine.connect() as conn:
            conn.execute(text("""
                INSERT INTO system_config
                (config_key, config_value, config_type, description, updated_at)
                VALUES (:k, :v, 'string', 'Permanent search closure', :ts)
                ON CONFLICT(config_key) DO NOTHING
            """), {'k': LOCK_KEY,
                   'v': json.dumps(record, default=str)[:2000],
                   'ts': int(now.timestamp())})
            conn.commit()
    except Exception as e:
        logger.error(f"kill-rule DB persist failed (sentinel still holds): {e}")

    logger.critical(
        "FAST LAB STRATEGY SEARCH PERMANENTLY CLOSED per the pre-registered "
        f"rule. Evidence: offline Bonferroni survivors = none; live record "
        f"= {evidence['live']}. Observation role continues.")
    if notifier:
        notifier.alert(
            'kill_rule', 'info',
            'Fast Lab strategy search closed (pre-registered rule)',
            'The 2026-08-07 deadline passed with nothing clearing the '
            'corrected bar after full costs. Search at this horizon is now '
            'permanently closed; the lab continues as an observation '
            'instrument. Full results: docs/FASTLAB_RESULTS.md',
            dedupe_key='kill_rule_closure')
    return record


def closure_record() -> Optional[Dict]:
    """The persisted closure record, for the dashboard banner."""
    if SENTINEL.exists():
        try:
            return json.loads(SENTINEL.read_text())
        except Exception:
            return {'closed_at': 'unknown (sentinel unreadable)'}
    return None
