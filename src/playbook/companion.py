"""
Playbook Companion core logic — the one piece of this repository that
serves real money, so it is deliberately boring:

  * REMINDER: once a day, check BTC's daily close against its 200-day
    moving average from the verified local data pipeline. One scheduled
    message on the DCA day (the 1st); an immediate message only on a
    regime FLIP; a loud message when data is stale or a check was missed.
    Nothing else, ever.
  * LEDGER: record the buys the human actually makes; compute cost basis,
    invested vs value, and an honest lump-sum comparison.

HARD GUARANTEES (enforced by tests/test_playbook.py, including a source
scan): this module never imports an exchange client, never reads an API
key, never places, sizes, or suggests orders beyond the fixed schedule.
"""

import json
import logging
from datetime import date, datetime, timedelta, timezone
from pathlib import Path
from typing import Dict, List, Optional

import pandas as pd
from sqlalchemy import text

logger = logging.getLogger(__name__)

TREND_WINDOW = 200            # the 200-day rule, exactly as in PLAYBOOK.md
DCA_DAY = 1                   # buy day: 1st of the month
LATE_WINDOW_DAYS = 7          # a missed DCA-day message may be sent late up to this
STALE_HOURS = 3.0             # newest 1h bar older than this => data stale
DIGEST_DIR = Path('docs/digests')
CHART_LINK = ("https://www.tradingview.com/chart/?symbol=BINANCE%3ABTCUSDT "
              "(add indicator 'Moving Average', length 200, on the 1D chart)")


def should_generate_digest(now: datetime, digest_dir: Path = DIGEST_DIR) -> bool:
    """Monthly digest trigger, with the same late-recovery window as a missed
    DCA-day message: fire on the 1st, and catch up within the first
    LATE_WINDOW_DAYS days if the month's digest file does not exist (the host
    may sleep through the entire 1st)."""
    if now.day == 1:
        return True
    if now.day <= LATE_WINDOW_DAYS:
        return not (digest_dir / f"{now.strftime('%Y-%m')}.md").exists()
    return False


# ── State (small key/value + ledger, in the companion's own DB) ─────────────

def ensure_tables(db):
    with db.get_connection() as conn:
        conn.execute("""
            CREATE TABLE IF NOT EXISTS dca_log (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                buy_date TEXT NOT NULL,
                amount_usd REAL NOT NULL,
                btc_price REAL NOT NULL,
                btc_amount REAL NOT NULL,
                note TEXT,
                created_at INTEGER DEFAULT (strftime('%s','now'))
            )""")
        conn.commit()


def get_state(db, key: str) -> Optional[str]:
    with db.engine.connect() as conn:
        row = conn.execute(text(
            "SELECT config_value FROM system_config WHERE config_key=:k"),
            {'k': f'playbook_{key}'}).fetchone()
    return row[0] if row else None


def set_state(db, key: str, value: str):
    with db.engine.connect() as conn:
        conn.execute(text("""
            INSERT INTO system_config
            (config_key, config_value, config_type, description, updated_at)
            VALUES (:k, :v, 'string', 'playbook companion state', :ts)
            ON CONFLICT(config_key) DO UPDATE
            SET config_value=:v, updated_at=:ts
        """), {'k': f'playbook_{key}', 'v': value,
               'ts': int(datetime.now(timezone.utc).timestamp())})
        conn.commit()


# ── Market data (read-only, from the verified pipeline) ─────────────────────

def btc_daily_closes(main_db) -> pd.Series:
    """Completed daily closes resampled from stored 1h bars."""
    h1 = main_db.get_ohlcv_data('BTC/USDT', '1h')
    h1 = h1.set_index('timestamp').sort_index()
    daily = h1['close'].resample('1D').last().dropna()
    today = pd.Timestamp.now(tz=timezone.utc).normalize().tz_localize(None)
    return daily[daily.index < today]


def data_staleness_hours(main_db) -> float:
    latest = main_db.get_latest_ohlcv_timestamp('BTC/USDT', '1h')
    if latest is None:
        return float('inf')
    return (datetime.now(timezone.utc).timestamp() - latest) / 3600


def check_regime(daily_closes: pd.Series) -> Optional[Dict]:
    """{'above': bool, 'close', 'sma200', 'distance_pct', 'as_of'} or None
    if there is not enough history to compute the rule."""
    if len(daily_closes) < TREND_WINDOW:
        return None
    sma = float(daily_closes.tail(TREND_WINDOW).mean())
    close = float(daily_closes.iloc[-1])
    return {'above': close > sma, 'close': close, 'sma200': sma,
            'distance_pct': close / sma - 1,
            'as_of': daily_closes.index[-1].date().isoformat()}


# ── The daily decision (pure logic; the service supplies db/notifier) ───────

def run_daily_check(main_db, state_db, notifier,
                    today: Optional[date] = None) -> List[str]:
    """One tick per day. Returns the list of message kinds sent (for tests).
    Never silently skips: staleness and missed runs are themselves alerts."""
    today = today or datetime.now(timezone.utc).date()
    sent: List[str] = []

    # missed-run detection (the companion itself was down)
    last_run = get_state(state_db, 'last_daily_run')
    if last_run:
        gap = (today - date.fromisoformat(last_run)).days
        if gap > 1:
            notifier.alert('playbook', 'warning',
                           'Playbook companion missed check-ins',
                           f'No daily check ran for {gap - 1} day(s) '
                           f'(service was down). Verify manually if in '
                           f'doubt: {CHART_LINK}',
                           dedupe_key=f'playbook_gap_{today}')
            sent.append('missed_runs')
    set_state(state_db, 'last_daily_run', today.isoformat())

    # data freshness — never guess on stale data
    stale_h = data_staleness_hours(main_db)
    if stale_h > STALE_HOURS:
        notifier.alert('playbook', 'warning',
                       'Playbook check: data stale — verify manually',
                       f'Newest local BTC data is {stale_h:.1f}h old, so the '
                       f'200-day check was NOT computed. Check the chart '
                       f'yourself: {CHART_LINK}',
                       dedupe_key=f'playbook_stale_{today}')
        sent.append('stale')
        return sent

    regime = check_regime(btc_daily_closes(main_db))
    if regime is None:
        notifier.alert('playbook', 'warning',
                       'Playbook check: not enough history',
                       f'Fewer than {TREND_WINDOW} daily closes available — '
                       f'verify manually: {CHART_LINK}',
                       dedupe_key=f'playbook_short_{today}')
        sent.append('short_history')
        return sent

    now_side = 'above' if regime['above'] else 'below'
    prev_side = get_state(state_db, 'regime')

    # regime FLIP -> immediate message (the only unscheduled message)
    if prev_side and now_side != prev_side:
        if now_side == 'below':
            set_state(state_db, 'pause_started', today.isoformat())
            message = (f"BTC closed BELOW its 200-day average "
                       f"(${regime['close']:,.0f} vs ${regime['sma200']:,.0f}, "
                       f"{regime['distance_pct']:+.1%}). Per the playbook: "
                       f"pause scheduled buys until it closes back above. "
                       f"No action needed today.")
        else:
            paused_since = get_state(state_db, 'pause_started')
            pause_note = ''
            if paused_since:
                days = (today - date.fromisoformat(paused_since)).days
                pause_note = f" The pause lasted {days} days."
            message = (f"BTC closed back ABOVE its 200-day average "
                       f"(${regime['close']:,.0f} vs ${regime['sma200']:,.0f})."
                       f"{pause_note} Per the playbook: resume scheduled "
                       f"buys, including any skipped amounts, on the next "
                       f"buy day.")
        notifier.alert('playbook', 'info', f'Trend flip: {now_side} 200-day',
                       message, dedupe_key=f'playbook_flip_{today}')
        sent.append(f'flip_{now_side}')
    if not prev_side:
        set_state(state_db, 'regime_since', today.isoformat())
    elif now_side != prev_side:
        set_state(state_db, 'regime_since', today.isoformat())
    set_state(state_db, 'regime', now_side)

    # DCA-day message (day 1; late-recovery within LATE_WINDOW_DAYS)
    this_month = today.strftime('%Y-%m')
    already = get_state(state_db, 'last_dca_msg_month') == this_month
    is_dca_day = today.day == DCA_DAY
    is_late_window = DCA_DAY < today.day <= DCA_DAY + LATE_WINDOW_DAYS

    if (is_dca_day or is_late_window) and not already:
        late = '' if is_dca_day else \
            f' (LATE — the companion could not send this on the 1st)'
        if regime['above']:
            message = (f"DCA day{late}: BTC is ABOVE its 200-day average "
                       f"(${regime['close']:,.0f} vs ${regime['sma200']:,.0f}, "
                       f"{regime['distance_pct']:+.1%}). Execute your "
                       f"scheduled buy, then log it: "
                       f"python scripts/playbook_log.py add --amount <usd>")
        else:
            paused_since = get_state(state_db, 'pause_started')
            days = (today - date.fromisoformat(paused_since)).days \
                if paused_since else 0
            message = (f"DCA day{late}: BTC is BELOW its 200-day average "
                       f"(${regime['close']:,.0f} vs ${regime['sma200']:,.0f}). "
                       f"Per the playbook: skip this buy and hold the cash — "
                       f"the pause has run {days} days. Add the skipped "
                       f"amount to your next buy when the trend recovers.")
        notifier.alert('playbook', 'info',
                       'Playbook: scheduled buy day', message,
                       dedupe_key=f'playbook_dca_{this_month}')
        set_state(state_db, 'last_dca_msg_month', this_month)
        sent.append('dca_late' if is_late_window else 'dca')

    return sent


# ── Ledger ───────────────────────────────────────────────────────────────────

def log_buy(state_db, main_db, amount_usd: float,
            price: Optional[float] = None,
            buy_date: Optional[date] = None, note: str = '') -> Dict:
    ensure_tables(state_db)
    buy_date = buy_date or datetime.now(timezone.utc).date()
    if price is None:
        closes = btc_daily_closes(main_db)
        at = closes[closes.index.date <= buy_date]
        if at.empty:
            raise ValueError('no price data on/before that date; pass --price')
        price = float(at.iloc[-1])
    btc = amount_usd / price
    with state_db.get_connection() as conn:
        conn.execute(
            "INSERT INTO dca_log (buy_date, amount_usd, btc_price, "
            "btc_amount, note) VALUES (?, ?, ?, ?, ?)",
            (buy_date.isoformat(), amount_usd, price, btc, note))
        conn.commit()
    return {'buy_date': buy_date.isoformat(), 'amount_usd': amount_usd,
            'btc_price': price, 'btc_amount': btc}


def ledger_summary(state_db, main_db) -> Dict:
    ensure_tables(state_db)
    log = pd.read_sql_query(
        "SELECT * FROM dca_log ORDER BY buy_date", state_db.engine)
    closes = btc_daily_closes(main_db)
    latest = float(closes.iloc[-1]) if not closes.empty else None
    out = {'log': log, 'latest_price': latest}
    if log.empty or latest is None:
        return out

    invested = float(log['amount_usd'].sum())
    btc_total = float(log['btc_amount'].sum())
    out.update({
        'invested': invested,
        'btc_total': btc_total,
        'cost_basis': invested / btc_total if btc_total else None,
        'current_value': btc_total * latest,
    })
    # honest lump-sum comparison: everything on the first logged day
    first = log['buy_date'].iloc[0]
    first_prices = closes[closes.index.date <= date.fromisoformat(first)]
    if not first_prices.empty:
        p0 = float(first_prices.iloc[-1])
        out['lump_sum_value'] = invested / p0 * latest
        out['lump_sum_price'] = p0
    return out
