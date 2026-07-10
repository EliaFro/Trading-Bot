"""Playbook Companion: the code that serves real money gets the strictest
tests — message contract, staleness honesty, ledger math, and the
never-trades guarantee enforced as a source scan."""

from datetime import date
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from src.playbook import companion as pc


class FakeNotifier:
    def __init__(self):
        self.alerts = []

    def alert(self, alert_type, severity, title, message, **kwargs):
        self.alerts.append({'type': alert_type, 'title': title,
                            'message': message})
        return True


class FakeMainDB:
    """Serves a controllable daily-close series through the same interface."""

    def __init__(self, closes: pd.Series, stale_hours: float = 0.5):
        self._closes = closes
        self._stale = stale_hours

    def get_ohlcv_data(self, symbol, tf, *a, **k):
        # 1h frame whose daily resample reproduces self._closes
        idx = pd.DatetimeIndex([d + pd.Timedelta(hours=12)
                                for d in self._closes.index])
        return pd.DataFrame({'timestamp': idx,
                             'open': self._closes.values,
                             'high': self._closes.values,
                             'low': self._closes.values,
                             'close': self._closes.values,
                             'volume': 1.0})

    def get_latest_ohlcv_timestamp(self, symbol, tf):
        import time
        return int(time.time() - self._stale * 3600)


def daily_series(n=300, uptrend=True, seed=1):
    rng = np.random.default_rng(seed)
    drift = 0.002 if uptrend else -0.002
    closes = 50_000 * np.exp(np.cumsum(rng.normal(drift, 0.01, n)))
    # index of COMPLETED days strictly before today
    end = pd.Timestamp.utcnow().normalize().tz_localize(None) - pd.Timedelta(days=1)
    idx = pd.date_range(end=end, periods=n, freq='1D')
    return pd.Series(closes, index=idx)


# ── Regime math ──────────────────────────────────────────────────────────────

def test_regime_above_and_below():
    up = pc.check_regime(daily_series(uptrend=True))
    assert up['above'] is True and up['distance_pct'] > 0
    down = pc.check_regime(daily_series(uptrend=False))
    assert down['above'] is False and down['distance_pct'] < 0


def test_regime_requires_history():
    assert pc.check_regime(daily_series(n=150)) is None


# ── Daily message contract ───────────────────────────────────────────────────

def test_dca_day_message_above(tmp_db):
    notifier = FakeNotifier()
    main_db = FakeMainDB(daily_series(uptrend=True))
    sent = pc.run_daily_check(main_db, tmp_db, notifier,
                              today=date(2026, 8, 1))
    assert 'dca' in sent
    assert any('Execute your scheduled buy' in a['message']
               for a in notifier.alerts)


def test_dca_day_message_below_reports_pause_length(tmp_db):
    notifier = FakeNotifier()
    main_db = FakeMainDB(daily_series(uptrend=False))
    pc.set_state(tmp_db, 'regime', 'below')
    pc.set_state(tmp_db, 'pause_started', '2026-07-10')
    sent = pc.run_daily_check(main_db, tmp_db, notifier,
                              today=date(2026, 8, 1))
    assert 'dca' in sent
    message = notifier.alerts[-1]['message']
    assert 'skip this buy' in message and '22 days' in message


def test_non_dca_day_sends_nothing_when_no_flip(tmp_db):
    notifier = FakeNotifier()
    main_db = FakeMainDB(daily_series(uptrend=True))
    pc.set_state(tmp_db, 'regime', 'above')
    pc.set_state(tmp_db, 'last_daily_run', '2026-08-14')
    sent = pc.run_daily_check(main_db, tmp_db, notifier,
                              today=date(2026, 8, 15))
    assert sent == []                      # low-noise by design
    assert notifier.alerts == []


def test_flip_sends_immediate_message(tmp_db):
    notifier = FakeNotifier()
    main_db = FakeMainDB(daily_series(uptrend=False))
    pc.set_state(tmp_db, 'regime', 'above')
    pc.set_state(tmp_db, 'last_daily_run', '2026-08-14')
    sent = pc.run_daily_check(main_db, tmp_db, notifier,
                              today=date(2026, 8, 15))
    assert sent == ['flip_below']
    assert 'pause scheduled buys' in notifier.alerts[0]['message']
    # flip back reports pause length
    main_db2 = FakeMainDB(daily_series(uptrend=True))
    pc.set_state(tmp_db, 'last_daily_run', '2026-08-24')
    sent2 = pc.run_daily_check(main_db2, tmp_db, notifier,
                               today=date(2026, 8, 25))
    assert sent2 == ['flip_above']
    assert 'lasted 10 days' in notifier.alerts[-1]['message']


def test_missed_dca_day_recovers_late_and_flags_it(tmp_db):
    notifier = FakeNotifier()
    main_db = FakeMainDB(daily_series(uptrend=True))
    pc.set_state(tmp_db, 'regime', 'above')
    sent = pc.run_daily_check(main_db, tmp_db, notifier,
                              today=date(2026, 8, 4))
    assert 'dca_late' in sent
    assert 'LATE' in notifier.alerts[-1]['message']
    # and never twice in the same month
    pc.set_state(tmp_db, 'last_daily_run', '2026-08-04')
    sent2 = pc.run_daily_check(main_db, tmp_db, notifier,
                               today=date(2026, 8, 5))
    assert 'dca' not in sent2 and 'dca_late' not in sent2


def test_stale_data_says_so_and_computes_nothing(tmp_db):
    notifier = FakeNotifier()
    main_db = FakeMainDB(daily_series(), stale_hours=9)
    sent = pc.run_daily_check(main_db, tmp_db, notifier,
                              today=date(2026, 8, 1))
    assert sent == ['stale']
    message = notifier.alerts[0]['message']
    assert 'verify manually' in message.lower() or 'Check the chart' in message
    assert 'tradingview' in message.lower()
    # no DCA message was fabricated on stale data
    assert not any('scheduled buy day' in a['title'] for a in notifier.alerts)


def test_service_downtime_is_itself_an_alert(tmp_db):
    notifier = FakeNotifier()
    main_db = FakeMainDB(daily_series(uptrend=True))
    pc.set_state(tmp_db, 'regime', 'above')
    pc.set_state(tmp_db, 'last_daily_run', '2026-08-10')
    sent = pc.run_daily_check(main_db, tmp_db, notifier,
                              today=date(2026, 8, 14))
    assert 'missed_runs' in sent
    assert any('missed check-ins' in a['title'] for a in notifier.alerts)


# ── Ledger math ──────────────────────────────────────────────────────────────

def test_ledger_cost_basis_and_lump_sum(tmp_db):
    closes = daily_series(uptrend=True)
    main_db = FakeMainDB(closes)
    d1 = closes.index[-100].date()
    d2 = closes.index[-50].date()
    pc.log_buy(tmp_db, main_db, 100.0, buy_date=d1)
    pc.log_buy(tmp_db, main_db, 100.0, buy_date=d2)

    s = pc.ledger_summary(tmp_db, main_db)
    p1 = float(closes.iloc[-100])
    p2 = float(closes.iloc[-50])
    latest = float(closes.iloc[-1])
    expected_btc = 100 / p1 + 100 / p2
    assert abs(s['btc_total'] - expected_btc) < 1e-12
    assert abs(s['cost_basis'] - 200 / expected_btc) < 1e-9
    assert abs(s['current_value'] - expected_btc * latest) < 1e-6
    assert abs(s['lump_sum_value'] - 200 / p1 * latest) < 1e-6


# ── The never-trades guarantee, enforced ─────────────────────────────────────

def test_playbook_code_cannot_trade_source_scan():
    """The companion must contain no path to an exchange order: no ccxt,
    no exchange client, no order verbs, no API-key access."""
    forbidden = ('import ccxt', 'create_order', 'createOrder', 'apiKey',
                 'BINANCE_API_KEY', 'BINANCE_SECRET', 'TradingEngine',
                 'sapi', 'order(')
    for path in (Path('src/playbook/companion.py'),
                 Path('src/playbook/__init__.py'),
                 Path('scripts/playbook_companion.py'),
                 Path('scripts/playbook_log.py')):
        source = path.read_text()
        for token in forbidden:
            assert token not in source, f"{path}: forbidden token '{token}'"
