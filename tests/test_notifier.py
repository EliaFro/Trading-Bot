"""Notifier: persistence, rate limiting, graceful no-channel operation."""

from src.utils.notifier import Notifier


def test_alert_persists_to_db_without_channels(tmp_db, monkeypatch):
    for var in ('TELEGRAM_BOT_TOKEN', 'TELEGRAM_CHAT_ID', 'ALERT_EMAIL',
                'SMTP_HOST'):
        monkeypatch.delenv(var, raising=False)
    notifier = Notifier(db=tmp_db)

    delivered = notifier.alert('risk', 'critical', 'Test alert',
                               'drawdown breached')
    assert delivered is False          # no external channel

    import pandas as pd
    from sqlalchemy import text
    with tmp_db.engine.connect() as conn:
        rows = conn.execute(text("SELECT alert_type, severity, title "
                                 "FROM alerts")).fetchall()
    assert len(rows) == 1
    assert rows[0][0] == 'risk' and rows[0][1] == 'critical'


def test_rate_limiting_dedupes_repeats(tmp_db, monkeypatch):
    monkeypatch.delenv('TELEGRAM_BOT_TOKEN', raising=False)
    notifier = Notifier(db=tmp_db, min_interval_per_key=3600)

    sent = []
    notifier._send_telegram = lambda text: sent.append(text) or True
    notifier.telegram_token = 'x'
    notifier.telegram_chat = 'y'

    assert notifier.alert('error', 'warning', 'Loop error', 'boom',
                          dedupe_key='loop') is True
    # Same key inside the window: persisted but not re-sent
    assert notifier.alert('error', 'warning', 'Loop error', 'boom again',
                          dedupe_key='loop') is False
    assert len(sent) == 1

    # Different key sends immediately
    assert notifier.alert('risk', 'critical', 'Other alert', 'x',
                          dedupe_key='other') is True
    assert len(sent) == 2


def test_daily_summary_formats(tmp_db, monkeypatch):
    notifier = Notifier(db=tmp_db)
    sent = []
    notifier._send_telegram = lambda text: sent.append(text) or True
    notifier.telegram_token = 'x'
    notifier.telegram_chat = 'y'

    assert notifier.daily_summary({
        'equity': 10123.45, 'daily_return': 0.0123, 'n_trades': 7,
        'wins': 5, 'losses': 2, 'realized_pnl': 123.45, 'fees': 6.78,
        'open_positions': 2, 'drawdown': -0.012, 'mode': 'paper',
        'uptime_hours': 25.5,
    })
    assert len(sent) == 1
    assert '10,123.45' in sent[0] and '+1.23%' in sent[0]
