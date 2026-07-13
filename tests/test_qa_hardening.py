"""Final QA pass regression tests: feature-width drift, network-failure
drills, database concurrency, and log-rotation guarantees."""

import threading
import time

import numpy as np
import pandas as pd
import pytest


# ── QA fix: pattern detector feature-width drift ─────────────────────────────

def test_detector_survives_feature_width_drift():
    """The detector's scaler/PCA fitted at one width must refit — not crash —
    when the extractor's width drifts (the hourly error loop found in QA)."""
    from src.patterns.detector import AdvancedPatternDetector
    det = AdvancedPatternDetector(enable_deep_learning=False)

    wide = np.random.default_rng(1).normal(size=(50, 223))
    out1 = det._preprocess_features(wide)
    assert out1.shape[0] == 50

    narrow = np.random.default_rng(2).normal(size=(50, 220))
    out2 = det._preprocess_features(narrow)          # crashed before the fix
    assert out2.shape[0] == 50

    # and detect_patterns end-to-end doesn't raise on either width
    det.detect_patterns(narrow)
    det.detect_patterns(wide)


# ── Drill: exchange/network unreachable ──────────────────────────────────────

def _engine(tmp_db, monkeypatch):
    from tests.test_paper_engine import make_engine
    return make_engine(tmp_db, monkeypatch)


def test_cycle_survives_network_down(tmp_db, monkeypatch):
    """Internet/Binance down: the cycle must complete gracefully, record the
    error, and not fabricate prices."""
    import ccxt
    import asyncio
    engine = _engine(tmp_db, monkeypatch)

    def dead_tickers(symbols):
        raise ccxt.NetworkError("simulated: network is down")

    def dead_update(symbol, timeframe, lookback_bars=300):
        raise ccxt.NetworkError("simulated: network is down")

    engine.market_data.fetch_tickers = dead_tickers
    engine.market_data.update_symbol = dead_update

    asyncio.run(engine.run_cycle())                  # must not raise
    assert engine.latest_prices == {}                # nothing fabricated
    errors = engine.metrics.get_recent_errors()
    assert errors, "network failure must be recorded, not swallowed silently"


def test_ml_loop_skips_day_without_prices(tmp_db, monkeypatch):
    """Covered by main loop logic: no prices -> no trade, slot not burned.
    Here: the engine-level guarantee that _open_position without a price
    refuses rather than inventing one."""
    engine = _engine(tmp_db, monkeypatch)
    assert engine._open_position({'symbol': 'BTC/USDT', 'action': 'BUY',
                                  'size': 0.1, 'confidence': 0.9,
                                  'metadata': {}}) is False
    assert 'BTC/USDT' not in engine.positions


# ── Drill: concurrent access (WAL) ───────────────────────────────────────────

def test_concurrent_writer_and_readers_no_drops(tmp_path):
    """Four services + dashboard share SQLite in WAL mode. Hammer: one
    writer streaming equity rows while readers poll — every write must land,
    no exception may escape."""
    from scripts.init_db import init_db
    from src.utils.database import DatabaseManager
    import sqlite3

    path = str(tmp_path / 'hammer.db')
    init_db(path)
    sqlite3.connect(path).execute("PRAGMA journal_mode=WAL").close()

    writer_db = DatabaseManager(path)
    n_writes, errors = 300, []

    def writer():
        for i in range(n_writes):
            if not writer_db.record_equity(equity=10_000 + i, cash=10_000,
                                           positions_value=0,
                                           active_positions=0):
                errors.append('write failed')

    def reader():
        rdb = DatabaseManager(path)
        for _ in range(150):
            try:
                rdb.get_equity_curve()
                rdb.get_recent_signals(5)
            except Exception as e:
                errors.append(f'reader: {e}')

    threads = [threading.Thread(target=writer)] + \
              [threading.Thread(target=reader) for _ in range(3)]
    for t in threads:
        t.start()
    for t in threads:
        t.join(timeout=60)

    assert not errors, errors[:3]
    final = pd.read_sql_query("SELECT COUNT(*) n FROM performance_tracking",
                              writer_db.engine)['n'].iloc[0]
    assert final == n_writes, f"dropped writes: {final}/{n_writes}"


# ── Log rotation is configured on every service ──────────────────────────────

def test_all_services_use_rotating_logs():
    from pathlib import Path
    for path, cap_mb in (('src/main.py', 50), ('scripts/fastlab_bot.py', 20),
                         ('scripts/playbook_companion.py', 5)):
        source = Path(path).read_text()
        assert 'RotatingFileHandler' in source, f"{path}: no rotation"
        assert f'{cap_mb} * 1024 * 1024' in source, f"{path}: cap changed"
