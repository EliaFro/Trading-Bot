#!/usr/bin/env python3
"""
Canonical database initialization for the AI Crypto Trading System.

THE single source of truth for the SQLite schema. DatabaseManager,
the dashboard, the pattern pipeline, and the trading engine all read/write
exactly these tables and column names.

Conventions:
  * every time column is an INTEGER unix epoch (UTC seconds)
  * trades use entry_time / exit_time (never a bare `timestamp`)
  * OHLCV lives in `ohlcv` keyed (symbol, timeframe, timestamp)

Usage:
    python scripts/init_db.py            # create/upgrade at $DB_PATH
    python scripts/init_db.py --reset    # drop and recreate (asks unless --yes)
"""

import argparse
import os
import sqlite3
import sys
from datetime import datetime, timezone
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parent.parent))

SCHEMA = """
CREATE TABLE IF NOT EXISTS ohlcv (
    symbol      TEXT NOT NULL,
    timeframe   TEXT NOT NULL,
    timestamp   INTEGER NOT NULL,          -- bar open time, epoch seconds UTC
    open        REAL NOT NULL,
    high        REAL NOT NULL,
    low         REAL NOT NULL,
    close       REAL NOT NULL,
    volume      REAL NOT NULL,
    PRIMARY KEY (symbol, timeframe, timestamp)
);

CREATE TABLE IF NOT EXISTS trades (
    id              TEXT PRIMARY KEY,
    symbol          TEXT NOT NULL,
    side            TEXT NOT NULL,          -- BUY / SELL
    quantity        REAL NOT NULL,
    entry_price     REAL NOT NULL,
    exit_price      REAL,
    stop_loss       REAL,
    take_profit     REAL,
    pnl             REAL,
    pnl_percentage  REAL,
    commission      REAL DEFAULT 0,
    slippage        REAL DEFAULT 0,
    strategy        TEXT NOT NULL,
    features        TEXT,                   -- JSON
    entry_time      INTEGER NOT NULL,       -- epoch seconds UTC
    exit_time       INTEGER,
    status          TEXT DEFAULT 'OPEN',    -- OPEN / CLOSED / CANCELLED
    mode            TEXT DEFAULT 'paper',   -- paper / live / backtest
    exit_reason     TEXT,                   -- signal/stop_loss/take_profit/shutdown/...
    pattern_id      TEXT,
    created_at      INTEGER DEFAULT (strftime('%s','now'))
);

CREATE TABLE IF NOT EXISTS signals (
    id          INTEGER PRIMARY KEY AUTOINCREMENT,
    timestamp   INTEGER NOT NULL,
    symbol      TEXT NOT NULL,
    action      TEXT NOT NULL,              -- BUY / SELL / HOLD
    confidence  REAL,
    size        REAL,
    stop_loss   REAL,
    take_profit REAL,
    strategy    TEXT,
    executed    INTEGER DEFAULT 0,          -- 1 if it became an order
    metadata    TEXT                        -- JSON
);

CREATE TABLE IF NOT EXISTS orders (
    id           TEXT PRIMARY KEY,
    trade_id     TEXT,
    symbol       TEXT NOT NULL,
    side         TEXT NOT NULL,
    order_type   TEXT NOT NULL,             -- LIMIT / MARKET / STOP_LOSS / TAKE_PROFIT
    quantity     REAL NOT NULL,
    price        REAL,
    status       TEXT DEFAULT 'PENDING',    -- PENDING / FILLED / CANCELLED / REJECTED
    filled_qty   REAL DEFAULT 0,
    fill_price   REAL,
    commission   REAL DEFAULT 0,
    slippage     REAL DEFAULT 0,
    exchange_id  TEXT,                      -- ccxt order id in live mode
    created_at   INTEGER NOT NULL,
    updated_at   INTEGER
);

CREATE TABLE IF NOT EXISTS performance_tracking (
    id               INTEGER PRIMARY KEY AUTOINCREMENT,
    timestamp        INTEGER NOT NULL,
    total_equity     REAL NOT NULL,
    cash             REAL,
    positions_value  REAL,
    daily_return     REAL,
    drawdown         REAL,
    active_positions INTEGER DEFAULT 0,
    mode             TEXT DEFAULT 'paper',
    benchmark_price  REAL                   -- BTC close, for buy-and-hold comparison
);

CREATE TABLE IF NOT EXISTS discovered_patterns (
    id             INTEGER PRIMARY KEY AUTOINCREMENT,
    pattern_id     TEXT UNIQUE,
    pattern_type   TEXT NOT NULL,
    symbol         TEXT,
    timeframe      TEXT,
    pattern_config TEXT,                    -- JSON
    confidence     REAL DEFAULT 0,
    performance    REAL NOT NULL DEFAULT 0,
    discovery_date INTEGER NOT NULL,
    status         TEXT DEFAULT 'candidate',-- candidate / active / expired
    success_rate   REAL DEFAULT 0,
    avg_return     REAL DEFAULT 0,
    times_used     INTEGER DEFAULT 0
);

CREATE TABLE IF NOT EXISTS model_versions (
    id                  INTEGER PRIMARY KEY AUTOINCREMENT,
    model_name          TEXT NOT NULL,
    model_type          TEXT,
    version             TEXT NOT NULL,
    file_path           TEXT,
    parameters          TEXT,               -- JSON
    performance_metrics TEXT,               -- JSON
    created_at          INTEGER NOT NULL,
    is_active           INTEGER DEFAULT 0
);

CREATE TABLE IF NOT EXISTS sentiment_scores (
    id              INTEGER PRIMARY KEY AUTOINCREMENT,
    symbol          TEXT NOT NULL,
    source          TEXT NOT NULL,
    timestamp       INTEGER NOT NULL,
    sentiment_score REAL NOT NULL,          -- -1 .. 1
    confidence      REAL NOT NULL,          -- 0 .. 1
    volume          INTEGER DEFAULT 0,
    metadata        TEXT                    -- JSON
);

CREATE TABLE IF NOT EXISTS alerts (
    id              INTEGER PRIMARY KEY AUTOINCREMENT,
    alert_type      TEXT NOT NULL,
    severity        TEXT NOT NULL,          -- info / warning / critical
    title           TEXT NOT NULL,
    message         TEXT NOT NULL,
    symbol          TEXT,
    data            TEXT,                   -- JSON
    timestamp       INTEGER NOT NULL,
    acknowledged    INTEGER DEFAULT 0
);

CREATE TABLE IF NOT EXISTS ml_predictions (
    id          INTEGER PRIMARY KEY AUTOINCREMENT,
    timestamp   INTEGER NOT NULL,           -- decision time, epoch seconds UTC
    symbol      TEXT NOT NULL,
    pred        TEXT NOT NULL,              -- UP / FLAT / DOWN
    p_up        REAL,
    p_down      REAL,
    model_version TEXT,
    executed    INTEGER DEFAULT 0
);

CREATE TABLE IF NOT EXISTS ml_retrain_log (
    id             INTEGER PRIMARY KEY AUTOINCREMENT,
    timestamp      INTEGER NOT NULL,
    old_version    TEXT,
    new_version    TEXT NOT NULL,
    decision       TEXT NOT NULL,           -- REPLACED / KEPT_OLD / INITIAL
    reason         TEXT,
    old_val_f1     REAL,
    new_val_f1     REAL,
    new_is_bal_acc REAL,                    -- in-sample (overfitting gauge)
    new_val_bal_acc REAL,                   -- validation slice (honest side)
    n_train        INTEGER,
    feature_importance TEXT                 -- JSON: top features + weights
);

CREATE TABLE IF NOT EXISTS system_config (
    config_key   TEXT PRIMARY KEY,
    config_value TEXT NOT NULL,
    config_type  TEXT NOT NULL,
    description  TEXT,
    updated_at   INTEGER NOT NULL
);

-- Indexes
CREATE INDEX IF NOT EXISTS idx_ohlcv_lookup       ON ohlcv(symbol, timeframe, timestamp);
CREATE INDEX IF NOT EXISTS idx_trades_status      ON trades(status);
CREATE INDEX IF NOT EXISTS idx_trades_symbol      ON trades(symbol, entry_time);
CREATE INDEX IF NOT EXISTS idx_trades_strategy    ON trades(strategy);
CREATE INDEX IF NOT EXISTS idx_signals_time       ON signals(timestamp);
CREATE INDEX IF NOT EXISTS idx_orders_status      ON orders(status);
CREATE INDEX IF NOT EXISTS idx_perf_time          ON performance_tracking(timestamp);
CREATE INDEX IF NOT EXISTS idx_sentiment_lookup   ON sentiment_scores(symbol, timestamp);
CREATE INDEX IF NOT EXISTS idx_patterns_status    ON discovered_patterns(status);
CREATE INDEX IF NOT EXISTS idx_alerts_time        ON alerts(timestamp);
"""

# Tables from the two legacy init scripts that conflict with this schema
LEGACY_TABLES = ('model_performance',)


def init_db(db_path: str, reset: bool = False) -> None:
    path = Path(db_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    Path('./logs').mkdir(exist_ok=True)
    Path('./data/cache').mkdir(parents=True, exist_ok=True)

    if reset and path.exists():
        path.unlink()
        print(f"Removed existing database {path}")

    conn = sqlite3.connect(path)
    try:
        cur = conn.cursor()

        # If a legacy `trades` table (single `timestamp` column) exists, it is
        # incompatible — refuse to guess, require explicit --reset.
        cur.execute("SELECT name FROM sqlite_master WHERE type='table' AND name='trades'")
        if cur.fetchone():
            cur.execute("PRAGMA table_info(trades)")
            cols = {row[1] for row in cur.fetchall()}
            if 'entry_time' not in cols:
                raise SystemExit(
                    "Existing 'trades' table uses the legacy schema "
                    "(no entry_time column). Re-run with --reset to rebuild. "
                    "Nothing was changed.")

        cur.executescript(SCHEMA)

        for table in LEGACY_TABLES:
            cur.execute(f"DROP TABLE IF EXISTS {table}")

        now = int(datetime.now(timezone.utc).timestamp())
        cur.execute("""
            INSERT OR REPLACE INTO system_config
            (config_key, config_value, config_type, description, updated_at)
            VALUES ('schema_version', '2', 'number', 'Canonical schema v2', ?)
        """, (now,))
        conn.commit()

        cur.execute("SELECT name FROM sqlite_master WHERE type='table' ORDER BY name")
        tables = [r[0] for r in cur.fetchall() if r[0] != 'sqlite_sequence']
        print(f"Database ready at {path}")
        for t in tables:
            cur.execute(f"SELECT COUNT(*) FROM {t}")
            print(f"  {t:24s} {cur.fetchone()[0]} rows")
    finally:
        conn.close()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--reset', action='store_true',
                        help='Drop and recreate the database')
    parser.add_argument('--yes', action='store_true',
                        help='Skip confirmation for --reset')
    parser.add_argument('--db-path', default=os.getenv('DB_PATH', './data/trading_system.db'))
    args = parser.parse_args()

    if args.reset and not args.yes:
        answer = input(f"Really DELETE {args.db_path} and recreate? [y/N] ")
        if answer.strip().lower() != 'y':
            sys.exit('Aborted.')

    init_db(args.db_path, reset=args.reset)
