"""Database round-trips on the canonical schema."""

from datetime import datetime, timedelta

import pandas as pd


def test_ohlcv_store_and_dedup(tmp_db):
    ts = pd.date_range('2025-06-01', periods=10, freq='5min', tz='UTC')
    df = pd.DataFrame({
        'timestamp': ts,
        'open': range(100, 110), 'high': range(101, 111),
        'low': range(99, 109), 'close': range(100, 110),
        'volume': [1.0] * 10,
    })
    assert tmp_db.store_ohlcv('BTC/USDT', '5m', df) == 10
    # Re-storing the same bars must not duplicate
    tmp_db.store_ohlcv('BTC/USDT', '5m', df)
    out = tmp_db.get_ohlcv_data('BTC/USDT', '5m')
    assert len(out) == 10
    assert list(out.columns) == ['timestamp', 'open', 'high', 'low', 'close', 'volume']
    assert out['timestamp'].is_monotonic_increasing


def test_latest_ohlcv_timestamp(tmp_db):
    ts = pd.date_range('2025-06-01', periods=5, freq='1h', tz='UTC')
    df = pd.DataFrame({'timestamp': ts, 'open': [1] * 5, 'high': [2] * 5,
                       'low': [0.5] * 5, 'close': [1.5] * 5, 'volume': [9] * 5})
    tmp_db.store_ohlcv('ETH/USDT', '1h', df)
    latest = tmp_db.get_latest_ohlcv_timestamp('ETH/USDT', '1h')
    assert latest == int(ts[-1].timestamp())
    assert tmp_db.get_latest_ohlcv_timestamp('XRP/USDT', '1h') is None


def test_trade_lifecycle(tmp_db):
    entry_time = datetime(2025, 6, 1, 12, 0)
    tmp_db.store_trade({
        'id': 'trade-1', 'symbol': 'BTC/USDT', 'side': 'BUY',
        'quantity': 0.01, 'entry_price': 60000.0, 'exit_price': None,
        'stop_loss': 58800.0, 'take_profit': 62400.0, 'pnl': None,
        'pnl_percentage': None, 'commission': 0.6, 'slippage': 0.3,
        'strategy': 'ma_crossover', 'features': {'note': 'test'},
        'entry_time': entry_time, 'exit_time': None, 'status': 'OPEN',
    })

    open_positions = tmp_db.get_active_positions()
    assert len(open_positions) == 1
    assert open_positions.iloc[0]['symbol'] == 'BTC/USDT'

    assert tmp_db.update_trade('trade-1', {
        'exit_price': 61000.0,
        'exit_time': entry_time + timedelta(hours=3),
        'pnl': 9.4, 'pnl_percentage': 0.0157, 'status': 'CLOSED',
    })
    assert tmp_db.get_active_positions().empty

    metrics = tmp_db.get_performance_metrics()
    assert metrics['total_trades'] == 1
    assert metrics['win_rate'] == 1.0


def test_signals_roundtrip(tmp_db):
    tmp_db.store_signal({
        'symbol': 'SOL/USDT', 'action': 'BUY', 'confidence': 0.8,
        'size': 0.05, 'stop_loss': 140.0, 'take_profit': 150.0,
        'executed': True, 'metadata': {'strategy': 'breakout'},
    })
    out = tmp_db.get_recent_signals()
    assert len(out) == 1
    assert out.iloc[0]['action'] == 'BUY'
    assert out.iloc[0]['executed'] == 1


def test_equity_curve_roundtrip(tmp_db):
    for equity in (10000.0, 10050.0, 10025.0):
        assert tmp_db.record_equity(equity=equity, cash=equity,
                                    positions_value=0.0, active_positions=0,
                                    mode='paper', benchmark_price=60000.0)
    curve = tmp_db.get_equity_curve()
    assert len(curve) == 3
    assert curve['total_equity'].iloc[-1] == 10025.0


def test_sentiment_roundtrip(tmp_db):
    assert tmp_db.store_sentiment('BTC/USDT', {
        'sentiment': 0.42, 'confidence': 0.7, 'volume': 12,
        'source': 'news', 'metadata': {'backend': 'vader'},
    })
    history = tmp_db.get_sentiment_history('BTC/USDT', hours=24)
    assert len(history) == 1
    assert abs(history.iloc[0]['sentiment_score'] - 0.42) < 1e-9
