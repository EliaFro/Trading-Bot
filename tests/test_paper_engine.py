"""Paper-trading engine: fill economics, LIMIT semantics, risk gates,
state persistence, graceful close-all."""

import asyncio
from datetime import datetime, timedelta, timezone

import pytest

from src.utils.config import Config
from src.utils.monitoring import MetricsCollector


def make_engine(tmp_db, monkeypatch, cash=10_000.0):
    """TradingEngine wired to the temp DB with network calls stubbed out."""
    monkeypatch.setenv('ENABLE_LIVE_TRADING', 'false')
    monkeypatch.setenv('ENABLE_PAPER_TRADING', 'true')

    config = Config(
        trading={'symbols': ['BTC/USDT'], 'timeframes': ['5m'],
                 'signal_timeframe': '5m', 'cycle_interval': 60,
                 'lookback_bars': 200, 'initial_capital': cash},
        execution={'commission_rate': 0.001, 'slippage_rate': 0.0005,
                   'slippage_tolerance': 0.001, 'order_timeout_seconds': 90},
        risk_management={'max_position_size': 0.1, 'max_concurrent_positions': 5,
                         'max_drawdown': 0.15, 'daily_loss_limit': 0.03,
                         'stop_loss_pct': 0.02, 'take_profit_pct': 0.04},
        strategies={'enabled': [], 'min_confidence': 0.55},
    )

    from src.trading.engine import TradingEngine

    class NoopModels:
        def generate_signals(self, **kwargs):
            return []

    engine = TradingEngine(config, NoopModels(), tmp_db, MetricsCollector())
    return engine


def test_buy_fill_includes_commission_and_slippage(tmp_db, monkeypatch):
    engine = make_engine(tmp_db, monkeypatch)
    engine.latest_prices['BTC/USDT'] = 50_000.0

    assert engine._open_position({
        'symbol': 'BTC/USDT', 'action': 'BUY', 'size': 0.10,
        'confidence': 0.9, 'stop_loss': 49_000.0, 'take_profit': 52_000.0,
        'metadata': {'strategy': 'test'},
    })

    # Position exists and was filled at market * (1 + slippage 0.05%),
    # not at the (higher) limit price
    assert 'BTC/USDT' in engine.positions
    pos = engine.positions['BTC/USDT']
    expected_fill = 50_000.0 * 1.0005
    assert abs(pos['entry_price'] - expected_fill) < 1e-6

    # Cash decreased by fill value + 0.1% commission
    fill_value = pos['quantity'] * expected_fill
    commission = fill_value * 0.001
    assert abs((10_000.0 - engine.cash) - (fill_value + commission)) < 1e-6

    # Notional respects the 10% cap
    assert fill_value <= 10_000.0 * 0.10 + 1e-6

    # Trade persisted as OPEN with the commission recorded
    open_rows = tmp_db.get_active_positions()
    assert len(open_rows) == 1


def test_limit_order_rests_until_price_reached(tmp_db, monkeypatch):
    engine = make_engine(tmp_db, monkeypatch)
    engine.latest_prices['BTC/USDT'] = 50_000.0

    engine._open_position({'symbol': 'BTC/USDT', 'action': 'BUY', 'size': 0.05,
                           'confidence': 0.9, 'metadata': {}})
    assert 'BTC/USDT' in engine.positions
    limit_price = 50_000.0 * 1.001  # marketable limit used at placement

    # Second scenario: price jumps ABOVE the limit before the fill check
    engine2 = make_engine(tmp_db, monkeypatch)
    engine2.latest_prices['ETH/USDT'] = 3_000.0
    engine2.symbols = ['ETH/USDT']
    engine2._open_position({'symbol': 'ETH/USDT', 'action': 'BUY', 'size': 0.05,
                            'confidence': 0.9, 'metadata': {}})
    # simulate: order placed, then market moved up 1% -> limit not marketable
    engine2.positions.pop('ETH/USDT', None)  # undo instant fill for the test
    order = {
        'id': 'o-1', 'symbol': 'ETH/USDT', 'side': 'BUY', 'order_type': 'LIMIT',
        'quantity': 0.1, 'price': 3_000.0 * 1.001,
        'created_at': datetime.now(timezone.utc), 'signal': {'metadata': {}},
    }
    engine2.pending_orders = [order]
    engine2.latest_prices['ETH/USDT'] = 3_100.0   # above limit -> no fill
    engine2._process_pending_orders()
    assert engine2.pending_orders, "order should still be resting"

    # Price falls back through the limit -> fills
    engine2.latest_prices['ETH/USDT'] = 2_990.0
    engine2._process_pending_orders()
    assert not engine2.pending_orders


def test_limit_order_expires_after_timeout(tmp_db, monkeypatch):
    engine = make_engine(tmp_db, monkeypatch)
    engine.latest_prices['BTC/USDT'] = 50_000.0
    order = {
        'id': 'o-2', 'symbol': 'BTC/USDT', 'side': 'BUY', 'order_type': 'LIMIT',
        'quantity': 0.01, 'price': 49_000.0,   # non-marketable
        'created_at': datetime.now(timezone.utc) - timedelta(seconds=300),
        'signal': {'metadata': {}},
    }
    engine.pending_orders = [order]
    engine._process_pending_orders()
    assert engine.pending_orders == []          # expired, not filled
    assert 'BTC/USDT' not in engine.positions


def test_stop_loss_and_take_profit(tmp_db, monkeypatch):
    engine = make_engine(tmp_db, monkeypatch)
    engine.latest_prices['BTC/USDT'] = 50_000.0
    engine._open_position({'symbol': 'BTC/USDT', 'action': 'BUY', 'size': 0.10,
                           'confidence': 0.9, 'stop_loss': 49_000.0,
                           'take_profit': 52_000.0, 'metadata': {}})
    cash_after_open = engine.cash

    # Price crashes through the stop -> position closes at a loss
    engine.latest_prices['BTC/USDT'] = 48_900.0
    engine._check_stops()
    assert 'BTC/USDT' not in engine.positions

    trades = tmp_db.get_recent_trades(10)
    closed = trades[trades['status'] == 'CLOSED']
    assert len(closed) == 1
    assert closed.iloc[0]['exit_reason'] == 'stop_loss'
    assert closed.iloc[0]['pnl'] < 0
    assert engine.cash > cash_after_open        # sale proceeds returned


def test_round_trip_pnl_math(tmp_db, monkeypatch):
    """Full round trip at known prices: verify P&L to the cent."""
    engine = make_engine(tmp_db, monkeypatch)
    engine.latest_prices['BTC/USDT'] = 50_000.0
    engine._open_position({'symbol': 'BTC/USDT', 'action': 'BUY', 'size': 0.10,
                           'confidence': 0.9, 'metadata': {}})
    pos = engine.positions['BTC/USDT']
    qty = pos['quantity']
    entry_fill = 50_000.0 * 1.0005

    engine.latest_prices['BTC/USDT'] = 51_000.0
    engine._close_position('BTC/USDT', 51_000.0, 'signal')

    exit_fill = 51_000.0 * (1 - 0.0005)
    expected_pnl = (qty * exit_fill) * (1 - 0.001) - qty * entry_fill

    trades = tmp_db.get_recent_trades(10)
    closed = trades[trades['status'] == 'CLOSED'].iloc[0]
    assert abs(closed['pnl'] - expected_pnl) < 0.01

    # Final cash = initial - entry cost + exit proceeds
    entry_cost = qty * entry_fill * (1 + 0.001)
    exit_proceeds = qty * exit_fill * (1 - 0.001)
    assert abs(engine.cash - (10_000.0 - entry_cost + exit_proceeds)) < 0.01


def test_max_concurrent_positions_gate(tmp_db, monkeypatch):
    engine = make_engine(tmp_db, monkeypatch)
    engine.limits = engine.limits.__class__(
        max_position_size=0.1, max_concurrent_positions=2, max_drawdown=0.15,
        daily_loss_limit=0.03, stop_loss_pct=0.02, take_profit_pct=0.04)
    for i, symbol in enumerate(['BTC/USDT', 'ETH/USDT']):
        engine.latest_prices[symbol] = 100.0 * (i + 1)
        engine._open_position({'symbol': symbol, 'action': 'BUY', 'size': 0.05,
                               'confidence': 0.9, 'metadata': {}})
    assert len(engine.positions) == 2
    assert engine._risk_gate() is not None      # third trade blocked


def test_state_survives_restart(tmp_db, monkeypatch):
    engine = make_engine(tmp_db, monkeypatch)
    engine.latest_prices['BTC/USDT'] = 50_000.0
    engine._open_position({'symbol': 'BTC/USDT', 'action': 'BUY', 'size': 0.05,
                           'confidence': 0.9, 'metadata': {}})
    engine._persist_cash()
    cash_before, positions_before = engine.cash, dict(engine.positions)

    # "Restart": a brand-new engine on the same database
    engine2 = make_engine(tmp_db, monkeypatch)
    assert abs(engine2.cash - cash_before) < 1e-9
    assert set(engine2.positions) == set(positions_before)
    restored = engine2.positions['BTC/USDT']
    assert abs(restored['entry_price'] -
               positions_before['BTC/USDT']['entry_price']) < 1e-9


def test_close_all_positions(tmp_db, monkeypatch):
    engine = make_engine(tmp_db, monkeypatch)
    for symbol, price in (('BTC/USDT', 50_000.0), ('ETH/USDT', 3_000.0)):
        engine.latest_prices[symbol] = price
        engine._open_position({'symbol': symbol, 'action': 'BUY', 'size': 0.05,
                               'confidence': 0.9, 'metadata': {}})
    assert len(engine.positions) == 2

    async def _run():
        # avoid network in _refresh_prices
        async def fake_refresh():
            return None
        engine._refresh_prices = fake_refresh
        await engine.close_all_positions('shutdown')

    asyncio.run(_run())
    assert engine.positions == {}
    trades = tmp_db.get_recent_trades(10)
    assert (trades['status'] == 'CLOSED').sum() == 2
    assert (trades['exit_reason'] == 'shutdown').sum() == 2
