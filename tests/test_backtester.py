"""Backtester correctness: strategies trade through the fixed runner, fees are
charged, and short accounting no longer double-counts cash."""

import numpy as np
import pandas as pd
import pytest

from src.backtesting.engine import (AdvancedBacktester, BacktestConfig,
                                    PositionSide)
from src.backtesting.backtest_module import BacktestRunner


def test_runner_produces_trades_on_real_structure(trending_ohlcv):
    """The fixed runner feeds history windows, so strategies actually trade."""
    runner = BacktestRunner(BacktestConfig(initial_capital=10_000.0))
    result = runner.run_strategy_backtest(
        'rsi_mean_reversion', {}, {'BTC/USDT': trending_ohlcv})

    assert len(result.trades) > 0, \
        "runner must produce trades on data with clear oversold episodes"
    assert result.total_commission > 0, "fees must be charged"
    assert result.equity_curve.iloc[0] > 0


def test_backtest_config_defaults_long_only():
    assert BacktestConfig().allow_shorting is False


def _one_bar_frame(price: float, ts: pd.Timestamp) -> pd.Series:
    return pd.Series({'open': price, 'high': price, 'low': price,
                      'close': price, 'volume': 1000.0}, name=ts)


def test_short_accounting_no_double_count():
    """Open a short at 100, close at 90 (10% favorable). Final equity must be
    initial + gross_pnl - fees - slippage; before the fix it was inflated by
    the entire position value."""
    config = BacktestConfig(initial_capital=10_000.0, allow_shorting=True,
                            commission_rate=0.001, slippage_rate=0.0)
    bt = AdvancedBacktester(config)
    bt.reset()

    ts1 = pd.Timestamp('2025-01-01 00:00')
    ts2 = pd.Timestamp('2025-01-01 01:00')

    bt.current_time = ts1
    bt._open_short_position('BTC/USDT',
                            {'symbol': 'BTC/USDT', 'action': 'SELL',
                             'size': 0.10}, _one_bar_frame(100.0, ts1))
    assert 'BTC/USDT' in bt.positions
    pos = bt.positions['BTC/USDT']
    qty = pos.quantity
    entry = pos.entry_price

    bt.current_time = ts2
    bt._close_position('BTC/USDT', _one_bar_frame(90.0, ts2), 'signal')

    trade = bt.trades[0]
    assert trade.side == PositionSide.SHORT
    gross = (entry - 90.0) * qty
    fees = entry * qty * 0.001 + 90.0 * qty * 0.001

    expected_final_cash = 10_000.0 + gross - fees
    assert abs(bt.cash - expected_final_cash) < 0.01, (
        f"cash {bt.cash:.2f} != expected {expected_final_cash:.2f} — "
        f"short proceeds double-counted?")
    # And the recorded trade P&L agrees
    assert abs(trade.pnl - (gross - fees)) < 0.01


def test_long_round_trip_charges_both_commissions(trending_ohlcv):
    config = BacktestConfig(initial_capital=10_000.0, commission_rate=0.001,
                            slippage_rate=0.0005)
    bt = AdvancedBacktester(config)
    bt.reset()

    ts1, ts2 = trending_ohlcv.index[0], trending_ohlcv.index[1]
    row1 = trending_ohlcv.iloc[0]
    row2 = trending_ohlcv.iloc[1].copy()
    row2.name = ts2

    bt.current_time = ts1
    bt._open_long_position('BTC/USDT',
                           {'symbol': 'BTC/USDT', 'action': 'BUY', 'size': 0.1},
                           row1)
    bt.current_time = ts2
    bt._close_position('BTC/USDT', row2, 'signal')

    assert len(bt.trades) == 1
    assert bt.metrics['total_commission'] > 0
    assert bt.metrics['total_slippage'] > 0
