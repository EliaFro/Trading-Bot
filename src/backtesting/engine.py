"""
Advanced backtesting engine for AI Crypto Trading System
Provides realistic simulation of trading with proper order execution, slippage, and fees
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional, Callable, Union
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from collections import defaultdict, deque
import logging
from enum import Enum
import uuid

logger = logging.getLogger(__name__)

# --- ENUMS AND DATA STRUCTURES ---

class OrderType(Enum):
    MARKET = "MARKET"
    LIMIT = "LIMIT"
    STOP_LOSS = "STOP_LOSS"
    TAKE_PROFIT = "TAKE_PROFIT"

class PositionSide(Enum):
    LONG = "LONG"
    SHORT = "SHORT"
    NEUTRAL = "NEUTRAL"

class OrderStatus(Enum):
    PENDING = "PENDING"
    FILLED = "FILLED"
    PARTIALLY_FILLED = "PARTIALLY_FILLED"
    CANCELLED = "CANCELLED"
    REJECTED = "REJECTED"

@dataclass
class BacktestConfig:
    """Configuration for backtesting"""
    initial_capital: float = 10000.0
    commission_rate: float = 0.001  # 0.1%
    slippage_rate: float = 0.0005  # 0.05%
    max_position_size: float = 0.1  # 10% of capital
    use_leverage: bool = False
    max_leverage: float = 3.0
    risk_free_rate: float = 0.02  # 2% annual
    rebalance_frequency: str = "daily"
    allow_shorting: bool = False  # v1 is long-only spot; enable only for research
    margin_call_level: float = 0.3  # 30% margin level triggers margin call
    funding_rate: float = 0.0001  # For perpetual futures
    
@dataclass
class Order:
    """Order data structure"""
    id: str = field(default_factory=lambda: str(uuid.uuid4()))
    symbol: str = ""
    side: str = ""  # BUY or SELL
    order_type: OrderType = OrderType.MARKET
    quantity: float = 0.0
    price: Optional[float] = None
    stop_price: Optional[float] = None
    time_in_force: str = "GTC"  # Good Till Cancelled
    timestamp: datetime = field(default_factory=datetime.now)
    status: OrderStatus = OrderStatus.PENDING
    filled_quantity: float = 0.0
    average_fill_price: float = 0.0
    commission: float = 0.0
    slippage: float = 0.0

@dataclass
class Position:
    """Position data structure"""
    symbol: str
    side: PositionSide
    entry_price: float
    quantity: float
    entry_time: datetime
    stop_loss: Optional[float] = None
    take_profit: Optional[float] = None
    leverage: float = 1.0
    unrealized_pnl: float = 0.0
    realized_pnl: float = 0.0
    entry_commission: float = 0.0
    entry_slippage: float = 0.0
    
    @property
    def value(self) -> float:
        """Get position value"""
        return self.quantity * self.entry_price
    
    @property
    def margin_required(self) -> float:
        """Get margin required for position"""
        return self.value / self.leverage
    
    def calculate_pnl(self, current_price: float) -> float:
        """Calculate P&L at current price"""
        if self.side == PositionSide.LONG:
            return (current_price - self.entry_price) * self.quantity
        else:  # SHORT
            return (self.entry_price - current_price) * self.quantity

@dataclass
class TradeResult:
    """Result of a completed trade"""
    id: str = field(default_factory=lambda: str(uuid.uuid4()))
    symbol: str = ""
    side: PositionSide = PositionSide.NEUTRAL
    entry_price: float = 0.0
    exit_price: float = 0.0
    quantity: float = 0.0
    entry_time: datetime = field(default_factory=datetime.now)
    exit_time: datetime = field(default_factory=datetime.now)
    pnl: float = 0.0
    pnl_percentage: float = 0.0
    commission: float = 0.0
    slippage: float = 0.0
    strategy: str = ""
    exit_reason: str = ""  # 'signal', 'stop_loss', 'take_profit', 'margin_call'
    metadata: Dict = field(default_factory=dict)

@dataclass
class BacktestResult:
    """Complete backtest results"""
    trades: List[TradeResult]
    equity_curve: pd.Series
    daily_returns: pd.Series
    total_return: float
    annualized_return: float
    sharpe_ratio: float
    sortino_ratio: float
    max_drawdown: float
    win_rate: float
    profit_factor: float
    avg_win: float
    avg_loss: float
    best_trade: float
    worst_trade: float
    avg_trade_duration: timedelta
    total_commission: float
    total_slippage: float
    calmar_ratio: float
    omega_ratio: float
    var_95: float  # Value at Risk
    cvar_95: float  # Conditional Value at Risk
    metadata: Dict = field(default_factory=dict)

# --- ADVANCED BACKTESTING ENGINE ---

class AdvancedBacktester:
    """Advanced backtesting engine with realistic simulation"""
    
    def __init__(self, config: BacktestConfig):
        self.config = config
        self.reset()
        
    def reset(self):
        """Reset backtester state"""
        self.cash = self.config.initial_capital
        self.initial_capital = self.config.initial_capital
        self.positions: Dict[str, Position] = {}
        self.orders: Dict[str, Order] = {}
        self.trades: List[TradeResult] = []
        self.equity_curve = []
        self.current_time = None
        self.margin_used = 0.0
        self.pending_orders: List[Order] = []
        
        # Performance tracking
        self.metrics = {
            'total_commission': 0.0,
            'total_slippage': 0.0,
            'total_funding': 0.0,
            'margin_calls': 0
        }
        
    def run_backtest(
        self,
        data: Dict[str, pd.DataFrame],
        strategy: Callable,
        start_date: Optional[datetime] = None,
        end_date: Optional[datetime] = None
    ) -> BacktestResult:
        """
        Run backtest on historical data using provided strategy
        
        Args:
            data: Dictionary of symbol -> DataFrame with OHLCV data
            strategy: Strategy function that returns trading signals
            start_date: Backtest start date
            end_date: Backtest end date
            
        Returns:
            BacktestResult with complete performance metrics
        """
        logger.info(f"Starting backtest from {start_date} to {end_date}")
        
        # Validate and prepare data
        aligned_data = self._align_data(data)
        
        if aligned_data.empty:
            logger.error("No data available for backtesting")
            return self._empty_result()
        
        # Filter by date range
        if start_date:
            aligned_data = aligned_data[aligned_data.index >= start_date]
        if end_date:
            aligned_data = aligned_data[aligned_data.index <= end_date]
        
        # Main backtest loop
        for timestamp, market_data in aligned_data.iterrows():
            self.current_time = timestamp
            
            # Process pending orders
            self._process_pending_orders(market_data)
            
            # Update positions with current prices
            self._update_positions(market_data)
            
            # Check margin requirements
            if self.config.use_leverage:
                self._check_margin_requirements(market_data)
            
            # Generate signals from strategy
            try:
                signals = strategy(market_data, self.positions, self.cash)
            except Exception as e:
                logger.error(f"Strategy error at {timestamp}: {e}")
                signals = []
            
            # Execute trades based on signals
            self._execute_signals(signals, market_data)
            
            # Record equity
            total_equity = self._calculate_total_equity(market_data)
            self.equity_curve.append({
                'timestamp': timestamp,
                'equity': total_equity,
                'cash': self.cash,
                'positions_value': total_equity - self.cash,
                'margin_used': self.margin_used
            })
        
        # Close all remaining positions at end
        self._close_all_positions(aligned_data.iloc[-1])
        
        # Calculate and return results
        return self._calculate_results()
    
    def _align_data(self, data: Dict[str, pd.DataFrame]) -> pd.DataFrame:
        """Align multiple dataframes to common timeline"""
        if not data:
            return pd.DataFrame()
        
        # Get the first symbol's data as base
        base_symbol = list(data.keys())[0]
        aligned = data[base_symbol].copy()
        
        # Ensure we have required columns
        required_cols = ['open', 'high', 'low', 'close', 'volume']
        for col in required_cols:
            if col not in aligned.columns:
                logger.error(f"Missing required column: {col}")
                return pd.DataFrame()
        
        # Add data for other symbols
        for symbol, df in data.items():
            if symbol == base_symbol:
                continue
            
            # Merge data with suffix
            for col in required_cols:
                if col in df.columns:
                    aligned[f'{symbol}_{col}'] = df[col]
        
        # Forward fill missing data
        aligned = aligned.ffill()
        
        # Drop rows with any NaN values
        aligned = aligned.dropna()
        
        return aligned
    
    def _process_pending_orders(self, market_data: pd.Series):
        """Process pending limit and stop orders"""
        filled_orders = []
        
        for order in self.pending_orders[:]:  # Copy list to allow modification
            symbol_price = self._get_current_price(order.symbol, market_data)
            
            if order.order_type == OrderType.LIMIT:
                # Check if limit order can be filled
                if order.side == 'BUY' and symbol_price <= order.price:
                    self._fill_order(order, symbol_price)
                    filled_orders.append(order)
                elif order.side == 'SELL' and symbol_price >= order.price:
                    self._fill_order(order, symbol_price)
                    filled_orders.append(order)
                    
            elif order.order_type == OrderType.STOP_LOSS:
                # Check if stop loss triggered
                if order.side == 'SELL' and symbol_price <= order.stop_price:
                    self._fill_order(order, symbol_price)
                    filled_orders.append(order)
                elif order.side == 'BUY' and symbol_price >= order.stop_price:
                    self._fill_order(order, symbol_price)
                    filled_orders.append(order)
        
        # Remove filled orders from pending
        for order in filled_orders:
            self.pending_orders.remove(order)
    
    def _update_positions(self, market_data: pd.Series):
        """Update positions with current market prices"""
        positions_to_close = []
        
        for symbol, position in self.positions.items():
            current_price = self._get_current_price(symbol, market_data)
            
            # Update unrealized P&L
            position.unrealized_pnl = position.calculate_pnl(current_price)
            
            # Check stop loss
            if position.stop_loss:
                if (position.side == PositionSide.LONG and current_price <= position.stop_loss) or \
                   (position.side == PositionSide.SHORT and current_price >= position.stop_loss):
                    positions_to_close.append((symbol, 'stop_loss'))
                    continue
            
            # Check take profit
            if position.take_profit:
                if (position.side == PositionSide.LONG and current_price >= position.take_profit) or \
                   (position.side == PositionSide.SHORT and current_price <= position.take_profit):
                    positions_to_close.append((symbol, 'take_profit'))
        
        # Close positions that hit stop loss or take profit
        for symbol, reason in positions_to_close:
            self._close_position(symbol, market_data, reason)
    
    def _check_margin_requirements(self, market_data: pd.Series):
        """Check margin requirements and trigger margin calls if needed"""
        if not self.config.use_leverage:
            return
        
        total_margin_required = 0.0
        total_equity = self._calculate_total_equity(market_data)
        
        for position in self.positions.values():
            total_margin_required += position.margin_required
        
        # Calculate margin level
        if total_margin_required > 0:
            margin_level = total_equity / total_margin_required
            
            # Trigger margin call if below threshold
            if margin_level < self.config.margin_call_level:
                logger.warning(f"Margin call triggered! Margin level: {margin_level:.2%}")
                self.metrics['margin_calls'] += 1
                
                # Close positions to restore margin
                self._handle_margin_call(market_data)
    
    def _handle_margin_call(self, market_data: pd.Series):
        """Handle margin call by closing positions"""
        # Sort positions by unrealized loss (close biggest losers first)
        positions_by_loss = sorted(
            self.positions.items(),
            key=lambda x: x[1].unrealized_pnl
        )
        
        # Close positions until margin is restored
        for symbol, position in positions_by_loss:
            if position.unrealized_pnl < 0:
                self._close_position(symbol, market_data, 'margin_call')
                
                # Check if margin restored
                total_equity = self._calculate_total_equity(market_data)
                total_margin_required = sum(p.margin_required for p in self.positions.values())
                
                if total_margin_required == 0 or total_equity / total_margin_required >= 1.0:
                    break
    
    def _execute_signals(self, signals: List[Dict], market_data: pd.Series):
        """Execute trading signals"""
        for signal in signals:
            try:
                symbol = signal.get('symbol')
                action = signal.get('action')
                
                if not symbol or not action:
                    continue
                
                if action == 'BUY':
                    self._open_long_position(symbol, signal, market_data)
                elif action == 'SELL':
                    if symbol in self.positions:
                        self._close_position(symbol, market_data, 'signal')
                    elif self.config.allow_shorting and symbol not in self.positions:
                        self._open_short_position(symbol, signal, market_data)
                elif action == 'CLOSE':
                    if symbol in self.positions:
                        self._close_position(symbol, market_data, 'signal')
                        
            except Exception as e:
                logger.error(f"Error executing signal: {e}")
    
    def _open_long_position(self, symbol: str, signal: Dict, market_data: pd.Series):
        """Open a long position"""
        if symbol in self.positions:
            logger.debug(f"Position already exists for {symbol}")
            return
        
        current_price = self._get_current_price(symbol, market_data)
        
        # Apply slippage for market orders
        if signal.get('order_type', 'MARKET') == 'MARKET':
            entry_price = current_price * (1 + self.config.slippage_rate)
        else:
            entry_price = signal.get('price', current_price)
        
        # Calculate position size
        position_size = self._calculate_position_size(signal, entry_price)
        if position_size <= 0:
            logger.debug(f"Invalid position size for {symbol}")
            return
        
        # Calculate costs
        position_value = position_size * entry_price
        commission = position_value * self.config.commission_rate
        slippage_cost = position_size * (entry_price - current_price)
        
        # Check available capital
        leverage = signal.get('leverage', 1.0) if self.config.use_leverage else 1.0
        margin_required = position_value / leverage
        
        if margin_required + commission > self.cash:
            logger.debug(f"Insufficient funds for {symbol} position")
            return
        
        # Create position
        position = Position(
            symbol=symbol,
            side=PositionSide.LONG,
            entry_price=entry_price,
            quantity=position_size,
            entry_time=self.current_time,
            stop_loss=signal.get('stop_loss'),
            take_profit=signal.get('take_profit'),
            leverage=leverage,
            entry_commission=commission,
            entry_slippage=slippage_cost
        )
        
        # Update state
        self.cash -= (margin_required + commission)
        self.margin_used += margin_required
        self.positions[symbol] = position
        self.metrics['total_commission'] += commission
        self.metrics['total_slippage'] += slippage_cost
        
        logger.debug(f"Opened LONG position: {symbol} @ {entry_price:.2f}, size: {position_size:.4f}")
    
    def _open_short_position(self, symbol: str, signal: Dict, market_data: pd.Series):
        """Open a short position"""
        if not self.config.allow_shorting:
            return
        
        if symbol in self.positions:
            logger.debug(f"Position already exists for {symbol}")
            return
        
        current_price = self._get_current_price(symbol, market_data)
        
        # Apply slippage (worse price for short)
        if signal.get('order_type', 'MARKET') == 'MARKET':
            entry_price = current_price * (1 - self.config.slippage_rate)
        else:
            entry_price = signal.get('price', current_price)
        
        # Calculate position size
        position_size = self._calculate_position_size(signal, entry_price)
        if position_size <= 0:
            return
        
        # Calculate costs
        position_value = position_size * entry_price
        commission = position_value * self.config.commission_rate
        slippage_cost = position_size * (current_price - entry_price)
        
        # For short positions, we receive cash but need margin
        leverage = signal.get('leverage', 1.0) if self.config.use_leverage else 1.0
        margin_required = position_value / leverage
        
        if margin_required > self.cash:
            logger.debug(f"Insufficient margin for short {symbol}")
            return
        
        # Create position
        position = Position(
            symbol=symbol,
            side=PositionSide.SHORT,
            entry_price=entry_price,
            quantity=position_size,
            entry_time=self.current_time,
            stop_loss=signal.get('stop_loss'),
            take_profit=signal.get('take_profit'),
            leverage=leverage,
            entry_commission=commission,
            entry_slippage=slippage_cost
        )
        
        # Update state.
        # Short-sale proceeds are NOT added to cash here: the buy-back cost is
        # already captured in gross_pnl at close (proceeds - buyback = pnl).
        # Adding proceeds here too double-counted the position value.
        self.cash -= (margin_required + commission)  # Reserve margin, pay commission
        self.margin_used += margin_required
        self.positions[symbol] = position
        self.metrics['total_commission'] += commission
        self.metrics['total_slippage'] += slippage_cost
        
        logger.debug(f"Opened SHORT position: {symbol} @ {entry_price:.2f}, size: {position_size:.4f}")
    
    def _close_position(self, symbol: str, market_data: pd.Series, reason: str):
        """Close an existing position"""
        if symbol not in self.positions:
            return
        
        position = self.positions[symbol]
        current_price = self._get_current_price(symbol, market_data)
        
        # Apply slippage
        if position.side == PositionSide.LONG:
            exit_price = current_price * (1 - self.config.slippage_rate)
        else:  # SHORT
            exit_price = current_price * (1 + self.config.slippage_rate)
        
        # Calculate P&L
        if position.side == PositionSide.LONG:
            gross_pnl = (exit_price - position.entry_price) * position.quantity
            cash_received = position.quantity * exit_price
        else:  # SHORT
            gross_pnl = (position.entry_price - exit_price) * position.quantity
            cash_paid = position.quantity * exit_price  # Buy back shares
            cash_received = -cash_paid  # Negative because we pay
        
        # Calculate costs (entry costs were stored on the position)
        exit_value = abs(position.quantity * exit_price)
        commission = exit_value * self.config.commission_rate
        slippage_cost = abs(position.quantity * (current_price - exit_price))
        total_commission = commission + position.entry_commission
        total_slippage = slippage_cost + position.entry_slippage

        # Net P&L after ALL costs (both legs)
        net_pnl = gross_pnl - total_commission - slippage_cost
        pnl_percentage = net_pnl / (position.quantity * position.entry_price)

        # Create trade result
        trade = TradeResult(
            symbol=symbol,
            side=position.side,
            entry_price=position.entry_price,
            exit_price=exit_price,
            quantity=position.quantity,
            entry_time=position.entry_time,
            exit_time=self.current_time,
            pnl=net_pnl,
            pnl_percentage=pnl_percentage,
            commission=total_commission,
            slippage=total_slippage,
            strategy='backtest',
            exit_reason=reason
        )
        
        # Update state
        if position.side == PositionSide.LONG:
            self.cash += (cash_received - commission)
        else:  # SHORT
            # Return margin and adjust for P&L
            self.cash += (position.margin_required + gross_pnl - commission)
        
        self.margin_used -= position.margin_required
        del self.positions[symbol]
        self.trades.append(trade)
        self.metrics['total_commission'] += commission
        self.metrics['total_slippage'] += slippage_cost
        
        logger.debug(
            f"Closed {position.side.value} position: {symbol} @ {exit_price:.2f}, "
            f"PnL: {net_pnl:.2f} ({pnl_percentage:.2%})"
        )
    
    def _calculate_position_size(self, signal: Dict, price: float) -> float:
        """Calculate position size based on risk management rules"""
        # Get base position size from signal or use default
        base_size_pct = signal.get('size', self.config.max_position_size)
        
        # Calculate position value
        position_value = self.cash * base_size_pct
        
        # Apply leverage if specified
        if self.config.use_leverage and 'leverage' in signal:
            leverage = min(signal['leverage'], self.config.max_leverage)
            position_value *= leverage
        
        # Convert to quantity
        quantity = position_value / price
        
        # Apply position limits
        max_position_value = self.cash * self.config.max_position_size
        max_quantity = max_position_value / price
        
        return min(quantity, max_quantity)
    
    def _get_current_price(self, symbol: str, market_data: pd.Series) -> float:
        """Get current price for a symbol"""
        # Try to get symbol-specific price first
        price_key = f'{symbol}_close'
        if price_key in market_data:
            return market_data[price_key]
        
        # Fall back to generic close price
        if 'close' in market_data:
            return market_data['close']
        
        raise ValueError(f"No price data found for {symbol}")
    
    def _calculate_total_equity(self, market_data: pd.Series) -> float:
        """Calculate total account equity"""
        positions_value = 0.0
        
        for symbol, position in self.positions.items():
            try:
                current_price = self._get_current_price(symbol, market_data)
                position_pnl = position.calculate_pnl(current_price)
                
                if position.side == PositionSide.LONG:
                    positions_value += position.quantity * current_price
                else:  # SHORT
                    # For short positions, we have the margin + unrealized P&L
                    positions_value += position.margin_required + position_pnl
                    
            except Exception as e:
                logger.error(f"Error calculating position value for {symbol}: {e}")
        
        return self.cash + positions_value
    
    def _close_all_positions(self, market_data: pd.Series):
        """Close all remaining positions at end of backtest"""
        symbols_to_close = list(self.positions.keys())
        for symbol in symbols_to_close:
            self._close_position(symbol, market_data, 'end_of_backtest')
    
    def _fill_order(self, order: Order, fill_price: float):
        """Fill a pending order"""
        order.status = OrderStatus.FILLED
        order.filled_quantity = order.quantity
        order.average_fill_price = fill_price
        
        # Calculate commission
        order.commission = order.quantity * fill_price * self.config.commission_rate
        
        # Calculate slippage
        if order.order_type == OrderType.MARKET:
            # Market orders have slippage
            if order.side == 'BUY':
                order.slippage = order.quantity * fill_price * self.config.slippage_rate
            else:
                order.slippage = order.quantity * fill_price * self.config.slippage_rate
        
        logger.debug(f"Filled order {order.id}: {order.side} {order.quantity} @ {fill_price}")
    
    def _calculate_results(self) -> BacktestResult:
        """Calculate comprehensive backtest results"""
        if not self.equity_curve:
            return self._empty_result()
        
        # Convert equity curve to DataFrame
        equity_df = pd.DataFrame(self.equity_curve)
        equity_df.set_index('timestamp', inplace=True)
        
        # Calculate returns
        equity_series = equity_df['equity']
        daily_returns = equity_series.pct_change().dropna()
        
        # Basic metrics
        total_return = (equity_series.iloc[-1] / equity_series.iloc[0]) - 1
        
        # Annualized return
        days = (equity_df.index[-1] - equity_df.index[0]).days
        years = max(days / 365.25, 1/365.25)  # Minimum 1 day
        annualized_return = (1 + total_return) ** (1/years) - 1
        
        # Risk metrics
        sharpe_ratio = self._calculate_sharpe_ratio(daily_returns)
        sortino_ratio = self._calculate_sortino_ratio(daily_returns)
        max_drawdown = self._calculate_max_drawdown(equity_series)
        calmar_ratio = annualized_return / abs(max_drawdown) if max_drawdown != 0 else 0
        
        # Trade statistics
        if self.trades:
            winning_trades = [t for t in self.trades if t.pnl > 0]
            losing_trades = [t for t in self.trades if t.pnl <= 0]
            
            win_rate = len(winning_trades) / len(self.trades)
            avg_win = np.mean([t.pnl for t in winning_trades]) if winning_trades else 0
            avg_loss = np.mean([t.pnl for t in losing_trades]) if losing_trades else 0
            
            gross_profit = sum(t.pnl for t in winning_trades)
            gross_loss = abs(sum(t.pnl for t in losing_trades))
            profit_factor = gross_profit / gross_loss if gross_loss > 0 else float('inf')
            
            best_trade = max(t.pnl_percentage for t in self.trades)
            worst_trade = min(t.pnl_percentage for t in self.trades)
            
            # Average trade duration
            durations = [(t.exit_time - t.entry_time) for t in self.trades]
            avg_duration = sum(durations, timedelta()) / len(durations)
        else:
            win_rate = 0
            avg_win = 0
            avg_loss = 0
            profit_factor = 0
            best_trade = 0
            worst_trade = 0
            avg_duration = timedelta()
        
        # Advanced metrics
        omega_ratio = self._calculate_omega_ratio(daily_returns)
        var_95 = self._calculate_var(daily_returns, 0.95)
        cvar_95 = self._calculate_cvar(daily_returns, 0.95)
        
        return BacktestResult(
            trades=self.trades,
            equity_curve=equity_series,
            daily_returns=daily_returns,
            total_return=total_return,
            annualized_return=annualized_return,
            sharpe_ratio=sharpe_ratio,
            sortino_ratio=sortino_ratio,
            max_drawdown=max_drawdown,
            win_rate=win_rate,
            profit_factor=profit_factor,
            avg_win=avg_win,
            avg_loss=avg_loss,
            best_trade=best_trade,
            worst_trade=worst_trade,
            avg_trade_duration=avg_duration,
            total_commission=self.metrics['total_commission'],
            total_slippage=self.metrics['total_slippage'],
            calmar_ratio=calmar_ratio,
            omega_ratio=omega_ratio,
            var_95=var_95,
            cvar_95=cvar_95,
            metadata={
                'total_trades': len(self.trades),
                'margin_calls': self.metrics['margin_calls'],
                'initial_capital': self.initial_capital,
                'final_capital': equity_series.iloc[-1]
            }
        )
    
    def _calculate_sharpe_ratio(self, returns: pd.Series) -> float:
        """Calculate Sharpe ratio"""
        if len(returns) < 2:
            return 0.0
        
        # Daily risk-free rate
        daily_rf = self.config.risk_free_rate / 252
        excess_returns = returns - daily_rf
        
        if returns.std() == 0:
            return 0.0
        
        # Annualized Sharpe ratio
        return np.sqrt(252) * excess_returns.mean() / returns.std()
    
    def _calculate_sortino_ratio(self, returns: pd.Series) -> float:
        """Calculate Sortino ratio (uses downside deviation)"""
        if len(returns) < 2:
            return 0.0
        
        # Daily risk-free rate
        daily_rf = self.config.risk_free_rate / 252
        excess_returns = returns - daily_rf
        
        # Downside deviation
        downside_returns = returns[returns < 0]
        if len(downside_returns) == 0:
            return float('inf')  # No downside risk
        
        downside_std = downside_returns.std()
        if downside_std == 0:
            return 0.0
        
        # Annualized Sortino ratio
        return np.sqrt(252) * excess_returns.mean() / downside_std
    
    def _calculate_max_drawdown(self, equity_series: pd.Series) -> float:
        """Calculate maximum drawdown"""
        if len(equity_series) < 2:
            return 0.0
        
        # Calculate running maximum
        running_max = equity_series.expanding().max()
        drawdown = (equity_series - running_max) / running_max
        
        return drawdown.min()
    
    def _calculate_omega_ratio(self, returns: pd.Series, threshold: float = 0) -> float:
        """Calculate Omega ratio"""
        if len(returns) == 0:
            return 0.0
        
        returns_above = returns[returns > threshold] - threshold
        returns_below = threshold - returns[returns <= threshold]
        
        if len(returns_below) == 0 or returns_below.sum() == 0:
            return float('inf')
        
        return returns_above.sum() / returns_below.sum()
    
    def _calculate_var(self, returns: pd.Series, confidence: float = 0.95) -> float:
        """Calculate Value at Risk"""
        if len(returns) == 0:
            return 0.0
        
        return np.percentile(returns, (1 - confidence) * 100)
    
    def _calculate_cvar(self, returns: pd.Series, confidence: float = 0.95) -> float:
        """Calculate Conditional Value at Risk (Expected Shortfall)"""
        var = self._calculate_var(returns, confidence)
        conditional_returns = returns[returns <= var]
        
        return conditional_returns.mean() if len(conditional_returns) > 0 else var
    
    def _empty_result(self) -> BacktestResult:
        """Return empty result structure"""
        return BacktestResult(
            trades=[],
            equity_curve=pd.Series(dtype=float),
            daily_returns=pd.Series(dtype=float),
            total_return=0,
            annualized_return=0,
            sharpe_ratio=0,
            sortino_ratio=0,
            max_drawdown=0,
            win_rate=0,
            profit_factor=0,
            avg_win=0,
            avg_loss=0,
            best_trade=0,
            worst_trade=0,
            avg_trade_duration=timedelta(),
            total_commission=0,
            total_slippage=0,
            calmar_ratio=0,
            omega_ratio=0,
            var_95=0,
            cvar_95=0
        )
