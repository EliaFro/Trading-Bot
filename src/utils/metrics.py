"""
Performance metrics and calculations for AI Crypto Trading System
Provides comprehensive metrics calculation and tracking
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple, Union
from datetime import datetime, timedelta
from dataclasses import dataclass, field
import logging
from collections import defaultdict
from scipy import stats
import warnings
warnings.filterwarnings('ignore')

logger = logging.getLogger(__name__)

@dataclass
class PerformanceMetrics:
    """Container for all performance metrics"""
    # Returns
    total_return: float = 0.0
    annualized_return: float = 0.0
    monthly_returns: Dict[str, float] = field(default_factory=dict)
    
    # Risk metrics
    volatility: float = 0.0
    sharpe_ratio: float = 0.0
    sortino_ratio: float = 0.0
    calmar_ratio: float = 0.0
    omega_ratio: float = 0.0
    
    # Drawdown metrics
    max_drawdown: float = 0.0
    max_drawdown_duration: int = 0
    current_drawdown: float = 0.0
    drawdown_periods: List[Dict] = field(default_factory=list)
    
    # Trade statistics
    total_trades: int = 0
    winning_trades: int = 0
    losing_trades: int = 0
    win_rate: float = 0.0
    avg_win: float = 0.0
    avg_loss: float = 0.0
    profit_factor: float = 0.0
    expectancy: float = 0.0
    
    # Risk metrics
    var_95: float = 0.0  # Value at Risk
    cvar_95: float = 0.0  # Conditional Value at Risk
    
    # Other metrics
    avg_trade_duration: timedelta = timedelta()
    total_commission: float = 0.0
    total_slippage: float = 0.0
    
    # Time-based metrics
    best_day: Tuple[datetime, float] = (datetime.now(), 0.0)
    worst_day: Tuple[datetime, float] = (datetime.now(), 0.0)
    best_month: Tuple[str, float] = ("", 0.0)
    worst_month: Tuple[str, float] = ("", 0.0)

class MetricsCalculator:
    """Calculate various trading performance metrics"""
    
    def __init__(self, risk_free_rate: float = 0.02):
        self.risk_free_rate = risk_free_rate  # Annual risk-free rate
        self.daily_rf = risk_free_rate / 252  # Daily risk-free rate
        
    def calculate_all_metrics(
        self,
        trades: pd.DataFrame,
        equity_curve: pd.Series,
        initial_capital: float = 10000
    ) -> PerformanceMetrics:
        """Calculate all performance metrics"""
        
        metrics = PerformanceMetrics()
        
        if trades.empty or equity_curve.empty:
            return metrics
        
        # Calculate returns
        returns = self.calculate_returns(equity_curve)
        
        # Return metrics
        metrics.total_return = self.calculate_total_return(equity_curve, initial_capital)
        metrics.annualized_return = self.calculate_annualized_return(
            metrics.total_return,
            (equity_curve.index[-1] - equity_curve.index[0]).days
        )
        metrics.monthly_returns = self.calculate_monthly_returns(equity_curve)
        
        # Risk metrics
        metrics.volatility = self.calculate_volatility(returns)
        metrics.sharpe_ratio = self.calculate_sharpe_ratio(returns)
        metrics.sortino_ratio = self.calculate_sortino_ratio(returns)
        
        # Drawdown metrics
        dd_stats = self.calculate_drawdown_statistics(equity_curve)
        metrics.max_drawdown = dd_stats['max_drawdown']
        metrics.max_drawdown_duration = dd_stats['max_duration']
        metrics.current_drawdown = dd_stats['current_drawdown']
        metrics.drawdown_periods = dd_stats['periods']
        
        metrics.calmar_ratio = self.calculate_calmar_ratio(
            metrics.annualized_return,
            metrics.max_drawdown
        )
        
        # Trade statistics
        trade_stats = self.calculate_trade_statistics(trades)
        metrics.total_trades = trade_stats['total_trades']
        metrics.winning_trades = trade_stats['winning_trades']
        metrics.losing_trades = trade_stats['losing_trades']
        metrics.win_rate = trade_stats['win_rate']
        metrics.avg_win = trade_stats['avg_win']
        metrics.avg_loss = trade_stats['avg_loss']
        metrics.profit_factor = trade_stats['profit_factor']
        metrics.expectancy = trade_stats['expectancy']
        metrics.avg_trade_duration = trade_stats['avg_duration']
        
        # Advanced risk metrics
        metrics.omega_ratio = self.calculate_omega_ratio(returns)
        metrics.var_95 = self.calculate_var(returns, 0.95)
        metrics.cvar_95 = self.calculate_cvar(returns, 0.95)
        
        # Cost metrics
        if 'commission' in trades.columns:
            metrics.total_commission = trades['commission'].sum()
        if 'slippage' in trades.columns:
            metrics.total_slippage = trades['slippage'].sum()
        
        # Time-based extremes
        daily_returns = returns.resample('D').sum()
        if not daily_returns.empty:
            metrics.best_day = (
                daily_returns.idxmax(),
                daily_returns.max()
            )
            metrics.worst_day = (
                daily_returns.idxmin(),
                daily_returns.min()
            )
        
        if metrics.monthly_returns:
            best_month = max(metrics.monthly_returns.items(), key=lambda x: x[1])
            worst_month = min(metrics.monthly_returns.items(), key=lambda x: x[1])
            metrics.best_month = best_month
            metrics.worst_month = worst_month
        
        return metrics
    
    def calculate_returns(self, equity_curve: pd.Series) -> pd.Series:
        """Calculate returns from equity curve"""
        return equity_curve.pct_change().dropna()
    
    def calculate_total_return(
        self,
        equity_curve: pd.Series,
        initial_capital: float
    ) -> float:
        """Calculate total return"""
        if equity_curve.empty:
            return 0.0
        
        return (equity_curve.iloc[-1] - initial_capital) / initial_capital
    
    def calculate_annualized_return(
        self,
        total_return: float,
        days: int
    ) -> float:
        """Calculate annualized return"""
        if days <= 0:
            return 0.0
        
        years = days / 365.25
        return (1 + total_return) ** (1 / years) - 1
    
    def calculate_monthly_returns(self, equity_curve: pd.Series) -> Dict[str, float]:
        """Calculate returns by month"""
        if equity_curve.empty:
            return {}
        
        monthly_returns = {}
        
        # Group by month
        monthly_equity = equity_curve.resample('M').last()
        
        for i in range(1, len(monthly_equity)):
            month_str = monthly_equity.index[i].strftime('%Y-%m')
            month_return = (monthly_equity.iloc[i] - monthly_equity.iloc[i-1]) / monthly_equity.iloc[i-1]
            monthly_returns[month_str] = month_return
        
        return monthly_returns
    
    def calculate_volatility(
        self,
        returns: pd.Series,
        annualize: bool = True
    ) -> float:
        """Calculate volatility (standard deviation of returns)"""
        if len(returns) < 2:
            return 0.0
        
        vol = returns.std()
        
        if annualize:
            # Annualize assuming daily returns
            vol *= np.sqrt(252)
        
        return vol
    
    def calculate_sharpe_ratio(self, returns: pd.Series) -> float:
        """Calculate Sharpe ratio"""
        if len(returns) < 2:
            return 0.0
        
        excess_returns = returns - self.daily_rf
        
        if returns.std() == 0:
            return 0.0
        
        # Annualized Sharpe ratio
        return np.sqrt(252) * excess_returns.mean() / returns.std()
    
    def calculate_sortino_ratio(self, returns: pd.Series) -> float:
        """Calculate Sortino ratio (uses downside deviation)"""
        if len(returns) < 2:
            return 0.0
        
        excess_returns = returns - self.daily_rf
        
        # Calculate downside deviation
        downside_returns = returns[returns < 0]
        
        if len(downside_returns) == 0:
            return float('inf')  # No downside risk
        
        downside_std = downside_returns.std()
        
        if downside_std == 0:
            return 0.0
        
        # Annualized Sortino ratio
        return np.sqrt(252) * excess_returns.mean() / downside_std
    
    def calculate_calmar_ratio(
        self,
        annualized_return: float,
        max_drawdown: float
    ) -> float:
        """Calculate Calmar ratio"""
        if max_drawdown == 0:
            return 0.0
        
        return annualized_return / abs(max_drawdown)
    
    def calculate_omega_ratio(
        self,
        returns: pd.Series,
        threshold: float = 0
    ) -> float:
        """Calculate Omega ratio"""
        if len(returns) == 0:
            return 0.0
        
        returns_above = returns[returns > threshold] - threshold
        returns_below = threshold - returns[returns <= threshold]
        
        if len(returns_below) == 0 or returns_below.sum() == 0:
            return float('inf')
        
        return returns_above.sum() / returns_below.sum()
    
    def calculate_drawdown_statistics(
        self,
        equity_curve: pd.Series
    ) -> Dict:
        """Calculate comprehensive drawdown statistics"""
        
        # Calculate running maximum
        running_max = equity_curve.expanding().max()
        
        # Calculate drawdown series
        drawdown = (equity_curve - running_max) / running_max
        
        # Maximum drawdown
        max_drawdown = drawdown.min()
        
        # Current drawdown
        current_drawdown = drawdown.iloc[-1]
        
        # Drawdown periods
        periods = []
        in_drawdown = False
        start_idx = None
        
        for i in range(len(drawdown)):
            if drawdown.iloc[i] < 0 and not in_drawdown:
                # Start of drawdown
                in_drawdown = True
                start_idx = i
            elif drawdown.iloc[i] == 0 and in_drawdown:
                # End of drawdown
                in_drawdown = False
                if start_idx is not None:
                    period = {
                        'start': equity_curve.index[start_idx],
                        'end': equity_curve.index[i],
                        'depth': drawdown.iloc[start_idx:i].min(),
                        'duration': i - start_idx
                    }
                    periods.append(period)
        
        # If still in drawdown
        if in_drawdown and start_idx is not None:
            period = {
                'start': equity_curve.index[start_idx],
                'end': equity_curve.index[-1],
                'depth': drawdown.iloc[start_idx:].min(),
                'duration': len(drawdown) - start_idx
            }
            periods.append(period)
        
        # Maximum drawdown duration
        max_duration = max([p['duration'] for p in periods]) if periods else 0
        
        return {
            'max_drawdown': max_drawdown,
            'current_drawdown': current_drawdown,
            'max_duration': max_duration,
            'periods': periods,
            'drawdown_series': drawdown
        }
    
    def calculate_trade_statistics(self, trades: pd.DataFrame) -> Dict:
        """Calculate trade-based statistics"""
        
        if trades.empty:
            return {
                'total_trades': 0,
                'winning_trades': 0,
                'losing_trades': 0,
                'win_rate': 0.0,
                'avg_win': 0.0,
                'avg_loss': 0.0,
                'profit_factor': 0.0,
                'expectancy': 0.0,
                'avg_duration': timedelta()
            }
        
        # Filter closed trades
        closed_trades = trades[trades['status'] == 'CLOSED'] if 'status' in trades.columns else trades
        
        if closed_trades.empty:
            return self._empty_trade_stats()
        
        # Basic counts
        total_trades = len(closed_trades)
        winning_trades = len(closed_trades[closed_trades['pnl'] > 0])
        losing_trades = len(closed_trades[closed_trades['pnl'] <= 0])
        
        # Win rate
        win_rate = winning_trades / total_trades if total_trades > 0 else 0
        
        # Average win/loss
        wins = closed_trades[closed_trades['pnl'] > 0]['pnl']
        losses = closed_trades[closed_trades['pnl'] <= 0]['pnl']
        
        avg_win = wins.mean() if len(wins) > 0 else 0
        avg_loss = losses.mean() if len(losses) > 0 else 0
        
        # Profit factor
        gross_profit = wins.sum() if len(wins) > 0 else 0
        gross_loss = abs(losses.sum()) if len(losses) > 0 else 0
        profit_factor = gross_profit / gross_loss if gross_loss > 0 else float('inf')
        
        # Expectancy
        expectancy = (win_rate * avg_win) + ((1 - win_rate) * avg_loss)
        
        # Average duration
        if 'entry_time' in closed_trades.columns and 'exit_time' in closed_trades.columns:
            durations = pd.to_datetime(closed_trades['exit_time']) - pd.to_datetime(closed_trades['entry_time'])
            avg_duration = durations.mean()
        else:
            avg_duration = timedelta()
        
        return {
            'total_trades': total_trades,
            'winning_trades': winning_trades,
            'losing_trades': losing_trades,
            'win_rate': win_rate,
            'avg_win': avg_win,
            'avg_loss': avg_loss,
            'profit_factor': profit_factor,
            'expectancy': expectancy,
            'avg_duration': avg_duration
        }
    
    def calculate_var(
        self,
        returns: pd.Series,
        confidence: float = 0.95
    ) -> float:
        """Calculate Value at Risk"""
        if len(returns) == 0:
            return 0.0
        
        # Historical VaR
        return np.percentile(returns, (1 - confidence) * 100)
    
    def calculate_cvar(
        self,
        returns: pd.Series,
        confidence: float = 0.95
    ) -> float:
        """Calculate Conditional Value at Risk (Expected Shortfall)"""
        var = self.calculate_var(returns, confidence)
        conditional_returns = returns[returns <= var]
        
        return conditional_returns.mean() if len(conditional_returns) > 0 else var
    
    def calculate_rolling_metrics(
        self,
        equity_curve: pd.Series,
        window: int = 252  # One year default
    ) -> pd.DataFrame:
        """Calculate rolling performance metrics"""
        
        returns = self.calculate_returns(equity_curve)
        
        rolling_metrics = pd.DataFrame(index=equity_curve.index)
        
        # Rolling returns
        rolling_metrics['rolling_return'] = returns.rolling(window).sum()
        
        # Rolling volatility
        rolling_metrics['rolling_volatility'] = returns.rolling(window).std() * np.sqrt(252)
        
        # Rolling Sharpe
        rolling_sharpe = []
        for i in range(len(returns)):
            if i < window:
                rolling_sharpe.append(np.nan)
            else:
                window_returns = returns.iloc[i-window:i]
                sharpe = self.calculate_sharpe_ratio(window_returns)
                rolling_sharpe.append(sharpe)
        
        rolling_metrics['rolling_sharpe'] = rolling_sharpe
        
        # Rolling max drawdown
        rolling_dd = []
        for i in range(len(equity_curve)):
            if i < window:
                rolling_dd.append(np.nan)
            else:
                window_equity = equity_curve.iloc[i-window:i]
                dd_stats = self.calculate_drawdown_statistics(window_equity)
                rolling_dd.append(dd_stats['max_drawdown'])
        
        rolling_metrics['rolling_max_drawdown'] = rolling_dd
        
        return rolling_metrics
    
    def calculate_correlation_metrics(
        self,
        returns: pd.Series,
        benchmark_returns: pd.Series
    ) -> Dict:
        """Calculate correlation-based metrics"""
        
        if len(returns) < 2 or len(benchmark_returns) < 2:
            return {
                'correlation': 0.0,
                'beta': 0.0,
                'alpha': 0.0,
                'tracking_error': 0.0,
                'information_ratio': 0.0
            }
        
        # Align series
        aligned = pd.DataFrame({
            'returns': returns,
            'benchmark': benchmark_returns
        }).dropna()
        
        if len(aligned) < 2:
            return self._empty_correlation_metrics()
        
        # Correlation
        correlation = aligned['returns'].corr(aligned['benchmark'])
        
        # Beta (systematic risk)
        covariance = aligned.cov().loc['returns', 'benchmark']
        benchmark_variance = aligned['benchmark'].var()
        beta = covariance / benchmark_variance if benchmark_variance > 0 else 0
        
        # Alpha (excess return)
        # Using CAPM: return = alpha + beta * benchmark_return + error
        annualized_return = aligned['returns'].mean() * 252
        annualized_benchmark = aligned['benchmark'].mean() * 252
        alpha = annualized_return - (self.risk_free_rate + beta * (annualized_benchmark - self.risk_free_rate))
        
        # Tracking error
        excess_returns = aligned['returns'] - aligned['benchmark']
        tracking_error = excess_returns.std() * np.sqrt(252)
        
        # Information ratio
        information_ratio = (excess_returns.mean() * 252) / tracking_error if tracking_error > 0 else 0
        
        return {
            'correlation': correlation,
            'beta': beta,
            'alpha': alpha,
            'tracking_error': tracking_error,
            'information_ratio': information_ratio
        }
    
    def calculate_strategy_metrics(
        self,
        trades: pd.DataFrame,
        by_strategy: bool = True
    ) -> pd.DataFrame:
        """Calculate metrics broken down by strategy"""
        
        if trades.empty or 'strategy' not in trades.columns:
            return pd.DataFrame()
        
        strategies = trades['strategy'].unique()
        metrics_list = []
        
        for strategy in strategies:
            strategy_trades = trades[trades['strategy'] == strategy]
            
            if strategy_trades.empty:
                continue
            
            stats = self.calculate_trade_statistics(strategy_trades)
            
            metrics_list.append({
                'strategy': strategy,
                'total_trades': stats['total_trades'],
                'win_rate': stats['win_rate'],
                'profit_factor': stats['profit_factor'],
                'avg_win': stats['avg_win'],
                'avg_loss': stats['avg_loss'],
                'expectancy': stats['expectancy'],
                'total_pnl': strategy_trades['pnl'].sum() if 'pnl' in strategy_trades else 0
            })
        
        return pd.DataFrame(metrics_list)
    
    def _empty_trade_stats(self) -> Dict:
        """Return empty trade statistics"""
        return {
            'total_trades': 0,
            'winning_trades': 0,
            'losing_trades': 0,
            'win_rate': 0.0,
            'avg_win': 0.0,
            'avg_loss': 0.0,
            'profit_factor': 0.0,
            'expectancy': 0.0,
            'avg_duration': timedelta()
        }
    
    def _empty_correlation_metrics(self) -> Dict:
        """Return empty correlation metrics"""
        return {
            'correlation': 0.0,
            'beta': 0.0,
            'alpha': 0.0,
            'tracking_error': 0.0,
            'information_ratio': 0.0
        }

class MetricsCollector:
    """Collect and store metrics for monitoring"""
    
    def __init__(self):
        self.metrics = defaultdict(list)
        self.timestamps = defaultdict(list)
        
    def record_metric(self, name: str, value: float, timestamp: Optional[datetime] = None):
        """Record a metric value"""
        if timestamp is None:
            timestamp = datetime.now()
        
        self.metrics[name].append(value)
        self.timestamps[name].append(timestamp)
        
        # Keep only last 10000 values to prevent memory issues
        if len(self.metrics[name]) > 10000:
            self.metrics[name] = self.metrics[name][-10000:]
            self.timestamps[name] = self.timestamps[name][-10000:]
    
    def get_metric(
        self,
        name: str,
        start_time: Optional[datetime] = None,
        end_time: Optional[datetime] = None
    ) -> pd.Series:
        """Get metric values as time series"""
        
        if name not in self.metrics:
            return pd.Series()
        
        # Create series
        series = pd.Series(
            self.metrics[name],
            index=self.timestamps[name]
        )
        
        # Filter by time range
        if start_time:
            series = series[series.index >= start_time]
        if end_time:
            series = series[series.index <= end_time]
        
        return series
    
    def get_latest(self, name: str) -> Optional[float]:
        """Get latest value of a metric"""
        if name in self.metrics and self.metrics[name]:
            return self.metrics[name][-1]
        return None
    
    def get_summary(self) -> Dict:
        """Get summary of all metrics"""
        summary = {}
        
        for name, values in self.metrics.items():
            if values:
                summary[name] = {
                    'latest': values[-1],
                    'mean': np.mean(values),
                    'std': np.std(values),
                    'min': min(values),
                    'max': max(values),
                    'count': len(values)
                }
        
        return summary
    
    def clear(self):
        """Clear all metrics"""
        self.metrics.clear()
        self.timestamps.clear()
    
    def export_to_prometheus(self) -> str:
        """Export metrics in Prometheus format"""
        lines = []
        
        for name, values in self.metrics.items():
            if values:
                # Clean metric name for Prometheus
                clean_name = name.replace('.', '_').replace('-', '_')
                
                # Add latest value
                lines.append(f"# TYPE {clean_name} gauge")
                lines.append(f"{clean_name} {values[-1]}")
        
        return '\n'.join(lines)

class PerformanceTracker:
    """Track performance over time"""
    
    def __init__(self, initial_capital: float = 10000):
        self.initial_capital = initial_capital
        self.current_capital = initial_capital
        self.equity_curve = []
        self.trades = []
        self.daily_returns = []
        self.metrics_calculator = MetricsCalculator()
        
    def update_equity(self, timestamp: datetime, equity: float):
        """Update equity curve"""
        self.equity_curve.append({
            'timestamp': timestamp,
            'equity': equity
        })
        self.current_capital = equity
    
    def add_trade(self, trade: Dict):
        """Add a completed trade"""
        self.trades.append(trade)
    
    def get_current_metrics(self) -> PerformanceMetrics:
        """Get current performance metrics"""
        
        if not self.equity_curve:
            return PerformanceMetrics()
        
        # Convert to pandas
        equity_df = pd.DataFrame(self.equity_curve)
        equity_series = pd.Series(
            equity_df['equity'].values,
            index=pd.to_datetime(equity_df['timestamp'])
        )
        
        trades_df = pd.DataFrame(self.trades) if self.trades else pd.DataFrame()
        
        # Calculate all metrics
        return self.metrics_calculator.calculate_all_metrics(
            trades_df,
            equity_series,
            self.initial_capital
        )
    
    def get_daily_summary(self, date: Optional[datetime] = None) -> Dict:
        """Get summary for a specific day"""
        if date is None:
            date = datetime.now().date()
        
        # Filter equity curve for the day
        day_equity = [
            e for e in self.equity_curve
            if pd.to_datetime(e['timestamp']).date() == date
        ]
        
        if not day_equity:
            return {}
        
        # Calculate daily metrics
        start_equity = day_equity[0]['equity']
        end_equity = day_equity[-1]['equity']
        daily_return = (end_equity - start_equity) / start_equity
        
        # Count trades
        day_trades = [
            t for t in self.trades
            if pd.to_datetime(t.get('entry_time', '')).date() == date
        ]
        
        return {
            'date': date,
            'starting_equity': start_equity,
            'ending_equity': end_equity,
            'daily_return': daily_return,
            'daily_pnl': end_equity - start_equity,
            'num_trades': len(day_trades),
            'winning_trades': len([t for t in day_trades if t.get('pnl', 0) > 0]),
            'losing_trades': len([t for t in day_trades if t.get('pnl', 0) <= 0])
        }
