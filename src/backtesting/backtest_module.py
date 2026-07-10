"""
Backtesting module for AI Crypto Trading System
Provides strategy implementations and backtesting utilities
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Callable, Tuple, Any
from datetime import datetime, timedelta
from dataclasses import dataclass
import logging
from abc import ABC, abstractmethod
from collections import defaultdict

try:
    import talib
except ImportError:  # pure-pandas fallback with TA-Lib-compatible signatures
    from src.utils import indicators as talib

from src.backtesting.engine import AdvancedBacktester, BacktestConfig, BacktestResult
from src.utils.indicators import TechnicalIndicators

logger = logging.getLogger(__name__)

# --- BASE STRATEGY CLASS ---

class BaseStrategy(ABC):
    """Abstract base class for trading strategies"""
    
    def __init__(self, params: Dict[str, Any] = None):
        self.params = params or {}
        self.name = self.__class__.__name__
        self.indicators = TechnicalIndicators()
        self.position_manager = PositionManager()
        
    @abstractmethod
    def generate_signals(
        self,
        data: pd.DataFrame,
        current_positions: Dict,
        portfolio_value: float,
        symbol: str = 'BTC/USDT'
    ) -> List[Dict]:
        """
        Generate trading signals for `symbol` from its OHLCV history `data`
        (most recent bar last; needs enough rows for the strategy's windows).

        Returns:
            List of signal dictionaries with format:
            {
                'symbol': 'BTC/USDT',
                'action': 'BUY' | 'SELL' | 'HOLD',
                'size': 0.1,  # Position size as fraction of portfolio
                'confidence': 0.8,  # Signal confidence
                'stop_loss': 45000,  # Optional
                'take_profit': 55000,  # Optional
                'metadata': {}  # Additional info
            }
        """
        pass
    
    def calculate_position_size(
        self,
        signal_strength: float,
        volatility: float,
        portfolio_value: float,
        current_positions: Dict
    ) -> float:
        """Calculate position size using Kelly Criterion"""
        # Kelly fraction with safety factor
        kelly_fraction = signal_strength * 0.25  # Conservative Kelly
        
        # Adjust for volatility
        vol_adjusted = kelly_fraction / (1 + volatility)
        
        # Apply maximum position size
        max_position = self.params.get('max_position_size', 0.1)
        position_size = min(vol_adjusted, max_position)
        
        # Check total exposure (positions may be dataclasses or dicts)
        def _pos_value(pos):
            if hasattr(pos, 'value'):
                return pos.value
            if isinstance(pos, dict):
                return pos.get('quantity', 0) * pos.get('entry_price', 0)
            return 0.0

        current_exposure = sum(_pos_value(pos) for pos in current_positions.values())
        max_total_exposure = self.params.get('max_total_exposure', 0.5)
        
        if (current_exposure + position_size * portfolio_value) > (max_total_exposure * portfolio_value):
            # Reduce position size to stay within limits
            available_exposure = max(0, (max_total_exposure * portfolio_value) - current_exposure)
            position_size = available_exposure / portfolio_value
        
        return position_size

# --- STRATEGY IMPLEMENTATIONS ---

class MovingAverageCrossoverStrategy(BaseStrategy):
    """Classic moving average crossover strategy"""
    
    def __init__(self, params: Dict[str, Any] = None):
        super().__init__(params)
        self.fast_period = self.params.get('fast_period', 20)
        self.slow_period = self.params.get('slow_period', 50)
        self.use_volume_filter = self.params.get('use_volume_filter', True)
        
    def generate_signals(
        self,
        data: pd.DataFrame,
        current_positions: Dict,
        portfolio_value: float,
        symbol: str = 'BTC/USDT'
    ) -> List[Dict]:

        signals = []

        if len(data) < self.slow_period + 2:
            return signals

        # Calculate moving averages
        data['sma_fast'] = data['close'].rolling(self.fast_period).mean()
        data['sma_slow'] = data['close'].rolling(self.slow_period).mean()
        
        # Volume filter
        if self.use_volume_filter:
            data['volume_sma'] = data['volume'].rolling(20).mean()
            volume_condition = data['volume'].iloc[-1] > data['volume_sma'].iloc[-1]
        else:
            volume_condition = True
        
        # Check for crossover
        current_fast = data['sma_fast'].iloc[-1]
        current_slow = data['sma_slow'].iloc[-1]
        prev_fast = data['sma_fast'].iloc[-2]
        prev_slow = data['sma_slow'].iloc[-2]
        
        # Golden cross (bullish)
        if prev_fast <= prev_slow and current_fast > current_slow and volume_condition:
            # Calculate signal strength
            cross_strength = abs(current_fast - current_slow) / current_slow
            volatility = data['close'].pct_change().rolling(20).std().iloc[-1]
            
            position_size = self.calculate_position_size(
                cross_strength,
                volatility,
                portfolio_value,
                current_positions
            )
            
            if position_size > 0:
                signals.append({
                    'symbol': symbol,
                    'action': 'BUY',
                    'size': position_size,
                    'confidence': min(cross_strength * 10, 0.9),
                    'stop_loss': data['close'].iloc[-1] * 0.98,
                    'take_profit': data['close'].iloc[-1] * 1.02,
                    'metadata': {
                        'strategy': self.name,
                        'fast_ma': current_fast,
                        'slow_ma': current_slow
                    }
                })

        # Death cross (bearish)
        elif prev_fast >= prev_slow and current_fast < current_slow:
            signals.append({
                'symbol': symbol,
                'action': 'SELL',
                'confidence': 0.6,
                'metadata': {'strategy': self.name}
            })
        
        return signals

class RSIMeanReversionStrategy(BaseStrategy):
    """RSI-based mean reversion strategy"""
    
    def __init__(self, params: Dict[str, Any] = None):
        super().__init__(params)
        self.rsi_period = self.params.get('rsi_period', 14)
        self.oversold_threshold = self.params.get('oversold_threshold', 30)
        self.overbought_threshold = self.params.get('overbought_threshold', 70)
        self.use_divergence = self.params.get('use_divergence', True)
        
    def generate_signals(
        self,
        data: pd.DataFrame,
        current_positions: Dict,
        portfolio_value: float,
        symbol: str = 'BTC/USDT'
    ) -> List[Dict]:

        signals = []

        if len(data) < max(self.rsi_period + 2, 25):
            return signals

        # Calculate RSI
        data['rsi'] = talib.RSI(data['close'].values, timeperiod=self.rsi_period)
        if pd.isna(data['rsi'].iloc[-1]):
            return signals
        
        # Check for divergence if enabled
        divergence = 0
        if self.use_divergence:
            divergence = self._check_divergence(data)
        
        current_rsi = data['rsi'].iloc[-1]
        current_price = data['close'].iloc[-1]
        
        # Oversold condition
        if current_rsi < self.oversold_threshold:
            # Calculate reversion probability
            historical_reversals = self._calculate_reversal_probability(data, 'oversold')
            
            signal_strength = (self.oversold_threshold - current_rsi) / self.oversold_threshold
            signal_strength *= (1 + divergence * 0.5)  # Boost if divergence present
            
            volatility = data['close'].pct_change().rolling(20).std().iloc[-1]
            position_size = self.calculate_position_size(
                signal_strength * historical_reversals,
                volatility,
                portfolio_value,
                current_positions
            )
            
            if position_size > 0:
                signals.append({
                    'symbol': symbol,
                    'action': 'BUY',
                    'size': position_size,
                    'confidence': signal_strength * historical_reversals,
                    'stop_loss': current_price * 0.97,
                    'take_profit': current_price * 1.015,
                    'metadata': {
                        'strategy': self.name,
                        'rsi': current_rsi,
                        'divergence': divergence
                    }
                })

        # Overbought condition
        elif current_rsi > self.overbought_threshold and symbol in current_positions:
            signals.append({
                'symbol': symbol,
                'action': 'SELL',
                'confidence': min((current_rsi - self.overbought_threshold)
                                  / (100 - self.overbought_threshold) + 0.5, 0.95),
                'metadata': {
                    'strategy': self.name,
                    'rsi': current_rsi
                }
            })
        
        return signals
    
    def _check_divergence(self, data: pd.DataFrame) -> float:
        """Check for RSI divergence"""
        # Simplified divergence detection
        lookback = 20
        
        if len(data) < lookback:
            return 0
        
        recent_data = data.tail(lookback)
        
        # Find peaks and troughs
        price_trend = np.polyfit(range(lookback), recent_data['close'].values, 1)[0]
        rsi_trend = np.polyfit(range(lookback), recent_data['rsi'].values, 1)[0]
        
        # Bullish divergence: price down, RSI up
        if price_trend < 0 and rsi_trend > 0:
            return 1.0
        # Bearish divergence: price up, RSI down
        elif price_trend > 0 and rsi_trend < 0:
            return -1.0
        
        return 0
    
    def _calculate_reversal_probability(self, data: pd.DataFrame, condition: str) -> float:
        """Calculate historical probability of reversal"""
        if len(data) < 100:
            return 0.5  # Default probability
        
        # Look at historical occurrences
        if condition == 'oversold':
            mask = data['rsi'] < self.oversold_threshold
        else:
            mask = data['rsi'] > self.overbought_threshold
        
        if mask.sum() == 0:
            return 0.5
        
        # Check how often price reversed after condition
        reversals = 0
        occurrences = 0
        
        for i in range(len(data) - 5):
            if mask.iloc[i]:
                occurrences += 1
                # Check if price moved favorably in next 5 periods
                future_return = (data['close'].iloc[i+5] - data['close'].iloc[i]) / data['close'].iloc[i]
                if condition == 'oversold' and future_return > 0:
                    reversals += 1
                elif condition == 'overbought' and future_return < 0:
                    reversals += 1
        
        return reversals / occurrences if occurrences > 0 else 0.5

class BreakoutStrategy(BaseStrategy):
    """Price breakout strategy with volume confirmation"""
    
    def __init__(self, params: Dict[str, Any] = None):
        super().__init__(params)
        self.lookback_period = self.params.get('lookback_period', 20)
        self.volume_multiplier = self.params.get('volume_multiplier', 1.5)
        self.use_atr_stops = self.params.get('use_atr_stops', True)
        self.atr_multiplier = self.params.get('atr_multiplier', 2.0)
        
    def generate_signals(
        self,
        data: pd.DataFrame,
        current_positions: Dict,
        portfolio_value: float,
        symbol: str = 'BTC/USDT'
    ) -> List[Dict]:

        signals = []

        if len(data) < max(self.lookback_period + 2, 25):
            return signals

        # Calculate indicators
        data['high_rolling'] = data['high'].rolling(self.lookback_period).max()
        data['low_rolling'] = data['low'].rolling(self.lookback_period).min()
        data['volume_sma'] = data['volume'].rolling(20).mean()
        data['atr'] = talib.ATR(data['high'].values, data['low'].values, data['close'].values)
        
        current_price = data['close'].iloc[-1]
        current_volume = data['volume'].iloc[-1]
        resistance = data['high_rolling'].iloc[-2]  # Previous period's resistance
        support = data['low_rolling'].iloc[-2]  # Previous period's support
        
        # Volume confirmation
        volume_confirmed = current_volume > (data['volume_sma'].iloc[-1] * self.volume_multiplier)
        
        # Breakout detection
        if current_price > resistance and volume_confirmed:
            # Upward breakout
            breakout_strength = (current_price - resistance) / resistance
            volatility = data['atr'].iloc[-1] / current_price
            
            position_size = self.calculate_position_size(
                breakout_strength * 2,  # Amplify signal
                volatility,
                portfolio_value,
                current_positions
            )
            
            if position_size > 0:
                # Calculate stops
                if self.use_atr_stops:
                    stop_loss = current_price - (data['atr'].iloc[-1] * self.atr_multiplier)
                    take_profit = current_price + (data['atr'].iloc[-1] * self.atr_multiplier * 2)
                else:
                    stop_loss = support
                    take_profit = current_price + (current_price - support) * 2
                
                signals.append({
                    'symbol': symbol,
                    'action': 'BUY',
                    'size': position_size,
                    'confidence': min(breakout_strength * 5 + 0.5, 0.9),
                    'stop_loss': stop_loss,
                    'take_profit': take_profit,
                    'metadata': {
                        'strategy': self.name,
                        'breakout_level': resistance,
                        'breakout_strength': breakout_strength,
                        'volume_ratio': current_volume / data['volume_sma'].iloc[-1]
                    }
                })

        elif current_price < support and volume_confirmed:
            # Downward breakout: exit signal (long-only v1 never opens shorts)
            breakout_strength = (support - current_price) / support

            signals.append({
                'symbol': symbol,
                'action': 'SELL',
                'confidence': min(breakout_strength * 5 + 0.5, 0.9),
                'metadata': {
                    'strategy': self.name,
                    'breakout_level': support,
                    'breakout_strength': breakout_strength
                }
            })
        
        return signals

class MLEnsembleStrategy(BaseStrategy):
    """Machine Learning ensemble strategy"""
    
    def __init__(self, params: Dict[str, Any] = None):
        super().__init__(params)
        self.model = params.get('model')  # Pre-trained model
        self.feature_calculator = FeatureCalculator()
        self.confidence_threshold = self.params.get('confidence_threshold', 0.6)
        self.use_sentiment = self.params.get('use_sentiment', True)
        
    def generate_signals(
        self,
        data: pd.DataFrame,
        current_positions: Dict,
        portfolio_value: float,
        symbol: str = 'BTC/USDT'
    ) -> List[Dict]:

        signals = []

        if len(data) < 40:
            return signals

        # Calculate features
        features = self.feature_calculator.calculate_features(data)
        
        # Add sentiment if available
        if self.use_sentiment:
            sentiment_score = self._get_sentiment_score()
            features['sentiment'] = sentiment_score
        
        # Get model prediction
        if self.model:
            prediction, confidence = self._get_model_prediction(features, data, symbol)
            
            if confidence > self.confidence_threshold:
                if prediction == 1:  # Buy signal
                    volatility = data['close'].pct_change().rolling(20).std().iloc[-1]
                    position_size = self.calculate_position_size(
                        confidence,
                        volatility,
                        portfolio_value,
                        current_positions
                    )
                    
                    if position_size > 0:
                        # Dynamic stop loss based on volatility
                        atr = talib.ATR(data['high'].values, data['low'].values, data['close'].values)[-1]
                        current_price = data['close'].iloc[-1]
                        
                        signals.append({
                            'symbol': symbol,
                            'action': 'BUY',
                            'size': position_size,
                            'confidence': confidence,
                            'stop_loss': current_price - (atr * 2),
                            'take_profit': current_price + (atr * 3),
                            'metadata': {
                                'strategy': self.name,
                                'model_confidence': confidence,
                                'features': features
                            }
                        })

                elif prediction == -1:  # Sell signal
                    signals.append({
                        'symbol': symbol,
                        'action': 'SELL',
                        'confidence': confidence,
                        'metadata': {
                            'strategy': self.name,
                            'model_confidence': confidence
                        }
                    })

        return signals

    def _get_model_prediction(self, features: Dict, data: pd.DataFrame,
                              symbol: str) -> Tuple[int, float]:
        """Prediction from the injected model (e.g. EnsembleModel.predict).
        Returns (0, 0.0) — no trade — when no model is available."""
        if self.model is None:
            return 0, 0.0
        try:
            if hasattr(self.model, 'predict'):
                return self.model.predict(data, symbol=symbol)
        except Exception as e:
            logger.error(f"Model prediction failed for {symbol}: {e}")
        return 0, 0.0

    def _get_sentiment_score(self) -> float:
        """Sentiment score injected via params by the trading engine.
        Neutral (0.0) when no sentiment feed is attached."""
        return float(self.params.get('sentiment_score', 0.0))

# --- HELPER CLASSES ---

class PositionManager:
    """Manages position sizing and risk"""
    
    def __init__(self, max_positions: int = 5, max_correlation: float = 0.7):
        self.max_positions = max_positions
        self.max_correlation = max_correlation
        
    def can_open_position(
        self,
        symbol: str,
        current_positions: Dict,
        correlation_matrix: Optional[pd.DataFrame] = None
    ) -> bool:
        """Check if we can open a new position"""
        
        # Check max positions
        if len(current_positions) >= self.max_positions:
            return False
        
        # Check correlation with existing positions
        if correlation_matrix is not None and symbol in current_positions:
            for existing_symbol in current_positions:
                if existing_symbol in correlation_matrix.index and symbol in correlation_matrix.columns:
                    correlation = correlation_matrix.loc[existing_symbol, symbol]
                    if abs(correlation) > self.max_correlation:
                        return False
        
        return True

class FeatureCalculator:
    """Calculate features for ML models"""
    
    def calculate_features(self, data: pd.DataFrame) -> Dict[str, float]:
        """Calculate all features for ML model"""
        
        features = {}
        
        # Price-based features
        features['returns_1h'] = (data['close'].iloc[-1] / data['close'].iloc[-2]) - 1
        features['returns_4h'] = (data['close'].iloc[-1] / data['close'].iloc[-4]) - 1 if len(data) > 4 else 0
        features['returns_24h'] = (data['close'].iloc[-1] / data['close'].iloc[-24]) - 1 if len(data) > 24 else 0
        
        # Volatility
        features['volatility_20'] = data['close'].pct_change().rolling(20).std().iloc[-1]
        
        # Technical indicators
        features['rsi'] = talib.RSI(data['close'].values)[-1]
        features['macd'], features['macd_signal'], _ = talib.MACD(data['close'].values)
        features['macd'] = features['macd'][-1] if len(features['macd']) > 0 else 0
        
        # Volume features
        features['volume_ratio'] = data['volume'].iloc[-1] / data['volume'].rolling(20).mean().iloc[-1]
        
        # Microstructure
        features['high_low_ratio'] = data['high'].iloc[-1] / data['low'].iloc[-1]
        features['close_to_high'] = (data['close'].iloc[-1] - data['low'].iloc[-1]) / (data['high'].iloc[-1] - data['low'].iloc[-1])
        
        return features

# --- STRATEGY FACTORY ---

class StrategyFactory:
    """Factory for creating strategy instances"""
    
    @staticmethod
    def create_strategy(name: str, params: Dict[str, Any] = None) -> BaseStrategy:
        """Create a strategy instance by name"""
        
        strategies = {
            'ma_crossover': MovingAverageCrossoverStrategy,
            'rsi_mean_reversion': RSIMeanReversionStrategy,
            'breakout': BreakoutStrategy,
            'ml_ensemble': MLEnsembleStrategy
        }
        
        if name not in strategies:
            raise ValueError(f"Unknown strategy: {name}")
        
        return strategies[name](params)
    
    @staticmethod
    def get_available_strategies() -> List[str]:
        """Get list of available strategies"""
        return [
            'ma_crossover',
            'rsi_mean_reversion',
            'breakout',
            'ml_ensemble'
        ]

# --- BACKTEST RUNNER ---

class BacktestRunner:
    """High-level backtest runner with analysis"""
    
    def __init__(self, config: BacktestConfig):
        self.config = config
        self.backtester = AdvancedBacktester(config)
        
    def run_strategy_backtest(
        self,
        strategy_name: str,
        strategy_params: Dict[str, Any],
        data: Dict[str, pd.DataFrame],
        start_date: Optional[datetime] = None,
        end_date: Optional[datetime] = None,
        lookback_bars: int = 200,
        model: Any = None
    ) -> BacktestResult:
        """Run backtest for a specific strategy.

        Strategies receive a rolling window of history per symbol (most
        recent bar last) — never a single row. Windows are sliced up to the
        current backtest timestamp only, so there is no look-ahead.
        """

        # Create strategy
        params = dict(strategy_params or {})
        if model is not None:
            params['model'] = model
        strategy = StrategyFactory.create_strategy(strategy_name, params)

        # Pre-sort each symbol's history once for fast slicing
        histories: Dict[str, pd.DataFrame] = {}
        for symbol, df in data.items():
            hist = df.sort_index()
            histories[symbol] = hist

        min_bars = 60  # below this, strategies can't compute their windows

        def strategy_func(market_data, positions, cash):
            timestamp = market_data.name  # backtester passes the aligned row
            signals = []
            for symbol, hist in histories.items():
                window = hist.loc[:timestamp]
                if len(window) < min_bars:
                    continue
                window = window.tail(lookback_bars).copy()
                try:
                    signals.extend(strategy.generate_signals(
                        window, positions, cash, symbol=symbol) or [])
                except Exception as e:
                    logger.debug(f"{strategy.name} error on {symbol} "
                                 f"@ {timestamp}: {e}")
            return signals

        # Run backtest
        self.backtester.reset()
        result = self.backtester.run_backtest(data, strategy_func, start_date, end_date)

        # Add strategy info to metadata
        result.metadata['strategy_name'] = strategy_name
        result.metadata['strategy_params'] = strategy_params

        return result
    
    def compare_strategies(
        self,
        strategies: List[Tuple[str, Dict]],
        data: Dict[str, pd.DataFrame],
        start_date: Optional[datetime] = None,
        end_date: Optional[datetime] = None
    ) -> pd.DataFrame:
        """Compare multiple strategies"""
        
        results = []
        
        for strategy_name, params in strategies:
            logger.info(f"Backtesting {strategy_name}...")
            
            result = self.run_strategy_backtest(
                strategy_name,
                params,
                data,
                start_date,
                end_date
            )
            
            results.append({
                'strategy': strategy_name,
                'total_return': result.total_return,
                'sharpe_ratio': result.sharpe_ratio,
                'max_drawdown': result.max_drawdown,
                'win_rate': result.win_rate,
                'profit_factor': result.profit_factor,
                'total_trades': len(result.trades),
                'avg_trade_duration': str(result.avg_trade_duration)
            })
        
        return pd.DataFrame(results)

# --- OPTIMIZATION ---

class StrategyOptimizer:
    """Optimize strategy parameters"""
    
    def __init__(self, backtester: BacktestRunner):
        self.backtester = backtester
        
    def optimize_parameters(
        self,
        strategy_name: str,
        param_grid: Dict[str, List],
        data: Dict[str, pd.DataFrame],
        optimization_metric: str = 'sharpe_ratio',
        n_jobs: int = 1
    ) -> Tuple[Dict, pd.DataFrame]:
        """Grid search optimization"""
        
        from itertools import product
        
        # Generate all parameter combinations
        param_names = list(param_grid.keys())
        param_values = list(param_grid.values())
        param_combinations = list(product(*param_values))
        
        results = []
        best_score = -float('inf')
        best_params = None
        
        for params_tuple in param_combinations:
            params = dict(zip(param_names, params_tuple))
            
            # Run backtest
            result = self.backtester.run_strategy_backtest(
                strategy_name,
                params,
                data
            )
            
            # Get optimization metric
            score = getattr(result, optimization_metric, 0)
            
            results.append({
                **params,
                optimization_metric: score,
                'total_return': result.total_return,
                'max_drawdown': result.max_drawdown,
                'total_trades': len(result.trades)
            })
            
            if score > best_score:
                best_score = score
                best_params = params
        
        results_df = pd.DataFrame(results)
        
        return best_params, results_df
