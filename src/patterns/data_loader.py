#!/usr/bin/env python3
"""
Advanced Data Loader for Pattern Discovery System
Handles OHLCV data loading, preprocessing, and preparation for ML pattern analysis
"""

import pandas as pd
import numpy as np
import sqlite3
import logging
from datetime import datetime, timedelta
from typing import Dict, List, Tuple, Optional
try:
    import talib
except ImportError:  # pure-pandas fallback with TA-Lib-compatible signatures
    from src.utils import indicators as talib
from sklearn.preprocessing import StandardScaler, MinMaxScaler
import asyncio
import aiohttp
from pathlib import Path

class PatternDataLoader:
    """
    Advanced data loader for pattern discovery with preprocessing capabilities
    """
    
    def __init__(self, db_path: str = None, cache_dir: str = "./data/cache",
                 allow_synthetic: bool = False):
        self.db_path = db_path
        self.allow_synthetic = allow_synthetic
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        
        # Data preprocessing parameters
        self.min_data_points = 100
        self.max_gap_size = 5  # Maximum allowed gaps in data
        
        # Scalers for different data types
        self.price_scaler = StandardScaler()
        self.volume_scaler = StandardScaler()
        self.indicator_scaler = MinMaxScaler()
        
        self.logger = logging.getLogger(__name__)
        
    def load_historical_data(self, 
                           symbol: str, 
                           timeframe: str = '1h',
                           start_date: datetime = None,
                           end_date: datetime = None,
                           min_periods: int = 1000) -> pd.DataFrame:
        """
        Load historical OHLCV data for pattern analysis
        
        Args:
            symbol: Trading symbol (e.g., 'BTC/USDT')
            timeframe: Data timeframe ('1m', '5m', '1h', '4h', '1d')
            start_date: Start date for data
            end_date: End date for data
            min_periods: Minimum number of periods required
            
        Returns:
            DataFrame with OHLCV data and basic indicators
        """
        # REAL DATA ONLY: synthetic data must be requested explicitly
        # (allow_synthetic=True) and exists for unit tests, never for trading.
        if self.db_path:
            df = self._load_from_database(symbol, timeframe, start_date, end_date)
            if df is not None and len(df) >= min_periods:
                return self._preprocess_data(df)

        cached = self._load_from_cache(symbol, timeframe, min_periods)
        if cached is not None:
            return cached

        if self.allow_synthetic:
            self.logger.warning(f"Using SYNTHETIC data for {symbol} "
                                f"(allow_synthetic=True — tests only)")
            return self._generate_sample_data(symbol, min_periods)

        raise ValueError(
            f"No real data for {symbol} {timeframe}: need >= {min_periods} "
            f"bars in the database. Run scripts/backfill.py first.")
    
    def _load_from_database(self, symbol: str, timeframe: str, 
                          start_date: datetime = None, 
                          end_date: datetime = None) -> pd.DataFrame:
        """Load data from SQLite database"""
        try:
            conn = sqlite3.connect(self.db_path)

            # Canonical schema: table `ohlcv`, epoch-second timestamps
            query = """
            SELECT timestamp, open, high, low, close, volume
            FROM ohlcv
            WHERE symbol = ? AND timeframe = ?
            """
            params = [symbol, timeframe]

            if start_date:
                query += " AND timestamp >= ?"
                params.append(int(start_date.timestamp()))

            if end_date:
                query += " AND timestamp <= ?"
                params.append(int(end_date.timestamp()))

            query += " ORDER BY timestamp ASC"

            df = pd.read_sql_query(query, conn, params=params)
            conn.close()

            if not df.empty:
                df['timestamp'] = pd.to_datetime(df['timestamp'], unit='s')
                df.set_index('timestamp', inplace=True)
                return df

        except Exception as e:
            self.logger.warning(f"Database load failed: {e}")

        return None

    def _load_from_cache(self, symbol: str, timeframe: str,
                         min_periods: int) -> Optional[pd.DataFrame]:
        """Load preprocessed data from the CSV cache, if present and complete."""
        cache_file = self.cache_dir / f"{symbol.replace('/', '_')}_{timeframe}.csv"

        if cache_file.exists():
            try:
                df = pd.read_csv(cache_file, index_col=0, parse_dates=True)
                if len(df) >= min_periods:
                    return self._preprocess_data(df)
            except Exception as e:
                self.logger.warning(f"Cache load failed: {e}")
        return None
    
    def _generate_sample_data(self, symbol: str, periods: int) -> pd.DataFrame:
        """Generate realistic sample OHLCV data for testing"""
        np.random.seed(42)  # For reproducible results
        
        # Base price for different symbols
        base_prices = {
            'BTC/USDT': 45000,
            'ETH/USDT': 3000,
            'SOL/USDT': 100,
            'BNB/USDT': 300,
            'ADA/USDT': 0.5,
            'DOT/USDT': 7.5
        }
        
        base_price = base_prices.get(symbol, 1000)
        
        # Generate timestamps
        end_time = datetime.now()
        start_time = end_time - timedelta(hours=periods)
        timestamps = pd.date_range(start=start_time, end=end_time, periods=periods)
        
        # Generate price data with realistic patterns
        returns = np.random.normal(0.0001, 0.02, periods)  # Small positive drift with volatility
        
        # Add some trending behavior
        trend = np.sin(np.linspace(0, 4*np.pi, periods)) * 0.001
        returns += trend
        
        # Add volatility clustering
        volatility = np.random.exponential(0.015, periods)
        returns *= volatility
        
        # Generate price series
        prices = base_price * np.exp(np.cumsum(returns))
        
        # Generate OHLC from prices
        df = pd.DataFrame(index=timestamps)
        df['close'] = prices
        
        # Generate realistic OHLC
        daily_range = np.random.uniform(0.005, 0.03, periods)  # 0.5% to 3% daily range
        
        df['high'] = df['close'] * (1 + daily_range * np.random.uniform(0.3, 1.0, periods))
        df['low'] = df['close'] * (1 - daily_range * np.random.uniform(0.3, 1.0, periods))
        df['open'] = df['low'] + (df['high'] - df['low']) * np.random.uniform(0.2, 0.8, periods)
        
        # Ensure OHLC relationships are correct
        df['high'] = np.maximum(df['high'], np.maximum(df['open'], df['close']))
        df['low'] = np.minimum(df['low'], np.minimum(df['open'], df['close']))
        
        # Generate volume
        base_volume = 1000000
        volume_trend = np.random.exponential(1, periods)
        price_volume_corr = np.abs(returns) * 0.5  # Higher volume during high volatility
        df['volume'] = base_volume * volume_trend * (1 + price_volume_corr)
        
        return self._preprocess_data(df)
    
    def _preprocess_data(self, df: pd.DataFrame) -> pd.DataFrame:
        """Preprocess OHLCV data for pattern analysis"""
        try:
            # Remove any NaN values
            df = df.dropna()
            
            # Ensure proper column order
            df = df[['open', 'high', 'low', 'close', 'volume']]
            
            # Add basic derived features
            df = self._add_basic_features(df)
            
            # Detect and handle outliers
            df = self._handle_outliers(df)
            
            # Add technical indicators
            df = self._add_technical_indicators(df)
            
            # Fill any remaining NaN values
            df = df.ffill().bfill()
            
            self.logger.info(f"Preprocessed data: {len(df)} periods")
            return df
            
        except Exception as e:
            self.logger.error(f"Error preprocessing data: {e}")
            raise
    
    def _add_basic_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add basic derived features"""
        # Price-based features
        df['hl2'] = (df['high'] + df['low']) / 2
        df['hlc3'] = (df['high'] + df['low'] + df['close']) / 3
        df['ohlc4'] = (df['open'] + df['high'] + df['low'] + df['close']) / 4
        
        # Returns
        df['returns'] = df['close'].pct_change()
        df['log_returns'] = np.log(df['close'] / df['close'].shift(1))
        
        # Price ranges
        df['true_range'] = np.maximum(
            df['high'] - df['low'],
            np.maximum(
                np.abs(df['high'] - df['close'].shift(1)),
                np.abs(df['low'] - df['close'].shift(1))
            )
        )
        
        # Volume features
        df['volume_sma'] = df['volume'].rolling(20).mean()
        df['volume_ratio'] = df['volume'] / df['volume_sma']
        
        # Price position within range
        df['price_position'] = (df['close'] - df['low']) / (df['high'] - df['low'])
        
        return df
    
    def _handle_outliers(self, df: pd.DataFrame, threshold: float = 3.0) -> pd.DataFrame:
        """Detect and handle outliers using Z-score method"""
        numeric_columns = ['open', 'high', 'low', 'close', 'volume']
        
        for col in numeric_columns:
            if col in df.columns:
                z_scores = np.abs((df[col] - df[col].mean()) / df[col].std())
                outliers = z_scores > threshold
                
                if outliers.sum() > 0:
                    self.logger.warning(f"Found {outliers.sum()} outliers in {col}")
                    # Replace outliers with median
                    df.loc[outliers, col] = df[col].median()
        
        return df
    
    def _add_technical_indicators(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add comprehensive technical indicators"""
        try:
            # Moving Averages
            for period in [5, 10, 20, 50, 100, 200]:
                df[f'sma_{period}'] = talib.SMA(df['close'], timeperiod=period)
                df[f'ema_{period}'] = talib.EMA(df['close'], timeperiod=period)
            
            # Momentum Indicators
            df['rsi'] = talib.RSI(df['close'], timeperiod=14)
            df['rsi_sma'] = talib.SMA(df['rsi'], timeperiod=14)
            
            df['macd'], df['macd_signal'], df['macd_hist'] = talib.MACD(df['close'])
            df['stoch_k'], df['stoch_d'] = talib.STOCH(df['high'], df['low'], df['close'])
            
            df['williams_r'] = talib.WILLR(df['high'], df['low'], df['close'])
            df['roc'] = talib.ROC(df['close'], timeperiod=10)
            
            # Volatility Indicators
            df['atr'] = talib.ATR(df['high'], df['low'], df['close'], timeperiod=14)
            df['bb_upper'], df['bb_middle'], df['bb_lower'] = talib.BBANDS(df['close'])
            df['bb_width'] = (df['bb_upper'] - df['bb_lower']) / df['bb_middle']
            df['bb_position'] = (df['close'] - df['bb_lower']) / (df['bb_upper'] - df['bb_lower'])
            
            # Volume Indicators
            df['obv'] = talib.OBV(df['close'], df['volume'])
            df['ad'] = talib.AD(df['high'], df['low'], df['close'], df['volume'])
            df['adosc'] = talib.ADOSC(df['high'], df['low'], df['close'], df['volume'])
            
            # Pattern Recognition Indicators
            df['doji'] = talib.CDLDOJI(df['open'], df['high'], df['low'], df['close'])
            df['hammer'] = talib.CDLHAMMER(df['open'], df['high'], df['low'], df['close'])
            df['shooting_star'] = talib.CDLSHOOTINGSTAR(df['open'], df['high'], df['low'], df['close'])
            df['engulfing'] = talib.CDLENGULFING(df['open'], df['high'], df['low'], df['close'])
            
            # Support/Resistance levels
            df = self._add_support_resistance(df)
            
            # Trend indicators
            df = self._add_trend_indicators(df)
            
        except Exception as e:
            self.logger.warning(f"Error adding technical indicators: {e}")
        
        return df
    
    def _add_support_resistance(self, df: pd.DataFrame, window: int = 20) -> pd.DataFrame:
        """Add support and resistance levels"""
        try:
            # Rolling support and resistance
            df['resistance'] = df['high'].rolling(window).max()
            df['support'] = df['low'].rolling(window).min()
            
            # Distance from support/resistance
            df['dist_to_resistance'] = (df['resistance'] - df['close']) / df['close']
            df['dist_to_support'] = (df['close'] - df['support']) / df['close']
            
            # Pivot points
            df['pivot'] = (df['high'].shift(1) + df['low'].shift(1) + df['close'].shift(1)) / 3
            df['r1'] = 2 * df['pivot'] - df['low'].shift(1)
            df['s1'] = 2 * df['pivot'] - df['high'].shift(1)
            
        except Exception as e:
            self.logger.warning(f"Error adding support/resistance: {e}")
        
        return df
    
    def _add_trend_indicators(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add trend strength and direction indicators"""
        try:
            # ADX for trend strength
            df['adx'] = talib.ADX(df['high'], df['low'], df['close'], timeperiod=14)
            df['plus_di'] = talib.PLUS_DI(df['high'], df['low'], df['close'], timeperiod=14)
            df['minus_di'] = talib.MINUS_DI(df['high'], df['low'], df['close'], timeperiod=14)
            
            # Parabolic SAR
            df['sar'] = talib.SAR(df['high'], df['low'])
            
            # Aroon indicators
            df['aroon_up'], df['aroon_down'] = talib.AROON(df['high'], df['low'], timeperiod=14)
            df['aroon_osc'] = df['aroon_up'] - df['aroon_down']
            
            # Moving average trends
            df['ma_trend_5_20'] = (df['sma_5'] > df['sma_20']).astype(int)
            df['ma_trend_20_50'] = (df['sma_20'] > df['sma_50']).astype(int)
            df['ma_trend_50_200'] = (df['sma_50'] > df['sma_200']).astype(int)
            
        except Exception as e:
            self.logger.warning(f"Error adding trend indicators: {e}")
        
        return df
    
    def prepare_pattern_windows(self, df: pd.DataFrame, 
                              window_sizes: List[int] = [10, 20, 50, 100]) -> Dict[int, np.ndarray]:
        """
        Prepare sliding windows of different sizes for pattern analysis
        
        Args:
            df: Preprocessed OHLCV dataframe
            window_sizes: List of window sizes to create
            
        Returns:
            Dictionary mapping window size to arrays of shape (n_windows, window_size, n_features)
        """
        windows = {}
        
        # Select features for pattern analysis
        feature_columns = [
            'open', 'high', 'low', 'close', 'volume',
            'returns', 'true_range', 'rsi', 'macd', 'bb_position',
            'volume_ratio', 'adx', 'aroon_osc'
        ]
        
        # Filter to available features
        available_features = [col for col in feature_columns if col in df.columns]
        feature_data = df[available_features].values
        
        for window_size in window_sizes:
            if len(df) < window_size:
                continue
                
            # Create sliding windows
            n_windows = len(df) - window_size + 1
            window_data = np.zeros((n_windows, window_size, len(available_features)))
            
            for i in range(n_windows):
                window_data[i] = feature_data[i:i + window_size]
            
            # Normalize each window
            normalized_windows = []
            for i in range(n_windows):
                window = window_data[i].copy()
                
                # Normalize price data (first 5 columns) relative to first close price
                if len(available_features) >= 5:
                    first_close = window[0, 3]  # Close price is 4th column (index 3)
                    window[:, :4] = window[:, :4] / first_close
                
                # Normalize volume separately
                if len(available_features) >= 5:
                    volume_col = window[:, 4]
                    if volume_col.std() > 0:
                        window[:, 4] = (volume_col - volume_col.mean()) / volume_col.std()
                
                # Other indicators are already in reasonable ranges
                normalized_windows.append(window)
            
            windows[window_size] = np.array(normalized_windows)
        
        return windows
    
    def get_data_quality_metrics(self, df: pd.DataFrame) -> Dict[str, float]:
        """Calculate data quality metrics"""
        metrics = {
            'total_periods': len(df),
            'missing_values_pct': df.isnull().sum().sum() / (len(df) * len(df.columns)) * 100,
            'duplicate_timestamps': df.index.duplicated().sum(),
            'avg_volume': df['volume'].mean() if 'volume' in df.columns else 0,
            'volatility': df['returns'].std() if 'returns' in df.columns else 0,
            'price_range_ratio': (df['high'].max() - df['low'].min()) / df['close'].mean(),
        }
        
        # Check for data gaps
        if len(df) > 1:
            time_diffs = df.index.to_series().diff()
            expected_freq = time_diffs.mode().iloc[0] if not time_diffs.mode().empty else pd.Timedelta(hours=1)
            gaps = (time_diffs > expected_freq * 1.5).sum()
            metrics['data_gaps'] = gaps
        
        return metrics
    
    async def load_multiple_symbols(self, symbols: List[str], 
                                  timeframe: str = '1h',
                                  min_periods: int = 1000) -> Dict[str, pd.DataFrame]:
        """Load data for multiple symbols concurrently"""
        tasks = []
        for symbol in symbols:
            task = asyncio.create_task(
                asyncio.to_thread(self.load_historical_data, symbol, timeframe, None, None, min_periods)
            )
            tasks.append((symbol, task))
        
        results = {}
        for symbol, task in tasks:
            try:
                results[symbol] = await task
                self.logger.info(f"Loaded data for {symbol}: {len(results[symbol])} periods")
            except Exception as e:
                self.logger.error(f"Failed to load data for {symbol}: {e}")
        
        return results
