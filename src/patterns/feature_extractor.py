#!/usr/bin/env python3
"""
Advanced Feature Extractor for Pattern Discovery
Extracts comprehensive features for ML-based pattern recognition
"""

import pandas as pd
import numpy as np
import logging
from typing import Dict, List, Tuple, Optional, Any
from sklearn.preprocessing import StandardScaler, RobustScaler
from scipy import stats
from scipy.signal import find_peaks, argrelextrema
from scipy.spatial.distance import euclidean
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
try:
    import talib
except ImportError:  # pure-pandas fallback with TA-Lib-compatible signatures
    from src.utils import indicators as talib
from dataclasses import dataclass
import warnings
warnings.filterwarnings('ignore')

@dataclass
class PatternFeatures:
    """Container for extracted pattern features"""
    geometric_features: np.ndarray
    statistical_features: np.ndarray
    technical_features: np.ndarray
    volume_features: np.ndarray
    fractal_features: np.ndarray
    sentiment_features: np.ndarray
    combined_features: np.ndarray
    feature_names: List[str]
    window_size: int
    timestamp: pd.Timestamp

class AdvancedFeatureExtractor:
    """
    Advanced feature extraction for pattern discovery using ML techniques
    """
    
    def __init__(self, enable_advanced_features: bool = True):
        self.enable_advanced_features = enable_advanced_features
        self.logger = logging.getLogger(__name__)
        
        # Scalers for different feature types
        self.price_scaler = RobustScaler()
        self.volume_scaler = StandardScaler()
        self.indicator_scaler = StandardScaler()
        
        # Pattern templates for template matching
        self.pattern_templates = self._initialize_pattern_templates()
        
        # PCA for dimensionality reduction
        self.pca_components = 10
        self.pca_model = PCA(n_components=self.pca_components)
        
        # Feature importance tracking
        self.feature_importance = {}
        self.feature_usage_count = {}
        
    def extract_features(self, 
                        data_window: np.ndarray, 
                        timestamp: pd.Timestamp = None,
                        symbol: str = None) -> PatternFeatures:
        """
        Extract comprehensive features from a data window
        
        Args:
            data_window: Array of shape (window_size, n_features) containing OHLCV and indicators
            timestamp: Timestamp for this window
            symbol: Trading symbol for context
            
        Returns:
            PatternFeatures object containing all extracted features
        """
        try:
            if data_window.ndim != 2:
                raise ValueError(f"Expected 2D array, got {data_window.ndim}D")
            
            window_size, n_input_features = data_window.shape
            
            # Extract different types of features
            geometric_features = self._extract_geometric_features(data_window)
            statistical_features = self._extract_statistical_features(data_window)
            technical_features = self._extract_technical_features(data_window)
            volume_features = self._extract_volume_features(data_window)
            fractal_features = self._extract_fractal_features(data_window)
            
            if self.enable_advanced_features:
                sentiment_features = self._extract_sentiment_features(data_window, symbol)
            else:
                sentiment_features = np.array([])
            
            # Combine all features
            all_features = [
                geometric_features,
                statistical_features,
                technical_features,
                volume_features,
                fractal_features,
                sentiment_features
            ]
            
            # Filter out empty arrays
            valid_features = [f for f in all_features if len(f) > 0]
            combined_features = np.concatenate(valid_features) if valid_features else np.array([])
            
            # Generate feature names
            feature_names = self._generate_feature_names(
                len(geometric_features),
                len(statistical_features), 
                len(technical_features),
                len(volume_features),
                len(fractal_features),
                len(sentiment_features)
            )
            
            return PatternFeatures(
                geometric_features=geometric_features,
                statistical_features=statistical_features,
                technical_features=technical_features,
                volume_features=volume_features,
                fractal_features=fractal_features,
                sentiment_features=sentiment_features,
                combined_features=combined_features,
                feature_names=feature_names,
                window_size=window_size,
                timestamp=timestamp or pd.Timestamp.now()
            )
            
        except Exception as e:
            self.logger.error(f"Error extracting features: {e}")
            return self._create_empty_features(data_window.shape[0])
    
    def _extract_geometric_features(self, data_window: np.ndarray) -> np.ndarray:
        """Extract geometric and shape-based features"""
        try:
            features = []
            
            # Assuming OHLCV are first 5 columns
            if data_window.shape[1] >= 5:
                prices = data_window[:, :4]  # OHLC
                close_prices = data_window[:, 3]  # Close prices
                volumes = data_window[:, 4]  # Volumes
                
                # Price shape features
                features.extend(self._extract_price_shape_features(close_prices))
                
                # Candlestick patterns
                features.extend(self._extract_candlestick_features(prices))
                
                # Support and resistance levels
                features.extend(self._extract_support_resistance_features(prices))
                
                # Trend line features
                features.extend(self._extract_trend_line_features(close_prices))
                
                # Pattern template matching
                features.extend(self._extract_template_matching_features(close_prices))
                
            return np.array(features, dtype=np.float32)
            
        except Exception as e:
            self.logger.warning(f"Error in geometric features: {e}")
            return np.array([])
    
    def _extract_price_shape_features(self, prices: np.ndarray) -> List[float]:
        """Extract features describing price shape and movement"""
        features = []
        
        if len(prices) < 3:
            return [0.0] * 15
        
        # Basic price statistics
        features.append(np.mean(prices))
        features.append(np.std(prices))
        features.append(stats.skew(prices))
        features.append(stats.kurtosis(prices))
        
        # Price range and position
        price_range = np.max(prices) - np.min(prices)
        features.append(price_range / np.mean(prices))  # Relative range
        features.append((prices[-1] - np.min(prices)) / price_range if price_range > 0 else 0.5)  # Position in range
        
        # Trend measurements
        x = np.arange(len(prices))
        slope, intercept, r_value, p_value, std_err = stats.linregress(x, prices)
        features.append(slope / np.mean(prices))  # Normalized slope
        features.append(r_value ** 2)  # R-squared
        features.append(std_err / np.mean(prices))  # Normalized standard error
        
        # Price acceleration (second derivative)
        if len(prices) >= 3:
            first_diff = np.diff(prices)
            second_diff = np.diff(first_diff)
            features.append(np.mean(second_diff))
            features.append(np.std(second_diff))
        else:
            features.extend([0.0, 0.0])
        
        # Autocorrelation at lag 1
        if len(prices) > 1:
            autocorr = np.corrcoef(prices[:-1], prices[1:])[0, 1]
            features.append(autocorr if not np.isnan(autocorr) else 0.0)
        else:
            features.append(0.0)
        
        # Peaks and valleys
        peaks, _ = find_peaks(prices, distance=max(1, len(prices)//10))
        valleys, _ = find_peaks(-prices, distance=max(1, len(prices)//10))
        features.append(len(peaks) / len(prices))  # Peak density
        features.append(len(valleys) / len(prices))  # Valley density
        
        return features
    
    def _extract_candlestick_features(self, ohlc: np.ndarray) -> List[float]:
        """Extract candlestick pattern features"""
        features = []
        
        if ohlc.shape[0] < 2 or ohlc.shape[1] < 4:
            return [0.0] * 20
        
        opens, highs, lows, closes = ohlc[:, 0], ohlc[:, 1], ohlc[:, 2], ohlc[:, 3]
        
        # Body and shadow ratios
        body_sizes = np.abs(closes - opens)
        upper_shadows = highs - np.maximum(opens, closes)
        lower_shadows = np.minimum(opens, closes) - lows
        total_ranges = highs - lows
        
        # Avoid division by zero
        safe_ranges = np.where(total_ranges > 0, total_ranges, 1e-8)
        
        # Ratios
        body_ratios = body_sizes / safe_ranges
        upper_shadow_ratios = upper_shadows / safe_ranges
        lower_shadow_ratios = lower_shadows / safe_ranges
        
        # Statistical features of ratios
        features.extend([
            np.mean(body_ratios),
            np.std(body_ratios),
            np.mean(upper_shadow_ratios),
            np.std(upper_shadow_ratios),
            np.mean(lower_shadow_ratios),
            np.std(lower_shadow_ratios)
        ])
        
        # Color pattern (bullish/bearish)
        bullish = (closes > opens).astype(float)
        features.append(np.mean(bullish))  # Bullish ratio
        
        # Consecutive patterns
        bullish_streaks = self._calculate_streaks(bullish)
        bearish_streaks = self._calculate_streaks(1 - bullish)
        features.extend([
            np.mean(bullish_streaks),
            np.max(bullish_streaks) if len(bullish_streaks) > 0 else 0,
            np.mean(bearish_streaks),
            np.max(bearish_streaks) if len(bearish_streaks) > 0 else 0
        ])
        
        # Gap analysis
        gaps = opens[1:] - closes[:-1]
        gap_ups = np.sum(gaps > 0)
        gap_downs = np.sum(gaps < 0)
        features.extend([
            gap_ups / len(gaps) if len(gaps) > 0 else 0,
            gap_downs / len(gaps) if len(gaps) > 0 else 0,
            np.mean(np.abs(gaps)) / np.mean(closes) if len(gaps) > 0 else 0
        ])
        
        # Doji patterns (small body relative to range)
        doji_threshold = 0.1
        doji_candles = body_ratios < doji_threshold
        features.append(np.mean(doji_candles))
        
        # Hammer/Shooting star patterns
        hammer_pattern = (lower_shadow_ratios > 0.6) & (upper_shadow_ratios < 0.1) & (body_ratios < 0.3)
        shooting_star_pattern = (upper_shadow_ratios > 0.6) & (lower_shadow_ratios < 0.1) & (body_ratios < 0.3)
        features.extend([
            np.mean(hammer_pattern),
            np.mean(shooting_star_pattern)
        ])
        
        # Spinning top patterns
        spinning_top = (body_ratios < 0.3) & (upper_shadow_ratios > 0.2) & (lower_shadow_ratios > 0.2)
        features.append(np.mean(spinning_top))
        
        return features
    
    def _extract_support_resistance_features(self, ohlc: np.ndarray) -> List[float]:
        """Extract support and resistance level features"""
        features = []
        
        if ohlc.shape[0] < 5:
            return [0.0] * 10
        
        highs, lows, closes = ohlc[:, 1], ohlc[:, 2], ohlc[:, 3]
        
        # Find local maxima and minima
        resistance_indices = argrelextrema(highs, np.greater, order=2)[0]
        support_indices = argrelextrema(lows, np.less, order=2)[0]
        
        resistance_levels = highs[resistance_indices] if len(resistance_indices) > 0 else np.array([np.max(highs)])
        support_levels = lows[support_indices] if len(support_indices) > 0 else np.array([np.min(lows)])
        
        current_price = closes[-1]
        
        # Distance to nearest support/resistance
        if len(resistance_levels) > 0:
            nearest_resistance = np.min(resistance_levels[resistance_levels >= current_price])
            if np.isfinite(nearest_resistance):
                features.append((nearest_resistance - current_price) / current_price)
            else:
                features.append(0.1)  # Default value
        else:
            features.append(0.1)
        
        if len(support_levels) > 0:
            nearest_support = np.max(support_levels[support_levels <= current_price])
            if np.isfinite(nearest_support):
                features.append((current_price - nearest_support) / current_price)
            else:
                features.append(0.1)  # Default value
        else:
            features.append(0.1)
        
        # Support/resistance strength (number of touches)
        features.append(len(resistance_levels) / len(highs))
        features.append(len(support_levels) / len(lows))
        
        # Range analysis
        price_range = np.max(highs) - np.min(lows)
        features.append(price_range / np.mean(closes))
        
        # Position within range
        range_position = (current_price - np.min(lows)) / price_range if price_range > 0 else 0.5
        features.append(range_position)
        
        # Breakout potential
        recent_highs = highs[-5:] if len(highs) >= 5 else highs
        recent_lows = lows[-5:] if len(lows) >= 5 else lows
        
        features.append((current_price - np.max(recent_highs)) / current_price)
        features.append((np.min(recent_lows) - current_price) / current_price)
        
        # Consolidation measure
        consolidation_range = np.max(recent_highs) - np.min(recent_lows)
        features.append(consolidation_range / np.mean(closes[-5:]) if len(closes) >= 5 else 0)
        
        # Squeeze indicator (range contraction)
        if len(highs) >= 10:
            early_range = np.max(highs[:5]) - np.min(lows[:5])
            late_range = np.max(highs[-5:]) - np.min(lows[-5:])
            features.append(late_range / early_range if early_range > 0 else 1.0)
        else:
            features.append(1.0)
        
        return features
    
    def _extract_trend_line_features(self, prices: np.ndarray) -> List[float]:
        """Extract trend line and channel features"""
        features = []
        
        if len(prices) < 10:
            return [0.0] * 8
        
        # Multiple timeframe trend analysis
        for period in [5, 10, len(prices)//2, len(prices)]:
            if period <= len(prices):
                recent_prices = prices[-period:]
                x = np.arange(len(recent_prices))
                
                # Linear regression for trend
                slope, intercept, r_value, p_value, std_err = stats.linregress(x, recent_prices)
                
                # Normalize slope by price level
                normalized_slope = slope / np.mean(recent_prices)
                features.append(normalized_slope)
                features.append(r_value ** 2)  # Trend strength
        
        return features
    
    def _extract_template_matching_features(self, prices: np.ndarray) -> List[float]:
        """Extract features based on template matching with known patterns"""
        features = []
        
        if len(prices) < 10:
            return [0.0] * len(self.pattern_templates)
        
        # Normalize prices for template matching
        normalized_prices = (prices - np.min(prices)) / (np.max(prices) - np.min(prices))
        
        for template_name, template in self.pattern_templates.items():
            if len(normalized_prices) >= len(template):
                # Find best match using sliding window
                best_correlation = -1
                
                for i in range(len(normalized_prices) - len(template) + 1):
                    window = normalized_prices[i:i + len(template)]
                    correlation = np.corrcoef(window, template)[0, 1]
                    
                    if not np.isnan(correlation) and correlation > best_correlation:
                        best_correlation = correlation
                
                features.append(best_correlation if best_correlation > -1 else 0)
            else:
                features.append(0)
        
        return features
    
    def _extract_statistical_features(self, data_window: np.ndarray) -> np.ndarray:
        """Extract statistical features from the data window"""
        try:
            features = []
            
            # For each column (feature) in the data
            for col_idx in range(data_window.shape[1]):
                column_data = data_window[:, col_idx]
                
                # Basic statistics
                features.extend([
                    np.mean(column_data),
                    np.std(column_data),
                    np.min(column_data),
                    np.max(column_data),
                    np.median(column_data)
                ])
                
                # Higher moments
                features.extend([
                    stats.skew(column_data),
                    stats.kurtosis(column_data)
                ])
                
                # Percentiles
                features.extend([
                    np.percentile(column_data, 25),
                    np.percentile(column_data, 75)
                ])
                
                # Variability measures
                features.append(np.std(column_data) / np.mean(column_data) if np.mean(column_data) != 0 else 0)
                
            # Cross-column correlations for first few columns
            if data_window.shape[1] >= 2:
                for i in range(min(5, data_window.shape[1])):
                    for j in range(i + 1, min(5, data_window.shape[1])):
                        corr = np.corrcoef(data_window[:, i], data_window[:, j])[0, 1]
                        features.append(corr if not np.isnan(corr) else 0)
            
            return np.array(features, dtype=np.float32)
            
        except Exception as e:
            self.logger.warning(f"Error in statistical features: {e}")
            return np.array([])
    
    def _extract_technical_features(self, data_window: np.ndarray) -> np.ndarray:
        """Extract technical analysis features"""
        try:
            features = []
            
            if data_window.shape[1] >= 5:
                # Assuming first 5 columns are OHLCV
                closes = data_window[:, 3]
                volumes = data_window[:, 4]
                
                # Moving average relationships
                if len(closes) >= 20:
                    sma_5 = np.mean(closes[-5:])
                    sma_10 = np.mean(closes[-10:])
                    sma_20 = np.mean(closes[-20:])
                    
                    features.extend([
                        closes[-1] / sma_5 - 1,
                        closes[-1] / sma_10 - 1,
                        closes[-1] / sma_20 - 1,
                        sma_5 / sma_10 - 1,
                        sma_10 / sma_20 - 1
                    ])
                
                # RSI calculation
                if len(closes) >= 14:
                    rsi = self._calculate_rsi(closes, 14)
                    features.append(rsi / 100.0)  # Normalize to 0-1
                
                # MACD features
                if len(closes) >= 26:
                    macd_line, signal_line = self._calculate_macd(closes)
                    features.extend([
                        macd_line,
                        signal_line,
                        macd_line - signal_line
                    ])
                
                # Volume analysis
                if len(volumes) >= 10:
                    volume_sma = np.mean(volumes[-10:])
                    features.append(volumes[-1] / volume_sma - 1)
                
                # Volatility features
                if len(closes) >= 10:
                    returns = np.diff(closes) / closes[:-1]
                    features.extend([
                        np.std(returns[-10:]) if len(returns) >= 10 else 0,
                        np.mean(np.abs(returns[-5:])) if len(returns) >= 5 else 0
                    ])
            
            return np.array(features, dtype=np.float32)
            
        except Exception as e:
            self.logger.warning(f"Error in technical features: {e}")
            return np.array([])
    
    def _extract_volume_features(self, data_window: np.ndarray) -> np.ndarray:
        """Extract volume-based features"""
        try:
            features = []
            
            if data_window.shape[1] >= 5:
                volumes = data_window[:, 4]
                closes = data_window[:, 3]
                
                # Volume trend
                if len(volumes) >= 3:
                    volume_slope = (volumes[-1] - volumes[0]) / len(volumes)
                    features.append(volume_slope / np.mean(volumes))
                
                # Volume-price correlation
                if len(volumes) >= 5:
                    price_changes = np.diff(closes)
                    volume_changes = np.diff(volumes)
                    
                    if len(price_changes) > 0 and len(volume_changes) > 0:
                        corr = np.corrcoef(price_changes[-min(10, len(price_changes)):], 
                                         volume_changes[-min(10, len(volume_changes)):])[0, 1]
                        features.append(corr if not np.isnan(corr) else 0)
                
                # Volume profile features
                features.extend([
                    np.std(volumes) / np.mean(volumes) if np.mean(volumes) > 0 else 0,
                    volumes[-1] / np.mean(volumes) if np.mean(volumes) > 0 else 1,
                ])
                
                # On-balance volume trend
                if len(closes) >= 2:
                    obv = self._calculate_obv(closes, volumes)
                    features.append(obv)
            
            return np.array(features, dtype=np.float32)
            
        except Exception as e:
            self.logger.warning(f"Error in volume features: {e}")
            return np.array([])
    
    def _extract_fractal_features(self, data_window: np.ndarray) -> np.ndarray:
        """Extract fractal and complexity features"""
        try:
            features = []
            
            if data_window.shape[1] >= 1:
                prices = data_window[:, 3] if data_window.shape[1] >= 4 else data_window[:, 0]
                
                # Hurst exponent (measure of long-range dependence)
                if len(prices) >= 10:
                    hurst = self._calculate_hurst_exponent(prices)
                    features.append(hurst)
                
                # Fractal dimension
                if len(prices) >= 5:
                    fractal_dim = self._calculate_fractal_dimension(prices)
                    features.append(fractal_dim)
                
                # Approximate entropy (measure of regularity)
                if len(prices) >= 10:
                    app_entropy = self._calculate_approximate_entropy(prices)
                    features.append(app_entropy)
                
                # Lyapunov exponent (measure of chaos)
                if len(prices) >= 15:
                    lyapunov = self._calculate_lyapunov_exponent(prices)
                    features.append(lyapunov)
            
            return np.array(features, dtype=np.float32)
            
        except Exception as e:
            self.logger.warning(f"Error in fractal features: {e}")
            return np.array([])
    
    def _extract_sentiment_features(self, data_window: np.ndarray, symbol: str = None) -> np.ndarray:
        """Extract sentiment and market microstructure features"""
        try:
            features = []
            
            if data_window.shape[1] >= 5:
                opens, highs, lows, closes, volumes = data_window[:, 0], data_window[:, 1], data_window[:, 2], data_window[:, 3], data_window[:, 4]
                
                # Market pressure indicators
                buying_pressure = np.sum((closes > opens) * volumes)
                selling_pressure = np.sum((closes < opens) * volumes)
                total_volume = np.sum(volumes)
                
                if total_volume > 0:
                    features.extend([
                        buying_pressure / total_volume,
                        selling_pressure / total_volume
                    ])
                
                # Price-volume efficiency
                price_moves = np.abs(closes - opens)
                if len(price_moves) > 0 and np.sum(volumes) > 0:
                    efficiency = np.sum(price_moves * volumes) / np.sum(volumes)
                    features.append(efficiency / np.mean(closes))
                
                # Accumulation/Distribution pattern
                if len(closes) >= 3:
                    acc_dist = self._calculate_accumulation_distribution(highs, lows, closes, volumes)
                    features.append(acc_dist)
            
            return np.array(features, dtype=np.float32)
            
        except Exception as e:
            self.logger.warning(f"Error in sentiment features: {e}")
            return np.array([])
    
    # Helper methods for calculations
    def _calculate_streaks(self, binary_array: np.ndarray) -> List[int]:
        """Calculate consecutive streaks in binary array"""
        if len(binary_array) == 0:
            return []
        
        streaks = []
        current_streak = 1
        
        for i in range(1, len(binary_array)):
            if binary_array[i] == binary_array[i-1]:
                current_streak += 1
            else:
                if binary_array[i-1] == 1:  # Only count positive streaks
                    streaks.append(current_streak)
                current_streak = 1
        
        if binary_array[-1] == 1:
            streaks.append(current_streak)
        
        return streaks if streaks else [0]
    
    def _calculate_rsi(self, prices: np.ndarray, period: int = 14) -> float:
        """Calculate RSI indicator"""
        if len(prices) < period + 1:
            return 50.0
        
        deltas = np.diff(prices)
        gains = np.where(deltas > 0, deltas, 0)
        losses = np.where(deltas < 0, -deltas, 0)
        
        avg_gain = np.mean(gains[-period:])
        avg_loss = np.mean(losses[-period:])
        
        if avg_loss == 0:
            return 100.0
        
        rs = avg_gain / avg_loss
        rsi = 100 - (100 / (1 + rs))
        return rsi
    
    def _calculate_macd(self, prices: np.ndarray, fast: int = 12, slow: int = 26, signal: int = 9) -> Tuple[float, float]:
        """Calculate MACD indicator"""
        if len(prices) < slow:
            return 0.0, 0.0
        
        ema_fast = self._calculate_ema(prices, fast)
        ema_slow = self._calculate_ema(prices, slow)
        
        macd_line = ema_fast - ema_slow
        
        # Simple approximation for signal line
        signal_line = macd_line * 0.9  # Simplified
        
        return macd_line / np.mean(prices), signal_line / np.mean(prices)
    
    def _calculate_ema(self, prices: np.ndarray, period: int) -> float:
        """Calculate Exponential Moving Average"""
        if len(prices) < period:
            return np.mean(prices)
        
        alpha = 2.0 / (period + 1)
        ema = prices[0]
        
        for price in prices[1:]:
            ema = alpha * price + (1 - alpha) * ema
        
        return ema
    
    def _calculate_obv(self, closes: np.ndarray, volumes: np.ndarray) -> float:
        """Calculate On-Balance Volume indicator"""
        if len(closes) < 2:
            return 0.0
        
        obv = 0
        for i in range(1, len(closes)):
            if closes[i] > closes[i-1]:
                obv += volumes[i]
            elif closes[i] < closes[i-1]:
                obv -= volumes[i]
        
        return obv / np.sum(volumes) if np.sum(volumes) > 0 else 0
    
    def _calculate_accumulation_distribution(self, highs: np.ndarray, lows: np.ndarray, 
                                           closes: np.ndarray, volumes: np.ndarray) -> float:
        """Calculate Accumulation/Distribution Line"""
        if len(highs) == 0:
            return 0.0
        
        clv = np.where(highs != lows, 
                      ((closes - lows) - (highs - closes)) / (highs - lows), 
                      0)
        
        ad_line = np.sum(clv * volumes)
        return ad_line / np.sum(volumes) if np.sum(volumes) > 0 else 0
    
    def _calculate_hurst_exponent(self, prices: np.ndarray) -> float:
        """Calculate Hurst exponent for measuring long-range dependence"""
        try:
            if len(prices) < 10:
                return 0.5
            
            log_prices = np.log(prices)
            n = len(log_prices)
            
            # Calculate ranges for different time scales
            ranges = []
            for i in range(2, min(n//2, 20)):
                # Divide series into segments
                segments = [log_prices[j:j+i] for j in range(0, n-i+1, i)]
                
                if len(segments) > 1:
                    # Calculate range for each segment
                    segment_ranges = []
                    for segment in segments:
                        if len(segment) >= 2:
                            cumsum = np.cumsum(segment - np.mean(segment))
                            segment_range = np.max(cumsum) - np.min(cumsum)
                            segment_ranges.append(segment_range)
                    
                    if segment_ranges:
                        ranges.append(np.mean(segment_ranges))
            
            if len(ranges) < 2:
                return 0.5
            
            # Fit power law
            scales = np.arange(2, len(ranges) + 2)
            log_ranges = np.log(ranges)
            log_scales = np.log(scales)
            
            slope, _, _, _, _ = stats.linregress(log_scales, log_ranges)
            hurst = slope
            
            # Clamp to reasonable range
            return np.clip(hurst, 0, 1)
            
        except Exception:
            return 0.5
    
    def _calculate_fractal_dimension(self, prices: np.ndarray) -> float:
        """Calculate fractal dimension using box-counting method"""
        try:
            if len(prices) < 5:
                return 1.5
            
            # Normalize prices
            normalized_prices = (prices - np.min(prices)) / (np.max(prices) - np.min(prices))
            
            # Box counting
            scales = [2, 4, 8, 16]
            counts = []
            
            for scale in scales:
                if scale < len(normalized_prices):
                    # Count boxes needed to cover the curve
                    boxes = set()
                    for i in range(len(normalized_prices) - 1):
                        x1, y1 = i / scale, normalized_prices[i] * scale
                        x2, y2 = (i + 1) / scale, normalized_prices[i + 1] * scale
                        
                        # Add boxes that the line segment passes through
                        for x in range(int(min(x1, x2)), int(max(x1, x2)) + 1):
                            for y in range(int(min(y1, y2)), int(max(y1, y2)) + 1):
                                boxes.add((x, y))
                    
                    counts.append(len(boxes))
            
            if len(counts) < 2:
                return 1.5
            
            # Calculate fractal dimension
            log_scales = np.log(scales[:len(counts)])
            log_counts = np.log(counts)
            
            slope, _, _, _, _ = stats.linregress(log_scales, log_counts)
            fractal_dim = -slope
            
            return np.clip(fractal_dim, 1, 2)
            
        except Exception:
            return 1.5
    
    def _calculate_approximate_entropy(self, prices: np.ndarray, m: int = 2, r: float = None) -> float:
        """Calculate approximate entropy"""
        try:
            if len(prices) < 10:
                return 0.5
            
            if r is None:
                r = 0.2 * np.std(prices)
            
            def _maxdist(xi, xj, m):
                return max([abs(ua - va) for ua, va in zip(xi, xj)])
            
            def _phi(m):
                patterns = np.array([prices[i:i + m] for i in range(len(prices) - m + 1)])
                C = []
                
                for i in range(len(patterns)):
                    template = patterns[i]
                    matches = sum([1 for j in range(len(patterns)) 
                                 if _maxdist(template, patterns[j], m) <= r])
                    C.append(matches / float(len(patterns)))
                
                phi = np.mean([np.log(c) for c in C if c > 0])
                return phi
            
            app_entropy = _phi(m) - _phi(m + 1)
            return abs(app_entropy)
            
        except Exception:
            return 0.5
    
    def _calculate_lyapunov_exponent(self, prices: np.ndarray) -> float:
        """Calculate largest Lyapunov exponent"""
        try:
            if len(prices) < 15:
                return 0.0
            
            # Embedding dimension and delay
            m = 3
            tau = 1
            
            # Create embedded vectors
            N = len(prices) - (m - 1) * tau
            embedded = np.zeros((N, m))
            
            for i in range(N):
                for j in range(m):
                    embedded[i, j] = prices[i + j * tau]
            
            # Find nearest neighbors and calculate divergence
            divergences = []
            
            for i in range(N // 2):
                # Find nearest neighbor
                distances = [euclidean(embedded[i], embedded[j]) 
                           for j in range(N) if abs(i - j) > 1]
                
                if distances:
                    min_dist_idx = np.argmin(distances)
                    # Map back to original index
                    j = min_dist_idx if min_dist_idx < i else min_dist_idx + 1
                    
                    # Calculate divergence over time
                    max_steps = min(10, N - max(i, j) - 1)
                    
                    for step in range(1, max_steps):
                        if i + step < N and j + step < N:
                            d0 = euclidean(embedded[i], embedded[j])
                            dt = euclidean(embedded[i + step], embedded[j + step])
                            
                            if d0 > 0 and dt > 0:
                                divergences.append(np.log(dt / d0) / step)
            
            if divergences:
                return np.mean(divergences)
            else:
                return 0.0
                
        except Exception:
            return 0.0
    
    def _initialize_pattern_templates(self) -> Dict[str, np.ndarray]:
        """Initialize pattern templates for template matching"""
        templates = {}
        
        # Create normalized pattern templates
        x = np.linspace(0, 1, 10)
        
        # Head and Shoulders
        templates['head_shoulders'] = np.array([0.3, 0.5, 0.3, 0.8, 0.3, 0.5, 0.3, 0.2, 0.2, 0.2])
        
        # Double Top
        templates['double_top'] = np.array([0.2, 0.7, 0.3, 0.2, 0.3, 0.7, 0.2, 0.1, 0.1, 0.1])
        
        # Double Bottom
        templates['double_bottom'] = np.array([0.8, 0.3, 0.7, 0.8, 0.7, 0.3, 0.8, 0.9, 0.9, 0.9])
        
        # Triangle (ascending)
        templates['ascending_triangle'] = np.array([0.2, 0.8, 0.3, 0.8, 0.4, 0.8, 0.5, 0.8, 0.6, 0.8])
        
        # Triangle (descending)
        templates['descending_triangle'] = np.array([0.8, 0.2, 0.7, 0.2, 0.6, 0.2, 0.5, 0.2, 0.4, 0.2])
        
        # Cup and Handle
        templates['cup_handle'] = np.array([0.8, 0.3, 0.2, 0.2, 0.2, 0.3, 0.8, 0.6, 0.7, 0.9])
        
        # Flag pattern
        templates['flag'] = np.array([0.2, 0.8, 0.7, 0.6, 0.5, 0.4, 0.3, 0.2, 0.1, 0.9])
        
        # Pennant
        templates['pennant'] = np.array([0.2, 0.8, 0.6, 0.4, 0.5, 0.3, 0.4, 0.6, 0.5, 0.9])
        
        return templates
    
    def _generate_feature_names(self, n_geometric: int, n_statistical: int, 
                               n_technical: int, n_volume: int, 
                               n_fractal: int, n_sentiment: int) -> List[str]:
        """Generate descriptive feature names"""
        names = []
        
        # Geometric feature names
        names.extend([f'geometric_{i}' for i in range(n_geometric)])
        
        # Statistical feature names
        names.extend([f'statistical_{i}' for i in range(n_statistical)])
        
        # Technical feature names
        names.extend([f'technical_{i}' for i in range(n_technical)])
        
        # Volume feature names
        names.extend([f'volume_{i}' for i in range(n_volume)])
        
        # Fractal feature names
        names.extend([f'fractal_{i}' for i in range(n_fractal)])
        
        # Sentiment feature names
        names.extend([f'sentiment_{i}' for i in range(n_sentiment)])
        
        return names
    
    def _create_empty_features(self, window_size: int) -> PatternFeatures:
        """Create empty PatternFeatures object for error cases"""
        return PatternFeatures(
            geometric_features=np.array([]),
            statistical_features=np.array([]),
            technical_features=np.array([]),
            volume_features=np.array([]),
            fractal_features=np.array([]),
            sentiment_features=np.array([]),
            combined_features=np.array([]),
            feature_names=[],
            window_size=window_size,
            timestamp=pd.Timestamp.now()
        )
    
    def update_feature_importance(self, feature_names: List[str], importance_scores: np.ndarray):
        """Update feature importance tracking for model improvement"""
        for name, score in zip(feature_names, importance_scores):
            if name not in self.feature_importance:
                self.feature_importance[name] = []
                self.feature_usage_count[name] = 0
            
            self.feature_importance[name].append(score)
            self.feature_usage_count[name] += 1
    
    def get_top_features(self, top_k: int = 20) -> List[Tuple[str, float]]:
        """Get top K most important features"""
        avg_importance = {}
        
        for feature_name, scores in self.feature_importance.items():
            avg_importance[feature_name] = np.mean(scores)
        
        sorted_features = sorted(avg_importance.items(), key=lambda x: x[1], reverse=True)
        return sorted_features[:top_k]
