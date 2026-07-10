#!/usr/bin/env python3
"""
Intelligent Pattern Postprocessor
Validates, filters, and enhances detected patterns using advanced analytics
"""

import numpy as np
import pandas as pd
import logging
from typing import Dict, List, Tuple, Optional, Any, Union
from dataclasses import dataclass, asdict
from datetime import datetime, timedelta
from enum import Enum
import json
from pathlib import Path

# Statistical and ML imports
from scipy import stats
from scipy.signal import find_peaks, savgol_filter
from sklearn.cluster import DBSCAN
from sklearn.preprocessing import StandardScaler
from sklearn.metrics.pairwise import cosine_similarity
import warnings
warnings.filterwarnings('ignore')

class PatternQuality(Enum):
    EXCELLENT = "excellent"
    GOOD = "good"
    AVERAGE = "average"
    POOR = "poor"
    INVALID = "invalid"

class PatternPhase(Enum):
    FORMATION = "formation"
    BREAKOUT = "breakout"
    CONTINUATION = "continuation"
    COMPLETION = "completion"
    INVALIDATION = "invalidation"

@dataclass
class ValidationResult:
    """Result of pattern validation"""
    is_valid: bool
    quality_score: float
    quality_rating: PatternQuality
    confidence_adjustment: float
    validation_reasons: List[str]
    technical_score: float
    volume_score: float
    timeframe_score: float
    market_context_score: float

@dataclass
class EnhancedPattern:
    """Enhanced pattern with additional analytics"""
    original_pattern: Any  # PatternDetection object
    validation_result: ValidationResult
    phase: PatternPhase
    target_prices: Dict[str, float]
    stop_loss_levels: Dict[str, float]
    support_resistance_levels: List[float]
    volume_analysis: Dict[str, Any]
    momentum_indicators: Dict[str, float]
    risk_metrics: Dict[str, float]
    trading_recommendations: Dict[str, Any]
    similar_historical_patterns: List[Dict[str, Any]]
    market_correlations: Dict[str, float]
    news_sentiment_impact: float
    seasonality_factor: float

class AdvancedPatternPostprocessor:
    """
    Advanced postprocessor for pattern validation and enhancement using AI techniques
    """
    
    def __init__(self, historical_data_path: str = None, enable_advanced_analytics: bool = True):
        self.historical_data_path = historical_data_path
        self.enable_advanced_analytics = enable_advanced_analytics
        self.logger = logging.getLogger(__name__)
        
        # Validation parameters
        self.min_confidence_threshold = 0.6
        self.min_quality_score = 0.5
        self.volume_threshold_multiplier = 1.2
        
        # Historical pattern database for similarity matching
        self.historical_patterns = {}
        self.pattern_success_rates = {}
        
        # Market context parameters
        self.volatility_regime_thresholds = {
            'low': 0.01,
            'medium': 0.03,
            'high': 0.05
        }
        
        # Technical analysis parameters
        self.support_resistance_tolerance = 0.02  # 2% tolerance for S/R levels
        self.trend_confirmation_periods = [5, 10, 20, 50]
        
        # Load historical data if available
        self._load_historical_patterns()
        
        # Initialize advanced analytics components
        if self.enable_advanced_analytics:
            self._initialize_advanced_components()
    
    def process_patterns(self, 
                        detected_patterns: List[Any],
                        market_data: pd.DataFrame,
                        symbol: str = None,
                        current_time: datetime = None) -> List[EnhancedPattern]:
        """
        Process detected patterns through comprehensive validation and enhancement
        
        Args:
            detected_patterns: List of PatternDetection objects
            market_data: OHLCV data with technical indicators
            symbol: Trading symbol
            current_time: Current timestamp
            
        Returns:
            List of validated and enhanced patterns
        """
        try:
            if not detected_patterns:
                return []
            
            enhanced_patterns = []
            current_time = current_time or datetime.now()
            
            self.logger.info(f"Processing {len(detected_patterns)} detected patterns")
            
            for pattern in detected_patterns:
                try:
                    # Validate pattern
                    validation = self._validate_pattern(pattern, market_data, symbol)
                    
                    if validation.is_valid:
                        # Enhance pattern with additional analytics
                        enhanced = self._enhance_pattern(pattern, validation, market_data, symbol, current_time)
                        enhanced_patterns.append(enhanced)
                        
                        self.logger.debug(f"Enhanced pattern: {pattern.pattern_type} "
                                        f"(Quality: {validation.quality_rating.value})")
                    else:
                        self.logger.debug(f"Pattern {pattern.pattern_type} failed validation: "
                                        f"{', '.join(validation.validation_reasons)}")
                
                except Exception as e:
                    self.logger.warning(f"Error processing pattern {pattern.pattern_type}: {e}")
                    continue
            
            # Rank and filter patterns
            enhanced_patterns = self._rank_and_filter_patterns(enhanced_patterns)
            
            # Update historical database
            self._update_historical_patterns(enhanced_patterns, symbol)
            
            self.logger.info(f"Processed patterns: {len(enhanced_patterns)} valid out of {len(detected_patterns)}")
            
            return enhanced_patterns
            
        except Exception as e:
            self.logger.error(f"Error in pattern processing: {e}")
            return []
    
    def _validate_pattern(self, pattern: Any, market_data: pd.DataFrame, symbol: str = None) -> ValidationResult:
        """Comprehensive pattern validation"""
        try:
            validation_reasons = []
            quality_factors = []
            
            # Basic validation checks
            if pattern.confidence < self.min_confidence_threshold:
                return ValidationResult(
                    is_valid=False,
                    quality_score=0.0,
                    quality_rating=PatternQuality.INVALID,
                    confidence_adjustment=0.0,
                    validation_reasons=["Confidence below threshold"],
                    technical_score=0.0,
                    volume_score=0.0,
                    timeframe_score=0.0,
                    market_context_score=0.0
                )
            
            # Technical validation
            technical_score = self._validate_technical_aspects(pattern, market_data)
            quality_factors.append(('technical', technical_score, 0.3))
            
            if technical_score < 0.4:
                validation_reasons.append("Poor technical pattern formation")
            
            # Volume validation
            volume_score = self._validate_volume_profile(pattern, market_data)
            quality_factors.append(('volume', volume_score, 0.25))
            
            if volume_score < 0.3:
                validation_reasons.append("Insufficient volume confirmation")
            
            # Timeframe validation
            timeframe_score = self._validate_timeframe_context(pattern, market_data)
            quality_factors.append(('timeframe', timeframe_score, 0.2))
            
            # Market context validation
            market_context_score = self._validate_market_context(pattern, market_data, symbol)
            quality_factors.append(('market_context', market_context_score, 0.25))
            
            # Calculate overall quality score
            quality_score = sum(score * weight for _, score, weight in quality_factors)
            
            # Determine quality rating
            quality_rating = self._determine_quality_rating(quality_score)
            
            # Calculate confidence adjustment
            confidence_adjustment = self._calculate_confidence_adjustment(quality_factors)
            
            # Determine if pattern is valid
            is_valid = (
                quality_score >= self.min_quality_score and
                technical_score >= 0.4 and
                len(validation_reasons) == 0
            )
            
            return ValidationResult(
                is_valid=is_valid,
                quality_score=quality_score,
                quality_rating=quality_rating,
                confidence_adjustment=confidence_adjustment,
                validation_reasons=validation_reasons,
                technical_score=technical_score,
                volume_score=volume_score,
                timeframe_score=timeframe_score,
                market_context_score=market_context_score
            )
            
        except Exception as e:
            self.logger.warning(f"Error validating pattern: {e}")
            return ValidationResult(
                is_valid=False,
                quality_score=0.0,
                quality_rating=PatternQuality.INVALID,
                confidence_adjustment=0.0,
                validation_reasons=["Validation error"],
                technical_score=0.0,
                volume_score=0.0,
                timeframe_score=0.0,
                market_context_score=0.0
            )
    
    def _validate_technical_aspects(self, pattern: Any, market_data: pd.DataFrame) -> float:
        """Validate technical aspects of the pattern"""
        try:
            scores = []
            
            # Get pattern data window
            start_idx = max(0, pattern.start_index)
            end_idx = min(len(market_data), pattern.end_index)
            pattern_data = market_data.iloc[start_idx:end_idx]
            
            if len(pattern_data) < 5:
                return 0.0
            
            # Price action consistency
            price_consistency = self._check_price_action_consistency(pattern, pattern_data)
            scores.append(price_consistency)
            
            # Support/Resistance validation
            sr_validation = self._validate_support_resistance(pattern, pattern_data)
            scores.append(sr_validation)
            
            # Pattern geometry validation
            geometry_score = self._validate_pattern_geometry(pattern, pattern_data)
            scores.append(geometry_score)
            
            # Trend context validation
            trend_context = self._validate_trend_context(pattern, pattern_data)
            scores.append(trend_context)
            
            # Moving average alignment
            ma_alignment = self._check_moving_average_alignment(pattern, pattern_data)
            scores.append(ma_alignment)
            
            return np.mean(scores)
            
        except Exception as e:
            self.logger.warning(f"Error in technical validation: {e}")
            return 0.0
    
    def _check_price_action_consistency(self, pattern: Any, data: pd.DataFrame) -> float:
        """Check if price action is consistent with pattern type"""
        try:
            if 'close' not in data.columns:
                return 0.5
            
            closes = data['close'].values
            pattern_type = pattern.pattern_type.lower()
            
            # Pattern-specific validation rules
            if 'double_top' in pattern_type:
                # Should have two peaks at similar levels
                peaks, _ = find_peaks(closes, distance=len(closes)//4)
                if len(peaks) >= 2:
                    peak_similarity = 1.0 - abs(closes[peaks[-2]] - closes[peaks[-1]]) / np.mean(closes)
                    return min(1.0, max(0.0, peak_similarity * 2))
            
            elif 'double_bottom' in pattern_type:
                # Should have two troughs at similar levels
                troughs, _ = find_peaks(-closes, distance=len(closes)//4)
                if len(troughs) >= 2:
                    trough_similarity = 1.0 - abs(closes[troughs[-2]] - closes[troughs[-1]]) / np.mean(closes)
                    return min(1.0, max(0.0, trough_similarity * 2))
            
            elif 'head_shoulders' in pattern_type:
                # Should have three peaks with middle one highest
                peaks, _ = find_peaks(closes, distance=len(closes)//5)
                if len(peaks) >= 3:
                    left_shoulder = closes[peaks[-3]]
                    head = closes[peaks[-2]]
                    right_shoulder = closes[peaks[-1]]
                    
                    if head > left_shoulder and head > right_shoulder:
                        shoulder_symmetry = 1.0 - abs(left_shoulder - right_shoulder) / np.mean(closes)
                        return min(1.0, max(0.0, shoulder_symmetry))
            
            elif 'triangle' in pattern_type:
                # Should show converging highs and lows
                highs = data['high'].values
                lows = data['low'].values
                
                # Check for converging trend lines
                high_slope = self._calculate_trend_slope(highs)
                low_slope = self._calculate_trend_slope(lows)
                
                convergence = abs(high_slope + low_slope)  # Should be close to 0 for symmetric triangle
                return max(0.0, 1.0 - convergence * 10)
            
            elif 'breakout' in pattern_type:
                # Should show clear break above resistance
                recent_high = np.max(closes[-5:]) if len(closes) >= 5 else closes[-1]
                prior_resistance = np.max(closes[:-5]) if len(closes) > 5 else closes[0]
                
                if recent_high > prior_resistance * 1.01:  # At least 1% breakout
                    return 0.8
            
            # Default consistency check - look for clean price action
            smoothed_prices = savgol_filter(closes, min(11, len(closes)//2*2+1), 3)
            price_noise = np.std(closes - smoothed_prices) / np.mean(closes)
            consistency = max(0.0, 1.0 - price_noise * 20)
            
            return consistency
            
        except Exception as e:
            self.logger.warning(f"Error checking price action consistency: {e}")
            return 0.5
    
    def _validate_support_resistance(self, pattern: Any, data: pd.DataFrame) -> float:
        """Validate support and resistance levels"""
        try:
            if 'high' not in data.columns or 'low' not in data.columns:
                return 0.5
            
            highs = data['high'].values
            lows = data['low'].values
            closes = data['close'].values
            
            # Find significant support and resistance levels
            resistance_levels = self._find_resistance_levels(highs, closes)
            support_levels = self._find_support_levels(lows, closes)
            
            score_factors = []
            
            # Check if current price respects these levels
            current_price = closes[-1]
            
            # Distance to nearest resistance
            if resistance_levels:
                nearest_resistance = min(resistance_levels, key=lambda x: abs(x - current_price))
                if current_price < nearest_resistance:
                    resistance_score = min(1.0, (nearest_resistance - current_price) / current_price * 20)
                    score_factors.append(resistance_score)
            
            # Distance to nearest support
            if support_levels:
                nearest_support = min(support_levels, key=lambda x: abs(x - current_price))
                if current_price > nearest_support:
                    support_score = min(1.0, (current_price - nearest_support) / current_price * 20)
                    score_factors.append(support_score)
            
            # Number of times levels have been tested
            level_strength = self._calculate_level_strength(resistance_levels + support_levels, highs, lows)
            score_factors.append(level_strength)
            
            return np.mean(score_factors) if score_factors else 0.5
            
        except Exception as e:
            self.logger.warning(f"Error validating support/resistance: {e}")
            return 0.5
    
    def _validate_pattern_geometry(self, pattern: Any, data: pd.DataFrame) -> float:
        """Validate geometric properties of the pattern"""
        try:
            if 'close' not in data.columns:
                return 0.5
            
            closes = data['close'].values
            pattern_type = pattern.pattern_type.lower()
            
            # Calculate pattern dimensions
            pattern_height = np.max(closes) - np.min(closes)
            pattern_width = len(closes)
            height_width_ratio = pattern_height / (np.mean(closes) * pattern_width / 100)
            
            # Pattern-specific geometry validation
            geometry_score = 0.5
            
            if 'triangle' in pattern_type:
                # Triangles should have reasonable width-to-height ratio
                ideal_ratio = 0.5  # Adjust based on timeframe
                ratio_score = 1.0 - abs(height_width_ratio - ideal_ratio) / ideal_ratio
                geometry_score = max(0.0, min(1.0, ratio_score))
            
            elif 'flag' in pattern_type or 'pennant' in pattern_type:
                # Flags should be relatively small compared to the preceding move
                if len(closes) >= 10:
                    flag_height = np.max(closes[-10:]) - np.min(closes[-10:])
                    pole_height = abs(closes[-10] - closes[0]) if len(closes) > 10 else pattern_height
                    
                    if pole_height > 0:
                        flag_ratio = flag_height / pole_height
                        geometry_score = max(0.0, 1.0 - flag_ratio) if flag_ratio < 0.5 else 0.2
            
            elif 'head_shoulders' in pattern_type:
                # Head and shoulders should have symmetrical shoulders
                if len(closes) >= 15:
                    third_point = len(closes) // 3
                    left_shoulder_avg = np.mean(closes[:third_point])
                    head_avg = np.mean(closes[third_point:2*third_point])
                    right_shoulder_avg = np.mean(closes[2*third_point:])
                    
                    shoulder_symmetry = 1.0 - abs(left_shoulder_avg - right_shoulder_avg) / np.mean(closes)
                    head_prominence = (head_avg - max(left_shoulder_avg, right_shoulder_avg)) / np.mean(closes)
                    
                    geometry_score = (shoulder_symmetry + min(1.0, head_prominence * 10)) / 2
            
            # Add general geometry health checks
            price_volatility = np.std(closes) / np.mean(closes)
            volatility_score = max(0.0, 1.0 - price_volatility * 5)  # Penalize excessive volatility
            
            final_score = (geometry_score + volatility_score) / 2
            return max(0.0, min(1.0, final_score))
            
        except Exception as e:
            self.logger.warning(f"Error validating pattern geometry: {e}")
            return 0.5
    
    def _validate_volume_profile(self, pattern: Any, market_data: pd.DataFrame) -> float:
        """Validate volume characteristics"""
        try:
            if 'volume' not in market_data.columns:
                return 0.5
            
            # Get pattern data window
            start_idx = max(0, pattern.start_index)
            end_idx = min(len(market_data), pattern.end_index)
            pattern_data = market_data.iloc[start_idx:end_idx]
            
            if len(pattern_data) < 3:
                return 0.5
            
            volumes = pattern_data['volume'].values
            closes = pattern_data['close'].values if 'close' in pattern_data.columns else np.ones(len(volumes))
            
            volume_scores = []
            
            # Volume trend analysis
            volume_trend = self._analyze_volume_trend(volumes, pattern.pattern_type)
            volume_scores.append(volume_trend)
            
            # Volume-price correlation
            if len(volumes) > 1 and len(closes) > 1:
                price_changes = np.diff(closes)
                volume_changes = np.diff(volumes)
                
                if np.std(price_changes) > 0 and np.std(volume_changes) > 0:
                    correlation = np.corrcoef(np.abs(price_changes), volume_changes[:-1] if len(volume_changes) > len(price_changes) else volume_changes)[0, 1]
                    correlation_score = abs(correlation) if not np.isnan(correlation) else 0.5
                    volume_scores.append(correlation_score)
            
            # Volume breakout confirmation
            if 'breakout' in pattern.pattern_type.lower():
                recent_volume = np.mean(volumes[-3:]) if len(volumes) >= 3 else volumes[-1]
                baseline_volume = np.mean(volumes[:-3]) if len(volumes) > 3 else recent_volume
                
                if baseline_volume > 0:
                    volume_surge = recent_volume / baseline_volume
                    breakout_score = min(1.0, max(0.0, (volume_surge - 1.0) / 1.0))  # Expect at least 100% increase
                    volume_scores.append(breakout_score)
            
            # Average volume relative to historical
            avg_volume = np.mean(volumes)
            
            # Get historical volume context if available
            if len(market_data) > len(pattern_data):
                historical_data = market_data.iloc[:start_idx]
                if len(historical_data) > 0 and 'volume' in historical_data.columns:
                    historical_avg = historical_data['volume'].mean()
                    if historical_avg > 0:
                        relative_volume = avg_volume / historical_avg
                        relative_score = min(1.0, relative_volume) if relative_volume > 0.5 else 0.2
                        volume_scores.append(relative_score)
            
            return np.mean(volume_scores) if volume_scores else 0.5
            
        except Exception as e:
            self.logger.warning(f"Error validating volume profile: {e}")
            return 0.5
    
    def _analyze_volume_trend(self, volumes: np.ndarray, pattern_type: str) -> float:
        """Analyze volume trend for pattern type"""
        try:
            if len(volumes) < 3:
                return 0.5
            
            # Calculate volume trend
            x = np.arange(len(volumes))
            slope, intercept, r_value, p_value, std_err = stats.linregress(x, volumes)
            
            pattern_type_lower = pattern_type.lower()
            
            # Pattern-specific volume expectations
            if 'breakout' in pattern_type_lower or 'breakdown' in pattern_type_lower:
                # Expect increasing volume
                if slope > 0 and r_value > 0.3:
                    return min(1.0, r_value * 2)
                else:
                    return 0.2
            
            elif 'consolidation' in pattern_type_lower or 'triangle' in pattern_type_lower:
                # Expect decreasing volume during formation
                if slope < 0 and abs(r_value) > 0.2:
                    return min(1.0, abs(r_value) * 2)
                else:
                    return 0.3
            
            elif 'reversal' in pattern_type_lower:
                # Expect volume spike at reversal point
                max_volume_idx = np.argmax(volumes)
                if max_volume_idx > len(volumes) * 0.7:  # Volume spike in last 30%
                    return 0.8
                else:
                    return 0.4
            
            # Default: moderate volume consistency
            volume_consistency = 1.0 - (np.std(volumes) / np.mean(volumes)) if np.mean(volumes) > 0 else 0.5
            return max(0.0, min(1.0, volume_consistency))
            
        except Exception as e:
            self.logger.warning(f"Error analyzing volume trend: {e}")
            return 0.5
    
    def _validate_timeframe_context(self, pattern: Any, market_data: pd.DataFrame) -> float:
        """Validate pattern in timeframe context"""
        try:
            pattern_duration = pattern.end_index - pattern.start_index
            
            # Check if pattern duration is reasonable for pattern type
            expected_durations = {
                'head_shoulders': (15, 50),
                'double_top': (10, 30),
                'double_bottom': (10, 30),
                'triangle': (15, 60),
                'flag': (5, 15),
                'pennant': (5, 15),
                'breakout': (1, 10),
                'consolidation': (20, 100)
            }
            
            pattern_base = pattern.pattern_type.lower().split('_')[0]
            if pattern_base in expected_durations:
                min_duration, max_duration = expected_durations[pattern_base]
                
                if min_duration <= pattern_duration <= max_duration:
                    return 1.0
                elif pattern_duration < min_duration:
                    return max(0.2, pattern_duration / min_duration)
                else:
                    return max(0.2, max_duration / pattern_duration)
            
            # Default scoring based on pattern maturity
            if pattern_duration < 5:
                return 0.3  # Too short
            elif pattern_duration > 100:
                return 0.4  # Too long
            else:
                return 0.7  # Reasonable duration
                
        except Exception as e:
            self.logger.warning(f"Error validating timeframe context: {e}")
            return 0.5
    
    def _validate_market_context(self, pattern: Any, market_data: pd.DataFrame, symbol: str = None) -> float:
        """Validate pattern against broader market context"""
        try:
            context_scores = []
            
            # Market volatility context
            if 'close' in market_data.columns and len(market_data) > 20:
                recent_prices = market_data['close'].tail(20)
                returns = recent_prices.pct_change().dropna()
                volatility = returns.std()
                
                # Determine volatility regime
                if volatility < self.volatility_regime_thresholds['low']:
                    vol_regime = 'low'
                elif volatility < self.volatility_regime_thresholds['medium']:
                    vol_regime = 'medium'
                else:
                    vol_regime = 'high'
                
                # Score based on pattern-volatility appropriateness
                pattern_type = pattern.pattern_type.lower()
                if 'breakout' in pattern_type and vol_regime in ['medium', 'high']:
                    context_scores.append(0.8)
                elif 'consolidation' in pattern_type and vol_regime == 'low':
                    context_scores.append(0.8)
                else:
                    context_scores.append(0.6)
            
            # Trend context
            if len(market_data) > 50:
                trend_score = self._assess_trend_context(market_data, pattern)
                context_scores.append(trend_score)
            
            # Market phase analysis
            market_phase_score = self._analyze_market_phase(market_data)
            context_scores.append(market_phase_score)
            
            return np.mean(context_scores) if context_scores else 0.5
            
        except Exception as e:
            self.logger.warning(f"Error validating market context: {e}")
            return 0.5
    
    def _enhance_pattern(self, 
                        pattern: Any, 
                        validation: ValidationResult,
                        market_data: pd.DataFrame,
                        symbol: str = None,
                        current_time: datetime = None) -> EnhancedPattern:
        """Enhance pattern with comprehensive analytics"""
        try:
            # Determine pattern phase
            phase = self._determine_pattern_phase(pattern, market_data)
            
            # Calculate target prices and stop losses
            target_prices = self._calculate_target_prices(pattern, market_data)
            stop_loss_levels = self._calculate_stop_loss_levels(pattern, market_data)
            
            # Identify support and resistance levels
            sr_levels = self._identify_support_resistance_levels(pattern, market_data)
            
            # Analyze volume characteristics
            volume_analysis = self._analyze_volume_characteristics(pattern, market_data)
            
            # Calculate momentum indicators
            momentum_indicators = self._calculate_momentum_indicators(pattern, market_data)
            
            # Assess risk metrics
            risk_metrics = self._calculate_risk_metrics(pattern, market_data, target_prices, stop_loss_levels)
            
            # Generate trading recommendations
            trading_recommendations = self._generate_trading_recommendations(
                pattern, validation, target_prices, stop_loss_levels, risk_metrics
            )
            
            # Find similar historical patterns
            similar_patterns = self._find_similar_historical_patterns(pattern, symbol)
            
            # Calculate market correlations
            market_correlations = self._calculate_market_correlations(pattern, market_data, symbol)
            
            # Assess news sentiment impact (placeholder)
            news_sentiment_impact = self._assess_news_sentiment_impact(pattern, symbol, current_time)
            
            # Calculate seasonality factor
            seasonality_factor = self._calculate_seasonality_factor(pattern, current_time)
            
            return EnhancedPattern(
                original_pattern=pattern,
                validation_result=validation,
                phase=phase,
                target_prices=target_prices,
                stop_loss_levels=stop_loss_levels,
                support_resistance_levels=sr_levels,
                volume_analysis=volume_analysis,
                momentum_indicators=momentum_indicators,
                risk_metrics=risk_metrics,
                trading_recommendations=trading_recommendations,
                similar_historical_patterns=similar_patterns,
                market_correlations=market_correlations,
                news_sentiment_impact=news_sentiment_impact,
                seasonality_factor=seasonality_factor
            )
            
        except Exception as e:
            self.logger.error(f"Error enhancing pattern: {e}")
            # Return minimal enhanced pattern
            return EnhancedPattern(
                original_pattern=pattern,
                validation_result=validation,
                phase=PatternPhase.FORMATION,
                target_prices={},
                stop_loss_levels={},
                support_resistance_levels=[],
                volume_analysis={},
                momentum_indicators={},
                risk_metrics={},
                trading_recommendations={},
                similar_historical_patterns=[],
                market_correlations={},
                news_sentiment_impact=0.0,
                seasonality_factor=1.0
            )
    
    def _calculate_target_prices(self, pattern: Any, market_data: pd.DataFrame) -> Dict[str, float]:
        """Calculate target prices based on pattern type and technical analysis"""
        try:
            targets = {}
            
            if 'close' not in market_data.columns:
                return targets
            
            # Get pattern data
            start_idx = max(0, pattern.start_index)
            end_idx = min(len(market_data), pattern.end_index)
            pattern_data = market_data.iloc[start_idx:end_idx]
            
            current_price = market_data['close'].iloc[-1]
            pattern_height = pattern_data['high'].max() - pattern_data['low'].min()
            
            pattern_type = pattern.pattern_type.lower()
            
            # Pattern-specific target calculations
            if 'double_top' in pattern_type:
                # Target is pattern height below the neckline
                neckline = pattern_data['low'].min()
                targets['primary'] = neckline - pattern_height
                targets['conservative'] = neckline - pattern_height * 0.5
                targets['aggressive'] = neckline - pattern_height * 1.5
                
            elif 'double_bottom' in pattern_type:
                # Target is pattern height above the neckline
                neckline = pattern_data['high'].max()
                targets['primary'] = neckline + pattern_height
                targets['conservative'] = neckline + pattern_height * 0.5
                targets['aggressive'] = neckline + pattern_height * 1.5
                
            elif 'head_shoulders' in pattern_type:
                # Target is head-to-neckline distance below neckline
                if len(pattern_data) >= 3:
                    head_price = pattern_data['high'].max()
                    neckline = pattern_data['low'].min()
                    head_height = head_price - neckline
                    
                    targets['primary'] = neckline - head_height
                    targets['conservative'] = neckline - head_height * 0.618  # Fibonacci
                    targets['aggressive'] = neckline - head_height * 1.618
                    
            elif 'triangle' in pattern_type:
                # Target is triangle height in breakout direction
                breakout_direction = 1 if current_price > pattern_data['high'].mean() else -1
                targets['primary'] = current_price + (pattern_height * breakout_direction)
                targets['conservative'] = current_price + (pattern_height * 0.5 * breakout_direction)
                targets['aggressive'] = current_price + (pattern_height * 1.5 * breakout_direction)
                
            elif 'breakout' in pattern_type:
                # Target based on recent consolidation range
                consolidation_height = pattern_height
                targets['primary'] = current_price + consolidation_height
                targets['conservative'] = current_price + consolidation_height * 0.618
                targets['aggressive'] = current_price + consolidation_height * 1.618
                
            elif 'flag' in pattern_type or 'pennant' in pattern_type:
                # Target is flagpole height added to breakout point
                if len(market_data) > end_idx:
                    flagpole_start = max(0, start_idx - 20)
                    flagpole_height = abs(market_data['close'].iloc[start_idx] - market_data['close'].iloc[flagpole_start])
                    
                    direction = 1 if 'bullish' in pattern_type else -1
                    targets['primary'] = current_price + (flagpole_height * direction)
                    targets['conservative'] = current_price + (flagpole_height * 0.618 * direction)
                    targets['aggressive'] = current_price + (flagpole_height * 1.618 * direction)
            
            # Apply Fibonacci levels for additional targets
            if targets:
                primary_target = targets.get('primary', current_price)
                target_distance = abs(primary_target - current_price)
                
                # Add Fibonacci-based targets
                targets['fibonacci_382'] = current_price + target_distance * 0.382 * (1 if primary_target > current_price else -1)
                targets['fibonacci_618'] = current_price + target_distance * 0.618 * (1 if primary_target > current_price else -1)
                targets['fibonacci_1618'] = current_price + target_distance * 1.618 * (1 if primary_target > current_price else -1)
            
            return targets
            
        except Exception as e:
            self.logger.warning(f"Error calculating target prices: {e}")
            return {}
    
    def _calculate_stop_loss_levels(self, pattern: Any, market_data: pd.DataFrame) -> Dict[str, float]:
        """Calculate stop loss levels"""
        try:
            stops = {}
            
            if 'close' not in market_data.columns:
                return stops
            
            current_price = market_data['close'].iloc[-1]
            
            # Get pattern data
            start_idx = max(0, pattern.start_index)
            end_idx = min(len(market_data), pattern.end_index)
            pattern_data = market_data.iloc[start_idx:end_idx]
            
            pattern_type = pattern.pattern_type.lower()
            
            # Pattern-specific stop loss calculations
            if 'breakout' in pattern_type:
                # Stop below/above the breakout level
                breakout_level = pattern_data['high'].max() if 'bullish' in pattern_type else pattern_data['low'].min()
                buffer = current_price * 0.01  # 1% buffer
                
                if 'bullish' in pattern_type:
                    stops['tight'] = breakout_level - buffer
                    stops['normal'] = breakout_level - buffer * 2
                    stops['wide'] = breakout_level - buffer * 3
                else:
                    stops['tight'] = breakout_level + buffer
                    stops['normal'] = breakout_level + buffer * 2
                    stops['wide'] = breakout_level + buffer * 3
                    
            elif 'double_top' in pattern_type:
                # Stop above the pattern high
                pattern_high = pattern_data['high'].max()
                buffer = current_price * 0.01
                stops['tight'] = pattern_high + buffer
                stops['normal'] = pattern_high + buffer * 2
                stops['wide'] = pattern_high + buffer * 3
                
            elif 'double_bottom' in pattern_type:
                # Stop below the pattern low
                pattern_low = pattern_data['low'].min()
                buffer = current_price * 0.01
                stops['tight'] = pattern_low - buffer
                stops['normal'] = pattern_low - buffer * 2
                stops['wide'] = pattern_low - buffer * 3
                
            elif 'triangle' in pattern_type:
                # Stop outside the triangle
                if current_price > pattern_data['close'].mean():
                    # Bullish breakout - stop below lower trend line
                    lower_line = pattern_data['low'].min()
                    buffer = current_price * 0.015
                    stops['tight'] = lower_line - buffer
                    stops['normal'] = lower_line - buffer * 2
                    stops['wide'] = lower_line - buffer * 3
                else:
                    # Bearish breakout - stop above upper trend line
                    upper_line = pattern_data['high'].max()
                    buffer = current_price * 0.015
                    stops['tight'] = upper_line + buffer
                    stops['normal'] = upper_line + buffer * 2
                    stops['wide'] = upper_line + buffer * 3
            
            # Default stops based on volatility
            if not stops and len(market_data) > 20:
                recent_data = market_data.tail(20)
                volatility = recent_data['close'].pct_change().std()
                
                stops['tight'] = current_price * (1 - volatility * 1.5)
                stops['normal'] = current_price * (1 - volatility * 2.5)
                stops['wide'] = current_price * (1 - volatility * 4.0)
            
            return stops
            
        except Exception as e:
            self.logger.warning(f"Error calculating stop loss levels: {e}")
            return {}
    
    def _rank_and_filter_patterns(self, patterns: List[EnhancedPattern]) -> List[EnhancedPattern]:
        """Rank patterns by quality and filter out low-quality ones"""
        try:
            if not patterns:
                return patterns
            
            # Calculate composite score for each pattern
            for pattern in patterns:
                scores = []
                
                # Base confidence and quality
                scores.append(pattern.original_pattern.confidence * 0.3)
                scores.append(pattern.validation_result.quality_score * 0.25)
                
                # Risk-reward ratio
                risk_reward = pattern.risk_metrics.get('risk_reward_ratio', 1.0)
                scores.append(min(1.0, risk_reward / 3.0) * 0.2)  # Normalize to 0-1
                
                # Pattern strength
                pattern_strength = pattern.original_pattern.pattern_strength
                scores.append(pattern_strength * 0.15)
                
                # Volume score
                volume_score = pattern.validation_result.volume_score
                scores.append(volume_score * 0.1)
                
                # Calculate composite score
                pattern.composite_score = sum(scores)
            
            # Sort by composite score
            patterns.sort(key=lambda x: x.composite_score, reverse=True)
            
            # Filter out patterns with very low scores
            min_composite_score = 0.6
            filtered_patterns = [p for p in patterns if p.composite_score >= min_composite_score]
            
            # Limit number of patterns to prevent overtrading
            max_patterns = 5
            return filtered_patterns[:max_patterns]
            
        except Exception as e:
            self.logger.warning(f"Error ranking and filtering patterns: {e}")
            return patterns
    
    # Helper methods for calculations
    def _calculate_trend_slope(self, prices: np.ndarray) -> float:
        """Calculate trend slope using linear regression"""
        if len(prices) < 2:
            return 0.0
        
        x = np.arange(len(prices))
        slope, _, _, _, _ = stats.linregress(x, prices)
        return slope / np.mean(prices)  # Normalize by price level
    
    def _find_resistance_levels(self, highs: np.ndarray, closes: np.ndarray) -> List[float]:
        """Find significant resistance levels"""
        if len(highs) < 5:
            return []
        
        # Find local maxima
        peaks, _ = find_peaks(highs, distance=max(1, len(highs)//10))
        
        if len(peaks) == 0:
            return []
        
        # Cluster peaks to find resistance levels
        peak_prices = highs[peaks]
        
        # Simple clustering - group peaks within 2% of each other
        resistance_levels = []
        tolerance = np.mean(closes) * self.support_resistance_tolerance
        
        for price in peak_prices:
            # Check if this price is close to an existing level
            found_level = False
            for level in resistance_levels:
                if abs(price - level) < tolerance:
                    found_level = True
                    break
            
            if not found_level:
                resistance_levels.append(price)
        
        return resistance_levels
    
    def _find_support_levels(self, lows: np.ndarray, closes: np.ndarray) -> List[float]:
        """Find significant support levels"""
        if len(lows) < 5:
            return []
        
        # Find local minima
        troughs, _ = find_peaks(-lows, distance=max(1, len(lows)//10))
        
        if len(troughs) == 0:
            return []
        
        # Cluster troughs to find support levels
        trough_prices = lows[troughs]
        
        # Simple clustering - group troughs within 2% of each other
        support_levels = []
        tolerance = np.mean(closes) * self.support_resistance_tolerance
        
        for price in trough_prices:
            # Check if this price is close to an existing level
            found_level = False
            for level in support_levels:
                if abs(price - level) < tolerance:
                    found_level = True
                    break
            
            if not found_level:
                support_levels.append(price)
        
        return support_levels
    
    def _calculate_level_strength(self, levels: List[float], highs: np.ndarray, lows: np.ndarray) -> float:
        """Calculate the strength of support/resistance levels based on number of touches"""
        if not levels:
            return 0.5
        
        total_touches = 0
        tolerance = np.mean(np.concatenate([highs, lows])) * self.support_resistance_tolerance
        
        for level in levels:
            # Count touches in highs
            high_touches = np.sum(np.abs(highs - level) < tolerance)
            low_touches = np.sum(np.abs(lows - level) < tolerance)
            total_touches += high_touches + low_touches
        
        # Normalize by number of levels and data points
        strength = total_touches / (len(levels) * (len(highs) + len(lows)))
        return min(1.0, strength * 10)  # Scale to 0-1 range
    
    def _determine_quality_rating(self, quality_score: float) -> PatternQuality:
        """Determine quality rating from score"""
        if quality_score >= 0.85:
            return PatternQuality.EXCELLENT
        elif quality_score >= 0.70:
            return PatternQuality.GOOD
        elif quality_score >= 0.55:
            return PatternQuality.AVERAGE
        elif quality_score >= 0.40:
            return PatternQuality.POOR
        else:
            return PatternQuality.INVALID
    
    def _calculate_confidence_adjustment(self, quality_factors: List[Tuple[str, float, float]]) -> float:
        """Calculate confidence adjustment based on quality factors"""
        adjustments = []
        
        for factor_name, score, weight in quality_factors:
            if score > 0.8:
                adjustments.append(0.1 * weight)  # Boost confidence
            elif score < 0.4:
                adjustments.append(-0.2 * weight)  # Reduce confidence
        
        return sum(adjustments)
    
    # Placeholder methods for advanced analytics
    def _initialize_advanced_components(self):
        """Initialize advanced analytics components"""
        # Placeholder for advanced initialization
        pass
    
    def _load_historical_patterns(self):
        """Load historical pattern database"""
        # Placeholder for loading historical patterns
        pass
    
    def _update_historical_patterns(self, patterns: List[EnhancedPattern], symbol: str):
        """Update historical pattern database"""
        # Placeholder for updating historical patterns
        pass
    
    def _determine_pattern_phase(self, pattern: Any, market_data: pd.DataFrame) -> PatternPhase:
        """Determine current phase of the pattern"""
        # Simplified phase determination
        return PatternPhase.FORMATION
    
    def _identify_support_resistance_levels(self, pattern: Any, market_data: pd.DataFrame) -> List[float]:
        """Identify key support and resistance levels"""
        try:
            if 'high' not in market_data.columns or 'low' not in market_data.columns:
                return []
            
            highs = market_data['high'].values
            lows = market_data['low'].values
            closes = market_data['close'].values if 'close' in market_data.columns else (highs + lows) / 2
            
            resistance_levels = self._find_resistance_levels(highs, closes)
            support_levels = self._find_support_levels(lows, closes)
            
            return resistance_levels + support_levels
            
        except Exception as e:
            self.logger.warning(f"Error identifying S/R levels: {e}")
            return []
    
    def _analyze_volume_characteristics(self, pattern: Any, market_data: pd.DataFrame) -> Dict[str, Any]:
        """Analyze volume characteristics"""
        analysis = {}
        
        try:
            if 'volume' not in market_data.columns:
                return analysis
            
            # Get pattern data
            start_idx = max(0, pattern.start_index)
            end_idx = min(len(market_data), pattern.end_index)
            pattern_data = market_data.iloc[start_idx:end_idx]
            
            volumes = pattern_data['volume'].values
            
            analysis['average_volume'] = np.mean(volumes)
            analysis['volume_trend'] = self._calculate_trend_slope(volumes)
            analysis['volume_volatility'] = np.std(volumes) / np.mean(volumes) if np.mean(volumes) > 0 else 0
            analysis['peak_volume'] = np.max(volumes)
            analysis['volume_distribution'] = {
                'q25': np.percentile(volumes, 25),
                'q50': np.percentile(volumes, 50),
                'q75': np.percentile(volumes, 75)
            }
            
        except Exception as e:
            self.logger.warning(f"Error analyzing volume characteristics: {e}")
        
        return analysis
    
    def _calculate_momentum_indicators(self, pattern: Any, market_data: pd.DataFrame) -> Dict[str, float]:
        """Calculate momentum indicators"""
        indicators = {}
        
        try:
            if 'close' not in market_data.columns:
                return indicators
            
            closes = market_data['close'].values
            
            # RSI (simplified)
            if len(closes) > 14:
                deltas = np.diff(closes)
                gains = np.where(deltas > 0, deltas, 0)
                losses = np.where(deltas < 0, -deltas, 0)
                
                avg_gain = np.mean(gains[-14:])
                avg_loss = np.mean(losses[-14:])
                
                if avg_loss > 0:
                    rs = avg_gain / avg_loss
                    rsi = 100 - (100 / (1 + rs))
                    indicators['rsi'] = rsi
            
            # MACD (simplified)
            if len(closes) > 26:
                ema_12 = self._calculate_ema(closes, 12)
                ema_26 = self._calculate_ema(closes, 26)
                macd = ema_12 - ema_26
                indicators['macd'] = macd / closes[-1]  # Normalize
            
            # Price momentum
            if len(closes) > 10:
                momentum = (closes[-1] - closes[-10]) / closes[-10]
                indicators['momentum_10'] = momentum
            
        except Exception as e:
            self.logger.warning(f"Error calculating momentum indicators: {e}")
        
        return indicators
    
    def _calculate_ema(self, prices: np.ndarray, period: int) -> float:
        """Calculate Exponential Moving Average"""
        if len(prices) < period:
            return np.mean(prices)
        
        alpha = 2.0 / (period + 1)
        ema = prices[0]
        
        for price in prices[1:]:
            ema = alpha * price + (1 - alpha) * ema
        
        return ema
    
    def _calculate_risk_metrics(self, 
                               pattern: Any, 
                               market_data: pd.DataFrame,
                               target_prices: Dict[str, float],
                               stop_loss_levels: Dict[str, float]) -> Dict[str, float]:
        """Calculate risk metrics"""
        metrics = {}
        
        try:
            current_price = market_data['close'].iloc[-1] if 'close' in market_data.columns else 0
            
            if current_price == 0:
                return metrics
            
            # Risk-reward ratios
            if target_prices and stop_loss_levels:
                primary_target = target_prices.get('primary', current_price)
                normal_stop = stop_loss_levels.get('normal', current_price)
                
                if normal_stop != current_price:
                    potential_reward = abs(primary_target - current_price)
                    potential_risk = abs(current_price - normal_stop)
                    
                    if potential_risk > 0:
                        risk_reward_ratio = potential_reward / potential_risk
                        metrics['risk_reward_ratio'] = risk_reward_ratio
            
            # Maximum adverse excursion
            if len(market_data) > pattern.start_index:
                pattern_data = market_data.iloc[pattern.start_index:pattern.end_index]
                if 'low' in pattern_data.columns:
                    max_adverse = (current_price - pattern_data['low'].min()) / current_price
                    metrics['max_adverse_excursion'] = max_adverse
            
            # Volatility-based risk
            if len(market_data) > 20:
                recent_returns = market_data['close'].tail(20).pct_change().dropna()
                volatility = recent_returns.std()
                metrics['volatility_risk'] = volatility
            
            # Pattern confidence risk adjustment
            confidence_risk = 1.0 - pattern.confidence
            metrics['confidence_risk'] = confidence_risk
            
        except Exception as e:
            self.logger.warning(f"Error calculating risk metrics: {e}")
        
        return metrics
    
    def _generate_trading_recommendations(self, 
                                        pattern: Any,
                                        validation: ValidationResult,
                                        target_prices: Dict[str, float],
                                        stop_loss_levels: Dict[str, float],
                                        risk_metrics: Dict[str, float]) -> Dict[str, Any]:
        """Generate trading recommendations"""
        recommendations = {}
        
        try:
            # Entry recommendation
            if validation.quality_rating in [PatternQuality.EXCELLENT, PatternQuality.GOOD]:
                recommendations['entry_signal'] = 'STRONG_BUY' if validation.quality_score > 0.8 else 'BUY'
            elif validation.quality_rating == PatternQuality.AVERAGE:
                recommendations['entry_signal'] = 'HOLD'
            else:
                recommendations['entry_signal'] = 'AVOID'
            
            # Position sizing
            risk_reward = risk_metrics.get('risk_reward_ratio', 1.0)
            confidence = pattern.confidence
            
            if risk_reward > 2.5 and confidence > 0.8:
                recommendations['position_size'] = 'LARGE'
            elif risk_reward > 1.8 and confidence > 0.7:
                recommendations['position_size'] = 'MEDIUM'
            elif risk_reward > 1.2 and confidence > 0.6:
                recommendations['position_size'] = 'SMALL'
            else:
                recommendations['position_size'] = 'AVOID'
            
            # Time horizon
            expected_duration = pattern.expected_duration
            if expected_duration < 10:
                recommendations['time_horizon'] = 'SHORT_TERM'
            elif expected_duration < 30:
                recommendations['time_horizon'] = 'MEDIUM_TERM'
            else:
                recommendations['time_horizon'] = 'LONG_TERM'
            
            # Risk level
            volatility_risk = risk_metrics.get('volatility_risk', 0.02)
            if volatility_risk < 0.015:
                recommendations['risk_level'] = 'LOW'
            elif volatility_risk < 0.03:
                recommendations['risk_level'] = 'MEDIUM'
            else:
                recommendations['risk_level'] = 'HIGH'
            
            # Recommended targets and stops
            recommendations['targets'] = target_prices
            recommendations['stop_losses'] = stop_loss_levels
            
        except Exception as e:
            self.logger.warning(f"Error generating trading recommendations: {e}")
        
        return recommendations
    
    # Placeholder methods for advanced features
    def _assess_trend_context(self, market_data: pd.DataFrame, pattern: Any) -> float:
        """Assess trend context"""
        # Simplified trend assessment
        return 0.7
    
    def _analyze_market_phase(self, market_data: pd.DataFrame) -> float:
        """Analyze current market phase"""
        # Simplified market phase analysis
        return 0.6
    
    def _find_similar_historical_patterns(self, pattern: Any, symbol: str = None) -> List[Dict[str, Any]]:
        """Find similar historical patterns"""
        # Placeholder for historical pattern matching
        return []
    
    def _calculate_market_correlations(self, pattern: Any, market_data: pd.DataFrame, symbol: str = None) -> Dict[str, float]:
        """Calculate market correlations"""
        # Placeholder for market correlation analysis
        return {}
    
    def _assess_news_sentiment_impact(self, pattern: Any, symbol: str = None, current_time: datetime = None) -> float:
        """Assess news sentiment impact"""
        # Placeholder for news sentiment analysis
        return 0.0
    
    def _calculate_seasonality_factor(self, pattern: Any, current_time: datetime = None) -> float:
        """Calculate seasonality factor"""
        # Simplified seasonality - could be enhanced with historical analysis
        if current_time:
            month = current_time.month
            # Simple seasonal adjustment (this could be much more sophisticated)
            seasonal_factors = {
                1: 1.05,   # January effect
                2: 0.98,
                3: 1.02,
                4: 1.03,
                5: 0.97,   # Sell in May
                6: 0.95,
                7: 0.96,
                8: 0.94,
                9: 0.98,
                10: 1.01,
                11: 1.04,  # Holiday rally
                12: 1.06
            }
            return seasonal_factors.get(month, 1.0)
        
        return 1.0
    
    def _check_moving_average_alignment(self, pattern: Any, data: pd.DataFrame) -> float:
        """Check moving average alignment"""
        try:
            if 'close' not in data.columns or len(data) < 50:
                return 0.5
            
            closes = data['close'].values
            current_price = closes[-1]
            
            # Calculate different MA periods
            mas = {}
            for period in [5, 10, 20, 50]:
                if len(closes) >= period:
                    mas[period] = np.mean(closes[-period:])
            
            if len(mas) < 2:
                return 0.5
            
            # Check alignment
            alignment_score = 0.0
            comparisons = 0
            
            # Bullish alignment: shorter MAs above longer MAs
            ma_periods = sorted(mas.keys())
            for i in range(len(ma_periods) - 1):
                shorter_ma = mas[ma_periods[i]]
                longer_ma = mas[ma_periods[i + 1]]
                
                if shorter_ma > longer_ma:
                    alignment_score += 1
                comparisons += 1
            
            # Price relative to MAs
            above_mas = sum(1 for ma in mas.values() if current_price > ma)
            ma_score = above_mas / len(mas)
            
            # Combine scores
            if comparisons > 0:
                ma_alignment = alignment_score / comparisons
                final_score = (ma_alignment + ma_score) / 2
                return final_score
            
            return ma_score
            
        except Exception as e:
            self.logger.warning(f"Error checking MA alignment: {e}")
            return 0.5
    
    def export_enhanced_patterns(self, patterns: List[EnhancedPattern], output_path: str = None) -> str:
        """Export enhanced patterns to JSON file"""
        try:
            if not output_path:
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                output_path = f"enhanced_patterns_{timestamp}.json"
            
            export_data = []
            
            for pattern in patterns:
                pattern_dict = {
                    'pattern_type': pattern.original_pattern.pattern_type,
                    'confidence': pattern.original_pattern.confidence,
                    'quality_score': pattern.validation_result.quality_score,
                    'quality_rating': pattern.validation_result.quality_rating.value,
                    'phase': pattern.phase.value,
                    'target_prices': pattern.target_prices,
                    'stop_loss_levels': pattern.stop_loss_levels,
                    'risk_metrics': pattern.risk_metrics,
                    'trading_recommendations': pattern.trading_recommendations,
                    'timestamp': pattern.original_pattern.timestamp.isoformat() if hasattr(pattern.original_pattern, 'timestamp') else datetime.now().isoformat()
                }
                export_data.append(pattern_dict)
            
            with open(output_path, 'w') as f:
                json.dump(export_data, f, indent=2, default=str)
            
            self.logger.info(f"Exported {len(patterns)} enhanced patterns to {output_path}")
            return output_path
            
        except Exception as e:
            self.logger.error(f"Error exporting enhanced patterns: {e}")
            return ""
