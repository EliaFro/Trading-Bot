#!/usr/bin/env python3
"""
Advanced Type Definitions for AI-Powered Trading Pattern Discovery System
Comprehensive dataclasses and enums for pattern detection, validation, and trading
"""

from dataclasses import dataclass, field, asdict
from typing import Dict, List, Tuple, Optional, Any, Union, Callable, TypeVar, Generic
from datetime import datetime
from enum import Enum, auto
import numpy as np
import pandas as pd
from abc import ABC, abstractmethod


# Type Variables for Generic Types
T = TypeVar('T')
PriceType = Union[float, np.float32, np.float64]
VolumeType = Union[int, float, np.int64, np.float64]
ArrayType = Union[np.ndarray, pd.Series, List[float]]


# ==================== ENUMS ====================

class PatternType(Enum):
    """Comprehensive pattern types for detection"""
    # Classic Chart Patterns
    HEAD_SHOULDERS = "head_shoulders"
    INVERSE_HEAD_SHOULDERS = "inverse_head_shoulders"
    DOUBLE_TOP = "double_top"
    DOUBLE_BOTTOM = "double_bottom"
    TRIPLE_TOP = "triple_top"
    TRIPLE_BOTTOM = "triple_bottom"
    
    # Triangle Patterns
    TRIANGLE_ASCENDING = "triangle_ascending"
    TRIANGLE_DESCENDING = "triangle_descending"
    TRIANGLE_SYMMETRIC = "triangle_symmetric"
    
    # Wedge Patterns
    WEDGE_RISING = "wedge_rising"
    WEDGE_FALLING = "wedge_falling"
    
    # Continuation Patterns
    FLAG_BULLISH = "flag_bullish"
    FLAG_BEARISH = "flag_bearish"
    PENNANT_BULLISH = "pennant_bullish"
    PENNANT_BEARISH = "pennant_bearish"
    
    # Reversal Patterns
    CUP_HANDLE = "cup_handle"
    INVERSE_CUP_HANDLE = "inverse_cup_handle"
    ROUNDING_TOP = "rounding_top"
    ROUNDING_BOTTOM = "rounding_bottom"
    
    # Rectangle/Channel Patterns
    RECTANGLE = "rectangle"
    CHANNEL_UP = "channel_up"
    CHANNEL_DOWN = "channel_down"
    
    # Breakout/Breakdown
    BREAKOUT = "breakout"
    BREAKDOWN = "breakdown"
    FALSE_BREAKOUT = "false_breakout"
    
    # Consolidation
    CONSOLIDATION = "consolidation"
    RANGE_BOUND = "range_bound"
    
    # Advanced Patterns
    HARMONIC_ABCD = "harmonic_abcd"
    HARMONIC_GARTLEY = "harmonic_gartley"
    HARMONIC_BUTTERFLY = "harmonic_butterfly"
    HARMONIC_BAT = "harmonic_bat"
    HARMONIC_CRAB = "harmonic_crab"
    
    # AI-Discovered Patterns
    AI_PATTERN_1 = "ai_pattern_1"
    AI_PATTERN_2 = "ai_pattern_2"
    AI_PATTERN_3 = "ai_pattern_3"
    ANOMALY_PATTERN = "anomaly_pattern"


class PatternQuality(Enum):
    """Quality rating for patterns"""
    EXCELLENT = "excellent"
    GOOD = "good"
    AVERAGE = "average"
    POOR = "poor"
    INVALID = "invalid"


class PatternPhase(Enum):
    """Current phase of pattern development"""
    FORMATION = "formation"
    CONFIRMATION = "confirmation"
    BREAKOUT = "breakout"
    CONTINUATION = "continuation"
    COMPLETION = "completion"
    INVALIDATION = "invalidation"
    FAILED = "failed"


class MarketRegime(Enum):
    """Market regime classification"""
    TRENDING_UP = "trending_up"
    TRENDING_DOWN = "trending_down"
    RANGING = "ranging"
    VOLATILE = "volatile"
    QUIET = "quiet"
    BREAKOUT_MODE = "breakout_mode"
    RISK_OFF = "risk_off"
    RISK_ON = "risk_on"


class TradingSignal(Enum):
    """Trading signal strength"""
    STRONG_BUY = "strong_buy"
    BUY = "buy"
    WEAK_BUY = "weak_buy"
    HOLD = "hold"
    WEAK_SELL = "weak_sell"
    SELL = "sell"
    STRONG_SELL = "strong_sell"
    NO_SIGNAL = "no_signal"


class TimeFrame(Enum):
    """Trading timeframes"""
    TICK = "tick"
    ONE_MIN = "1m"
    FIVE_MIN = "5m"
    FIFTEEN_MIN = "15m"
    THIRTY_MIN = "30m"
    ONE_HOUR = "1h"
    FOUR_HOUR = "4h"
    ONE_DAY = "1d"
    ONE_WEEK = "1w"
    ONE_MONTH = "1M"


class RiskLevel(Enum):
    """Risk level classification"""
    VERY_LOW = "very_low"
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    VERY_HIGH = "very_high"
    EXTREME = "extreme"


class TrainingPhase(Enum):
    """Model training phases"""
    INITIALIZATION = "initialization"
    DATA_PREPARATION = "data_preparation"
    FEATURE_ENGINEERING = "feature_engineering"
    MODEL_SELECTION = "model_selection"
    HYPERPARAMETER_TUNING = "hyperparameter_tuning"
    TRAINING = "training"
    VALIDATION = "validation"
    ENSEMBLE_CREATION = "ensemble_creation"
    DEPLOYMENT = "deployment"
    MONITORING = "monitoring"
    RETRAINING = "retraining"


class LearningStrategy(Enum):
    """Machine learning strategies"""
    SUPERVISED = "supervised"
    SEMI_SUPERVISED = "semi_supervised"
    UNSUPERVISED = "unsupervised"
    REINFORCEMENT = "reinforcement"
    ACTIVE_LEARNING = "active_learning"
    TRANSFER_LEARNING = "transfer_learning"
    META_LEARNING = "meta_learning"
    CONTINUAL_LEARNING = "continual_learning"
    FEDERATED_LEARNING = "federated_learning"


class ModelType(Enum):
    """Model type classification"""
    RANDOM_FOREST = "random_forest"
    GRADIENT_BOOSTING = "gradient_boosting"
    NEURAL_NETWORK = "neural_network"
    LSTM = "lstm"
    CNN = "cnn"
    TRANSFORMER = "transformer"
    ENSEMBLE = "ensemble"
    SVM = "svm"
    LOGISTIC_REGRESSION = "logistic_regression"
    CUSTOM_AI = "custom_ai"


# ==================== CONFIGURATION DATACLASSES ====================

@dataclass
class SystemConfiguration:
    """Master configuration for the entire system"""
    # System settings
    system_name: str = "AI Trading Pattern Discovery System"
    version: str = "2.0.0"
    environment: str = "production"
    
    # Performance settings
    enable_gpu: bool = True
    max_workers: int = -1  # -1 for auto
    memory_limit_gb: float = 16.0
    
    # Data settings
    data_retention_days: int = 365
    cache_enabled: bool = True
    cache_size_gb: float = 10.0
    
    # Model settings
    enable_deep_learning: bool = True
    enable_ensemble: bool = True
    model_update_frequency_hours: int = 24
    
    # Risk management
    max_concurrent_positions: int = 10
    max_position_size_percent: float = 5.0
    max_daily_loss_percent: float = 2.0
    
    # Monitoring
    enable_real_time_monitoring: bool = True
    alert_threshold_performance_drop: float = 0.1
    log_level: str = "INFO"


@dataclass
class DataConfiguration:
    """Configuration for data handling"""
    # Data sources
    primary_data_source: str = "binance"
    backup_data_sources: List[str] = field(default_factory=lambda: ["coinbase", "kraken"])
    
    # Data parameters
    min_data_points: int = 1000
    max_data_points: int = 100000
    lookback_periods: int = 5000
    
    # Preprocessing
    handle_missing_data: str = "interpolate"  # "drop", "interpolate", "forward_fill"
    outlier_detection_method: str = "isolation_forest"
    outlier_threshold: float = 3.0
    
    # Feature engineering
    enable_technical_indicators: bool = True
    enable_custom_features: bool = True
    feature_scaling_method: str = "robust"  # "standard", "robust", "minmax"
    
    # Data quality
    min_data_quality_score: float = 0.8
    require_volume_data: bool = True
    require_bid_ask_data: bool = False


@dataclass
class TradingConfiguration:
    """Configuration for trading parameters"""
    # Trading settings
    enable_live_trading: bool = False
    enable_paper_trading: bool = True
    default_position_size: float = 1000.0
    
    # Order types
    use_market_orders: bool = False
    use_limit_orders: bool = True
    limit_order_offset_percent: float = 0.1
    
    # Risk management
    stop_loss_type: str = "trailing"  # "fixed", "trailing", "dynamic"
    default_stop_loss_percent: float = 2.0
    take_profit_multiplier: float = 2.5
    
    # Timing
    max_holding_period_hours: int = 168  # 1 week
    min_holding_period_minutes: int = 60
    
    # Filters
    min_volume_filter: float = 1000000.0  # USD
    min_pattern_confidence: float = 0.7
    min_risk_reward_ratio: float = 1.5


@dataclass
class TrainingConfiguration:
    """Configuration for model training"""
    # Learning settings
    learning_strategy: LearningStrategy = LearningStrategy.SUPERVISED
    enable_hyperparameter_tuning: bool = True
    enable_feature_selection: bool = True
    enable_ensemble: bool = True
    enable_deep_learning: bool = True
    
    # Validation settings
    cross_validation_folds: int = 5
    test_size: float = 0.2
    validation_size: float = 0.2
    time_series_split: bool = True
    
    # Training parameters
    random_state: int = 42
    n_jobs: int = -1
    max_training_time_hours: float = 24.0
    early_stopping_patience: int = 10
    
    # Hyperparameter search
    hyperparameter_search_iterations: int = 100
    hyperparameter_search_method: str = "optuna"  # "grid", "random", "optuna", "hyperopt"
    
    # Model settings
    ensemble_size: int = 5
    model_selection_metric: str = 'f1_weighted'
    retraining_threshold: float = 0.05
    
    # Advanced features
    enable_uncertainty_quantification: bool = True
    enable_explainability: bool = True
    enable_online_learning: bool = False


# ==================== PATTERN DETECTION DATACLASSES ====================

@dataclass
class PatternDetection:
    """Comprehensive pattern detection result"""
    # Basic information
    pattern_id: str
    pattern_type: PatternType
    symbol: str
    timeframe: TimeFrame
    
    # Detection metrics
    confidence: float
    probability_scores: Dict[PatternType, float]
    pattern_strength: float
    
    # Location
    start_index: int
    end_index: int
    start_time: datetime
    end_time: datetime
    
    # Pattern characteristics
    pattern_height: float
    pattern_width: int
    volume_profile: str  # "increasing", "decreasing", "stable", "irregular"
    
    # Model information
    detecting_models: List[ModelType]
    model_predictions: Dict[str, Any]
    features_used: List[str]
    feature_importance: Dict[str, float]
    
    # Market context
    market_regime: MarketRegime
    trend_direction: str  # "up", "down", "sideways"
    volatility_context: str  # "low", "medium", "high"
    
    # Risk metrics
    risk_reward_ratio: float
    expected_duration: int
    success_probability: float
    
    # Metadata
    detection_timestamp: datetime = field(default_factory=datetime.now)
    detection_version: str = "2.0.0"
    notes: str = ""
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization"""
        return asdict(self)


@dataclass
class PatternFeatures:
    """Container for all extracted pattern features"""
    # Feature arrays
    geometric_features: np.ndarray
    statistical_features: np.ndarray
    technical_features: np.ndarray
    volume_features: np.ndarray
    microstructure_features: np.ndarray
    fractal_features: np.ndarray
    sentiment_features: np.ndarray
    market_regime_features: np.ndarray
    
    # Combined features
    combined_features: np.ndarray
    normalized_features: np.ndarray
    engineered_features: np.ndarray
    
    # Feature metadata
    feature_names: List[str]
    feature_importance_scores: Dict[str, float]
    feature_statistics: Dict[str, Dict[str, float]]
    
    # Context
    window_size: int
    overlap_size: int
    timestamp: pd.Timestamp
    
    # Quality metrics
    feature_quality_score: float
    missing_features: List[str]
    feature_correlations: Optional[np.ndarray] = None


@dataclass
class ValidationResult:
    """Comprehensive pattern validation result"""
    # Validation outcome
    is_valid: bool
    validation_score: float
    quality_rating: PatternQuality
    confidence_adjustment: float
    
    # Detailed scores
    technical_score: float
    volume_score: float
    timeframe_score: float
    market_context_score: float
    statistical_significance: float
    
    # Validation details
    validation_reasons: List[str]
    validation_warnings: List[str]
    failed_checks: List[str]
    
    # Recommendations
    recommended_action: TradingSignal
    suggested_improvements: List[str]
    
    # Metadata
    validation_timestamp: datetime = field(default_factory=datetime.now)
    validator_version: str = "2.0.0"


@dataclass
class EnhancedPattern:
    """Pattern enhanced with comprehensive analytics and predictions"""
    # Core pattern
    original_pattern: PatternDetection
    validation_result: ValidationResult
    
    # Pattern state
    phase: PatternPhase
    phase_completion_percent: float
    time_to_completion: Optional[int]
    
    # Price predictions
    target_prices: Dict[str, PriceType]
    stop_loss_levels: Dict[str, PriceType]
    entry_zones: List[Tuple[PriceType, PriceType]]
    
    # Support/Resistance
    support_levels: List[PriceType]
    resistance_levels: List[PriceType]
    key_levels: Dict[str, PriceType]
    
    # Volume analysis
    volume_analysis: Dict[str, Any]
    volume_patterns: List[str]
    unusual_volume_detected: bool
    
    # Technical indicators
    momentum_indicators: Dict[str, float]
    trend_indicators: Dict[str, float]
    volatility_indicators: Dict[str, float]
    
    # Risk analysis
    risk_metrics: Dict[str, float]
    risk_level: RiskLevel
    max_drawdown_expected: float
    var_95: float  # Value at Risk 95%
    
    # Trading recommendations
    trading_signal: TradingSignal
    position_size_recommendation: float
    entry_strategy: Dict[str, Any]
    exit_strategy: Dict[str, Any]
    
    # Historical analysis
    similar_historical_patterns: List[Dict[str, Any]]
    historical_success_rate: float
    average_return: float
    
    # Market context
    market_correlations: Dict[str, float]
    sector_performance: Dict[str, float]
    news_sentiment_score: float
    social_sentiment_score: float
    
    # AI insights
    ai_interpretation: str
    ai_confidence_explanation: str
    pattern_evolution_prediction: List[Dict[str, Any]]
    
    # Scoring
    composite_score: float = 0.0
    profitability_score: float = 0.0
    reliability_score: float = 0.0
    
    # Metadata
    enhancement_timestamp: datetime = field(default_factory=datetime.now)
    last_update: datetime = field(default_factory=datetime.now)


# ==================== MODEL TRAINING DATACLASSES ====================

@dataclass
class ModelMetrics:
    """Comprehensive model performance metrics"""
    # Basic metrics
    accuracy: float
    precision: float
    recall: float
    f1_score: float
    
    # Advanced metrics
    auc_score: float
    log_loss: float
    matthews_corrcoef: float
    cohen_kappa: float
    
    # Class-specific metrics
    per_class_precision: Dict[str, float]
    per_class_recall: Dict[str, float]
    per_class_f1: Dict[str, float]
    
    # Arrays
    confusion_matrix: np.ndarray
    precision_recall_curve: Optional[Tuple[np.ndarray, np.ndarray, np.ndarray]] = None
    roc_curve: Optional[Tuple[np.ndarray, np.ndarray, np.ndarray]] = None
    
    # Feature analysis
    feature_importance: Dict[str, float]
    feature_selection_mask: Optional[np.ndarray] = None
    
    # Performance metrics
    training_time: float
    inference_time: float
    model_size_mb: float
    memory_usage_mb: float
    
    # Complexity metrics
    model_complexity: int
    number_of_parameters: int
    
    # Uncertainty metrics
    uncertainty_metrics: Dict[str, float] = field(default_factory=dict)
    calibration_error: Optional[float] = None
    
    # Validation metrics
    cross_validation_scores: List[float] = field(default_factory=list)
    out_of_sample_performance: Optional[float] = None
    
    # Learning curves
    learning_curve_data: Dict[str, List[float]] = field(default_factory=dict)
    validation_curve_data: Dict[str, List[float]] = field(default_factory=dict)


@dataclass
class ModelCard:
    """Model documentation and metadata"""
    # Identification
    model_id: str
    model_name: str
    model_type: ModelType
    version: str
    
    # Description
    description: str
    use_case: str
    limitations: List[str]
    ethical_considerations: List[str]
    
    # Training details
    training_data_description: str
    training_algorithm: str
    hyperparameters: Dict[str, Any]
    
    # Performance summary
    performance_summary: Dict[str, float]
    benchmark_comparison: Dict[str, float]
    
    # Deployment info
    deployment_status: str
    api_endpoint: Optional[str]
    update_frequency: str
    
    # Metadata
    created_by: str
    created_date: datetime
    last_updated: datetime
    tags: List[str]


@dataclass
class TrainingResult:
    """Comprehensive training session result"""
    # Identification
    training_id: str
    model_id: str
    model_type: ModelType
    
    # Training details
    training_phase: TrainingPhase
    training_configuration: TrainingConfiguration
    training_duration_hours: float
    
    # Performance
    metrics: ModelMetrics
    model_card: ModelCard
    
    # Data information
    training_data_hash: str
    training_samples: int
    feature_count: int
    feature_names: List[str]
    label_distribution: Dict[str, int]
    
    # Model artifacts
    model_path: str
    preprocessor_path: Optional[str]
    feature_selector_path: Optional[str]
    
    # Validation
    validation_strategy: str
    validation_results: Dict[str, Any]
    
    # Explainability
    feature_explanations: Optional[Dict[str, Any]] = None
    model_explanations: Optional[Dict[str, Any]] = None
    
    # Timestamps
    start_time: datetime
    end_time: datetime
    
    # Status
    status: str  # "completed", "failed", "partial"
    errors: List[str] = field(default_factory=list)
    warnings: List[str] = field(default_factory=list)


# ==================== TRADING DATACLASSES ====================

@dataclass
class TradingPosition:
    """Active trading position"""
    # Identification
    position_id: str
    pattern_id: str
    symbol: str
    
    # Position details
    side: str  # "long" or "short"
    entry_price: PriceType
    current_price: PriceType
    quantity: float
    
    # Timing
    entry_time: datetime
    expected_holding_time: int
    max_holding_time: int
    
    # Risk management
    stop_loss: PriceType
    take_profit: PriceType
    trailing_stop_distance: Optional[float]
    
    # Performance
    unrealized_pnl: float
    unrealized_pnl_percent: float
    max_profit: float
    max_drawdown: float
    
    # Pattern linkage
    pattern_confidence: float
    pattern_phase_at_entry: PatternPhase
    
    # Status
    status: str  # "open", "closing", "closed"
    close_reason: Optional[str] = None
    
    def calculate_risk_reward(self) -> float:
        """Calculate current risk-reward ratio"""
        risk = abs(self.entry_price - self.stop_loss)
        reward = abs(self.take_profit - self.entry_price)
        return reward / risk if risk > 0 else 0


@dataclass
class BacktestResult:
    """Comprehensive backtest result"""
    # Identification
    backtest_id: str
    strategy_name: str
    
    # Period
    start_date: datetime
    end_date: datetime
    total_days: int
    
    # Performance metrics
    total_return: float
    annualized_return: float
    sharpe_ratio: float
    sortino_ratio: float
    calmar_ratio: float
    
    # Risk metrics
    max_drawdown: float
    max_drawdown_duration_days: int
    var_95: float
    cvar_95: float
    
    # Trading statistics
    total_trades: int
    winning_trades: int
    losing_trades: int
    win_rate: float
    profit_factor: float
    
    # Trade analysis
    average_win: float
    average_loss: float
    largest_win: float
    largest_loss: float
    
    # Pattern analysis
    pattern_performance: Dict[PatternType, Dict[str, float]]
    best_performing_patterns: List[Tuple[PatternType, float]]
    worst_performing_patterns: List[Tuple[PatternType, float]]
    
    # Time analysis
    trades_by_hour: Dict[int, int]
    trades_by_day: Dict[str, int]
    trades_by_month: Dict[int, int]
    
    # Detailed results
    equity_curve: List[float]
    drawdown_curve: List[float]
    trade_log: List[Dict[str, Any]]


# ==================== MARKET DATA DATACLASSES ====================

@dataclass
class MarketData:
    """Enhanced market data container"""
    # Basic OHLCV
    symbol: str
    timeframe: TimeFrame
    timestamps: np.ndarray
    open: np.ndarray
    high: np.ndarray
    low: np.ndarray
    close: np.ndarray
    volume: np.ndarray
    
    # Additional data
    bid: Optional[np.ndarray] = None
    ask: Optional[np.ndarray] = None
    trades_count: Optional[np.ndarray] = None
    vwap: Optional[np.ndarray] = None
    
    # Technical indicators (computed)
    indicators: Dict[str, np.ndarray] = field(default_factory=dict)
    
    # Metadata
    source: str = ""
    quality_score: float = 1.0
    gaps_filled: int = 0
    outliers_removed: int = 0
    
    def get_dataframe(self) -> pd.DataFrame:
        """Convert to pandas DataFrame"""
        df = pd.DataFrame({
            'open': self.open,
            'high': self.high,
            'low': self.low,
            'close': self.close,
            'volume': self.volume
        }, index=pd.DatetimeIndex(self.timestamps))
        
        # Add indicators
        for name, values in self.indicators.items():
            df[name] = values
            
        return df


@dataclass
class MarketContext:
    """Current market context and conditions"""
    # Market state
    timestamp: datetime
    market_regime: MarketRegime
    volatility_regime: str  # "low", "normal", "high", "extreme"
    trend_strength: float
    
    # Key levels
    daily_high: PriceType
    daily_low: PriceType
    weekly_high: PriceType
    weekly_low: PriceType
    
    # Market internals
    breadth: float  # advance/decline ratio
    volume_ratio: float  # up volume / down volume
    put_call_ratio: Optional[float] = None
    
    # Sentiment
    fear_greed_index: Optional[float] = None
    social_sentiment: Optional[float] = None
    news_sentiment: Optional[float] = None
    
    # Correlations
    correlation_matrix: Optional[np.ndarray] = None
    beta_to_market: Optional[float] = None
    
    # Risk indicators
    vix_level: Optional[float] = None
    term_structure: Optional[Dict[str, float]] = None


# ==================== ALERT AND NOTIFICATION DATACLASSES ====================

@dataclass
class Alert:
    """Trading alert"""
    alert_id: str
    alert_type: str  # "pattern_detected", "position_update", "risk_warning", etc.
    severity: str  # "info", "warning", "critical"
    
    title: str
    message: str
    
    # Context
    symbol: Optional[str] = None
    pattern_id: Optional[str] = None
    position_id: Optional[str] = None
    
    # Actions
    recommended_action: Optional[str] = None
    auto_execute: bool = False
    
    # Timing
    created_at: datetime = field(default_factory=datetime.now)
    expires_at: Optional[datetime] = None
    
    # Status
    acknowledged: bool = False
    acknowledged_by: Optional[str] = None
    acknowledged_at: Optional[datetime] = None


# ==================== PERFORMANCE TRACKING DATACLASSES ====================

@dataclass
class SystemPerformance:
    """System-wide performance metrics"""
    # System health
    timestamp: datetime
    uptime_hours: float
    cpu_usage_percent: float
    memory_usage_percent: float
    gpu_usage_percent: Optional[float] = None
    
    # Processing metrics
    patterns_detected_24h: int
    patterns_validated_24h: int
    models_trained_24h: int
    predictions_made_24h: int
    
    # Latency metrics
    avg_detection_latency_ms: float
    avg_validation_latency_ms: float
    avg_prediction_latency_ms: float
    
    # Accuracy metrics
    detection_accuracy_24h: float
    validation_accuracy_24h: float
    prediction_accuracy_24h: float
    
    # Trading performance
    active_positions: int
    positions_opened_24h: int
    positions_closed_24h: int
    win_rate_24h: float
    pnl_24h: float
    
    # Error tracking
    errors_24h: int
    warnings_24h: int
    failed_detections_24h: int
    
    # Model performance
    model_drift_detected: bool
    models_requiring_retraining: List[str]
    
    # Alerts
    active_alerts: int
    critical_alerts: int


# ==================== ABSTRACT BASE CLASSES ====================

class PatternDetectorBase(ABC):
    """Abstract base class for pattern detectors"""
    
    @abstractmethod
    def detect(self, market_data: MarketData) -> List[PatternDetection]:
        """Detect patterns in market data"""
        pass
    
    @abstractmethod
    def get_supported_patterns(self) -> List[PatternType]:
        """Get list of supported pattern types"""
        pass


class FeatureExtractorBase(ABC):
    """Abstract base class for feature extractors"""
    
    @abstractmethod
    def extract(self, market_data: MarketData, window_size: int) -> PatternFeatures:
        """Extract features from market data"""
        pass
    
    @abstractmethod
    def get_feature_names(self) -> List[str]:
        """Get list of feature names"""
        pass


class ModelTrainerBase(ABC):
    """Abstract base class for model trainers"""
    
    @abstractmethod
    def train(self, features: np.ndarray, labels: np.ndarray) -> TrainingResult:
        """Train model with features and labels"""
        pass
    
    @abstractmethod
    def predict(self, features: np.ndarray) -> np.ndarray:
        """Make predictions with trained model"""
        pass


# ==================== UTILITY FUNCTIONS ====================

def create_pattern_id(pattern_type: PatternType, symbol: str, timestamp: datetime) -> str:
    """Create unique pattern ID"""
    timestamp_str = timestamp.strftime("%Y%m%d_%H%M%S")
    return f"{pattern_type.value}_{symbol}_{timestamp_str}"


def calculate_position_size(
    capital: float,
    risk_percent: float,
    entry_price: float,
    stop_loss: float
) -> float:
    """Calculate position size based on risk management"""
    risk_amount = capital * (risk_percent / 100)
    price_risk = abs(entry_price - stop_loss)
    
    if price_risk > 0:
        position_size = risk_amount / price_risk
        return position_size
    return 0.0


def merge_patterns(patterns: List[PatternDetection], overlap_threshold: float = 0.5) -> List[PatternDetection]:
    """Merge overlapping patterns"""
    if not patterns:
        return patterns
    
    # Sort by start time
    sorted_patterns = sorted(patterns, key=lambda p: p.start_time)
    merged = []
    
    for pattern in sorted_patterns:
        if not merged:
            merged.append(pattern)
            continue
            
        last_pattern = merged[-1]
        
        # Check for overlap
        overlap_start = max(last_pattern.start_index, pattern.start_index)
        overlap_end = min(last_pattern.end_index, pattern.end_index)
        overlap_size = max(0, overlap_end - overlap_start)
        
        pattern_size = pattern.end_index - pattern.start_index
        overlap_ratio = overlap_size / pattern_size if pattern_size > 0 else 0
        
        if overlap_ratio < overlap_threshold:
            merged.append(pattern)
        else:
            # Update the last pattern if this one has higher confidence
            if pattern.confidence > last_pattern.confidence:
                merged[-1] = pattern
    
    return merged


def validate_market_data(data: MarketData) -> Tuple[bool, List[str]]:
    """Validate market data integrity"""
    errors = []
    
    # Check data length consistency
    lengths = [
        len(data.timestamps),
        len(data.open),
        len(data.high),
        len(data.low),
        len(data.close),
        len(data.volume)
    ]
    
    if len(set(lengths)) > 1:
        errors.append("Inconsistent data lengths")
    
    # Check OHLC relationships
    for i in range(len(data.close)):
        if data.high[i] < data.low[i]:
            errors.append(f"High < Low at index {i}")
        if data.high[i] < data.open[i] or data.high[i] < data.close[i]:
            errors.append(f"High price violation at index {i}")
        if data.low[i] > data.open[i] or data.low[i] > data.close[i]:
            errors.append(f"Low price violation at index {i}")
    
    # Check for negative prices
    if np.any(data.open < 0) or np.any(data.close < 0):
        errors.append("Negative prices detected")
    
    # Check for zero or negative volume
    if np.any(data.volume <= 0):
        errors.append("Zero or negative volume detected")
    
    # Check timestamp order
    if len(data.timestamps) > 1:
        time_diffs = np.diff(data.timestamps.astype('int64'))
        if np.any(time_diffs <= 0):
            errors.append("Timestamps not in ascending order")
    
    is_valid = len(errors) == 0
    return is_valid, errors


# ==================== TYPE ALIASES ====================

PatternList = List[PatternDetection]
EnhancedPatternList = List[EnhancedPattern]
PriceArray = np.ndarray
VolumeArray = np.ndarray
FeatureMatrix = np.ndarray
PredictionArray = np.ndarray


# ==================== CONSTANTS ====================

# Pattern detection thresholds
MIN_PATTERN_CONFIDENCE = 0.6
MIN_PATTERN_QUALITY = 0.5
MIN_PATTERN_POINTS = 5

# Risk management constants
MAX_POSITION_SIZE_PERCENT = 10.0
MAX_PORTFOLIO_RISK_PERCENT = 2.0
DEFAULT_STOP_LOSS_PERCENT = 2.0

# Model performance thresholds
MIN_MODEL_ACCURACY = 0.65
MIN_MODEL_F1_SCORE = 0.60
MAX_MODEL_DRIFT_THRESHOLD = 0.1

# Time constants
SECONDS_IN_MINUTE = 60
MINUTES_IN_HOUR = 60
HOURS_IN_DAY = 24
DAYS_IN_WEEK = 7
DAYS_IN_MONTH = 30
DAYS_IN_YEAR = 365


# ==================== EXPORT ALL ====================

__all__ = [
    # Enums
    'PatternType', 'PatternQuality', 'PatternPhase', 'MarketRegime',
    'TradingSignal', 'TimeFrame', 'RiskLevel', 'TrainingPhase',
    'LearningStrategy', 'ModelType',
    
    # Configuration classes
    'SystemConfiguration', 'DataConfiguration', 'TradingConfiguration',
    'TrainingConfiguration',
    
    # Pattern detection classes
    'PatternDetection', 'PatternFeatures', 'ValidationResult',
    'EnhancedPattern',
    
    # Model training classes
    'ModelMetrics', 'ModelCard', 'TrainingResult',
    
    # Trading classes
    'TradingPosition', 'BacktestResult',
    
    # Market data classes
    'MarketData', 'MarketContext',
    
    # Alert and performance classes
    'Alert', 'SystemPerformance',
    
    # Abstract base classes
    'PatternDetectorBase', 'FeatureExtractorBase', 'ModelTrainerBase',
    
    # Utility functions
    'create_pattern_id', 'calculate_position_size', 'merge_patterns',
    'validate_market_data',
    
    # Type aliases
    'PatternList', 'EnhancedPatternList', 'PriceArray', 'VolumeArray',
    'FeatureMatrix', 'PredictionArray',
    
    # Constants
    'MIN_PATTERN_CONFIDENCE', 'MIN_PATTERN_QUALITY', 'MIN_PATTERN_POINTS',
    'MAX_POSITION_SIZE_PERCENT', 'MAX_PORTFOLIO_RISK_PERCENT',
    'DEFAULT_STOP_LOSS_PERCENT', 'MIN_MODEL_ACCURACY', 'MIN_MODEL_F1_SCORE',
    'MAX_MODEL_DRIFT_THRESHOLD'
]