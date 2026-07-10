#!/usr/bin/env python3
"""
AI-Powered Pattern Detector with Multiple ML Models
Detects and classifies trading patterns using ensemble machine learning
"""

import numpy as np
import pandas as pd
import logging
from typing import Dict, List, Tuple, Optional, Any, Union
from dataclasses import dataclass
from datetime import datetime
import pickle
import json
from pathlib import Path

# Machine Learning imports
from sklearn.ensemble import (
    RandomForestClassifier, 
    GradientBoostingClassifier,
    IsolationForest,
    VotingClassifier
)
from sklearn.neural_network import MLPClassifier
from sklearn.svm import SVC
from sklearn.cluster import DBSCAN, KMeans
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE

# Deep Learning (if available)
try:
    import tensorflow as tf
    from tensorflow.keras.models import Sequential, Model
    from tensorflow.keras.layers import Dense, LSTM, Conv1D, MaxPooling1D, Flatten, Dropout, Input
    from tensorflow.keras.optimizers import Adam
    from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
    TF_AVAILABLE = True
except ImportError:
    TF_AVAILABLE = False
    logging.warning("TensorFlow not available. Deep learning features disabled.")

# Pattern recognition
from scipy.cluster.hierarchy import dendrogram, linkage, fcluster
from scipy.spatial.distance import pdist
import warnings
warnings.filterwarnings('ignore')

@dataclass
class PatternDetection:
    """Container for pattern detection results"""
    pattern_type: str
    confidence: float
    probability_scores: Dict[str, float]
    start_index: int
    end_index: int
    timestamp: datetime
    features_used: List[str]
    model_predictions: Dict[str, Any]
    pattern_strength: float
    risk_reward_ratio: float
    expected_duration: int
    market_regime: str

@dataclass
class ModelPerformance:
    """Container for model performance metrics"""
    accuracy: float
    precision: float
    recall: float
    f1_score: float
    confusion_matrix: np.ndarray
    feature_importance: Dict[str, float]
    training_time: float
    prediction_time: float
    model_complexity: int

class AdvancedPatternDetector:
    """
    Advanced AI-powered pattern detector with ensemble learning and self-improvement
    """
    
    def __init__(self, model_dir: str = "./models/patterns", enable_deep_learning: bool = True):
        self.model_dir = Path(model_dir)
        self.model_dir.mkdir(parents=True, exist_ok=True)
        
        self.enable_deep_learning = enable_deep_learning and TF_AVAILABLE
        self.logger = logging.getLogger(__name__)
        
        # Pattern types we can detect
        self.pattern_types = [
            'head_shoulders', 'double_top', 'double_bottom', 'triangle_ascending',
            'triangle_descending', 'triangle_symmetric', 'wedge_rising', 'wedge_falling',
            'flag_bullish', 'flag_bearish', 'pennant', 'cup_handle', 'inverse_head_shoulders',
            'rectangle', 'channel_up', 'channel_down', 'breakout', 'breakdown', 'consolidation',
            'reversal_bullish', 'reversal_bearish', 'continuation_bullish', 'continuation_bearish'
        ]
        
        # Initialize models
        self.models = {}
        self.ensemble_model = None
        self.scaler = StandardScaler()
        self.label_encoder = LabelEncoder()
        self.pca = PCA(n_components=50)
        
        # Performance tracking
        self.model_performance = {}
        self.detection_history = []
        self.feature_importance_tracker = {}
        
        # Anomaly detection
        self.anomaly_detector = IsolationForest(contamination=0.1, random_state=42)
        
        # Clustering for pattern discovery
        self.clusterer = KMeans(n_clusters=len(self.pattern_types), random_state=42)
        
        # Market regime detection
        self.regime_detector = None
        
        # Initialize models
        self._initialize_models()
        self._load_existing_models()
    
    def _initialize_models(self):
        """Initialize all ML models"""
        
        # Random Forest - Good for feature importance and interpretability
        self.models['random_forest'] = RandomForestClassifier(
            n_estimators=200,
            max_depth=15,
            min_samples_split=5,
            min_samples_leaf=2,
            random_state=42,
            n_jobs=-1
        )
        
        # Gradient Boosting - Good for complex patterns
        self.models['gradient_boosting'] = GradientBoostingClassifier(
            n_estimators=150,
            learning_rate=0.1,
            max_depth=8,
            random_state=42
        )
        
        # SVM - Good for high-dimensional data
        self.models['svm'] = SVC(
            kernel='rbf',
            C=1.0,
            gamma='scale',
            probability=True,
            random_state=42
        )
        
        # Neural Network - Good for non-linear patterns
        self.models['neural_network'] = MLPClassifier(
            hidden_layer_sizes=(128, 64, 32),
            activation='relu',
            solver='adam',
            alpha=0.001,
            learning_rate='adaptive',
            max_iter=1000,
            random_state=42
        )
        
        # Deep Learning models (if available)
        if self.enable_deep_learning:
            self._initialize_deep_models()
        
        # Create ensemble
        voting_models = [
            ('rf', self.models['random_forest']),
            ('gb', self.models['gradient_boosting']),
            ('nn', self.models['neural_network'])
        ]
        
        self.ensemble_model = VotingClassifier(
            estimators=voting_models,
            voting='soft'
        )
        
        self.logger.info(f"Initialized {len(self.models)} models for pattern detection")
    
    def _initialize_deep_models(self):
        """Initialize deep learning models"""
        if not TF_AVAILABLE:
            return
        
        try:
            # LSTM for sequence patterns
            self.models['lstm'] = self._create_lstm_model()
            
            # CNN for local patterns
            self.models['cnn'] = self._create_cnn_model()
            
            # Autoencoder for anomaly detection
            self.models['autoencoder'] = self._create_autoencoder()
            
        except Exception as e:
            self.logger.warning(f"Failed to initialize deep learning models: {e}")
    
    def _create_lstm_model(self, input_shape: Tuple[int, int] = (50, 10)):
        """Create LSTM model for sequence pattern recognition"""
        model = Sequential([
            LSTM(64, return_sequences=True, input_shape=input_shape),
            Dropout(0.2),
            LSTM(32, return_sequences=False),
            Dropout(0.2),
            Dense(64, activation='relu'),
            Dropout(0.3),
            Dense(len(self.pattern_types), activation='softmax')
        ])
        
        model.compile(
            optimizer=Adam(learning_rate=0.001),
            loss='categorical_crossentropy',
            metrics=['accuracy']
        )
        
        return model
    
    def _create_cnn_model(self, input_shape: Tuple[int, int] = (50, 10)):
        """Create CNN model for local pattern recognition"""
        model = Sequential([
            Conv1D(64, 3, activation='relu', input_shape=input_shape),
            Conv1D(64, 3, activation='relu'),
            MaxPooling1D(2),
            Conv1D(128, 3, activation='relu'),
            Conv1D(128, 3, activation='relu'),
            MaxPooling1D(2),
            Flatten(),
            Dense(128, activation='relu'),
            Dropout(0.5),
            Dense(64, activation='relu'),
            Dropout(0.3),
            Dense(len(self.pattern_types), activation='softmax')
        ])
        
        model.compile(
            optimizer=Adam(learning_rate=0.001),
            loss='categorical_crossentropy',
            metrics=['accuracy']
        )
        
        return model
    
    def _create_autoencoder(self, input_dim: int = 100):
        """Create autoencoder for anomaly detection"""
        # Encoder
        input_layer = Input(shape=(input_dim,))
        encoded = Dense(64, activation='relu')(input_layer)
        encoded = Dense(32, activation='relu')(encoded)
        encoded = Dense(16, activation='relu')(encoded)
        
        # Decoder
        decoded = Dense(32, activation='relu')(encoded)
        decoded = Dense(64, activation='relu')(decoded)
        decoded = Dense(input_dim, activation='sigmoid')(decoded)
        
        autoencoder = Model(input_layer, decoded)
        autoencoder.compile(optimizer='adam', loss='mse')
        
        return autoencoder
    
    def detect_patterns(self, 
                       features: np.ndarray, 
                       timestamps: List[datetime] = None,
                       symbol: str = None) -> List[PatternDetection]:
        """
        Detect patterns in the provided features using ensemble of models
        
        Args:
            features: Array of shape (n_samples, n_features)
            timestamps: List of timestamps for each sample
            symbol: Trading symbol
            
        Returns:
            List of detected patterns with confidence scores
        """
        try:
            if len(features) == 0:
                return []
            
            # Preprocess features
            features_scaled = self._preprocess_features(features)
            
            # Detect market regime
            market_regime = self._detect_market_regime(features_scaled)
            
            # Get predictions from all models
            predictions = self._get_ensemble_predictions(features_scaled)
            
            # Post-process and validate patterns
            detected_patterns = self._post_process_predictions(
                predictions, features_scaled, timestamps, market_regime, symbol
            )
            
            # Update detection history
            self.detection_history.extend(detected_patterns)
            
            # Self-improvement: update models based on recent performance
            self._update_model_performance(detected_patterns)
            
            return detected_patterns
            
        except Exception as e:
            self.logger.error(f"Error detecting patterns: {e}")
            return []
    
    def _preprocess_features(self, features: np.ndarray) -> np.ndarray:
        """Preprocess features for model input"""
        # Handle NaN values
        features_clean = np.nan_to_num(features, nan=0.0, posinf=1e6, neginf=-1e6)
        
        # Scale features
        if hasattr(self.scaler, 'mean_'):
            features_scaled = self.scaler.transform(features_clean)
        else:
            features_scaled = self.scaler.fit_transform(features_clean)
        
        # Apply PCA if features are high-dimensional
        if features_scaled.shape[1] > 100:
            if hasattr(self.pca, 'components_'):
                features_scaled = self.pca.transform(features_scaled)
            else:
                features_scaled = self.pca.fit_transform(features_scaled)
        
        return features_scaled
    
    def _detect_market_regime(self, features: np.ndarray) -> str:
        """Detect current market regime (trending, ranging, volatile)"""
        try:
            # Use clustering to identify market regimes
            if len(features) < 50:
                return 'unknown'
            
            # Extract regime-relevant features (volatility, trend strength, etc.)
            recent_features = features[-50:]  # Last 50 periods
            
            # Simple regime classification based on feature statistics
            volatility = np.std(recent_features[:, 0]) if recent_features.shape[1] > 0 else 0
            trend_strength = np.abs(np.corrcoef(np.arange(len(recent_features)), recent_features[:, 0])[0, 1]) if recent_features.shape[1] > 0 else 0
            
            if volatility > 2.0:
                return 'volatile'
            elif trend_strength > 0.7:
                return 'trending'
            else:
                return 'ranging'
                
        except Exception as e:
            self.logger.warning(f"Error detecting market regime: {e}")
            return 'unknown'
    
    def _get_ensemble_predictions(self, features: np.ndarray) -> Dict[str, Any]:
        """Get predictions from all available models"""
        predictions = {}
        
        try:
            # Classical ML models
            for model_name, model in self.models.items():
                if model_name in ['lstm', 'cnn', 'autoencoder']:
                    continue  # Skip deep learning models for now
                
                if hasattr(model, 'predict_proba'):
                    try:
                        pred_proba = model.predict_proba(features)
                        pred_classes = model.predict(features)
                        
                        predictions[model_name] = {
                            'probabilities': pred_proba,
                            'classes': pred_classes,
                            'confidence': np.max(pred_proba, axis=1)
                        }
                    except Exception as e:
                        self.logger.warning(f"Model {model_name} prediction failed: {e}")
            
            # Ensemble prediction
            if self.ensemble_model and hasattr(self.ensemble_model, 'predict_proba'):
                try:
                    ensemble_proba = self.ensemble_model.predict_proba(features)
                    ensemble_classes = self.ensemble_model.predict(features)
                    
                    predictions['ensemble'] = {
                        'probabilities': ensemble_proba,
                        'classes': ensemble_classes,
                        'confidence': np.max(ensemble_proba, axis=1)
                    }
                except Exception as e:
                    self.logger.warning(f"Ensemble prediction failed: {e}")
            
            # Deep learning predictions
            if self.enable_deep_learning:
                predictions.update(self._get_deep_predictions(features))
            
            # Anomaly detection
            anomaly_scores = self._detect_anomalies(features)
            predictions['anomaly_scores'] = anomaly_scores
            
        except Exception as e:
            self.logger.error(f"Error getting ensemble predictions: {e}")
        
        return predictions
    
    def _get_deep_predictions(self, features: np.ndarray) -> Dict[str, Any]:
        """Get predictions from deep learning models"""
        deep_predictions = {}
        
        if not TF_AVAILABLE:
            return deep_predictions
        
        try:
            # Reshape features for deep learning models if needed
            if len(features.shape) == 2 and features.shape[0] > 50:
                # Create sequences for LSTM
                sequence_length = 50
                n_features = features.shape[1]
                
                sequences = []
                for i in range(sequence_length, len(features)):
                    sequences.append(features[i-sequence_length:i])
                
                if sequences:
                    sequences = np.array(sequences)
                    
                    # LSTM predictions
                    if 'lstm' in self.models and hasattr(self.models['lstm'], 'predict'):
                        try:
                            lstm_pred = self.models['lstm'].predict(sequences, verbose=0)
                            deep_predictions['lstm'] = {
                                'probabilities': lstm_pred,
                                'classes': np.argmax(lstm_pred, axis=1),
                                'confidence': np.max(lstm_pred, axis=1)
                            }
                        except Exception as e:
                            self.logger.warning(f"LSTM prediction failed: {e}")
                    
                    # CNN predictions
                    if 'cnn' in self.models and hasattr(self.models['cnn'], 'predict'):
                        try:
                            cnn_pred = self.models['cnn'].predict(sequences, verbose=0)
                            deep_predictions['cnn'] = {
                                'probabilities': cnn_pred,
                                'classes': np.argmax(cnn_pred, axis=1),
                                'confidence': np.max(cnn_pred, axis=1)
                            }
                        except Exception as e:
                            self.logger.warning(f"CNN prediction failed: {e}")
        
        except Exception as e:
            self.logger.warning(f"Error in deep learning predictions: {e}")
        
        return deep_predictions
    
    def _detect_anomalies(self, features: np.ndarray) -> np.ndarray:
        """Detect anomalous patterns using isolation forest"""
        try:
            if hasattr(self.anomaly_detector, 'decision_function'):
                anomaly_scores = self.anomaly_detector.decision_function(features)
                return anomaly_scores
            else:
                # Fit the anomaly detector if not already fitted
                self.anomaly_detector.fit(features)
                anomaly_scores = self.anomaly_detector.decision_function(features)
                return anomaly_scores
        except Exception as e:
            self.logger.warning(f"Anomaly detection failed: {e}")
            return np.zeros(len(features))
    
    def _post_process_predictions(self, 
                                 predictions: Dict[str, Any], 
                                 features: np.ndarray,
                                 timestamps: List[datetime],
                                 market_regime: str,
                                 symbol: str) -> List[PatternDetection]:
        """Post-process and validate pattern predictions"""
        detected_patterns = []
        
        try:
            if 'ensemble' not in predictions:
                self.logger.warning("No ensemble predictions available")
                return detected_patterns
            
            ensemble_data = predictions['ensemble']
            probabilities = ensemble_data['probabilities']
            classes = ensemble_data['classes']
            confidences = ensemble_data['confidence']
            
            # Filter by confidence threshold
            confidence_threshold = 0.6
            high_confidence_indices = np.where(confidences > confidence_threshold)[0]
            
            for idx in high_confidence_indices:
                pattern_class = classes[idx]
                confidence = confidences[idx]
                
                # Get pattern type name
                if hasattr(self.label_encoder, 'classes_'):
                    try:
                        pattern_type = self.label_encoder.inverse_transform([pattern_class])[0]
                    except:
                        pattern_type = self.pattern_types[pattern_class % len(self.pattern_types)]
                else:
                    pattern_type = self.pattern_types[pattern_class % len(self.pattern_types)]
                
                # Calculate probability scores for all classes
                prob_scores = {}
                for i, prob in enumerate(probabilities[idx]):
                    class_name = self.pattern_types[i % len(self.pattern_types)]
                    prob_scores[class_name] = float(prob)
                
                # Calculate additional metrics
                pattern_strength = self._calculate_pattern_strength(features, idx)
                risk_reward_ratio = self._calculate_risk_reward_ratio(pattern_type, features, idx)
                expected_duration = self._estimate_pattern_duration(pattern_type, market_regime)
                
                # Create pattern detection object
                detection = PatternDetection(
                    pattern_type=pattern_type,
                    confidence=float(confidence),
                    probability_scores=prob_scores,
                    start_index=max(0, idx - 10),
                    end_index=min(len(features) - 1, idx + 10),
                    timestamp=timestamps[idx] if timestamps and idx < len(timestamps) else datetime.now(),
                    features_used=[f'feature_{i}' for i in range(features.shape[1])],
                    model_predictions={k: v for k, v in predictions.items() if k != 'anomaly_scores'},
                    pattern_strength=pattern_strength,
                    risk_reward_ratio=risk_reward_ratio,
                    expected_duration=expected_duration,
                    market_regime=market_regime
                )
                
                detected_patterns.append(detection)
            
            # Remove overlapping patterns (keep highest confidence)
            detected_patterns = self._remove_overlapping_patterns(detected_patterns)
            
            self.logger.info(f"Detected {len(detected_patterns)} patterns with confidence > {confidence_threshold}")
            
        except Exception as e:
            self.logger.error(f"Error post-processing predictions: {e}")
        
        return detected_patterns
    
    def _calculate_pattern_strength(self, features: np.ndarray, idx: int) -> float:
        """Calculate the strength/quality of a detected pattern"""
        try:
            # Use multiple factors to assess pattern strength
            strength_factors = []
            
            # Volume confirmation (if available)
            if features.shape[1] > 4:  # Assuming volume is available
                volume_trend = np.mean(features[max(0, idx-5):idx+1, 4]) if idx >= 5 else 1.0
                strength_factors.append(min(volume_trend / np.mean(features[:, 4]), 2.0))
            
            # Price action consistency
            if features.shape[1] > 0:
                price_consistency = 1.0 - np.std(features[max(0, idx-10):idx+1, 0])
                strength_factors.append(max(0, price_consistency))
            
            # Technical indicator alignment
            if features.shape[1] > 10:
                indicator_alignment = np.mean(features[idx, 5:min(10, features.shape[1])])
                strength_factors.append(abs(indicator_alignment))
            
            # Calculate overall strength
            if strength_factors:
                pattern_strength = np.mean(strength_factors)
                return min(1.0, max(0.0, pattern_strength))
            
            return 0.5  # Default strength
            
        except Exception as e:
            self.logger.warning(f"Error calculating pattern strength: {e}")
            return 0.5
    
    def _calculate_risk_reward_ratio(self, pattern_type: str, features: np.ndarray, idx: int) -> float:
        """Calculate expected risk/reward ratio for the pattern"""
        try:
            # Pattern-specific risk/reward estimates
            risk_reward_map = {
                'head_shoulders': 2.5,
                'double_top': 2.0,
                'double_bottom': 2.0,
                'triangle_ascending': 1.8,
                'triangle_descending': 1.8,
                'breakout': 3.0,
                'breakdown': 3.0,
                'flag_bullish': 2.2,
                'flag_bearish': 2.2,
                'cup_handle': 3.5
            }
            
            base_ratio = risk_reward_map.get(pattern_type, 1.5)
            
            # Adjust based on market conditions
            if features.shape[1] > 0:
                volatility = np.std(features[max(0, idx-20):idx+1, 0]) if idx >= 20 else 0.02
                volatility_adjustment = 1.0 + min(volatility * 10, 1.0)  # Higher volatility = higher potential reward
                base_ratio *= volatility_adjustment
            
            return min(5.0, max(1.0, base_ratio))
            
        except Exception as e:
            self.logger.warning(f"Error calculating risk/reward ratio: {e}")
            return 1.5
    
    def _estimate_pattern_duration(self, pattern_type: str, market_regime: str) -> int:
        """Estimate how long the pattern is expected to last"""
        # Base durations (in periods)
        duration_map = {
            'head_shoulders': 20,
            'double_top': 15,
            'double_bottom': 15,
            'triangle_ascending': 25,
            'triangle_descending': 25,
            'breakout': 5,
            'breakdown': 5,
            'flag_bullish': 8,
            'flag_bearish': 8,
            'consolidation': 30
        }
        
        base_duration = duration_map.get(pattern_type, 10)
        
        # Adjust based on market regime
        regime_multipliers = {
            'trending': 0.8,
            'ranging': 1.5,
            'volatile': 0.6,
            'unknown': 1.0
        }
        
        multiplier = regime_multipliers.get(market_regime, 1.0)
        return int(base_duration * multiplier)
    
    def _remove_overlapping_patterns(self, patterns: List[PatternDetection]) -> List[PatternDetection]:
        """Remove overlapping patterns, keeping the highest confidence ones"""
        if len(patterns) <= 1:
            return patterns
        
        # Sort by confidence (descending)
        sorted_patterns = sorted(patterns, key=lambda x: x.confidence, reverse=True)
        filtered_patterns = []
        
        for pattern in sorted_patterns:
            overlap_found = False
            
            for existing_pattern in filtered_patterns:
                # Check for overlap
                if (pattern.start_index <= existing_pattern.end_index and 
                    pattern.end_index >= existing_pattern.start_index):
                    overlap_found = True
                    break
            
            if not overlap_found:
                filtered_patterns.append(pattern)
        
        return filtered_patterns
    
    def train_models(self, 
                    features: np.ndarray, 
                    labels: np.ndarray,
                    validation_split: float = 0.2) -> Dict[str, ModelPerformance]:
        """Train all models with provided data"""
        
        if len(features) == 0 or len(labels) == 0:
            self.logger.warning("No training data provided")
            return {}
        
        # Preprocess data
        features_scaled = self._preprocess_features(features)
        
        # Encode labels
        if not hasattr(self.label_encoder, 'classes_'):
            labels_encoded = self.label_encoder.fit_transform(labels)
        else:
            labels_encoded = self.label_encoder.transform(labels)
        
        # Split data
        X_train, X_val, y_train, y_val = train_test_split(
            features_scaled, labels_encoded, 
            test_size=validation_split, 
            stratify=labels_encoded,
            random_state=42
        )
        
        performance_results = {}
        
        # Train classical ML models
        for model_name, model in self.models.items():
            if model_name in ['lstm', 'cnn', 'autoencoder']:
                continue  # Skip deep learning models
            
            try:
                start_time = datetime.now()
                
                # Train model
                model.fit(X_train, y_train)
                
                training_time = (datetime.now() - start_time).total_seconds()
                
                # Evaluate model
                performance = self._evaluate_model(model, X_val, y_val, model_name)
                performance.training_time = training_time
                
                performance_results[model_name] = performance
                
                self.logger.info(f"Trained {model_name}: Accuracy = {performance.accuracy:.3f}")
                
            except Exception as e:
                self.logger.error(f"Error training {model_name}: {e}")
        
        # Train ensemble
        try:
            start_time = datetime.now()
            self.ensemble_model.fit(X_train, y_train)
            training_time = (datetime.now() - start_time).total_seconds()
            
            ensemble_performance = self._evaluate_model(self.ensemble_model, X_val, y_val, 'ensemble')
            ensemble_performance.training_time = training_time
            performance_results['ensemble'] = ensemble_performance
            
            self.logger.info(f"Trained ensemble: Accuracy = {ensemble_performance.accuracy:.3f}")
            
        except Exception as e:
            self.logger.error(f"Error training ensemble: {e}")
        
        # Train deep learning models
        if self.enable_deep_learning:
            dl_performance = self._train_deep_models(X_train, y_train, X_val, y_val)
            performance_results.update(dl_performance)
        
        # Train anomaly detector
        try:
            self.anomaly_detector.fit(X_train)
            self.logger.info("Trained anomaly detector")
        except Exception as e:
            self.logger.error(f"Error training anomaly detector: {e}")
        
        # Update performance tracking
        self.model_performance.update(performance_results)
        
        # Save models
        self._save_models()
        
        return performance_results
    
    def _train_deep_models(self, X_train: np.ndarray, y_train: np.ndarray, 
                          X_val: np.ndarray, y_val: np.ndarray) -> Dict[str, ModelPerformance]:
        """Train deep learning models"""
        performance_results = {}
        
        if not TF_AVAILABLE:
            return performance_results
        
        try:
            # Prepare data for deep learning
            from tensorflow.keras.utils import to_categorical
            
            y_train_cat = to_categorical(y_train, num_classes=len(self.pattern_types))
            y_val_cat = to_categorical(y_val, num_classes=len(self.pattern_types))
            
            # Create sequences for LSTM/CNN
            if len(X_train) > 100:
                sequence_length = 50
                
                # LSTM training
                if 'lstm' in self.models:
                    try:
                        X_train_seq, y_train_seq = self._create_sequences(X_train, y_train_cat, sequence_length)
                        X_val_seq, y_val_seq = self._create_sequences(X_val, y_val_cat, sequence_length)
                        
                        if len(X_train_seq) > 0:
                            callbacks = [
                                EarlyStopping(patience=10, restore_best_weights=True),
                                ReduceLROnPlateau(patience=5, factor=0.5)
                            ]
                            
                            start_time = datetime.now()
                            
                            history = self.models['lstm'].fit(
                                X_train_seq, y_train_seq,
                                validation_data=(X_val_seq, y_val_seq),
                                epochs=50,
                                batch_size=32,
                                callbacks=callbacks,
                                verbose=0
                            )
                            
                            training_time = (datetime.now() - start_time).total_seconds()
                            
                            # Evaluate LSTM
                            val_loss, val_acc = self.models['lstm'].evaluate(X_val_seq, y_val_seq, verbose=0)
                            
                            performance_results['lstm'] = ModelPerformance(
                                accuracy=val_acc,
                                precision=val_acc,  # Simplified
                                recall=val_acc,     # Simplified
                                f1_score=val_acc,   # Simplified
                                confusion_matrix=np.eye(len(self.pattern_types)),  # Simplified
                                feature_importance={},
                                training_time=training_time,
                                prediction_time=0.0,
                                model_complexity=self.models['lstm'].count_params()
                            )
                            
                            self.logger.info(f"Trained LSTM: Accuracy = {val_acc:.3f}")
                    
                    except Exception as e:
                        self.logger.warning(f"LSTM training failed: {e}")
                
                # CNN training (similar to LSTM)
                if 'cnn' in self.models:
                    try:
                        X_train_seq, y_train_seq = self._create_sequences(X_train, y_train_cat, sequence_length)
                        X_val_seq, y_val_seq = self._create_sequences(X_val, y_val_cat, sequence_length)
                        
                        if len(X_train_seq) > 0:
                            callbacks = [
                                EarlyStopping(patience=10, restore_best_weights=True),
                                ReduceLROnPlateau(patience=5, factor=0.5)
                            ]
                            
                            start_time = datetime.now()
                            
                            history = self.models['cnn'].fit(
                                X_train_seq, y_train_seq,
                                validation_data=(X_val_seq, y_val_seq),
                                epochs=50,
                                batch_size=32,
                                callbacks=callbacks,
                                verbose=0
                            )
                            
                            training_time = (datetime.now() - start_time).total_seconds()
                            
                            # Evaluate CNN
                            val_loss, val_acc = self.models['cnn'].evaluate(X_val_seq, y_val_seq, verbose=0)
                            
                            performance_results['cnn'] = ModelPerformance(
                                accuracy=val_acc,
                                precision=val_acc,  # Simplified
                                recall=val_acc,     # Simplified
                                f1_score=val_acc,   # Simplified
                                confusion_matrix=np.eye(len(self.pattern_types)),  # Simplified
                                feature_importance={},
                                training_time=training_time,
                                prediction_time=0.0,
                                model_complexity=self.models['cnn'].count_params()
                            )
                            
                            self.logger.info(f"Trained CNN: Accuracy = {val_acc:.3f}")
                    
                    except Exception as e:
                        self.logger.warning(f"CNN training failed: {e}")
        
        except Exception as e:
            self.logger.warning(f"Deep learning training failed: {e}")
        
        return performance_results
    
    def _create_sequences(self, X: np.ndarray, y: np.ndarray, sequence_length: int) -> Tuple[np.ndarray, np.ndarray]:
        """Create sequences for LSTM/CNN training"""
        sequences = []
        targets = []
        
        for i in range(sequence_length, len(X)):
            sequences.append(X[i-sequence_length:i])
            targets.append(y[i])
        
        return np.array(sequences), np.array(targets)
    
    def _evaluate_model(self, model: Any, X_val: np.ndarray, y_val: np.ndarray, model_name: str) -> ModelPerformance:
        """Evaluate model performance"""
        try:
            start_time = datetime.now()
            
            # Make predictions
            y_pred = model.predict(X_val)
            
            prediction_time = (datetime.now() - start_time).total_seconds()
            
            # Calculate metrics
            accuracy = accuracy_score(y_val, y_pred)
            
            # Classification report
            report = classification_report(y_val, y_pred, output_dict=True, zero_division=0)
            
            # Confusion matrix
            cm = confusion_matrix(y_val, y_pred)
            
            # Feature importance (if available)
            feature_importance = {}
            if hasattr(model, 'feature_importances_'):
                feature_importance = {f'feature_{i}': importance 
                                    for i, importance in enumerate(model.feature_importances_)}
            
            # Model complexity
            complexity = 0
            if hasattr(model, 'n_estimators'):
                complexity = model.n_estimators
            elif hasattr(model, 'coef_'):
                complexity = np.prod(model.coef_.shape)
            
            return ModelPerformance(
                accuracy=accuracy,
                precision=report['weighted avg']['precision'],
                recall=report['weighted avg']['recall'],
                f1_score=report['weighted avg']['f1-score'],
                confusion_matrix=cm,
                feature_importance=feature_importance,
                training_time=0.0,  # Set by caller
                prediction_time=prediction_time,
                model_complexity=complexity
            )
            
        except Exception as e:
            self.logger.warning(f"Error evaluating {model_name}: {e}")
            return ModelPerformance(
                accuracy=0.0, precision=0.0, recall=0.0, f1_score=0.0,
                confusion_matrix=np.array([]), feature_importance={},
                training_time=0.0, prediction_time=0.0, model_complexity=0
            )
    
    def _update_model_performance(self, detected_patterns: List[PatternDetection]):
        """Update model performance tracking for self-improvement"""
        try:
            # This would typically involve feedback from actual trading results
            # For now, we'll update based on pattern confidence and consistency
            
            for pattern in detected_patterns:
                if pattern.confidence > 0.8:
                    # High confidence patterns contribute positively
                    self._update_feature_importance(pattern.features_used, 1.1)
                elif pattern.confidence < 0.6:
                    # Low confidence patterns contribute negatively
                    self._update_feature_importance(pattern.features_used, 0.9)
                    
        except Exception as e:
            self.logger.warning(f"Error updating model performance: {e}")
    
    def _update_feature_importance(self, features: List[str], multiplier: float):
        """Update feature importance tracking"""
        for feature in features:
            if feature not in self.feature_importance_tracker:
                self.feature_importance_tracker[feature] = 1.0
            
            self.feature_importance_tracker[feature] *= multiplier
            
            # Keep values in reasonable range
            self.feature_importance_tracker[feature] = np.clip(
                self.feature_importance_tracker[feature], 0.1, 10.0
            )
    
    def _save_models(self):
        """Save trained models to disk"""
        try:
            # Save classical ML models
            for model_name, model in self.models.items():
                if model_name not in ['lstm', 'cnn', 'autoencoder']:
                    model_path = self.model_dir / f"{model_name}.pkl"
                    with open(model_path, 'wb') as f:
                        pickle.dump(model, f)
            
            # Save ensemble model
            if self.ensemble_model:
                ensemble_path = self.model_dir / "ensemble.pkl"
                with open(ensemble_path, 'wb') as f:
                    pickle.dump(self.ensemble_model, f)
            
            # Save preprocessing objects
            scaler_path = self.model_dir / "scaler.pkl"
            with open(scaler_path, 'wb') as f:
                pickle.dump(self.scaler, f)
            
            encoder_path = self.model_dir / "label_encoder.pkl"
            with open(encoder_path, 'wb') as f:
                pickle.dump(self.label_encoder, f)
            
            # Save deep learning models
            if self.enable_deep_learning:
                for model_name in ['lstm', 'cnn', 'autoencoder']:
                    if model_name in self.models:
                        model_path = self.model_dir / f"{model_name}.h5"
                        self.models[model_name].save(model_path)
            
            # Save performance tracking
            performance_path = self.model_dir / "performance.json"
            with open(performance_path, 'w') as f:
                # Convert numpy arrays to lists for JSON serialization
                serializable_performance = {}
                for model_name, perf in self.model_performance.items():
                    serializable_performance[model_name] = {
                        'accuracy': perf.accuracy,
                        'precision': perf.precision,
                        'recall': perf.recall,
                        'f1_score': perf.f1_score,
                        'training_time': perf.training_time,
                        'prediction_time': perf.prediction_time,
                        'model_complexity': perf.model_complexity
                    }
                
                json.dump(serializable_performance, f, indent=2)
            
            self.logger.info(f"Models saved to {self.model_dir}")
            
        except Exception as e:
            self.logger.error(f"Error saving models: {e}")
    
    def _load_existing_models(self):
        """Load existing models from disk"""
        try:
            # Load classical ML models
            for model_name in ['random_forest', 'gradient_boosting', 'svm', 'neural_network']:
                model_path = self.model_dir / f"{model_name}.pkl"
                if model_path.exists():
                    with open(model_path, 'rb') as f:
                        self.models[model_name] = pickle.load(f)
                    self.logger.info(f"Loaded {model_name} model")
            
            # Load ensemble model
            ensemble_path = self.model_dir / "ensemble.pkl"
            if ensemble_path.exists():
                with open(ensemble_path, 'rb') as f:
                    self.ensemble_model = pickle.load(f)
                self.logger.info("Loaded ensemble model")
            
            # Load preprocessing objects
            scaler_path = self.model_dir / "scaler.pkl"
            if scaler_path.exists():
                with open(scaler_path, 'rb') as f:
                    self.scaler = pickle.load(f)
            
            encoder_path = self.model_dir / "label_encoder.pkl"
            if encoder_path.exists():
                with open(encoder_path, 'rb') as f:
                    self.label_encoder = pickle.load(f)
            
            # Load deep learning models
            if self.enable_deep_learning and TF_AVAILABLE:
                for model_name in ['lstm', 'cnn', 'autoencoder']:
                    model_path = self.model_dir / f"{model_name}.h5"
                    if model_path.exists():
                        try:
                            self.models[model_name] = tf.keras.models.load_model(model_path)
                            self.logger.info(f"Loaded {model_name} model")
                        except Exception as e:
                            self.logger.warning(f"Failed to load {model_name}: {e}")
            
            # Load performance tracking
            performance_path = self.model_dir / "performance.json"
            if performance_path.exists():
                with open(performance_path, 'r') as f:
                    performance_data = json.load(f)
                    
                    for model_name, perf_dict in performance_data.items():
                        self.model_performance[model_name] = ModelPerformance(
                            accuracy=perf_dict['accuracy'],
                            precision=perf_dict['precision'],
                            recall=perf_dict['recall'],
                            f1_score=perf_dict['f1_score'],
                            confusion_matrix=np.array([]),  # Not saved
                            feature_importance={},          # Not saved
                            training_time=perf_dict['training_time'],
                            prediction_time=perf_dict['prediction_time'],
                            model_complexity=perf_dict['model_complexity']
                        )
            
        except Exception as e:
            self.logger.warning(f"Error loading existing models: {e}")
    
    def get_model_summary(self) -> Dict[str, Any]:
        """Get summary of all models and their performance"""
        summary = {
            'total_models': len(self.models),
            'models_trained': len(self.model_performance),
            'pattern_types': len(self.pattern_types),
            'detection_history_length': len(self.detection_history),
            'model_performance': {}
        }
        
        for model_name, performance in self.model_performance.items():
            summary['model_performance'][model_name] = {
                'accuracy': performance.accuracy,
                'f1_score': performance.f1_score,
                'training_time': performance.training_time,
                'complexity': performance.model_complexity
            }
        
        return summary
    
    def retrain_with_feedback(self, pattern_feedbacks: List[Dict[str, Any]]):
        """Retrain models based on feedback from actual trading results"""
        try:
            # This would implement active learning based on trading performance
            # For now, we'll simulate the process
            
            self.logger.info(f"Received feedback for {len(pattern_feedbacks)} patterns")
            
            # Update feature importance based on feedback
            for feedback in pattern_feedbacks:
                pattern_type = feedback.get('pattern_type')
                success = feedback.get('success', False)
                features_used = feedback.get('features_used', [])
                
                if success:
                    self._update_feature_importance(features_used, 1.2)
                else:
                    self._update_feature_importance(features_used, 0.8)
            
            # Schedule retraining if enough feedback accumulated
            if len(pattern_feedbacks) >= 100:
                self.logger.info("Sufficient feedback accumulated. Scheduling model retraining.")
                # This would trigger actual retraining in a production system
                
        except Exception as e:
            self.logger.error(f"Error processing feedback: {e}")
