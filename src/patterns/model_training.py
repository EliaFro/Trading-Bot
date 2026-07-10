#!/usr/bin/env python3
"""
Self-Improving AI Model Training System for Pattern Discovery
Implements advanced machine learning techniques with continuous learning capabilities
"""

import numpy as np
import pandas as pd
import logging
from typing import Dict, List, Tuple, Optional, Any, Union, Callable
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from pathlib import Path
import json
import pickle
import sqlite3
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor
import threading
import queue
import time
from enum import Enum
import warnings
warnings.filterwarnings('ignore')

# Machine Learning imports
from sklearn.model_selection import (
    train_test_split, cross_val_score, GridSearchCV, 
    RandomizedSearchCV, StratifiedKFold, TimeSeriesSplit
)
from sklearn.ensemble import (
    RandomForestClassifier, GradientBoostingClassifier,
    ExtraTreesClassifier, AdaBoostClassifier, VotingClassifier
)
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.neural_network import MLPClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import StandardScaler, LabelEncoder, RobustScaler
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    classification_report, confusion_matrix, roc_auc_score,
    precision_recall_curve, roc_curve
)
from sklearn.decomposition import PCA
from sklearn.feature_selection import SelectKBest, f_classif, RFE
from sklearn.pipeline import Pipeline
from sklearn.base import BaseEstimator, ClassifierMixin
import optuna
from scipy import stats

# Deep Learning (if available)
try:
    import tensorflow as tf
    from tensorflow.keras.models import Sequential, Model, load_model
    from tensorflow.keras.layers import (
        Dense, LSTM, Conv1D, MaxPooling1D, Flatten, Dropout, 
        BatchNormalization, Attention, MultiHeadAttention,
        Input, Concatenate, Add, GlobalAveragePooling1D
    )
    from tensorflow.keras.optimizers import Adam, RMSprop, SGD
    from tensorflow.keras.callbacks import (
        EarlyStopping, ReduceLROnPlateau, ModelCheckpoint,
        TensorBoard, LearningRateScheduler
    )
    from tensorflow.keras.utils import to_categorical
    import tensorflow_probability as tfp
    TF_AVAILABLE = True
except ImportError:
    TF_AVAILABLE = False
    logging.warning("TensorFlow not available. Deep learning features disabled.")

class TrainingPhase(Enum):
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
    SUPERVISED = "supervised"
    SEMI_SUPERVISED = "semi_supervised"
    ACTIVE_LEARNING = "active_learning"
    TRANSFER_LEARNING = "transfer_learning"
    META_LEARNING = "meta_learning"
    CONTINUAL_LEARNING = "continual_learning"

@dataclass
class TrainingConfiguration:
    """Configuration for model training"""
    learning_strategy: LearningStrategy = LearningStrategy.SUPERVISED
    enable_hyperparameter_tuning: bool = True
    enable_feature_selection: bool = True
    enable_ensemble: bool = True
    enable_deep_learning: bool = True
    cross_validation_folds: int = 5
    test_size: float = 0.2
    validation_size: float = 0.2
    random_state: int = 42
    n_jobs: int = -1
    max_training_time_hours: float = 24.0
    early_stopping_patience: int = 10
    hyperparameter_search_iterations: int = 100
    ensemble_size: int = 5
    auto_feature_engineering: bool = True
    enable_uncertainty_quantification: bool = True
    model_selection_metric: str = 'f1_weighted'
    retraining_threshold: float = 0.05  # Performance drop threshold for retraining

@dataclass
class ModelMetrics:
    """Comprehensive model performance metrics"""
    accuracy: float = 0.0
    precision: float = 0.0
    recall: float = 0.0
    f1_score: float = 0.0
    auc_score: float = 0.0
    confusion_matrix: np.ndarray = field(default_factory=lambda: np.array([]))
    classification_report: Dict[str, Any] = field(default_factory=dict)
    feature_importance: Dict[str, float] = field(default_factory=dict)
    training_time: float = 0.0
    inference_time: float = 0.0
    memory_usage: float = 0.0
    model_complexity: int = 0
    uncertainty_metrics: Dict[str, float] = field(default_factory=dict)
    cross_validation_scores: List[float] = field(default_factory=list)
    learning_curve_data: Dict[str, List[float]] = field(default_factory=dict)

@dataclass
class TrainingResult:
    """Result of model training session"""
    model_id: str
    model_type: str
    training_phase: TrainingPhase
    metrics: ModelMetrics
    model_path: str
    configuration: TrainingConfiguration
    training_data_hash: str
    feature_names: List[str]
    label_mapping: Dict[str, int]
    timestamp: datetime = field(default_factory=datetime.now)
    notes: str = ""
    hyperparameters: Dict[str, Any] = field(default_factory=dict)
    validation_strategy: str = ""

class AdvancedModelTrainer:
    """
    Advanced AI model training system with self-improvement capabilities
    """
    
    def __init__(self, 
                 config: TrainingConfiguration = None,
                 models_dir: str = "./models/patterns",
                 data_dir: str = "./data/training",
                 enable_gpu: bool = True):
        
        self.config = config or TrainingConfiguration()
        self.models_dir = Path(models_dir)
        self.data_dir = Path(data_dir)
        self.models_dir.mkdir(parents=True, exist_ok=True)
        self.data_dir.mkdir(parents=True, exist_ok=True)
        
        self.logger = logging.getLogger(__name__)
        self.enable_gpu = enable_gpu and TF_AVAILABLE
        
        # Initialize components
        self.scaler = StandardScaler()
        self.label_encoder = LabelEncoder()
        self.feature_selector = None
        self.pca = None
        
        # Model registry
        self.model_registry = {}
        self.training_history = []
        self.performance_tracking = {}
        
        # Active learning components
        self.uncertainty_threshold = 0.3
        self.query_strategy = 'uncertainty_sampling'
        self.labeled_pool = []
        self.unlabeled_pool = []
        
        # Continuous learning
        self.model_versions = {}
        self.performance_drift_detector = None
        
        # Initialize ML models
        self._initialize_model_pool()
        
        # Setup GPU if available
        if self.enable_gpu and TF_AVAILABLE:
            self._setup_gpu()
        
        # Load existing models and history
        self._load_training_history()
        
        # Start monitoring thread
        self.monitoring_active = True
        self.monitoring_thread = threading.Thread(target=self._monitor_performance, daemon=True)
        self.monitoring_thread.start()
    
    def _initialize_model_pool(self):
        """Initialize pool of available models"""
        
        # Classical ML models
        self.classical_models = {
            'random_forest': {
                'model': RandomForestClassifier,
                'param_grid': {
                    'n_estimators': [100, 200, 300, 500],
                    'max_depth': [10, 15, 20, None],
                    'min_samples_split': [2, 5, 10],
                    'min_samples_leaf': [1, 2, 4],
                    'max_features': ['sqrt', 'log2', None]
                }
            },
            'gradient_boosting': {
                'model': GradientBoostingClassifier,
                'param_grid': {
                    'n_estimators': [100, 200, 300],
                    'learning_rate': [0.01, 0.1, 0.2],
                    'max_depth': [3, 5, 7, 9],
                    'subsample': [0.8, 0.9, 1.0]
                }
            },
            'extra_trees': {
                'model': ExtraTreesClassifier,
                'param_grid': {
                    'n_estimators': [100, 200, 300],
                    'max_depth': [10, 15, 20, None],
                    'min_samples_split': [2, 5, 10],
                    'min_samples_leaf': [1, 2, 4]
                }
            },
            'svm': {
                'model': SVC,
                'param_grid': {
                    'C': [0.1, 1, 10, 100],
                    'gamma': ['scale', 'auto', 0.001, 0.01, 0.1],
                    'kernel': ['rbf', 'poly', 'sigmoid']
                }
            },
            'neural_network': {
                'model': MLPClassifier,
                'param_grid': {
                    'hidden_layer_sizes': [(50,), (100,), (50, 50), (100, 50), (100, 100)],
                    'activation': ['relu', 'tanh'],
                    'alpha': [0.0001, 0.001, 0.01],
                    'learning_rate': ['constant', 'adaptive'],
                    'max_iter': [500, 1000, 2000]
                }
            },
            'logistic_regression': {
                'model': LogisticRegression,
                'param_grid': {
                    'C': [0.01, 0.1, 1, 10, 100],
                    'penalty': ['l1', 'l2'],
                    'solver': ['liblinear', 'saga'],
                    'max_iter': [1000, 2000]
                }
            }
        }
        
        # Deep learning architectures
        if TF_AVAILABLE:
            self.deep_learning_architectures = {
                'lstm_classifier': self._create_lstm_classifier,
                'cnn_classifier': self._create_cnn_classifier,
                'transformer_classifier': self._create_transformer_classifier,
                'hybrid_cnn_lstm': self._create_hybrid_cnn_lstm,
                'attention_network': self._create_attention_network
            }
    
    def train_models(self, 
                    features: np.ndarray,
                    labels: np.ndarray,
                    feature_names: List[str] = None,
                    validation_data: Tuple[np.ndarray, np.ndarray] = None,
                    incremental: bool = False) -> List[TrainingResult]:
        """
        Train multiple models with comprehensive evaluation
        
        Args:
            features: Training features
            labels: Training labels
            feature_names: Names of features
            validation_data: Optional validation data
            incremental: Whether to perform incremental learning
            
        Returns:
            List of training results for all models
        """
        try:
            self.logger.info(f"Starting model training with {len(features)} samples")
            
            if len(features) == 0:
                raise ValueError("No training data provided")
            
            # Data preparation
            X_processed, y_processed = self._prepare_training_data(features, labels)
            
            # Feature engineering
            if self.config.auto_feature_engineering:
                X_processed = self._engineer_features(X_processed, feature_names)
            
            # Split data
            if validation_data is None:
                X_train, X_val, y_train, y_val = self._split_data(X_processed, y_processed)
            else:
                X_train, y_train = X_processed, y_processed
                X_val, y_val = validation_data
            
            # Feature selection
            if self.config.enable_feature_selection:
                X_train, X_val = self._select_features(X_train, y_train, X_val)
            
            training_results = []
            
            # Train classical ML models
            classical_results = self._train_classical_models(X_train, y_train, X_val, y_val, feature_names)
            training_results.extend(classical_results)
            
            # Train deep learning models
            if self.config.enable_deep_learning and TF_AVAILABLE:
                dl_results = self._train_deep_learning_models(X_train, y_train, X_val, y_val, feature_names)
                training_results.extend(dl_results)
            
            # Create ensemble
            if self.config.enable_ensemble and len(training_results) > 1:
                ensemble_result = self._create_ensemble(training_results, X_val, y_val, feature_names)
                training_results.append(ensemble_result)
            
            # Update training history
            self.training_history.extend(training_results)
            
            # Save results
            self._save_training_results(training_results)
            
            # Update performance tracking
            self._update_performance_tracking(training_results)
            
            self.logger.info(f"Training completed. {len(training_results)} models trained.")
            
            return training_results
            
        except Exception as e:
            self.logger.error(f"Error in model training: {e}")
            raise
    
    def _prepare_training_data(self, features: np.ndarray, labels: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Prepare and clean training data"""
        try:
            # Handle missing values
            features_clean = np.nan_to_num(features, nan=0.0, posinf=1e6, neginf=-1e6)
            
            # Remove samples with all zeros (likely invalid)
            valid_samples = ~np.all(features_clean == 0, axis=1)
            features_clean = features_clean[valid_samples]
            labels_clean = labels[valid_samples]
            
            # Scale features
            features_scaled = self.scaler.fit_transform(features_clean)
            
            # Encode labels
            labels_encoded = self.label_encoder.fit_transform(labels_clean)
            
            self.logger.info(f"Data prepared: {len(features_scaled)} samples, {features_scaled.shape[1]} features")
            
            return features_scaled, labels_encoded
            
        except Exception as e:
            self.logger.error(f"Error preparing training data: {e}")
            raise
    
    def _engineer_features(self, features: np.ndarray, feature_names: List[str] = None) -> np.ndarray:
        """Automatic feature engineering"""
        try:
            engineered_features = [features]
            
            # Polynomial features (degree 2, interaction only)
            if features.shape[1] <= 20:  # Only for smaller feature sets
                from sklearn.preprocessing import PolynomialFeatures
                poly = PolynomialFeatures(degree=2, interaction_only=True, include_bias=False)
                poly_features = poly.fit_transform(features)
                engineered_features.append(poly_features[:, features.shape[1]:])  # Only new features
            
            # Statistical features
            if features.shape[1] >= 5:
                # Rolling statistics
                window_sizes = [3, 5, 10]
                for window in window_sizes:
                    if features.shape[0] > window:
                        rolling_mean = np.array([np.mean(features[max(0, i-window):i+1], axis=0) 
                                               for i in range(len(features))])
                        rolling_std = np.array([np.std(features[max(0, i-window):i+1], axis=0) 
                                              for i in range(len(features))])
                        engineered_features.extend([rolling_mean, rolling_std])
            
            # Combine all features
            all_features = np.hstack(engineered_features)
            
            # Remove highly correlated features
            correlation_matrix = np.corrcoef(all_features.T)
            high_corr_pairs = np.where((np.abs(correlation_matrix) > 0.95) & 
                                     (correlation_matrix != 1.0))
            
            features_to_remove = set()
            for i, j in zip(high_corr_pairs[0], high_corr_pairs[1]):
                if i != j:
                    features_to_remove.add(max(i, j))
            
            if features_to_remove:
                keep_features = [i for i in range(all_features.shape[1]) if i not in features_to_remove]
                all_features = all_features[:, keep_features]
            
            self.logger.info(f"Feature engineering: {features.shape[1]} -> {all_features.shape[1]} features")
            
            return all_features
            
        except Exception as e:
            self.logger.warning(f"Error in feature engineering: {e}")
            return features
    
    def _split_data(self, features: np.ndarray, labels: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """Split data into training and validation sets"""
        # Use stratified split to maintain class distribution
        X_train, X_val, y_train, y_val = train_test_split(
            features, labels,
            test_size=self.config.validation_size,
            stratify=labels,
            random_state=self.config.random_state
        )
        
        self.logger.info(f"Data split: {len(X_train)} train, {len(X_val)} validation")
        
        return X_train, X_val, y_train, y_val
    
    def _select_features(self, X_train: np.ndarray, y_train: np.ndarray, X_val: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Select most important features"""
        try:
            # Use multiple feature selection methods
            
            # 1. Statistical feature selection
            k_best = min(50, X_train.shape[1] // 2)  # Select top 50 or half of features
            statistical_selector = SelectKBest(f_classif, k=k_best)
            X_train_stat = statistical_selector.fit_transform(X_train, y_train)
            X_val_stat = statistical_selector.transform(X_val)
            
            # 2. Recursive feature elimination with Random Forest
            if X_train.shape[1] > 20:
                rf = RandomForestClassifier(n_estimators=50, random_state=self.config.random_state)
                rfe_selector = RFE(rf, n_features_to_select=min(30, X_train.shape[1]))
                X_train_rfe = rfe_selector.fit_transform(X_train, y_train)
                X_val_rfe = rfe_selector.transform(X_val)
                
                # Combine selections
                stat_features = statistical_selector.get_support()
                rfe_features = rfe_selector.get_support()
                combined_features = stat_features | rfe_features
                
                X_train_selected = X_train[:, combined_features]
                X_val_selected = X_val[:, combined_features]
            else:
                X_train_selected = X_train_stat
                X_val_selected = X_val_stat
            
            # Store feature selector for later use
            self.feature_selector = statistical_selector
            
            self.logger.info(f"Feature selection: {X_train.shape[1]} -> {X_train_selected.shape[1]} features")
            
            return X_train_selected, X_val_selected
            
        except Exception as e:
            self.logger.warning(f"Error in feature selection: {e}")
            return X_train, X_val
    
    def _train_classical_models(self, X_train: np.ndarray, y_train: np.ndarray,
                               X_val: np.ndarray, y_val: np.ndarray,
                               feature_names: List[str] = None) -> List[TrainingResult]:
        """Train classical machine learning models"""
        
        results = []
        
        for model_name, model_config in self.classical_models.items():
            try:
                self.logger.info(f"Training {model_name}")
                
                start_time = time.time()
                
                # Get base model
                model_class = model_config['model']
                
                # Hyperparameter tuning
                if self.config.enable_hyperparameter_tuning:
                    best_model = self._tune_hyperparameters(
                        model_class, model_config['param_grid'], 
                        X_train, y_train, model_name
                    )
                else:
                    best_model = model_class(random_state=self.config.random_state)
                    best_model.fit(X_train, y_train)
                
                training_time = time.time() - start_time
                
                # Evaluate model
                metrics = self._evaluate_model(best_model, X_val, y_val, X_train, y_train)
                metrics.training_time = training_time
                
                # Save model
                model_path = self._save_model(best_model, model_name)
                
                # Create training result
                result = TrainingResult(
                    model_id=f"{model_name}_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
                    model_type=model_name,
                    training_phase=TrainingPhase.TRAINING,
                    metrics=metrics,
                    model_path=model_path,
                    configuration=self.config,
                    training_data_hash=self._hash_data(X_train, y_train),
                    feature_names=feature_names or [f"feature_{i}" for i in range(X_train.shape[1])],
                    label_mapping=dict(zip(self.label_encoder.classes_, range(len(self.label_encoder.classes_)))),
                    hyperparameters=best_model.get_params() if hasattr(best_model, 'get_params') else {},
                    validation_strategy="holdout"
                )
                
                results.append(result)
                
                self.logger.info(f"{model_name} trained: {metrics.f1_score:.3f} F1-score")
                
            except Exception as e:
                self.logger.error(f"Error training {model_name}: {e}")
                continue
        
        return results
    
    def _tune_hyperparameters(self, model_class, param_grid: Dict, 
                             X_train: np.ndarray, y_train: np.ndarray,
                             model_name: str):
        """Tune hyperparameters using Optuna or GridSearch"""
        
        try:
            # Use Optuna for more efficient hyperparameter optimization
            def objective(trial):
                params = {}
                
                # Convert param_grid to Optuna suggestions
                for param, values in param_grid.items():
                    if isinstance(values[0], int):
                        params[param] = trial.suggest_int(param, min(values), max(values))
                    elif isinstance(values[0], float):
                        params[param] = trial.suggest_float(param, min(values), max(values))
                    else:
                        params[param] = trial.suggest_categorical(param, values)
                
                # Add random_state if the model supports it
                if 'random_state' in model_class().get_params():
                    params['random_state'] = self.config.random_state
                
                # Create and train model
                model = model_class(**params)
                
                # Use cross-validation for more robust evaluation
                cv_scores = cross_val_score(
                    model, X_train, y_train, 
                    cv=min(5, len(np.unique(y_train))),
                    scoring=self.config.model_selection_metric,
                    n_jobs=1  # Avoid nested parallelism
                )
                
                return np.mean(cv_scores)
            
            # Create study
            study = optuna.create_study(direction='maximize', study_name=f"{model_name}_tuning")
            study.optimize(objective, n_trials=min(self.config.hyperparameter_search_iterations, 50))
            
            # Get best parameters
            best_params = study.best_params
            if 'random_state' in model_class().get_params():
                best_params['random_state'] = self.config.random_state
            
            # Train final model with best parameters
            best_model = model_class(**best_params)
            best_model.fit(X_train, y_train)
            
            self.logger.info(f"{model_name} hyperparameter tuning completed. Best score: {study.best_value:.3f}")
            
            return best_model
            
        except Exception as e:
            self.logger.warning(f"Hyperparameter tuning failed for {model_name}: {e}")
            # Fallback to default parameters
            model = model_class()
            if hasattr(model, 'set_params') and 'random_state' in model.get_params():
                model.set_params(random_state=self.config.random_state)
            model.fit(X_train, y_train)
            return model
    
    def _train_deep_learning_models(self, X_train: np.ndarray, y_train: np.ndarray,
                                   X_val: np.ndarray, y_val: np.ndarray,
                                   feature_names: List[str] = None) -> List[TrainingResult]:
        """Train deep learning models"""
        
        if not TF_AVAILABLE:
            return []
        
        results = []
        
        # Prepare data for deep learning
        y_train_cat = to_categorical(y_train, num_classes=len(np.unique(y_train)))
        y_val_cat = to_categorical(y_val, num_classes=len(np.unique(y_val)))
        
        for arch_name, arch_func in self.deep_learning_architectures.items():
            try:
                self.logger.info(f"Training {arch_name}")
                
                start_time = time.time()
                
                # Create model architecture
                model = arch_func(input_shape=(X_train.shape[1],), num_classes=len(np.unique(y_train)))
                
                # Prepare callbacks
                callbacks = self._get_training_callbacks(arch_name)
                
                # Train model
                history = model.fit(
                    X_train, y_train_cat,
                    validation_data=(X_val, y_val_cat),
                    epochs=100,
                    batch_size=min(32, len(X_train) // 10),
                    callbacks=callbacks,
                    verbose=0
                )
                
                training_time = time.time() - start_time
                
                # Evaluate model
                y_pred_proba = model.predict(X_val, verbose=0)
                y_pred = np.argmax(y_pred_proba, axis=1)
                
                metrics = self._calculate_metrics(y_val, y_pred, y_pred_proba)
                metrics.training_time = training_time
                
                # Save model
                model_path = self._save_deep_model(model, arch_name)
                
                # Create training result
                result = TrainingResult(
                    model_id=f"{arch_name}_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
                    model_type=arch_name,
                    training_phase=TrainingPhase.TRAINING,
                    metrics=metrics,
                    model_path=model_path,
                    configuration=self.config,
                    training_data_hash=self._hash_data(X_train, y_train),
                    feature_names=feature_names or [f"feature_{i}" for i in range(X_train.shape[1])],
                    label_mapping=dict(zip(self.label_encoder.classes_, range(len(self.label_encoder.classes_)))),
                    validation_strategy="holdout"
                )
                
                results.append(result)
                
                self.logger.info(f"{arch_name} trained: {metrics.f1_score:.3f} F1-score")
                
            except Exception as e:
                self.logger.error(f"Error training {arch_name}: {e}")
                continue
        
        return results
    
    def _create_lstm_classifier(self, input_shape: Tuple[int], num_classes: int):
        """Create LSTM classifier"""
        model = Sequential([
            Dense(128, activation='relu', input_shape=input_shape),
            Dropout(0.3),
            Dense(64, activation='relu'),
            Dropout(0.3),
            Dense(32, activation='relu'),
            Dropout(0.2),
            Dense(num_classes, activation='softmax')
        ])
        
        model.compile(
            optimizer=Adam(learning_rate=0.001),
            loss='categorical_crossentropy',
            metrics=['accuracy']
        )
        
        return model
    
    def _create_cnn_classifier(self, input_shape: Tuple[int], num_classes: int):
        """Create 1D CNN classifier"""
        # Reshape input for 1D CNN
        input_layer = Input(shape=input_shape)
        reshaped = tf.expand_dims(input_layer, axis=-1)
        
        # CNN layers
        conv1 = Conv1D(64, 3, activation='relu', padding='same')(reshaped)
        conv1 = BatchNormalization()(conv1)
        pool1 = MaxPooling1D(2)(conv1)
        
        conv2 = Conv1D(128, 3, activation='relu', padding='same')(pool1)
        conv2 = BatchNormalization()(conv2)
        pool2 = MaxPooling1D(2)(conv2)
        
        # Global pooling and dense layers
        gap = GlobalAveragePooling1D()(pool2)
        dense1 = Dense(128, activation='relu')(gap)
        dropout1 = Dropout(0.5)(dense1)
        dense2 = Dense(64, activation='relu')(dropout1)
        dropout2 = Dropout(0.3)(dense2)
        output = Dense(num_classes, activation='softmax')(dropout2)
        
        model = Model(inputs=input_layer, outputs=output)
        
        model.compile(
            optimizer=Adam(learning_rate=0.001),
            loss='categorical_crossentropy',
            metrics=['accuracy']
        )
        
        return model
    
    def _create_transformer_classifier(self, input_shape: Tuple[int], num_classes: int):
        """Create Transformer-based classifier"""
        input_layer = Input(shape=input_shape)
        
        # Reshape for attention mechanism
        reshaped = tf.expand_dims(input_layer, axis=1)  # Add sequence dimension
        
        # Multi-head attention
        attention = MultiHeadAttention(num_heads=4, key_dim=64)(reshaped, reshaped)
        attention = Dropout(0.1)(attention)
        attention = Add()([reshaped, attention])  # Residual connection
        
        # Feed-forward network
        ffn = Dense(128, activation='relu')(attention)
        ffn = Dropout(0.1)(ffn)
        ffn = Dense(input_shape[0])(ffn)
        ffn = Add()([attention, ffn])  # Residual connection
        
        # Global pooling and classification
        pooled = GlobalAveragePooling1D()(ffn)
        dense = Dense(64, activation='relu')(pooled)
        dropout = Dropout(0.3)(dense)
        output = Dense(num_classes, activation='softmax')(dropout)
        
        model = Model(inputs=input_layer, outputs=output)
        
        model.compile(
            optimizer=Adam(learning_rate=0.001),
            loss='categorical_crossentropy',
            metrics=['accuracy']
        )
        
        return model
    
    def _create_hybrid_cnn_lstm(self, input_shape: Tuple[int], num_classes: int):
        """Create hybrid CNN-LSTM classifier"""
        input_layer = Input(shape=input_shape)
        
        # Reshape for CNN
        reshaped = tf.expand_dims(input_layer, axis=-1)
        
        # CNN feature extraction
        conv1 = Conv1D(64, 3, activation='relu')(reshaped)
        conv1 = BatchNormalization()(conv1)
        conv2 = Conv1D(128, 3, activation='relu')(conv1)
        conv2 = BatchNormalization()(conv2)
        
        # LSTM for sequence modeling
        lstm = LSTM(64, return_sequences=False)(conv2)
        
        # Dense layers
        dense1 = Dense(64, activation='relu')(lstm)
        dropout = Dropout(0.3)(dense1)
        output = Dense(num_classes, activation='softmax')(dropout)
        
        model = Model(inputs=input_layer, outputs=output)
        
        model.compile(
            optimizer=Adam(learning_rate=0.001),
            loss='categorical_crossentropy',
            metrics=['accuracy']
        )
        
        return model
    
    def _create_attention_network(self, input_shape: Tuple[int], num_classes: int):
        """Create attention-based network"""
        input_layer = Input(shape=input_shape)
        
        # Dense layers with attention
        dense1 = Dense(128, activation='relu')(input_layer)
        dense1 = Dropout(0.3)(dense1)
        
        dense2 = Dense(64, activation='relu')(dense1)
        dense2 = Dropout(0.3)(dense2)
        
        # Self-attention mechanism (simplified)
        attention_weights = Dense(64, activation='softmax')(dense2)
        attended_features = tf.multiply(dense2, attention_weights)
        
        # Final classification
        output = Dense(num_classes, activation='softmax')(attended_features)
        
        model = Model(inputs=input_layer, outputs=output)
        
        model.compile(
            optimizer=Adam(learning_rate=0.001),
            loss='categorical_crossentropy',
            metrics=['accuracy']
        )
        
        return model
    
    def _get_training_callbacks(self, model_name: str) -> List:
        """Get training callbacks for deep learning models"""
        callbacks = []
        
        # Early stopping
        early_stopping = EarlyStopping(
            monitor='val_loss',
            patience=self.config.early_stopping_patience,
            restore_best_weights=True
        )
        callbacks.append(early_stopping)
        
        # Learning rate reduction
        lr_reduction = ReduceLROnPlateau(
            monitor='val_loss',
            factor=0.5,
            patience=5,
            min_lr=1e-7
        )
        callbacks.append(lr_reduction)
        
        # Model checkpoint
        checkpoint_path = self.models_dir / f"{model_name}_checkpoint.h5"
        checkpoint = ModelCheckpoint(
            str(checkpoint_path),
            monitor='val_loss',
            save_best_only=True,
            save_weights_only=False
        )
        callbacks.append(checkpoint)
        
        return callbacks
    
    def _evaluate_model(self, model, X_val: np.ndarray, y_val: np.ndarray,
                       X_train: np.ndarray = None, y_train: np.ndarray = None) -> ModelMetrics:
        """Comprehensive model evaluation"""
        
        start_time = time.time()
        
        # Make predictions
        y_pred = model.predict(X_val)
        y_pred_proba = None
        
        if hasattr(model, 'predict_proba'):
            y_pred_proba = model.predict_proba(X_val)
        
        inference_time = (time.time() - start_time) / len(X_val)
        
        # Calculate metrics
        metrics = self._calculate_metrics(y_val, y_pred, y_pred_proba)
        metrics.inference_time = inference_time
        
        # Feature importance
        if hasattr(model, 'feature_importances_'):
            feature_importance = {f'feature_{i}': importance 
                                for i, importance in enumerate(model.feature_importances_)}
            metrics.feature_importance = feature_importance
        
        # Cross-validation scores
        if X_train is not None and y_train is not None:
            try:
                cv_scores = cross_val_score(
                    model, X_train, y_train,
                    cv=min(self.config.cross_validation_folds, len(np.unique(y_train))),
                    scoring=self.config.model_selection_metric,
                    n_jobs=1
                )
                metrics.cross_validation_scores = cv_scores.tolist()
            except Exception as e:
                self.logger.warning(f"Cross-validation failed: {e}")
        
        # Model complexity
        complexity = 0
        if hasattr(model, 'n_estimators'):
            complexity = model.n_estimators
        elif hasattr(model, 'coef_'):
            complexity = np.prod(model.coef_.shape)
        elif hasattr(model, 'count_params'):
            complexity = model.count_params()
        
        metrics.model_complexity = complexity
        
        return metrics
    
    def _calculate_metrics(self, y_true: np.ndarray, y_pred: np.ndarray, 
                          y_pred_proba: np.ndarray = None) -> ModelMetrics:
        """Calculate comprehensive performance metrics"""
        
        metrics = ModelMetrics()
        
        try:
            # Basic metrics
            metrics.accuracy = accuracy_score(y_true, y_pred)
            metrics.precision = precision_score(y_true, y_pred, average='weighted', zero_division=0)
            metrics.recall = recall_score(y_true, y_pred, average='weighted', zero_division=0)
            metrics.f1_score = f1_score(y_true, y_pred, average='weighted', zero_division=0)
            
            # Confusion matrix
            metrics.confusion_matrix = confusion_matrix(y_true, y_pred)
            
            # Classification report
            metrics.classification_report = classification_report(y_true, y_pred, output_dict=True, zero_division=0)
            
            # AUC score (for multi-class, use ovr strategy)
            if y_pred_proba is not None and len(np.unique(y_true)) > 2:
                try:
                    metrics.auc_score = roc_auc_score(y_true, y_pred_proba, multi_class='ovr', average='weighted')
                except Exception:
                    metrics.auc_score = 0.0
            elif y_pred_proba is not None and len(np.unique(y_true)) == 2:
                try:
                    metrics.auc_score = roc_auc_score(y_true, y_pred_proba[:, 1])
                except Exception:
                    metrics.auc_score = 0.0
            
            # Uncertainty quantification
            if y_pred_proba is not None and self.config.enable_uncertainty_quantification:
                metrics.uncertainty_metrics = self._calculate_uncertainty_metrics(y_pred_proba)
            
        except Exception as e:
            self.logger.warning(f"Error calculating metrics: {e}")
        
        return metrics
    
    def _calculate_uncertainty_metrics(self, y_pred_proba: np.ndarray) -> Dict[str, float]:
        """Calculate uncertainty metrics"""
        uncertainty_metrics = {}
        
        try:
            # Prediction entropy (measure of uncertainty)
            entropy = -np.sum(y_pred_proba * np.log(y_pred_proba + 1e-8), axis=1)
            uncertainty_metrics['mean_entropy'] = np.mean(entropy)
            uncertainty_metrics['std_entropy'] = np.std(entropy)
            
            # Maximum probability (confidence)
            max_probs = np.max(y_pred_proba, axis=1)
            uncertainty_metrics['mean_confidence'] = np.mean(max_probs)
            uncertainty_metrics['std_confidence'] = np.std(max_probs)
            
            # Prediction margin (difference between top two predictions)
            sorted_probs = np.sort(y_pred_proba, axis=1)
            margins = sorted_probs[:, -1] - sorted_probs[:, -2]
            uncertainty_metrics['mean_margin'] = np.mean(margins)
            uncertainty_metrics['std_margin'] = np.std(margins)
            
        except Exception as e:
            self.logger.warning(f"Error calculating uncertainty metrics: {e}")
        
        return uncertainty_metrics
    
    def _create_ensemble(self, training_results: List[TrainingResult],
                        X_val: np.ndarray, y_val: np.ndarray,
                        feature_names: List[str] = None) -> TrainingResult:
        """Create ensemble model from trained models"""
        
        try:
            self.logger.info("Creating ensemble model")
            
            # Select top performing models
            sorted_results = sorted(training_results, 
                                  key=lambda x: x.metrics.f1_score, 
                                  reverse=True)
            
            top_models = sorted_results[:self.config.ensemble_size]
            
            # Load models
            models = []
            for result in top_models:
                try:
                    if result.model_type in self.deep_learning_architectures and TF_AVAILABLE:
                        model = load_model(result.model_path)
                    else:
                        with open(result.model_path, 'rb') as f:
                            model = pickle.load(f)
                    models.append((result.model_type, model))
                except Exception as e:
                    self.logger.warning(f"Failed to load model {result.model_type}: {e}")
            
            if not models:
                raise ValueError("No models could be loaded for ensemble")
            
            # Create ensemble predictions
            ensemble_predictions = []
            
            for model_name, model in models:
                try:
                    if model_name in self.deep_learning_architectures:
                        pred = model.predict(X_val, verbose=0)
                        if pred.shape[1] > 1:  # Multi-class
                            pred = np.argmax(pred, axis=1)
                    else:
                        pred = model.predict(X_val)
                    
                    ensemble_predictions.append(pred)
                except Exception as e:
                    self.logger.warning(f"Failed to get predictions from {model_name}: {e}")
            
            if not ensemble_predictions:
                raise ValueError("No predictions could be obtained from ensemble models")
            
            # Combine predictions using majority voting
            ensemble_predictions = np.array(ensemble_predictions)
            final_predictions = stats.mode(ensemble_predictions, axis=0)[0].flatten()
            
            # Calculate ensemble metrics
            metrics = self._calculate_metrics(y_val, final_predictions)
            
            # Create ensemble model object (simplified)
            ensemble_model = {
                'models': [(name, model) for name, model in models],
                'voting_strategy': 'majority',
                'model_weights': [result.metrics.f1_score for result in top_models]
            }
            
            # Save ensemble
            ensemble_path = self._save_model(ensemble_model, 'ensemble')
            
            # Create training result
            ensemble_result = TrainingResult(
                model_id=f"ensemble_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
                model_type='ensemble',
                training_phase=TrainingPhase.ENSEMBLE_CREATION,
                metrics=metrics,
                model_path=ensemble_path,
                configuration=self.config,
                training_data_hash=top_models[0].training_data_hash,
                feature_names=feature_names or [f"feature_{i}" for i in range(X_val.shape[1])],
                label_mapping=top_models[0].label_mapping,
                hyperparameters={'ensemble_size': len(models), 'base_models': [name for name, _ in models]},
                validation_strategy="holdout"
            )
            
            self.logger.info(f"Ensemble created: {metrics.f1_score:.3f} F1-score")
            
            return ensemble_result
            
        except Exception as e:
            self.logger.error(f"Error creating ensemble: {e}")
            # Return best individual model as fallback
            return max(training_results, key=lambda x: x.metrics.f1_score)
    
    def _save_model(self, model, model_name: str) -> str:
        """Save trained model"""
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        model_path = self.models_dir / f"{model_name}_{timestamp}.pkl"
        
        try:
            with open(model_path, 'wb') as f:
                pickle.dump(model, f)
            return str(model_path)
        except Exception as e:
            self.logger.error(f"Error saving model {model_name}: {e}")
            return ""
    
    def _save_deep_model(self, model, model_name: str) -> str:
        """Save deep learning model"""
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        model_path = self.models_dir / f"{model_name}_{timestamp}.h5"
        
        try:
            model.save(str(model_path))
            return str(model_path)
        except Exception as e:
            self.logger.error(f"Error saving deep model {model_name}: {e}")
            return ""
    
    def _save_training_results(self, results: List[TrainingResult]):
        """Save training results to database"""
        try:
            db_path = self.data_dir / "training_history.db"
            conn = sqlite3.connect(str(db_path))
            
            for result in results:
                # Convert result to dictionary
                result_dict = {
                    'model_id': result.model_id,
                    'model_type': result.model_type,
                    'training_phase': result.training_phase.value,
                    'accuracy': result.metrics.accuracy,
                    'precision': result.metrics.precision,
                    'recall': result.metrics.recall,
                    'f1_score': result.metrics.f1_score,
                    'auc_score': result.metrics.auc_score,
                    'training_time': result.metrics.training_time,
                    'inference_time': result.metrics.inference_time,
                    'model_complexity': result.metrics.model_complexity,
                    'model_path': result.model_path,
                    'training_data_hash': result.training_data_hash,
                    'timestamp': result.timestamp.isoformat(),
                    'hyperparameters': json.dumps(result.hyperparameters),
                    'validation_strategy': result.validation_strategy
                }
                
                # Insert into database
                columns = ', '.join(result_dict.keys())
                placeholders = ', '.join(['?' for _ in result_dict.values()])
                
                conn.execute(f'''
                    CREATE TABLE IF NOT EXISTS training_results (
                        id INTEGER PRIMARY KEY AUTOINCREMENT,
                        model_id TEXT,
                        model_type TEXT,
                        training_phase TEXT,
                        accuracy REAL,
                        precision REAL,
                        recall REAL,
                        f1_score REAL,
                        auc_score REAL,
                        training_time REAL,
                        inference_time REAL,
                        model_complexity INTEGER,
                        model_path TEXT,
                        training_data_hash TEXT,
                        timestamp TEXT,
                        hyperparameters TEXT,
                        validation_strategy TEXT
                    )
                ''')
                
                conn.execute(f'INSERT INTO training_results ({columns}) VALUES ({placeholders})',
                           list(result_dict.values()))
            
            conn.commit()
            conn.close()
            
            self.logger.info(f"Saved {len(results)} training results to database")
            
        except Exception as e:
            self.logger.error(f"Error saving training results: {e}")
    
    def _load_training_history(self):
        """Load existing training history"""
        try:
            db_path = self.data_dir / "training_history.db"
            if not db_path.exists():
                return
            
            conn = sqlite3.connect(str(db_path))
            
            # Load recent training results
            query = '''
                SELECT * FROM training_results 
                ORDER BY timestamp DESC 
                LIMIT 1000
            '''
            
            df = pd.read_sql_query(query, conn)
            conn.close()
            
            self.logger.info(f"Loaded {len(df)} training history records")
            
        except Exception as e:
            self.logger.warning(f"Error loading training history: {e}")
    
    def _update_performance_tracking(self, results: List[TrainingResult]):
        """Update performance tracking for model improvement"""
        try:
            for result in results:
                model_type = result.model_type
                
                if model_type not in self.performance_tracking:
                    self.performance_tracking[model_type] = {
                        'scores': [],
                        'timestamps': [],
                        'best_score': 0.0,
                        'trend': 'stable'
                    }
                
                tracking = self.performance_tracking[model_type]
                tracking['scores'].append(result.metrics.f1_score)
                tracking['timestamps'].append(result.timestamp)
                tracking['best_score'] = max(tracking['best_score'], result.metrics.f1_score)
                
                # Analyze trend
                if len(tracking['scores']) >= 3:
                    recent_scores = tracking['scores'][-3:]
                    if all(recent_scores[i] > recent_scores[i-1] for i in range(1, len(recent_scores))):
                        tracking['trend'] = 'improving'
                    elif all(recent_scores[i] < recent_scores[i-1] for i in range(1, len(recent_scores))):
                        tracking['trend'] = 'declining'
                    else:
                        tracking['trend'] = 'stable'
                
                # Keep only recent history
                if len(tracking['scores']) > 100:
                    tracking['scores'] = tracking['scores'][-100:]
                    tracking['timestamps'] = tracking['timestamps'][-100:]
            
        except Exception as e:
            self.logger.warning(f"Error updating performance tracking: {e}")
    
    def _monitor_performance(self):
        """Background thread for performance monitoring"""
        while self.monitoring_active:
            try:
                time.sleep(3600)  # Check every hour
                
                # Check for performance drift
                for model_type, tracking in self.performance_tracking.items():
                    if tracking['trend'] == 'declining' and len(tracking['scores']) >= 5:
                        recent_avg = np.mean(tracking['scores'][-5:])
                        best_score = tracking['best_score']
                        
                        if (best_score - recent_avg) > self.config.retraining_threshold:
                            self.logger.warning(f"Performance drift detected for {model_type}. "
                                              f"Best: {best_score:.3f}, Recent: {recent_avg:.3f}")
                            # Trigger retraining notification
                            self._schedule_retraining(model_type)
                
            except Exception as e:
                self.logger.error(f"Error in performance monitoring: {e}")
                time.sleep(300)  # Wait 5 minutes before retrying
    
    def _schedule_retraining(self, model_type: str):
        """Schedule model retraining"""
        self.logger.info(f"Scheduling retraining for {model_type}")
        # This could trigger an automated retraining process
        # For now, just log the event
    
    def _hash_data(self, X: np.ndarray, y: np.ndarray) -> str:
        """Create hash of training data for versioning"""
        import hashlib
        data_string = f"{X.shape}_{y.shape}_{np.sum(X)}_{np.sum(y)}"
        return hashlib.md5(data_string.encode()).hexdigest()
    
    def _setup_gpu(self):
        """Setup GPU for TensorFlow"""
        try:
            if TF_AVAILABLE:
                gpus = tf.config.experimental.list_physical_devices('GPU')
                if gpus:
                    # Enable memory growth
                    for gpu in gpus:
                        tf.config.experimental.set_memory_growth(gpu, True)
                    self.logger.info(f"GPU setup completed. Found {len(gpus)} GPU(s)")
                else:
                    self.logger.info("No GPUs found. Using CPU.")
        except Exception as e:
            self.logger.warning(f"GPU setup failed: {e}")
    
    def get_best_models(self, top_k: int = 5) -> List[TrainingResult]:
        """Get top performing models"""
        if not self.training_history:
            return []
        
        sorted_models = sorted(self.training_history, 
                             key=lambda x: x.metrics.f1_score, 
                             reverse=True)
        
        return sorted_models[:top_k]
    
    def get_performance_summary(self) -> Dict[str, Any]:
        """Get performance summary across all models"""
        if not self.training_history:
            return {}
        
        summary = {
            'total_models_trained': len(self.training_history),
            'best_f1_score': max(r.metrics.f1_score for r in self.training_history),
            'average_f1_score': np.mean([r.metrics.f1_score for r in self.training_history]),
            'model_type_performance': {},
            'training_time_stats': {
                'total_hours': sum(r.metrics.training_time for r in self.training_history) / 3600,
                'average_minutes': np.mean([r.metrics.training_time for r in self.training_history]) / 60
            }
        }
        
        # Performance by model type
        for result in self.training_history:
            model_type = result.model_type
            if model_type not in summary['model_type_performance']:
                summary['model_type_performance'][model_type] = []
            summary['model_type_performance'][model_type].append(result.metrics.f1_score)
        
        # Average performance by type
        for model_type, scores in summary['model_type_performance'].items():
            summary['model_type_performance'][model_type] = {
                'average_f1': np.mean(scores),
                'best_f1': np.max(scores),
                'count': len(scores)
            }
        
        return summary
    
    def active_learning_iteration(self, 
                                 unlabeled_features: np.ndarray,
                                 oracle_function: Callable,
                                 budget: int = 100) -> Tuple[np.ndarray, np.ndarray]:
        """Perform one iteration of active learning"""
        
        if not self.training_history:
            raise ValueError("No trained models available for active learning")
        
        # Get best model
        best_model_result = max(self.training_history, key=lambda x: x.metrics.f1_score)
        
        # Load best model
        try:
            if best_model_result.model_type in self.deep_learning_architectures and TF_AVAILABLE:
                model = load_model(best_model_result.model_path)
            else:
                with open(best_model_result.model_path, 'rb') as f:
                    model = pickle.load(f)
        except Exception as e:
            self.logger.error(f"Failed to load model for active learning: {e}")
            return np.array([]), np.array([])
        
        # Get predictions and uncertainty
        if best_model_result.model_type in self.deep_learning_architectures:
            predictions = model.predict(unlabeled_features, verbose=0)
            uncertainties = self._calculate_prediction_uncertainty(predictions)
        else:
            if hasattr(model, 'predict_proba'):
                predictions = model.predict_proba(unlabeled_features)
                uncertainties = self._calculate_prediction_uncertainty(predictions)
            else:
                # Fallback to random sampling
                uncertainties = np.random.random(len(unlabeled_features))
        
        # Select samples with highest uncertainty
        query_indices = np.argsort(uncertainties)[-budget:]
        
        # Query oracle for labels
        query_features = unlabeled_features[query_indices]
        query_labels = []
        
        for features in query_features:
            label = oracle_function(features)
            query_labels.append(label)
        
        return query_features, np.array(query_labels)
    
    def _calculate_prediction_uncertainty(self, predictions: np.ndarray) -> np.ndarray:
        """Calculate prediction uncertainty"""
        if len(predictions.shape) == 1:
            # Binary classification
            uncertainties = np.abs(predictions - 0.5)  # Distance from decision boundary
        else:
            # Multi-class classification - use entropy
            uncertainties = -np.sum(predictions * np.log(predictions + 1e-8), axis=1)
        
        return uncertainties
    
    def shutdown(self):
        """Cleanup and shutdown"""
        self.monitoring_active = False
        if hasattr(self, 'monitoring_thread'):
            self.monitoring_thread.join(timeout=5)
        
        self.logger.info("Model trainer shutdown completed")

# Example usage and integration functions
def create_training_pipeline(config: TrainingConfiguration = None) -> AdvancedModelTrainer:
    """Create a complete training pipeline"""
    return AdvancedModelTrainer(config)

def train_pattern_models(features: np.ndarray, 
                        labels: np.ndarray,
                        config: TrainingConfiguration = None) -> List[TrainingResult]:
    """High-level function to train pattern recognition models"""
    
    trainer = create_training_pipeline(config)
    results = trainer.train_models(features, labels)
    
    # Return best models
    return trainer.get_best_models(top_k=3)

if __name__ == "__main__":
    # Example usage
    logging.basicConfig(level=logging.INFO)
    
    # Create sample data
    np.random.seed(42)
    X = np.random.randn(1000, 50)
    y = np.random.randint(0, 5, 1000)
    
    # Create configuration
    config = TrainingConfiguration(
        enable_hyperparameter_tuning=True,
        enable_ensemble=True,
        hyperparameter_search_iterations=50
    )
    
    # Train models
    trainer = AdvancedModelTrainer(config)
    results = trainer.train_models(X, y)
    
    # Print summary
    summary = trainer.get_performance_summary()
    print(f"Trained {summary['total_models_trained']} models")
    print(f"Best F1-score: {summary['best_f1_score']:.3f}")
    
    # Cleanup
    trainer.shutdown()
