"""
CatBoost Machine Learning Model with RFE Feature Selection
MySQL migration support
"""

import pandas as pd
import numpy as np
import joblib
import os
import time
from typing import Dict, List, Optional, Tuple, Any
from loguru import logger
from datetime import datetime, timedelta
import sqlite3

from catboost import CatBoostClassifier, CatBoostRegressor
from sklearn.feature_selection import RFE, RFECV
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import classification_report, accuracy_score, precision_recall_fscore_support

# Import database manager for MySQL migration
from src.database.db_manager import db_manager

from config.settings import *

class CatBoostTradingModel:
    """Advanced CatBoost model for cryptocurrency trading predictions"""
    
    def __init__(self):
        self.model = None
        self.rfe_selector = None
        self.label_encoder = None
        self.selected_features = []
        self.feature_importance = {}
        self.feature_weights = {}
        self.model_performance = {}
        self.training_progress = 0
        self.is_trained = False
        self.model_path = "models/"
        self.db_path = DATABASE_URL.replace("sqlite:///", "")
        
        # Training control and fallback (MySQL migration)
        self.last_training_attempt = None
        self.training_cooldown_until = None
        self.prediction_warning_counter = 0  # Reduce warning spam
        
        # Training diagnostics and metadata (Follow-up fixes)
        self.last_training_error = None
        self.last_training_time = None
        self.class_distribution = {}
        self.selected_feature_count = 0
        self.numeric_feature_count = 0
        self.last_sanitization_stats = {}
        self.next_retry_at = None
        self.class_weights = {}
        
        # Complete Pipeline Restructure additions
        self.training_stages = [
            'sanitizing',
            'building', 
            'feature_selection',
            'fitting',
            'validating',
            'finalizing'
        ]
        self.current_training_stage = None
        self.stage_progress = 0.0
        self.overall_training_progress = 0.0
        
        # Sliding window accuracy tracking
        self.recent_predictions = []  # Store recent (prediction, actual) tuples
        self.accuracy_window_size = ACCURACY_SLIDING_WINDOW
        self.accuracy_live = 0.0
        self.model_version = 1
        self.training_count = 0
        
        # Model preservation for OHLCV-only mode
        self.model_prev = None  # Previous model backup
        self.model_prev_performance = None  # Previous model metrics
        self.model_prev_features = None  # Previous model selected features
        
        # Initialize feature weights (active=1.0, inactive=0.01)
        self.active_weight = ACTIVE_FEATURE_WEIGHT
        self.inactive_weight = INACTIVE_FEATURE_WEIGHT
        
    async def initialize(self):
        """Initialize the ML model"""
        try:
            logger.info("Initializing CatBoost ML model...")
            
            # Create models directory
            os.makedirs(self.model_path, exist_ok=True)
            
            # Initialize CatBoost model with high performance settings
            # Note: Bayesian bootstrap is incompatible with subsample parameter
            self.model = CatBoostClassifier(
                iterations=CATBOOST_ITERATIONS,
                depth=CATBOOST_DEPTH,
                learning_rate=CATBOOST_LEARNING_RATE,
                loss_function='MultiClass',
                eval_metric='MultiClass',
                random_seed=42,
                logging_level='Silent',
                bootstrap_type='Bayesian',
                bagging_temperature=1.0,
                # subsample removed - incompatible with Bayesian bootstrap
                colsample_bylevel=0.8,
                reg_lambda=3.0,
                thread_count=-1,
                use_best_model=True,
                early_stopping_rounds=50
            )
            
            # Initialize label encoder
            self.label_encoder = LabelEncoder()
            
            # Load existing model if available
            await self.load_model()
            
            logger.success("CatBoost ML model initialized")
            
        except Exception as e:
            logger.error(f"Failed to initialize ML model: {e}")
            raise
    
    def _sanitize_features(self, feature_df: pd.DataFrame) -> tuple:
        """Sanitize features and return both sanitized data and metadata
        
        Returns:
            tuple: (sanitized_dataframe, metadata_dict)
        """
        try:
            logger.info("[TRAIN] Starting feature sanitization...")
            sanitized_df = feature_df.copy()
            initial_feature_count = len(sanitized_df.columns)
            dropped_features = []
            converted_features = []
            
            for column in feature_df.columns:
                col_data = sanitized_df[column]
                
                # Check if column is already numeric
                if pd.api.types.is_numeric_dtype(col_data):
                    continue
                
                # Handle object/string columns
                if col_data.dtype == 'object':
                    unique_values = col_data.dropna().unique()
                    unique_count = len(unique_values)
                    
                    logger.info(f"[TRAIN] Found non-numeric column '{column}' with {unique_count} unique values")
                    
                    # Special handling for symbol column - encode or drop
                    if column.lower() == 'symbol':
                        logger.info(f"[TRAIN] Processing symbol column: {column}")
                        if unique_count <= 1:
                            # Constant symbol - drop it as it provides no information
                            logger.info(f"[TRAIN] Dropping constant symbol column: {column} (value: {unique_values[0] if len(unique_values) > 0 else 'N/A'})")
                            sanitized_df = sanitized_df.drop(columns=[column])
                            dropped_features.append(f"{column} (constant_symbol)")
                        elif unique_count <= 50:  # Reasonable number of symbols for encoding
                            # Encode symbol to numerical codes
                            from sklearn.preprocessing import LabelEncoder
                            le = LabelEncoder()
                            symbol_codes = le.fit_transform(col_data.astype(str))
                            sanitized_df[f'{column}_code'] = symbol_codes
                            sanitized_df = sanitized_df.drop(columns=[column])
                            converted_features.append(f"{column} → {column}_code (symbol_encoded)")
                            logger.info(f"[TRAIN] Encoded symbol column {column} to {column}_code with {unique_count} categories")
                        else:
                            # Too many symbols - drop it
                            logger.info(f"[TRAIN] Dropping high-cardinality symbol column: {column} ({unique_count} symbols)")
                            sanitized_df = sanitized_df.drop(columns=[column])
                            dropped_features.append(f"{column} (high_cardinality_symbol)")
                        continue
                    
                    # Drop constant columns (only one unique value)
                    if unique_count <= 1:
                        logger.info(f"[TRAIN] Dropping constant column: {column}")
                        sanitized_df = sanitized_df.drop(columns=[column])
                        dropped_features.append(column)
                        continue
                    
                    # For categorical columns with few unique values, try label encoding
                    elif unique_count <= 10:
                        try:
                            # Attempt to convert to numeric first
                            numeric_col = pd.to_numeric(col_data, errors='coerce')
                            if not numeric_col.isna().all():
                                sanitized_df[column] = numeric_col
                                converted_features.append(f"{column} (to_numeric)")
                                continue
                            
                            # If conversion fails, use label encoding for low-cardinality categorical
                            from sklearn.preprocessing import LabelEncoder
                            le = LabelEncoder()
                            sanitized_df[column] = le.fit_transform(col_data.astype(str))
                            converted_features.append(f"{column} (label_encoded)")
                            
                        except Exception as e:
                            logger.warning(f"[TRAIN] Could not convert column '{column}': {e}")
                            sanitized_df = sanitized_df.drop(columns=[column])
                            dropped_features.append(column)
                    
                    # Drop high-cardinality categorical columns
                    else:
                        logger.info(f"[TRAIN] Dropping high-cardinality column: {column} ({unique_count} unique values)")
                        sanitized_df = sanitized_df.drop(columns=[column])
                        dropped_features.append(column)
                
                # Handle other non-numeric types
                else:
                    try:
                        # Try to convert to numeric
                        sanitized_df[column] = pd.to_numeric(col_data, errors='coerce')
                        converted_features.append(f"{column} (forced_numeric)")
                    except:
                        logger.warning(f"[TRAIN] Dropping unconvertible column: {column} (dtype: {col_data.dtype})")
                        sanitized_df = sanitized_df.drop(columns=[column])
                        dropped_features.append(column)
            
            # Final check: remove any columns that became all NaN after conversion
            for column in sanitized_df.columns:
                if sanitized_df[column].isna().all():
                    logger.info(f"[TRAIN] Dropping all-NaN column after conversion: {column}")
                    sanitized_df = sanitized_df.drop(columns=[column])
                    dropped_features.append(column)
            
            # Remove any remaining rows with NaN values
            rows_before = len(sanitized_df)
            sanitized_df = sanitized_df.dropna()
            rows_after = len(sanitized_df)
            
            # Log sanitization results
            final_feature_count = len(sanitized_df.columns)
            logger.info(f"[TRAIN] Feature sanitization completed:")
            logger.info(f"[TRAIN]   Features: {initial_feature_count} → {final_feature_count}")
            logger.info(f"[TRAIN]   Samples: {rows_before} → {rows_after}")
            
            if dropped_features:
                logger.info(f"[TRAIN]   Dropped features ({len(dropped_features)}): {', '.join(dropped_features[:5])}" + 
                          (f" and {len(dropped_features)-5} more" if len(dropped_features) > 5 else ""))
            
            if converted_features:
                logger.info(f"[TRAIN]   Converted features ({len(converted_features)}): {', '.join(converted_features[:3])}" + 
                          (f" and {len(converted_features)-3} more" if len(converted_features) > 3 else ""))
            
            # Create metadata dict
            metadata = {
                'initial_feature_count': initial_feature_count,
                'final_feature_count': final_feature_count,
                'dropped_constant': len([f for f in dropped_features if 'constant' in str(f)]),
                'dropped_high_cardinality': len([f for f in dropped_features if 'high-cardinality' in str(f) or 'high_cardinality' in str(f)]),
                'encoded_categorical': len([f for f in converted_features if 'label_encoded' in f or 'symbol_encoded' in f]),
                'converted_numeric': len([f for f in converted_features if 'to_numeric' in f or 'forced_numeric' in f]),
                'samples_before': rows_before,
                'samples_after': rows_after
            }
            
            # Store sanitization stats for reporting
            self.last_sanitization_stats = metadata
            
            return sanitized_df, metadata
            
        except Exception as e:
            logger.error(f"[TRAIN] Feature sanitization failed: {e}")
            # Return original dataframe as fallback with empty metadata
            return feature_df, {
                'initial_feature_count': len(feature_df.columns),
                'final_feature_count': len(feature_df.columns), 
                'error': str(e)
            }
    
    async def prepare_features_and_labels(self, data: Dict[str, Any], symbol: str) -> Tuple[pd.DataFrame, pd.Series]:
        """Prepare features and labels from indicator data (MySQL migration improved)"""
        try:
            logger.info("[TRAIN] Preparing features and labels...")
            
            # Convert indicators to DataFrame
            # Collect valid indicators first to avoid DataFrame fragmentation
            valid_indicators = {}
            for indicator_name, values in data.items():
                if isinstance(values, (pd.Series, list, np.ndarray)):
                    valid_indicators[indicator_name] = values
            
            if not valid_indicators:
                logger.error("[TRAIN] No valid indicators found")
                return pd.DataFrame(), pd.Series()
            
            # Create DataFrame all at once to avoid fragmentation warning
            feature_df = pd.DataFrame(valid_indicators)
            raw_feature_count = len(feature_df.columns)
            raw_sample_count = len(feature_df)
            
            logger.info(f"[TRAIN] Raw features: {raw_feature_count}, samples: {raw_sample_count}")
            
            # Remove rows with NaN values
            feature_df = feature_df.dropna()
            clean_sample_count = len(feature_df)
            
            if clean_sample_count == 0:
                logger.warning("[TRAIN] No valid feature data after cleaning")
                return pd.DataFrame(), pd.Series()
            
            logger.info(f"[TRAIN] Samples after dropna: {clean_sample_count} (dropped {raw_sample_count - clean_sample_count})")
            
            # Robust feature sanitization to handle non-numeric columns
            feature_df, sanitization_stats = self._sanitize_features(feature_df)
            self.last_sanitization_stats = sanitization_stats  # Store for diagnostics
            
            if len(feature_df.columns) == 0:
                logger.error("[TRAIN] No valid features after sanitization")
                return pd.DataFrame(), pd.Series()
            
            # Generate labels based on future price movement
            labels = await self.generate_labels(feature_df, symbol)
            
            # Align features and labels
            min_length = min(len(feature_df), len(labels))
            if min_length != len(feature_df):
                logger.info(f"[TRAIN] Aligning data length to {min_length}")
            
            feature_df = feature_df.iloc[:min_length]
            labels = labels.iloc[:min_length]
            
            # Log class distribution (MySQL migration)
            if len(labels) > 0:
                class_dist = labels.value_counts().sort_index()
                logger.info(f"[TRAIN] Class distribution - SELL: {class_dist.get(0, 0)}, HOLD: {class_dist.get(1, 0)}, BUY: {class_dist.get(2, 0)}")
            
            logger.info(f"[TRAIN] Final dataset: {len(feature_df)} samples with {len(feature_df.columns)} features")
            
            return feature_df, labels
            
        except Exception as e:
            logger.error(f"[TRAIN] Failed to prepare features and labels: {e}")
            return pd.DataFrame(), pd.Series()
    
    async def generate_labels(self, feature_df: pd.DataFrame, symbol: str) -> pd.Series:
        """Generate trading labels (BUY=2, HOLD=1, SELL=0) based on future returns (MySQL migration)"""
        try:
            # Get price data from database using db_manager
            if db_manager.backend == 'mysql':
                query = '''
                    SELECT close, timestamp 
                    FROM market_data 
                    WHERE symbol = %s AND timeframe = %s 
                    ORDER BY timestamp ASC
                '''
            else:
                query = '''
                    SELECT close, timestamp 
                    FROM market_data 
                    WHERE symbol = ? AND timeframe = ? 
                    ORDER BY timestamp ASC
                '''
            
            conn = db_manager.get_pandas_connection()
            price_df = pd.read_sql_query(query, conn, params=(symbol, DEFAULT_TIMEFRAME))
            conn.close()
            
            if len(price_df) == 0:
                logger.warning(f"No price data found for {symbol}")
                return pd.Series([1] * len(feature_df))  # Default to HOLD
            
            # Align feature_df length to price_df (MySQL migration improvement)
            if len(price_df) < len(feature_df):
                logger.warning(f"Price data ({len(price_df)}) shorter than features ({len(feature_df)})")
                feature_df = feature_df.iloc[:len(price_df)]
            
            # Calculate future returns (looking ahead 1-3 periods)
            price_df['future_return_1'] = price_df['close'].pct_change(1).shift(-1)
            price_df['future_return_3'] = price_df['close'].pct_change(3).shift(-3)
            
            # Guard against empty future returns (MySQL migration improvement)
            valid_returns = price_df[['future_return_1', 'future_return_3']].dropna()
            if len(valid_returns) < 10:
                logger.warning(f"Insufficient future return data ({len(valid_returns)} rows), using HOLD labels")
                return pd.Series([1] * len(feature_df))
            
            # Define thresholds for buy/sell signals
            buy_threshold = 0.02   # 2% gain
            sell_threshold = -0.02  # 2% loss
            
            labels = []
            for i in range(len(feature_df)):
                if i < len(price_df):
                    ret_1 = price_df['future_return_1'].iloc[i] if not pd.isna(price_df['future_return_1'].iloc[i]) else 0
                    ret_3 = price_df['future_return_3'].iloc[i] if not pd.isna(price_df['future_return_3'].iloc[i]) else 0
                    
                    # Strong positive signal
                    if ret_1 > buy_threshold or ret_3 > buy_threshold * 1.5:
                        labels.append(2)  # BUY
                    # Strong negative signal
                    elif ret_1 < sell_threshold or ret_3 < sell_threshold * 1.5:
                        labels.append(0)  # SELL
                    else:
                        labels.append(1)  # HOLD
                else:
                    labels.append(1)  # Default HOLD
            
            return pd.Series(labels[:len(feature_df)])
            
        except Exception as e:
            logger.error(f"Failed to generate labels: {e}")
            return pd.Series([1] * len(feature_df))  # Default to HOLD
    
    async def perform_rfe_selection(self, X: pd.DataFrame, y: pd.Series, rfe_eligible_features: List[str], X_recent: Optional[pd.DataFrame] = None, y_recent: Optional[pd.Series] = None) -> List[str]:
        """Perform Recursive Feature Elimination on eligible features (OHLCV-only mode with recent window)"""
        try:
            logger.info("[TRAIN] Performing RFE feature selection (OHLCV-only mode)...")
            self.training_progress = 10
            
            # Import RFE settings for OHLCV-only mode
            try:
                from config.settings import RFE_SELECTION_CANDLES, BASE_MUST_KEEP_FEATURES
                rfe_window_size = RFE_SELECTION_CANDLES
                base_must_keep = BASE_MUST_KEEP_FEATURES
            except ImportError:
                rfe_window_size = 1000
                base_must_keep = ["open", "high", "low", "close", "volume"]
            
            # Use recent window data if provided, otherwise slice from full dataset
            if X_recent is not None and y_recent is not None:
                X_rfe_window = X_recent
                y_rfe_window = y_recent
            else:
                # Use most recent RFE_SELECTION_CANDLES from full dataset
                window_size = min(rfe_window_size, len(X))
                X_rfe_window = X.tail(window_size)
                y_rfe_window = y.tail(window_size)
            
            logger.info(f"[TRAIN] RFE window: {len(X_rfe_window)} samples (target: {rfe_window_size})")
            
            # Sanitize RFE window features to handle symbol encoding and other issues
            logger.info("[TRAIN] Sanitizing RFE window features...")
            X_rfe_window, rfe_sanitization_stats = self._sanitize_features(X_rfe_window)
            logger.info(f"[TRAIN] RFE sanitization: {rfe_sanitization_stats['initial_feature_count']} → {rfe_sanitization_stats['final_feature_count']} features")
            
            # Guard: Check minimum samples for RFE
            if len(X_rfe_window) < MIN_RFE_SAMPLES:
                logger.warning(f"[TRAIN] Insufficient samples for RFE ({len(X_rfe_window)} < {MIN_RFE_SAMPLES}), using baseline feature selection")
                return await self.select_features_by_importance(X, y, rfe_eligible_features)
            
            # Filter to only RFE-eligible features (exclude BASE_MUST_KEEP_FEATURES) 
            # Note: Use sanitized column names after sanitization
            rfe_features = [col for col in X_rfe_window.columns if col in rfe_eligible_features and col not in base_must_keep]
            
            if len(rfe_features) == 0:
                logger.warning("[TRAIN] No RFE-eligible features found")
                must_keep_features = [col for col in X.columns if col not in rfe_eligible_features or col in base_must_keep]
                return must_keep_features
            
            X_rfe_data = X_rfe_window[rfe_features].copy()
            
            # Create dedicated train directory for RFE CatBoost temporary models
            rfe_train_dir = os.path.join(self.model_path, "catboost_rfe_tmp")
            os.makedirs(rfe_train_dir, exist_ok=True)
            
            # Use a simpler model for RFE to speed up the process
            estimator = CatBoostClassifier(
                iterations=100,
                depth=6,
                learning_rate=0.1,
                logging_level='Silent',
                random_seed=42,
                train_dir=rfe_train_dir,
                allow_writing_files=True,
                verbose=False
            )
            
            # Perform RFE with cross-validation (target 25 features)
            target_features = min(RFE_N_FEATURES, len(rfe_features))
            rfe = RFECV(
                estimator=estimator,
                step=max(1, len(rfe_features) // 20),  # Remove features progressively
                cv=3,     # 3-fold cross-validation
                scoring='accuracy',
                n_jobs=-1,
                min_features_to_select=target_features
            )
            
            self.training_progress = 30
            
            rfe.fit(X_rfe_data, y_rfe_window)
            
            # Get selected RFE features
            selected_rfe_features = X_rfe_data.columns[rfe.support_].tolist()
            
            # Add back must-keep features (BASE_MUST_KEEP_FEATURES + other required)
            must_keep_features = [col for col in X.columns if col not in rfe_eligible_features or col in base_must_keep]
            all_selected_features = selected_rfe_features + must_keep_features
            
            # Remove duplicates while preserving order
            final_features = []
            seen = set()
            for feature in all_selected_features:
                if feature not in seen and feature in X.columns:
                    final_features.append(feature)
                    seen.add(feature)
            
            logger.info(f"[TRAIN] RFE selected {len(selected_rfe_features)} features out of {len(rfe_features)} eligible")
            logger.info(f"[TRAIN] Must-keep features: {len(must_keep_features)} (including {len(base_must_keep)} base OHLCV)")
            logger.info(f"[TRAIN] Final feature count: {len(final_features)} (target: {RFE_N_FEATURES + len(base_must_keep)})")
            
            # Store RFE results and selected features
            self.rfe_selector = rfe
            self.selected_features = final_features
            
            self.training_progress = 40
            
            return final_features
            
        except Exception as e:
            logger.error(f"[TRAIN] RFE selection failed: {e}")
            # Fallback: select top features by importance
            return await self.select_features_by_importance(X, y, rfe_eligible_features)
    
    async def select_features_by_importance(self, X: pd.DataFrame, y: pd.Series, rfe_eligible_features: List[str]) -> List[str]:
        """Fallback feature selection based on feature importance (MySQL migration improved)"""
        try:
            logger.info("[TRAIN] Using feature importance for selection...")
            
            # Sanitize features if not already sanitized
            logger.info("[TRAIN] Sanitizing features for importance selection...")
            X_sanitized, importance_sanitization_stats = self._sanitize_features(X)
            logger.info(f"[TRAIN] Importance sanitization: {importance_sanitization_stats['initial_feature_count']} → {importance_sanitization_stats['final_feature_count']} features")
            
            # Cap features to MAX_BASELINE_FEATURES when RFE is skipped (MySQL migration)
            max_features = min(MAX_BASELINE_FEATURES, len(rfe_eligible_features))
            
            # Create dedicated train directory for importance CatBoost temporary models
            importance_train_dir = os.path.join(self.model_path, "catboost_importance_tmp")
            os.makedirs(importance_train_dir, exist_ok=True)
            
            # Train a quick model to get feature importance
            temp_model = CatBoostClassifier(
                iterations=100,
                depth=4,
                learning_rate=0.1,
                logging_level='Silent',
                random_seed=42,
                train_dir=importance_train_dir,
                allow_writing_files=True,
                verbose=False
            )
            
            temp_model.fit(X_sanitized, y)
            
            # Get feature importance
            importance_df = pd.DataFrame({
                'feature': X_sanitized.columns,
                'importance': temp_model.get_feature_importance()
            }).sort_values('importance', ascending=False)
            
            # Select top features (use variance proxy if many features) (MySQL migration)
            rfe_features = importance_df[
                importance_df['feature'].isin(rfe_eligible_features)
            ].head(max_features)['feature'].tolist()
            
            # Add required features (from sanitized columns)
            required_features = [col for col in X_sanitized.columns if col not in rfe_eligible_features]
            all_selected_features = rfe_features + required_features
            
            logger.info(f"[TRAIN] Selected {len(rfe_features)} top important features (max {max_features})")
            
            return all_selected_features
            
        except Exception as e:
            logger.error(f"[TRAIN] Feature importance selection failed: {e}")
            # Try to return sanitized features if available, otherwise original features
            try:
                X_sanitized, _ = self._sanitize_features(X)
                return list(X_sanitized.columns)
            except:
                return list(X.columns)  # Return all original features as final fallback
    
    def setup_feature_weights(self, selected_features: List[str], all_features: List[str]):
        """Setup feature weights (active vs inactive)"""
        self.feature_weights = {}
        
        for feature in all_features:
            if feature in selected_features:
                self.feature_weights[feature] = self.active_weight
            else:
                self.feature_weights[feature] = self.inactive_weight
    
    def preserve_current_model(self):
        """Preserve current model before retraining (OHLCV-only mode)"""
        try:
            if self.model is not None and self.is_trained:
                # Deep copy current model and metadata
                import copy
                self.model_prev = copy.deepcopy(self.model)
                self.model_prev_performance = copy.deepcopy(self.model_performance)
                self.model_prev_features = copy.deepcopy(self.selected_features)
                logger.info(f"[TRAIN] Previous model preserved (v{self.model_version}, acc: {self.model_prev_performance.get('accuracy', 0.0):.4f})")
            else:
                logger.debug("[TRAIN] No trained model to preserve")
        except Exception as e:
            logger.error(f"[TRAIN] Failed to preserve current model: {e}")
    
    def revert_to_previous_model(self):
        """Revert to previous model if new training underperforms (OHLCV-only mode)"""
        try:
            if self.model_prev is not None and self.model_prev_performance is not None:
                self.model = self.model_prev
                self.model_performance = self.model_prev_performance
                self.selected_features = self.model_prev_features or []
                self.is_trained = True
                
                prev_accuracy = self.model_prev_performance.get('accuracy', 0.0)
                logger.warning(f"[TRAIN] Reverted to previous model (acc: {prev_accuracy:.4f})")
                logger.info(f"[TRAIN] Previous model features count: {len(self.selected_features)}")
                return True
            else:
                logger.error("[TRAIN] No previous model available for reversion")
                return False
        except Exception as e:
            logger.error(f"[TRAIN] Failed to revert to previous model: {e}")
            return False
    
    def should_keep_new_model(self, new_accuracy: float) -> bool:
        """Determine if new model should be kept based on performance (OHLCV-only mode)"""
        if self.model_prev_performance is None:
            # No previous model, keep new one regardless
            return True
        
        prev_accuracy = self.model_prev_performance.get('accuracy', 0.0)
        
        # Keep new model if it's better or within reasonable threshold
        improvement_threshold = 0.01  # Allow small degradation for feature reduction benefits
        
        if new_accuracy >= (prev_accuracy - improvement_threshold):
            logger.info(f"[TRAIN] New model performance acceptable: {new_accuracy:.4f} vs {prev_accuracy:.4f}")
            return True
        else:
            logger.warning(f"[TRAIN] New model underperforms: {new_accuracy:.4f} vs {prev_accuracy:.4f}")
            return False
        
        logger.info(f"Setup weights: {len(selected_features)} active, {len(all_features) - len(selected_features)} inactive")
    
    async def train_model(self, X: pd.DataFrame, y: pd.Series, rfe_eligible_features: List[str], X_recent: Optional[pd.DataFrame] = None, y_recent: Optional[pd.Series] = None):
        """Train the CatBoost model with staged progress and model preservation (OHLCV-only mode)"""
        try:
            logger.info("[TRAIN] Starting model training with staged progress...")
            
            # Preserve current model before retraining (OHLCV-only mode)
            self.preserve_current_model()
            
            # Initialize training progress
            self.overall_training_progress = 0.0
            self.current_training_stage = None
            self.last_training_error = None
            self.training_count += 1
            training_start_time = datetime.now()
            
            if len(X) == 0 or len(y) == 0:
                error_msg = "No training data available"
                logger.error(f"[TRAIN] {error_msg}")
                self.last_training_error = error_msg
                return False
            
            # Stage 1: Sanitizing (0-15%)
            self.current_training_stage = 'sanitizing'
            logger.info("[TRAIN] Stage 1/6: Data sanitization...")
            
            # Sanitize main training features early
            logger.info("[TRAIN] Sanitizing main training features...")
            X_sanitized, main_sanitization_stats = self._sanitize_features(X)
            logger.info(f"[TRAIN] Main sanitization: {main_sanitization_stats['initial_feature_count']} → {main_sanitization_stats['final_feature_count']} features")
            
            # Sanitize recent features if provided
            X_recent_sanitized = None
            if X_recent is not None:
                logger.info("[TRAIN] Sanitizing recent training features...")
                X_recent_sanitized, recent_sanitization_stats = self._sanitize_features(X_recent)
                logger.info(f"[TRAIN] Recent sanitization: {recent_sanitization_stats['initial_feature_count']} → {recent_sanitization_stats['final_feature_count']} features")
            
            # Encode labels and calculate class distribution
            y_encoded = self.label_encoder.fit_transform(y)
            
            # Calculate class distribution for diagnostics
            unique_classes, class_counts = np.unique(y, return_counts=True)
            self.class_distribution = dict(zip(unique_classes, class_counts.tolist()))
            
            # Calculate class weights (inverse frequency)
            total_samples = len(y)
            n_classes = len(unique_classes)
            self.class_weights = {}
            for i, class_name in enumerate(unique_classes):
                class_weight = total_samples / (n_classes * class_counts[i])
                self.class_weights[class_name] = class_weight
            
            logger.info(f"[TRAIN] Class distribution: {self.class_distribution}")
            logger.info(f"[TRAIN] Class weights: {self.class_weights}")
            self.overall_training_progress = 15.0
            
            # Stage 2: Building (15-30%)
            self.current_training_stage = 'building'
            logger.info("[TRAIN] Stage 2/6: Building training/test split...")
            
            # Split data
            X_train, X_test, y_train, y_test = train_test_split(
                X_sanitized, y_encoded, test_size=0.2, random_state=42, stratify=y_encoded
            )
            
            logger.info(f"[TRAIN] Train samples: {len(X_train)}, Test samples: {len(X_test)}")
            self.overall_training_progress = 30.0
            
            # Stage 3: Feature Selection (30-50%)
            self.current_training_stage = 'feature_selection'
            logger.info("[TRAIN] Stage 3/6: Feature selection...")
            
            # Perform RFE feature selection with progress tracking (using sanitized data)
            selected_features = await self.perform_rfe_selection(X_train, y_train, rfe_eligible_features, X_recent_sanitized, y_recent)
            self.selected_features = selected_features
            self.selected_feature_count = len(selected_features)
            
            # Setup feature weights (use sanitized column names)
            self.setup_feature_weights(selected_features, list(X_sanitized.columns))
            
            # Apply feature selection
            X_train_selected = X_train[selected_features]
            X_test_selected = X_test[selected_features]
            
            logger.info(f"[TRAIN] Selected {len(selected_features)} features")
            self.overall_training_progress = 50.0
            
            # Stage 4: Fitting (50-80%)
            self.current_training_stage = 'fitting'
            logger.info("[TRAIN] Stage 4/6: Model fitting...")
            
            # Train the main model with class weights
            logger.info("[TRAIN] Training main CatBoost model...")
            
            # Convert class weights to CatBoost format - fix class_weights parameter issue
            # CatBoost expects class_weights parameter in constructor, not fit method
            catboost_class_weights = {}
            for encoded_label in range(len(unique_classes)):
                original_label = self.label_encoder.inverse_transform([encoded_label])[0]
                catboost_class_weights[encoded_label] = self.class_weights[original_label]
            
            logger.info(f"[TRAIN] CatBoost class weights: {catboost_class_weights}")
            
            # Create a new model instance with class weights in constructor
            self.model = CatBoostClassifier(
                iterations=CATBOOST_ITERATIONS,
                depth=CATBOOST_DEPTH,
                learning_rate=CATBOOST_LEARNING_RATE,
                loss_function='MultiClass',
                eval_metric='MultiClass',
                random_seed=42,
                logging_level='Silent',
                bootstrap_type='Bayesian',
                bagging_temperature=1.0,
                colsample_bylevel=0.8,
                reg_lambda=3.0,
                thread_count=-1,
                use_best_model=True,
                early_stopping_rounds=50,
                class_weights=catboost_class_weights  # Set in constructor
            )
            
            # Use validation set for early stopping
            X_train_final, X_val, y_train_final, y_val = train_test_split(
                X_train_selected, y_train, test_size=0.2, random_state=42
            )
            
            logger.info(f"[TRAIN] Fitting model with {len(X_train_final)} training samples...")
            self.model.fit(
                X_train_final, y_train_final,
                eval_set=(X_val, y_val),
                verbose=False,
                plot=False
                # class_weights removed from fit() call - now in constructor
            )
            
            self.overall_training_progress = 80.0
            
            # Stage 5: Validating (80-95%)
            self.current_training_stage = 'validating'
            logger.info("[TRAIN] Stage 5/6: Model validation...")
            
            # Evaluate model
            y_pred = self.model.predict(X_test_selected)
            accuracy = accuracy_score(y_test, y_pred)
            precision, recall, f1, support = precision_recall_fscore_support(y_test, y_pred, average='weighted')
            
            self.model_performance = {
                'accuracy': accuracy,
                'precision': precision,
                'recall': recall,
                'f1_score': f1,
                'training_samples': len(X_train),
                'test_samples': len(X_test),
                'selected_features': len(selected_features),
                'training_date': datetime.now().isoformat(),
                'class_distribution': self.class_distribution,
                'class_weights': self.class_weights
            }
            
            # Initialize live accuracy with training accuracy
            self.accuracy_live = accuracy
            
            # Store additional diagnostics
            self.selected_feature_count = len(selected_features)
            self.numeric_feature_count = len(X.columns)
            self.last_training_time = datetime.now()
            
            # Get feature importance
            feature_names = X_train_selected.columns
            importance_scores = self.model.get_feature_importance()
            self.feature_importance = dict(zip(feature_names, importance_scores))
            
            logger.info(f"[TRAIN] Validation results: Accuracy: {accuracy:.4f}, F1: {f1:.4f}")
            self.overall_training_progress = 95.0
            
            # Model preservation decision (OHLCV-only mode)
            if not self.should_keep_new_model(accuracy):
                logger.warning("[TRAIN] New model underperforms, reverting to previous model")
                if self.revert_to_previous_model():
                    # Update progress and return success with reverted model
                    self.current_training_stage = 'finalizing'
                    self.overall_training_progress = 100.0
                    logger.info(f"[TRAIN] Reverted to previous model (v{self.model_version})")
                    return True
                else:
                    logger.error("[TRAIN] Reversion failed, keeping new model despite poor performance")
            else:
                logger.info("[TRAIN] New model performance acceptable, keeping new model")
                # Clear previous model to save memory
                self.model_prev = None
                self.model_prev_performance = None
                self.model_prev_features = None
            
            # Stage 6: Finalizing (95-100%)
            self.current_training_stage = 'finalizing'
            logger.info("[TRAIN] Stage 6/6: Finalizing...")
            
            # Save model
            await self.save_model()
            
            # Update model version
            self.model_version += 1
            self.is_trained = True
            self.current_training_stage = None
            self.overall_training_progress = 100.0
            
            training_duration = (datetime.now() - training_start_time).total_seconds()
            
            logger.success(f"[TRAIN] Training completed successfully in {training_duration:.1f}s")
            logger.info(f"[TRAIN] Model v{self.model_version}: Accuracy: {accuracy:.4f}, F1: {f1:.4f}")
            logger.info(f"[TRAIN] Features: {len(selected_features)}/{len(X.columns)}, Samples: {len(X_train)}")
            
            return True
            
        except Exception as e:
            error_msg = f"Model training failed: {e}"
            logger.error(f"[TRAIN] {error_msg}")
            self.last_training_error = error_msg
            self.current_training_stage = None
            self.overall_training_progress = 0.0
            return False
    
    async def predict(self, X: pd.DataFrame) -> Dict[str, Any]:
        """Make predictions with confidence scores"""
        try:
            if not self.is_trained or self.model is None:
                # Reduce warning spam by logging warning only every N predictions (MySQL migration)
                self.prediction_warning_counter += 1
                if self.prediction_warning_counter % 10 == 1:  # Log every 10th warning
                    logger.warning(f"Model is not trained (warning #{self.prediction_warning_counter})")
                
                return {
                    'prediction': 'HOLD',
                    'confidence': 0.0,
                    'probabilities': {'SELL': 0.33, 'HOLD': 0.34, 'BUY': 0.33}
                }
            
            # Apply feature weights and selection
            X_processed = self.apply_feature_weights(X)
            
            # Select only trained features
            if self.selected_features:
                available_features = [f for f in self.selected_features if f in X_processed.columns]
                
                # Log once if missing > 20% of expected features (MySQL migration)
                missing_pct = (len(self.selected_features) - len(available_features)) / len(self.selected_features)
                if missing_pct > 0.2:
                    if not hasattr(self, '_missing_features_warned'):
                        logger.warning(f"Missing {missing_pct:.1%} of expected features ({len(self.selected_features) - len(available_features)}/{len(self.selected_features)})")
                        self._missing_features_warned = True
                
                if available_features:
                    X_processed = X_processed[available_features]
                else:
                    logger.warning("No trained features available")
                    return {
                        'prediction': 'HOLD',
                        'confidence': 0.0,
                        'probabilities': {'SELL': 0.33, 'HOLD': 0.34, 'BUY': 0.33}
                    }
            
            # Make prediction
            prediction_proba = self.model.predict_proba(X_processed)
            prediction = self.model.predict(X_processed)
            
            # Convert back to labels
            prediction_label = self.label_encoder.inverse_transform(prediction)[0]
            
            # Get class probabilities
            class_names = self.label_encoder.classes_
            probabilities = dict(zip(
                ['SELL', 'HOLD', 'BUY'] if len(class_names) == 3 else [str(c) for c in class_names],
                prediction_proba[0]
            ))
            
            # Calculate confidence as the maximum probability
            confidence = float(np.max(prediction_proba[0]) * 100)
            
            # Map numeric labels to string labels
            label_mapping = {0: 'SELL', 1: 'HOLD', 2: 'BUY'}
            final_prediction = label_mapping.get(prediction_label, 'HOLD')
            
            return {
                'prediction': final_prediction,
                'confidence': confidence,
                'probabilities': probabilities
            }
            
        except Exception as e:
            logger.error(f"Prediction failed: {e}")
            return {
                'prediction': 'HOLD',
                'confidence': 0.0,
                'probabilities': {'SELL': 0.33, 'HOLD': 0.34, 'BUY': 0.33}
            }
    
    def apply_feature_weights(self, X: pd.DataFrame) -> pd.DataFrame:
        """Apply feature weights (active vs inactive features)"""
        X_weighted = X.copy()
        
        for feature in X.columns:
            weight = self.feature_weights.get(feature, self.active_weight)
            if weight != 1.0:  # Only apply if weight is different from 1.0
                X_weighted[feature] = X_weighted[feature] * weight
        
        return X_weighted
    
    async def retrain_online(self, new_data: Dict[str, Any], symbol: str, rfe_eligible_features: List[str]):
        """Online learning with new market data"""
        try:
            logger.info("Starting online retraining...")
            
            # Prepare new features and labels
            X_new, y_new = await self.prepare_features_and_labels(new_data, symbol)
            
            if len(X_new) == 0:
                logger.warning("No new training data available")
                return False
            
            # Get recent data (last ML_LOOKBACK_PERIODS candles)
            X_recent = X_new.tail(ML_LOOKBACK_PERIODS)
            y_recent = y_new.tail(ML_LOOKBACK_PERIODS)
            
            # Re-run feature selection on recent data
            if len(X_recent) >= 50:  # Minimum samples for RFE
                training_success = await self.train_model(X_recent, y_recent, rfe_eligible_features)
                if training_success:
                    logger.info("Online retraining completed")
                    return True
                else:
                    logger.warning("Online retraining failed during model training")
                    return False
            else:
                logger.warning("Insufficient data for retraining")
                return False
                
        except Exception as e:
            logger.error(f"Online retraining failed: {e}")
            return False
    
    async def save_model(self):
        """Save the trained model and metadata"""
        try:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            
            # Save CatBoost model
            model_file = os.path.join(self.model_path, f"catboost_model_{timestamp}.cbm")
            self.model.save_model(model_file)
            
            # Save metadata
            metadata = {
                'selected_features': self.selected_features,
                'feature_importance': self.feature_importance,
                'feature_weights': self.feature_weights,
                'model_performance': self.model_performance,
                'training_progress': self.training_progress,
                'model_file': model_file,
                'timestamp': timestamp
            }
            
            metadata_file = os.path.join(self.model_path, f"model_metadata_{timestamp}.joblib")
            joblib.dump(metadata, metadata_file)
            
            # Save as latest
            latest_model_file = os.path.join(self.model_path, "latest_model.cbm")
            latest_metadata_file = os.path.join(self.model_path, "latest_metadata.joblib")
            
            self.model.save_model(latest_model_file)
            joblib.dump(metadata, latest_metadata_file)
            
            # Save label encoder
            label_encoder_file = os.path.join(self.model_path, "label_encoder.joblib")
            joblib.dump(self.label_encoder, label_encoder_file)
            
            logger.info(f"Model saved successfully: {model_file}")
            
        except Exception as e:
            logger.error(f"Failed to save model: {e}")
    
    async def load_model(self):
        """Load existing model if available"""
        try:
            latest_model_file = os.path.join(self.model_path, "latest_model.cbm")
            latest_metadata_file = os.path.join(self.model_path, "latest_metadata.joblib")
            label_encoder_file = os.path.join(self.model_path, "label_encoder.joblib")
            
            if os.path.exists(latest_model_file) and os.path.exists(latest_metadata_file):
                # Load CatBoost model
                self.model.load_model(latest_model_file)
                
                # Load metadata
                metadata = joblib.load(latest_metadata_file)
                self.selected_features = metadata.get('selected_features', [])
                self.feature_importance = metadata.get('feature_importance', {})
                self.feature_weights = metadata.get('feature_weights', {})
                self.model_performance = metadata.get('model_performance', {})
                self.training_progress = metadata.get('training_progress', 0)
                
                # Load label encoder
                if os.path.exists(label_encoder_file):
                    self.label_encoder = joblib.load(label_encoder_file)
                
                self.is_trained = True
                logger.info("Existing model loaded successfully")
                
            else:
                logger.info("No existing model found, will train new model")
                
        except Exception as e:
            logger.warning(f"Failed to load existing model: {e}")
    
    # ==================== SLIDING WINDOW ACCURACY TRACKING ====================
    
    def add_prediction_result(self, prediction: str, actual: str):
        """Add a prediction result to the sliding window for live accuracy tracking"""
        try:
            # Add to recent predictions
            self.recent_predictions.append((prediction, actual))
            
            # Maintain sliding window size
            if len(self.recent_predictions) > self.accuracy_window_size:
                self.recent_predictions = self.recent_predictions[-self.accuracy_window_size:]
            
            # Update live accuracy
            self._update_live_accuracy()
            
        except Exception as e:
            logger.debug(f"[ACCURACY] Failed to add prediction result: {e}")
    
    def _update_live_accuracy(self):
        """Update the live accuracy based on recent predictions"""
        try:
            if not self.recent_predictions:
                self.accuracy_live = 0.0
                return
            
            correct_predictions = 0
            for prediction, actual in self.recent_predictions:
                if prediction == actual:
                    correct_predictions += 1
            
            self.accuracy_live = correct_predictions / len(self.recent_predictions)
            
        except Exception as e:
            logger.debug(f"[ACCURACY] Failed to update live accuracy: {e}")
            self.accuracy_live = 0.0
    
    def get_live_accuracy(self) -> float:
        """Get the current live accuracy"""
        return self.accuracy_live
    
    def get_accuracy_window_stats(self) -> Dict[str, Any]:
        """Get detailed statistics about the accuracy sliding window"""
        try:
            if not self.recent_predictions:
                return {
                    'window_size': 0,
                    'accuracy': 0.0,
                    'correct_predictions': 0,
                    'total_predictions': 0
                }
            
            correct = sum(1 for pred, actual in self.recent_predictions if pred == actual)
            total = len(self.recent_predictions)
            
            return {
                'window_size': total,
                'accuracy': correct / total if total > 0 else 0.0,
                'correct_predictions': correct,
                'total_predictions': total,
                'max_window_size': self.accuracy_window_size
            }
            
        except Exception as e:
            logger.debug(f"[ACCURACY] Failed to get window stats: {e}")
            return {
                'window_size': 0,
                'accuracy': 0.0,
                'correct_predictions': 0,
                'total_predictions': 0,
                'error': str(e)
            }
    
    def get_training_progress_info(self) -> Dict[str, Any]:
        """Get detailed training progress information (Complete Pipeline Restructure)"""
        return {
            'overall_progress': self.overall_training_progress,
            'current_stage': self.current_training_stage,
            'stages': self.training_stages,
            'stage_descriptions': {
                'sanitizing': 'Data sanitization and preprocessing',
                'building': 'Building training/test splits',
                'feature_selection': 'RFE feature selection',
                'fitting': 'Model training and fitting',
                'validating': 'Model validation and evaluation',
                'finalizing': 'Saving model and cleanup'
            }
        }
    
    # ==================== END SLIDING WINDOW ACCURACY TRACKING ====================
    
    def get_model_info(self) -> Dict[str, Any]:
        """Get model information for web interface (Complete Pipeline Restructure + User Feedback Adjustments)"""
        # Standard class mapping (User Feedback Adjustments)
        class_mapping = {"0": "SELL", "1": "HOLD", "2": "BUY"}
        
        # Get accuracy window stats
        accuracy_stats = self.get_accuracy_window_stats()
        
        return {
            'is_trained': self.is_trained,
            'training_progress': self.overall_training_progress,  # Use new overall progress
            'model_performance': self.model_performance,
            'selected_features_count': len(self.selected_features),
            'total_features_count': len(self.feature_weights),
            'active_features': [f for f, w in self.feature_weights.items() if w == self.active_weight],
            'inactive_features': [f for f, w in self.feature_weights.items() if w == self.inactive_weight],
            'feature_importance': self.feature_importance,
            # MySQL migration enhancements
            'fallback_active': not self.is_trained,
            'last_training_time': self.last_training_time.isoformat() if self.last_training_time else None,
            'samples_in_last_train': self.model_performance.get('training_samples', 0),
            'class_distribution': self.class_distribution,
            'training_cooldown_active': (
                self.training_cooldown_until and 
                datetime.now() < self.training_cooldown_until
            ) if hasattr(self, 'training_cooldown_until') and self.training_cooldown_until else False,
            'prediction_warning_count': getattr(self, 'prediction_warning_counter', 0),
            # Follow-up fixes: Additional diagnostics
            'last_training_error': self.last_training_error,
            'numeric_feature_count': self.numeric_feature_count,
            'selected_feature_count': self.selected_feature_count,
            'class_weights': self.class_weights,
            'sanitization': self.last_sanitization_stats,
            'next_retry_at': self.next_retry_at.isoformat() if self.next_retry_at else None,
            # Complete Pipeline Restructure additions
            'model_version': self.model_version,
            'training_count': self.training_count,
            'current_training_stage': self.current_training_stage,
            'accuracy_live': self.accuracy_live,
            'last_accuracy': self.model_performance.get('accuracy', 0.0),
            'accuracy_window_stats': accuracy_stats,
            'training_progress_info': self.get_training_progress_info(),
            # User Feedback Adjustments
            'class_mapping': class_mapping,
            'accuracy_live_count': accuracy_stats.get('live_count', 0),
            'collected_valid_samples': self.model_performance.get('training_samples', 0)  # Valid samples after sanitization
        }