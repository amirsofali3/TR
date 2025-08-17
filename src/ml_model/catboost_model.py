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
                verbose=False,  # Use verbose instead of logging_level to avoid conflicts
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
        """Sanitize features NON-DESTRUCTIVELY (no row drops) and return both sanitized data and metadata
        
        Phase 3 Fix: NEVER drop rows, only columns. Use imputation for missing values.
        
        Returns:
            tuple: (sanitized_dataframe, metadata_dict)
        """
        try:
            logger.info("[TRAIN] Starting NON-DESTRUCTIVE feature sanitization...")
            sanitized_df = feature_df.copy()
            initial_feature_count = len(sanitized_df.columns)
            initial_row_count = len(sanitized_df)
            dropped_features = []
            converted_features = []
            imputed_columns = []
            
            for column in feature_df.columns:
                col_data = sanitized_df[column]
                
                # Check if column is already numeric
                if pd.api.types.is_numeric_dtype(col_data):
                    # Check for NaN values that need imputation
                    if col_data.isna().any():
                        nan_count = col_data.isna().sum()
                        sanitized_df[column] = self._impute_missing(col_data)
                        imputed_columns.append(f"{column} ({nan_count} NaNs)")
                    continue
                
                # Handle object/string columns  
                if col_data.dtype == 'object':
                    unique_values = col_data.dropna().unique()
                    unique_count = len(unique_values)
                    
                    logger.info(f"[TRAIN] Found non-numeric column '{column}' with {unique_count} unique values")
                    
                    # Special handling for symbol column - encode without dropping rows
                    if column.lower() == 'symbol':
                        logger.info(f"[TRAIN] Processing symbol column: {column}")
                        if unique_count <= 1:
                            # Constant symbol - drop column but keep rows
                            logger.info(f"[TRAIN] Dropping constant symbol column: {column} (value: {unique_values[0] if len(unique_values) > 0 else 'N/A'})")
                            sanitized_df = sanitized_df.drop(columns=[column])
                            dropped_features.append(f"{column} (constant_symbol)")
                        elif unique_count <= 50:  # Reasonable number of symbols for encoding
                            # Encode symbol to numerical codes, handling NaNs
                            from sklearn.preprocessing import LabelEncoder
                            le = LabelEncoder()
                            # Fill NaNs with a placeholder for encoding, then restore as numeric NaN
                            col_filled = col_data.fillna('__MISSING__').astype(str)
                            symbol_codes = le.fit_transform(col_filled)
                            # Convert back to float and set missing values as NaN for imputation
                            symbol_codes = symbol_codes.astype(float)
                            symbol_codes[col_filled == '__MISSING__'] = np.nan
                            # Apply imputation to handle any NaNs
                            sanitized_df[f'{column}_code'] = self._impute_missing(pd.Series(symbol_codes, index=col_data.index))
                            sanitized_df = sanitized_df.drop(columns=[column])
                            converted_features.append(f"{column} → {column}_code (symbol_encoded)")
                            logger.info(f"[TRAIN] Encoded symbol column {column} to {column}_code with {unique_count} categories")
                        else:
                            # Too many symbols - drop column but keep rows
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
                                # Apply imputation to handle conversion NaNs
                                sanitized_df[column] = self._impute_missing(numeric_col)
                                converted_features.append(f"{column} (to_numeric)")
                                if numeric_col.isna().any():
                                    imputed_columns.append(f"{column} ({numeric_col.isna().sum()} conversion NaNs)")
                                continue
                            
                            # If conversion fails, use label encoding for low-cardinality categorical
                            from sklearn.preprocessing import LabelEncoder
                            le = LabelEncoder()
                            col_filled = col_data.fillna('__MISSING__').astype(str)
                            encoded_values = le.fit_transform(col_filled)
                            # Convert to float and handle missing placeholder
                            encoded_values = encoded_values.astype(float)
                            encoded_values[col_filled == '__MISSING__'] = np.nan
                            sanitized_df[column] = self._impute_missing(pd.Series(encoded_values, index=col_data.index))
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
                        numeric_col = pd.to_numeric(col_data, errors='coerce')
                        sanitized_df[column] = self._impute_missing(numeric_col)
                        converted_features.append(f"{column} (forced_numeric)")
                        if numeric_col.isna().any():
                            imputed_columns.append(f"{column} ({numeric_col.isna().sum()} conversion NaNs)")
                    except:
                        logger.warning(f"[TRAIN] Dropping unconvertible column: {column} (dtype: {col_data.dtype})")
                        sanitized_df = sanitized_df.drop(columns=[column])
                        dropped_features.append(column)
            
            # Final check: remove any columns that became all NaN after conversion (but keep rows)
            all_nan_columns = []
            for column in sanitized_df.columns:
                if sanitized_df[column].isna().all():
                    logger.info(f"[TRAIN] Dropping all-NaN column after conversion: {column}")
                    sanitized_df = sanitized_df.drop(columns=[column])
                    all_nan_columns.append(column)
                    dropped_features.append(f"{column} (all_NaN)")
            
            # CRITICAL: Ensure no rows are dropped - only columns
            final_row_count = len(sanitized_df)
            
            # Log sanitization results
            final_feature_count = len(sanitized_df.columns)
            logger.info(f"[TRAIN] NON-DESTRUCTIVE feature sanitization completed:")
            logger.info(f"[TRAIN]   Features: {initial_feature_count} → {final_feature_count}")
            logger.info(f"[TRAIN]   Samples: {initial_row_count} → {final_row_count} (MUST BE EQUAL)")
            
            # Assert row count preservation (critical for X/y alignment)
            if final_row_count != initial_row_count:
                logger.error(f"[TRAIN] CRITICAL ERROR: Row count changed during sanitization! {initial_row_count} → {final_row_count}")
                logger.error("[TRAIN] This will cause X/y length mismatch!")
                # This should not happen with the new implementation
                
            if dropped_features:
                logger.info(f"[TRAIN]   Dropped features ({len(dropped_features)}): {', '.join(dropped_features[:5])}" + 
                          (f" and {len(dropped_features)-5} more" if len(dropped_features) > 5 else ""))
            
            if converted_features:
                logger.info(f"[TRAIN]   Converted features ({len(converted_features)}): {', '.join(converted_features[:3])}" + 
                          (f" and {len(converted_features)-3} more" if len(converted_features) > 3 else ""))
                          
            if imputed_columns:
                logger.info(f"[TRAIN]   Imputed features ({len(imputed_columns)}): {', '.join(imputed_columns[:3])}" + 
                          (f" and {len(imputed_columns)-3} more" if len(imputed_columns) > 3 else ""))
            
            # Create metadata dict
            metadata = {
                'initial_feature_count': initial_feature_count,
                'final_feature_count': final_feature_count,
                'dropped_constant': len([f for f in dropped_features if 'constant' in str(f)]),
                'dropped_high_cardinality': len([f for f in dropped_features if 'high-cardinality' in str(f) or 'high_cardinality' in str(f)]),
                'encoded_categorical': len([f for f in converted_features if 'label_encoded' in f or 'symbol_encoded' in f]),
                'converted_numeric': len([f for f in converted_features if 'to_numeric' in f or 'forced_numeric' in f]),
                'imputed_columns': len(imputed_columns),
                'samples_before': initial_row_count,
                'samples_after': final_row_count,
                'rows_preserved': final_row_count == initial_row_count
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
                'samples_before': len(feature_df),
                'samples_after': len(feature_df),
                'error': str(e)
            }

    def _impute_missing(self, series: pd.Series) -> pd.Series:
        """Impute missing values using forward fill → back fill → zero strategy
        
        Phase 3 addition: Handle NaN values without dropping rows
        """
        try:
            if not series.isna().any():
                return series
                
            # Strategy: forward fill → back fill → zero
            imputed = series.copy()
            
            # Forward fill
            imputed = imputed.fillna(method='ffill')
            
            # Back fill for any remaining NaNs
            imputed = imputed.fillna(method='bfill')
            
            # Fill any remaining NaNs with zero (or median for float columns)
            if imputed.isna().any():
                if pd.api.types.is_numeric_dtype(imputed):
                    # Use median for numeric columns, fallback to 0
                    fill_value = imputed.median() if not imputed.isna().all() else 0
                else:
                    fill_value = 0
                imputed = imputed.fillna(fill_value)
            
            return imputed
            
        except Exception as e:
            logger.warning(f"[TRAIN] Imputation failed for series: {e}, filling with zeros")
            return series.fillna(0)
    
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
            from config.settings import RFE_VERBOSE
            estimator = CatBoostClassifier(
                iterations=100,
                depth=6,
                learning_rate=0.1,
                random_seed=42,
                train_dir=rfe_train_dir,
                allow_writing_files=True,
                verbose=RFE_VERBOSE  # Use config setting, avoiding logging_level conflict
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
            self.selection_method = 'RFE'  # Mark successful RFE
            self.fallback_reason = None  # Clear any previous fallback reason
            
            self.training_progress = 40
            
            return final_features
            
        except Exception as e:
            logger.error(f"[TRAIN] RFE selection failed: {e}")
            # Fallback: select top features by importance
            self.fallback_reason = f"RFE failed: {str(e)}"  # Record fallback reason
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
            from config.settings import RFE_VERBOSE
            temp_model = CatBoostClassifier(
                iterations=100,
                depth=4,
                learning_rate=0.1,
                random_seed=42,
                train_dir=importance_train_dir,
                allow_writing_files=True,
                verbose=RFE_VERBOSE  # Use config setting, avoiding logging_level conflict
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
            
            # Mark selection method if not already set by RFE
            if not hasattr(self, 'selection_method') or self.selection_method != 'RFE':
                self.selection_method = 'importance'
            
            logger.info(f"[TRAIN] Selected {len(rfe_features)} top important features (max {max_features})")
            
            return all_selected_features
            
        except Exception as e:
            logger.error(f"[TRAIN] Feature importance selection failed: {e}")
            # Mark as fallback to all features
            self.selection_method = 'all'
            if not hasattr(self, 'fallback_reason'):
                self.fallback_reason = f"Both RFE and importance selection failed: {str(e)}"
            # Try to return sanitized features if available, otherwise original features
            try:
                X_sanitized, _ = self._sanitize_features(X)
                return list(X_sanitized.columns)
            except:
                return list(X.columns)  # Return all original features as final fallback
    
    def setup_feature_weights(self, selected_features: List[str], all_features: List[str]):
        """Setup feature weights (active vs inactive) with audit logging (Phase 4)"""
        from config.settings import BASE_MUST_KEEP_FEATURES
        
        self.feature_weights = {}
        
        # Calculate feature counts for audit logging
        must_keep_count = 0
        for feature in all_features:
            if feature in selected_features:
                self.feature_weights[feature] = self.active_weight
                if feature in BASE_MUST_KEEP_FEATURES:
                    must_keep_count += 1
            else:
                self.feature_weights[feature] = self.inactive_weight
        
        # Phase 4: Audit log feature distribution
        selected_count = len(selected_features)
        inactive_count = len(all_features) - selected_count
        logger.info(f"[TRAIN] Feature distribution: selected={selected_count}, must_keep={must_keep_count}, inactive={inactive_count}")
        logger.info(f"Setup weights: {selected_count} active, {inactive_count} inactive")
    
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
    
    async def perform_full_training_flow(self, X: pd.DataFrame, y: pd.Series, rfe_eligible_features: List[str], X_recent: Optional[pd.DataFrame] = None, y_recent: Optional[pd.Series] = None) -> bool:
        """Comprehensive training flow with multiple fallback strategies - Phase 3 addition"""
        try:
            logger.info("[TRAIN] Starting comprehensive training flow with fallback strategies...")
            
            # Attempt 1: Full training with RFE (if enabled)
            try:
                success = await self.train_model(X, y, rfe_eligible_features, X_recent, y_recent)
                if success:
                    logger.success("[TRAIN] Full training flow completed successfully")
                    return True
                else:
                    logger.warning("[TRAIN] Full training flow failed, trying fallback strategies...")
            except Exception as e:
                logger.error(f"[TRAIN] Full training failed: {e}")
                self.last_training_error = f"Full training: {str(e)}"
            
            # Attempt 2: Simplified training with all numeric features (no RFE)
            try:
                logger.info("[TRAIN] Attempting simplified training with all numeric features...")
                
                # Ensure we have numeric data
                X_simplified, _ = self._sanitize_features(X)
                if len(X_simplified.columns) == 0:
                    logger.error("[TRAIN] No numeric features available for simplified training")
                    return False
                
                # Use all numeric features as "selected"
                all_numeric_features = list(X_simplified.columns)
                self.selected_features = all_numeric_features
                self.selected_feature_count = len(all_numeric_features)
                
                logger.info(f"[TRAIN] Simplified training with {len(all_numeric_features)} features")
                
                # Try basic model training without complex RFE
                success = await self._train_simplified_model(X_simplified, y)
                if success:
                    logger.success("[TRAIN] Simplified training completed successfully")
                    return True
                    
            except Exception as e:
                logger.error(f"[TRAIN] Simplified training failed: {e}")
                self.last_training_error = f"Simplified training: {str(e)}"
            
            # Attempt 3: Minimal model with just must-keep features
            try:
                logger.info("[TRAIN] Attempting minimal training with must-keep features only...")
                
                from config.settings import BASE_MUST_KEEP_FEATURES
                must_keep_features = BASE_MUST_KEEP_FEATURES or ["open", "high", "low", "close", "volume"]
                
                # Filter to existing columns
                available_must_keep = [col for col in must_keep_features if col in X.columns]
                
                if len(available_must_keep) < 3:
                    logger.error("[TRAIN] Not enough must-keep features for minimal training")
                    return False
                
                X_minimal = X[available_must_keep]
                self.selected_features = available_must_keep
                self.selected_feature_count = len(available_must_keep)
                
                logger.info(f"[TRAIN] Minimal training with {len(available_must_keep)} must-keep features")
                
                success = await self._train_simplified_model(X_minimal, y)
                if success:
                    logger.success("[TRAIN] Minimal training completed successfully")
                    return True
                    
            except Exception as e:
                logger.error(f"[TRAIN] Minimal training failed: {e}")
                self.last_training_error = f"Minimal training: {str(e)}"
            
            # All attempts failed
            logger.error("[TRAIN] All training strategies failed")
            self.last_training_error = "All training strategies failed"
            return False
            
        except Exception as e:
            logger.error(f"[TRAIN] Comprehensive training flow failed: {e}")
            self.last_training_error = f"Training flow error: {str(e)}"
            return False
    
    async def _train_simplified_model(self, X: pd.DataFrame, y: pd.Series) -> bool:
        """Train a simplified model without complex features - Phase 3 helper"""
        try:
            # Basic data validation
            if len(X) == 0 or len(y) == 0 or len(X) != len(y):
                logger.error(f"[TRAIN] Invalid data for simplified training: X={len(X)}, y={len(y)}")
                return False
            
            # Encode labels
            if self.label_encoder is None:
                self.label_encoder = LabelEncoder()
            
            y_encoded = self.label_encoder.fit_transform(y)
            
            # Simple train/test split
            X_train, X_test, y_train, y_test = train_test_split(
                X, y_encoded, test_size=0.2, random_state=42, stratify=y_encoded
            )
            
            # Create simple CatBoost model
            simple_train_dir = os.path.join(self.model_path, "catboost_simple")
            os.makedirs(simple_train_dir, exist_ok=True)
            
            self.model = CatBoostClassifier(
                iterations=min(500, CATBOOST_ITERATIONS // 2),  # Fewer iterations for simplified model
                depth=min(6, CATBOOST_DEPTH),
                learning_rate=CATBOOST_LEARNING_RATE,
                loss_function='MultiClass',
                eval_metric='MultiClass',
                random_seed=42,
                verbose=False,  # Use verbose instead of logging_level 
                train_dir=simple_train_dir,
                allow_writing_files=True
            )
            
            # Train the model
            self.model.fit(X_train, y_train, verbose=False)
            
            # Evaluate
            y_pred = self.model.predict(X_test)
            accuracy = accuracy_score(y_test, y_pred)
            
            # Store basic performance
            self.model_performance = {
                'accuracy': accuracy,
                'training_samples': len(X_train),
                'test_samples': len(X_test),
                'selected_features': len(X.columns),
                'training_date': datetime.now().isoformat(),
                'model_type': 'simplified'
            }
            
            self.is_trained = True
            self.last_training_time = datetime.now()
            logger.info(f"[TRAIN] Simplified model trained with accuracy: {accuracy:.4f}")
            
            return True
            
        except Exception as e:
            logger.error(f"[TRAIN] Simplified model training failed: {e}")
            return False
    
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
            
            # CRITICAL: Verify row alignment after sanitization 
            if len(X_sanitized) != len(y):
                logger.error(f"[TRAIN] CRITICAL X/y length mismatch after sanitization! X: {len(X_sanitized)}, y: {len(y)}")
                if len(X_sanitized) < len(y):
                    # If X is shorter, align y to X's index
                    logger.warning(f"[TRAIN] Realigning y to X's index to fix mismatch")
                    y = y.loc[X_sanitized.index]
                    logger.info(f"[TRAIN] Realigned y length: {len(y)}")
                else:
                    # This should not happen with the new sanitization, but handle it
                    logger.error(f"[TRAIN] X is longer than y - this should not happen!")
                    return False
            else:
                logger.info(f"[TRAIN] X/y alignment verified: {len(X_sanitized)} samples")
            
            # Sanitize recent features if provided
            X_recent_sanitized = None
            if X_recent is not None:
                logger.info("[TRAIN] Sanitizing recent training features...")
                X_recent_sanitized, recent_sanitization_stats = self._sanitize_features(X_recent)
                logger.info(f"[TRAIN] Recent sanitization: {recent_sanitization_stats['initial_feature_count']} → {recent_sanitization_stats['final_feature_count']} features")
                
                # Verify recent features alignment if y_recent provided
                if y_recent is not None and len(X_recent_sanitized) != len(y_recent):
                    logger.warning(f"[TRAIN] Recent X/y length mismatch: X: {len(X_recent_sanitized)}, y: {len(y_recent)}")
                    if len(X_recent_sanitized) < len(y_recent):
                        y_recent = y_recent.loc[X_recent_sanitized.index]
                        logger.info(f"[TRAIN] Realigned recent y length: {len(y_recent)}")
            
            
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
            
            # Check ENABLE_RFE setting for optional RFE (Phase 3 improvement)
            from config.settings import ENABLE_RFE, RFE_FALLBACK_TOP_N, BASE_MUST_KEEP_FEATURES
            
            if ENABLE_RFE:
                logger.info("[TRAIN] RFE is enabled, performing feature selection...")
                # Perform RFE feature selection with progress tracking (using sanitized data)
                selected_features = await self.perform_rfe_selection(X_train, y_train, rfe_eligible_features, X_recent_sanitized, y_recent)
            else:
                logger.info("[TRAIN] RFE is disabled, using importance-based fallback selection...")
                # Use importance-based selection instead of RFE
                selected_features = await self.select_features_by_importance(X_train, y_train, rfe_eligible_features)
            
            # Ensure selected_features is never empty (critical fallback)
            if not selected_features:
                logger.warning("[TRAIN] No features selected! Using must-keep features as fallback")
                must_keep_features = BASE_MUST_KEEP_FEATURES or ["open", "high", "low", "close", "volume"]
                # Take must-keep features that exist in the data
                selected_features = [col for col in must_keep_features if col in X_train.columns]
                
                # If still empty, take all numeric features (last resort)
                if not selected_features:
                    logger.warning("[TRAIN] Must-keep features not found! Using all available features")
                    selected_features = list(X_train.columns)
                
                logger.info(f"[TRAIN] Fallback selected {len(selected_features)} features: {selected_features[:5]}...")
                
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
            
            # Create stable train directory for CatBoost artifacts (Phase 3 improvement)
            main_train_dir = os.path.join(self.model_path, "catboost_main")
            os.makedirs(main_train_dir, exist_ok=True)
            
            # Create a new model instance with class weights in constructor
            self.model = CatBoostClassifier(
                iterations=CATBOOST_ITERATIONS,
                depth=CATBOOST_DEPTH,
                learning_rate=CATBOOST_LEARNING_RATE,
                loss_function='MultiClass',
                eval_metric='MultiClass',
                random_seed=42,
                verbose=False,  # Use verbose instead of logging_level
                bootstrap_type='Bayesian',
                bagging_temperature=1.0,
                colsample_bylevel=0.8,
                reg_lambda=3.0,
                thread_count=-1,
                use_best_model=True,
                early_stopping_rounds=50,
                train_dir=main_train_dir,  # Stable train directory
                allow_writing_files=True,
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
            
            # Phase 4: Persist model metadata for continuity
            self.persist_model_metadata()
            
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
            'collected_valid_samples': self.model_performance.get('training_samples', 0),  # Valid samples after sanitization
            # Phase 4 additions: Feature selection metadata
            'feature_selection': {
                'method': getattr(self, 'selection_method', 'unknown'),
                'selected_count': len(self.selected_features),
                'must_keep_count': len([f for f, w in self.feature_weights.items() if w == self.active_weight and f in BASE_MUST_KEEP_FEATURES]),
                'inactive_count': len([f for f, w in self.feature_weights.items() if w == self.inactive_weight]),
                'rfe_enabled': ENABLE_RFE,
                'fallback_reason': getattr(self, 'fallback_reason', None)
            }
        }
    
    def get_feature_breakdown(self) -> Dict[str, Any]:
        """Get comprehensive feature breakdown for API endpoints (Phase 4)"""
        try:
            from config.settings import BASE_MUST_KEEP_FEATURES
            
            # Get active and inactive features from feature weights
            active_features = [f for f, w in self.feature_weights.items() if w == self.active_weight]
            inactive_features = [f for f, w in self.feature_weights.items() if w == self.inactive_weight]
            
            # Calculate must-keep features 
            must_keep_features = [f for f in active_features if f in BASE_MUST_KEEP_FEATURES]
            
            # Get feature importance data
            feature_importance = self.feature_importance or {}
            
            # Build detailed breakdown
            selected_detailed = []
            for feature in active_features:
                selected_detailed.append({
                    'name': feature,
                    'importance': feature_importance.get(feature, 0.0),
                    'is_must_keep': feature in BASE_MUST_KEEP_FEATURES,
                    'status': 'selected'
                })
            
            inactive_detailed = []
            for feature in inactive_features:
                inactive_detailed.append({
                    'name': feature,
                    'importance': feature_importance.get(feature, 0.0),
                    'status': 'inactive'
                })
            
            return {
                'selected': selected_detailed,
                'inactive': inactive_detailed,
                'must_keep': must_keep_features,
                'selection_method': getattr(self, 'selection_method', 'unknown'),
                'timestamp': datetime.now().isoformat(),
                'metadata': {
                    'total_selected': len(active_features),
                    'total_inactive': len(inactive_features),
                    'total_features': len(active_features) + len(inactive_features),
                    'must_keep_count': len(must_keep_features),
                    'model_version': self.model_version,
                    'last_training_time': self.last_training_time.isoformat() if self.last_training_time else None,
                    'rfe_enabled': ENABLE_RFE,
                    'fallback_reason': getattr(self, 'fallback_reason', None)
                }
            }
            
        except Exception as e:
            logger.error(f"Error getting feature breakdown: {e}")
            return {
                'selected': [],
                'inactive': [],
                'must_keep': [],
                'selection_method': 'error',
                'timestamp': datetime.now().isoformat(),
                'metadata': {'error': str(e)}
            }
    
    def persist_model_metadata(self):
        """Persist model metadata to JSON file for continuity (Phase 4)"""
        try:
            import json
            from pathlib import Path
            
            # Create models directory if it doesn't exist
            models_dir = Path(self.model_path)
            models_dir.mkdir(exist_ok=True)
            
            # Build metadata
            metadata = {
                'model_version': self.model_version,
                'training_count': self.training_count,
                'last_training_time': self.last_training_time.isoformat() if self.last_training_time else None,
                'selected_features': self.selected_features,
                'selection_method': getattr(self, 'selection_method', 'unknown'),
                'fallback_reason': getattr(self, 'fallback_reason', None),
                'feature_count': len(self.selected_features),
                'model_performance': self.model_performance,
                'is_trained': self.is_trained,
                'created_at': datetime.now().isoformat()
            }
            
            # Save to JSON file
            metadata_file = models_dir / 'last_model_meta.json'
            with open(metadata_file, 'w') as f:
                json.dump(metadata, f, indent=2)
            
            logger.info(f"[TRAIN] Model metadata persisted to {metadata_file}")
            
        except Exception as e:
            logger.warning(f"[TRAIN] Failed to persist metadata: {e}")