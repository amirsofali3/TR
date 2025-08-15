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
                subsample=0.8,
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
    
    async def perform_rfe_selection(self, X: pd.DataFrame, y: pd.Series, rfe_eligible_features: List[str]) -> List[str]:
        """Perform Recursive Feature Elimination on eligible features (MySQL migration improved)"""
        try:
            logger.info("[TRAIN] Performing RFE feature selection...")
            self.training_progress = 10
            
            # Guard: Check minimum samples for RFE (MySQL migration)
            if len(X) < MIN_RFE_SAMPLES:
                logger.warning(f"[TRAIN] Insufficient samples for RFE ({len(X)} < {MIN_RFE_SAMPLES}), using baseline feature selection")
                return await self.select_features_by_importance(X, y, rfe_eligible_features)
            
            # Filter to only RFE-eligible features
            rfe_features = [col for col in X.columns if col in rfe_eligible_features]
            
            if len(rfe_features) == 0:
                logger.warning("[TRAIN] No RFE-eligible features found")
                return list(X.columns)
            
            X_rfe = X[rfe_features].copy()
            
            # Use a simpler model for RFE to speed up the process
            estimator = CatBoostClassifier(
                iterations=100,
                depth=6,
                learning_rate=0.1,
                logging_level='Silent',
                random_seed=42
            )
            
            # Perform RFE with cross-validation
            rfe = RFECV(
                estimator=estimator,
                step=10,  # Remove 10 features at a time
                cv=3,     # 3-fold cross-validation
                scoring='accuracy',
                n_jobs=-1,
                min_features_to_select=min(RFE_N_FEATURES, len(rfe_features))
            )
            
            self.training_progress = 30
            
            rfe.fit(X_rfe, y)
            
            # Get selected features
            selected_rfe_features = X_rfe.columns[rfe.support_].tolist()
            
            # Add back required features (must keep)
            required_features = [col for col in X.columns if col not in rfe_features]
            all_selected_features = selected_rfe_features + required_features
            
            logger.info(f"[TRAIN] RFE selected {len(selected_rfe_features)} features out of {len(rfe_features)} eligible")
            logger.info(f"[TRAIN] Total features after adding required: {len(all_selected_features)}")
            
            # Store RFE results
            self.rfe_selector = rfe
            
            self.training_progress = 40
            
            return all_selected_features
            
        except Exception as e:
            logger.error(f"[TRAIN] RFE selection failed: {e}")
            # Fallback: select top features by importance
            return await self.select_features_by_importance(X, y, rfe_eligible_features)
    
    async def select_features_by_importance(self, X: pd.DataFrame, y: pd.Series, rfe_eligible_features: List[str]) -> List[str]:
        """Fallback feature selection based on feature importance (MySQL migration improved)"""
        try:
            logger.info("[TRAIN] Using feature importance for selection...")
            
            # Cap features to MAX_BASELINE_FEATURES when RFE is skipped (MySQL migration)
            max_features = min(MAX_BASELINE_FEATURES, len(rfe_eligible_features))
            
            # Train a quick model to get feature importance
            temp_model = CatBoostClassifier(
                iterations=100,
                depth=4,
                learning_rate=0.1,
                logging_level='Silent',
                random_seed=42
            )
            
            temp_model.fit(X, y)
            
            # Get feature importance
            importance_df = pd.DataFrame({
                'feature': X.columns,
                'importance': temp_model.get_feature_importance()
            }).sort_values('importance', ascending=False)
            
            # Select top features (use variance proxy if many features) (MySQL migration)
            rfe_features = importance_df[
                importance_df['feature'].isin(rfe_eligible_features)
            ].head(max_features)['feature'].tolist()
            
            # Add required features
            required_features = [col for col in X.columns if col not in rfe_eligible_features]
            all_selected_features = rfe_features + required_features
            
            logger.info(f"[TRAIN] Selected {len(rfe_features)} top important features (max {max_features})")
            
            return all_selected_features
            
        except Exception as e:
            logger.error(f"[TRAIN] Feature importance selection failed: {e}")
            return list(X.columns)  # Return all features as fallback
    
    def setup_feature_weights(self, selected_features: List[str], all_features: List[str]):
        """Setup feature weights (active vs inactive)"""
        self.feature_weights = {}
        
        for feature in all_features:
            if feature in selected_features:
                self.feature_weights[feature] = self.active_weight
            else:
                self.feature_weights[feature] = self.inactive_weight
        
        logger.info(f"Setup weights: {len(selected_features)} active, {len(all_features) - len(selected_features)} inactive")
    
    async def train_model(self, X: pd.DataFrame, y: pd.Series, rfe_eligible_features: List[str]):
        """Train the CatBoost model with feature selection"""
        try:
            logger.info("Starting model training...")
            self.training_progress = 0
            
            if len(X) == 0 or len(y) == 0:
                logger.error("No training data available")
                return False
            
            # Encode labels
            y_encoded = self.label_encoder.fit_transform(y)
            
            # Split data
            X_train, X_test, y_train, y_test = train_test_split(
                X, y_encoded, test_size=0.2, random_state=42, stratify=y_encoded
            )
            
            self.training_progress = 5
            
            # Perform RFE feature selection
            selected_features = await self.perform_rfe_selection(X_train, y_train, rfe_eligible_features)
            self.selected_features = selected_features
            
            # Setup feature weights
            self.setup_feature_weights(selected_features, list(X.columns))
            
            # Apply feature selection
            X_train_selected = X_train[selected_features]
            X_test_selected = X_test[selected_features]
            
            self.training_progress = 50
            
            # Train the main model
            logger.info("Training main CatBoost model...")
            
            # Use validation set for early stopping
            X_train_final, X_val, y_train_final, y_val = train_test_split(
                X_train_selected, y_train, test_size=0.2, random_state=42
            )
            
            self.model.fit(
                X_train_final, y_train_final,
                eval_set=(X_val, y_val),
                verbose=False,
                plot=False
            )
            
            self.training_progress = 80
            
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
                'training_date': datetime.now().isoformat()
            }
            
            # Get feature importance
            feature_names = X_train_selected.columns
            importance_scores = self.model.get_feature_importance()
            self.feature_importance = dict(zip(feature_names, importance_scores))
            
            self.training_progress = 90
            
            # Save model
            await self.save_model()
            
            self.is_trained = True
            self.training_progress = 100
            
            logger.success(f"Model training completed. Accuracy: {accuracy:.4f}, F1: {f1:.4f}")
            
            return True
            
        except Exception as e:
            logger.error(f"Model training failed: {e}")
            self.training_progress = 0
            return False
    
    async def predict(self, X: pd.DataFrame) -> Dict[str, Any]:
        """Make predictions with confidence scores"""
        try:
            if not self.is_trained or self.model is None:
                logger.warning("Model is not trained")
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
                if available_features:
                    X_processed = X_processed[available_features]
            
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
                await self.train_model(X_recent, y_recent, rfe_eligible_features)
                logger.info("Online retraining completed")
                return True
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
    
    def get_model_info(self) -> Dict[str, Any]:
        """Get model information for web interface"""
        return {
            'is_trained': self.is_trained,
            'training_progress': self.training_progress,
            'model_performance': self.model_performance,
            'selected_features_count': len(self.selected_features),
            'total_features_count': len(self.feature_weights),
            'active_features': [f for f, w in self.feature_weights.items() if w == self.active_weight],
            'inactive_features': [f for f, w in self.feature_weights.items() if w == self.inactive_weight],
            'feature_importance': self.feature_importance
        }