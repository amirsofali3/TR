"""
Trading Engine - Main trading logic and signal generation
"""

import asyncio
import time
from typing import Dict, List, Optional, Tuple, Any
from loguru import logger
from datetime import datetime, timedelta
import pandas as pd

from config.settings import *

class TradingEngine:
    """Main trading engine that coordinates all components"""
    
    def __init__(self, data_collector, indicator_engine, ml_model):
        self.data_collector = data_collector
        self.indicator_engine = indicator_engine
        self.ml_model = ml_model
        self.risk_manager = None
        
        self.latest_signals = {}
        self.analysis_results = {}
        self.running = False
        
        # Online Learning & Continuous Updates (Complete Pipeline Restructure)
        self.last_retrain_time = None
        self.next_retrain_time = None
        self.accumulated_samples = 0  # Count of new samples since last retrain
        self.online_learning_active = False
        self.last_accuracy_update = None
        
        # Analysis Interval Management (User Feedback Adjustments)
        self.current_analysis_interval = BASE_ANALYSIS_INTERVAL_SEC
        self._pending_interval_change = None
        
        # Collection metrics (User Feedback Adjustments)
        self.collector_active = False
        self.ticks_per_sec = 0.0  # EMA over last 60s
        self._tick_count = 0
        self._tick_time_window_start = None
        
        # Import risk manager here to avoid circular imports
        from src.risk_management.risk_manager import RiskManagementSystem
        self.risk_manager = RiskManagementSystem()
    
    async def initialize(self):
        """Initialize trading engine (Complete Pipeline Restructure)"""
        try:
            logger.info("Initializing trading engine...")
            
            # Initialize risk manager
            await self.risk_manager.initialize()
            
            # Start risk manager monitoring
            asyncio.create_task(self.risk_manager.start_monitoring())
            
            # NOTE: Initial training is now handled by the main flow orchestration
            # Do not trigger training here - wait for bootstrap collection to complete
            
            logger.success("Trading engine initialized (bootstrap mode)")
            
        except Exception as e:
            logger.error(f"Failed to initialize trading engine: {e}")
            raise
    
    async def start_post_bootstrap_training(self, features_df=None, labels=None, recent_data_df=None, must_keep_features=None, rfe_candidates=None):
        """Start training and online learning (OHLCV-only mode)"""
        try:
            logger.info("[TRAIN] Starting OHLCV-only training phase...")
            
            # Use provided data if available (OHLCV-only mode)
            if features_df is not None and labels is not None:
                logger.info("[TRAIN] Using preloaded OHLCV features and labels")
                await self.trigger_ohlcv_training(features_df, labels, recent_data_df, must_keep_features, rfe_candidates)
            else:
                # Fallback to legacy training method
                logger.info("[TRAIN] Falling back to legacy training method")
                await self.trigger_initial_training()
            
            # Start online learning if training was successful
            if self.ml_model.is_trained:
                await self.start_online_learning()
            
            logger.success("[TRAIN] OHLCV-only training phase completed")
            
        except Exception as e:
            logger.error(f"[TRAIN] OHLCV-only training failed: {e}")
            raise
    
    async def trigger_ohlcv_training(self, features_df, labels, recent_data_df=None, must_keep_features=None, rfe_candidates=None):
        """Trigger OHLCV-only model training with preloaded data"""
        try:
            logger.info("[TRAIN] Starting OHLCV-only model training...")
            
            if len(features_df) < MIN_INITIAL_TRAIN_SAMPLES:
                logger.warning(f"Insufficient data for OHLCV training ({len(features_df)} < {MIN_INITIAL_TRAIN_SAMPLES})")
                return
            
            # Ensure features and labels are aligned
            min_length = min(len(features_df), len(labels))
            features_aligned = features_df.iloc[:min_length]
            labels_aligned = labels.iloc[:min_length]
            
            logger.info(f"[TRAIN] Training with {len(features_aligned)} samples, {len(features_aligned.columns)} features")
            
            # Train the model with OHLCV data
            if rfe_candidates is None:
                rfe_candidates = self.indicator_engine.get_rfe_candidates()
            
            # Prepare recent data for RFE if provided
            recent_features_df = None
            recent_labels = None
            if recent_data_df is not None:
                # Calculate indicators on recent data
                recent_indicators = await self.indicator_engine.calculate_all_indicators(recent_data_df)
                if recent_indicators and 'dataframe' in recent_indicators:
                    recent_features_df = recent_indicators['dataframe']
                    recent_labels = self.generate_training_labels(recent_features_df)
                    
                    # Align recent data
                    if len(recent_features_df) > 0 and len(recent_labels) > 0:
                        min_recent_length = min(len(recent_features_df), len(recent_labels))
                        recent_features_df = recent_features_df.iloc[:min_recent_length]
                        recent_labels = recent_labels.iloc[:min_recent_length]
                        
                        logger.info(f"[TRAIN] Recent data for RFE: {len(recent_features_df)} samples")
            
            # Start model training
            training_success = await self.ml_model.train_model(
                features_aligned, 
                labels_aligned, 
                rfe_candidates,
                X_recent=recent_features_df,
                y_recent=recent_labels
            )
            
            if training_success:
                logger.success("[TRAIN] OHLCV-only model training completed successfully")
            else:
                logger.warning("[TRAIN] OHLCV-only model training failed")
                
        except Exception as e:
            logger.error(f"[TRAIN] OHLCV-only training failed: {e}")
            # Set cooldown for retry
            self.ml_model.training_cooldown_until = datetime.now() + timedelta(seconds=TRAIN_RETRY_COOLDOWN_SEC)
            raise
    
    def generate_training_labels(self, df):
        """Generate training labels for recent data (helper method)"""
        try:
            if 'close' not in df.columns:
                return pd.Series(['HOLD'] * len(df), index=df.index)
            
            # Calculate future returns
            future_returns = df['close'].pct_change().shift(-1)
            
            # Create labels
            labels = pd.Series(index=df.index, dtype='object')
            labels[future_returns > 0.002] = 'BUY'
            labels[future_returns < -0.002] = 'SELL'
            labels[labels.isna()] = 'HOLD'
            
            return labels[:-1]  # Remove last row
            
        except Exception as e:
            logger.error(f"[TRAIN] Failed to generate labels: {e}")
            return pd.Series(['HOLD'] * (len(df) - 1), index=df.index[:-1])
    
    async def trigger_initial_training(self):
        """Trigger initial model training if model is not trained and sufficient data available (MySQL migration)"""
        try:
            if self.ml_model.is_trained:
                logger.info("ML model already trained, skipping initial training")
                return
            
            # Check if training cooldown is active
            now = datetime.now()
            if (self.ml_model.training_cooldown_until and now < self.ml_model.training_cooldown_until):
                remaining = self.ml_model.training_cooldown_until - now
                logger.info(f"Training cooldown active, {remaining.total_seconds():.0f}s remaining")
                return
            elif self.ml_model.training_cooldown_until and now >= self.ml_model.training_cooldown_until:
                # Cooldown expired, clear it
                logger.info("Training cooldown expired, clearing retry scheduling")
                self.ml_model.training_cooldown_until = None
                self.ml_model.next_retry_at = None
            
            logger.info("ML model not trained, attempting initial training...")
            
            # Get data for primary symbol
            primary_symbol = 'BTCUSDT'
            historical_data = await self.data_collector.get_historical_data(
                primary_symbol, DEFAULT_TIMEFRAME, ML_LOOKBACK_PERIODS
            )
            
            if historical_data is None or len(historical_data) < MIN_INITIAL_TRAIN_SAMPLES:
                logger.warning(f"Insufficient data for initial training ({len(historical_data) if historical_data is not None else 0} < {MIN_INITIAL_TRAIN_SAMPLES})")
                return
            
            # Calculate indicators
            logger.info("Calculating indicators for initial training...")
            indicators = await self.indicator_engine.calculate_all_indicators(
                historical_data, primary_symbol
            )
            
            if not indicators:
                logger.warning("No indicators available for initial training")
                return
            
            # Get RFE eligible features
            rfe_eligible = self.indicator_engine.get_rfe_eligible_indicators()
            
            # Trigger training asynchronously to avoid blocking initialization
            asyncio.create_task(self._perform_initial_training(indicators, primary_symbol, rfe_eligible))
            
        except Exception as e:
            logger.error(f"Initial training trigger failed: {e}")
    
    async def _perform_initial_training(self, indicators: Dict, symbol: str, rfe_eligible: List[str]):
        """Perform the actual initial training (MySQL migration)"""
        try:
            logger.info("[TRAIN] Starting initial model training...")
            
            success = await self.ml_model.retrain_online(indicators, symbol, rfe_eligible)
            
            if success:
                logger.success("[TRAIN] Initial model training completed successfully!")
                # Clear any previous retry scheduling
                self.ml_model.next_retry_at = None
            else:
                logger.warning("[TRAIN] Initial model training failed, will use fallback signals")
                # Set cooldown to prevent immediate retry using new configuration
                from config.settings import TRAIN_RETRY_COOLDOWN_SEC
                cooldown_seconds = TRAIN_RETRY_COOLDOWN_SEC
                retry_time = datetime.now() + timedelta(seconds=cooldown_seconds)
                
                self.ml_model.training_cooldown_until = retry_time
                self.ml_model.next_retry_at = retry_time
                
                logger.info(f"[TRAIN] Next training retry scheduled for: {retry_time.strftime('%Y-%m-%d %H:%M:%S')}")
                
        except Exception as e:
            logger.error(f"[TRAIN] Initial training execution failed: {e}")
            # Set cooldown on failure using new configuration  
            from config.settings import TRAIN_RETRY_COOLDOWN_SEC
            cooldown_seconds = TRAIN_RETRY_COOLDOWN_SEC
            retry_time = datetime.now() + timedelta(seconds=cooldown_seconds)
            
            self.ml_model.training_cooldown_until = retry_time
            self.ml_model.next_retry_at = retry_time
            
            logger.info(f"[TRAIN] Next training retry scheduled for: {retry_time.strftime('%Y-%m-%d %H:%M:%S')}")
    
    async def analyze_markets(self):
        """Analyze all supported markets and generate signals"""
        try:
            logger.info("Starting market analysis...")
            
            # Check for training retry if model not trained
            if not self.ml_model.is_trained:
                await self.check_and_retry_training()
            
            # Get current prices first for risk management
            current_prices = {}
            for symbol in SUPPORTED_PAIRS:
                price_data = await self.data_collector.get_real_time_price(symbol)
                if price_data:
                    current_prices[symbol] = price_data['price']
            
            # Update positions with current prices
            if current_prices:
                await self.risk_manager.update_positions(current_prices)
            
            # Analyze each symbol
            analysis_tasks = []
            for symbol in SUPPORTED_PAIRS:
                task = asyncio.create_task(self.analyze_symbol(symbol))
                analysis_tasks.append(task)
            
            # Wait for all analysis to complete
            results = await asyncio.gather(*analysis_tasks, return_exceptions=True)
            
            # Process results
            successful_analyses = 0
            for i, result in enumerate(results):
                symbol = SUPPORTED_PAIRS[i]
                if isinstance(result, Exception):
                    logger.warning(f"Analysis failed for {symbol}: {result}")
                    continue
                
                if result:
                    self.analysis_results[symbol] = result
                    successful_analyses += 1
                    
                    # Generate trading signal if confidence is high enough
                    await self.process_signal(symbol, result)
            
            logger.info(f"Market analysis completed: {successful_analyses}/{len(SUPPORTED_PAIRS)} successful")
            
        except Exception as e:
            logger.error(f"Market analysis failed: {e}")
    
    async def analyze_symbol(self, symbol: str) -> Optional[Dict[str, Any]]:
        """Analyze a single symbol and generate prediction (MySQL migration improved + Phase 3 data fallback)"""
        try:
            # Phase 3: Get historical data with fallback logic
            from config.settings import MIN_ANALYSIS_CANDLES
            
            # First attempt: get data with normal lookback
            historical_data = await self.data_collector.get_historical_data(
                symbol, DEFAULT_TIMEFRAME, ML_LOOKBACK_PERIODS
            )
            
            # Phase 3 fallback: If insufficient data, try fetching more from DB
            if historical_data is None or len(historical_data) < MIN_ANALYSIS_CANDLES:
                current_size = len(historical_data) if historical_data is not None else 0
                logger.warning(f"Insufficient live data for {symbol} ({current_size} samples), fetching more from historical DB...")
                
                # Try fetching more data from database
                extended_data = await self.data_collector.get_historical_data(
                    symbol, DEFAULT_TIMEFRAME, MIN_ANALYSIS_CANDLES * 2  # Get more to ensure sufficiency
                )
                
                if extended_data is not None and len(extended_data) >= MIN_ANALYSIS_CANDLES:
                    historical_data = extended_data
                    logger.info(f"Extended data fetch successful: {len(historical_data)} samples for {symbol}")
                else:
                    logger.warning(f"Still insufficient data for {symbol} even after extended fetch")
            
            # Final check for minimum data requirement
            if historical_data is None or len(historical_data) < 100:
                logger.warning(f"Insufficient data for {symbol}: {len(historical_data) if historical_data is not None else 0} samples")
                return None
            
            logger.debug(f"Analysis data for {symbol}: {len(historical_data)} samples")
            
            # Calculate all indicators
            indicators = await self.indicator_engine.calculate_all_indicators(
                historical_data, symbol
            )
            
            if not indicators:
                logger.warning(f"No indicators calculated for {symbol}")
                return None
            
            # Check if model is trained, use fallback if not (MySQL migration)
            if not self.ml_model.is_trained:
                return await self.generate_fallback_signal(symbol, historical_data, indicators)
            
            # Prepare feature data (use last row for prediction)
            # Collect valid indicators first to avoid DataFrame fragmentation
            valid_indicators = {}
            for indicator_name, values in indicators.items():
                if isinstance(values, (pd.Series, list)):
                    if len(values) > 0:
                        valid_indicators[indicator_name] = values
            
            # Create DataFrame all at once to avoid fragmentation warning
            feature_df = pd.DataFrame(valid_indicators)
            
            if len(feature_df) == 0:
                logger.warning(f"No feature data for {symbol}")
                return None
            
            # Get last row for prediction
            latest_features = feature_df.tail(1)
            
            # Make prediction
            prediction_result = await self.ml_model.predict(latest_features)
            
            # Get current price
            price_data = await self.data_collector.get_real_time_price(symbol)
            current_price = price_data['price'] if price_data else None
            
            # Prepare analysis result
            analysis_result = {
                'symbol': symbol,
                'timestamp': datetime.now().isoformat(),
                'current_price': current_price,
                'prediction': prediction_result['prediction'],
                'confidence': prediction_result['confidence'],
                'probabilities': prediction_result['probabilities'],
                'feature_count': len(latest_features.columns),
                'data_quality': self.assess_data_quality(historical_data, indicators),
                'fallback': False  # MySQL migration - indicate this is ML prediction
            }
            
            logger.debug(f"Analysis for {symbol}: {prediction_result['prediction']} ({prediction_result['confidence']:.1f}%)")
            
            return analysis_result
            
        except Exception as e:
            logger.error(f"Symbol analysis failed for {symbol}: {e}")
            return None
    
    async def generate_fallback_signal(self, symbol: str, historical_data: pd.DataFrame, indicators: Dict) -> Optional[Dict[str, Any]]:
        """Generate simple fallback signal when ML model not trained (MySQL migration)"""
        try:
            logger.debug(f"Generating fallback signal for {symbol} (model not trained)")
            
            # Simple fallback strategy: SMA(14) vs SMA(50) crossover + RSI thresholds
            close_prices = historical_data['close']
            
            # Calculate simple moving averages
            sma_14 = close_prices.rolling(window=14).mean()
            sma_50 = close_prices.rolling(window=50).mean()
            
            # Calculate RSI (simplified)
            delta = close_prices.diff()
            gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
            loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
            rs = gain / loss
            rsi = 100 - (100 / (1 + rs))
            
            # Get current values
            current_price = close_prices.iloc[-1]
            current_sma_14 = sma_14.iloc[-1]
            current_sma_50 = sma_50.iloc[-1]
            current_rsi = rsi.iloc[-1]
            
            # Generate signal
            if pd.isna(current_sma_14) or pd.isna(current_sma_50) or pd.isna(current_rsi):
                prediction = 'HOLD'
                confidence = 50.0
            elif current_sma_14 > current_sma_50 and current_rsi < 70:
                prediction = 'BUY'
                confidence = min(65.0, 55.0 + (current_sma_14 - current_sma_50) / current_sma_50 * 100)
            elif current_sma_14 < current_sma_50 and current_rsi > 30:
                prediction = 'SELL'
                confidence = min(65.0, 55.0 + (current_sma_50 - current_sma_14) / current_sma_50 * 100)
            else:
                prediction = 'HOLD'
                confidence = 55.0
            
            # Create fallback analysis result
            analysis_result = {
                'symbol': symbol,
                'timestamp': datetime.now().isoformat(),
                'current_price': float(current_price),
                'prediction': prediction,
                'confidence': float(confidence),
                'probabilities': {
                    'BUY': 0.5 if prediction == 'BUY' else 0.25,
                    'HOLD': 0.5 if prediction == 'HOLD' else 0.25,
                    'SELL': 0.5 if prediction == 'SELL' else 0.25
                },
                'feature_count': 3,  # SMA14, SMA50, RSI
                'data_quality': self.assess_data_quality(historical_data, indicators),
                'fallback': True,  # MySQL migration - indicate this is fallback
                'fallback_indicators': {
                    'sma_14': float(current_sma_14) if not pd.isna(current_sma_14) else None,
                    'sma_50': float(current_sma_50) if not pd.isna(current_sma_50) else None,
                    'rsi': float(current_rsi) if not pd.isna(current_rsi) else None
                }
            }
            
            logger.debug(f"Fallback signal for {symbol}: {prediction} ({confidence:.1f}%)")
            return analysis_result
            
        except Exception as e:
            logger.error(f"Fallback signal generation failed for {symbol}: {e}")
            return None
    
    def assess_data_quality(self, historical_data: pd.DataFrame, indicators: Dict) -> Dict[str, Any]:
        """Assess the quality of data and indicators"""
        try:
            data_points = len(historical_data)
            indicator_coverage = len([v for v in indicators.values() if v is not None])
            total_indicators = len(indicators)
            
            # Check for recent data
            if len(historical_data) > 0:
                latest_timestamp = historical_data['timestamp'].iloc[-1]
                current_timestamp = time.time() * 1000
                data_age_hours = (current_timestamp - latest_timestamp) / (1000 * 3600)
            else:
                data_age_hours = float('inf')
            
            quality_score = min(100, (
                (min(data_points, 500) / 500) * 40 +  # Data quantity (40%)
                (indicator_coverage / max(total_indicators, 1)) * 40 +  # Indicator coverage (40%)
                (100 if data_age_hours < 24 else max(0, 100 - data_age_hours)) * 0.2  # Data freshness (20%)
            ))
            
            return {
                'score': quality_score,
                'data_points': data_points,
                'indicator_coverage': f"{indicator_coverage}/{total_indicators}",
                'data_age_hours': data_age_hours,
                'is_acceptable': quality_score >= 70
            }
            
        except Exception as e:
            logger.error(f"Data quality assessment failed: {e}")
            return {'score': 0, 'is_acceptable': False}
    
    async def process_signal(self, symbol: str, analysis: Dict[str, Any]):
        """Process trading signal and execute if conditions are met"""
        try:
            prediction = analysis['prediction']
            confidence = analysis['confidence']
            current_price = analysis['current_price']
            
            if current_price is None:
                logger.warning(f"No current price for {symbol}")
                return
            
            # Store latest signal
            self.latest_signals[symbol] = {
                'timestamp': analysis['timestamp'],
                'prediction': prediction,
                'confidence': confidence,
                'price': current_price,
                'executed': False
            }
            
            # Check if confidence meets threshold for execution
            if confidence < CONFIDENCE_THRESHOLD:
                logger.debug(f"Signal for {symbol} below confidence threshold: {confidence:.1f}% < {CONFIDENCE_THRESHOLD}%")
                return
            
            # Check if we should execute the signal
            should_execute = await self.should_execute_signal(symbol, prediction, confidence, current_price)
            
            if should_execute:
                await self.execute_signal(symbol, prediction, confidence, current_price)
                self.latest_signals[symbol]['executed'] = True
            
        except Exception as e:
            logger.error(f"Signal processing failed for {symbol}: {e}")
    
    async def should_execute_signal(self, symbol: str, prediction: str, confidence: float, price: float) -> bool:
        """Determine if a signal should be executed"""
        try:
            # Skip HOLD signals
            if prediction == 'HOLD':
                return False
            
            # Check if we're in demo mode and have sufficient balance
            if DEMO_MODE:
                if self.risk_manager.portfolio_balance < 100:  # Minimum $100 for new trades
                    logger.warning("Insufficient demo balance for new trades")
                    return False
            
            # Check position limits
            config = self.risk_manager.get_tp_sl_config(symbol)
            active_positions = [p for p in self.risk_manager.positions.values() 
                             if p.symbol == symbol and p.is_active]
            
            if len(active_positions) >= config.max_positions:
                logger.debug(f"Max positions reached for {symbol}")
                return False
            
            # Check for conflicting positions
            for position in active_positions:
                if (prediction == 'BUY' and position.side == 'SELL') or \
                   (prediction == 'SELL' and position.side == 'BUY'):
                    logger.debug(f"Conflicting position exists for {symbol}")
                    return False
            
            # Additional filters can be added here
            # e.g., market conditions, volatility checks, correlation with other positions
            
            return True
            
        except Exception as e:
            logger.error(f"Signal execution check failed: {e}")
            return False
    
    async def execute_signal(self, symbol: str, prediction: str, confidence: float, price: float):
        """Execute trading signal"""
        try:
            logger.info(f"Executing {prediction} signal for {symbol} at {price:.4f} (confidence: {confidence:.1f}%)")
            
            if DEMO_MODE:
                # Demo mode - simulate trade
                position_id = await self.risk_manager.open_position(
                    symbol=symbol,
                    side=prediction,  # 'BUY' or 'SELL'
                    price=price,
                    confidence=confidence
                )
                
                if position_id:
                    logger.success(f"Demo position opened: {position_id}")
                else:
                    logger.error(f"Failed to open demo position for {symbol}")
            else:
                # Live mode - actual trade execution
                # This would integrate with actual exchange API
                logger.warning("Live trading not implemented - use demo mode")
            
        except Exception as e:
            logger.error(f"Signal execution failed for {symbol}: {e}")
    
    async def retrain_models(self):
        """Retrain ML models with recent market data"""
        try:
            logger.info("Starting model retraining...")
            
            # Get RFE eligible features
            rfe_eligible = self.indicator_engine.get_rfe_eligible_indicators()
            
            # Retrain for each major symbol
            major_symbols = ['BTCUSDT', 'ETHUSDT', 'ADAUSDT', 'DOTUSDT', 'LINKUSDT']
            
            for symbol in major_symbols:
                try:
                    # Get recent data
                    historical_data = await self.data_collector.get_historical_data(
                        symbol, DEFAULT_TIMEFRAME, ML_LOOKBACK_PERIODS
                    )
                    
                    if historical_data is None or len(historical_data) < 200:
                        logger.warning(f"Insufficient data for retraining {symbol}")
                        continue
                    
                    # Calculate indicators
                    indicators = await self.indicator_engine.calculate_all_indicators(
                        historical_data, symbol
                    )
                    
                    if not indicators:
                        logger.warning(f"No indicators for retraining {symbol}")
                        continue
                    
                    # Retrain model
                    success = await self.ml_model.retrain_online(indicators, symbol, rfe_eligible)
                    
                    if success:
                        logger.info(f"Model retrained successfully for {symbol}")
                    else:
                        logger.warning(f"Model retraining failed for {symbol}")
                    
                    # Break after first successful retrain to avoid overtraining
                    if success:
                        break
                        
                except Exception as e:
                    logger.error(f"Retraining failed for {symbol}: {e}")
                    continue
            
            logger.info("Model retraining completed")
            
        except Exception as e:
            logger.error(f"Model retraining failed: {e}")
    
    async def update_data(self):
        """Update market data and retrain model periodically"""
        try:
            # Update market data
            await self.data_collector.update_data()
            
            # Check if we need to retrain (every 6 hours)
            current_hour = datetime.now().hour
            if current_hour % 6 == 0:  # Every 6 hours
                await self.retrain_models()
                
        except Exception as e:
            logger.error(f"Data update failed: {e}")
    
    def get_latest_signals(self, limit: int = 20) -> List[Dict[str, Any]]:
        """Get latest trading signals for web interface"""
        try:
            signals = []
            
            # Sort signals by timestamp
            sorted_signals = sorted(
                self.latest_signals.items(),
                key=lambda x: x[1]['timestamp'],
                reverse=True
            )
            
            for symbol, signal in sorted_signals[:limit]:
                signals.append({
                    'symbol': symbol,
                    'timestamp': signal['timestamp'],
                    'prediction': signal['prediction'],
                    'confidence': signal['confidence'],
                    'price': signal['price'],
                    'executed': signal['executed']
                })
            
            return signals
            
        except Exception as e:
            logger.error(f"Failed to get latest signals: {e}")
            return []
    
    # ==================== ONLINE LEARNING & CONTINUOUS UPDATES ====================
    
    async def start_online_learning(self):
        """Start continuous online learning process (Complete Pipeline Restructure)"""
        try:
            self.online_learning_active = True
            self.last_retrain_time = datetime.now()
            self.accumulated_samples = 0
            self._schedule_next_retrain()
            
            logger.info("[ONLINE] Online learning activated")
            logger.info(f"[ONLINE] Next retrain scheduled for: {self.next_retrain_time}")
            logger.info(f"[ONLINE] Retrain triggers: {MIN_NEW_SAMPLES_FOR_RETRAIN} samples OR {ONLINE_RETRAIN_INTERVAL_SEC}s")
            
            # Start background tasks
            asyncio.create_task(self._online_learning_loop())
            asyncio.create_task(self._accuracy_update_loop())
            
        except Exception as e:
            logger.error(f"[ONLINE] Failed to start online learning: {e}")
    
    def _schedule_next_retrain(self):
        """Schedule the next automatic retrain"""
        self.next_retrain_time = datetime.now() + timedelta(seconds=ONLINE_RETRAIN_INTERVAL_SEC)
    
    async def _online_learning_loop(self):
        """Background loop for online learning monitoring"""
        try:
            while self.online_learning_active and self.running:
                try:
                    # Check if retrain is needed
                    should_retrain = await self._should_trigger_retrain()
                    
                    if should_retrain:
                        await self._trigger_online_retrain()
                    
                    # Sleep for a minute between checks
                    await asyncio.sleep(60)
                    
                except Exception as e:
                    logger.debug(f"[ONLINE] Online learning loop error: {e}")
                    await asyncio.sleep(60)
                    
        except Exception as e:
            logger.error(f"[ONLINE] Online learning loop failed: {e}")
    
    async def _accuracy_update_loop(self):
        """Background loop for live accuracy updates"""
        try:
            while self.online_learning_active and self.running:
                try:
                    # Update live accuracy from recent predictions
                    self._update_live_accuracy()
                    self.last_accuracy_update = datetime.now()
                    
                    # Sleep for accuracy update interval
                    await asyncio.sleep(ACCURACY_UPDATE_INTERVAL_SEC)
                    
                except Exception as e:
                    logger.debug(f"[ONLINE] Accuracy update loop error: {e}")
                    await asyncio.sleep(ACCURACY_UPDATE_INTERVAL_SEC)
                    
        except Exception as e:
            logger.error(f"[ONLINE] Accuracy update loop failed: {e}")
    
    async def _should_trigger_retrain(self) -> bool:
        """Check if automatic retrain should be triggered"""
        try:
            now = datetime.now()
            
            # Time-based trigger
            if self.next_retrain_time and now >= self.next_retrain_time:
                logger.info("[ONLINE] Time-based retrain trigger activated")
                return True
            
            # Sample-count trigger
            if self.accumulated_samples >= MIN_NEW_SAMPLES_FOR_RETRAIN:
                logger.info(f"[ONLINE] Sample-count retrain trigger activated ({self.accumulated_samples} >= {MIN_NEW_SAMPLES_FOR_RETRAIN})")
                return True
            
            return False
            
        except Exception as e:
            logger.debug(f"[ONLINE] Error checking retrain triggers: {e}")
            return False
    
    async def _trigger_online_retrain(self):
        """Trigger an automatic online retrain"""
        try:
            logger.info("[ONLINE] Starting automatic online retrain...")
            
            # Get recent data for retraining
            primary_symbol = 'BTCUSDT'
            historical_data = await self.data_collector.get_historical_data(
                primary_symbol, DEFAULT_TIMEFRAME, ML_LOOKBACK_PERIODS
            )
            
            if historical_data is None or len(historical_data) < MIN_INITIAL_TRAIN_SAMPLES:
                logger.warning(f"[ONLINE] Insufficient data for retrain ({len(historical_data) if historical_data is not None else 0} samples)")
                self._schedule_next_retrain()  # Try again later
                return
            
            # Calculate indicators
            indicators = await self.indicator_engine.calculate_all_indicators(
                historical_data, primary_symbol
            )
            
            if not indicators:
                logger.warning("[ONLINE] No indicators available for retrain")
                self._schedule_next_retrain()
                return
            
            # Get RFE eligible features
            rfe_eligible = self.indicator_engine.get_rfe_eligible_indicators()
            
            # Perform retrain
            retrain_success = await self.ml_model.retrain_online(indicators, primary_symbol, rfe_eligible)
            
            if retrain_success:
                logger.success("[ONLINE] Online retrain completed successfully")
                self.accumulated_samples = 0  # Reset sample counter
                self.last_retrain_time = datetime.now()
                self._schedule_next_retrain()
            else:
                logger.warning("[ONLINE] Online retrain failed")
                # Schedule retry sooner
                self.next_retrain_time = datetime.now() + timedelta(seconds=300)  # 5 minutes
                
        except Exception as e:
            logger.error(f"[ONLINE] Online retrain failed: {e}")
            self._schedule_next_retrain()
    
    def _update_live_accuracy(self):
        """Update live accuracy and emit WebSocket event if available"""
        try:
            # The ML model handles its own sliding window accuracy
            # This method can be used to trigger WebSocket updates
            pass
            
        except Exception as e:
            logger.debug(f"[ONLINE] Live accuracy update error: {e}")
    
    def increment_sample_count(self, count: int = 1):
        """Increment the accumulated sample count for retrain trigger"""
        self.accumulated_samples += count
    
    async def manual_retrain(self, retrain_type: str = "fast") -> Dict[str, Any]:
        """Manually trigger a retrain (OHLCV-only mode)"""
        try:
            logger.info(f"[RETRAIN] Starting manual retrain (OHLCV-only mode)")
            
            # Import candle data manager
            from src.data_access.candle_data_manager import candle_data_manager
            
            # Step 1: Fetch latest 1000 candles for RFE
            logger.info(f"[RETRAIN] Fetching latest {RFE_SELECTION_CANDLES} candles for RFE...")
            recent_data = await candle_data_manager.get_recent_candles(SUPPORTED_PAIRS, RFE_SELECTION_CANDLES)
            
            # Step 2: Load full historical data for final training
            logger.info("[RETRAIN] Loading full historical data for training...")
            full_data = await candle_data_manager.get_full_historical_data(SUPPORTED_PAIRS)
            
            if full_data.empty:
                return {
                    "success": False,
                    "message": "No historical data available for retrain",
                    "error": "NO_DATA"
                }
            
            # Check minimum data requirements
            if len(full_data) < MIN_INITIAL_TRAIN_SAMPLES:
                return {
                    "success": False,
                    "message": f"Insufficient data for retrain ({len(full_data)} < {MIN_INITIAL_TRAIN_SAMPLES})",
                    "error": "INSUFFICIENT_DATA",
                    "samples_found": len(full_data),
                    "samples_required": MIN_INITIAL_TRAIN_SAMPLES
                }
            
            # Step 3: Prepare data for indicator calculation
            prepared_full_data = candle_data_manager.prepare_features_dataframe(full_data)
            prepared_recent_data = None
            
            if not recent_data.empty:
                prepared_recent_data = candle_data_manager.prepare_features_dataframe(recent_data)
                logger.info(f"[RETRAIN] RFE window: {len(prepared_recent_data)} recent candles")
            else:
                logger.warning("[RETRAIN] No recent data for RFE, using full dataset")
            
            # Step 4: Calculate indicators on full dataset
            logger.info("[RETRAIN] Calculating OHLCV indicators...")
            indicator_results = await self.indicator_engine.calculate_all_indicators(prepared_full_data)
            
            if not indicator_results or 'dataframe' not in indicator_results:
                return {
                    "success": False,
                    "message": "Failed to calculate indicators for retrain",
                    "error": "INDICATOR_CALCULATION_FAILED"
                }
            
            features_df = indicator_results['dataframe']
            logger.info(f"[RETRAIN] Calculated indicators: {len(features_df.columns)} features, {len(features_df)} samples")
            
            # Step 5: Generate training labels
            labels = self.generate_training_labels(features_df)
            
            # Step 6: Get feature lists
            must_keep_features = self.indicator_engine.get_must_keep_features()
            rfe_candidates = self.indicator_engine.get_rfe_candidates()
            
            logger.info(f"[RETRAIN] Must keep: {len(must_keep_features)}, RFE candidates: {len(rfe_candidates)}")
            
            # Step 7: Run retraining with OHLCV data
            await self.trigger_ohlcv_training(features_df, labels, prepared_recent_data, must_keep_features, rfe_candidates)
            
            if self.ml_model.is_trained:
                # Reset counters
                self.accumulated_samples = 0
                self.last_retrain_time = datetime.now()
                self._schedule_next_retrain()
                
                logger.success(f"[RETRAIN] OHLCV-only retrain completed successfully")
                
                return {
                    "success": True,
                    "message": "OHLCV-only retrain completed successfully",
                    "model_version": self.ml_model.model_version,
                    "training_count": self.ml_model.training_count,
                    "samples_used": len(features_df),
                    "selected_features": len(self.ml_model.selected_features),
                    "accuracy": self.ml_model.model_performance.get('accuracy', 0.0),
                    "mode": "ohlcv_only",
                    "timeframe": "1m",
                    "rfe_window_size": len(prepared_recent_data) if prepared_recent_data is not None else "full_dataset"
                }
            else:
                return {
                    "success": False,
                    "message": "OHLCV-only retrain failed during training",
                    "error": "TRAINING_FAILED"
                }
                
        except Exception as e:
            logger.error(f"[RETRAIN] OHLCV-only retrain failed: {e}")
            return {
                "success": False,
                "message": f"OHLCV-only retrain failed: {str(e)}",
                "error": "EXCEPTION"
            }
    
    def get_online_learning_status(self) -> Dict[str, Any]:
        """Get online learning status information"""
        return {
            "active": self.online_learning_active,
            "last_retrain_time": self.last_retrain_time.isoformat() if self.last_retrain_time else None,
            "next_retrain_time": self.next_retrain_time.isoformat() if self.next_retrain_time else None,
            "accumulated_samples": self.accumulated_samples,
            "min_samples_for_retrain": MIN_NEW_SAMPLES_FOR_RETRAIN,
            "retrain_interval_sec": ONLINE_RETRAIN_INTERVAL_SEC,
            "last_accuracy_update": self.last_accuracy_update.isoformat() if self.last_accuracy_update else None
        }
    
    # ==================== USER FEEDBACK ADJUSTMENTS ====================
    
    def _check_pending_interval_change(self):
        """Check and apply any pending analysis interval changes"""
        if self._pending_interval_change is not None:
            old_interval = self.current_analysis_interval
            new_interval = self._pending_interval_change
            
            # Apply the interval change
            self.current_analysis_interval = new_interval
            self._pending_interval_change = None
            
            logger.info(f"[ANALYSIS] Applied interval change: {old_interval}s -> {new_interval}s")
    
    def _get_indicator_progress(self) -> Dict[str, Any]:
        """Get indicator computation progress (separate from training progress)"""
        try:
            indicators_summary = self.indicator_engine.get_indicators_summary()
            
            # Calculate skipped indicators
            skipped = indicators_summary.get('skipped', [])
            
            return {
                'phase': 'completed' if indicators_summary.get('computed_count', 0) > 0 else 'loading',
                'total_defined': indicators_summary.get('total_defined', 0),
                'to_compute': indicators_summary.get('rfe_candidate_count', 0) + indicators_summary.get('must_keep_count', 0),
                'computed_count': indicators_summary.get('computed_count', 0),
                'percent': round((indicators_summary.get('computed_count', 0) / max(indicators_summary.get('total_defined', 1), 1)) * 100, 1),
                'skipped_count': len(skipped),
                'skipped': [{'name': s.get('name', ''), 'reason': s.get('reason', '')} for s in skipped[:10]]  # Limit to 10 for WebSocket
            }
        except Exception as e:
            logger.error(f"Failed to get indicator progress: {e}")
            return {
                'phase': 'error',
                'total_defined': 0,
                'to_compute': 0,
                'computed_count': 0,
                'percent': 0,
                'skipped_count': 0,
                'skipped': []
            }
    
    def _get_features_info(self) -> Dict[str, Any]:
        """Get features information summary for status API"""
        try:
            model_info = self.ml_model.get_model_info()
            selected_features = model_info.get('active_features', [])
            inactive_features = model_info.get('inactive_features', [])
            
            # For status API, provide summary with truncated lists (first 100)
            return {
                'selected': selected_features[:100],  # Truncate for WebSocket
                'inactive': inactive_features[:100],  # Truncate for WebSocket
                'counts': {
                    'selected': len(selected_features),
                    'inactive': len(inactive_features),
                    'total': len(selected_features) + len(inactive_features)
                }
            }
        except Exception as e:
            logger.error(f"Failed to get features info: {e}")
            return {
                'selected': [],
                'inactive': [],
                'counts': {'selected': 0, 'inactive': 0, 'total': 0}
            }
    
    def _get_ohlcv_features_info(self) -> Dict[str, Any]:
        """Get OHLCV-specific features information for status API"""
        try:
            model_info = self.ml_model.get_model_info()
            selected_features = model_info.get('active_features', [])
            inactive_features = model_info.get('inactive_features', [])
            
            # Get must-keep features from indicator engine
            must_keep_features = self.indicator_engine.get_must_keep_features()
            rfe_candidates = self.indicator_engine.get_rfe_candidates()
            
            # Categorize features with case-insensitive matching
            base_ohlcv = []
            must_keep_other = []
            rfe_selected = []
            rfe_not_selected = []
            
            try:
                from config.settings import BASE_MUST_KEEP_FEATURES
                base_features = BASE_MUST_KEEP_FEATURES + ['timestamp', 'symbol']  # Include meta features
            except ImportError:
                base_features = ["open", "high", "low", "close", "volume", "timestamp", "symbol"]
            
            # Create case-insensitive lookup sets
            base_lower = {b.lower().strip() for b in base_features}
            must_keep_lower = {m.lower().strip() for m in must_keep_features}
            rfe_candidates_lower = {r.lower().strip() for r in rfe_candidates}
            
            # Helper function for case-insensitive feature name sanitization
            def sanitize_feature_name(name: str) -> str:
                """Sanitize feature name for consistent matching"""
                return name.lower().strip()
            
            for feature in selected_features:
                sanitized_feature = sanitize_feature_name(feature)
                if sanitized_feature in base_lower:
                    base_ohlcv.append(feature)
                elif sanitized_feature in must_keep_lower:
                    must_keep_other.append(feature)
                elif sanitized_feature in rfe_candidates_lower:
                    rfe_selected.append(feature)
            
            for feature in inactive_features:
                sanitized_feature = sanitize_feature_name(feature)
                if sanitized_feature in rfe_candidates_lower:
                    rfe_not_selected.append(feature)
            
            return {
                'base_ohlcv': base_ohlcv[:50],  # Core OHLCV features
                'must_keep_other': must_keep_other[:50],  # Other must-keep features
                'rfe_selected': rfe_selected[:50],  # RFE-selected features
                'rfe_not_selected': rfe_not_selected[:50],  # RFE candidates not selected
                'counts': {
                    'base_ohlcv': len(base_ohlcv),
                    'must_keep_other': len(must_keep_other),
                    'rfe_selected': len(rfe_selected),
                    'rfe_not_selected': len(rfe_not_selected),
                    'total_selected': len(selected_features),
                    'total_candidates': len(rfe_candidates),
                    'rfe_target': RFE_N_FEATURES
                },
                'mode': 'ohlcv_only'
            }
        except Exception as e:
            logger.error(f"Failed to get OHLCV features info: {e}")
            return {
                'base_ohlcv': [],
                'must_keep_other': [],
                'rfe_selected': [],
                'rfe_not_selected': [],
                'counts': {
                    'base_ohlcv': 0,
                    'must_keep_other': 0,
                    'rfe_selected': 0,
                    'rfe_not_selected': 0,
                    'total_selected': 0,
                    'total_candidates': 0,
                    'rfe_target': RFE_N_FEATURES
                },
                'mode': 'ohlcv_only'
            }
    
    def _check_insufficient_samples(self) -> bool:
        """Check if there are insufficient samples after sanitization"""
        try:
            model_info = self.ml_model.get_model_info()
            valid_samples = model_info.get('collected_valid_samples', 0)
            return valid_samples > 0 and valid_samples < MIN_VALID_SAMPLES
        except Exception:
            return False
    
    def update_tick_metrics(self):
        """Update ticks per second metrics (called by data collector)"""
        import time
        current_time = time.time()
        
        if self._tick_time_window_start is None:
            self._tick_time_window_start = current_time
            self._tick_count = 0
        
        self._tick_count += 1
        
        # Calculate TPS over 60-second window
        window_duration = current_time - self._tick_time_window_start
        if window_duration >= 60:  # 60-second window
            current_tps = self._tick_count / window_duration
            
            # EMA with alpha=0.3 for smoothing
            if self.ticks_per_sec == 0:
                self.ticks_per_sec = current_tps
            else:
                self.ticks_per_sec = 0.3 * current_tps + 0.7 * self.ticks_per_sec
            
            # Reset window
            self._tick_time_window_start = current_time
            self._tick_count = 0
    
    # ==================== END USER FEEDBACK ADJUSTMENTS ====================
    
    def get_system_status(self) -> Dict[str, Any]:
        """Get comprehensive system status for web interface (OHLCV-only mode)"""
        try:
            # Check for pending interval changes
            self._check_pending_interval_change()
            
            # Model info
            model_info = self.ml_model.get_model_info()
            
            # Portfolio summary
            portfolio = self.risk_manager.get_portfolio_summary()
            
            # Recent analysis results
            recent_analysis = {}
            for symbol, analysis in self.analysis_results.items():
                recent_analysis[symbol] = {
                    'timestamp': analysis['timestamp'],
                    'prediction': analysis['prediction'],
                    'confidence': analysis['confidence'],
                    'current_price': analysis['current_price'],
                    'data_quality': analysis['data_quality']
                }
            
            # OHLCV-only mode: Skip bootstrap collection status
            collection_status = {
                'active': False,
                'progress_percent': 100.0,  # Always complete in OHLCV-only mode
                'status': 'completed',
                'message': 'Bootstrap collection bypassed in OHLCV-only mode',
                'mode': 'ohlcv_only'
            }
            
            # Get indicators summary
            indicators_summary = self.indicator_engine.get_indicators_summary()
            
            # OHLCV-only mode: Enhanced indicators summary
            indicators_summary['mode'] = 'ohlcv_only'
            indicators_summary['csv_source'] = 'ohlcv_only_indicators.csv'
            
            # Get database backend info
            backend_info = {}
            try:
                from src.database.db_manager import db_manager
                backend_info = db_manager.get_backend_info()
            except Exception:
                backend_info = {"error": "Database manager not available"}
            
            # Get online learning status
            online_learning_status = self.get_online_learning_status()
            
            # Get feature information (OHLCV-only mode enhanced)
            features_info = self._get_ohlcv_features_info()
            
            # Check for insufficient samples flag
            insufficient_samples_flag = self._check_insufficient_samples()
            
            return {
                # Core system status (OHLCV-only mode)
                'system_running': self.running,
                'demo_mode': DEMO_MODE,
                'supported_pairs': SUPPORTED_PAIRS,
                'confidence_threshold': CONFIDENCE_THRESHOLD,
                'timestamp': datetime.now().isoformat(),
                'mode': 'ohlcv_only',
                'timeframe': '1m',
                
                # OHLCV-only mode collection status
                'collection': collection_status,
                
                # Analysis configuration (1m timeframe)
                'analysis': {
                    'interval_sec': self.current_analysis_interval,
                    'timeframe': '1m',
                    'raw_collection_bypassed': True
                },
                
                # Training status with OHLCV-only enhancements
                'training': {
                    'phase': model_info.get('current_training_stage', 'idle'),
                    'progress_percent': model_info.get('training_progress', 0.0),
                    'last_training_error': model_info.get('last_training_error'),
                    'last_accuracy': model_info.get('last_accuracy', 0.0),
                    'accuracy_live': model_info.get('accuracy_live', 0.0),
                    'accuracy_window_size': ACCURACY_SLIDING_WINDOW,
                    'accuracy_live_count': model_info.get('accuracy_live_count', 0),
                    'accuracy_warming_up': model_info.get('accuracy_live_count', 0) < ACCURACY_SLIDING_WINDOW,
                    'model_version': model_info.get('model_version', 1),
                    'training_count': model_info.get('training_count', 0),
                    'class_distribution': model_info.get('class_distribution', {}),
                    'class_weights': model_info.get('class_weights', {}),
                    'class_mapping': model_info.get('class_mapping', {"0": "SELL", "1": "HOLD", "2": "BUY"}),
                    'selected_feature_count': model_info.get('selected_feature_count', 0),
                    'collected_valid_samples': model_info.get('collected_valid_samples', 0),
                    'flag_insufficient_samples': insufficient_samples_flag,
                    'warning': "INSUFFICIENT_SAMPLES" if insufficient_samples_flag else None,
                    'features': features_info,
                    'last_training_time': model_info.get('last_training_time'),
                    'next_retrain_at': online_learning_status.get('next_retrain_time'),
                    'accuracy_window_stats': model_info.get('accuracy_window_stats', {}),
                    'training_progress_info': model_info.get('training_progress_info', {}),
                    # OHLCV-only specific fields
                    'rfe_target_features': RFE_N_FEATURES,
                    'rfe_window_size': RFE_SELECTION_CANDLES,
                    'mode': 'ohlcv_only'
                },
                
                # Indicators status (OHLCV-only mode)
                'indicators': indicators_summary,
                
                # Database backend info
                'backend': backend_info,
                
                # Online learning status  
                'online_learning': online_learning_status,
                
                # Legacy fields for backward compatibility
                'model': model_info,
                'portfolio': portfolio,
                'latest_signals': self.get_latest_signals(10),
                'recent_analysis': recent_analysis
            }
            
        except Exception as e:
            logger.error(f"Failed to get system status: {e}")
            return {'error': str(e), 'timestamp': datetime.now().isoformat()}
    
    async def check_and_retry_training(self):
        """Check if training retry should be attempted and execute if needed"""
        try:
            if not self.ml_model or self.ml_model.is_trained:
                return False
            
            now = datetime.now()
            
            # Check if it's time to retry
            if (self.ml_model.next_retry_at and 
                now >= self.ml_model.next_retry_at and
                (not self.ml_model.training_cooldown_until or 
                 now >= self.ml_model.training_cooldown_until)):
                
                logger.info("[TRAIN] Retry cooldown expired, attempting training retry...")
                
                # Clear retry scheduling
                self.ml_model.next_retry_at = None
                self.ml_model.training_cooldown_until = None
                
                # Attempt training retry
                await self.attempt_initial_training()
                
                return True
                
            return False
            
        except Exception as e:
            logger.error(f"[TRAIN] Retry training check failed: {e}")
            return False
    
    async def start_trading(self):
        """Start the trading process"""
        self.running = True
        logger.info("Trading engine started")
    
    async def stop_trading(self):
        """Stop the trading process"""
        self.running = False
        if self.risk_manager:
            self.risk_manager.stop()
        logger.info("Trading engine stopped")
    
    async def stop(self):
        """Stop the trading engine"""
        await self.stop_trading()
