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
    
    async def start_post_bootstrap_training(self):
        """Start training and online learning after bootstrap collection completes"""
        try:
            logger.info("[TRAIN] Starting post-bootstrap training phase...")
            
            # Trigger initial ML model training
            await self.trigger_initial_training()
            
            # Start online learning if training was successful
            if self.ml_model.is_trained:
                await self.start_online_learning()
            
            logger.success("[TRAIN] Post-bootstrap training phase completed")
            
        except Exception as e:
            logger.error(f"[TRAIN] Post-bootstrap training failed: {e}")
            raise
    
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
        """Analyze a single symbol and generate prediction (MySQL migration improved)"""
        try:
            # Get historical data
            historical_data = await self.data_collector.get_historical_data(
                symbol, DEFAULT_TIMEFRAME, ML_LOOKBACK_PERIODS
            )
            
            if historical_data is None or len(historical_data) < 100:
                logger.warning(f"Insufficient data for {symbol}")
                return None
            
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
        """Manually trigger a retrain (Complete Pipeline Restructure)"""
        try:
            if retrain_type == "full":
                return {
                    "success": False,
                    "message": "Full retrain cycle (re-bootstrap) not implemented yet",
                    "error": "NOT_IMPLEMENTED"
                }
            
            # Fast retrain on accumulated data
            logger.info("[RETRAIN] Manual fast retrain triggered...")
            
            # Get recent data
            primary_symbol = 'BTCUSDT'
            historical_data = await self.data_collector.get_historical_data(
                primary_symbol, DEFAULT_TIMEFRAME, ML_LOOKBACK_PERIODS
            )
            
            if historical_data is None or len(historical_data) < MIN_INITIAL_TRAIN_SAMPLES:
                return {
                    "success": False,
                    "message": f"Insufficient data for retrain ({len(historical_data) if historical_data is not None else 0} samples)",
                    "error": "INSUFFICIENT_DATA"
                }
            
            # Calculate indicators
            indicators = await self.indicator_engine.calculate_all_indicators(
                historical_data, primary_symbol
            )
            
            if not indicators:
                return {
                    "success": False,
                    "message": "No indicators available for retrain",
                    "error": "NO_INDICATORS"
                }
            
            # Get RFE eligible features
            rfe_eligible = self.indicator_engine.get_rfe_eligible_indicators()
            
            # Perform retrain
            retrain_success = await self.ml_model.retrain_online(indicators, primary_symbol, rfe_eligible)
            
            if retrain_success:
                # Reset counters
                self.accumulated_samples = 0
                self.last_retrain_time = datetime.now()
                self._schedule_next_retrain()
                
                return {
                    "success": True,
                    "message": "Manual retrain completed successfully",
                    "model_version": self.ml_model.model_version,
                    "training_count": self.ml_model.training_count
                }
            else:
                return {
                    "success": False,
                    "message": "Manual retrain failed during training",
                    "error": "TRAINING_FAILED"
                }
                
        except Exception as e:
            logger.error(f"[RETRAIN] Manual retrain failed: {e}")
            return {
                "success": False,
                "message": f"Manual retrain failed: {str(e)}",
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
    
    # ==================== END ONLINE LEARNING ====================
    
    def get_system_status(self) -> Dict[str, Any]:
        """Get comprehensive system status for web interface (Complete Pipeline Restructure)"""
        try:
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
            
            # Get bootstrap collection status
            collection_status = self.data_collector.get_bootstrap_status()
            
            # Get indicators summary
            indicators_summary = self.indicator_engine.get_indicators_summary()
            
            # Get database backend info
            backend_info = {}
            try:
                from src.database.db_manager import db_manager
                backend_info = db_manager.get_backend_info()
            except Exception:
                backend_info = {"error": "Database manager not available"}
            
            # Get online learning status
            online_learning_status = self.get_online_learning_status()
            
            return {
                # Core system status
                'system_running': self.running,
                'demo_mode': DEMO_MODE,
                'supported_pairs': SUPPORTED_PAIRS,
                'confidence_threshold': CONFIDENCE_THRESHOLD,
                'timestamp': datetime.now().isoformat(),
                
                # Bootstrap collection status (Complete Pipeline Restructure)
                'collection': collection_status,
                
                # Training status with enhanced fields (Complete Pipeline Restructure)
                'training': {
                    'phase': model_info.get('current_training_stage', 'idle'),
                    'progress_percent': model_info.get('training_progress', 0.0),
                    'last_training_error': model_info.get('last_training_error'),
                    'last_accuracy': model_info.get('last_accuracy', 0.0),
                    'accuracy_live': model_info.get('accuracy_live', 0.0),
                    'model_version': model_info.get('model_version', 1),
                    'training_count': model_info.get('training_count', 0),
                    'class_distribution': model_info.get('class_distribution', {}),
                    'class_weights': model_info.get('class_weights', {}),
                    'selected_feature_count': model_info.get('selected_feature_count', 0),
                    'last_training_time': model_info.get('last_training_time'),
                    'next_retrain_at': online_learning_status.get('next_retrain_time'),
                    'accuracy_window_stats': model_info.get('accuracy_window_stats', {}),
                    'training_progress_info': model_info.get('training_progress_info', {})
                },
                
                # Indicators status (Complete Pipeline Restructure)
                'indicators': indicators_summary,
                
                # Database backend info (Complete Pipeline Restructure)
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