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
        
        # Import risk manager here to avoid circular imports
        from src.risk_management.risk_manager import RiskManagementSystem
        self.risk_manager = RiskManagementSystem()
    
    async def initialize(self):
        """Initialize trading engine (MySQL migration improved)"""
        try:
            logger.info("Initializing trading engine...")
            
            # Initialize risk manager
            await self.risk_manager.initialize()
            
            # Start risk manager monitoring
            asyncio.create_task(self.risk_manager.start_monitoring())
            
            # Trigger initial ML model training if needed (MySQL migration)
            await self.trigger_initial_training()
            
            logger.success("Trading engine initialized")
            
        except Exception as e:
            logger.error(f"Failed to initialize trading engine: {e}")
            raise
    
    async def trigger_initial_training(self):
        """Trigger initial model training if model is not trained and sufficient data available (MySQL migration)"""
        try:
            if self.ml_model.is_trained:
                logger.info("ML model already trained, skipping initial training")
                return
            
            # Check if training cooldown is active
            if (self.ml_model.training_cooldown_until and 
                datetime.now() < self.ml_model.training_cooldown_until):
                logger.info("Training cooldown active, skipping initial training")
                return
            
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
            else:
                logger.warning("[TRAIN] Initial model training failed, will use fallback signals")
                # Set cooldown to prevent immediate retry
                self.ml_model.training_cooldown_until = datetime.now() + timedelta(minutes=TRAIN_RETRY_COOLDOWN_MIN)
                
        except Exception as e:
            logger.error(f"[TRAIN] Initial training execution failed: {e}")
            # Set cooldown on failure
            self.ml_model.training_cooldown_until = datetime.now() + timedelta(minutes=TRAIN_RETRY_COOLDOWN_MIN)
    
    async def analyze_markets(self):
        """Analyze all supported markets and generate signals"""
        try:
            logger.info("Starting market analysis...")
            
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
    
    def get_system_status(self) -> Dict[str, Any]:
        """Get system status for web interface"""
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
            
            return {
                'system_running': self.running,
                'demo_mode': DEMO_MODE,
                'supported_pairs': SUPPORTED_PAIRS,
                'confidence_threshold': CONFIDENCE_THRESHOLD,
                'model': model_info,
                'portfolio': portfolio,
                'latest_signals': self.get_latest_signals(10),
                'recent_analysis': recent_analysis,
                'timestamp': datetime.now().isoformat()
            }
            
        except Exception as e:
            logger.error(f"Failed to get system status: {e}")
            return {}
    
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