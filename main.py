"""
Main entry point for the Crypto Trading AI System
"""

import asyncio
import signal
import sys
import os
import time
from loguru import logger
from concurrent.futures import ThreadPoolExecutor

# Add src to path
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

from src.data_collector.binance_collector import BinanceDataCollector
from src.indicators.indicator_engine import IndicatorEngine
from src.ml_model.catboost_model import CatBoostTradingModel
from src.trading_engine.trading_engine import TradingEngine
from src.web_app.app import create_app
from config.settings import *

class TradingSystem:
    """Main Trading System Controller"""
    
    def __init__(self):
        self.running = False
        self.data_collector = None
        self.indicator_engine = None
        self.ml_model = None
        self.trading_engine = None
        self.web_app = None
        self.executor = ThreadPoolExecutor(max_workers=10)
        
        # Setup logging
        self.setup_logging()
        logger.info("Trading System initialized")
    
    def setup_logging(self):
        """Setup logging configuration"""
        logger.remove()  # Remove default handler
        
        # Console logging
        logger.add(
            sys.stderr,
            level=LOG_LEVEL,
            format="<green>{time:YYYY-MM-DD HH:mm:ss}</green> | <level>{level: <8}</level> | <cyan>{name}</cyan>:<cyan>{function}</cyan>:<cyan>{line}</cyan> - <level>{message}</level>"
        )
        
        # File logging
        os.makedirs(os.path.dirname(LOG_FILE), exist_ok=True)
        logger.add(
            LOG_FILE,
            level=LOG_LEVEL,
            format="{time:YYYY-MM-DD HH:mm:ss} | {level: <8} | {name}:{function}:{line} - {message}",
            rotation=f"{MAX_LOG_SIZE} MB",
            retention=f"{BACKUP_COUNT} files"
        )
    
    async def initialize_components(self):
        """Initialize all system components"""
        try:
            logger.info("Initializing system components...")
            
            # Initialize data collector
            self.data_collector = BinanceDataCollector()
            await self.data_collector.initialize()
            logger.info("Data collector initialized")
            
            # Initialize indicator engine
            self.indicator_engine = IndicatorEngine()
            await self.indicator_engine.initialize()
            logger.info("Indicator engine initialized")
            
            # Initialize ML model
            self.ml_model = CatBoostTradingModel()
            await self.ml_model.initialize()
            logger.info("ML model initialized")
            
            # Initialize trading engine
            self.trading_engine = TradingEngine(
                data_collector=self.data_collector,
                indicator_engine=self.indicator_engine,
                ml_model=self.ml_model
            )
            await self.trading_engine.initialize()
            logger.info("Trading engine initialized")
            
            logger.success("All components initialized successfully")
            
        except Exception as e:
            logger.error(f"Failed to initialize components: {e}")
            raise
    
    async def start_web_app(self):
        """Start the web application"""
        try:
            self.web_app = create_app(
                data_collector=self.data_collector,
                indicator_engine=self.indicator_engine,
                ml_model=self.ml_model,
                trading_engine=self.trading_engine
            )
            
            # Run web app in executor to avoid blocking
            self.executor.submit(
                self.web_app.run,
                host=WEB_HOST,
                port=WEB_PORT,
                debug=False
            )
            
            logger.info(f"Web application started on http://{WEB_HOST}:{WEB_PORT}")
            
        except Exception as e:
            logger.error(f"Failed to start web application: {e}")
            raise
    
    async def run_trading_loop(self):
        """Main trading loop"""
        logger.info("Starting main trading loop...")
        
        while self.running:
            try:
                # Update data and analyze markets
                await self.trading_engine.analyze_markets()
                
                # Wait for next update
                await asyncio.sleep(UPDATE_INTERVAL)
                
            except KeyboardInterrupt:
                logger.info("Received keyboard interrupt, stopping...")
                break
            except Exception as e:
                logger.error(f"Error in trading loop: {e}")
                await asyncio.sleep(5)  # Wait before retrying
    
    async def start(self):
        """Start the trading system"""
        try:
            logger.info("Starting Crypto Trading AI System...")
            
            # Initialize all components
            await self.initialize_components()
            
            # Start web application
            await self.start_web_app()
            
            # Set running flag
            self.running = True
            
            # Start trading loop
            await self.run_trading_loop()
            
        except Exception as e:
            logger.error(f"Failed to start trading system: {e}")
            raise
    
    async def stop(self):
        """Stop the trading system"""
        logger.info("Stopping trading system...")
        self.running = False
        
        # Stop components
        if self.trading_engine:
            await self.trading_engine.stop()
        
        if self.data_collector:
            await self.data_collector.stop()
        
        # Shutdown executor
        self.executor.shutdown(wait=True)
        
        logger.info("Trading system stopped")
    
    def signal_handler(self, signum, frame):
        """Handle system signals"""
        logger.info(f"Received signal {signum}, initiating shutdown...")
        asyncio.create_task(self.stop())

async def main():
    """Main function"""
    # Create trading system instance
    trading_system = TradingSystem()
    
    # Setup signal handlers
    for sig in [signal.SIGINT, signal.SIGTERM]:
        signal.signal(sig, trading_system.signal_handler)
    
    try:
        # Start the system
        await trading_system.start()
    except KeyboardInterrupt:
        logger.info("Keyboard interrupt received")
    except Exception as e:
        logger.error(f"System error: {e}")
    finally:
        await trading_system.stop()

if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        print("\nShutdown complete")
    except Exception as e:
        print(f"Fatal error: {e}")
        sys.exit(1)