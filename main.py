"""
Main entry point for the Crypto Trading AI System
"""

import asyncio
import signal
import sys
import os
import time
from concurrent.futures import ThreadPoolExecutor

# Handle missing loguru gracefully
try:
    from loguru import logger
except ImportError:
    print("[ERROR] loguru is not installed. Please install it with: pip install loguru")
    print("[INFO] Using basic logging as fallback...")
    import logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s | %(levelname)-8s | %(message)s'
    )
    logger = logging.getLogger(__name__)
    logger.success = logger.info  # Add success method for compatibility

# Add src to path
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

# Import configuration first
try:
    from config.settings import *
    logger.info("Configuration loaded successfully")
except Exception as e:
    print(f"[ERROR] Failed to load configuration: {e}")
    sys.exit(1)

# Import components with error handling
missing_modules = []

try:
    from src.data_collector.binance_collector import BinanceDataCollector
except ImportError as e:
    missing_modules.append(f"data_collector: {e}")
    BinanceDataCollector = None

try:
    from src.indicators.indicator_engine import IndicatorEngine
except ImportError as e:
    missing_modules.append(f"indicator_engine: {e}")
    IndicatorEngine = None

try:
    from src.ml_model.catboost_model import CatBoostTradingModel
except ImportError as e:
    missing_modules.append(f"ml_model: {e}")
    CatBoostTradingModel = None

try:
    from src.trading_engine.trading_engine import TradingEngine
except ImportError as e:
    missing_modules.append(f"trading_engine: {e}")
    TradingEngine = None

try:
    from src.web_app.app import create_app
except ImportError as e:
    missing_modules.append(f"web_app: {e}")
    create_app = None

# Report missing modules
if missing_modules:
    logger.error("Some modules could not be imported:")
    for module in missing_modules:
        logger.error(f"  - {module}")
    logger.error("Please install missing dependencies or check the module paths")
    print("\n[SOLUTION] Try running: pip install pandas numpy flask aiohttp loguru python-binance scikit-learn catboost")
    print("Or run: python install_packages.py")
    sys.exit(1)

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
            retention=BACKUP_COUNT
        )
    
    async def initialize_components(self):
        """Initialize all system components"""
        try:
            logger.info("Initializing system components...")
            
            # Check API keys configuration
            if BINANCE_API_KEY == "your_binance_api_key_here" or not BINANCE_API_KEY:
                logger.warning("Binance API keys not configured - running in demo mode only")
                logger.info("To enable real trading, set BINANCE_API_KEY and BINANCE_SECRET_KEY in config/settings.py")
            
            # Initialize data collector
            logger.info("Initializing data collector...")
            self.data_collector = BinanceDataCollector()
            await self.data_collector.initialize()
            logger.success("✅ Data collector initialized")
            
            # Initialize indicator engine  
            logger.info("Initializing indicator engine...")
            self.indicator_engine = IndicatorEngine()
            await self.indicator_engine.initialize()
            logger.success("✅ Indicator engine initialized")
            
            # Initialize ML model
            logger.info("Initializing ML model...")
            self.ml_model = CatBoostTradingModel()
            await self.ml_model.initialize()
            logger.success("✅ ML model initialized")
            
            # Initialize trading engine
            logger.info("Initializing trading engine...")
            self.trading_engine = TradingEngine(
                data_collector=self.data_collector,
                indicator_engine=self.indicator_engine,
                ml_model=self.ml_model
            )
            await self.trading_engine.initialize()
            logger.success("✅ Trading engine initialized")
            
            logger.success("🎉 All components initialized successfully")
            logger.info(f"📊 Monitoring {len(SUPPORTED_PAIRS)} trading pairs: {', '.join(SUPPORTED_PAIRS)}")
            logger.info(f"⏱️  Analysis interval: {UPDATE_INTERVAL} seconds")
            
        except Exception as e:
            logger.error(f"❌ Failed to initialize components: {e}")
            logger.error("This is likely due to missing dependencies or configuration issues")
            logger.error("Check the logs above for specific error details")
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
        logger.info("🔄 Starting main trading loop...")
        logger.info(f"📈 System will analyze markets every {UPDATE_INTERVAL} seconds")
        
        analysis_count = 0
        while self.running:
            try:
                analysis_count += 1
                logger.info(f"🔍 Starting market analysis #{analysis_count}")
                start_time = time.time()
                
                # Update data and analyze markets
                await self.trading_engine.analyze_markets()
                
                end_time = time.time()
                duration = end_time - start_time
                logger.info(f"✅ Market analysis #{analysis_count} completed in {duration:.2f} seconds")
                
                # Wait for next update
                logger.debug(f"😴 Waiting {UPDATE_INTERVAL} seconds until next analysis...")
                await asyncio.sleep(UPDATE_INTERVAL)
                
            except KeyboardInterrupt:
                logger.info("⏹️  Received keyboard interrupt, stopping...")
                break
            except Exception as e:
                logger.error(f"❌ Error in trading loop (analysis #{analysis_count}): {e}")
                logger.error("Will retry in 5 seconds...")
                import traceback
                logger.debug(traceback.format_exc())
                await asyncio.sleep(5)  # Wait before retrying
    
    async def validate_system_requirements(self):
        """Validate system requirements before startup"""
        logger.info("🔧 Validating system requirements...")
        
        # Check if data directory exists and is writable
        data_dir = os.path.dirname(DATABASE_URL.replace("sqlite:///", ""))
        if data_dir and not os.path.exists(data_dir):
            try:
                os.makedirs(data_dir, exist_ok=True)
                logger.info(f"✅ Created data directory: {data_dir}")
            except Exception as e:
                logger.error(f"❌ Cannot create data directory {data_dir}: {e}")
                raise
        
        # Check if logs directory exists
        log_dir = os.path.dirname(LOG_FILE)
        if log_dir and not os.path.exists(log_dir):
            try:
                os.makedirs(log_dir, exist_ok=True)
                logger.info(f"✅ Created logs directory: {log_dir}")
            except Exception as e:
                logger.error(f"❌ Cannot create logs directory {log_dir}: {e}")
                raise
        
        # Test database connection
        try:
            import sqlite3
            db_path = DATABASE_URL.replace("sqlite:///", "")
            conn = sqlite3.connect(db_path)
            conn.close()
            logger.info("✅ Database connection test passed")
        except Exception as e:
            logger.error(f"❌ Database connection test failed: {e}")
            raise
        
        logger.success("✅ System requirements validation passed")

    async def start(self):
        """Start the trading system"""
        try:
            logger.info("🚀 Starting Crypto Trading AI System...")
            logger.info(f"📋 Configuration: {len(SUPPORTED_PAIRS)} pairs, {UPDATE_INTERVAL}s interval, {'DEMO' if DEMO_MODE else 'LIVE'} mode")
            
            # Validate system requirements
            await self.validate_system_requirements()
            
            # Initialize all components
            await self.initialize_components()
            
            # Start web application
            logger.info("🌐 Starting web application...")
            await self.start_web_app()
            
            # Set running flag
            self.running = True
            
            logger.success("🎉 System startup completed successfully!")
            logger.info("📊 Beginning market analysis...")
            
            # Start trading loop
            await self.run_trading_loop()
            
        except Exception as e:
            logger.error(f"❌ Failed to start trading system: {e}")
            logger.error("💡 Common solutions:")
            logger.error("   1. Install missing packages: pip install -r requirements.txt") 
            logger.error("   2. Check API key configuration in config/settings.py")
            logger.error("   3. Ensure database directory is writable")
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