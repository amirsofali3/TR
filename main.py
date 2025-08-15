"""
Main entry point for the Crypto Trading AI System
"""

import asyncio
import signal
import sys
import os
import time
from concurrent.futures import ThreadPoolExecutor

from dotenv import load_dotenv
load_dotenv()

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
    
    def create_app(self, data_collector=None, indicator_engine=None, ml_model=None, trading_engine=None):
        """Create the Flask app (used for testing)"""
        return create_app(
            data_collector=data_collector,
            indicator_engine=indicator_engine, 
            ml_model=ml_model,
            trading_engine=trading_engine
        )
    
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
        """Validate system requirements before startup (Complete Pipeline Restructure)"""
        logger.info("🔧 Validating system requirements...")
        
        # Check MySQL enforcement first
        try:
            from src.database.db_manager import db_manager
            backend_info = db_manager.get_backend_info()
            
            logger.info(f"[DB] Database backend: {backend_info.get('db_engine', 'unknown')}")
            
            if backend_info.get('db_engine') == 'mysql':
                logger.info(f"[DB] MySQL connection: {backend_info.get('mysql_host')}:{backend_info.get('mysql_port')}")
                logger.info(f"[DB] MySQL database: {backend_info.get('mysql_database')}")
            else:
                logger.info(f"[DB] SQLite path: {backend_info.get('sqlite_path')}")
            
            if backend_info.get('force_mysql_only'):
                logger.info("[DB] MySQL-only mode enforced")
            else:
                logger.info("[DB] MySQL fallback mode enabled")
                
        except Exception as e:
            logger.error(f"[DB] Database validation failed: {e}")
            raise
        
        # Check if data directory exists and is writable (for SQLite fallback)
        if backend_info.get('db_engine') == 'sqlite':
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
            if backend_info.get('db_engine') == 'mysql':
                # Test MySQL connection
                from src.database.db_manager import db_manager
                with db_manager.get_cursor() as cursor:
                    cursor.execute("SELECT 1")
                    result = cursor.fetchone()
                    if result:
                        logger.info("✅ MySQL database connection test passed")
            else:
                # Test SQLite connection
                import sqlite3
                db_path = DATABASE_URL.replace("sqlite:///", "")
                conn = sqlite3.connect(db_path)
                conn.close()
                logger.info("✅ SQLite database connection test passed")
                
        except Exception as e:
            logger.error(f"❌ Database connection test failed: {e}")
            raise
        
        logger.success("✅ System requirements validation passed")

    async def start(self):
        """Start the trading system with Complete Pipeline Restructure flow"""
        try:
            logger.info("🚀 Starting Crypto Trading AI System (Complete Pipeline Restructure)...")
            logger.info(f"📋 Configuration: {len(SUPPORTED_PAIRS)} pairs, {INITIAL_COLLECTION_DURATION_SEC}s bootstrap, {'DEMO' if DEMO_MODE else 'LIVE'} mode")
            
            # Phase 1: System Requirements & Database Validation
            logger.info("[BOOTSTRAP] Phase 1: System validation...")
            await self.validate_system_requirements()
            
            # Phase 2: Component Initialization (Web First!)
            logger.info("[BOOTSTRAP] Phase 2: Component initialization...")
            await self.initialize_components()
            
            # Phase 3: Start Web Application FIRST
            logger.info("[BOOTSTRAP] Phase 3: Starting web panel...")
            await self.start_web_app()
            
            # Set running flag for web app
            self.running = True
            
            logger.success("🌐 Web application started - accessible at http://{WEB_HOST}:{WEB_PORT}")
            logger.info("[BOOTSTRAP] System ready for bootstrap data collection...")
            
            # Phase 4: Bootstrap Data Collection
            logger.info("[BOOTSTRAP] Phase 4: Starting bootstrap data collection...")
            bootstrap_success = await self.run_bootstrap_collection()
            
            if not bootstrap_success:
                logger.error("[BOOTSTRAP] Bootstrap collection failed - starting with limited functionality")
                # Continue anyway but with warnings
            
            # Phase 5: Indicator Preparation (After Collection)
            logger.info("[BOOTSTRAP] Phase 5: Preparing indicators...")
            await self.prepare_indicators()
            
            # Phase 6: Model Training (After Indicators)
            logger.info("[BOOTSTRAP] Phase 6: Starting model training...")
            await self.run_initial_training()
            
            # Phase 7: Start Trading & Online Learning
            logger.info("[BOOTSTRAP] Phase 7: Starting trading operations...")
            await self.start_trading_operations()
            
            logger.success("🎉 Complete Pipeline Restructure startup completed successfully!")
            logger.info("[BOOTSTRAP] System fully operational - all phases complete")
            
            # Start main trading loop
            await self.run_trading_loop()
            
        except Exception as e:
            logger.error(f"❌ Failed to start trading system: {e}")
            logger.error("💡 Common solutions:")
            logger.error("   1. Install missing packages: pip install -r requirements.txt") 
            logger.error("   2. Check MySQL configuration (MYSQL_HOST, MYSQL_USER, etc.)")
            logger.error("   3. Ensure FORCE_MYSQL_ONLY=False if MySQL not available")
            raise
    
    # ==================== COMPLETE PIPELINE RESTRUCTURE PHASES ====================
    
    async def run_bootstrap_collection(self) -> bool:
        """Run the bootstrap data collection phase"""
        try:
            logger.info(f"[COLLECT] Starting {INITIAL_COLLECTION_DURATION_SEC}s bootstrap collection...")
            
            # Start bootstrap collection
            collection_success = await self.data_collector.start_bootstrap_collection(
                duration=INITIAL_COLLECTION_DURATION_SEC
            )
            
            if collection_success:
                logger.success(f"[COLLECT] Bootstrap collection completed successfully")
                return True
            else:
                logger.warning(f"[COLLECT] Bootstrap collection failed or incomplete")
                return False
                
        except Exception as e:
            logger.error(f"[COLLECT] Bootstrap collection phase failed: {e}")
            return False
    
    async def prepare_indicators(self):
        """Prepare indicators after bootstrap collection"""
        try:
            logger.info("[INDICATORS] Loading and parsing encyclopedia...")
            
            # The indicator engine was already initialized, but let's get a summary
            indicators_summary = self.indicator_engine.get_indicators_summary()
            
            logger.info(f"[INDICATORS] Encyclopedia summary:")
            logger.info(f"[INDICATORS]   - Total defined: {indicators_summary.get('total_defined', 0)}")
            logger.info(f"[INDICATORS]   - Must keep: {indicators_summary.get('must_keep_count', 0)}")
            logger.info(f"[INDICATORS]   - RFE candidates: {indicators_summary.get('rfe_candidate_count', 0)}")
            logger.info(f"[INDICATORS]   - Computed: {indicators_summary.get('computed_count', 0)}")
            
            skipped = indicators_summary.get('skipped', [])
            if skipped:
                logger.info(f"[INDICATORS] Skipped {len(skipped)} indicators:")
                for skip in skipped[:5]:  # Show first 5
                    logger.info(f"[INDICATORS]   - {skip['name']}: {skip['reason']}")
                if len(skipped) > 5:
                    logger.info(f"[INDICATORS]   - ... and {len(skipped) - 5} more")
            
            logger.success("[INDICATORS] Indicator preparation completed")
            
        except Exception as e:
            logger.error(f"[INDICATORS] Indicator preparation failed: {e}")
            raise
    
    async def run_initial_training(self):
        """Run the initial model training after bootstrap collection"""
        try:
            logger.info("[TRAIN] Starting initial model training phase...")
            
            # Start post-bootstrap training
            await self.trading_engine.start_post_bootstrap_training()
            
            # Check if training was successful
            if self.ml_model.is_trained:
                logger.success(f"[TRAIN] Initial training completed successfully")
                logger.info(f"[TRAIN] Model version: {self.ml_model.model_version}")
                logger.info(f"[TRAIN] Training accuracy: {self.ml_model.model_performance.get('accuracy', 0.0):.4f}")
                logger.info(f"[TRAIN] Selected features: {self.ml_model.selected_feature_count}")
            else:
                logger.warning("[TRAIN] Initial training failed - using fallback signals")
                logger.info("[TRAIN] System will continue with technical indicator fallbacks")
            
        except Exception as e:
            logger.error(f"[TRAIN] Initial training phase failed: {e}")
            # Don't raise - continue with fallback functionality
            logger.warning("[TRAIN] Continuing with fallback functionality")
    
    async def start_trading_operations(self):
        """Start trading operations and online learning"""
        try:
            logger.info("[ONLINE] Starting trading operations...")
            
            # Start online learning if model is trained
            if self.ml_model.is_trained:
                online_learning_status = self.trading_engine.get_online_learning_status()
                if online_learning_status.get('active'):
                    logger.info("[ONLINE] Online learning already active")
                else:
                    logger.info("[ONLINE] Starting online learning system...")
                    # Online learning should have been started by start_post_bootstrap_training
            else:
                logger.info("[ONLINE] Model not trained - skipping online learning")
            
            logger.success("[ONLINE] Trading operations started")
            
        except Exception as e:
            logger.error(f"[ONLINE] Failed to start trading operations: {e}")
            # Don't raise - basic functionality should still work
            logger.warning("[ONLINE] Continuing with basic trading functionality")
    
    # ==================== END COMPLETE PIPELINE RESTRUCTURE PHASES ====================
    
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
