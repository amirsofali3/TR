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
        """Start the trading system with OHLCV-only mode flow (bypass bootstrap)"""
        try:
            logger.info("🚀 Starting Crypto Trading AI System (OHLCV-only mode)...")
            logger.info(f"📋 Configuration: {len(SUPPORTED_PAIRS)} pairs, 1m timeframe, {'DEMO' if DEMO_MODE else 'LIVE'} mode")
            
            # Phase 1: System Requirements & Database Validation
            logger.info("[STARTUP] Phase 1: System validation...")
            await self.validate_system_requirements()
            
            # Phase 2: Component Initialization 
            logger.info("[STARTUP] Phase 2: Component initialization...")
            await self.initialize_components()
            
            # Phase 3: Start Web Application
            logger.info("[STARTUP] Phase 3: Starting web panel...")
            await self.start_web_app()
            
            # Set running flag for web app
            self.running = True
            
            logger.success(f"🌐 Web application started - accessible at http://{WEB_HOST}:{WEB_PORT}")
            
            # Phase 4: Initialize Candle Data Manager (OHLCV-only mode)
            logger.info("[STARTUP] Phase 4: Initializing candle data access...")
            await self.initialize_candle_data_manager()
            
            # Phase 5: Load Historical OHLCV Data & Calculate Indicators
            logger.info("[STARTUP] Phase 5: Loading historical OHLCV data...")
            await self.load_and_prepare_ohlcv_data()
            
            # Phase 6: Model Training (OHLCV-only mode with RFE)
            logger.info("[STARTUP] Phase 6: Starting OHLCV-based model training...")
            await self.run_ohlcv_model_training()
            
            # Phase 7: Start Trading Operations
            logger.info("[STARTUP] Phase 7: Starting trading operations...")
            await self.start_trading_operations()
            
            logger.success("🎉 OHLCV-only mode startup completed successfully!")
            logger.info("[STARTUP] System fully operational - ready for 1m timeframe trading")
            
            # Start main trading loop
            await self.run_trading_loop()
            
        except Exception as e:
            logger.error(f"❌ Failed to start trading system: {e}")
            logger.error("💡 Common solutions:")
            logger.error("   1. Ensure candles table has data for symbols: " + ", ".join(SUPPORTED_PAIRS)) 
            logger.error("   2. Check MySQL configuration (MYSQL_HOST, MYSQL_USER, etc.)")
            logger.error("   3. Verify ohlcv_only_indicators.csv exists in project root")
            raise
    
    # ==================== OHLCV-ONLY MODE PHASES ====================
    
    async def initialize_candle_data_manager(self):
        """Initialize the candle data manager for OHLCV-only mode"""
        try:
            logger.info("[CANDLES] Initializing candle data manager...")
            
            # Import and initialize candle data manager
            from src.data_access.candle_data_manager import candle_data_manager
            await candle_data_manager.initialize()
            
            # Validate data availability for supported symbols
            validation_results = await candle_data_manager.validate_data_availability(
                SUPPORTED_PAIRS, 
                min_samples=MIN_INITIAL_TRAIN_SAMPLES
            )
            
            available_symbols = [symbol for symbol, valid in validation_results.items() if valid]
            missing_symbols = [symbol for symbol, valid in validation_results.items() if not valid]
            
            if missing_symbols:
                logger.warning(f"[CANDLES] Symbols with insufficient data: {missing_symbols}")
            
            if not available_symbols:
                raise ValueError("No symbols have sufficient data for training")
            
            logger.info(f"[CANDLES] Available symbols for training: {available_symbols}")
            logger.success("[CANDLES] Candle data manager initialized successfully")
            
        except Exception as e:
            logger.error(f"[CANDLES] Failed to initialize candle data manager: {e}")
            raise
    
    async def load_and_prepare_ohlcv_data(self):
        """Load historical OHLCV data and prepare for indicator calculation"""
        try:
            from src.data_access.candle_data_manager import candle_data_manager
            
            logger.info("[OHLCV] Loading full historical OHLCV data...")
            
            # Load full historical data for all supported pairs
            full_data = await candle_data_manager.get_full_historical_data(SUPPORTED_PAIRS)
            
            if full_data.empty:
                raise ValueError("No historical OHLCV data available")
            
            logger.info(f"[OHLCV] Loaded {len(full_data)} historical candles")
            
            # Prepare data for indicator calculation
            prepared_data = candle_data_manager.prepare_features_dataframe(full_data)
            
            if prepared_data.empty:
                raise ValueError("Failed to prepare OHLCV data for indicators")
            
            # Store data for later use
            self.historical_ohlcv_data = prepared_data
            
            logger.success(f"[OHLCV] Historical data prepared: {len(prepared_data)} rows")
            
        except Exception as e:
            logger.error(f"[OHLCV] Failed to load historical data: {e}")
            raise
    
    async def run_ohlcv_model_training(self):
        """Run OHLCV-based model training with RFE on recent window"""
        try:
            logger.info("[TRAIN] Starting OHLCV-based model training...")
            
            from src.data_access.candle_data_manager import candle_data_manager
            
            # Load recent data for RFE window
            logger.info(f"[TRAIN] Loading recent {RFE_SELECTION_CANDLES} candles for RFE...")
            recent_data = await candle_data_manager.get_recent_candles(SUPPORTED_PAIRS, RFE_SELECTION_CANDLES)
            
            if recent_data.empty:
                logger.warning("[TRAIN] No recent data available, using full dataset for RFE")
                recent_data = None
            else:
                recent_data = candle_data_manager.prepare_features_dataframe(recent_data)
                logger.info(f"[TRAIN] RFE window: {len(recent_data)} recent candles")
            
            # Calculate indicators on full historical data
            logger.info("[TRAIN] Calculating OHLCV-based indicators...")
            indicator_results = await self.indicator_engine.calculate_all_indicators(
                self.historical_ohlcv_data
            )
            
            if not indicator_results or 'dataframe' not in indicator_results:
                raise ValueError("Indicator calculation failed")
            
            features_df = indicator_results['dataframe']
            logger.info(f"[TRAIN] Calculated indicators: {len(features_df.columns)} features, {len(features_df)} samples")
            
            # Get feature lists for RFE
            must_keep_features = self.indicator_engine.get_must_keep_features()
            rfe_candidates = self.indicator_engine.get_rfe_candidates()
            
            logger.info(f"[TRAIN] Must keep features: {len(must_keep_features)}")
            logger.info(f"[TRAIN] RFE candidate features: {len(rfe_candidates)}")
            
            # Generate labels (simple momentum strategy for demo)
            labels = self.generate_training_labels(features_df)
            
            # Start model training with RFE
            await self.trading_engine.start_post_bootstrap_training(
                features_df=features_df,
                labels=labels,
                recent_data_df=recent_data,
                must_keep_features=must_keep_features,
                rfe_candidates=rfe_candidates
            )
            
            # Check if training was successful
            if self.ml_model.is_trained:
                logger.success(f"[TRAIN] OHLCV-based training completed successfully")
                logger.info(f"[TRAIN] Model version: {self.ml_model.model_version}")
                logger.info(f"[TRAIN] Training accuracy: {self.ml_model.model_performance.get('accuracy', 0.0):.4f}")
                logger.info(f"[TRAIN] Selected features: {len(self.ml_model.selected_features)}")
            else:
                logger.warning("[TRAIN] OHLCV-based training failed - using fallback signals")
            
        except Exception as e:
            logger.error(f"[TRAIN] OHLCV-based training failed: {e}")
            # Don't raise - continue with fallback functionality
            logger.warning("[TRAIN] Continuing with fallback functionality")
    
    def generate_training_labels(self, df: pd.DataFrame) -> pd.Series:
        """Generate simple training labels based on price movement (for demo)"""
        try:
            # Simple momentum-based labeling
            # This is a placeholder - in production you'd want more sophisticated labeling
            if 'close' not in df.columns:
                raise ValueError("Close price not available for labeling")
            
            # Calculate future returns (next period)
            future_returns = df['close'].pct_change().shift(-1)
            
            # Create labels based on return thresholds
            labels = pd.Series(index=df.index, dtype='object')
            labels[future_returns > 0.002] = 'BUY'    # >0.2% return
            labels[future_returns < -0.002] = 'SELL'  # <-0.2% return  
            labels[labels.isna()] = 'HOLD'             # Everything else
            
            # Remove last row (no future data)
            labels = labels[:-1]
            
            logger.info(f"[TRAIN] Generated labels: {labels.value_counts().to_dict()}")
            return labels
            
        except Exception as e:
            logger.error(f"[TRAIN] Failed to generate labels: {e}")
            # Return dummy labels
            dummy_labels = pd.Series(['HOLD'] * len(df), index=df.index)
            return dummy_labels[:-1]
    
    # ==================== LEGACY BOOTSTRAP PHASES (DISABLED IN OHLCV-ONLY MODE) ====================
    
    async def run_bootstrap_collection(self) -> bool:
        """Legacy bootstrap collection - disabled in OHLCV-only mode"""
        logger.info("[BOOTSTRAP] Bootstrap collection disabled in OHLCV-only mode")
        logger.info("[BOOTSTRAP] Using preloaded candles table data instead")
        return True
    
    async def prepare_indicators(self):
        """Prepare OHLCV-only indicators (legacy method - now integrated into training)"""
        try:
            logger.info("[INDICATORS] OHLCV-only indicator preparation...")
            
            # The indicator engine was already initialized with OHLCV-only config
            indicators_summary = self.indicator_engine.get_indicators_summary()
            
            logger.info(f"[INDICATORS] OHLCV-only summary:")
            logger.info(f"[INDICATORS]   - Total defined: {indicators_summary.get('total_defined', 0)}")
            logger.info(f"[INDICATORS]   - Must keep: {indicators_summary.get('must_keep_count', 0)}")
            logger.info(f"[INDICATORS]   - RFE candidates: {indicators_summary.get('rfe_candidate_count', 0)}")
            
            logger.success("[INDICATORS] OHLCV-only indicator preparation completed")
            
        except Exception as e:
            logger.error(f"[INDICATORS] Indicator preparation failed: {e}")
            raise
    
    async def run_initial_training(self):
        """Legacy initial training method - now handled by run_ohlcv_model_training"""
        logger.info("[TRAIN] Legacy training method - training now handled in run_ohlcv_model_training")
        logger.info("[TRAIN] This method is kept for backward compatibility only")
    
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
