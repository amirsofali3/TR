# Crypto Trading AI System Configuration

# API Configuration
BINANCE_API_KEY = "your_binance_api_key_here"
BINANCE_SECRET_KEY = "your_binance_secret_key_here"

# Trading Configuration
DEFAULT_TIMEFRAME = "1m"  # Default timeframe for analysis (OHLCV-only mode)
UPDATE_INTERVAL = 60  # Update interval in seconds (1 minute)
CONFIDENCE_THRESHOLD = 70.0  # Minimum confidence for trade execution
MAX_POSITIONS = 10  # Maximum number of open positions

# Supported Cryptocurrencies (OHLCV-only mode)
SUPPORTED_PAIRS = [
    "BTCUSDT", "ETHUSDT", "SOLUSDT", "DOGEUSDT"
]

# Database Configuration
DATABASE_URL = "sqlite:///data/trading_system.db"

# MySQL Configuration (MySQL migration)
# Set MYSQL_ENABLED=true in environment to use MySQL instead of SQLite
MYSQL_ENABLED = False
MYSQL_HOST = "localhost"
MYSQL_PORT = 3306
MYSQL_USER = "root" 
MYSQL_PASSWORD = ""
MYSQL_DB = "trading_system"
MYSQL_CHARSET = "utf8mb4"
MYSQL_TIMEZONE = "+00:00"

# Machine Learning Configuration (OHLCV-only mode)
ML_LOOKBACK_PERIODS = 500  # Number of recent candles for RFE selection
CATBOOST_ITERATIONS = 1000
CATBOOST_DEPTH = 10
CATBOOST_LEARNING_RATE = 0.1
RFE_N_FEATURES = 25  # Number of features to select with RFE (reduced for OHLCV-only)

# ML Training Pipeline Configuration (MySQL migration)
MIN_INITIAL_TRAIN_SAMPLES = 400  # Minimum samples required for initial training
MIN_RFE_SAMPLES = 150  # Minimum samples for RFE, otherwise use baseline features
MAX_BASELINE_FEATURES = 120  # Max features when RFE is skipped
TRAIN_RETRY_COOLDOWN_MIN = 10  # Minutes to wait before retrying failed training

# Auto Schema and Database Configuration (Follow-up fixes)
AUTO_CREATE_SCHEMA = True  # Automatically create database tables if missing
AUTO_FALLBACK_DB = True  # Fallback to SQLite if MySQL credentials missing
MYSQL_MARKET_DATA_TABLE = "market_data"  # Configurable market data table name
TRAIN_RETRY_COOLDOWN_SEC = 600  # Training retry cooldown in seconds (10 min)

# Fallback Strategy Configuration (MySQL migration) 
ALLOW_FALLBACK_EXECUTION = False  # Allow fallback signals to execute trades

# Risk Management Configuration
DEFAULT_TP_LEVELS = [0.02, 0.04, 0.06, 0.08, 0.10]  # TP levels as percentages
DEFAULT_SL_PERCENTAGE = 0.02  # Initial SL as percentage
TP_SL_TRAILING_STEP = 0.01  # Step size for trailing SL

# Web Application Configuration
WEB_HOST = "0.0.0.0"
WEB_PORT = 5000
SECRET_KEY = "your_secret_key_here_change_this"

# Logging Configuration
LOG_LEVEL = "INFO"
LOG_FILE = "logs/trading_system.log"
MAX_LOG_SIZE = 10  # MB
BACKUP_COUNT = 5

# Demo Mode Configuration
DEMO_MODE = True  # Start in demo mode by default
DEMO_BALANCE = 10000.0  # Starting balance in USDT for demo mode

# Feature Weights Configuration
ACTIVE_FEATURE_WEIGHT = 1.0
INACTIVE_FEATURE_WEIGHT = 0.01

# Data Collection Configuration
MAX_HISTORICAL_DAYS = 365  # Maximum days of historical data to fetch
DATA_UPDATE_INTERVAL = 3600  # Update historical data every hour

# Bootstrap Data Collection Configuration (Complete Pipeline Restructure)
INITIAL_COLLECTION_DURATION_SEC = 0  # No bootstrap collection in OHLCV-only mode
INITIAL_COLLECTION_TIMEFRAME = "1s"  # Raw tick collection timeframe (legacy)
ONLINE_RETRAIN_INTERVAL_SEC = 900  # 15 minutes between automatic retrains
MIN_NEW_SAMPLES_FOR_RETRAIN = 300  # Minimum new samples to trigger retrain
ACCURACY_SLIDING_WINDOW = 1000  # Number of recent predictions for live accuracy
ACCURACY_UPDATE_INTERVAL_SEC = 60  # Update live accuracy every 60 seconds

# OHLCV-only Mode Configuration
INDICATOR_CSV_PATH = "ohlcv_only_indicators.csv"  # Path to OHLCV-only indicators CSV
ENABLE_TICK_BOOTSTRAP = False  # Disable tick-based bootstrap entirely
RFE_SELECTION_CANDLES = 1000  # Recent candles window for RFE selection
BASE_MUST_KEEP_FEATURES = ["open", "high", "low", "close", "volume"]  # Base OHLCV features always kept

# Data Collection & Analysis Separation (User Feedback Adjustments)
RAW_COLLECTION_INTERVAL_SEC = 1  # Fixed raw data collection interval
BASE_ANALYSIS_INTERVAL_SEC = 5  # Default analysis interval (user-adjustable)

# Training Sample Requirements (User Feedback Adjustments)
MIN_VALID_SAMPLES = 150  # Minimum valid samples after sanitization

# Phase 3 Stabilization Settings
ENABLE_RFE = True  # Enable/disable Recursive Feature Elimination
MIN_ANALYSIS_CANDLES = 150  # Minimum candles for live analysis (fetch fallback)
RFE_FALLBACK_TOP_N = 50  # Top N features when RFE disabled or fails

# Phase 4 Feature Selection & Signal Generation Settings
FEATURE_API_ENABLED = True  # Enable /api/features endpoint
SIGNAL_MIN_FEATURE_COVERAGE = 0.9  # Minimum feature coverage for signal generation
HISTORICAL_FETCH_LIMIT = 500  # Max candles to fetch for fallback analysis
MAX_INACTIVE_FEATURE_LIST = 50  # UI truncation limit for inactive features
RFE_VERBOSE = False  # CatBoost verbosity for RFE (replaces logging_level)

# Retrain Configuration (User Feedback Adjustments)
RETRAIN_USE_FULL_HISTORY = True  # Use full history for retrain by default
RETRAIN_HISTORY_MINUTES = 180  # History window when RETRAIN_USE_FULL_HISTORY=False

# MySQL Enforcement Configuration (Complete Pipeline Restructure) 
FORCE_MYSQL_ONLY = True  # Prevent SQLite fallback, require MySQL credentials
MYSQL_MARKET_DATA_TABLE = "market_data"  # Configurable market data table name

# Additional MySQL Table Names
MYSQL_MARKET_TICKS_TABLE = "market_ticks"  # Raw tick data
MYSQL_OHLC_1S_TABLE = "ohlc_1s"  # 1-second OHLC data
MYSQL_OHLC_1M_TABLE = "ohlc_1m"  # 1-minute aggregated OHLC data
MYSQL_INDICATORS_CACHE_TABLE = "indicators_cache"  # Indicator cache
MYSQL_MODEL_TRAINING_RUNS_TABLE = "model_training_runs"  # Training audit trail
MYSQL_MODEL_METRICS_TABLE = "model_metrics"  # Model performance metrics