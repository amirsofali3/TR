# Crypto Trading AI System Configuration

# API Configuration
BINANCE_API_KEY = "your_binance_api_key_here"
BINANCE_SECRET_KEY = "your_binance_secret_key_here"

# Trading Configuration
DEFAULT_TIMEFRAME = "4h"  # Default timeframe for analysis
UPDATE_INTERVAL = 60  # Update interval in seconds (1 minute)
CONFIDENCE_THRESHOLD = 70.0  # Minimum confidence for trade execution
MAX_POSITIONS = 10  # Maximum number of open positions

# Supported Cryptocurrencies (limited for testing)
SUPPORTED_PAIRS = [
    "BTCUSDT", "ETHUSDT", "DOGEUSDT"
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

# Machine Learning Configuration
ML_LOOKBACK_PERIODS = 500  # Number of recent candles for RFE selection
CATBOOST_ITERATIONS = 1000
CATBOOST_DEPTH = 10
CATBOOST_LEARNING_RATE = 0.1
RFE_N_FEATURES = 50  # Number of features to select with RFE

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