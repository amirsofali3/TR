"""
Database abstraction layer supporting both SQLite and MySQL
MySQL migration support for the crypto trading system
"""

import os
import sqlite3
import asyncio
from typing import Optional, List, Dict, Any, Union
from contextlib import contextmanager
from loguru import logger

from dotenv import load_dotenv
load_dotenv()  # به صورت پیش‌فرض دنبال فایل .env در همان دایرکتوری فعلی می‌گردد

# MySQL support (optional)
try:
    import pymysql
    MYSQL_AVAILABLE = True
except ImportError:
    MYSQL_AVAILABLE = False
    logger.warning("PyMySQL not available - MySQL support disabled")

from config.settings import DATABASE_URL

class DatabaseManager:
    """Unified database manager supporting SQLite and MySQL"""
    
    def __init__(self):
        self.backend = None
        self.mysql_config = {}
        self.sqlite_path = None
        self._detect_backend()
    
    def _detect_backend(self):
        """Auto-detect database backend based on configuration with MySQL enforcement"""
        force_mysql_only = os.getenv('FORCE_MYSQL_ONLY', '').lower()
        if force_mysql_only == '':
            try:
                from config.settings import FORCE_MYSQL_ONLY
                force_mysql_only = FORCE_MYSQL_ONLY
            except ImportError:
                force_mysql_only = True
        else:
            force_mysql_only = force_mysql_only == 'true'
        
        allow_empty_pw = os.getenv('ALLOW_EMPTY_MYSQL_PASSWORD', 'false').lower() == 'true'
        
        mysql_enabled = os.getenv('MYSQL_ENABLED', 'false').lower() == 'true'
        mysql_db_set = os.getenv('MYSQL_DB') is not None
        
        required_vars = ['MYSQL_HOST', 'MYSQL_USER', 'MYSQL_PASSWORD', 'MYSQL_DB']
        missing_vars = []
        for var in required_vars:
            val = os.getenv(var)
            if val is None:
                missing_vars.append(var)
            else:
                stripped = val.strip()
                if var == 'MYSQL_PASSWORD':
                    if stripped == '' and not allow_empty_pw:
                        missing_vars.append(var)
                else:
                    if stripped == '':
                        missing_vars.append(var)
        
        if force_mysql_only:
            if not MYSQL_AVAILABLE:
                raise ImportError("[DB] FORCE_MYSQL_ONLY=True but PyMySQL not installed. Install with: pip install pymysql")
            
            if missing_vars:
                logger.error(f"[DB] FORCE_MYSQL_ONLY=True but missing required MySQL environment variables:")
                for var in missing_vars:
                    logger.error(f"[DB]   - {var} (not set)")
                logger.error(f"[DB] Required variables: {', '.join(required_vars)}")
                if allow_empty_pw:
                    logger.error("[DB] Note: ALLOW_EMPTY_MYSQL_PASSWORD=True only bypasses empty password check if MYSQL_PASSWORD is defined (even if blank).")
                raise ValueError(f"MySQL enforcement enabled but missing required variables: {', '.join(missing_vars)}")
            
            pw = os.getenv('MYSQL_PASSWORD', '')
            if pw.strip() == '' and allow_empty_pw:
                logger.warning("[DB] MySQL running with EMPTY PASSWORD (ALLOW_EMPTY_MYSQL_PASSWORD=True) — SECURITY RISK (dev only)")
            
            self.backend = 'mysql'
            self.mysql_config = {
                'host': os.getenv('MYSQL_HOST', 'localhost'),
                'port': int(os.getenv('MYSQL_PORT', 3306)),
                'user': os.getenv('MYSQL_USER', 'root'),
                'password': pw,
                'database': os.getenv('MYSQL_DB', 'trading_system'),
                'charset': os.getenv('MYSQL_CHARSET', 'utf8mb4'),
                'autocommit': True
            }
            logger.info(f"[DB] MySQL-only mode: {self.mysql_config['user']}@{self.mysql_config['host']}:{self.mysql_config['port']}/{self.mysql_config['database']}")
            return
        
        # Fallback logic (unchanged except password handling already covered)
        if mysql_enabled and MYSQL_AVAILABLE:
            if missing_vars:
                auto_fallback = os.getenv('AUTO_FALLBACK_DB', '').lower() == 'true'
                if not auto_fallback:
                    try:
                        from config.settings import AUTO_FALLBACK_DB
                        auto_fallback = AUTO_FALLBACK_DB
                    except ImportError:
                        auto_fallback = True
                logger.warning(f"[DB] MySQL enabled but missing environment variables: {', '.join(missing_vars)}")
                if auto_fallback:
                    logger.info("[DB] AUTO_FALLBACK_DB=True, using SQLite fallback")
                    self._setup_sqlite_backend()
                    return
                else:
                    raise ValueError(f"MySQL enabled but missing vars: {', '.join(missing_vars)}. Set AUTO_FALLBACK_DB=True to fallback.")
            
            pw = os.getenv('MYSQL_PASSWORD', '')
            if pw.strip() == '' and allow_empty_pw:
                logger.warning("[DB] MySQL (fallback mode) with EMPTY PASSWORD (ALLOW_EMPTY_MYSQL_PASSWORD=True) — SECURITY RISK")
            
            self.backend = 'mysql'
            self.mysql_config = {
                'host': os.getenv('MYSQL_HOST', 'localhost'),
                'port': int(os.getenv('MYSQL_PORT', 3306)),
                'user': os.getenv('MYSQL_USER', 'root'),
                'password': pw,
                'database': os.getenv('MYSQL_DB', 'trading_system'),
                'charset': os.getenv('MYSQL_CHARSET', 'utf8mb4'),
                'autocommit': True
            }
            logger.info(f"[DB] Using MySQL backend: {self.mysql_config['user']}@{self.mysql_config['host']}:{self.mysql_config['port']}/{self.mysql_config['database']}")
        elif mysql_db_set and not mysql_enabled:
            logger.warning(f"[DB] MYSQL_DB set but MYSQL_ENABLED!='true' — using SQLite.")
            logger.info("[DB] To use MySQL set MYSQL_ENABLED=true")
            self._setup_sqlite_backend()
        else:
            self._setup_sqlite_backend()
    
    def _setup_sqlite_backend(self):
        """Setup SQLite backend configuration"""
        self.backend = 'sqlite'
        self.sqlite_path = DATABASE_URL.replace("sqlite:///", "")
        # Create directory if needed
        os.makedirs(os.path.dirname(self.sqlite_path), exist_ok=True)
        logger.info(f"Using SQLite backend: {self.sqlite_path}")
    
    def get_connection(self):
        """Get database connection based on configured backend"""
        if self.backend == 'mysql':
            if not MYSQL_AVAILABLE:
                raise ImportError("MySQL backend selected but PyMySQL not available")
            return pymysql.connect(**self.mysql_config)
        else:
            return sqlite3.connect(self.sqlite_path)
    
    @contextmanager
    def get_cursor(self):
        """Context manager for database operations"""
        conn = None
        cursor = None
        try:
            conn = self.get_connection()
            cursor = conn.cursor()
            yield cursor
            conn.commit()
        except Exception as e:
            if conn:
                conn.rollback()
            raise e
        finally:
            if cursor:
                cursor.close()
            if conn:
                conn.close()
    
    def execute(self, query: str, params: Optional[tuple] = None) -> int:
        """Execute a query and return affected rows count"""
        with self.get_cursor() as cursor:
            if params:
                cursor.execute(query, params)
            else:
                cursor.execute(query)
            
            # Return affected rows
            if hasattr(cursor, 'rowcount'):
                return cursor.rowcount
            return 0
    
    def executemany(self, query: str, params_list: List[tuple]) -> int:
        """Execute query with multiple parameter sets"""
        with self.get_cursor() as cursor:
            cursor.executemany(query, params_list)
            
            if hasattr(cursor, 'rowcount'):
                return cursor.rowcount
            return 0
    
    def fetchall(self, query: str, params: Optional[tuple] = None) -> List[tuple]:
        """Execute query and fetch all results"""
        with self.get_cursor() as cursor:
            if params:
                cursor.execute(query, params)
            else:
                cursor.execute(query)
            return cursor.fetchall()
    
    def fetchone(self, query: str, params: Optional[tuple] = None) -> Optional[tuple]:
        """Execute query and fetch one result"""
        with self.get_cursor() as cursor:
            if params:
                cursor.execute(query, params)
            else:
                cursor.execute(query)
            return cursor.fetchone()
    
    def get_pandas_connection(self):
        """Get connection suitable for pandas operations"""
        return self.get_connection()
    
    def close(self):
        """Close database connections (placeholder for connection pooling)"""
        # In future versions, this could close connection pools
        pass
    
    def get_backend_info(self) -> Dict[str, Any]:
        """Get information about current database backend"""
        # Determine FORCE_MYSQL_ONLY setting
        force_mysql_only = os.getenv('FORCE_MYSQL_ONLY', '').lower()
        if force_mysql_only == '':
            try:
                from config.settings import FORCE_MYSQL_ONLY
                force_mysql_only = FORCE_MYSQL_ONLY
            except ImportError:
                force_mysql_only = True
        else:
            force_mysql_only = force_mysql_only == 'true'
        
        if self.backend == 'mysql':
            return {
                'db_engine': 'mysql',
                'mysql_host': self.mysql_config['host'],
                'mysql_port': self.mysql_config['port'],
                'mysql_database': self.mysql_config['database'],
                'mysql_charset': self.mysql_config['charset'],
                'table_market_data': os.getenv('MYSQL_MARKET_DATA_TABLE', 'market_data'),
                'force_mysql_only': force_mysql_only
            }
        else:
            return {
                'db_engine': 'sqlite',
                'sqlite_path': self.sqlite_path,
                'sqlite_exists': os.path.exists(self.sqlite_path) if self.sqlite_path else False,
                'table_market_data': 'market_data',
                'force_mysql_only': force_mysql_only
            }
    
    def ensure_schema(self) -> bool:
        """Ensure database schema exists, create missing tables"""
        try:
            from config.settings import AUTO_CREATE_SCHEMA, MYSQL_MARKET_DATA_TABLE
            
            if not AUTO_CREATE_SCHEMA:
                logger.info("[DB] Auto schema creation disabled")
                return True
            
            logger.info("[DB] Checking database schema...")
            
            # Define table schemas
            schemas = self._get_table_schemas()
            created_tables = []
            
            with self.get_cursor() as cursor:
                for table_name, schema in schemas.items():
                    try:
                        # Check if table exists
                        if self.backend == 'mysql':
                            cursor.execute(f"SHOW TABLES LIKE '{table_name}'")
                            exists = cursor.fetchone() is not None
                        else:  # sqlite
                            cursor.execute(f"SELECT name FROM sqlite_master WHERE type='table' AND name='{table_name}'")
                            exists = cursor.fetchone() is not None
                        
                        if not exists:
                            logger.info(f"[DB] Creating missing table: {table_name}")
                            cursor.execute(schema)
                            created_tables.append(table_name)
                    except Exception as e:
                        logger.error(f"[DB] Failed to create table {table_name}: {e}")
                        continue
            
            if created_tables:
                logger.info(f"[DB] Created {len(created_tables)} missing tables: {', '.join(created_tables)}")
            else:
                logger.info("[DB] All required tables exist")
            
            return True
            
        except Exception as e:
            logger.error(f"[DB] Schema creation failed: {e}")
            return False
    
    def _get_table_schemas(self) -> Dict[str, str]:
        """Get table creation schemas for both MySQL and SQLite with complete pipeline tables"""
        # Use configurable table names
        market_table = os.getenv('MYSQL_MARKET_DATA_TABLE') or 'market_data'
        ticks_table = os.getenv('MYSQL_MARKET_TICKS_TABLE') or 'market_ticks'
        ohlc_1s_table = os.getenv('MYSQL_OHLC_1S_TABLE') or 'ohlc_1s'
        ohlc_1m_table = os.getenv('MYSQL_OHLC_1M_TABLE') or 'ohlc_1m'
        indicators_cache_table = os.getenv('MYSQL_INDICATORS_CACHE_TABLE') or 'indicators_cache'
        training_runs_table = os.getenv('MYSQL_MODEL_TRAINING_RUNS_TABLE') or 'model_training_runs'
        model_metrics_table = os.getenv('MYSQL_MODEL_METRICS_TABLE') or 'model_metrics'
        
        try:
            from config.settings import (
                MYSQL_MARKET_DATA_TABLE, MYSQL_MARKET_TICKS_TABLE, MYSQL_OHLC_1S_TABLE,
                MYSQL_OHLC_1M_TABLE, MYSQL_INDICATORS_CACHE_TABLE, 
                MYSQL_MODEL_TRAINING_RUNS_TABLE, MYSQL_MODEL_METRICS_TABLE
            )
            market_table = market_table or MYSQL_MARKET_DATA_TABLE
            ticks_table = ticks_table or MYSQL_MARKET_TICKS_TABLE
            ohlc_1s_table = ohlc_1s_table or MYSQL_OHLC_1S_TABLE
            ohlc_1m_table = ohlc_1m_table or MYSQL_OHLC_1M_TABLE
            indicators_cache_table = indicators_cache_table or MYSQL_INDICATORS_CACHE_TABLE
            training_runs_table = training_runs_table or MYSQL_MODEL_TRAINING_RUNS_TABLE
            model_metrics_table = model_metrics_table or MYSQL_MODEL_METRICS_TABLE
        except ImportError:
            pass  # Use defaults
        
        if self.backend == 'mysql':
            return {
                # Raw tick data (1-second or sub-second)
                ticks_table: f"""
                    CREATE TABLE IF NOT EXISTS {ticks_table} (
                        id BIGINT AUTO_INCREMENT PRIMARY KEY,
                        symbol VARCHAR(20) NOT NULL,
                        timestamp BIGINT NOT NULL,
                        price DECIMAL(20,8) NOT NULL,
                        volume DECIMAL(20,8) NOT NULL,
                        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                        INDEX idx_symbol_timestamp (symbol, timestamp)
                    ) ENGINE=InnoDB
                """,
                
                # 1-second OHLC data (aggregated from ticks)
                ohlc_1s_table: f"""
                    CREATE TABLE IF NOT EXISTS {ohlc_1s_table} (
                        id BIGINT AUTO_INCREMENT PRIMARY KEY,
                        symbol VARCHAR(20) NOT NULL,
                        timestamp BIGINT NOT NULL,
                        open DECIMAL(20,8) NOT NULL,
                        high DECIMAL(20,8) NOT NULL,
                        low DECIMAL(20,8) NOT NULL,
                        close DECIMAL(20,8) NOT NULL,
                        volume DECIMAL(20,8) NOT NULL,
                        tick_count INT DEFAULT 1,
                        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                        UNIQUE KEY unique_candle_1s (symbol, timestamp)
                    ) ENGINE=InnoDB
                """,
                
                # 1-minute OHLC data (aggregated from 1s data)
                ohlc_1m_table: f"""
                    CREATE TABLE IF NOT EXISTS {ohlc_1m_table} (
                        id BIGINT AUTO_INCREMENT PRIMARY KEY,
                        symbol VARCHAR(20) NOT NULL,
                        timestamp BIGINT NOT NULL,
                        open DECIMAL(20,8) NOT NULL,
                        high DECIMAL(20,8) NOT NULL,
                        low DECIMAL(20,8) NOT NULL,
                        close DECIMAL(20,8) NOT NULL,
                        volume DECIMAL(20,8) NOT NULL,
                        tick_count INT DEFAULT 60,
                        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                        UNIQUE KEY unique_candle_1m (symbol, timestamp)
                    ) ENGINE=InnoDB
                """,
                
                # Original market data table (for backward compatibility)
                market_table: f"""
                    CREATE TABLE IF NOT EXISTS {market_table} (
                        id INT AUTO_INCREMENT PRIMARY KEY,
                        symbol VARCHAR(20) NOT NULL,
                        timeframe VARCHAR(10) NOT NULL,
                        timestamp BIGINT NOT NULL,
                        open DECIMAL(20,8) NOT NULL,
                        high DECIMAL(20,8) NOT NULL,
                        low DECIMAL(20,8) NOT NULL,
                        close DECIMAL(20,8) NOT NULL,
                        volume DECIMAL(20,8) NOT NULL,
                        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                        UNIQUE KEY unique_candle (symbol, timeframe, timestamp)
                    ) ENGINE=InnoDB
                """,
                
                # Indicators cache for performance
                indicators_cache_table: f"""
                    CREATE TABLE IF NOT EXISTS {indicators_cache_table} (
                        id BIGINT AUTO_INCREMENT PRIMARY KEY,
                        symbol VARCHAR(20) NOT NULL,
                        timestamp BIGINT NOT NULL,
                        indicator_name VARCHAR(100) NOT NULL,
                        indicator_value DECIMAL(20,8),
                        indicator_json TEXT,
                        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                        INDEX idx_symbol_timestamp_indicator (symbol, timestamp, indicator_name)
                    ) ENGINE=InnoDB
                """,
                
                # Model training runs audit trail
                training_runs_table: f"""
                    CREATE TABLE IF NOT EXISTS {training_runs_table} (
                        id BIGINT AUTO_INCREMENT PRIMARY KEY,
                        model_version INT NOT NULL,
                        training_started_at TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP,
                        training_completed_at TIMESTAMP NULL DEFAULT NULL,
                        samples_count INT,
                        features_selected INT,
                        accuracy DECIMAL(5,4),
                        training_error TEXT,
                        status ENUM('started', 'completed', 'failed') DEFAULT 'started',
                        config_json TEXT,
                        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                    ) ENGINE=InnoDB
                """,
                
                # Model metrics key-value store
                model_metrics_table: f"""
                    CREATE TABLE IF NOT EXISTS {model_metrics_table} (
                        metric_key VARCHAR(100) PRIMARY KEY,
                        metric_value TEXT,
                        metric_type ENUM('int', 'float', 'string', 'json') DEFAULT 'string',
                        updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP ON UPDATE CURRENT_TIMESTAMP
                    ) ENGINE=InnoDB
                """,
                
                # Real-time prices table
                'real_time_prices': """
                    CREATE TABLE IF NOT EXISTS real_time_prices (
                        symbol VARCHAR(20) PRIMARY KEY,
                        price DECIMAL(20,8) NOT NULL,
                        bid_price DECIMAL(20,8),
                        ask_price DECIMAL(20,8),
                        bid_size DECIMAL(20,8),
                        ask_size DECIMAL(20,8),
                        timestamp BIGINT NOT NULL,
                        updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP ON UPDATE CURRENT_TIMESTAMP
                    ) ENGINE=InnoDB
                """,
                
                # Positions table
                'positions': """
                    CREATE TABLE IF NOT EXISTS positions (
                        position_id VARCHAR(100) PRIMARY KEY,
                        symbol VARCHAR(20) NOT NULL,
                        side VARCHAR(10) NOT NULL,
                        entry_price DECIMAL(20,8) NOT NULL,
                        quantity DECIMAL(20,8) NOT NULL,
                        remaining_quantity DECIMAL(20,8) NOT NULL,
                        entry_time TIMESTAMP NOT NULL,
                        tp_levels TEXT,
                        tp_quantities TEXT,
                        current_tp_level INT DEFAULT 0,
                        initial_sl DECIMAL(20,8),
                        current_sl DECIMAL(20,8),
                        realized_pnl DECIMAL(20,8) DEFAULT 0,
                        unrealized_pnl DECIMAL(20,8) DEFAULT 0,
                        confidence DECIMAL(5,2) DEFAULT 0,
                        is_active BOOLEAN DEFAULT TRUE,
                        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                    ) ENGINE=InnoDB
                """
            }
        else:  # SQLite schemas (simplified for compatibility)
            return {
                # Raw tick data
                ticks_table: f"""
                    CREATE TABLE IF NOT EXISTS {ticks_table} (
                        id INTEGER PRIMARY KEY AUTOINCREMENT,
                        symbol TEXT NOT NULL,
                        timestamp INTEGER NOT NULL,
                        price REAL NOT NULL,
                        volume REAL NOT NULL,
                        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                    )
                """,
                
                # 1-second OHLC data
                ohlc_1s_table: f"""
                    CREATE TABLE IF NOT EXISTS {ohlc_1s_table} (
                        id INTEGER PRIMARY KEY AUTOINCREMENT,
                        symbol TEXT NOT NULL,
                        timestamp INTEGER NOT NULL,
                        open REAL NOT NULL,
                        high REAL NOT NULL,
                        low REAL NOT NULL,
                        close REAL NOT NULL,
                        volume REAL NOT NULL,
                        tick_count INTEGER DEFAULT 1,
                        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                        UNIQUE(symbol, timestamp)
                    )
                """,
                
                # 1-minute OHLC data
                ohlc_1m_table: f"""
                    CREATE TABLE IF NOT EXISTS {ohlc_1m_table} (
                        id INTEGER PRIMARY KEY AUTOINCREMENT,
                        symbol TEXT NOT NULL,
                        timestamp INTEGER NOT NULL,
                        open REAL NOT NULL,
                        high REAL NOT NULL,
                        low REAL NOT NULL,
                        close REAL NOT NULL,
                        volume REAL NOT NULL,
                        tick_count INTEGER DEFAULT 60,
                        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                        UNIQUE(symbol, timestamp)
                    )
                """,
                
                # Original market data table
                market_table: f"""
                    CREATE TABLE IF NOT EXISTS {market_table} (
                        id INTEGER PRIMARY KEY AUTOINCREMENT,
                        symbol TEXT NOT NULL,
                        timeframe TEXT NOT NULL,
                        timestamp INTEGER NOT NULL,
                        open REAL NOT NULL,
                        high REAL NOT NULL,
                        low REAL NOT NULL,
                        close REAL NOT NULL,
                        volume REAL NOT NULL,
                        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                        UNIQUE(symbol, timeframe, timestamp)
                    )
                """,
                
                # Indicators cache
                indicators_cache_table: f"""
                    CREATE TABLE IF NOT EXISTS {indicators_cache_table} (
                        id INTEGER PRIMARY KEY AUTOINCREMENT,
                        symbol TEXT NOT NULL,
                        timestamp INTEGER NOT NULL,
                        indicator_name TEXT NOT NULL,
                        indicator_value REAL,
                        indicator_json TEXT,
                        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                    )
                """,
                
                # Model training runs
                training_runs_table: f"""
                    CREATE TABLE IF NOT EXISTS {training_runs_table} (
                        id INTEGER PRIMARY KEY AUTOINCREMENT,
                        model_version INTEGER NOT NULL,
                        training_started_at TIMESTAMP NOT NULL,
                        training_completed_at TIMESTAMP,
                        samples_count INTEGER,
                        features_selected INTEGER,
                        accuracy REAL,
                        training_error TEXT,
                        status TEXT DEFAULT 'started',
                        config_json TEXT,
                        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                    )
                """,
                
                # Model metrics
                model_metrics_table: f"""
                    CREATE TABLE IF NOT EXISTS {model_metrics_table} (
                        metric_key TEXT PRIMARY KEY,
                        metric_value TEXT,
                        metric_type TEXT DEFAULT 'string',
                        updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                    )
                """,
                
                # Real-time prices
                'real_time_prices': """
                    CREATE TABLE IF NOT EXISTS real_time_prices (
                        symbol TEXT PRIMARY KEY,
                        price REAL NOT NULL,
                        bid_price REAL,
                        ask_price REAL,
                        bid_size REAL,
                        ask_size REAL,
                        timestamp INTEGER NOT NULL,
                        updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                    )
                """,
                
                # Positions
                'positions': """
                    CREATE TABLE IF NOT EXISTS positions (
                        position_id TEXT PRIMARY KEY,
                        symbol TEXT NOT NULL,
                        side TEXT NOT NULL,
                        entry_price REAL NOT NULL,
                        quantity REAL NOT NULL,
                        remaining_quantity REAL NOT NULL,
                        entry_time TIMESTAMP NOT NULL,
                        tp_levels TEXT,
                        tp_quantities TEXT,
                        current_tp_level INTEGER DEFAULT 0,
                        initial_sl REAL,
                        current_sl REAL,
                        realized_pnl REAL DEFAULT 0,
                        unrealized_pnl REAL DEFAULT 0,
                        confidence REAL DEFAULT 0,
                        is_active BOOLEAN DEFAULT TRUE,
                        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                    )
                """
            }

# Global instance
db_manager = DatabaseManager()
