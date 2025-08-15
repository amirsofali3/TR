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
        """Auto-detect database backend based on configuration"""
        # Check for MySQL configuration via environment variables
        mysql_enabled = os.getenv('MYSQL_ENABLED', 'false').lower() == 'true'
        mysql_db_set = os.getenv('MYSQL_DB') is not None
        
        if mysql_enabled and MYSQL_AVAILABLE:
            # Validate required MySQL credentials
            required_vars = ['MYSQL_HOST', 'MYSQL_USER', 'MYSQL_PASSWORD', 'MYSQL_DB']
            missing_vars = [var for var in required_vars if not os.getenv(var)]
            
            if missing_vars:
                # Check AUTO_FALLBACK_DB from env var first, then settings
                auto_fallback = os.getenv('AUTO_FALLBACK_DB', '').lower() == 'true'
                if not auto_fallback:
                    try:
                        from config.settings import AUTO_FALLBACK_DB
                        auto_fallback = AUTO_FALLBACK_DB
                    except ImportError:
                        auto_fallback = True  # Default to True if settings not available
                        
                logger.warning(f"[DB] MySQL enabled but missing required environment variables: {', '.join(missing_vars)}")
                
                if auto_fallback:
                    logger.info("[DB] AUTO_FALLBACK_DB=True, falling back to SQLite")
                    self._setup_sqlite_backend()
                    return
                else:
                    raise ValueError(f"MySQL enabled but missing required variables: {', '.join(missing_vars)}. Set AUTO_FALLBACK_DB=True to enable fallback.")
            
            # Setup MySQL backend
            self.backend = 'mysql'
            self.mysql_config = {
                'host': os.getenv('MYSQL_HOST', 'localhost'),
                'port': int(os.getenv('MYSQL_PORT', 3306)),
                'user': os.getenv('MYSQL_USER', 'root'),
                'password': os.getenv('MYSQL_PASSWORD', ''),
                'database': os.getenv('MYSQL_DB', 'trading_system'),
                'charset': os.getenv('MYSQL_CHARSET', 'utf8mb4'),
                'autocommit': True
            }
            logger.info(f"Using MySQL backend: {self.mysql_config['user']}@{self.mysql_config['host']}:{self.mysql_config['port']}/{self.mysql_config['database']}")
            
        elif mysql_db_set and not mysql_enabled:
            # User set MYSQL_DB but didn't enable MySQL - provide helpful message
            logger.warning(f"MYSQL_DB is set to '{os.getenv('MYSQL_DB')}' but MYSQL_ENABLED is not 'true'. Using SQLite instead.")
            logger.info("To use MySQL, set: export MYSQL_ENABLED=true")
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
        if self.backend == 'mysql':
            return {
                'backend': 'mysql',
                'host': self.mysql_config['host'],
                'port': self.mysql_config['port'],
                'database': self.mysql_config['database'],
                'charset': self.mysql_config['charset']
            }
        else:
            return {
                'backend': 'sqlite',
                'path': self.sqlite_path,
                'exists': os.path.exists(self.sqlite_path)
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
        """Get table creation schemas for both MySQL and SQLite"""
        # Use configurable table name - check env var first, then settings
        market_table = os.getenv('MYSQL_MARKET_DATA_TABLE')
        if not market_table:
            from config.settings import MYSQL_MARKET_DATA_TABLE
            market_table = MYSQL_MARKET_DATA_TABLE
        
        if self.backend == 'mysql':
            return {
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
        else:  # SQLite schemas
            return {
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