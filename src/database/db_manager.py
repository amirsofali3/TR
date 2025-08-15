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
            self.backend = 'sqlite'
            self.sqlite_path = DATABASE_URL.replace("sqlite:///", "")
            # Create directory if needed
            os.makedirs(os.path.dirname(self.sqlite_path), exist_ok=True)
            logger.info(f"Using SQLite backend: {self.sqlite_path}")
        else:
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

# Global instance
db_manager = DatabaseManager()