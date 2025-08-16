"""
Data Access Helper for OHLCV-only mode
Handles reading from candles table with multiple symbols
"""

import pandas as pd
import numpy as np
from typing import List, Optional, Dict, Any
from loguru import logger
from datetime import datetime, timedelta

from src.database.db_manager import db_manager
from config.settings import SUPPORTED_PAIRS


class CandleDataManager:
    """Manager for accessing OHLCV data from candles table"""
    
    def __init__(self):
        self.candles_table = "candles"
        
    async def initialize(self):
        """Initialize the data manager"""
        try:
            logger.info("[CANDLES] Initializing candle data manager...")
            
            # Check if candles table exists
            await self.ensure_candles_table_exists()
            
            # Log basic info about available data
            summary = await self.get_data_summary()
            logger.info(f"[CANDLES] Data summary: {summary}")
            
            logger.success("[CANDLES] Candle data manager initialized")
            
        except Exception as e:
            logger.error(f"[CANDLES] Failed to initialize: {e}")
            raise
    
    async def ensure_candles_table_exists(self):
        """Ensure candles table exists with proper schema"""
        try:
            backend = db_manager.backend
            
            if backend == 'mysql':
                create_table_sql = f"""
                CREATE TABLE IF NOT EXISTS {self.candles_table} (
                    id BIGINT AUTO_INCREMENT PRIMARY KEY,
                    symbol VARCHAR(20) NOT NULL,
                    timestamp BIGINT NOT NULL,
                    open DECIMAL(20,8) NOT NULL,
                    high DECIMAL(20,8) NOT NULL,
                    low DECIMAL(20,8) NOT NULL,
                    close DECIMAL(20,8) NOT NULL,
                    volume DECIMAL(20,8) NOT NULL,
                    datetime DATETIME NOT NULL,
                    INDEX idx_symbol_timestamp (symbol, timestamp),
                    INDEX idx_datetime (datetime),
                    UNIQUE KEY unique_symbol_timestamp (symbol, timestamp)
                )
                """
            else:
                create_table_sql = f"""
                CREATE TABLE IF NOT EXISTS {self.candles_table} (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    symbol TEXT NOT NULL,
                    timestamp INTEGER NOT NULL,
                    open REAL NOT NULL,
                    high REAL NOT NULL,
                    low REAL NOT NULL,
                    close REAL NOT NULL,
                    volume REAL NOT NULL,
                    datetime TEXT NOT NULL,
                    UNIQUE(symbol, timestamp)
                )
                """
            
            db_manager.execute(create_table_sql)
            logger.debug(f"[CANDLES] Ensured {self.candles_table} table exists")
            
        except Exception as e:
            logger.error(f"[CANDLES] Failed to create table: {e}")
            raise
    
    async def get_data_summary(self) -> Dict[str, Any]:
        """Get summary of available candle data"""
        try:
            # Get count by symbol
            count_sql = f"""
            SELECT symbol, COUNT(*) as count, 
                   MIN(datetime) as earliest, 
                   MAX(datetime) as latest
            FROM {self.candles_table}
            GROUP BY symbol
            ORDER BY symbol
            """
            
            results = db_manager.fetchall(count_sql)
            
            summary = {
                'symbols': [],
                'total_candles': 0,
                'date_range': {'earliest': None, 'latest': None}
            }
            
            overall_earliest = None
            overall_latest = None
            
            for row in results:
                symbol, count, earliest, latest = row
                summary['symbols'].append({
                    'symbol': symbol,
                    'count': count,
                    'earliest': earliest,
                    'latest': latest
                })
                summary['total_candles'] += count
                
                # Track overall date range
                if overall_earliest is None or (earliest and earliest < overall_earliest):
                    overall_earliest = earliest
                if overall_latest is None or (latest and latest > overall_latest):
                    overall_latest = latest
            
            summary['date_range']['earliest'] = overall_earliest
            summary['date_range']['latest'] = overall_latest
            
            return summary
            
        except Exception as e:
            logger.error(f"[CANDLES] Failed to get data summary: {e}")
            return {'symbols': [], 'total_candles': 0, 'date_range': {'earliest': None, 'latest': None}}
    
    async def load_candle_data(self, symbols: Optional[List[str]] = None, 
                              limit: Optional[int] = None, 
                              recent_only: bool = False,
                              recent_count: int = 1000) -> pd.DataFrame:
        """Load OHLCV candle data for specified symbols with safeguards"""
        try:
            if symbols is None:
                symbols = SUPPORTED_PAIRS
            
            logger.info(f"[CANDLES] Loading candle data for symbols: {symbols}")
            
            # Safeguard: Check if candles table has any data first
            total_count_sql = f"SELECT COUNT(*) FROM {self.candles_table}"
            total_result = db_manager.fetchall(total_count_sql)
            total_candles = total_result[0][0] if total_result else 0
            
            if total_candles == 0:
                logger.warning(f"[CANDLES] Candles table is empty - no data available")
                return pd.DataFrame()
            
            logger.debug(f"[CANDLES] Total candles in table: {total_candles}")
            
            # Build query with safeguards for missing symbols
            symbol_placeholders = ','.join(['%s'] * len(symbols))
            
            if db_manager.backend == 'mysql':
                if recent_only:
                    # Get recent candles per symbol
                    # NOTE: Explicitly select only required 8 columns to avoid rn column mismatch in DataFrame
                    base_sql = f"""
                    SELECT symbol, timestamp, open, high, low, close, volume, datetime FROM (
                        SELECT symbol, timestamp, open, high, low, close, volume, datetime,
                               ROW_NUMBER() OVER (PARTITION BY symbol ORDER BY timestamp DESC) as rn
                        FROM {self.candles_table}
                        WHERE symbol IN ({symbol_placeholders})
                    ) ranked
                    WHERE rn <= %s
                    ORDER BY symbol, timestamp
                    """
                    params = tuple(symbols) + (recent_count,)
                else:
                    base_sql = f"""
                    SELECT symbol, timestamp, open, high, low, close, volume, datetime
                    FROM {self.candles_table}
                    WHERE symbol IN ({symbol_placeholders})
                    ORDER BY symbol, timestamp
                    """
                    params = tuple(symbols)
            else:
                # SQLite version
                if recent_only:
                    # NOTE: Explicitly select only required 8 columns to avoid rn column mismatch in DataFrame
                    base_sql = f"""
                    SELECT symbol, timestamp, open, high, low, close, volume, datetime FROM (
                        SELECT symbol, timestamp, open, high, low, close, volume, datetime,
                               ROW_NUMBER() OVER (PARTITION BY symbol ORDER BY timestamp DESC) as rn
                        FROM {self.candles_table}
                        WHERE symbol IN ({symbol_placeholders})
                    ) ranked
                    WHERE rn <= ?
                    ORDER BY symbol, timestamp
                    """
                    params = tuple(symbols) + (recent_count,)
                else:
                    base_sql = f"""
                    SELECT symbol, timestamp, open, high, low, close, volume, datetime
                    FROM {self.candles_table}
                    WHERE symbol IN ({symbol_placeholders})
                    ORDER BY symbol, timestamp
                    """
                    params = tuple(symbols)
            
            if limit and not recent_only:
                base_sql += f" LIMIT {limit}"
            
            # Execute query
            results = db_manager.fetchall(base_sql, params)
            
            if not results:
                logger.warning(f"[CANDLES] No data found for symbols: {symbols}")
                
                # Check if any of the symbols exist in the table
                symbol_check_sql = f"SELECT DISTINCT symbol FROM {self.candles_table}"
                available_symbols_result = db_manager.fetchall(symbol_check_sql)
                available_symbols = [row[0] for row in available_symbols_result] if available_symbols_result else []
                
                missing_symbols = [sym for sym in symbols if sym not in available_symbols]
                if missing_symbols:
                    logger.warning(f"[CANDLES] Missing symbols in candles table: {missing_symbols}")
                    logger.info(f"[CANDLES] Available symbols: {available_symbols}")
                
                return pd.DataFrame()
            
            # Convert to DataFrame
            # Defensive safeguard: if results have unexpected extra columns (e.g., rn), strip them
            if results and len(results[0]) > 8:
                logger.warning(f"[CANDLES] Query returned {len(results[0])} columns, expected 8. Trimming extra columns.")
                results = [row[:8] for row in results]
            
            df = pd.DataFrame(results, columns=[
                'symbol', 'timestamp', 'open', 'high', 'low', 'close', 'volume', 'datetime'
            ])
            
            # Safeguard: Validate data types and handle conversion errors
            try:
                df['timestamp'] = pd.to_numeric(df['timestamp'], errors='coerce')
                df['open'] = pd.to_numeric(df['open'], errors='coerce')
                df['high'] = pd.to_numeric(df['high'], errors='coerce')
                df['low'] = pd.to_numeric(df['low'], errors='coerce')
                df['close'] = pd.to_numeric(df['close'], errors='coerce')
                df['volume'] = pd.to_numeric(df['volume'], errors='coerce')
                df['datetime'] = pd.to_datetime(df['datetime'], errors='coerce')
                
                # Drop rows with invalid data
                initial_count = len(df)
                df = df.dropna(subset=['open', 'high', 'low', 'close', 'volume', 'timestamp'])
                final_count = len(df)
                
                if initial_count != final_count:
                    logger.warning(f"[CANDLES] Dropped {initial_count - final_count} rows with invalid data")
                
            except Exception as e:
                logger.error(f"[CANDLES] Data type conversion failed: {e}")
                return pd.DataFrame()
            
            logger.info(f"[CANDLES] Loaded {len(df)} valid candles for {len(symbols)} symbols")
            
            # Log per-symbol counts with warnings
            symbol_counts = df['symbol'].value_counts()
            for symbol in symbols:
                count = symbol_counts.get(symbol, 0)
                if count == 0:
                    logger.warning(f"[CANDLES] No data found for symbol: {symbol}")
                else:
                    # Enhanced logging for recent window loads - reuse existing pattern
                    if recent_only:
                        logger.info(f"[CANDLES] Recent window - {symbol}: {count} candles")
                    else:
                        logger.debug(f"[CANDLES] {symbol}: {count} candles")
                    
                    # Safeguard: Check for reasonable data distribution
                    if count < 100:
                        logger.warning(f"[CANDLES] Symbol {symbol} has very few candles: {count}")
            
            
            return df
            
        except Exception as e:
            logger.error(f"[CANDLES] Failed to load candle data: {e}")
            return pd.DataFrame()
    
    async def get_recent_candles(self, symbols: Optional[List[str]] = None, count: int = 1000) -> pd.DataFrame:
        """Get recent candles for RFE selection window"""
        return await self.load_candle_data(symbols=symbols, recent_only=True, recent_count=count)
    
    async def get_full_historical_data(self, symbols: Optional[List[str]] = None) -> pd.DataFrame:
        """Get full historical data for final model training"""
        return await self.load_candle_data(symbols=symbols, recent_only=False)
    
    def prepare_features_dataframe(self, df: pd.DataFrame) -> pd.DataFrame:
        """Prepare DataFrame with basic OHLCV features for indicator calculation"""
        if df.empty:
            return df
        
        # Ensure required columns are present
        required_cols = ['open', 'high', 'low', 'close', 'volume', 'timestamp']
        missing_cols = [col for col in required_cols if col not in df.columns]
        
        if missing_cols:
            logger.error(f"[CANDLES] Missing required columns: {missing_cols}")
            return pd.DataFrame()
        
        # Sort by timestamp to ensure proper order
        df_sorted = df.sort_values(['symbol', 'timestamp']).copy()
        
        logger.info(f"[CANDLES] Prepared {len(df_sorted)} rows for feature calculation")
        return df_sorted
    
    async def validate_data_availability(self, symbols: Optional[List[str]] = None, min_samples: int = 400) -> Dict[str, bool]:
        """Validate that sufficient data is available for each symbol"""
        if symbols is None:
            symbols = SUPPORTED_PAIRS
        
        validation_results = {}
        
        for symbol in symbols:
            try:
                count_sql = f"SELECT COUNT(*) FROM {self.candles_table} WHERE symbol = %s"
                if db_manager.backend != 'mysql':
                    count_sql = f"SELECT COUNT(*) FROM {self.candles_table} WHERE symbol = ?"
                
                result = db_manager.fetchall(count_sql, (symbol,))
                count = result[0][0] if result else 0
                
                validation_results[symbol] = count >= min_samples
                
                if count < min_samples:
                    logger.warning(f"[CANDLES] Insufficient data for {symbol}: {count} < {min_samples}")
                else:
                    logger.debug(f"[CANDLES] {symbol} validation passed: {count} samples")
                    
            except Exception as e:
                logger.error(f"[CANDLES] Failed to validate {symbol}: {e}")
                validation_results[symbol] = False
        
        return validation_results


# Global instance
candle_data_manager = CandleDataManager()