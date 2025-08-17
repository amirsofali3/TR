"""
Binance Data Collector for fetching cryptocurrency market data
"""

import asyncio
import aiohttp
import os
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple
from loguru import logger
import time
import sqlite3
# Import binance with fallback to stub
try:
    from binance.client import Client
    from binance.exceptions import BinanceAPIException
    BINANCE_AVAILABLE = True
except ImportError:
    from binance_stub import Client, BinanceAPIException  
    BINANCE_AVAILABLE = False
    logger.warning("python-binance not available - using stub for testing")

try:
    import ccxt.async_support as ccxt
    CCXT_AVAILABLE = True
except ImportError:
    CCXT_AVAILABLE = False
    logger.warning("ccxt not available - using public API only")

# Import database manager for MySQL migration
from src.database.db_manager import db_manager

from config.settings import *

class BinanceDataCollector:
    """Handles data collection from Binance API with bootstrap support"""
    
    def __init__(self):
        self.client = None
        self.ws_client = None
        self.ccxt_exchange = None
        self.db_path = DATABASE_URL.replace("sqlite:///", "")
        self.price_cache = {}
        self.data_cache = {}
        self.running = False
        
        # Bootstrap collection state
        self.bootstrap_active = False
        self.bootstrap_start_time = None
        self.bootstrap_duration = INITIAL_COLLECTION_DURATION_SEC
        self.bootstrap_progress = 0.0
        self.bootstrap_records_collected = {}
        self.bootstrap_total_records = 0
        
        # Diagnostics and failure tracking
        self.tick_store_failures = 0
        self.tick_store_successes = 0
        self.last_progress_emit = 0
        self.zero_records_warning_sent = False
        
    async def initialize(self):
        """Initialize the data collector"""
        try:
            logger.info("Initializing Binance data collector...")
            
            # Check if API keys are configured
            has_api_keys = (BINANCE_API_KEY != "your_binance_api_key_here" and 
                           BINANCE_SECRET_KEY != "your_binance_secret_key_here" and 
                           BINANCE_API_KEY and BINANCE_SECRET_KEY)
            
            if has_api_keys:
                logger.info("✅ API keys configured - initializing authenticated client")
                # Initialize Binance client
                self.client = Client(
                    api_key=BINANCE_API_KEY,
                    api_secret=BINANCE_SECRET_KEY,
                    testnet=DEMO_MODE
                )
                
                # Initialize CCXT for additional functionality if available
                if CCXT_AVAILABLE:
                    self.ccxt_exchange = ccxt.binance({
                        'apiKey': BINANCE_API_KEY,
                        'secret': BINANCE_SECRET_KEY,
                        'sandbox': DEMO_MODE,
                        'enableRateLimit': True,
                    })
                else:
                    logger.info("CCXT not available - using python-binance only")
                
                # Test API connection
                try:
                    account_info = self.client.get_account()
                    logger.success(f"✅ API connection successful - Account status: {account_info.get('status', 'UNKNOWN')}")
                except Exception as api_error:
                    logger.warning(f"⚠️  API keys configured but connection failed: {api_error}")
                    logger.info("Falling back to public API mode...")
                    self.client = None
                    self.ccxt_exchange = None
            else:
                logger.info("⚠️  API keys not configured - using public API only")
                logger.info("This limits functionality but allows basic market analysis")
                self.client = None
                self.ccxt_exchange = None
            
            # Initialize database
            await self.init_database()
            
            # Bootstrap mode is handled separately by start_bootstrap_collection()
            # Don't fetch initial data automatically anymore
            logger.info("📊 Data collector initialized - ready for bootstrap collection")
            
            logger.success("✅ Binance data collector initialized successfully")
            
        except Exception as e:
            logger.error(f"❌ Failed to initialize data collector: {e}")
            logger.error("This may be due to:")
            logger.error("  1. Missing dependencies (pandas, aiohttp, etc.)")
            logger.error("  2. Network connectivity issues")
            logger.error("  3. Invalid API keys")
            logger.error("  4. Database initialization problems")
            raise
    
    async def init_database(self):
        """Initialize database for storing market data (Complete Pipeline Restructure)"""
        try:
            logger.info("[DB] Initializing database schema...")
            
            # Use the enhanced database manager's schema creation
            schema_success = db_manager.ensure_schema()
            if not schema_success:
                logger.error("[DB] Failed to create database schema")
                raise Exception("Database schema creation failed")
                
            logger.success("[DB] Database schema initialized successfully")
            
        except Exception as e:
            logger.error(f"❌ Database initialization failed: {e}")
            raise
    
    async def fetch_initial_data(self):
        """Fetch initial historical data for all supported pairs"""
        try:
            logger.info(f"📈 Fetching initial historical data for {len(SUPPORTED_PAIRS)} pairs...")
            logger.info(f"⏱️  This may take a few moments...")
            
            successful_fetches = 0
            total_candles = 0
            
            for i, symbol in enumerate(SUPPORTED_PAIRS, 1):
                try:
                    logger.info(f"📊 Fetching data for {symbol} ({i}/{len(SUPPORTED_PAIRS)})...")
                    
                    # Fetch OHLCV data for the default timeframe
                    data = await self.fetch_historical_data(symbol, DEFAULT_TIMEFRAME, MAX_HISTORICAL_DAYS)
                    if data is not None and len(data) > 0:
                        await self.store_market_data(symbol, DEFAULT_TIMEFRAME, data)
                        successful_fetches += 1
                        total_candles += len(data)
                        logger.success(f"✅ Fetched {len(data)} candles for {symbol}")
                    else:
                        logger.warning(f"⚠️  No data received for {symbol}")
                    
                    # Small delay to avoid rate limits
                    await asyncio.sleep(0.1)
                    
                except Exception as e:
                    logger.error(f"❌ Failed to fetch initial data for {symbol}: {e}")
                    continue
            
            if successful_fetches > 0:
                logger.success(f"🎉 Initial data fetch completed: {successful_fetches}/{len(SUPPORTED_PAIRS)} pairs, {total_candles} total candles")
            else:
                logger.error("❌ No initial data could be fetched for any pairs")
                logger.error("This may prevent market analysis from working properly")
                logger.error("Check your internet connection and try again")
            
        except Exception as e:
            logger.error(f"❌ Failed to fetch initial data: {e}")
            logger.error("The system will continue but analysis may be limited")
    
    async def fetch_historical_data(self, symbol: str, timeframe: str, days: int) -> Optional[pd.DataFrame]:
        """Fetch historical OHLCV data for a symbol"""
        try:
            # Calculate start time
            end_time = datetime.now()
            start_time = end_time - timedelta(days=days)
            
            # Convert to milliseconds
            start_ms = int(start_time.timestamp() * 1000)
            end_ms = int(end_time.timestamp() * 1000)
            
            # Fetch data from Binance
            if self.client:
                klines = self.client.get_historical_klines(
                    symbol=symbol,
                    interval=timeframe,
                    start_str=str(start_ms),
                    end_str=str(end_ms)
                )
            else:
                # Fallback to public API
                async with aiohttp.ClientSession() as session:
                    url = f"https://api.binance.com/api/v3/klines"
                    params = {
                        'symbol': symbol,
                        'interval': timeframe,
                        'startTime': start_ms,
                        'endTime': end_ms,
                        'limit': 1000
                    }
                    async with session.get(url, params=params) as response:
                        if response.status == 200:
                            klines = await response.json()
                        else:
                            logger.error(f"Failed to fetch data for {symbol}: {response.status}")
                            return None
            
            if not klines:
                return None
            
            # Convert to DataFrame
            df = pd.DataFrame(klines, columns=[
                'timestamp', 'open', 'high', 'low', 'close', 'volume',
                'close_time', 'quote_asset_volume', 'number_of_trades',
                'taker_buy_base_asset_volume', 'taker_buy_quote_asset_volume', 'ignore'
            ])
            
            # Keep only required columns and convert types
            df = df[['timestamp', 'open', 'high', 'low', 'close', 'volume']].copy()
            df['timestamp'] = pd.to_numeric(df['timestamp'])
            df['open'] = pd.to_numeric(df['open'])
            df['high'] = pd.to_numeric(df['high'])
            df['low'] = pd.to_numeric(df['low'])
            df['close'] = pd.to_numeric(df['close'])
            df['volume'] = pd.to_numeric(df['volume'])
            
            # Add datetime column for convenience
            df['datetime'] = pd.to_datetime(df['timestamp'], unit='ms')
            
            return df
            
        except Exception as e:
            logger.error(f"Failed to fetch historical data for {symbol}: {e}")
            return None
    
    async def store_market_data(self, symbol: str, timeframe: str, data: pd.DataFrame):
        """Store market data in database (MySQL migration)"""
        try:
            # Prepare data for insertion
            data_to_insert = data[['timestamp', 'open', 'high', 'low', 'close', 'volume']].copy()
            data_to_insert['symbol'] = symbol
            data_to_insert['timeframe'] = timeframe
            
            # Use appropriate INSERT statement based on backend
            if db_manager.backend == 'mysql':
                insert_query = '''
                    INSERT IGNORE INTO market_data 
                    (symbol, timeframe, timestamp, open, high, low, close, volume)
                    VALUES (%s, %s, %s, %s, %s, %s, %s, %s)
                '''
            else:
                insert_query = '''
                    INSERT OR REPLACE INTO market_data 
                    (symbol, timeframe, timestamp, open, high, low, close, volume)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                '''
            
            # Batch insert for efficiency
            params_list = []
            for _, row in data_to_insert.iterrows():
                params_list.append((
                    row['symbol'], row['timeframe'], row['timestamp'],
                    row['open'], row['high'], row['low'], row['close'], row['volume']
                ))
            
            db_manager.executemany(insert_query, params_list)
            
        except Exception as e:
            logger.error(f"Failed to store market data for {symbol}: {e}")
    
    async def get_historical_data(self, symbol: str, timeframe: str, limit: int = 500) -> Optional[pd.DataFrame]:
        """Get historical data from database (MySQL migration)"""
        try:
            # Use appropriate parameter placeholder for backend
            if db_manager.backend == 'mysql':
                query = '''
                    SELECT timestamp, open, high, low, close, volume 
                    FROM market_data 
                    WHERE symbol = %s AND timeframe = %s 
                    ORDER BY timestamp DESC 
                    LIMIT %s
                '''
            else:
                query = '''
                    SELECT timestamp, open, high, low, close, volume 
                    FROM market_data 
                    WHERE symbol = ? AND timeframe = ? 
                    ORDER BY timestamp DESC 
                    LIMIT ?
                '''
            
            conn = db_manager.get_pandas_connection()
            df = pd.read_sql_query(query, conn, params=(symbol, timeframe, limit))
            conn.close()
            
            if len(df) > 0:
                # Sort by timestamp ascending
                df = df.sort_values('timestamp').reset_index(drop=True)
                df['datetime'] = pd.to_datetime(df['timestamp'], unit='ms')
                return df
            
            return None
            
        except Exception as e:
            logger.error(f"Failed to get historical data for {symbol}: {e}")
            return None
    
    async def get_real_time_price(self, symbol: str) -> Optional[Dict]:
        """Get real-time price for a symbol"""
        try:
            if symbol in self.price_cache:
                cache_time = self.price_cache[symbol].get('timestamp', 0)
                if time.time() - cache_time < 5:  # Use cache if less than 5 seconds old
                    return self.price_cache[symbol]
            
            # Fetch current price
            if self.client:
                ticker = self.client.get_symbol_ticker(symbol=symbol)
                order_book = self.client.get_orderbook_ticker(symbol=symbol)
                
                price_data = {
                    'symbol': symbol,
                    'price': float(ticker['price']),
                    'bid_price': float(order_book['bidPrice']),
                    'ask_price': float(order_book['askPrice']),
                    'bid_size': float(order_book['bidQty']),
                    'ask_size': float(order_book['askQty']),
                    'timestamp': time.time()
                }
            else:
                # Fallback to public API
                async with aiohttp.ClientSession() as session:
                    url = f"https://api.binance.com/api/v3/ticker/price?symbol={symbol}"
                    async with session.get(url) as response:
                        if response.status == 200:
                            ticker = await response.json()
                            price_data = {
                                'symbol': symbol,
                                'price': float(ticker['price']),
                                'bid_price': None,
                                'ask_price': None,
                                'bid_size': None,
                                'ask_size': None,
                                'timestamp': time.time()
                            }
                        else:
                            return None
            
            # Cache the price
            self.price_cache[symbol] = price_data
            
            # Store in database
            await self.store_real_time_price(price_data)
            
            return price_data
            
        except Exception as e:
            logger.error(f"Failed to get real-time price for {symbol}: {e}")
            return None
    
    async def store_real_time_price(self, price_data: Dict):
        """Store real-time price in database (MySQL migration)"""
        try:
            # Use appropriate INSERT statement based on backend
            if db_manager.backend == 'mysql':
                query = '''
                    INSERT INTO real_time_prices 
                    (symbol, price, bid_price, ask_price, bid_size, ask_size, timestamp) 
                    VALUES (%s, %s, %s, %s, %s, %s, %s)
                    ON DUPLICATE KEY UPDATE 
                    price=VALUES(price), bid_price=VALUES(bid_price), ask_price=VALUES(ask_price),
                    bid_size=VALUES(bid_size), ask_size=VALUES(ask_size), timestamp=VALUES(timestamp),
                    updated_at=CURRENT_TIMESTAMP
                '''
            else:
                query = '''
                    INSERT OR REPLACE INTO real_time_prices 
                    (symbol, price, bid_price, ask_price, bid_size, ask_size, timestamp) 
                    VALUES (?, ?, ?, ?, ?, ?, ?)
                '''
            
            db_manager.execute(query, (
                price_data['symbol'],
                price_data['price'],
                price_data.get('bid_price'),
                price_data.get('ask_price'),
                price_data.get('bid_size'),
                price_data.get('ask_size'),
                int(price_data['timestamp'] * 1000)
            ))
            
        except Exception as e:
            logger.error(f"Failed to store real-time price: {e}")
    
    async def update_data(self):
        """Update market data for all supported pairs"""
        try:
            for symbol in SUPPORTED_PAIRS:
                try:
                    # Fetch latest data
                    data = await self.fetch_historical_data(symbol, DEFAULT_TIMEFRAME, 2)  # Last 2 days
                    if data is not None and len(data) > 0:
                        await self.store_market_data(symbol, DEFAULT_TIMEFRAME, data)
                    
                    # Update real-time price
                    await self.get_real_time_price(symbol)
                    
                    # Small delay
                    await asyncio.sleep(0.1)
                    
                except Exception as e:
                    logger.warning(f"Failed to update data for {symbol}: {e}")
                    continue
            
        except Exception as e:
            logger.error(f"Failed to update market data: {e}")
    
    async def start_real_time_updates(self):
        """Start real-time data updates"""
        self.running = True
        logger.info("Starting real-time data updates...")
        
        while self.running:
            try:
                await self.update_data()
                await asyncio.sleep(UPDATE_INTERVAL)
                
            except Exception as e:
                logger.error(f"Error in real-time updates: {e}")
                await asyncio.sleep(5)
    
    # ==================== BOOTSTRAP COLLECTION SYSTEM ====================
    
    async def start_bootstrap_collection(self, duration: int = None) -> bool:
        """Start bootstrap data collection phase (Complete Pipeline Restructure)"""
        try:
            self.bootstrap_duration = duration or INITIAL_COLLECTION_DURATION_SEC
            self.bootstrap_active = True
            self.bootstrap_start_time = datetime.now()
            self.bootstrap_progress = 0.0
            self.bootstrap_records_collected = {symbol: 0 for symbol in SUPPORTED_PAIRS}
            self.bootstrap_total_records = 0
            
            logger.info(f"[BOOTSTRAP] Starting bootstrap data collection for {self.bootstrap_duration} seconds")
            logger.info(f"[BOOTSTRAP] Collecting {INITIAL_COLLECTION_TIMEFRAME} data for {len(SUPPORTED_PAIRS)} symbols")
            logger.info(f"[BOOTSTRAP] Symbols: {', '.join(SUPPORTED_PAIRS)}")
            
            # Start collection loop
            collection_success = await self._run_bootstrap_collection()
            
            # Mark bootstrap as complete
            self.bootstrap_active = False
            
            if collection_success:
                logger.success(f"[BOOTSTRAP] Data collection completed successfully!")
                logger.info(f"[BOOTSTRAP] Total records collected: {self.bootstrap_total_records}")
                logger.info(f"[BOOTSTRAP] Records per symbol: {dict(self.bootstrap_records_collected)}")
                
                # Aggregate data to 1-minute OHLC if needed
                await self._aggregate_bootstrap_data()
                
                return True
            else:
                logger.error("[BOOTSTRAP] Data collection failed")
                return False
                
        except Exception as e:
            logger.error(f"[BOOTSTRAP] Failed to start bootstrap collection: {e}")
            self.bootstrap_active = False
            return False
    
    async def _run_bootstrap_collection(self) -> bool:
        """Run the actual bootstrap data collection loop"""
        try:
            start_time = time.time()
            end_time = start_time + self.bootstrap_duration
            
            logger.info(f"[COLLECT] Starting collection loop for {self.bootstrap_duration}s")
            
            while time.time() < end_time and self.bootstrap_active:
                collection_start = time.time()
                
                # Update progress
                elapsed = time.time() - start_time
                self.bootstrap_progress = min((elapsed / self.bootstrap_duration) * 100, 100.0)
                
                # Collect data for all symbols
                for symbol in SUPPORTED_PAIRS:
                    try:
                        # Fetch current price data (simulating 1-second ticks)
                        await self._collect_tick_data(symbol)
                        
                    except Exception as e:
                        logger.debug(f"[COLLECT] Failed to collect tick for {symbol}: {e}")
                        continue
                
                # Calculate sleep time to maintain 1-second intervals
                collection_time = time.time() - collection_start
                sleep_time = max(0, 1.0 - collection_time)
                
                if sleep_time > 0:
                    await asyncio.sleep(sleep_time)
                
                # Log progress every 60 seconds and emit WebSocket events every 5 seconds
                current_time = time.time()
                if int(elapsed) % 60 == 0 and int(elapsed) > 0:
                    remaining = max(0, self.bootstrap_duration - elapsed)
                    logger.info(f"[COLLECT] Progress: {self.bootstrap_progress:.1f}% | "
                              f"Elapsed: {int(elapsed)}s | Remaining: {int(remaining)}s | "
                              f"Total records: {self.bootstrap_total_records}")
                
                # Emit WebSocket progress event every 5 seconds
                if current_time - self.last_progress_emit >= 5.0:
                    self.last_progress_emit = current_time
                    await self._emit_progress_event()
                
                # Check for zero records warning after 120 seconds
                if elapsed >= 120 and self.bootstrap_total_records == 0 and not self.zero_records_warning_sent:
                    logger.error("[COLLECT] No records collected after 120 seconds - potential data collection failure")
                    await self._emit_collection_warning("No records collected after 120 seconds")
                    self.zero_records_warning_sent = True
            
            logger.success(f"[COLLECT] Collection loop completed")
            return True
            
        except Exception as e:
            logger.error(f"[COLLECT] Collection loop failed: {e}")
            return False
    
    async def _collect_tick_data(self, symbol: str):
        """Collect a single tick of data for a symbol"""
        try:
            # Get current price from Binance API
            if self.client:
                ticker = self.client.get_symbol_ticker(symbol=symbol)
                price = float(ticker['price'])
            else:
                # Fallback to public API
                async with aiohttp.ClientSession() as session:
                    url = f"https://api.binance.com/api/v3/ticker/price"
                    params = {'symbol': symbol}
                    async with session.get(url, params=params) as response:
                        if response.status == 200:
                            data = await response.json()
                            price = float(data['price'])
                        else:
                            return  # Skip this tick
            
            # Store tick data
            timestamp = int(time.time() * 1000)  # Milliseconds
            await self._store_tick_data(symbol, timestamp, price, 1.0)  # Volume = 1 for simplicity
            
        except Exception as e:
            logger.debug(f"[COLLECT] Failed to collect tick for {symbol}: {e}")
    
    async def _store_tick_data(self, symbol: str, timestamp: int, price: float, volume: float):
        """Store tick data in the database"""
        try:
            # Get table name
            ticks_table = os.getenv('MYSQL_MARKET_TICKS_TABLE', 'market_ticks')
            
            if db_manager.backend == 'mysql':
                insert_query = f'''
                    INSERT INTO {ticks_table} (symbol, timestamp, price, volume)
                    VALUES (%s, %s, %s, %s)
                '''
                db_manager.execute(insert_query, (symbol, timestamp, price, volume))
            else:
                insert_query = f'''
                    INSERT INTO {ticks_table} (symbol, timestamp, price, volume)
                    VALUES (?, ?, ?, ?)
                '''
                db_manager.execute(insert_query, (symbol, timestamp, price, volume))
            
            # Success tracking
            self.tick_store_failures = 0  # Reset failure counter on success
            self.tick_store_successes += 1
            self.bootstrap_total_records += 1
            
            # Increment per-symbol counter
            if symbol in self.bootstrap_records_collected:
                self.bootstrap_records_collected[symbol] += 1
            
            # Log first 50 successes
            if self.tick_store_successes <= 50:
                logger.debug(f"[COLLECT] Stored tick for {symbol}: price={price}, timestamp={timestamp}")
                if self.tick_store_successes == 50:
                    logger.info("[COLLECT] First 50 ticks stored successfully")
            
        except Exception as e:
            # Failure tracking and escalated logging
            self.tick_store_failures += 1
            
            if self.tick_store_failures <= 5:
                logger.warning(f"[COLLECT] Failed to store tick data for {symbol}: {e}")
            else:
                logger.error(f"[COLLECT] Failed to store tick data for {symbol} (failure #{self.tick_store_failures}): {e}")
    
    async def _aggregate_bootstrap_data(self):
        """Aggregate collected tick data to 1-minute OHLC"""
        try:
            logger.info("[COLLECT] Aggregating tick data to 1-minute OHLC...")
            
            for symbol in SUPPORTED_PAIRS:
                await self._aggregate_symbol_to_1m(symbol)
            
            logger.success("[COLLECT] Data aggregation completed")
            
        except Exception as e:
            logger.error(f"[COLLECT] Data aggregation failed: {e}")
    
    async def _aggregate_symbol_to_1m(self, symbol: str):
        """Aggregate tick data for a symbol to 1-minute OHLC"""
        try:
            # Get table names
            ticks_table = os.getenv('MYSQL_MARKET_TICKS_TABLE', 'market_ticks')
            ohlc_1m_table = os.getenv('MYSQL_OHLC_1M_TABLE', 'ohlc_1m')
            
            # Query tick data
            if db_manager.backend == 'mysql':
                query = f'''
                    SELECT timestamp, price, volume FROM {ticks_table}
                    WHERE symbol = %s ORDER BY timestamp
                '''
                rows = db_manager.fetchall(query, (symbol,))
            else:
                query = f'''
                    SELECT timestamp, price, volume FROM {ticks_table}
                    WHERE symbol = ? ORDER BY timestamp
                '''
                rows = db_manager.fetchall(query, (symbol,))
            
            if not rows:
                return
            
            # Group by minute and calculate OHLC
            minute_data = {}
            for timestamp, price, volume in rows:
                # Round to minute
                minute_ts = (timestamp // 60000) * 60000
                
                if minute_ts not in minute_data:
                    minute_data[minute_ts] = {
                        'open': price,
                        'high': price,
                        'low': price,
                        'close': price,
                        'volume': volume,
                        'tick_count': 1
                    }
                else:
                    data = minute_data[minute_ts]
                    data['high'] = max(data['high'], price)
                    data['low'] = min(data['low'], price)
                    data['close'] = price  # Last price is close
                    data['volume'] += volume
                    data['tick_count'] += 1
            
            # Insert aggregated data
            for minute_ts, ohlc_data in minute_data.items():
                if db_manager.backend == 'mysql':
                    insert_query = f'''
                        INSERT INTO {ohlc_1m_table} 
                        (symbol, timestamp, open, high, low, close, volume, tick_count)
                        VALUES (%s, %s, %s, %s, %s, %s, %s, %s)
                        ON DUPLICATE KEY UPDATE
                        open = VALUES(open), high = VALUES(high), low = VALUES(low),
                        close = VALUES(close), volume = VALUES(volume), tick_count = VALUES(tick_count)
                    '''
                    params = (symbol, minute_ts, ohlc_data['open'], ohlc_data['high'], 
                             ohlc_data['low'], ohlc_data['close'], ohlc_data['volume'], 
                             ohlc_data['tick_count'])
                else:
                    insert_query = f'''
                        INSERT OR REPLACE INTO {ohlc_1m_table}
                        (symbol, timestamp, open, high, low, close, volume, tick_count)
                        VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                    '''
                    params = (symbol, minute_ts, ohlc_data['open'], ohlc_data['high'], 
                             ohlc_data['low'], ohlc_data['close'], ohlc_data['volume'], 
                             ohlc_data['tick_count'])
                
                db_manager.execute(insert_query, params)
            
            logger.info(f"[COLLECT] Aggregated {len(minute_data)} 1-minute candles for {symbol}")
            
        except Exception as e:
            logger.error(f"[COLLECT] Failed to aggregate data for {symbol}: {e}")
    
    async def start_realtime_data_collection(self):
        """Start continuous real-time data collection for candles table (NEW)"""
        try:
            logger.info("[REALTIME] Starting continuous real-time data collection...")
            
            from src.data_access.candle_data_manager import candle_data_manager
            
            while self.running:
                try:
                    # Collect latest 1-minute candles for all supported pairs
                    for symbol in SUPPORTED_PAIRS:
                        await self._collect_latest_candle(symbol)
                    
                    # Wait for next update cycle (1 second as configured)
                    from config.settings import REALTIME_DATA_UPDATE_INTERVAL
                    await asyncio.sleep(REALTIME_DATA_UPDATE_INTERVAL)
                    
                except Exception as e:
                    logger.error(f"[REALTIME] Error in data collection cycle: {e}")
                    await asyncio.sleep(5)  # Wait 5 seconds on error
            
            logger.info("[REALTIME] Real-time data collection stopped")
            
        except Exception as e:
            logger.error(f"[REALTIME] Failed to start real-time data collection: {e}")
    
    async def _collect_latest_candle(self, symbol: str):
        """Collect the latest 1-minute candle for a symbol (NEW)"""
        try:
            # Get latest kline data from Binance
            if self.client:
                # Use authenticated client if available
                klines = self.client.get_klines(
                    symbol=symbol,
                    interval=Client.KLINE_INTERVAL_1MINUTE,
                    limit=1
                )
            else:
                # Use public API via aiohttp
                url = f"https://api.binance.com/api/v3/klines"
                params = {
                    'symbol': symbol,
                    'interval': '1m',
                    'limit': 1
                }
                
                async with aiohttp.ClientSession() as session:
                    async with session.get(url, params=params) as response:
                        if response.status == 200:
                            klines = await response.json()
                        else:
                            logger.error(f"[REALTIME] Failed to get data for {symbol}: HTTP {response.status}")
                            return
            
            if not klines:
                return
                
            # Extract OHLCV data
            kline = klines[0]
            timestamp = int(kline[0])  # Open time
            open_price = float(kline[1])
            high_price = float(kline[2])
            low_price = float(kline[3])
            close_price = float(kline[4])
            volume = float(kline[5])
            
            # Update candles table via candle_data_manager
            from src.data_access.candle_data_manager import candle_data_manager
            await candle_data_manager.update_realtime_candle(
                symbol, timestamp, open_price, high_price, low_price, close_price, volume
            )
            
            logger.debug(f"[REALTIME] Updated candle for {symbol}: {close_price}")
            
        except Exception as e:
            logger.error(f"[REALTIME] Failed to collect latest candle for {symbol}: {e}")
    
    async def get_current_prices(self, symbols: Optional[List[str]] = None) -> Dict[str, float]:
        """Get current prices for symbols (NEW - for real-time indicator calculations)"""
        try:
            if symbols is None:
                symbols = SUPPORTED_PAIRS
            
            prices = {}
            
            if self.client:
                # Use authenticated client for ticker prices
                ticker_prices = self.client.get_all_tickers()
                price_map = {ticker['symbol']: float(ticker['price']) for ticker in ticker_prices}
                
                for symbol in symbols:
                    if symbol in price_map:
                        prices[symbol] = price_map[symbol]
            else:
                # Use public API
                url = "https://api.binance.com/api/v3/ticker/price"
                
                async with aiohttp.ClientSession() as session:
                    async with session.get(url) as response:
                        if response.status == 200:
                            ticker_data = await response.json()
                            price_map = {ticker['symbol']: float(ticker['price']) for ticker in ticker_data}
                            
                            for symbol in symbols:
                                if symbol in price_map:
                                    prices[symbol] = price_map[symbol]
            
            logger.debug(f"[REALTIME] Retrieved current prices for {len(prices)} symbols")
            return prices
            
        except Exception as e:
            logger.error(f"[REALTIME] Failed to get current prices: {e}")
            return {}
    
    async def load_historical_data_to_candles(self, symbols: Optional[List[str]] = None, days_back: int = 30):
        """Load historical 1-minute candle data into candles table for initial training (NEW)"""
        try:
            if symbols is None:
                symbols = SUPPORTED_PAIRS
            
            logger.info(f"[HISTORICAL] Loading {days_back} days of historical data for {len(symbols)} symbols...")
            
            from src.data_access.candle_data_manager import candle_data_manager
            
            for symbol in symbols:
                try:
                    logger.info(f"[HISTORICAL] Loading data for {symbol}...")
                    
                    if self.client:
                        # Use authenticated client
                        klines = self.client.get_historical_klines(
                            symbol=symbol,
                            interval=Client.KLINE_INTERVAL_1MINUTE,
                            start_str=f"{days_back} days ago UTC"
                        )
                    else:
                        # Use public API
                        url = f"https://api.binance.com/api/v3/klines"
                        params = {
                            'symbol': symbol,
                            'interval': '1m',
                            'limit': 1000  # Max limit for public API
                        }
                        
                        async with aiohttp.ClientSession() as session:
                            async with session.get(url, params=params) as response:
                                if response.status == 200:
                                    klines = await response.json()
                                else:
                                    logger.error(f"[HISTORICAL] Failed to get data for {symbol}: HTTP {response.status}")
                                    continue
                    
                    if not klines:
                        logger.warning(f"[HISTORICAL] No historical data found for {symbol}")
                        continue
                    
                    # Process klines and update candles table
                    candle_batch = []
                    for kline in klines:
                        candle_data = {
                            'symbol': symbol,
                            'timestamp': int(kline[0]),  # Open time
                            'open': float(kline[1]),
                            'high': float(kline[2]),
                            'low': float(kline[3]),
                            'close': float(kline[4]),
                            'volume': float(kline[5])
                        }
                        candle_batch.append(candle_data)
                    
                    # Batch update candles table
                    await candle_data_manager.batch_update_realtime_candles(candle_batch)
                    
                    logger.success(f"[HISTORICAL] Loaded {len(candle_batch)} candles for {symbol}")
                    
                    # Small delay to avoid rate limits
                    await asyncio.sleep(0.5)
                    
                except Exception as e:
                    logger.error(f"[HISTORICAL] Failed to load historical data for {symbol}: {e}")
                    continue
            
            logger.success(f"[HISTORICAL] Historical data loading completed for {len(symbols)} symbols")
            
        except Exception as e:
            logger.error(f"[HISTORICAL] Failed to load historical data: {e}")
    
    async def _emit_progress_event(self):
        """Emit WebSocket progress event"""
        try:
            # Try to import and emit via Flask app context if available
            from flask import current_app
            if hasattr(current_app, 'extensions') and 'socketio' in current_app.extensions:
                socketio = current_app.extensions['socketio']
                progress_data = {
                    'active': self.bootstrap_active,
                    'progress': self.bootstrap_progress,
                    'elapsed_sec': int((datetime.now() - self.bootstrap_start_time).total_seconds()) if self.bootstrap_start_time else 0,
                    'remaining_sec': max(0, self.bootstrap_duration - int((datetime.now() - self.bootstrap_start_time).total_seconds())) if self.bootstrap_start_time else 0,
                    'total_records': self.bootstrap_total_records,
                    'per_symbol': dict(self.bootstrap_records_collected)
                }
                socketio.emit('collection_progress', progress_data)
        except Exception as e:
            logger.debug(f"[COLLECT] Failed to emit progress event: {e}")
    
    async def _emit_collection_warning(self, message: str):
        """Emit WebSocket warning event"""
        try:
            from flask import current_app
            if hasattr(current_app, 'extensions') and 'socketio' in current_app.extensions:
                socketio = current_app.extensions['socketio']
                socketio.emit('collection_warning', {'message': message})
        except Exception as e:
            logger.debug(f"[COLLECT] Failed to emit warning event: {e}")
    
    def get_bootstrap_status(self) -> Dict:
        """Get bootstrap collection status"""
        if not self.bootstrap_active:
            return {
                'enabled': False,
                'duration_sec': self.bootstrap_duration,
                'elapsed_sec': 0,
                'remaining_sec': 0,
                'percent': 100.0 if hasattr(self, 'bootstrap_start_time') and self.bootstrap_start_time else 0.0,
                'records_total': self.bootstrap_total_records,
                'records_per_symbol': dict(self.bootstrap_records_collected)
            }
        
        elapsed = (datetime.now() - self.bootstrap_start_time).total_seconds() if self.bootstrap_start_time else 0
        remaining = max(0, self.bootstrap_duration - elapsed)
        
        return {
            'enabled': True,
            'duration_sec': self.bootstrap_duration,
            'elapsed_sec': int(elapsed),
            'remaining_sec': int(remaining),
            'percent': self.bootstrap_progress,
            'records_total': self.bootstrap_total_records,
            'records_per_symbol': dict(self.bootstrap_records_collected)
        }
    
    # ==================== END BOOTSTRAP COLLECTION SYSTEM ====================
    
    async def stop(self):
        """Stop data collector"""
        self.running = False
        if self.ccxt_exchange:
            await self.ccxt_exchange.close()
        logger.info("Data collector stopped")