"""
Binance Data Collector for fetching cryptocurrency market data
"""

import asyncio
import aiohttp
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple
from loguru import logger
import time
import sqlite3
from binance.client import Client
from binance.exceptions import BinanceAPIException
import ccxt.async_support as ccxt

from config.settings import *

class BinanceDataCollector:
    """Handles data collection from Binance API"""
    
    def __init__(self):
        self.client = None
        self.ws_client = None
        self.ccxt_exchange = None
        self.db_path = DATABASE_URL.replace("sqlite:///", "")
        self.price_cache = {}
        self.data_cache = {}
        self.running = False
        
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
                
                # Initialize CCXT for additional functionality
                self.ccxt_exchange = ccxt.binance({
                    'apiKey': BINANCE_API_KEY,
                    'secret': BINANCE_SECRET_KEY,
                    'sandbox': DEMO_MODE,
                    'enableRateLimit': True,
                })
                
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
            
            # Fetch initial historical data
            logger.info("📊 Fetching initial market data...")
            await self.fetch_initial_data()
            
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
        """Initialize SQLite database for storing market data"""
        try:
            # Create data directory if it doesn't exist
            import os
            os.makedirs(os.path.dirname(self.db_path), exist_ok=True)
            
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            # Create tables for market data
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS market_data (
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
            ''')
            
            # Create table for real-time prices
            cursor.execute('''
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
            ''')
            
            # Create indexes for better performance
            cursor.execute('CREATE INDEX IF NOT EXISTS idx_market_data_symbol_time ON market_data(symbol, timeframe, timestamp)')
            cursor.execute('CREATE INDEX IF NOT EXISTS idx_prices_symbol ON real_time_prices(symbol)')
            
            conn.commit()
            conn.close()
            
            logger.info("Database initialized successfully")
            
        except Exception as e:
            logger.error(f"Failed to initialize database: {e}")
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
        """Store market data in database"""
        try:
            conn = sqlite3.connect(self.db_path)
            
            # Prepare data for insertion
            data_to_insert = data[['timestamp', 'open', 'high', 'low', 'close', 'volume']].copy()
            data_to_insert['symbol'] = symbol
            data_to_insert['timeframe'] = timeframe
            
            # Use INSERT OR REPLACE to handle duplicates
            cursor = conn.cursor()
            for _, row in data_to_insert.iterrows():
                cursor.execute('''
                    INSERT OR REPLACE INTO market_data 
                    (symbol, timeframe, timestamp, open, high, low, close, volume)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                ''', (
                    row['symbol'], row['timeframe'], row['timestamp'],
                    row['open'], row['high'], row['low'], row['close'], row['volume']
                ))
            
            conn.commit()
            conn.close()
            
        except Exception as e:
            logger.error(f"Failed to store market data for {symbol}: {e}")
    
    async def get_historical_data(self, symbol: str, timeframe: str, limit: int = 500) -> Optional[pd.DataFrame]:
        """Get historical data from database"""
        try:
            conn = sqlite3.connect(self.db_path)
            
            query = '''
                SELECT timestamp, open, high, low, close, volume 
                FROM market_data 
                WHERE symbol = ? AND timeframe = ? 
                ORDER BY timestamp DESC 
                LIMIT ?
            '''
            
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
        """Store real-time price in database"""
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            cursor.execute('''
                INSERT OR REPLACE INTO real_time_prices 
                (symbol, price, bid_price, ask_price, bid_size, ask_size, timestamp) 
                VALUES (?, ?, ?, ?, ?, ?, ?)
            ''', (
                price_data['symbol'],
                price_data['price'],
                price_data.get('bid_price'),
                price_data.get('ask_price'),
                price_data.get('bid_size'),
                price_data.get('ask_size'),
                int(price_data['timestamp'] * 1000)
            ))
            
            conn.commit()
            conn.close()
            
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
    
    async def stop(self):
        """Stop data collector"""
        self.running = False
        if self.ccxt_exchange:
            await self.ccxt_exchange.close()
        logger.info("Data collector stopped")