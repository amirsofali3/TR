"""
Advanced Risk Management System with Step-wise TP/SL
"""

import asyncio
import time
from typing import Dict, List, Optional, Tuple, Any
from loguru import logger
from dataclasses import dataclass, field
from datetime import datetime, timedelta
import sqlite3
import json

from config.settings import *

@dataclass
class Position:
    """Trading position with step-wise TP/SL"""
    symbol: str
    side: str  # 'BUY' or 'SELL'
    entry_price: float
    quantity: float
    entry_time: datetime
    
    # Take Profit levels
    tp_levels: List[float] = field(default_factory=list)
    tp_quantities: List[float] = field(default_factory=list)
    current_tp_level: int = 0
    
    # Stop Loss
    initial_sl: float = 0.0
    current_sl: float = 0.0
    
    # Position tracking
    remaining_quantity: float = 0.0
    realized_pnl: float = 0.0
    unrealized_pnl: float = 0.0
    is_active: bool = True
    
    # Metadata
    confidence: float = 0.0
    position_id: str = ""

@dataclass
class TPSLConfig:
    """Configuration for TP/SL levels for each symbol"""
    symbol: str
    tp_levels: List[float] = field(default_factory=lambda: DEFAULT_TP_LEVELS)
    sl_percentage: float = DEFAULT_SL_PERCENTAGE
    trailing_step: float = TP_SL_TRAILING_STEP
    max_positions: int = 1

class RiskManagementSystem:
    """Advanced risk management with step-wise TP/SL"""
    
    def __init__(self):
        self.positions: Dict[str, Position] = {}
        self.tp_sl_configs: Dict[str, TPSLConfig] = {}
        self.portfolio_balance = DEMO_BALANCE
        self.total_pnl = 0.0
        self.daily_pnl = {}
        self.running = False
        self.db_path = DATABASE_URL.replace("sqlite:///", "")
        
        # Initialize default configs for all supported pairs
        for symbol in SUPPORTED_PAIRS:
            self.tp_sl_configs[symbol] = TPSLConfig(symbol=symbol)
    
    async def initialize(self):
        """Initialize risk management system"""
        try:
            logger.info("Initializing risk management system...")
            
            # Initialize database
            await self.init_database()
            
            # Load existing positions
            await self.load_positions()
            
            # Load TP/SL configurations
            await self.load_tp_sl_configs()
            
            logger.success("Risk management system initialized")
            
        except Exception as e:
            logger.error(f"Failed to initialize risk management: {e}")
            raise
    
    async def init_database(self):
        """Initialize database tables for positions and P&L tracking"""
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            # Positions table
            cursor.execute('''
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
            ''')
            
            # P&L tracking table
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS pnl_history (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    date TEXT NOT NULL,
                    symbol TEXT,
                    realized_pnl REAL DEFAULT 0,
                    unrealized_pnl REAL DEFAULT 0,
                    total_pnl REAL DEFAULT 0,
                    portfolio_balance REAL DEFAULT 0,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    UNIQUE(date, symbol)
                )
            ''')
            
            # TP/SL configurations table
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS tp_sl_configs (
                    symbol TEXT PRIMARY KEY,
                    tp_levels TEXT,
                    sl_percentage REAL,
                    trailing_step REAL,
                    max_positions INTEGER,
                    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                )
            ''')
            
            # Trade history table
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS trade_history (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    position_id TEXT,
                    symbol TEXT NOT NULL,
                    side TEXT NOT NULL,
                    action TEXT NOT NULL,  -- 'OPEN', 'TP', 'SL', 'CLOSE'
                    price REAL NOT NULL,
                    quantity REAL NOT NULL,
                    pnl REAL DEFAULT 0,
                    timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                )
            ''')
            
            conn.commit()
            conn.close()
            
            logger.info("Risk management database initialized")
            
        except Exception as e:
            logger.error(f"Failed to initialize database: {e}")
            raise
    
    async def open_position(self, symbol: str, side: str, price: float, confidence: float, 
                          quantity: Optional[float] = None) -> Optional[str]:
        """Open a new position with step-wise TP/SL setup"""
        try:
            config = self.tp_sl_configs.get(symbol)
            if not config:
                logger.error(f"No TP/SL config found for {symbol}")
                return None
            
            # Check if we already have max positions for this symbol
            active_positions = [p for p in self.positions.values() 
                             if p.symbol == symbol and p.is_active]
            
            if len(active_positions) >= config.max_positions:
                logger.warning(f"Max positions reached for {symbol}")
                return None
            
            # Calculate position size if not provided
            if quantity is None:
                quantity = await self.calculate_position_size(symbol, price, confidence)
            
            # Generate position ID
            position_id = f"{symbol}_{side}_{int(time.time())}"
            
            # Setup TP levels
            tp_levels = []
            tp_quantities = []
            
            if side == 'BUY':
                for tp_pct in config.tp_levels:
                    tp_price = price * (1 + tp_pct)
                    tp_levels.append(tp_price)
                    tp_quantities.append(quantity / len(config.tp_levels))
                
                # Setup stop loss
                initial_sl = price * (1 - config.sl_percentage)
                
            else:  # SELL
                for tp_pct in config.tp_levels:
                    tp_price = price * (1 - tp_pct)
                    tp_levels.append(tp_price)
                    tp_quantities.append(quantity / len(config.tp_levels))
                
                # Setup stop loss
                initial_sl = price * (1 + config.sl_percentage)
            
            # Create position
            position = Position(
                symbol=symbol,
                side=side,
                entry_price=price,
                quantity=quantity,
                entry_time=datetime.now(),
                tp_levels=tp_levels,
                tp_quantities=tp_quantities,
                current_tp_level=0,
                initial_sl=initial_sl,
                current_sl=initial_sl,
                remaining_quantity=quantity,
                confidence=confidence,
                position_id=position_id
            )
            
            # Store position
            self.positions[position_id] = position
            await self.save_position(position)
            
            # Record trade
            await self.record_trade(position_id, symbol, side, 'OPEN', price, quantity, 0)
            
            logger.info(f"Opened {side} position for {symbol} at {price:.4f} (ID: {position_id})")
            
            return position_id
            
        except Exception as e:
            logger.error(f"Failed to open position: {e}")
            return None
    
    async def calculate_position_size(self, symbol: str, price: float, confidence: float) -> float:
        """Calculate position size based on risk management rules"""
        try:
            # Base position size as percentage of portfolio
            base_risk_pct = 0.02  # 2% of portfolio per trade
            
            # Adjust risk based on confidence
            confidence_multiplier = min(confidence / 100.0, 1.0)  # Scale confidence to 0-1
            adjusted_risk_pct = base_risk_pct * confidence_multiplier
            
            # Calculate position size in USDT
            risk_amount = self.portfolio_balance * adjusted_risk_pct
            
            # Calculate quantity based on price
            quantity = risk_amount / price
            
            return round(quantity, 6)
            
        except Exception as e:
            logger.error(f"Failed to calculate position size: {e}")
            return 0.0
    
    async def update_positions(self, current_prices: Dict[str, float]):
        """Update all active positions with current prices"""
        try:
            for position_id, position in self.positions.items():
                if not position.is_active:
                    continue
                
                current_price = current_prices.get(position.symbol)
                if current_price is None:
                    continue
                
                # Update unrealized P&L
                await self.update_unrealized_pnl(position, current_price)
                
                # Check TP levels
                await self.check_take_profit(position, current_price)
                
                # Check and update trailing stop loss
                await self.update_trailing_stop_loss(position, current_price)
                
                # Check stop loss
                await self.check_stop_loss(position, current_price)
                
        except Exception as e:
            logger.error(f"Failed to update positions: {e}")
    
    async def update_unrealized_pnl(self, position: Position, current_price: float):
        """Update unrealized P&L for a position"""
        try:
            if position.side == 'BUY':
                pnl = (current_price - position.entry_price) * position.remaining_quantity
            else:  # SELL
                pnl = (position.entry_price - current_price) * position.remaining_quantity
            
            position.unrealized_pnl = pnl
            
        except Exception as e:
            logger.error(f"Failed to update unrealized P&L: {e}")
    
    async def check_take_profit(self, position: Position, current_price: float):
        """Check and execute take profit levels"""
        try:
            if position.current_tp_level >= len(position.tp_levels):
                return  # All TP levels already hit
            
            current_tp_price = position.tp_levels[position.current_tp_level]
            tp_quantity = position.tp_quantities[position.current_tp_level]
            
            # Check if TP level is hit
            tp_hit = False
            if position.side == 'BUY' and current_price >= current_tp_price:
                tp_hit = True
            elif position.side == 'SELL' and current_price <= current_tp_price:
                tp_hit = True
            
            if tp_hit:
                # Execute partial close at TP level
                await self.execute_partial_close(position, current_tp_price, tp_quantity, 'TP')
                
                # Move to next TP level
                position.current_tp_level += 1
                
                # Update trailing stop loss to previous TP level
                if position.current_tp_level > 1:
                    prev_tp_price = position.tp_levels[position.current_tp_level - 2]
                    position.current_sl = prev_tp_price
                    
                    logger.info(f"Updated trailing SL for {position.symbol} to {prev_tp_price:.4f}")
                
                await self.save_position(position)
                
        except Exception as e:
            logger.error(f"Failed to check take profit: {e}")
    
    async def update_trailing_stop_loss(self, position: Position, current_price: float):
        """Update trailing stop loss based on price movement"""
        try:
            config = self.tp_sl_configs.get(position.symbol)
            if not config:
                return
            
            # Only trail if we haven't hit any TP levels yet
            if position.current_tp_level > 0:
                return
            
            if position.side == 'BUY':
                # Trail stop loss upward
                new_sl = current_price * (1 - config.sl_percentage)
                if new_sl > position.current_sl:
                    position.current_sl = new_sl
                    await self.save_position(position)
                    logger.debug(f"Trailed SL up for {position.symbol} to {new_sl:.4f}")
                    
            else:  # SELL
                # Trail stop loss downward
                new_sl = current_price * (1 + config.sl_percentage)
                if new_sl < position.current_sl:
                    position.current_sl = new_sl
                    await self.save_position(position)
                    logger.debug(f"Trailed SL down for {position.symbol} to {new_sl:.4f}")
            
        except Exception as e:
            logger.error(f"Failed to update trailing stop loss: {e}")
    
    async def check_stop_loss(self, position: Position, current_price: float):
        """Check and execute stop loss"""
        try:
            sl_hit = False
            
            if position.side == 'BUY' and current_price <= position.current_sl:
                sl_hit = True
            elif position.side == 'SELL' and current_price >= position.current_sl:
                sl_hit = True
            
            if sl_hit:
                # Close entire remaining position at SL
                await self.execute_partial_close(
                    position, position.current_sl, position.remaining_quantity, 'SL'
                )
                
                position.is_active = False
                await self.save_position(position)
                
                logger.warning(f"Stop loss hit for {position.symbol} at {position.current_sl:.4f}")
                
        except Exception as e:
            logger.error(f"Failed to check stop loss: {e}")
    
    async def execute_partial_close(self, position: Position, price: float, quantity: float, reason: str):
        """Execute partial position close"""
        try:
            # Calculate P&L for this partial close
            if position.side == 'BUY':
                pnl = (price - position.entry_price) * quantity
            else:  # SELL
                pnl = (position.entry_price - price) * quantity
            
            # Update position
            position.remaining_quantity -= quantity
            position.realized_pnl += pnl
            
            # Update portfolio
            self.portfolio_balance += (quantity * price) + pnl  # In demo mode
            self.total_pnl += pnl
            
            # Record trade
            await self.record_trade(
                position.position_id, position.symbol, position.side, 
                reason, price, quantity, pnl
            )
            
            # Check if position is fully closed
            if position.remaining_quantity <= 0.001:  # Close to zero
                position.is_active = False
                logger.info(f"Position {position.position_id} fully closed. Total P&L: {position.realized_pnl:.2f}")
            else:
                logger.info(f"Partial close {reason} for {position.symbol}: {quantity:.4f} at {price:.4f}, P&L: {pnl:.2f}")
            
            await self.save_position(position)
            
        except Exception as e:
            logger.error(f"Failed to execute partial close: {e}")
    
    async def close_position(self, position_id: str, price: float, reason: str = 'MANUAL'):
        """Manually close a position"""
        try:
            position = self.positions.get(position_id)
            if not position or not position.is_active:
                return False
            
            await self.execute_partial_close(
                position, price, position.remaining_quantity, reason
            )
            
            position.is_active = False
            await self.save_position(position)
            
            return True
            
        except Exception as e:
            logger.error(f"Failed to close position: {e}")
            return False
    
    async def save_position(self, position: Position):
        """Save position to database"""
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            cursor.execute('''
                INSERT OR REPLACE INTO positions 
                (position_id, symbol, side, entry_price, quantity, remaining_quantity,
                 entry_time, tp_levels, tp_quantities, current_tp_level, initial_sl, 
                 current_sl, realized_pnl, unrealized_pnl, confidence, is_active)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            ''', (
                position.position_id, position.symbol, position.side, position.entry_price,
                position.quantity, position.remaining_quantity, position.entry_time,
                json.dumps(position.tp_levels), json.dumps(position.tp_quantities),
                position.current_tp_level, position.initial_sl, position.current_sl,
                position.realized_pnl, position.unrealized_pnl, position.confidence, position.is_active
            ))
            
            conn.commit()
            conn.close()
            
        except Exception as e:
            logger.error(f"Failed to save position: {e}")
    
    async def load_positions(self):
        """Load positions from database"""
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            cursor.execute('SELECT * FROM positions WHERE is_active = TRUE')
            rows = cursor.fetchall()
            
            for row in rows:
                position = Position(
                    position_id=row[0],
                    symbol=row[1],
                    side=row[2],
                    entry_price=row[3],
                    quantity=row[4],
                    remaining_quantity=row[5],
                    entry_time=datetime.fromisoformat(row[6]),
                    tp_levels=json.loads(row[7]) if row[7] else [],
                    tp_quantities=json.loads(row[8]) if row[8] else [],
                    current_tp_level=row[9],
                    initial_sl=row[10],
                    current_sl=row[11],
                    realized_pnl=row[12],
                    unrealized_pnl=row[13],
                    confidence=row[14],
                    is_active=bool(row[15])
                )
                
                self.positions[position.position_id] = position
            
            conn.close()
            logger.info(f"Loaded {len(self.positions)} active positions")
            
        except Exception as e:
            logger.error(f"Failed to load positions: {e}")
    
    async def record_trade(self, position_id: str, symbol: str, side: str, action: str, 
                          price: float, quantity: float, pnl: float):
        """Record trade in database"""
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            cursor.execute('''
                INSERT INTO trade_history 
                (position_id, symbol, side, action, price, quantity, pnl)
                VALUES (?, ?, ?, ?, ?, ?, ?)
            ''', (position_id, symbol, side, action, price, quantity, pnl))
            
            conn.commit()
            conn.close()
            
        except Exception as e:
            logger.error(f"Failed to record trade: {e}")
    
    async def update_daily_pnl(self):
        """Update daily P&L tracking"""
        try:
            today = datetime.now().date().isoformat()
            
            # Calculate total unrealized P&L
            total_unrealized = sum(p.unrealized_pnl for p in self.positions.values() if p.is_active)
            
            # Calculate daily realized P&L
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            cursor.execute('''
                SELECT SUM(pnl) FROM trade_history 
                WHERE DATE(timestamp) = ?
            ''', (today,))
            
            result = cursor.fetchone()
            daily_realized = result[0] if result[0] else 0.0
            
            # Store daily P&L
            cursor.execute('''
                INSERT OR REPLACE INTO pnl_history 
                (date, realized_pnl, unrealized_pnl, total_pnl, portfolio_balance)
                VALUES (?, ?, ?, ?, ?)
            ''', (today, daily_realized, total_unrealized, 
                  daily_realized + total_unrealized, self.portfolio_balance))
            
            conn.commit()
            conn.close()
            
            self.daily_pnl[today] = {
                'realized': daily_realized,
                'unrealized': total_unrealized,
                'total': daily_realized + total_unrealized
            }
            
        except Exception as e:
            logger.error(f"Failed to update daily P&L: {e}")
    
    def get_tp_sl_config(self, symbol: str) -> TPSLConfig:
        """Get TP/SL configuration for a symbol"""
        return self.tp_sl_configs.get(symbol, TPSLConfig(symbol=symbol))
    
    def update_tp_sl_config(self, symbol: str, config: TPSLConfig):
        """Update TP/SL configuration for a symbol"""
        self.tp_sl_configs[symbol] = config
        # Save to database
        asyncio.create_task(self.save_tp_sl_config(config))
    
    async def save_tp_sl_config(self, config: TPSLConfig):
        """Save TP/SL configuration to database"""
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            cursor.execute('''
                INSERT OR REPLACE INTO tp_sl_configs 
                (symbol, tp_levels, sl_percentage, trailing_step, max_positions)
                VALUES (?, ?, ?, ?, ?)
            ''', (
                config.symbol, json.dumps(config.tp_levels), 
                config.sl_percentage, config.trailing_step, config.max_positions
            ))
            
            conn.commit()
            conn.close()
            
        except Exception as e:
            logger.error(f"Failed to save TP/SL config: {e}")
    
    async def load_tp_sl_configs(self):
        """Load TP/SL configurations from database"""
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            cursor.execute('SELECT * FROM tp_sl_configs')
            rows = cursor.fetchall()
            
            for row in rows:
                config = TPSLConfig(
                    symbol=row[0],
                    tp_levels=json.loads(row[1]) if row[1] else DEFAULT_TP_LEVELS,
                    sl_percentage=row[2],
                    trailing_step=row[3],
                    max_positions=row[4]
                )
                
                self.tp_sl_configs[config.symbol] = config
            
            conn.close()
            
        except Exception as e:
            logger.error(f"Failed to load TP/SL configs: {e}")
    
    def get_portfolio_summary(self) -> Dict[str, Any]:
        """Get portfolio summary for web interface"""
        try:
            active_positions = [p for p in self.positions.values() if p.is_active]
            
            total_unrealized = sum(p.unrealized_pnl for p in active_positions)
            total_realized = sum(p.realized_pnl for p in self.positions.values())
            
            return {
                'portfolio_balance': self.portfolio_balance,
                'active_positions': len(active_positions),
                'total_positions': len(self.positions),
                'total_realized_pnl': total_realized,
                'total_unrealized_pnl': total_unrealized,
                'total_pnl': total_realized + total_unrealized,
                'daily_pnl': self.daily_pnl,
                'positions': [
                    {
                        'position_id': p.position_id,
                        'symbol': p.symbol,
                        'side': p.side,
                        'entry_price': p.entry_price,
                        'quantity': p.quantity,
                        'remaining_quantity': p.remaining_quantity,
                        'current_tp_level': p.current_tp_level,
                        'total_tp_levels': len(p.tp_levels),
                        'current_sl': p.current_sl,
                        'realized_pnl': p.realized_pnl,
                        'unrealized_pnl': p.unrealized_pnl,
                        'confidence': p.confidence
                    }
                    for p in active_positions
                ]
            }
            
        except Exception as e:
            logger.error(f"Failed to get portfolio summary: {e}")
            return {}
    
    async def start_monitoring(self):
        """Start monitoring positions"""
        self.running = True
        logger.info("Starting position monitoring...")
        
        while self.running:
            try:
                # Update daily P&L
                await self.update_daily_pnl()
                
                # Wait for next update
                await asyncio.sleep(10)  # Check every 10 seconds
                
            except Exception as e:
                logger.error(f"Error in position monitoring: {e}")
                await asyncio.sleep(5)
    
    def stop(self):
        """Stop position monitoring"""
        self.running = False
        logger.info("Position monitoring stopped")