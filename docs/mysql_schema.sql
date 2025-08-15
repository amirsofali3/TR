-- MySQL Schema for Crypto Trading System
-- Compatible with existing SQLite schema

CREATE DATABASE IF NOT EXISTS trading_system CHARACTER SET utf8mb4 COLLATE utf8mb4_unicode_ci;
USE trading_system;

-- Market data table
CREATE TABLE IF NOT EXISTS market_data (
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
) ENGINE=InnoDB;

-- Real-time prices table
CREATE TABLE IF NOT EXISTS real_time_prices (
    symbol VARCHAR(20) PRIMARY KEY,
    price DECIMAL(20,8) NOT NULL,
    bid_price DECIMAL(20,8),
    ask_price DECIMAL(20,8),
    bid_size DECIMAL(20,8),
    ask_size DECIMAL(20,8),
    timestamp BIGINT NOT NULL,
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP ON UPDATE CURRENT_TIMESTAMP
) ENGINE=InnoDB;

-- Positions table (risk management)
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
) ENGINE=InnoDB;

-- P&L history table
CREATE TABLE IF NOT EXISTS pnl_history (
    id INT AUTO_INCREMENT PRIMARY KEY,
    date VARCHAR(20) NOT NULL,
    symbol VARCHAR(20),
    realized_pnl DECIMAL(20,8) DEFAULT 0,
    unrealized_pnl DECIMAL(20,8) DEFAULT 0,
    total_pnl DECIMAL(20,8) DEFAULT 0,
    portfolio_balance DECIMAL(20,8) DEFAULT 0,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    UNIQUE KEY unique_pnl_date (date, symbol)
) ENGINE=InnoDB;

-- TP/SL configurations table
CREATE TABLE IF NOT EXISTS tp_sl_configs (
    symbol VARCHAR(20) PRIMARY KEY,
    tp_levels TEXT,
    sl_percentage DECIMAL(5,4),
    trailing_step DECIMAL(5,4),
    max_positions INT,
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP ON UPDATE CURRENT_TIMESTAMP
) ENGINE=InnoDB;

-- Trade history table
CREATE TABLE IF NOT EXISTS trade_history (
    id INT AUTO_INCREMENT PRIMARY KEY,
    position_id VARCHAR(100),
    symbol VARCHAR(20) NOT NULL,
    side VARCHAR(10) NOT NULL,
    action VARCHAR(10) NOT NULL,  -- 'OPEN', 'TP', 'SL', 'CLOSE'
    price DECIMAL(20,8) NOT NULL,
    quantity DECIMAL(20,8) NOT NULL,
    pnl DECIMAL(20,8) DEFAULT 0,
    timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP
) ENGINE=InnoDB;

-- Indexes for better performance
CREATE INDEX idx_market_data_symbol_time ON market_data(symbol, timeframe, timestamp);
CREATE INDEX idx_prices_symbol ON real_time_prices(symbol);
CREATE INDEX idx_positions_active ON positions(is_active, symbol);
CREATE INDEX idx_pnl_date ON pnl_history(date);
CREATE INDEX idx_trade_history_symbol ON trade_history(symbol, timestamp);

-- Optional: Table for consolidated trading data (if user has 'tr' table ready)
-- Uncomment below if MERGE_INTO_SINGLE_TABLE environment variable is set
/*
CREATE TABLE IF NOT EXISTS tr (
    id INT AUTO_INCREMENT PRIMARY KEY,
    symbol VARCHAR(20) NOT NULL,
    timestamp BIGINT NOT NULL,
    open DECIMAL(20,8) NOT NULL,
    high DECIMAL(20,8) NOT NULL,
    low DECIMAL(20,8) NOT NULL,
    close DECIMAL(20,8) NOT NULL,
    volume DECIMAL(20,8) NOT NULL,
    -- Add any additional columns that the existing 'tr' table might have
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    UNIQUE KEY unique_tr_entry (symbol, timestamp)
) ENGINE=InnoDB;
*/