"""
Synthetic data generator for testing crypto trading system
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Optional
from datetime import datetime, timedelta

class SyntheticDataGenerator:
    """Generate synthetic market data and indicators for testing"""
    
    def __init__(self, seed: int = 42):
        self.seed = seed
        np.random.seed(seed)
    
    def generate_ohlcv_data(
        self,
        num_samples: int = 500,
        base_price: float = 50000.0,
        volatility: float = 0.02,
        trend: float = 0.0001,  # Daily trend as decimal
        start_time: Optional[int] = None,
        interval_ms: int = 14400000  # 4 hours in milliseconds
    ) -> pd.DataFrame:
        """
        Generate synthetic OHLCV data
        
        Args:
            num_samples: Number of candles to generate
            base_price: Starting price
            volatility: Price volatility (standard deviation of returns)
            trend: Daily trend (positive for upward, negative for downward)
            start_time: Starting timestamp in milliseconds
            interval_ms: Time between candles in milliseconds
        
        Returns:
            DataFrame with OHLCV data
        """
        if start_time is None:
            start_time = int(datetime(2023, 1, 1).timestamp() * 1000)
        
        # Generate price series using geometric Brownian motion
        timestamps = []
        opens = []
        highs = []
        lows = []
        closes = []
        volumes = []
        
        current_price = base_price
        
        for i in range(num_samples):
            # Calculate timestamp
            timestamp = start_time + i * interval_ms
            timestamps.append(timestamp)
            
            # Generate price movement
            drift = trend
            shock = np.random.normal(0, volatility)
            price_change = drift + shock
            
            # OHLC generation
            open_price = current_price
            close_price = current_price * (1 + price_change)
            
            # High and low based on intraday volatility
            intraday_volatility = volatility * 0.3
            high_offset = abs(np.random.normal(0, intraday_volatility))
            low_offset = abs(np.random.normal(0, intraday_volatility))
            
            high_price = max(open_price, close_price) * (1 + high_offset)
            low_price = min(open_price, close_price) * (1 - low_offset)
            
            # Volume (correlated with volatility)
            volume_base = 500
            volume_volatility = abs(price_change) * 1000
            volume = max(100, np.random.normal(volume_base + volume_volatility, 200))
            
            opens.append(open_price)
            highs.append(high_price)
            lows.append(low_price)
            closes.append(close_price)
            volumes.append(volume)
            
            current_price = close_price
        
        return pd.DataFrame({
            'timestamp': timestamps,
            'open': opens,
            'high': highs,
            'low': lows,
            'close': closes,
            'volume': volumes
        })
    
    def generate_trending_data(
        self,
        num_samples: int = 500,
        base_price: float = 50000.0,
        trend_strength: float = 0.001  # Positive for up, negative for down
    ) -> pd.DataFrame:
        """Generate data with a clear trend"""
        return self.generate_ohlcv_data(
            num_samples=num_samples,
            base_price=base_price,
            volatility=0.015,
            trend=trend_strength
        )
    
    def generate_sideways_data(
        self,
        num_samples: int = 500,
        base_price: float = 50000.0,
        volatility: float = 0.01
    ) -> pd.DataFrame:
        """Generate sideways/range-bound data"""
        return self.generate_ohlcv_data(
            num_samples=num_samples,
            base_price=base_price,
            volatility=volatility,
            trend=0.0  # No trend
        )
    
    def generate_indicators(
        self,
        ohlcv_data: pd.DataFrame,
        include_advanced: bool = True
    ) -> Dict[str, pd.Series]:
        """
        Generate technical indicators from OHLCV data
        
        Args:
            ohlcv_data: DataFrame with OHLCV data
            include_advanced: Whether to include advanced indicators
            
        Returns:
            Dictionary of indicator name -> Series
        """
        indicators = {}
        
        close_prices = ohlcv_data['close']
        high_prices = ohlcv_data['high']
        low_prices = ohlcv_data['low']
        volumes = ohlcv_data['volume']
        
        # Simple Moving Averages
        for period in [5, 10, 14, 20, 50, 100, 200]:
            if len(close_prices) >= period:
                indicators[f'SMA_{period}'] = close_prices.rolling(window=period).mean()
        
        # Exponential Moving Averages
        for period in [5, 10, 12, 14, 20, 26, 50]:
            if len(close_prices) >= period:
                indicators[f'EMA_{period}'] = close_prices.ewm(span=period).mean()
        
        # RSI
        if len(close_prices) >= 14:
            delta = close_prices.diff()
            gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
            loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
            rs = gain / loss
            indicators['RSI'] = 100 - (100 / (1 + rs))
        
        # MACD
        if len(close_prices) >= 26:
            ema_12 = close_prices.ewm(span=12).mean()
            ema_26 = close_prices.ewm(span=26).mean()
            indicators['MACD'] = ema_12 - ema_26
            indicators['MACD_Signal'] = indicators['MACD'].ewm(span=9).mean()
        
        # Bollinger Bands
        for period in [14, 20]:
            if len(close_prices) >= period:
                sma = close_prices.rolling(window=period).mean()
                std = close_prices.rolling(window=period).std()
                indicators[f'BB_Upper_{period}'] = sma + (2 * std)
                indicators[f'BB_Lower_{period}'] = sma - (2 * std)
                indicators[f'BB_Middle_{period}'] = sma
        
        # Volume indicators
        for period in [10, 20]:
            if len(volumes) >= period:
                indicators[f'Volume_SMA_{period}'] = volumes.rolling(window=period).mean()
        
        # Price-based indicators
        indicators['High_Low_Pct'] = (high_prices - low_prices) / close_prices * 100
        indicators['Open_Close_Pct'] = (close_prices - ohlcv_data['open']) / ohlcv_data['open'] * 100
        
        if include_advanced:
            # Stochastic Oscillator
            if len(close_prices) >= 14:
                low_14 = low_prices.rolling(window=14).min()
                high_14 = high_prices.rolling(window=14).max()
                indicators['Stoch_K'] = ((close_prices - low_14) / (high_14 - low_14)) * 100
                indicators['Stoch_D'] = indicators['Stoch_K'].rolling(window=3).mean()
            
            # Williams %R
            if len(close_prices) >= 14:
                high_14 = high_prices.rolling(window=14).max()
                low_14 = low_prices.rolling(window=14).min()
                indicators['Williams_R'] = ((high_14 - close_prices) / (high_14 - low_14)) * -100
            
            # Average True Range (ATR)
            if len(close_prices) >= 14:
                high_low = high_prices - low_prices
                high_close = abs(high_prices - close_prices.shift())
                low_close = abs(low_prices - close_prices.shift())
                true_range = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
                indicators['ATR'] = true_range.rolling(window=14).mean()
        
        # Clean up any NaN values by forward-filling
        for name, series in indicators.items():
            indicators[name] = series.fillna(method='ffill').fillna(0)
        
        return indicators
    
    def create_test_dataset(
        self,
        scenario: str = 'mixed',
        num_samples: int = 500,
        base_price: float = 50000.0
    ):
        """
        Create a complete test dataset with OHLCV and indicators
        
        Args:
            scenario: 'uptrend', 'downtrend', 'sideways', 'mixed', or 'volatile'
            num_samples: Number of samples to generate
            base_price: Starting price
            
        Returns:
            Tuple of (ohlcv_data, indicators)
        """
        if scenario == 'uptrend':
            ohlcv_data = self.generate_trending_data(
                num_samples=num_samples,
                base_price=base_price,
                trend_strength=0.0015  # Strong uptrend
            )
        elif scenario == 'downtrend':
            ohlcv_data = self.generate_trending_data(
                num_samples=num_samples,
                base_price=base_price,
                trend_strength=-0.0015  # Strong downtrend
            )
        elif scenario == 'sideways':
            ohlcv_data = self.generate_sideways_data(
                num_samples=num_samples,
                base_price=base_price,
                volatility=0.008  # Low volatility sideways
            )
        elif scenario == 'volatile':
            ohlcv_data = self.generate_ohlcv_data(
                num_samples=num_samples,
                base_price=base_price,
                volatility=0.04,  # High volatility
                trend=0.0
            )
        else:  # mixed
            # Create mixed data with different phases
            phase_size = num_samples // 3
            
            # Phase 1: Uptrend
            phase1 = self.generate_trending_data(
                num_samples=phase_size,
                base_price=base_price,
                trend_strength=0.001
            )
            
            # Phase 2: Sideways (starting from last price of phase 1)
            phase2_start = phase1['close'].iloc[-1]
            phase2 = self.generate_sideways_data(
                num_samples=phase_size,
                base_price=phase2_start,
                volatility=0.01
            )
            # Adjust timestamps
            phase2['timestamp'] = phase2['timestamp'] + phase1['timestamp'].iloc[-1]
            
            # Phase 3: Downtrend (starting from last price of phase 2)
            phase3_start = phase2['close'].iloc[-1]
            remaining_samples = num_samples - 2 * phase_size
            phase3 = self.generate_trending_data(
                num_samples=remaining_samples,
                base_price=phase3_start,
                trend_strength=-0.0008
            )
            # Adjust timestamps
            phase3['timestamp'] = phase3['timestamp'] + phase2['timestamp'].iloc[-1]
            
            # Combine phases
            ohlcv_data = pd.concat([phase1, phase2, phase3], ignore_index=True)
        
        # Generate indicators
        indicators = self.generate_indicators(ohlcv_data, include_advanced=True)
        
        return ohlcv_data, indicators


# Convenience functions for quick testing
def create_uptrend_data(samples: int = 500):
    """Quick function to create uptrending test data"""
    generator = SyntheticDataGenerator(seed=42)
    return generator.create_test_dataset('uptrend', samples)

def create_downtrend_data(samples: int = 500):
    """Quick function to create downtrending test data"""
    generator = SyntheticDataGenerator(seed=123)
    return generator.create_test_dataset('downtrend', samples)

def create_sideways_data(samples: int = 500):
    """Quick function to create sideways test data"""
    generator = SyntheticDataGenerator(seed=456)
    return generator.create_test_dataset('sideways', samples)

def create_mixed_data(samples: int = 500):
    """Quick function to create mixed scenario test data"""
    generator = SyntheticDataGenerator(seed=789)
    return generator.create_test_dataset('mixed', samples)