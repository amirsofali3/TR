"""
Technical Indicator Engine for calculating all trading indicators
"""

import pandas as pd
import numpy as np
import talib
from typing import Dict, List, Optional, Any
from loguru import logger
import os
import csv
from scipy import stats
from sklearn.linear_model import LinearRegression

class IndicatorEngine:
    """Calculates all technical indicators from the CSV file"""
    
    def __init__(self):
        self.indicators_config = {}
        self.required_indicators = set()
        self.rfe_eligible_indicators = set()
        self.indicator_functions = {}
        self.prerequisite_map = {}
        
    async def initialize(self):
        """Initialize the indicator engine"""
        try:
            logger.info("Initializing indicator engine...")
            
            # Load indicators configuration from CSV
            await self.load_indicators_config()
            
            # Setup indicator functions
            self.setup_indicator_functions()
            
            logger.success(f"Indicator engine initialized with {len(self.indicators_config)} indicators")
            
        except Exception as e:
            logger.error(f"Failed to initialize indicator engine: {e}")
            raise
    
    async def load_indicators_config(self):
        """Load indicators configuration from CSV file"""
        try:
            # Use relative path from project root
            import os
            project_root = os.path.dirname(os.path.dirname(os.path.dirname(__file__)))
            csv_path = os.path.join(project_root, "crypto_trading_feature_encyclopedia.csv")
            
            with open(csv_path, 'r', encoding='utf-8') as file:
                reader = csv.DictReader(file)
                
                for row in reader:
                    indicator_name = row['Indicator']
                    
                    self.indicators_config[indicator_name] = {
                        'category': row['Category'],
                        'required_inputs': row['Required Inputs'],
                        'formula': row['Formula / Calculation'],
                        'must_keep': row['Must Keep (Not in RFE)'].lower() == 'yes',
                        'rfe_eligible': row['RFE Eligible'].lower() == 'yes',
                        'prerequisite_for': row['Prerequisite For'],
                        'parameters': row['Parameters'],
                        'outputs': row['Outputs']
                    }
                    
                    # Track required indicators (must keep)
                    if row['Must Keep (Not in RFE)'].lower() == 'yes':
                        self.required_indicators.add(indicator_name)
                    
                    # Track RFE eligible indicators
                    if row['RFE Eligible'].lower() == 'yes':
                        self.rfe_eligible_indicators.add(indicator_name)
                    
                    # Track prerequisites
                    if row['Prerequisite For']:
                        for prereq in row['Prerequisite For'].split(','):
                            prereq = prereq.strip()
                            if prereq not in self.prerequisite_map:
                                self.prerequisite_map[prereq] = []
                            self.prerequisite_map[prereq].append(indicator_name)
            
            logger.info(f"Loaded {len(self.indicators_config)} indicators")
            logger.info(f"Required indicators: {len(self.required_indicators)}")
            logger.info(f"RFE eligible indicators: {len(self.rfe_eligible_indicators)}")
            
        except FileNotFoundError:
            logger.error(f"Indicators CSV file not found at {csv_path}")
            logger.info("Falling back to default indicators configuration")
            # Set up minimal default indicators
            self._setup_default_indicators()
        except Exception as e:
            logger.error(f"Failed to load indicators config: {e}")
            logger.info("Falling back to default indicators configuration")
            self._setup_default_indicators()
    
    def _setup_default_indicators(self):
        """Setup default indicators when CSV file is not available"""
        # Core price data indicators (always required)
        default_indicators = {
            'Timestamp': {'category': 'Core Price Data', 'must_keep': True, 'rfe_eligible': False},
            'Open': {'category': 'Core Price Data', 'must_keep': True, 'rfe_eligible': False},
            'High': {'category': 'Core Price Data', 'must_keep': True, 'rfe_eligible': False},
            'Low': {'category': 'Core Price Data', 'must_keep': True, 'rfe_eligible': False},
            'Close': {'category': 'Core Price Data', 'must_keep': True, 'rfe_eligible': False},
            'Volume': {'category': 'Core Price Data', 'must_keep': True, 'rfe_eligible': False},
            'Symbol': {'category': 'Core Price Data', 'must_keep': True, 'rfe_eligible': False},
            # Basic technical indicators
            'SMA_5': {'category': 'Trend Indicators', 'must_keep': False, 'rfe_eligible': True},
            'SMA_10': {'category': 'Trend Indicators', 'must_keep': False, 'rfe_eligible': True},
            'EMA_5': {'category': 'Trend Indicators', 'must_keep': False, 'rfe_eligible': True},
            'EMA_10': {'category': 'Trend Indicators', 'must_keep': False, 'rfe_eligible': True},
            'RSI_14': {'category': 'Momentum Indicators', 'must_keep': False, 'rfe_eligible': True},
        }
        
        self.indicators_config = default_indicators
        
        for name, config in default_indicators.items():
            if config['must_keep']:
                self.required_indicators.add(name)
            if config['rfe_eligible']:
                self.rfe_eligible_indicators.add(name)
        
        logger.warning(f"Using default configuration with {len(default_indicators)} indicators")
    
    def setup_indicator_functions(self):
        """Setup functions for calculating indicators"""
        
        # Core Price Data functions
        self.indicator_functions.update({
            'Timestamp': self.calc_timestamp,
            'Open': self.calc_open,
            'High': self.calc_high,
            'Low': self.calc_low,
            'Close': self.calc_close,
            'Volume': self.calc_volume,
            'Symbol': self.calc_symbol,
        })
        
        # Prerequisite functions
        self.indicator_functions.update({
            'Prev Close': lambda df: df['close'].shift(1),
            'Prev High': lambda df: df['high'].shift(1),
            'Prev Low': lambda df: df['low'].shift(1),
            'Typical Price (TP)': lambda df: (df['high'] + df['low'] + df['close']) / 3,
            'Median Price (MP)': lambda df: (df['high'] + df['low']) / 2,
            'HLC3': lambda df: (df['high'] + df['low'] + df['close']) / 3,
            'OHLC4': lambda df: (df['open'] + df['high'] + df['low'] + df['close']) / 4,
            'True Range (TR)': self.calc_true_range,
            'Log Return (1)': lambda df: np.log(df['close'] / df['close'].shift(1)),
        })
        
        # SMA functions
        for period in [3, 5, 7, 9, 10, 12, 14, 20, 21, 26, 30, 34, 40, 50, 55, 60, 89, 100, 120, 144, 150, 200]:
            self.indicator_functions[f'SMA_{period}'] = lambda df, p=period: talib.SMA(df['close'], timeperiod=p)
        
        # EMA functions
        for period in [3, 5, 7, 9, 10, 12, 14, 20, 21, 26, 30, 34, 40, 50, 55, 60, 89, 100, 120, 144, 150, 200]:
            self.indicator_functions[f'EMA_{period}'] = lambda df, p=period: talib.EMA(df['close'], timeperiod=p)
        
        # WMA functions
        for period in [5, 9, 14, 20, 30, 50, 100, 200]:
            self.indicator_functions[f'WMA_{period}'] = lambda df, p=period: talib.WMA(df['close'], timeperiod=p)
        
        # HMA functions
        for period in [9, 14, 20, 30, 50, 100]:
            self.indicator_functions[f'HMA_{period}'] = lambda df, p=period: self.calc_hull_ma(df['close'], p)
        
        # DEMA and TEMA functions
        for period in [10, 20, 30, 50, 100]:
            self.indicator_functions[f'DEMA_{period}'] = lambda df, p=period: talib.DEMA(df['close'], timeperiod=p)
            self.indicator_functions[f'TEMA_{period}'] = lambda df, p=period: talib.TEMA(df['close'], timeperiod=p)
        
        # KAMA functions
        for period in [10, 20, 30, 50]:
            self.indicator_functions[f'KAMA_{period}'] = lambda df, p=period: talib.KAMA(df['close'], timeperiod=p)
        
        # ZLEMA functions (Zero-lag EMA)
        for period in [10, 20, 30, 50, 100]:
            self.indicator_functions[f'ZLEMA_{period}'] = lambda df, p=period: self.calc_zlema(df['close'], p)
        
        # LSMA functions (Least Squares Moving Average)
        for period in [14, 21, 50, 100]:
            self.indicator_functions[f'LSMA_{period}'] = lambda df, p=period: self.calc_lsma(df['close'], p)
        
        # MACD functions
        macd_configs = [
            (12, 26, 9), (5, 35, 5), (8, 24, 9), (10, 20, 7), (20, 50, 9)
        ]
        for fast, slow, signal in macd_configs:
            self.indicator_functions[f'MACD_{fast}_{slow}_{signal}'] = lambda df, f=fast, s=slow, sig=signal: self.calc_macd(df['close'], f, s, sig)
        
        # Ichimoku functions
        self.indicator_functions.update({
            'Ichimoku_Tenkan': lambda df: self.calc_ichimoku_tenkan(df, 9),
            'Ichimoku_Kijun': lambda df: self.calc_ichimoku_kijun(df, 26),
            'Ichimoku_SenkouA': lambda df: self.calc_ichimoku_senkou_a(df),
            'Ichimoku_SenkouB': lambda df: self.calc_ichimoku_senkou_b(df, 52),
            'Ichimoku_Chikou': lambda df: df['close'].shift(-26),
        })
        
        # Parabolic SAR
        self.indicator_functions['Parabolic_SAR'] = lambda df: talib.SAR(df['high'], df['low'])
        
        # SuperTrend functions
        st_configs = [
            (7, 1.5), (7, 2.0), (7, 2.5), (7, 3.0),
            (10, 1.5), (10, 2.0), (10, 2.5), (10, 3.0),
            (14, 1.5), (14, 2.0), (14, 2.5), (14, 3.0)
        ]
        for atr_period, mult in st_configs:
            self.indicator_functions[f'SuperTrend_ATR{atr_period}_M{mult}'] = lambda df, atr=atr_period, m=mult: self.calc_supertrend(df, atr, m)
        
        # RSI functions
        for period in [2, 3, 5, 7, 9, 14, 21, 28, 50]:
            self.indicator_functions[f'RSI_{period}'] = lambda df, p=period: talib.RSI(df['close'], timeperiod=p)
        
        # Stochastic functions
        for period in [5, 9, 14]:
            self.indicator_functions[f'Stoch_%K_{period}'] = lambda df, p=period: talib.STOCH(df['high'], df['low'], df['close'], fastk_period=p)[0]
            self.indicator_functions[f'Stoch_%D_{period}'] = lambda df, p=period: talib.STOCH(df['high'], df['low'], df['close'], fastk_period=p)[1]
        
        # CCI functions
        for period in [14, 20, 50]:
            self.indicator_functions[f'CCI_{period}'] = lambda df, p=period: talib.CCI(df['high'], df['low'], df['close'], timeperiod=p)
        
        # Williams %R functions
        for period in [7, 14, 20]:
            self.indicator_functions[f'Williams_%R_{period}'] = lambda df, p=period: talib.WILLR(df['high'], df['low'], df['close'], timeperiod=p)
        
        # ROC functions
        for period in [1, 2, 3, 5, 7, 9, 14, 21, 28]:
            self.indicator_functions[f'ROC_{period}'] = lambda df, p=period: talib.ROC(df['close'], timeperiod=p)
        
        # MFI functions
        for period in [10, 14, 20, 50]:
            self.indicator_functions[f'MFI_{period}'] = lambda df, p=period: talib.MFI(df['high'], df['low'], df['close'], df['volume'], timeperiod=p)
        
        # Momentum functions
        for period in [10, 20, 50, 100]:
            self.indicator_functions[f'Momentum_{period}'] = lambda df, p=period: talib.MOM(df['close'], timeperiod=p)
        
        # ATR functions
        for period in [7, 10, 14, 21]:
            self.indicator_functions[f'ATR_{period}'] = lambda df, p=period: talib.ATR(df['high'], df['low'], df['close'], timeperiod=p)
        
        # Bollinger Bands functions
        bb_configs = [(10, 1.0), (10, 2.0), (10, 3.0), (14, 1.0), (14, 2.0), (14, 3.0), (20, 1.0), (20, 2.0), (20, 3.0)]
        for period, mult in bb_configs:
            self.indicator_functions[f'Bollinger_{period}_x{mult}'] = lambda df, p=period, m=mult: self.calc_bollinger_bands(df['close'], p, m)
            self.indicator_functions[f'Bollinger_Width_{period}_x{mult}'] = lambda df, p=period, m=mult: self.calc_bollinger_width(df['close'], p, m)
        
        # Add more indicator functions as needed...
        # This is a comprehensive start covering the major categories
    
    def calc_timestamp(self, df):
        return df['timestamp'] if 'timestamp' in df else df.index
    
    def calc_open(self, df):
        return df['open']
    
    def calc_high(self, df):
        return df['high']
    
    def calc_low(self, df):
        return df['low']
    
    def calc_close(self, df):
        return df['close']
    
    def calc_volume(self, df):
        return df['volume']
    
    def calc_symbol(self, df):
        return df.get('symbol', 'UNKNOWN')
    
    def calc_true_range(self, df):
        """Calculate True Range"""
        high_low = df['high'] - df['low']
        high_close_prev = np.abs(df['high'] - df['close'].shift(1))
        low_close_prev = np.abs(df['low'] - df['close'].shift(1))
        return np.maximum(high_low, np.maximum(high_close_prev, low_close_prev))
    
    def calc_hull_ma(self, series, period):
        """Calculate Hull Moving Average"""
        half_period = int(period / 2)
        sqrt_period = int(np.sqrt(period))
        
        wma_half = talib.WMA(series, timeperiod=half_period)
        wma_full = talib.WMA(series, timeperiod=period)
        
        raw_hma = 2 * wma_half - wma_full
        return talib.WMA(raw_hma, timeperiod=sqrt_period)
    
    def calc_zlema(self, series, period):
        """Calculate Zero-Lag EMA"""
        lag = int((period - 1) / 2)
        ema_data = series + (series - series.shift(lag))
        return talib.EMA(ema_data, timeperiod=period)
    
    def calc_lsma(self, series, period):
        """Calculate Least Squares Moving Average (Linear Regression)"""
        lsma = pd.Series(index=series.index, dtype=float)
        
        for i in range(period - 1, len(series)):
            y = series.iloc[i - period + 1:i + 1].values
            x = np.arange(period)
            slope, intercept, _, _, _ = stats.linregress(x, y)
            lsma.iloc[i] = slope * (period - 1) + intercept
        
        return lsma
    
    def calc_macd(self, series, fast_period, slow_period, signal_period):
        """Calculate MACD with all outputs"""
        macd_line, signal_line, histogram = talib.MACD(series, fastperiod=fast_period, slowperiod=slow_period, signalperiod=signal_period)
        return {'macd': macd_line, 'signal': signal_line, 'hist': histogram}
    
    def calc_ichimoku_tenkan(self, df, period):
        """Calculate Ichimoku Tenkan line"""
        high_max = df['high'].rolling(window=period).max()
        low_min = df['low'].rolling(window=period).min()
        return (high_max + low_min) / 2
    
    def calc_ichimoku_kijun(self, df, period):
        """Calculate Ichimoku Kijun line"""
        high_max = df['high'].rolling(window=period).max()
        low_min = df['low'].rolling(window=period).min()
        return (high_max + low_min) / 2
    
    def calc_ichimoku_senkou_a(self, df):
        """Calculate Ichimoku Senkou Span A"""
        tenkan = self.calc_ichimoku_tenkan(df, 9)
        kijun = self.calc_ichimoku_kijun(df, 26)
        return ((tenkan + kijun) / 2).shift(26)
    
    def calc_ichimoku_senkou_b(self, df, period):
        """Calculate Ichimoku Senkou Span B"""
        high_max = df['high'].rolling(window=period).max()
        low_min = df['low'].rolling(window=period).min()
        return ((high_max + low_min) / 2).shift(26)
    
    def calc_supertrend(self, df, atr_period, multiplier):
        """Calculate SuperTrend indicator"""
        atr = talib.ATR(df['high'], df['low'], df['close'], timeperiod=atr_period)
        mp = (df['high'] + df['low']) / 2
        
        upper_band = mp + (multiplier * atr)
        lower_band = mp - (multiplier * atr)
        
        supertrend = pd.Series(index=df.index, dtype=float)
        direction = pd.Series(index=df.index, dtype=float)
        
        for i in range(1, len(df)):
            if df['close'].iloc[i] <= lower_band.iloc[i-1]:
                supertrend.iloc[i] = upper_band.iloc[i]
                direction.iloc[i] = -1
            elif df['close'].iloc[i] >= upper_band.iloc[i-1]:
                supertrend.iloc[i] = lower_band.iloc[i]
                direction.iloc[i] = 1
            else:
                supertrend.iloc[i] = supertrend.iloc[i-1]
                direction.iloc[i] = direction.iloc[i-1]
        
        return {'direction': direction, 'band': supertrend}
    
    def calc_bollinger_bands(self, series, period, std_mult):
        """Calculate Bollinger Bands"""
        sma = talib.SMA(series, timeperiod=period)
        std = series.rolling(window=period).std()
        
        upper = sma + (std * std_mult)
        lower = sma - (std * std_mult)
        
        return {'upper': upper, 'middle': sma, 'lower': lower}
    
    def calc_bollinger_width(self, series, period, std_mult):
        """Calculate Bollinger Bands Width"""
        bb = self.calc_bollinger_bands(series, period, std_mult)
        return (bb['upper'] - bb['lower']) / bb['middle']
    
    async def calculate_all_indicators(self, df: pd.DataFrame, symbol: str = None) -> Dict[str, Any]:
        """Calculate all indicators for the given data"""
        try:
            results = {}
            calculated = set()
            
            # Add symbol if provided
            if symbol:
                df = df.copy()
                df['symbol'] = symbol
            
            # Calculate indicators in dependency order
            max_iterations = 10  # Prevent infinite loops
            iteration = 0
            
            while len(calculated) < len(self.indicator_functions) and iteration < max_iterations:
                iteration += 1
                initial_count = len(calculated)
                
                for indicator_name, calc_function in self.indicator_functions.items():
                    if indicator_name in calculated:
                        continue
                    
                    try:
                        # Check if we have all prerequisites
                        if self.can_calculate_indicator(indicator_name, calculated):
                            result = calc_function(df)
                            
                            if isinstance(result, dict):
                                # Handle multi-output indicators
                                for key, value in result.items():
                                    results[f"{indicator_name}_{key}"] = value
                            else:
                                results[indicator_name] = result
                            
                            calculated.add(indicator_name)
                            
                    except Exception as e:
                        logger.debug(f"Could not calculate {indicator_name}: {e}")
                        continue
                
                # If no progress was made, break to avoid infinite loop
                if len(calculated) == initial_count:
                    break
            
            logger.info(f"Calculated {len(calculated)} out of {len(self.indicator_functions)} indicators")
            
            return results
            
        except Exception as e:
            logger.error(f"Failed to calculate indicators: {e}")
            return {}
    
    def can_calculate_indicator(self, indicator_name: str, calculated: set) -> bool:
        """Check if an indicator can be calculated based on prerequisites"""
        # For now, assume all can be calculated if we have basic OHLCV data
        # This can be enhanced to check specific prerequisites
        return True
    
    def get_rfe_eligible_indicators(self) -> List[str]:
        """Get list of indicators eligible for RFE"""
        return list(self.rfe_eligible_indicators)
    
    def get_required_indicators(self) -> List[str]:
        """Get list of indicators that must be kept"""
        return list(self.required_indicators)