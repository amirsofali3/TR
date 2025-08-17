"""
Technical Indicator Engine for calculating all trading indicators
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Any
from loguru import logger
import os
import csv

# Optional imports - will be gracefully handled if missing
try:
    import talib
    TALIB_AVAILABLE = True
except ImportError:
    TALIB_AVAILABLE = False
    logger.warning("TA-Lib not available - some indicators will be skipped")

try:
    from scipy import stats
    SCIPY_AVAILABLE = True
except ImportError:
    SCIPY_AVAILABLE = False
    logger.warning("SciPy not available - some statistical indicators will be skipped")

try:
    from sklearn.linear_model import LinearRegression
    SKLEARN_AVAILABLE = True
except ImportError:
    SKLEARN_AVAILABLE = False
    logger.warning("Scikit-learn not available - some ML-based indicators will be skipped")

class IndicatorEngine:
    """Calculates all technical indicators from the CSV file with encyclopedia support"""
    
    def __init__(self):
        self.indicators_config = {}
        self.required_indicators = set()  # must_keep indicators
        self.rfe_eligible_indicators = set()
        self.indicator_functions = {}
        self.prerequisite_map = {}
        
        # New fields for Complete Pipeline Restructure
        self.skipped_indicators = []  # List of {name, reason} for skipped indicators
        self.computed_indicators = set()  # Successfully computed indicators
        self.must_keep_features = []  # Final list of must-keep feature names
        self.rfe_candidates = []  # Final list of RFE candidate feature names
        
        # Classification sets for improved must-keep and RFE detection
        self.must_keep_indicator_set = set()  # Case-insensitive must-keep indicator names
        self.rfe_indicator_set = set()  # Case-insensitive RFE-eligible indicator names
        
        # OHLCV Separation (NEW - FIXED)
        self.ohlcv_base_features = []  # Base OHLCV features (never processed by RFE)
        self.technical_indicators = []  # Only technical indicators (for RFE processing)
        
        # Feature tracking file path
        self.feature_tracking_file = "selected_features.json"
    
    def _sanitize_indicator_name(self, name: str) -> str:
        """Sanitize indicator name for consistent matching (case-insensitive, normalized)"""
        if not name:
            return ""
        # Convert to lowercase and strip whitespace
        return name.lower().strip()
    
    def _build_indicator_classification_sets(self):
        """Build case-insensitive classification sets from loaded CSV configuration"""
        self.must_keep_indicator_set.clear()
        self.rfe_indicator_set.clear()
        
        for indicator_name, config in self.indicators_config.items():
            sanitized_name = self._sanitize_indicator_name(indicator_name)
            
            if config.get('must_keep', False):
                self.must_keep_indicator_set.add(sanitized_name)
                logger.debug(f"[INDICATORS] Added to must-keep set: {sanitized_name}")
            
            if config.get('rfe_eligible', False):
                self.rfe_indicator_set.add(sanitized_name)
                logger.debug(f"[INDICATORS] Added to RFE set: {sanitized_name}")
        
        logger.info(f"[INDICATORS] Built classification sets - Must-keep: {len(self.must_keep_indicator_set)}, RFE-eligible: {len(self.rfe_indicator_set)}")
    
    def _extract_base_indicator_name(self, feature_name: str) -> str:
        """Extract base indicator name from feature name, handling complex cases like Stoch_%K_5, Williams_%R_14"""
        if not feature_name:
            return ""
        
        # Handle special cases with % signs first
        if '%' in feature_name:
            # Cases like "Stoch_%K_5", "Williams_%R_14" 
            # Try to find the longest match from our known indicators
            sanitized_feature = self._sanitize_indicator_name(feature_name)
            
            # Direct match first
            if sanitized_feature in self.must_keep_indicator_set or sanitized_feature in self.rfe_indicator_set:
                return sanitized_feature
                
            # Try progressive splitting for complex names
            parts = feature_name.split('_')
            for i in range(len(parts), 0, -1):
                potential_base = '_'.join(parts[:i])
                sanitized_base = self._sanitize_indicator_name(potential_base)
                if sanitized_base in self.must_keep_indicator_set or sanitized_base in self.rfe_indicator_set:
                    return sanitized_base
        
        # Standard processing for names with underscores
        if '_' in feature_name:
            parts = feature_name.split('_')
            if len(parts) >= 2:
                # First try first part only
                potential_base = parts[0]
                sanitized_base = self._sanitize_indicator_name(potential_base)
                if sanitized_base in self.must_keep_indicator_set or sanitized_base in self.rfe_indicator_set:
                    return sanitized_base
                
                # Then try first two parts joined
                potential_base = '_'.join(parts[:2])
                sanitized_base = self._sanitize_indicator_name(potential_base)
                if sanitized_base in self.must_keep_indicator_set or sanitized_base in self.rfe_indicator_set:
                    return sanitized_base
                
                # Finally try the whole name minus last part (for cases like "SuperTrend_ATR14_M2")
                if len(parts) > 2:
                    potential_base = '_'.join(parts[:-1])
                    sanitized_base = self._sanitize_indicator_name(potential_base)
                    if sanitized_base in self.must_keep_indicator_set or sanitized_base in self.rfe_indicator_set:
                        return sanitized_base
        
        # Return sanitized full name as fallback
        return self._sanitize_indicator_name(feature_name)
        
    async def initialize(self):
        """Initialize the indicator engine with OHLCV separation"""
        try:
            logger.info("[INDICATORS] Initializing indicator engine with OHLCV separation...")
            
            # Initialize OHLCV base features first (never processed by RFE)
            self._initialize_ohlcv_base_features()
            
            # Load technical indicators configuration from CSV (excluding OHLCV)
            await self.load_indicators_config()
            
            # Setup indicator functions
            self.setup_indicator_functions()
            
            logger.success(f"[INDICATORS] Engine initialized - OHLCV base: {len(self.ohlcv_base_features)}, Technical indicators: {len(self.indicators_config)}")
            
        except Exception as e:
            logger.error(f"[INDICATORS] Failed to initialize: {e}")
            raise
    
    def _initialize_ohlcv_base_features(self):
        """Initialize OHLCV base features that are never processed by RFE"""
        try:
            from config.settings import OHLCV_BASE_FEATURES
            self.ohlcv_base_features = OHLCV_BASE_FEATURES.copy()
        except ImportError:
            self.ohlcv_base_features = ["timestamp", "open", "high", "low", "close", "volume", "symbol"]
        
        logger.info(f"[INDICATORS] OHLCV base features: {self.ohlcv_base_features}")
        logger.info("[INDICATORS] These features will NEVER be processed by RFE - they are always available as base data")
    
    async def load_indicators_config(self):
        """Load indicators configuration from CSV file (OHLCV-only mode)"""
        try:
            # Import settings to get INDICATOR_CSV_PATH
            from config.settings import INDICATOR_CSV_PATH
            
            # Use relative path from project root
            import os
            project_root = os.path.dirname(os.path.dirname(os.path.dirname(__file__)))
            csv_path = os.path.join(project_root, INDICATOR_CSV_PATH)
            
            logger.info(f"[INDICATORS] Loading OHLCV-only indicators from: {csv_path}")
            
            if not os.path.exists(csv_path):
                logger.warning(f"[INDICATORS] CSV file not found: {csv_path}")
                logger.info("[INDICATORS] Falling back to default OHLCV indicators")
                self._setup_default_ohlcv_indicators()
                return
            
            with open(csv_path, 'r', encoding='utf-8') as file:
                reader = csv.DictReader(file)
                
                total_indicators = 0
                must_keep_count = 0
                rfe_eligible_count = 0
                
                for row in reader:
                    indicator_name = row['Indicator'].strip()
                    if not indicator_name:
                        continue
                    
                    total_indicators += 1
                    
                    # Parse boolean fields safely
                    must_keep = row.get('Must Keep (Not in RFE)', '').lower().strip() == 'yes'
                    rfe_eligible = row.get('RFE Eligible', '').lower().strip() == 'yes'
                    
                    self.indicators_config[indicator_name] = {
                        'category': row.get('Category', '').strip(),
                        'required_inputs': row.get('Required Inputs', '').strip(),
                        'formula': row.get('Formula / Calculation', '').strip(),
                        'must_keep': must_keep,
                        'rfe_eligible': rfe_eligible,
                        'prerequisite_for': row.get('Prerequisite For', '').strip(),
                        'parameters': row.get('Parameters', '').strip(),
                        'outputs': row.get('Outputs', '').strip()
                    }
                    
                    # Track required indicators (must keep)
                    if must_keep:
                        self.required_indicators.add(indicator_name)
                        must_keep_count += 1
                    
                    # Track RFE eligible indicators
                    if rfe_eligible:
                        self.rfe_eligible_indicators.add(indicator_name)
                        rfe_eligible_count += 1
                    
                    # Track prerequisites
                    prereq_for = row.get('Prerequisite For', '').strip()
                    if prereq_for:
                        for prereq in prereq_for.split(','):
                            prereq = prereq.strip()
                            if prereq:
                                if prereq not in self.prerequisite_map:
                                    self.prerequisite_map[prereq] = []
                                self.prerequisite_map[prereq].append(indicator_name)
            
            logger.success(f"[INDICATORS] OHLCV-only indicators loaded successfully:")
            logger.info(f"[INDICATORS]   - Total defined: {total_indicators}")
            logger.info(f"[INDICATORS]   - Must keep: {must_keep_count}")
            logger.info(f"[INDICATORS]   - RFE eligible: {rfe_eligible_count}")
            logger.info(f"[INDICATORS]   - With prerequisites: {len(self.prerequisite_map)}")
            
            # Build classification sets after loading configuration
            self._build_indicator_classification_sets()
            
        except FileNotFoundError:
            logger.error(f"[INDICATORS] OHLCV indicators CSV file not found at {csv_path}")
            logger.info("[INDICATORS] Falling back to default OHLCV indicators")
            self._setup_default_ohlcv_indicators()
        except Exception as e:
            logger.error(f"[INDICATORS] Failed to load OHLCV indicators: {e}")
            logger.info("[INDICATORS] Falling back to default OHLCV indicators")
            self._setup_default_ohlcv_indicators()
    
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
    
    def _setup_default_ohlcv_indicators(self):
        """Setup default OHLCV-only indicators when CSV file is not available"""
        # Based on BASE_MUST_KEEP_FEATURES and essential OHLCV prerequisites
        from config.settings import BASE_MUST_KEEP_FEATURES
        
        # Core OHLCV data indicators (always required)
        default_ohlcv_indicators = {
            'Timestamp': {'category': 'Core Price Data', 'must_keep': True, 'rfe_eligible': False},
            'Open': {'category': 'Core Price Data', 'must_keep': True, 'rfe_eligible': False},
            'High': {'category': 'Core Price Data', 'must_keep': True, 'rfe_eligible': False},
            'Low': {'category': 'Core Price Data', 'must_keep': True, 'rfe_eligible': False},
            'Close': {'category': 'Core Price Data', 'must_keep': True, 'rfe_eligible': False},
            'Volume': {'category': 'Core Price Data', 'must_keep': True, 'rfe_eligible': False},
            'Symbol': {'category': 'Core Price Data', 'must_keep': True, 'rfe_eligible': False},
            
            # Essential OHLCV prerequisites
            'Prev Close': {'category': 'Prereq', 'must_keep': True, 'rfe_eligible': False},
            'Prev High': {'category': 'Prereq', 'must_keep': True, 'rfe_eligible': False}, 
            'Prev Low': {'category': 'Prereq', 'must_keep': True, 'rfe_eligible': False},
            'Typical Price (TP)': {'category': 'Prereq', 'must_keep': True, 'rfe_eligible': False},
            'Median Price (MP)': {'category': 'Prereq', 'must_keep': True, 'rfe_eligible': False},
            'HLC3': {'category': 'Prereq', 'must_keep': True, 'rfe_eligible': False},
            'OHLC4': {'category': 'Prereq', 'must_keep': True, 'rfe_eligible': False},
            
            # Essential RFE-eligible OHLCV-based indicators
            'Return_1': {'category': 'Price Action', 'must_keep': False, 'rfe_eligible': True},
            'Return_2': {'category': 'Price Action', 'must_keep': False, 'rfe_eligible': True},
            'Return_3': {'category': 'Price Action', 'must_keep': False, 'rfe_eligible': True},
            'Return_5': {'category': 'Price Action', 'must_keep': False, 'rfe_eligible': True},
            'Return_7': {'category': 'Price Action', 'must_keep': False, 'rfe_eligible': True},
            'Return_10': {'category': 'Price Action', 'must_keep': False, 'rfe_eligible': True},
            'Return_14': {'category': 'Price Action', 'must_keep': False, 'rfe_eligible': True},
            'Return_21': {'category': 'Price Action', 'must_keep': False, 'rfe_eligible': True},
            'HighLowRange_1': {'category': 'Price Action', 'must_keep': False, 'rfe_eligible': True},
            'HighLowRange_2': {'category': 'Price Action', 'must_keep': False, 'rfe_eligible': True},
            'HighLowRange_3': {'category': 'Price Action', 'must_keep': False, 'rfe_eligible': True},
            'HighLowRange_5': {'category': 'Price Action', 'must_keep': False, 'rfe_eligible': True},
            'CloseToRange_1': {'category': 'Price Action', 'must_keep': False, 'rfe_eligible': True},
            'CloseToRange_2': {'category': 'Price Action', 'must_keep': False, 'rfe_eligible': True},
            'CloseToRange_3': {'category': 'Price Action', 'must_keep': False, 'rfe_eligible': True},
            'CloseToRange_5': {'category': 'Price Action', 'must_keep': False, 'rfe_eligible': True},
            'Body': {'category': 'Price Action', 'must_keep': False, 'rfe_eligible': True},
            'UpperWick': {'category': 'Price Action', 'must_keep': False, 'rfe_eligible': True},
            'LowerWick': {'category': 'Price Action', 'must_keep': False, 'rfe_eligible': True},
            'BodyToRange': {'category': 'Price Action', 'must_keep': False, 'rfe_eligible': True},
            'Volume_SMA_10': {'category': 'Volume', 'must_keep': False, 'rfe_eligible': True},
            'Volume_SMA_20': {'category': 'Volume', 'must_keep': False, 'rfe_eligible': True}
        }
        
        self.indicators_config = default_ohlcv_indicators
        self.required_indicators = set()
        self.rfe_eligible_indicators = set()
        
        for name, config in default_ohlcv_indicators.items():
            if config['must_keep']:
                self.required_indicators.add(name)
            if config['rfe_eligible']:
                self.rfe_eligible_indicators.add(name)
        
        logger.info(f"[INDICATORS] Using default OHLCV-only configuration with {len(default_ohlcv_indicators)} indicators")
        logger.info(f"[INDICATORS] Must keep: {len(self.required_indicators)}, RFE eligible: {len(self.rfe_eligible_indicators)}")
        
        # Build classification sets after setting up default configuration
        self._build_indicator_classification_sets()
    
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
        """Calculate all indicators for the given data (Complete Pipeline Restructure)"""
        try:
            results = {}
            calculated = set()
            self.skipped_indicators = []  # Reset skipped indicators list
            self.computed_indicators = set()
            
            if symbol:
                df = df.copy()
                df['symbol'] = symbol
            
            logger.info(f"[INDICATORS] Calculating indicators for {len(df)} data points...")
            
            max_iterations = 10
            iteration = 0
            
            # Phase 1: Calculate all indicators
            while len(calculated) < len(self.indicator_functions) and iteration < max_iterations:
                iteration += 1
                initial_count = len(calculated)
                
                for indicator_name, calc_function in self.indicator_functions.items():
                    if indicator_name in calculated:
                        continue
                    try:
                        if self.can_calculate_indicator(indicator_name, calculated, df):
                            result = calc_function(df)
                            if result is None or (hasattr(result, '__len__') and len(result) == 0):
                                self.skipped_indicators.append({'name': indicator_name, 'reason': 'Empty or None result returned'})
                                continue
                            if isinstance(result, dict):
                                for key, value in result.items():
                                    feature_name = f"{indicator_name}_{key}"
                                    results[feature_name] = value
                                    self.computed_indicators.add(feature_name)
                            else:
                                results[indicator_name] = result
                                self.computed_indicators.add(indicator_name)
                            calculated.add(indicator_name)
                    except Exception as e:
                        self.skipped_indicators.append({'name': indicator_name,'reason': f'Calculation error: {str(e)}'})
                        logger.debug(f"[INDICATORS] Could not calculate {indicator_name}: {e}")
                        continue
                if len(calculated) == initial_count:
                    break
            
            for indicator_name in self.indicator_functions.keys():
                if indicator_name not in calculated:
                    self.skipped_indicators.append({'name': indicator_name,'reason': 'Missing dependencies or insufficient iterations'})
            
            # Phase 2: Build classification map for improved feature categorization
            classification_map = {}
            for feature_name in self.computed_indicators:
                # Improved base indicator extraction - handle multi-output indicators and special characters
                base_indicator = self._extract_base_indicator_name(feature_name)
                classification_map[feature_name] = base_indicator
            
            # Phase 3: Classify features into must-keep and RFE candidates  
            self.must_keep_features = []
            self.rfe_candidates = []
            
            # Always include base OHLCV features in must_keep, even if not computed by indicators
            try:
                from config.settings import BASE_MUST_KEEP_FEATURES
                base_features = BASE_MUST_KEEP_FEATURES + ['timestamp', 'symbol']  # Include meta features
            except ImportError:
                base_features = ["open", "high", "low", "close", "volume", "timestamp", "symbol"]
            
            # Map base features to actual computed feature names (case-insensitive)
            computed_lower_map = {self._sanitize_indicator_name(c): c for c in self.computed_indicators}
            original_columns_lower = {self._sanitize_indicator_name(c): c for c in df.columns}
            
            for base_feature in base_features:
                sanitized_base = self._sanitize_indicator_name(base_feature)
                actual_feature_name = None
                
                # First try to find in computed indicators
                if sanitized_base in computed_lower_map:
                    actual_feature_name = computed_lower_map[sanitized_base]
                # Fall back to original DataFrame columns
                elif sanitized_base in original_columns_lower:
                    actual_feature_name = original_columns_lower[sanitized_base]
                    # Add to computed indicators if found in original columns
                    self.computed_indicators.add(actual_feature_name)
                
                if actual_feature_name and actual_feature_name not in self.must_keep_features:
                    self.must_keep_features.append(actual_feature_name)
                    logger.debug(f"[INDICATORS] Added base feature to must-keep: {actual_feature_name}")
            
            # Classify computed features using improved classification
            for feature_name in self.computed_indicators:
                if feature_name in self.must_keep_features:
                    continue  # Already classified as must-keep
                    
                base_indicator = classification_map.get(feature_name, self._sanitize_indicator_name(feature_name))
                
                if base_indicator in self.must_keep_indicator_set:
                    self.must_keep_features.append(feature_name)
                    logger.debug(f"[INDICATORS] Classified as must-keep: {feature_name} (base: {base_indicator})")
                elif base_indicator in self.rfe_indicator_set:
                    self.rfe_candidates.append(feature_name)
                    logger.debug(f"[INDICATORS] Classified as RFE candidate: {feature_name} (base: {base_indicator})")
                else:
                    # Default: add to RFE candidates if not explicitly must-keep
                    self.rfe_candidates.append(feature_name)
                    logger.debug(f"[INDICATORS] Defaulting to RFE candidate: {feature_name} (base: {base_indicator})")
            
            # Ensure RFE candidates exclude must-keep features
            self.rfe_candidates = [f for f in self.rfe_candidates if f not in self.must_keep_features]
            
            logger.success("[INDICATORS] Calculation completed")
            logger.info(f"[INDICATORS]   - Computed: {len(self.computed_indicators)}")
            logger.info(f"[INDICATORS]   - Must keep: {len(self.must_keep_features)}")
            logger.info(f"[INDICATORS]   - RFE candidates: {len(self.rfe_candidates)}")
            logger.info(f"[INDICATORS]   - Skipped: {len(self.skipped_indicators)}")
            
            if self.skipped_indicators:
                logger.warning(f"[INDICATORS] {len(self.skipped_indicators)} indicators were skipped")
                for skipped in self.skipped_indicators[:5]:
                    logger.debug(f"[INDICATORS]   - {skipped['name']}: {skipped['reason']}")
                if len(self.skipped_indicators) > 5:
                    logger.debug(f"[INDICATORS]   - ... and {len(self.skipped_indicators) - 5} more")
            
            # Phase 4: Build final DataFrame with single concat to avoid fragmentation warnings
            if results:
                # Create indicators DataFrame from results
                indicators_df = pd.DataFrame(results, index=df.index)
                
                # Combine original data with indicators using concat
                features_df = pd.concat([df, indicators_df], axis=1)
            else:
                features_df = df.copy()
                
            # Drop all-NaN columns
            empty_cols = [c for c in features_df.columns if features_df[c].isna().all()]
            if empty_cols:
                features_df.drop(columns=empty_cols, inplace=True)
                logger.debug(f"[INDICATORS] Dropped {len(empty_cols)} empty columns")
                
            logger.info(f"[INDICATORS] Final features dataframe shape: {features_df.shape} (added {len(results)} indicator columns)")
            
            return {
                'dataframe': features_df,
                'computed_features': sorted(list(self.computed_indicators)),
                'must_keep_features': self.get_must_keep_features(),
                'rfe_candidates': sorted(self.rfe_candidates),
                'skipped': self.get_skipped_indicators()
            }
        except Exception as e:
            logger.error(f"[INDICATORS] Failed to calculate indicators: {e}")
            return {}
    
    def can_calculate_indicator(self, indicator_name: str, calculated: set, df: pd.DataFrame = None) -> bool:
        """Check if an indicator can be calculated based on prerequisites (Complete Pipeline Restructure)"""
        try:
            # Check if we have the basic OHLCV data required
            if df is not None:
                required_columns = ['open', 'high', 'low', 'close', 'volume']
                if not all(col in df.columns for col in required_columns):
                    return False
                
                # Check for sufficient data points
                if len(df) < 2:
                    return False
            
            # Check specific prerequisites from configuration
            config = self.indicators_config.get(indicator_name, {})
            required_inputs = config.get('required_inputs', '')
            
            if required_inputs:
                # Parse required inputs and check if they're available
                inputs = [inp.strip() for inp in required_inputs.split(',') if inp.strip()]
                for required_input in inputs:
                    # Check if required input is in calculated indicators or basic OHLCV
                    if (required_input not in calculated and 
                        required_input.lower() not in ['ohlcv', 'ohlc', 'price', 'volume', 'timestamp']):
                        return False
            
            return True
            
        except Exception as e:
            logger.debug(f"[INDICATORS] Error checking prerequisites for {indicator_name}: {e}")
            return False
    
    def get_rfe_eligible_indicators(self) -> List[str]:
        """Get list of indicators eligible for RFE"""
        return list(self.rfe_eligible_indicators)
    
    def get_required_indicators(self) -> List[str]:
        """Get list of indicators that must be kept"""
        return list(self.required_indicators)
    
    # New methods for Complete Pipeline Restructure
    
    def get_must_keep_features(self) -> List[str]:
        """Get final list of must-keep feature names after calculation (OHLCV separation FIXED)"""
        # OHLCV base features are ALWAYS kept but NEVER processed by RFE
        ohlcv_features = self.ohlcv_base_features.copy()
        
        # Add any computed technical indicators that are marked as must-keep
        computed_map = {c.lower(): c for c in self.must_keep_features}
        ordered = ohlcv_features.copy()  # Start with OHLCV base
        
        # Add other must_keep technical indicators 
        for f in self.must_keep_features:
            if f.lower() not in [ohlcv.lower() for ohlcv in self.ohlcv_base_features]:
                if f not in ordered:
                    ordered.append(f)
        
        logger.debug(f"[INDICATORS] Must-keep features: OHLCV base ({len(ohlcv_features)}) + Technical ({len(ordered) - len(ohlcv_features)}) = {len(ordered)} total")
        return ordered
    
    def get_rfe_candidates(self) -> List[str]:
        """Get final list of RFE candidate feature names after calculation (OHLCV separation FIXED)"""
        # Only technical indicators can be RFE candidates - NEVER OHLCV base features
        technical_rfe_candidates = []
        
        for candidate in self.rfe_candidates:
            # Exclude OHLCV base features from RFE candidates completely
            if candidate.lower() not in [ohlcv.lower() for ohlcv in self.ohlcv_base_features]:
                technical_rfe_candidates.append(candidate)
        
        logger.debug(f"[INDICATORS] RFE candidates: {len(technical_rfe_candidates)} technical indicators (OHLCV excluded)")
        return technical_rfe_candidates
    
    def save_selected_features_file(self, selected_technical_features: List[str], model_version: str = "1.0"):
        """Save selected technical features to JSON file for model synchronization (NEW)"""
        try:
            import json
            from datetime import datetime
            
            feature_data = {
                "model_version": model_version,
                "timestamp": datetime.now().isoformat(),
                "ohlcv_base_features": self.ohlcv_base_features.copy(),
                "selected_technical_features": selected_technical_features.copy(),
                "total_features": len(self.ohlcv_base_features) + len(selected_technical_features),
                "rfe_config": {
                    "target_features": getattr(self, '_rfe_target_features', 30),
                    "selection_method": "RFE",
                    "impact_threshold": "100%"
                }
            }
            
            with open(self.feature_tracking_file, 'w') as f:
                json.dump(feature_data, f, indent=2)
            
            logger.success(f"[INDICATORS] Feature tracking file saved: {self.feature_tracking_file}")
            logger.info(f"[INDICATORS] Model will use: {len(self.ohlcv_base_features)} OHLCV + {len(selected_technical_features)} technical features")
            
        except Exception as e:
            logger.error(f"[INDICATORS] Failed to save feature tracking file: {e}")
    
    def load_selected_features_file(self) -> Dict[str, Any]:
        """Load selected features from JSON file for model synchronization (NEW)"""
        try:
            import json
            import os
            
            if not os.path.exists(self.feature_tracking_file):
                logger.warning(f"[INDICATORS] Feature tracking file not found: {self.feature_tracking_file}")
                return {}
            
            with open(self.feature_tracking_file, 'r') as f:
                feature_data = json.load(f)
            
            logger.success(f"[INDICATORS] Feature tracking file loaded: {self.feature_tracking_file}")
            logger.info(f"[INDICATORS] Model version: {feature_data.get('model_version', 'unknown')}")
            
            return feature_data
            
        except Exception as e:
            logger.error(f"[INDICATORS] Failed to load feature tracking file: {e}")
            return {}
    
    def get_skipped_indicators(self) -> List[Dict[str, str]]:
        """Get list of indicators that were skipped with reasons"""
        return self.skipped_indicators.copy()
    
    def get_indicators_summary(self) -> Dict[str, Any]:
        """Get comprehensive summary of indicators status (OHLCV separation FIXED)"""
        return {
            'total_defined': len(self.indicators_config),
            'must_keep_count': len(self.required_indicators),
            'rfe_candidate_count': len(self.rfe_eligible_indicators),
            'computed_count': len(self.computed_indicators),
            'skipped': self.get_skipped_indicators(),
            'ohlcv_base_features': self.ohlcv_base_features,
            'ohlcv_base_count': len(self.ohlcv_base_features),
            'must_keep_features': self.get_must_keep_features(),
            'rfe_candidates': self.get_rfe_candidates(),
            'technical_indicators_only_rfe': True  # Indicates OHLCV separation is active
        }
    
    def get_all_feature_names(self) -> List[str]:
        """Get all feature names (base OHLCV + computed indicators) - Phase 3 addition"""
        try:
            from config.settings import BASE_MUST_KEEP_FEATURES
            base_features = BASE_MUST_KEEP_FEATURES or ["open", "high", "low", "close", "volume"]
        except ImportError:
            base_features = ["open", "high", "low", "close", "volume"]
        
        # Combine base features with computed indicators
        all_features = base_features.copy()
        
        # Add all computed indicators (from successful calculations)
        for indicator_name in self.computed_indicators:
            if indicator_name not in all_features:
                all_features.append(indicator_name)
        
        # Also add timestamp and symbol_code if they exist
        additional_features = ["timestamp", "symbol_code"]
        for feat in additional_features:
            if feat not in all_features:
                all_features.append(feat)
        
        logger.debug(f"[INDICATORS] All feature names: {len(all_features)} features")
        return all_features
    
    def build_final_feature_sets(self, selected_features: List[str], must_keep: List[str], all_features: List[str]) -> Dict[str, List[str]]:
        """Build final active/inactive feature sets with proper deduplication (Phase 4)"""
        try:
            from config.settings import BASE_MUST_KEEP_FEATURES
            
            # Normalize feature names for case-insensitive comparison
            def normalize_name(name: str) -> str:
                """Normalize feature name for consistent comparison"""
                return str(name).lower().strip().replace(' ', '_')
            
            # Create normalized lookup sets
            selected_normalized = {normalize_name(f): f for f in selected_features}
            must_keep_normalized = {normalize_name(f): f for f in must_keep}
            all_normalized = {normalize_name(f): f for f in all_features}
            
            # Build active features (selected + must_keep, deduplicated)
            active_features = []
            seen_normalized = set()
            
            # Add selected features first
            for feature in selected_features:
                norm_name = normalize_name(feature)
                if norm_name not in seen_normalized and norm_name in all_normalized:
                    active_features.append(feature)
                    seen_normalized.add(norm_name)
            
            # Add must_keep features if not already included
            for feature in must_keep:
                norm_name = normalize_name(feature)
                if norm_name not in seen_normalized and norm_name in all_normalized:
                    active_features.append(feature)
                    seen_normalized.add(norm_name)
            
            # Build inactive features (all_features - active, deduplicated)
            inactive_features = []
            for feature in all_features:
                norm_name = normalize_name(feature)
                if norm_name not in seen_normalized:
                    inactive_features.append(feature)
            
            # Log the results for debugging
            logger.info(f"[INDICATORS] Final feature sets: {len(active_features)} active, {len(inactive_features)} inactive")
            logger.debug(f"[INDICATORS] Active features: {active_features[:10]}{'...' if len(active_features) > 10 else ''}")
            logger.debug(f"[INDICATORS] Inactive features: {inactive_features[:10]}{'...' if len(inactive_features) > 10 else ''}")
            
            return {
                'active': active_features,
                'inactive': inactive_features,
                'total_count': len(active_features) + len(inactive_features),
                'active_count': len(active_features),
                'inactive_count': len(inactive_features)
            }
            
        except Exception as e:
            logger.error(f"[INDICATORS] Error building final feature sets: {e}")
            # Return safe fallback
            return {
                'active': selected_features or must_keep or [],
                'inactive': [],
                'total_count': len(selected_features) if selected_features else len(must_keep) if must_keep else 0,
                'active_count': len(selected_features) if selected_features else len(must_keep) if must_keep else 0,
                'inactive_count': 0
            }
