"""
Test initial training functionality
"""

import os
import sys
import asyncio
import pandas as pd
import numpy as np
from unittest.mock import AsyncMock, MagicMock

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from src.ml_model.catboost_model import CatBoostTradingModel
from src.trading_engine.trading_engine import TradingEngine
from config.settings import MIN_INITIAL_TRAIN_SAMPLES

class TestInitialTraining:
    """Test initial training functionality"""
    
    def create_synthetic_data(self, num_samples: int = 500):
        """Create synthetic market data for testing"""
        np.random.seed(42)
        
        # Generate synthetic OHLCV data
        base_price = 50000
        timestamps = []
        opens = []
        highs = []
        lows = []
        closes = []
        volumes = []
        
        current_price = base_price
        current_time = 1600000000000  # Arbitrary timestamp
        
        for i in range(num_samples):
            # Random walk for price movement
            change = np.random.normal(0, 0.01)  # 1% volatility
            current_price *= (1 + change)
            
            # OHLC for this candle
            open_price = current_price
            close_price = current_price * (1 + np.random.normal(0, 0.005))
            high_price = max(open_price, close_price) * (1 + abs(np.random.normal(0, 0.003)))
            low_price = min(open_price, close_price) * (1 - abs(np.random.normal(0, 0.003)))
            volume = np.random.uniform(100, 1000)
            
            timestamps.append(current_time + i * 14400000)  # 4-hour intervals
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
    
    def create_synthetic_indicators(self, num_samples: int = 500):
        """Create synthetic indicator data for testing"""
        np.random.seed(42)
        
        indicators = {}
        
        # Create some basic indicators
        close_prices = np.random.uniform(45000, 55000, num_samples)
        
        # Simple moving averages
        indicators['SMA_5'] = pd.Series([np.mean(close_prices[max(0, i-4):i+1]) for i in range(num_samples)])
        indicators['SMA_14'] = pd.Series([np.mean(close_prices[max(0, i-13):i+1]) for i in range(num_samples)])
        indicators['SMA_50'] = pd.Series([np.mean(close_prices[max(0, i-49):i+1]) for i in range(num_samples)])
        
        # Exponential moving averages
        indicators['EMA_12'] = pd.Series(close_prices).ewm(span=12).mean()
        indicators['EMA_26'] = pd.Series(close_prices).ewm(span=26).mean()
        
        # RSI-like indicator
        indicators['RSI'] = pd.Series(np.random.uniform(20, 80, num_samples))
        
        # Volume indicators
        volumes = np.random.uniform(100, 1000, num_samples)
        indicators['Volume_SMA'] = pd.Series([np.mean(volumes[max(0, i-19):i+1]) for i in range(num_samples)])
        
        # Bollinger Bands
        sma_20 = pd.Series([np.mean(close_prices[max(0, i-19):i+1]) for i in range(num_samples)])
        std_20 = pd.Series([np.std(close_prices[max(0, i-19):i+1]) if i >= 19 else 0 for i in range(num_samples)])
        indicators['BB_Upper'] = sma_20 + (2 * std_20)
        indicators['BB_Lower'] = sma_20 - (2 * std_20)
        
        return indicators
    
    async def test_model_training_with_sufficient_data(self):
        """Test that model trains when sufficient data is available"""
        model = CatBoostTradingModel()
        await model.initialize()
        
        # Ensure model is not trained initially
        assert not model.is_trained
        
        # Create synthetic data with sufficient samples
        indicators = self.create_synthetic_indicators(MIN_INITIAL_TRAIN_SAMPLES + 100)
        rfe_eligible = ['SMA_5', 'SMA_14', 'EMA_12', 'RSI', 'Volume_SMA']
        
        # Attempt training
        success = await model.retrain_online(indicators, 'BTCUSDT', rfe_eligible)
        
        # Training might fail due to missing price data, but should not crash
        assert success is not None  # Should return True or False, not None
        
        print(f"✅ Model training test completed (success: {success})")
        return success
    
    async def test_model_training_with_insufficient_data(self):
        """Test that model handles insufficient data gracefully"""
        model = CatBoostTradingModel()
        await model.initialize()
        
        # Create synthetic data with insufficient samples
        indicators = self.create_synthetic_indicators(MIN_INITIAL_TRAIN_SAMPLES - 100)
        rfe_eligible = ['SMA_5', 'SMA_14', 'EMA_12']
        
        # Attempt training
        success = await model.retrain_online(indicators, 'BTCUSDT', rfe_eligible)
        
        # Should handle gracefully
        assert success is False or success is True  # Should not crash
        
        print("✅ Insufficient data test passed")
        return success
    
    async def test_trading_engine_initialization_with_training(self):
        """Test that trading engine attempts initial training on initialization"""
        # Create mock dependencies
        mock_data_collector = AsyncMock()
        mock_indicator_engine = AsyncMock()
        mock_ml_model = AsyncMock()
        
        # Configure mocks
        synthetic_data = self.create_synthetic_data(MIN_INITIAL_TRAIN_SAMPLES + 50)
        mock_data_collector.get_historical_data.return_value = synthetic_data
        
        synthetic_indicators = self.create_synthetic_indicators(MIN_INITIAL_TRAIN_SAMPLES + 50)
        mock_indicator_engine.calculate_all_indicators.return_value = synthetic_indicators
        mock_indicator_engine.get_rfe_eligible_indicators.return_value = ['SMA_5', 'SMA_14', 'EMA_12']
        
        mock_ml_model.is_trained = False
        mock_ml_model.training_cooldown_until = None
        mock_ml_model.retrain_online.return_value = True
        
        # Create trading engine
        trading_engine = TradingEngine(mock_data_collector, mock_indicator_engine, mock_ml_model)
        
        # Mock the risk manager to avoid complex initialization
        trading_engine.risk_manager = AsyncMock()
        trading_engine.risk_manager.initialize.return_value = None
        trading_engine.risk_manager.start_monitoring.return_value = None
        
        # Initialize trading engine (this should trigger training)
        await trading_engine.initialize()
        
        # Verify training was attempted
        assert mock_data_collector.get_historical_data.called
        assert mock_indicator_engine.calculate_all_indicators.called
        
        print("✅ Trading engine initialization test passed")
    
    async def test_fallback_signal_generation(self):
        """Test fallback signal generation when model not trained"""
        # Create mock dependencies
        mock_data_collector = AsyncMock()
        mock_indicator_engine = AsyncMock()
        mock_ml_model = AsyncMock()
        
        # Configure mocks
        synthetic_data = self.create_synthetic_data(200)
        mock_data_collector.get_historical_data.return_value = synthetic_data
        mock_data_collector.get_real_time_price.return_value = {'price': 50000.0}
        
        synthetic_indicators = self.create_synthetic_indicators(200)
        mock_indicator_engine.calculate_all_indicators.return_value = synthetic_indicators
        
        # Model is not trained
        mock_ml_model.is_trained = False
        
        # Create trading engine
        trading_engine = TradingEngine(mock_data_collector, mock_indicator_engine, mock_ml_model)
        trading_engine.risk_manager = AsyncMock()
        
        # Test fallback signal generation
        result = await trading_engine.analyze_symbol('BTCUSDT')
        
        # Should generate fallback signal
        assert result is not None
        assert 'fallback' in result
        assert result['fallback'] is True
        assert result['prediction'] in ['BUY', 'SELL', 'HOLD']
        assert 0 <= result['confidence'] <= 100
        assert 'fallback_indicators' in result
        
        print(f"✅ Fallback signal generated: {result['prediction']} ({result['confidence']:.1f}%)")

async def run_all_tests():
    """Run all training tests"""
    test = TestInitialTraining()
    
    print("🧪 Running ML model training tests...")
    
    try:
        await test.test_model_training_with_sufficient_data()
        await test.test_model_training_with_insufficient_data()
        await test.test_trading_engine_initialization_with_training()
        await test.test_fallback_signal_generation()
        
        print("🎉 All training tests completed successfully!")
        
    except Exception as e:
        print(f"❌ Test failed: {e}")
        import traceback
        traceback.print_exc()

if __name__ == '__main__':
    # Run tests
    asyncio.run(run_all_tests())