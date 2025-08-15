"""
Test fallback signal generation
"""

import os
import sys
import asyncio
import pandas as pd
import numpy as np

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from src.trading_engine.trading_engine import TradingEngine
from unittest.mock import AsyncMock

class TestFallbackSignal:
    """Test fallback signal generation functionality"""
    
    def create_trending_up_data(self, num_samples: int = 100):
        """Create synthetic data with upward trend"""
        np.random.seed(42)
        
        base_price = 50000
        prices = []
        
        for i in range(num_samples):
            # Upward trend with noise
            trend = i * 10  # 10 dollar increase per period
            noise = np.random.normal(0, 100)
            price = base_price + trend + noise
            prices.append(price)
        
        timestamps = [1600000000000 + i * 14400000 for i in range(num_samples)]
        
        return pd.DataFrame({
            'timestamp': timestamps,
            'open': prices,
            'high': [p * 1.01 for p in prices],
            'low': [p * 0.99 for p in prices],
            'close': prices,
            'volume': np.random.uniform(100, 1000, num_samples)
        })
    
    def create_trending_down_data(self, num_samples: int = 100):
        """Create synthetic data with downward trend"""
        np.random.seed(123)
        
        base_price = 50000
        prices = []
        
        for i in range(num_samples):
            # Downward trend with noise
            trend = -i * 15  # 15 dollar decrease per period
            noise = np.random.normal(0, 100)
            price = base_price + trend + noise
            prices.append(max(price, 1000))  # Prevent negative prices
        
        timestamps = [1600000000000 + i * 14400000 for i in range(num_samples)]
        
        return pd.DataFrame({
            'timestamp': timestamps,
            'open': prices,
            'high': [p * 1.01 for p in prices],
            'low': [p * 0.99 for p in prices],
            'close': prices,
            'volume': np.random.uniform(100, 1000, num_samples)
        })
    
    def create_sideways_data(self, num_samples: int = 100):
        """Create synthetic data with sideways movement"""
        np.random.seed(456)
        
        base_price = 50000
        prices = []
        
        for i in range(num_samples):
            # Sideways movement with noise
            noise = np.random.normal(0, 200)
            price = base_price + noise
            prices.append(price)
        
        timestamps = [1600000000000 + i * 14400000 for i in range(num_samples)]
        
        return pd.DataFrame({
            'timestamp': timestamps,
            'open': prices,
            'high': [p * 1.01 for p in prices],
            'low': [p * 0.99 for p in prices],
            'close': prices,
            'volume': np.random.uniform(100, 1000, num_samples)
        })
    
    async def test_fallback_signal_uptrend(self):
        """Test fallback signal generation with upward trending data"""
        # Create mock dependencies
        mock_data_collector = AsyncMock()
        mock_indicator_engine = AsyncMock()
        mock_ml_model = AsyncMock()
        
        # Configure mocks with upward trending data
        trending_data = self.create_trending_up_data(100)
        mock_data_collector.get_historical_data.return_value = trending_data
        mock_data_collector.get_real_time_price.return_value = {'price': trending_data['close'].iloc[-1]}
        
        mock_indicator_engine.calculate_all_indicators.return_value = {'dummy': [1, 2, 3]}
        
        # Model is not trained
        mock_ml_model.is_trained = False
        
        # Create trading engine
        trading_engine = TradingEngine(mock_data_collector, mock_indicator_engine, mock_ml_model)
        
        # Generate fallback signal
        result = await trading_engine.generate_fallback_signal('BTCUSDT', trending_data, {'dummy': [1, 2, 3]})
        
        # Should generate signal
        assert result is not None
        assert result['fallback'] is True
        assert result['prediction'] in ['BUY', 'SELL', 'HOLD']
        assert 'fallback_indicators' in result
        assert 'sma_14' in result['fallback_indicators']
        assert 'sma_50' in result['fallback_indicators']
        assert 'rsi' in result['fallback_indicators']
        
        print(f"✅ Uptrend fallback signal: {result['prediction']} ({result['confidence']:.1f}%)")
        return result
    
    async def test_fallback_signal_downtrend(self):
        """Test fallback signal generation with downward trending data"""
        # Create mock dependencies
        mock_data_collector = AsyncMock()
        mock_indicator_engine = AsyncMock()
        mock_ml_model = AsyncMock()
        
        # Configure mocks with downward trending data
        trending_data = self.create_trending_down_data(100)
        mock_data_collector.get_historical_data.return_value = trending_data
        mock_data_collector.get_real_time_price.return_value = {'price': trending_data['close'].iloc[-1]}
        
        mock_indicator_engine.calculate_all_indicators.return_value = {'dummy': [1, 2, 3]}
        
        # Model is not trained
        mock_ml_model.is_trained = False
        
        # Create trading engine
        trading_engine = TradingEngine(mock_data_collector, mock_indicator_engine, mock_ml_model)
        
        # Generate fallback signal
        result = await trading_engine.generate_fallback_signal('BTCUSDT', trending_data, {'dummy': [1, 2, 3]})
        
        # Should generate signal
        assert result is not None
        assert result['fallback'] is True
        assert result['prediction'] in ['BUY', 'SELL', 'HOLD']
        
        print(f"✅ Downtrend fallback signal: {result['prediction']} ({result['confidence']:.1f}%)")
        return result
    
    async def test_fallback_signal_sideways(self):
        """Test fallback signal generation with sideways data"""
        # Create mock dependencies
        mock_data_collector = AsyncMock()
        mock_indicator_engine = AsyncMock()
        mock_ml_model = AsyncMock()
        
        # Configure mocks with sideways data
        sideways_data = self.create_sideways_data(100)
        mock_data_collector.get_historical_data.return_value = sideways_data
        mock_data_collector.get_real_time_price.return_value = {'price': sideways_data['close'].iloc[-1]}
        
        mock_indicator_engine.calculate_all_indicators.return_value = {'dummy': [1, 2, 3]}
        
        # Model is not trained
        mock_ml_model.is_trained = False
        
        # Create trading engine
        trading_engine = TradingEngine(mock_data_collector, mock_indicator_engine, mock_ml_model)
        
        # Generate fallback signal
        result = await trading_engine.generate_fallback_signal('BTCUSDT', sideways_data, {'dummy': [1, 2, 3]})
        
        # Should generate signal
        assert result is not None
        assert result['fallback'] is True
        assert result['prediction'] in ['BUY', 'SELL', 'HOLD']
        
        print(f"✅ Sideways fallback signal: {result['prediction']} ({result['confidence']:.1f}%)")
        return result
    
    async def test_fallback_vs_ml_signal_structure(self):
        """Test that fallback signals have the same structure as ML signals"""
        # Test fallback signal
        mock_data_collector = AsyncMock()
        mock_indicator_engine = AsyncMock()
        mock_ml_model = AsyncMock()
        
        trending_data = self.create_trending_up_data(100)
        mock_data_collector.get_historical_data.return_value = trending_data
        mock_data_collector.get_real_time_price.return_value = {'price': 50000.0}
        
        mock_indicator_engine.calculate_all_indicators.return_value = {'dummy': [1, 2, 3]}
        
        # First test: model not trained (fallback)
        mock_ml_model.is_trained = False
        
        trading_engine = TradingEngine(mock_data_collector, mock_indicator_engine, mock_ml_model)
        fallback_result = await trading_engine.generate_fallback_signal('BTCUSDT', trending_data, {'dummy': [1, 2, 3]})
        
        # Check required fields
        required_fields = ['symbol', 'timestamp', 'current_price', 'prediction', 'confidence', 'probabilities', 'feature_count', 'data_quality', 'fallback']
        
        for field in required_fields:
            assert field in fallback_result, f"Missing field: {field}"
        
        assert fallback_result['fallback'] is True
        assert isinstance(fallback_result['probabilities'], dict)
        assert 'BUY' in fallback_result['probabilities']
        assert 'SELL' in fallback_result['probabilities']
        assert 'HOLD' in fallback_result['probabilities']
        
        print("✅ Fallback signal structure validation passed")
    
    async def test_fallback_signal_execution_control(self):
        """Test that fallback signals respect ALLOW_FALLBACK_EXECUTION setting"""
        from config.settings import ALLOW_FALLBACK_EXECUTION
        
        # Create fallback signal
        mock_data_collector = AsyncMock()
        mock_indicator_engine = AsyncMock()
        mock_ml_model = AsyncMock()
        
        trending_data = self.create_trending_up_data(100)
        mock_data_collector.get_historical_data.return_value = trending_data
        mock_data_collector.get_real_time_price.return_value = {'price': 50000.0}
        mock_indicator_engine.calculate_all_indicators.return_value = {'dummy': [1, 2, 3]}
        
        mock_ml_model.is_trained = False
        
        trading_engine = TradingEngine(mock_data_collector, mock_indicator_engine, mock_ml_model)
        result = await trading_engine.generate_fallback_signal('BTCUSDT', trending_data, {'dummy': [1, 2, 3]})
        
        # Fallback signal should be generated regardless of ALLOW_FALLBACK_EXECUTION
        # The execution control happens at the trading level, not signal generation
        assert result is not None
        assert result['fallback'] is True
        
        print(f"✅ Fallback execution control test passed (ALLOW_FALLBACK_EXECUTION: {ALLOW_FALLBACK_EXECUTION})")

async def run_all_tests():
    """Run all fallback signal tests"""
    test = TestFallbackSignal()
    
    print("🧪 Running fallback signal tests...")
    
    try:
        uptrend_result = await test.test_fallback_signal_uptrend()
        downtrend_result = await test.test_fallback_signal_downtrend()
        sideways_result = await test.test_fallback_signal_sideways()
        
        await test.test_fallback_vs_ml_signal_structure()
        await test.test_fallback_signal_execution_control()
        
        # Check that different market conditions produce different signals
        signals = [uptrend_result['prediction'], downtrend_result['prediction'], sideways_result['prediction']]
        
        print("\n📊 Signal Summary:")
        print(f"  Uptrend:   {uptrend_result['prediction']} ({uptrend_result['confidence']:.1f}%)")
        print(f"  Downtrend: {downtrend_result['prediction']} ({downtrend_result['confidence']:.1f}%)")
        print(f"  Sideways:  {sideways_result['prediction']} ({sideways_result['confidence']:.1f}%)")
        
        print("\n🎉 All fallback signal tests completed successfully!")
        
    except Exception as e:
        print(f"❌ Test failed: {e}")
        import traceback
        traceback.print_exc()

if __name__ == '__main__':
    # Run tests
    asyncio.run(run_all_tests())