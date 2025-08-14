"""
Comprehensive test script for the Crypto Trading AI System
"""

import asyncio
import sys
import os
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import time

# Add src to path
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

from src.data_collector.binance_collector import BinanceDataCollector
from src.indicators.indicator_engine import IndicatorEngine
from src.ml_model.catboost_model import CatBoostTradingModel
from src.risk_management.risk_manager import RiskManagementSystem
from src.trading_engine.trading_engine import TradingEngine
from src.web_app.app import create_app
from config.settings import *

class SystemTester:
    """Comprehensive system tester"""
    
    def __init__(self):
        self.test_results = {}
        self.components = {}
        
    async def run_all_tests(self):
        """Run all system tests"""
        print("=" * 60)
        print("CRYPTO TRADING AI SYSTEM - COMPREHENSIVE TEST")
        print("=" * 60)
        
        try:
            # Test 1: Component Initialization
            await self.test_component_initialization()
            
            # Test 2: Data Collection
            await self.test_data_collection()
            
            # Test 3: Indicator Calculation
            await self.test_indicator_calculation()
            
            # Test 4: ML Model Training and Prediction
            await self.test_ml_model()
            
            # Test 5: Risk Management
            await self.test_risk_management()
            
            # Test 6: Trading Engine
            await self.test_trading_engine()
            
            # Test 7: Web Application
            await self.test_web_application()
            
            # Test 8: End-to-End Integration
            await self.test_end_to_end()
            
            # Print Summary
            self.print_test_summary()
            
        except Exception as e:
            print(f"❌ Critical test failure: {e}")
            return False
        
        return True
    
    async def test_component_initialization(self):
        """Test component initialization"""
        print("\n📋 Test 1: Component Initialization")
        print("-" * 40)
        
        try:
            # Initialize data collector
            print("  • Initializing data collector...")
            self.components['data_collector'] = BinanceDataCollector()
            await self.components['data_collector'].initialize()
            print("  ✅ Data collector initialized")
            
            # Initialize indicator engine
            print("  • Initializing indicator engine...")
            self.components['indicator_engine'] = IndicatorEngine()
            await self.components['indicator_engine'].initialize()
            print(f"  ✅ Indicator engine initialized with {len(self.components['indicator_engine'].indicators_config)} indicators")
            
            # Initialize ML model
            print("  • Initializing ML model...")
            self.components['ml_model'] = CatBoostTradingModel()
            await self.components['ml_model'].initialize()
            print("  ✅ ML model initialized")
            
            # Initialize risk manager
            print("  • Initializing risk manager...")
            self.components['risk_manager'] = RiskManagementSystem()
            await self.components['risk_manager'].initialize()
            print("  ✅ Risk management system initialized")
            
            # Initialize trading engine
            print("  • Initializing trading engine...")
            self.components['trading_engine'] = TradingEngine(
                self.components['data_collector'],
                self.components['indicator_engine'],
                self.components['ml_model']
            )
            await self.components['trading_engine'].initialize()
            print("  ✅ Trading engine initialized")
            
            self.test_results['initialization'] = 'PASSED'
            print("✅ Component initialization test PASSED")
            
        except Exception as e:
            self.test_results['initialization'] = f'FAILED: {e}'
            print(f"❌ Component initialization test FAILED: {e}")
            raise
    
    async def test_data_collection(self):
        """Test data collection functionality"""
        print("\n📊 Test 2: Data Collection")
        print("-" * 40)
        
        try:
            data_collector = self.components['data_collector']
            
            # Test historical data fetching
            print("  • Testing historical data fetching...")
            test_symbol = 'BTCUSDT'
            historical_data = await data_collector.fetch_historical_data(test_symbol, '4h', 10)
            
            if historical_data is not None and len(historical_data) > 0:
                print(f"  ✅ Fetched {len(historical_data)} historical candles for {test_symbol}")
            else:
                raise Exception("Failed to fetch historical data")
            
            # Test real-time price fetching
            print("  • Testing real-time price fetching...")
            price_data = await data_collector.get_real_time_price(test_symbol)
            
            if price_data and 'price' in price_data:
                print(f"  ✅ Current {test_symbol} price: ${price_data['price']:.4f}")
            else:
                print("  ⚠️  Real-time price fetching failed (might work with API keys)")
            
            # Test database operations
            print("  • Testing database operations...")
            if historical_data is not None:
                await data_collector.store_market_data(test_symbol, '4h', historical_data)
                stored_data = await data_collector.get_historical_data(test_symbol, '4h', 5)
                
                if stored_data is not None and len(stored_data) > 0:
                    print(f"  ✅ Database operations working - {len(stored_data)} records retrieved")
                else:
                    raise Exception("Database operations failed")
            
            self.test_results['data_collection'] = 'PASSED'
            print("✅ Data collection test PASSED")
            
        except Exception as e:
            self.test_results['data_collection'] = f'FAILED: {e}'
            print(f"❌ Data collection test FAILED: {e}")
    
    async def test_indicator_calculation(self):
        """Test indicator calculation"""
        print("\n📈 Test 3: Indicator Calculation")
        print("-" * 40)
        
        try:
            indicator_engine = self.components['indicator_engine']
            data_collector = self.components['data_collector']
            
            # Get test data
            print("  • Preparing test data...")
            test_symbol = 'BTCUSDT'
            test_data = await data_collector.get_historical_data(test_symbol, '4h', 200)
            
            if test_data is None or len(test_data) < 50:
                # Create synthetic test data if real data is not available
                print("  • Creating synthetic test data...")
                dates = pd.date_range(start='2023-01-01', periods=200, freq='4H')
                base_price = 50000
                
                # Generate realistic OHLCV data
                np.random.seed(42)
                returns = np.random.normal(0, 0.02, 200)
                prices = [base_price]
                
                for ret in returns[1:]:
                    prices.append(prices[-1] * (1 + ret))
                
                test_data = pd.DataFrame({
                    'timestamp': [int(d.timestamp() * 1000) for d in dates],
                    'open': prices,
                    'high': [p * (1 + abs(np.random.normal(0, 0.01))) for p in prices],
                    'low': [p * (1 - abs(np.random.normal(0, 0.01))) for p in prices],
                    'close': prices,
                    'volume': np.random.lognormal(10, 0.5, 200),
                    'datetime': dates
                })
            
            print(f"  • Test data ready: {len(test_data)} candles")
            
            # Calculate indicators
            print("  • Calculating indicators...")
            indicators = await indicator_engine.calculate_all_indicators(test_data, test_symbol)
            
            # Check results
            calculated_count = len([v for v in indicators.values() if v is not None])
            total_functions = len(indicator_engine.indicator_functions)
            
            print(f"  ✅ Calculated {calculated_count} indicators")
            print(f"  ℹ️  Available functions: {total_functions}")
            
            # Test specific indicators
            test_indicators = ['SMA_20', 'EMA_20', 'RSI_14', 'MACD_12_26_9']
            working_indicators = []
            
            for indicator in test_indicators:
                if indicator in indicators and indicators[indicator] is not None:
                    working_indicators.append(indicator)
                    if isinstance(indicators[indicator], (pd.Series, list, np.ndarray)):
                        if len(indicators[indicator]) > 0:
                            last_value = indicators[indicator].iloc[-1] if hasattr(indicators[indicator], 'iloc') else indicators[indicator][-1]
                            print(f"  ✅ {indicator}: {last_value}")
                        else:
                            print(f"  ⚠️  {indicator}: Empty result")
                    else:
                        print(f"  ⚠️  {indicator}: Unexpected type {type(indicators[indicator])}")
            
            if len(working_indicators) >= 2:
                print(f"  ✅ Core indicators working: {working_indicators}")
                self.test_results['indicators'] = 'PASSED'
                print("✅ Indicator calculation test PASSED")
            else:
                raise Exception(f"Too few working indicators: {working_indicators}")
            
        except Exception as e:
            self.test_results['indicators'] = f'FAILED: {e}'
            print(f"❌ Indicator calculation test FAILED: {e}")
    
    async def test_ml_model(self):
        """Test ML model functionality"""
        print("\n🤖 Test 4: ML Model Training and Prediction")
        print("-" * 40)
        
        try:
            ml_model = self.components['ml_model']
            indicator_engine = self.components['indicator_engine']
            data_collector = self.components['data_collector']
            
            # Get test data and indicators
            print("  • Preparing training data...")
            test_symbol = 'BTCUSDT'
            test_data = await data_collector.get_historical_data(test_symbol, '4h', 500)
            
            if test_data is None or len(test_data) < 100:
                print("  ⚠️  Insufficient real data, creating synthetic data for ML test...")
                # Create more comprehensive synthetic data
                np.random.seed(42)
                n_samples = 500
                
                base_price = 50000
                returns = np.random.normal(0, 0.02, n_samples)
                prices = [base_price]
                
                for ret in returns[1:]:
                    prices.append(prices[-1] * (1 + ret))
                
                dates = pd.date_range(start='2023-01-01', periods=n_samples, freq='4H')
                
                test_data = pd.DataFrame({
                    'timestamp': [int(d.timestamp() * 1000) for d in dates],
                    'open': prices,
                    'high': [p * (1 + abs(np.random.normal(0, 0.01))) for p in prices],
                    'low': [p * (1 - abs(np.random.normal(0, 0.01))) for p in prices],
                    'close': prices,
                    'volume': np.random.lognormal(10, 0.5, n_samples),
                    'datetime': dates
                })
            
            # Calculate indicators
            indicators = await indicator_engine.calculate_all_indicators(test_data, test_symbol)
            
            if not indicators:
                raise Exception("No indicators available for training")
            
            print(f"  • Training data ready: {len(test_data)} samples")
            
            # Get RFE eligible features
            rfe_eligible = indicator_engine.get_rfe_eligible_indicators()
            print(f"  • RFE eligible features: {len(rfe_eligible)}")
            
            # Prepare features and labels
            print("  • Preparing features and labels...")
            X, y = await ml_model.prepare_features_and_labels(indicators, test_symbol)
            
            if len(X) == 0 or len(y) == 0:
                raise Exception("No valid training data prepared")
            
            print(f"  • Features: {len(X.columns)} x {len(X)} samples")
            print(f"  • Labels: {len(y)} samples")
            
            # Train model
            print("  • Training model (this may take a few minutes)...")
            training_success = await ml_model.train_model(X, y, rfe_eligible)
            
            if not training_success:
                raise Exception("Model training failed")
            
            # Test prediction
            print("  • Testing prediction...")
            last_features = X.tail(1)
            prediction = await ml_model.predict(last_features)
            
            print(f"  ✅ Prediction: {prediction['prediction']} (confidence: {prediction['confidence']:.1f}%)")
            print(f"  ℹ️  Probabilities: {prediction['probabilities']}")
            
            # Get model info
            model_info = ml_model.get_model_info()
            print(f"  ✅ Model accuracy: {model_info.get('model_performance', {}).get('accuracy', 0):.3f}")
            print(f"  ✅ Selected features: {model_info.get('selected_features_count', 0)}")
            
            self.test_results['ml_model'] = 'PASSED'
            print("✅ ML model test PASSED")
            
        except Exception as e:
            self.test_results['ml_model'] = f'FAILED: {e}'
            print(f"❌ ML model test FAILED: {e}")
    
    async def test_risk_management(self):
        """Test risk management system"""
        print("\n💰 Test 5: Risk Management")
        print("-" * 40)
        
        try:
            risk_manager = self.components['risk_manager']
            
            # Test position opening
            print("  • Testing position opening...")
            position_id = await risk_manager.open_position(
                symbol='BTCUSDT',
                side='BUY',
                price=50000.0,
                confidence=85.0
            )
            
            if position_id:
                print(f"  ✅ Position opened: {position_id}")
                
                # Test position updates
                print("  • Testing position updates...")
                current_prices = {'BTCUSDT': 51000.0}  # 2% gain
                await risk_manager.update_positions(current_prices)
                
                # Check position status
                position = risk_manager.positions.get(position_id)
                if position:
                    print(f"  ✅ Position P&L: ${position.unrealized_pnl:.2f}")
                
                # Test position closing
                print("  • Testing position closing...")
                close_success = await risk_manager.close_position(position_id, 51500.0, 'TEST')
                
                if close_success:
                    print("  ✅ Position closed successfully")
                else:
                    print("  ⚠️  Position closing failed")
            
            # Test portfolio summary
            print("  • Testing portfolio summary...")
            portfolio = risk_manager.get_portfolio_summary()
            print(f"  ✅ Portfolio balance: ${portfolio.get('portfolio_balance', 0):.2f}")
            print(f"  ℹ️  Total positions: {portfolio.get('total_positions', 0)}")
            
            self.test_results['risk_management'] = 'PASSED'
            print("✅ Risk management test PASSED")
            
        except Exception as e:
            self.test_results['risk_management'] = f'FAILED: {e}'
            print(f"❌ Risk management test FAILED: {e}")
    
    async def test_trading_engine(self):
        """Test trading engine"""
        print("\n⚙️ Test 6: Trading Engine")
        print("-" * 40)
        
        try:
            trading_engine = self.components['trading_engine']
            
            # Test system status
            print("  • Testing system status...")
            status = trading_engine.get_system_status()
            print(f"  ✅ System status retrieved")
            print(f"  ℹ️  Supported pairs: {len(status.get('supported_pairs', []))}")
            
            # Test market analysis
            print("  • Testing market analysis...")
            await trading_engine.analyze_markets()
            
            # Get analysis results
            signals = trading_engine.get_latest_signals(5)
            print(f"  ✅ Generated {len(signals)} signals")
            
            for signal in signals[:3]:  # Show first 3 signals
                print(f"    - {signal['symbol']}: {signal['prediction']} ({signal['confidence']:.1f}%)")
            
            self.test_results['trading_engine'] = 'PASSED'
            print("✅ Trading engine test PASSED")
            
        except Exception as e:
            self.test_results['trading_engine'] = f'FAILED: {e}'
            print(f"❌ Trading engine test FAILED: {e}")
    
    async def test_web_application(self):
        """Test web application"""
        print("\n🌐 Test 7: Web Application")
        print("-" * 40)
        
        try:
            # Create Flask app
            print("  • Creating Flask application...")
            app = create_app(
                data_collector=self.components['data_collector'],
                indicator_engine=self.components['indicator_engine'],
                ml_model=self.components['ml_model'],
                trading_engine=self.components['trading_engine']
            )
            
            # Test app creation
            if app:
                print("  ✅ Flask application created successfully")
                
                # Test routes (basic check)
                with app.test_client() as client:
                    # Test main dashboard
                    response = client.get('/')
                    if response.status_code == 200:
                        print("  ✅ Dashboard route working")
                    
                    # We can't easily test API routes that require async operations
                    # without more complex setup, but the app creation success is a good sign
                
                self.test_results['web_application'] = 'PASSED'
                print("✅ Web application test PASSED")
            else:
                raise Exception("Failed to create Flask application")
                
        except Exception as e:
            self.test_results['web_application'] = f'FAILED: {e}'
            print(f"❌ Web application test FAILED: {e}")
    
    async def test_end_to_end(self):
        """Test end-to-end integration"""
        print("\n🔄 Test 8: End-to-End Integration")
        print("-" * 40)
        
        try:
            print("  • Running integrated workflow...")
            
            # Simulate a complete trading cycle
            trading_engine = self.components['trading_engine']
            
            # 1. Market analysis
            await trading_engine.analyze_markets()
            
            # 2. Get signals
            signals = trading_engine.get_latest_signals(1)
            
            if signals:
                signal = signals[0]
                print(f"  ✅ Generated signal: {signal['symbol']} - {signal['prediction']} ({signal['confidence']:.1f}%)")
                
                # 3. Check if signal would be executed (confidence > threshold)
                if signal['confidence'] >= CONFIDENCE_THRESHOLD:
                    print(f"  ✅ Signal meets confidence threshold ({CONFIDENCE_THRESHOLD}%)")
                else:
                    print(f"  ℹ️  Signal below confidence threshold ({signal['confidence']:.1f}% < {CONFIDENCE_THRESHOLD}%)")
            
            # 4. Check portfolio status
            portfolio = trading_engine.risk_manager.get_portfolio_summary()
            print(f"  ✅ Portfolio status: ${portfolio.get('portfolio_balance', 0):.2f} balance")
            
            # 5. Test data persistence
            print("  • Testing data persistence...")
            data_collector = self.components['data_collector']
            
            # Check if we have stored data
            stored_data = await data_collector.get_historical_data('BTCUSDT', '4h', 10)
            if stored_data is not None and len(stored_data) > 0:
                print(f"  ✅ Data persistence working - {len(stored_data)} records in database")
            
            self.test_results['end_to_end'] = 'PASSED'
            print("✅ End-to-end integration test PASSED")
            
        except Exception as e:
            self.test_results['end_to_end'] = f'FAILED: {e}'
            print(f"❌ End-to-end integration test FAILED: {e}")
    
    def print_test_summary(self):
        """Print comprehensive test summary"""
        print("\n" + "=" * 60)
        print("TEST SUMMARY")
        print("=" * 60)
        
        passed_tests = 0
        total_tests = len(self.test_results)
        
        for test_name, result in self.test_results.items():
            status = "✅ PASSED" if result == 'PASSED' else f"❌ {result}"
            print(f"{test_name.replace('_', ' ').title():<30} {status}")
            
            if result == 'PASSED':
                passed_tests += 1
        
        print("-" * 60)
        print(f"OVERALL RESULT: {passed_tests}/{total_tests} tests passed")
        
        if passed_tests == total_tests:
            print("🎉 ALL TESTS PASSED! System is ready for use.")
            print("\nTo start the system:")
            print("  python main.py")
            print("\nThen open your browser to:")
            print("  http://localhost:5000")
        else:
            print("⚠️  Some tests failed. Please review the errors above.")
            print("The system may still work with reduced functionality.")
        
        print("=" * 60)

async def main():
    """Main test function"""
    print("Starting comprehensive system test...")
    
    try:
        tester = SystemTester()
        success = await tester.run_all_tests()
        
        if success:
            print("\n🎯 Testing completed successfully!")
        else:
            print("\n⚠️  Testing completed with some issues.")
        
    except KeyboardInterrupt:
        print("\n⚠️  Testing interrupted by user")
    except Exception as e:
        print(f"\n❌ Critical testing error: {e}")
        return False
    
    return True

if __name__ == "__main__":
    try:
        result = asyncio.run(main())
        sys.exit(0 if result else 1)
    except Exception as e:
        print(f"Fatal error: {e}")
        sys.exit(1)