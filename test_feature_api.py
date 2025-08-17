#!/usr/bin/env python3
"""
Basic integration test for Phase 4 feature API fixes
Tests that endpoints return expected data structures and no RFE parameter conflicts
"""

import sys
import os
import asyncio
from unittest.mock import Mock, MagicMock
import pandas as pd
import numpy as np

# Add src to path for imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

def test_config_settings():
    """Test that new config settings are available"""
    try:
        from config.settings import (
            FEATURE_API_ENABLED, RFE_VERBOSE, SIGNAL_MIN_FEATURE_COVERAGE,
            HISTORICAL_FETCH_LIMIT, MAX_INACTIVE_FEATURE_LIST
        )
        print("✅ Config settings loaded successfully")
        print(f"  - FEATURE_API_ENABLED: {FEATURE_API_ENABLED}")
        print(f"  - RFE_VERBOSE: {RFE_VERBOSE}")
        print(f"  - SIGNAL_MIN_FEATURE_COVERAGE: {SIGNAL_MIN_FEATURE_COVERAGE}")
        return True
    except Exception as e:
        print(f"❌ Config settings test failed: {e}")
        return False

def test_catboost_model_initialization():
    """Test CatBoost model can be initialized without parameter conflicts"""
    try:
        from src.ml_model.catboost_model import CatBoostTradingModel
        
        # Create mock dependencies to avoid database connection issues
        sys.modules['src.database.db_manager'] = Mock()
        sys.modules['src.database.db_manager'].db_manager = Mock()
        
        model = CatBoostTradingModel()
        print("✅ CatBoost model created successfully")
        
        # Test get_feature_breakdown method exists
        if hasattr(model, 'get_feature_breakdown'):
            print("✅ get_feature_breakdown method available")
        else:
            print("❌ get_feature_breakdown method missing")
            return False
            
        return True
    except Exception as e:
        print(f"❌ CatBoost model test failed: {e}")
        return False

def test_indicator_engine_feature_sets():
    """Test indicator engine build_final_feature_sets method"""
    try:
        from src.indicators.indicator_engine import IndicatorEngine
        
        engine = IndicatorEngine()
        
        # Test data
        selected_features = ['open', 'high', 'low', 'close', 'volume', 'rsi_14']
        must_keep = ['open', 'high', 'low', 'close', 'volume']
        all_features = ['open', 'high', 'low', 'close', 'volume', 'rsi_14', 'sma_20', 'ema_12']
        
        # Test build_final_feature_sets
        result = engine.build_final_feature_sets(selected_features, must_keep, all_features)
        
        print("✅ build_final_feature_sets method works")
        print(f"  - Active features: {len(result['active'])}")
        print(f"  - Inactive features: {len(result['inactive'])}")
        
        # Validate no duplicates between active and inactive
        active_set = set(result['active'])
        inactive_set = set(result['inactive'])
        duplicates = active_set.intersection(inactive_set)
        
        if not duplicates:
            print("✅ No duplicates between active and inactive features")
        else:
            print(f"❌ Found duplicates: {duplicates}")
            return False
            
        return True
    except Exception as e:
        print(f"❌ Indicator engine test failed: {e}")
        return False

def test_trading_engine_fetch_candles():
    """Test trading engine fetch_historical_candles method"""
    try:
        from src.trading_engine.trading_engine import TradingEngine
        
        # Create mock dependencies
        mock_data_collector = Mock()
        mock_indicator_engine = Mock()
        mock_ml_model = Mock()
        
        engine = TradingEngine(mock_data_collector, mock_indicator_engine, mock_ml_model)
        
        # Test method exists
        if hasattr(engine, 'fetch_historical_candles'):
            print("✅ fetch_historical_candles method available")
        else:
            print("❌ fetch_historical_candles method missing")
            return False
            
        return True
    except Exception as e:
        print(f"❌ Trading engine test failed: {e}")
        return False

def test_web_app_features_endpoint():
    """Test web app /api/features endpoint structure"""
    try:
        from src.web_app.app import create_app
        
        # Create mock components
        mock_ml_model = Mock()
        mock_ml_model.get_feature_breakdown.return_value = {
            'selected': [{'name': 'close', 'importance': 0.5, 'status': 'selected'}],
            'inactive': [{'name': 'sma_20', 'importance': 0.1, 'status': 'inactive'}],
            'must_keep': ['close'],
            'selection_method': 'RFE',
            'timestamp': '2024-01-01T00:00:00',
            'metadata': {
                'total_selected': 1,
                'total_inactive': 1,
                'total_features': 2
            }
        }
        
        app = create_app(ml_model=mock_ml_model)
        
        with app.test_client() as client:
            response = client.get('/api/features')
            data = response.get_json()
            
            if response.status_code == 200:
                print("✅ /api/features endpoint responds successfully")
                print(f"  - Selected features: {len(data.get('selected', []))}")
                print(f"  - Selection method: {data.get('selection_method', 'unknown')}")
                return True
            else:
                print(f"❌ /api/features endpoint failed: {response.status_code}")
                return False
                
    except Exception as e:
        print(f"❌ Web app test failed: {e}")
        return False

def run_all_tests():
    """Run all tests and report results"""
    print("🧪 Running Phase 4 Feature API Tests\n")
    
    tests = [
        ("Config Settings", test_config_settings),
        ("CatBoost Model", test_catboost_model_initialization),
        ("Indicator Engine", test_indicator_engine_feature_sets),
        ("Trading Engine", test_trading_engine_fetch_candles),
        ("Web App API", test_web_app_features_endpoint),
    ]
    
    results = []
    for test_name, test_func in tests:
        print(f"\n📋 Testing {test_name}...")
        try:
            success = test_func()
            results.append((test_name, success))
        except Exception as e:
            print(f"❌ {test_name} test crashed: {e}")
            results.append((test_name, False))
    
    # Summary
    print("\n" + "="*50)
    print("📊 TEST SUMMARY")
    print("="*50)
    
    passed = 0
    for test_name, success in results:
        status = "✅ PASS" if success else "❌ FAIL"
        print(f"{status} {test_name}")
        if success:
            passed += 1
    
    total = len(results)
    print(f"\nResult: {passed}/{total} tests passed")
    
    if passed == total:
        print("🎉 All tests passed! Phase 4 fixes are working correctly.")
        return True
    else:
        print("⚠️  Some tests failed. Please review the implementation.")
        return False

if __name__ == "__main__":
    success = run_all_tests()
    sys.exit(0 if success else 1)