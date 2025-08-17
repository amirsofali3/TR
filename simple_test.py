#!/usr/bin/env python3
"""
Simple test for Phase 4 core functionality without external dependencies
"""

import sys
import os

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

def test_config_updates():
    """Test that config has new Phase 4 settings"""
    try:
        from config.settings import (
            FEATURE_API_ENABLED, RFE_VERBOSE, SIGNAL_MIN_FEATURE_COVERAGE,
            HISTORICAL_FETCH_LIMIT, MAX_INACTIVE_FEATURE_LIST, BASE_MUST_KEEP_FEATURES
        )
        
        print("✅ All new config settings loaded:")
        print(f"  FEATURE_API_ENABLED = {FEATURE_API_ENABLED}")
        print(f"  RFE_VERBOSE = {RFE_VERBOSE}")
        print(f"  SIGNAL_MIN_FEATURE_COVERAGE = {SIGNAL_MIN_FEATURE_COVERAGE}")
        print(f"  HISTORICAL_FETCH_LIMIT = {HISTORICAL_FETCH_LIMIT}")
        print(f"  MAX_INACTIVE_FEATURE_LIST = {MAX_INACTIVE_FEATURE_LIST}")
        print(f"  BASE_MUST_KEEP_FEATURES = {BASE_MUST_KEEP_FEATURES}")
        
        # Validate values
        assert FEATURE_API_ENABLED == True
        assert RFE_VERBOSE == False
        assert SIGNAL_MIN_FEATURE_COVERAGE == 0.9
        assert HISTORICAL_FETCH_LIMIT == 500
        assert MAX_INACTIVE_FEATURE_LIST == 50
        
        return True
    except Exception as e:
        print(f"❌ Config test failed: {e}")
        return False

def test_indicator_feature_sets():
    """Test indicator engine feature set building"""
    try:
        # Mock the logger to avoid import issues
        import sys
        from unittest.mock import Mock
        
        # Create a mock logger
        mock_logger = Mock()
        sys.modules['loguru'] = Mock()
        sys.modules['loguru'].logger = mock_logger
        
        from src.indicators.indicator_engine import IndicatorEngine
        
        engine = IndicatorEngine()
        
        # Test case: OHLCV features with some indicators
        selected_features = ['open', 'high', 'low', 'close', 'volume', 'rsi_14', 'sma_20']
        must_keep = ['open', 'high', 'low', 'close', 'volume'] 
        all_features = ['open', 'high', 'low', 'close', 'volume', 'rsi_14', 'sma_20', 'ema_12', 'macd_12_26_9']
        
        result = engine.build_final_feature_sets(selected_features, must_keep, all_features)
        
        print("✅ Feature set building works:")
        print(f"  Active features: {result['active']}")
        print(f"  Inactive features: {result['inactive']}")
        print(f"  Active count: {result['active_count']}")
        print(f"  Inactive count: {result['inactive_count']}")
        
        # Validate no duplicates
        active_set = set(result['active'])
        inactive_set = set(result['inactive'])
        duplicates = active_set.intersection(inactive_set)
        
        assert len(duplicates) == 0, f"Found duplicates: {duplicates}"
        print("✅ No duplicates between active and inactive")
        
        # Validate all base OHLCV features are active
        base_features = ['open', 'high', 'low', 'close', 'volume']
        for feature in base_features:
            assert feature in result['active'], f"Base feature {feature} not in active list"
        print("✅ All base OHLCV features are active")
        
        return True
    except Exception as e:
        print(f"❌ Indicator test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_catboost_params():
    """Test that CatBoost parameter conflicts are resolved"""
    try:
        # Read the CatBoost model file and check for parameter conflicts
        with open('src/ml_model/catboost_model.py', 'r') as f:
            content = f.read()
        
        # Count occurrences of problematic parameters
        logging_level_count = content.count('logging_level=')
        verbose_count = content.count('verbose=')
        
        print(f"✅ CatBoost parameter check:")
        print(f"  logging_level parameters: {logging_level_count} (should be 0)")
        print(f"  verbose parameters: {verbose_count} (should be > 0)")
        
        # Should have no logging_level and some verbose parameters
        assert logging_level_count == 0, f"Found {logging_level_count} logging_level parameters, should be 0"
        assert verbose_count > 0, f"Found {verbose_count} verbose parameters, should be > 0"
        
        # Check that RFE_VERBOSE is used from config
        assert 'RFE_VERBOSE' in content, "RFE_VERBOSE config not found in CatBoost model"
        
        print("✅ CatBoost parameter conflicts resolved")
        return True
        
    except Exception as e:
        print(f"❌ CatBoost parameter test failed: {e}")
        return False

def main():
    """Run all simple tests"""
    print("🧪 Running Phase 4 Simple Validation Tests\n")
    
    tests = [
        ("Config Updates", test_config_updates),
        ("Indicator Feature Sets", test_indicator_feature_sets), 
        ("CatBoost Parameters", test_catboost_params),
    ]
    
    results = []
    for test_name, test_func in tests:
        print(f"\n📋 {test_name}...")
        try:
            success = test_func()
            results.append(success)
        except Exception as e:
            print(f"❌ {test_name} crashed: {e}")
            results.append(False)
    
    passed = sum(results)
    total = len(results)
    
    print(f"\n📊 RESULTS: {passed}/{total} tests passed")
    
    if passed == total:
        print("🎉 All core Phase 4 fixes validated successfully!")
        print("\nKey fixes verified:")
        print("✅ Config settings added")
        print("✅ Feature classification works without duplicates")
        print("✅ CatBoost parameter conflicts resolved")
        return True
    else:
        print("⚠️  Some tests failed.")
        return False

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)