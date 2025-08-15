#!/usr/bin/env python3
"""
Test script for feature sanitization functionality
"""

import os
import sys
import asyncio
import pandas as pd
import numpy as np

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

from src.ml_model.catboost_model import CatBoostTradingModel

async def test_feature_sanitization():
    """Test the feature sanitization functionality"""
    print("🧪 Testing Feature Sanitization...")
    
    # Create a CatBoost model instance
    model = CatBoostTradingModel()
    await model.initialize()
    
    # Create problematic test data similar to the user's issue
    test_data = {
        # Numeric features (should be kept)
        'SMA_14': [50000.1, 50100.2, 50200.3, 50300.4, 50400.5] * 20,
        'RSI': [65.5, 67.2, 69.1, 70.8, 72.3] * 20,
        'Volume': [1000.0, 1100.0, 1200.0, 1300.0, 1400.0] * 20,
        
        # String constant (should be dropped - this was causing the original error)
        'Symbol': ['BTCUSDT'] * 100,
        
        # String with few categories (should be label encoded)
        'Market_Phase': ['bull', 'bear', 'sideways', 'bull', 'bear'] * 20,
        
        # String with many categories (should be dropped)
        'High_Cardinality': [f'category_{i}' for i in range(100)],
        
        # Mixed numeric/string that can be converted
        'Mixed_Numeric': ['123.45', '234.56', '345.67', '456.78', '567.89'] * 20,
        
        # Boolean that should be converted
        'Boolean_Feature': [True, False, True, False, True] * 20,
        
        # All NaN column (should be dropped after conversion attempts)
        'All_NaN': [np.nan] * 100,
    }
    
    # Create DataFrame
    feature_df = pd.DataFrame(test_data)
    print(f"📊 Original DataFrame shape: {feature_df.shape}")
    print(f"📊 Column types:\n{feature_df.dtypes}")
    print(f"📊 Sample data:\n{feature_df.head()}")
    
    # Test the sanitization method
    try:
        sanitized_df = model._sanitize_features(feature_df)
        
        print(f"\n✅ Sanitization completed!")
        print(f"📊 Sanitized DataFrame shape: {sanitized_df.shape}")
        print(f"📊 Sanitized column types:\n{sanitized_df.dtypes}")
        print(f"📊 Sanitized columns: {list(sanitized_df.columns)}")
        
        # Verify all columns are numeric
        non_numeric_cols = [col for col in sanitized_df.columns 
                           if not pd.api.types.is_numeric_dtype(sanitized_df[col])]
        
        if non_numeric_cols:
            print(f"❌ ERROR: Found non-numeric columns after sanitization: {non_numeric_cols}")
            return False
        else:
            print("✅ All columns are numeric after sanitization")
        
        # Verify no NaN values remain
        nan_count = sanitized_df.isna().sum().sum()
        if nan_count > 0:
            print(f"❌ ERROR: Found {nan_count} NaN values after sanitization")
            return False
        else:
            print("✅ No NaN values remain after sanitization")
            
        # Test prepare_features_and_labels with this problematic data
        print("\n🧪 Testing prepare_features_and_labels with problematic data...")
        X, y = await model.prepare_features_and_labels(test_data, 'BTCUSDT')
        
        if len(X) > 0 and len(y) > 0:
            print(f"✅ prepare_features_and_labels succeeded: X{X.shape}, y{y.shape}")
            
            # Verify we can create a basic CatBoost model with this data
            try:
                from catboost import CatBoostClassifier
                temp_model = CatBoostClassifier(
                    iterations=10,  # Just a few iterations for testing
                    depth=2,
                    learning_rate=0.1,
                    logging_level='Silent',
                    random_seed=42
                )
                temp_model.fit(X, y)
                print("✅ CatBoost training test succeeded - no more numeric conversion errors!")
                return True
                
            except Exception as e:
                print(f"❌ CatBoost training still fails: {e}")
                return False
        else:
            print(f"❌ prepare_features_and_labels failed: X{X.shape}, y{y.shape}")
            return False
            
    except Exception as e:
        print(f"❌ Sanitization failed with error: {e}")
        import traceback
        traceback.print_exc()
        return False

async def test_training_success_reporting():
    """Test that training success is reported correctly"""
    print("\n🧪 Testing Training Success Reporting...")
    
    model = CatBoostTradingModel()
    await model.initialize()
    
    # Test with empty data (should return False)
    empty_data = {}
    result = await model.retrain_online(empty_data, 'BTCUSDT', ['SMA_14', 'RSI'])
    
    if result is False:
        print("✅ retrain_online correctly returns False for empty data")
    else:
        print(f"❌ retrain_online should return False for empty data, got: {result}")
        return False
    
    # Test with insufficient data (should return False)
    small_data = {
        'SMA_14': [1, 2, 3],  # Too few samples
        'RSI': [60, 65, 70]
    }
    result = await model.retrain_online(small_data, 'BTCUSDT', ['SMA_14', 'RSI'])
    
    if result is False:
        print("✅ retrain_online correctly returns False for insufficient data")
        return True
    else:
        print(f"❌ retrain_online should return False for insufficient data, got: {result}")
        return False

async def main():
    """Run all tests"""
    print("🚀 Running Feature Sanitization and Training Success Tests\n")
    
    test1_passed = await test_feature_sanitization()
    test2_passed = await test_training_success_reporting()
    
    print(f"\n📊 Test Results:")
    print(f"   Feature Sanitization: {'✅ PASSED' if test1_passed else '❌ FAILED'}")
    print(f"   Training Success Reporting: {'✅ PASSED' if test2_passed else '❌ FAILED'}")
    
    if test1_passed and test2_passed:
        print("\n🎉 All tests passed! The fixes should resolve the original issues.")
        return 0
    else:
        print("\n💥 Some tests failed. Please review the implementation.")
        return 1

if __name__ == "__main__":
    exit_code = asyncio.run(main())
    sys.exit(exit_code)