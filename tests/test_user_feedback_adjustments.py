#!/usr/bin/env python3
"""
Test User Feedback Adjustments to Pipeline Restructure

Tests the specific changes requested by user:
- Configuration defaults
- Retrain behavior changes  
- New API endpoints
- Enhanced status fields
- Analysis interval changes
"""

import os
import sys
import unittest
from unittest.mock import AsyncMock, MagicMock
import asyncio
import json

# Add src to path
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'src'))
sys.path.append(os.path.dirname(__file__) + "/..")

# Set test environment
os.environ['FORCE_MYSQL_ONLY'] = 'false'
os.environ['INITIAL_COLLECTION_DURATION_SEC'] = '5'

from config.settings import *

class TestUserFeedbackAdjustments(unittest.TestCase):
    """Test User Feedback Adjustments"""
    
    def test_configuration_defaults(self):
        """Test that new configuration defaults are correct"""
        # Test new configuration values
        self.assertEqual(BASE_ANALYSIS_INTERVAL_SEC, 5, "BASE_ANALYSIS_INTERVAL_SEC should default to 5")
        self.assertEqual(RAW_COLLECTION_INTERVAL_SEC, 1, "RAW_COLLECTION_INTERVAL_SEC should be 1")
        self.assertEqual(ACCURACY_UPDATE_INTERVAL_SEC, 60, "ACCURACY_UPDATE_INTERVAL_SEC should be 60")
        self.assertEqual(MIN_VALID_SAMPLES, 150, "MIN_VALID_SAMPLES should be 150")
        self.assertEqual(RETRAIN_USE_FULL_HISTORY, True, "RETRAIN_USE_FULL_HISTORY should default to True")
        self.assertEqual(RETRAIN_HISTORY_MINUTES, 180, "RETRAIN_HISTORY_MINUTES should be 180")
        
        print("✅ Configuration defaults test passed")

    def test_retrain_behavior_changes(self):
        """Test retrain behavior changes"""
        from src.trading_engine.trading_engine import TradingEngine
        
        # Create mock components
        data_collector = MagicMock()
        indicator_engine = MagicMock()
        ml_model = MagicMock()
        
        # Setup mock methods
        data_collector.get_historical_data = AsyncMock(return_value=None)
        indicator_engine.calculate_all_indicators = AsyncMock(return_value=None)
        indicator_engine.get_rfe_eligible_indicators = MagicMock(return_value=[])
        
        trading_engine = TradingEngine(data_collector, indicator_engine, ml_model)
        
        async def test_retrain():
            # Test that "full" retrain type is converted to "fast"
            result = await trading_engine.manual_retrain("full")
            
            # Should fail due to insufficient data, but should log "fast" mode
            self.assertFalse(result['success'])
            self.assertIn('error', result)
            
            # Test that fast retrain uses existing data only
            result = await trading_engine.manual_retrain("fast") 
            self.assertFalse(result['success'])  # Expected to fail with no data
            self.assertEqual(result.get('error'), 'INSUFFICIENT_DATA')
            
        asyncio.run(test_retrain())
        print("✅ Retrain behavior changes test passed")

    def test_trading_engine_intervals(self):
        """Test analysis interval management"""
        from src.trading_engine.trading_engine import TradingEngine
        
        # Create mock components
        data_collector = MagicMock()
        indicator_engine = MagicMock()
        ml_model = MagicMock()
        
        trading_engine = TradingEngine(data_collector, indicator_engine, ml_model)
        
        # Test initial interval
        self.assertEqual(trading_engine.current_analysis_interval, BASE_ANALYSIS_INTERVAL_SEC)
        
        # Test pending interval change
        trading_engine._pending_interval_change = 15
        trading_engine._check_pending_interval_change()
        
        self.assertEqual(trading_engine.current_analysis_interval, 15)
        self.assertIsNone(trading_engine._pending_interval_change)
        
        print("✅ Analysis interval management test passed")

    def test_model_class_mapping(self):
        """Test class mapping in model info"""
        from src.ml_model.catboost_model import CatBoostTradingModel
        
        model = CatBoostTradingModel()
        model_info = model.get_model_info()
        
        # Test class mapping
        self.assertIn('class_mapping', model_info)
        class_mapping = model_info['class_mapping']
        
        expected_mapping = {"0": "SELL", "1": "HOLD", "2": "BUY"}
        self.assertEqual(class_mapping, expected_mapping)
        
        print("✅ Class mapping test passed")

    def test_web_app_new_endpoints(self):
        """Test new web app endpoints"""
        from src.web_app.app import create_app
        
        # Create mock components
        data_collector = MagicMock()
        indicator_engine = MagicMock()
        ml_model = MagicMock()
        trading_engine = MagicMock()
        
        # Setup mock responses
        ml_model.get_model_info.return_value = {
            'active_features': ['feature1', 'feature2'],
            'inactive_features': ['feature3', 'feature4'],
            'feature_importance': {'feature1': 0.5, 'feature2': 0.3},
            'model_version': 1,
            'last_training_time': None
        }
        
        trading_engine.get_system_status.return_value = {
            'analysis': {'interval_sec': 5},
            'training': {'class_mapping': {"0": "SELL", "1": "HOLD", "2": "BUY"}}
        }
        
        app = create_app(data_collector, indicator_engine, ml_model, trading_engine)
        
        with app.test_client() as client:
            # Test /api/features endpoint
            response = client.get('/api/features')
            self.assertEqual(response.status_code, 200)
            
            data = response.get_json()
            self.assertIn('selected', data)
            self.assertIn('inactive', data)
            self.assertIn('metadata', data)
            
            # Test structure
            self.assertEqual(len(data['selected']), 2)
            self.assertEqual(len(data['inactive']), 2)
            
            # Test /api/analysis/interval endpoint
            response = client.post('/api/analysis/interval', 
                                 json={'interval_sec': 10},
                                 content_type='application/json')
            self.assertEqual(response.status_code, 200)
            
            result = response.get_json()
            self.assertTrue(result['success'])
            self.assertEqual(result['new_interval_sec'], 10)
            
            # Test invalid interval
            response = client.post('/api/analysis/interval',
                                 json={'interval_sec': 100},
                                 content_type='application/json')
            self.assertEqual(response.status_code, 400)
            
            # Test /api/retrain endpoint (always fast mode)
            response = client.post('/api/retrain')
            self.assertEqual(response.status_code, 202)  # Should return 202 Accepted
            
            result = response.get_json()
            self.assertTrue(result['started'])
            self.assertEqual(result['mode'], 'fast')
            
        print("✅ New web endpoints test passed")


if __name__ == '__main__':
    print("🧪 Testing User Feedback Adjustments")
    print("=" * 50)
    
    # Run tests
    unittest.main(verbosity=2, exit=False)
    
    print("\n" + "=" * 50)
    print("✅ User Feedback Adjustments Tests Completed")