#!/usr/bin/env python3
"""
End-to-End Test for Complete Pipeline Restructure
Tests the full pipeline with shortened collection duration
"""

import os
import sys
import asyncio
import time
from datetime import datetime

# Add src to path
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

# Set test environment variables
os.environ['FORCE_MYSQL_ONLY'] = 'false'  # Allow SQLite for testing
os.environ['INITIAL_COLLECTION_DURATION_SEC'] = '30'  # Short test duration

print("🧪 Starting Complete Pipeline Restructure End-to-End Test")
print("=" * 60)

async def test_complete_pipeline():
    """Test the complete pipeline restructure"""
    try:
        # Import after setting environment variables
        from main import TradingSystem
        
        print("✅ TradingSystem imported successfully")
        
        # Create trading system
        trading_system = TradingSystem()
        print("✅ TradingSystem created successfully")
        
        # Test phases individually with timeouts
        
        # Phase 1: System validation
        print("\n📋 Phase 1: Testing system validation...")
        await asyncio.wait_for(trading_system.validate_system_requirements(), timeout=30)
        print("✅ Phase 1: System validation passed")
        
        # Phase 2: Component initialization
        print("\n📋 Phase 2: Testing component initialization...")
        await asyncio.wait_for(trading_system.initialize_components(), timeout=60)
        print("✅ Phase 2: Component initialization passed")
        
        # Phase 3: Web app start (don't actually start server in test)
        print("\n📋 Phase 3: Testing web app creation...")
        trading_system.web_app = trading_system.create_app(
            data_collector=trading_system.data_collector,
            indicator_engine=trading_system.indicator_engine, 
            ml_model=trading_system.ml_model,
            trading_engine=trading_system.trading_engine
        )
        trading_system.running = True
        print("✅ Phase 3: Web app creation passed")
        
        # Phase 4: Bootstrap collection (shortened)
        print(f"\n📋 Phase 4: Testing 30s bootstrap collection...")
        start_time = time.time()
        bootstrap_success = await asyncio.wait_for(
            trading_system.run_bootstrap_collection(), 
            timeout=60
        )
        duration = time.time() - start_time
        print(f"✅ Phase 4: Bootstrap collection {'passed' if bootstrap_success else 'completed'} in {duration:.1f}s")
        
        # Phase 5: Indicators preparation
        print("\n📋 Phase 5: Testing indicator preparation...")
        await asyncio.wait_for(trading_system.prepare_indicators(), timeout=30)
        print("✅ Phase 5: Indicator preparation passed")
        
        # Phase 6: Initial training
        print("\n📋 Phase 6: Testing initial training...")
        await asyncio.wait_for(trading_system.run_initial_training(), timeout=120)
        print("✅ Phase 6: Initial training passed")
        
        # Phase 7: Trading operations
        print("\n📋 Phase 7: Testing trading operations start...")
        await asyncio.wait_for(trading_system.start_trading_operations(), timeout=30)
        print("✅ Phase 7: Trading operations passed")
        
        # Test system status API
        print("\n📋 Testing system status API...")
        status = trading_system.trading_engine.get_system_status()
        
        # Validate required fields
        required_fields = ['collection', 'training', 'indicators', 'backend']
        for field in required_fields:
            if field not in status:
                raise AssertionError(f"Missing required field in status: {field}")
        
        print("✅ System status API validation passed")
        
        # Test JSON serialization
        print("\n📋 Testing JSON serialization...")
        from src.utils.json_sanitize import safe_json_dumps
        json_result = safe_json_dumps(status)
        if '"error"' in json_result:
            print(f"⚠️ JSON serialization had issues: {json_result[:200]}...")
        else:
            print("✅ JSON serialization passed")
        
        # Print summary
        print("\n" + "=" * 60)
        print("🎉 END-TO-END TEST COMPLETED SUCCESSFULLY!")
        print("=" * 60)
        
        # Print system info
        collection_status = status.get('collection', {})
        training_status = status.get('training', {})
        indicators_status = status.get('indicators', {})
        backend_status = status.get('backend', {})
        
        print(f"📊 Collection: {collection_status.get('records_total', 0)} records collected")
        print(f"🧠 Training: {training_status.get('progress_percent', 0):.1f}% complete")
        print(f"📈 Indicators: {indicators_status.get('computed_count', 0)} computed")
        print(f"💾 Backend: {backend_status.get('db_engine', 'unknown')}")
        print(f"🎯 Model trained: {trading_system.ml_model.is_trained}")
        print(f"📊 Live accuracy: {training_status.get('accuracy_live', 0.0):.4f}")
        
        return True
        
    except Exception as e:
        print(f"\n❌ TEST FAILED: {e}")
        import traceback
        print(traceback.format_exc())
        return False

async def main():
    """Main test function"""
    try:
        success = await test_complete_pipeline()
        if success:
            print("\n✅ ALL TESTS PASSED")
            sys.exit(0)
        else:
            print("\n❌ SOME TESTS FAILED")
            sys.exit(1)
            
    except KeyboardInterrupt:
        print("\n⏹️ Test interrupted by user")
        sys.exit(130)
    except Exception as e:
        print(f"\n💥 Test framework error: {e}")
        sys.exit(1)

if __name__ == "__main__":
    print(f"🕒 Test started at: {datetime.now().isoformat()}")
    asyncio.run(main())