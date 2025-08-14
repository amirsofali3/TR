#!/usr/bin/env python3
"""
System diagnostics and test script for the Trading System
This helps identify what's preventing the system from starting analysis
"""

import sys
import os
import asyncio
from datetime import datetime

def check_python_version():
    """Check if Python version is compatible"""
    version = sys.version_info
    if version.major >= 3 and version.minor >= 8:
        print(f"✅ Python version {version.major}.{version.minor}.{version.micro} is compatible")
        return True
    else:
        print(f"❌ Python version {version.major}.{version.minor}.{version.micro} is too old (need 3.8+)")
        return False

def check_dependencies():
    """Check if required dependencies are available"""
    required_modules = [
        ('loguru', 'Logging system'),
        ('aiohttp', 'HTTP client for API calls'),
        ('pandas', 'Data manipulation'),
        ('numpy', 'Numerical operations'),
        ('flask', 'Web framework'),
        ('requests', 'HTTP requests'),
    ]
    
    optional_modules = [
        ('binance', 'Binance API client'),
        ('sklearn', 'Machine learning'),
        ('ccxt', 'Cryptocurrency exchange library'),
    ]
    
    print("🔍 Checking required dependencies...")
    missing_required = []
    
    for module_name, description in required_modules:
        try:
            __import__(module_name)
            print(f"✅ {module_name} - {description}")
        except ImportError:
            print(f"❌ {module_name} - {description} - MISSING")
            missing_required.append(module_name)
    
    print("\n🔍 Checking optional dependencies...")
    missing_optional = []
    
    for module_name, description in optional_modules:
        try:
            __import__(module_name)
            print(f"✅ {module_name} - {description}")
        except ImportError:
            print(f"⚠️  {module_name} - {description} - Missing (optional)")
            missing_optional.append(module_name)
    
    return missing_required, missing_optional

def check_configuration():
    """Check system configuration"""
    print("\n📋 Checking configuration...")
    
    try:
        # Add src to path
        sys.path.append(os.path.join(os.getcwd(), 'src'))
        
        # Import configuration
        import config.settings as settings
        
        print(f"✅ Configuration loaded successfully")
        print(f"   📊 Supported pairs: {len(settings.SUPPORTED_PAIRS)} ({', '.join(settings.SUPPORTED_PAIRS)})")
        print(f"   ⏱️  Update interval: {settings.UPDATE_INTERVAL} seconds")
        print(f"   🏦 Demo mode: {'Yes' if settings.DEMO_MODE else 'No'}")
        
        # Check API keys
        api_configured = (settings.BINANCE_API_KEY != "your_binance_api_key_here" and 
                         settings.BINANCE_SECRET_KEY != "your_binance_secret_key_here" and
                         settings.BINANCE_API_KEY and settings.BINANCE_SECRET_KEY)
        
        if api_configured:
            print(f"   🔑 API keys: Configured")
        else:
            print(f"   🔑 API keys: Not configured (will use public API)")
            
        return True
        
    except Exception as e:
        print(f"❌ Configuration check failed: {e}")
        return False

def check_directories():
    """Check if required directories exist and are writable"""
    print("\n📁 Checking directories...")
    
    try:
        import config.settings as settings
        
        # Check data directory
        data_dir = os.path.dirname(settings.DATABASE_URL.replace("sqlite:///", ""))
        if data_dir:
            if os.path.exists(data_dir):
                print(f"✅ Data directory exists: {data_dir}")
            else:
                try:
                    os.makedirs(data_dir, exist_ok=True)
                    print(f"✅ Created data directory: {data_dir}")
                except Exception as e:
                    print(f"❌ Cannot create data directory {data_dir}: {e}")
                    return False
        
        # Check logs directory  
        log_dir = os.path.dirname(settings.LOG_FILE)
        if log_dir:
            if os.path.exists(log_dir):
                print(f"✅ Logs directory exists: {log_dir}")
            else:
                try:
                    os.makedirs(log_dir, exist_ok=True)
                    print(f"✅ Created logs directory: {log_dir}")
                except Exception as e:
                    print(f"❌ Cannot create logs directory {log_dir}: {e}")
                    return False
        
        return True
        
    except Exception as e:
        print(f"❌ Directory check failed: {e}")
        return False

async def test_network_connectivity():
    """Test network connectivity to Binance API"""
    print("\n🌐 Testing network connectivity...")
    
    try:
        import aiohttp
        
        async with aiohttp.ClientSession() as session:
            # Test public Binance API
            url = "https://api.binance.com/api/v3/ping"
            async with session.get(url, timeout=aiohttp.ClientTimeout(total=10)) as response:
                if response.status == 200:
                    print("✅ Binance API connectivity: OK")
                    return True
                else:
                    print(f"❌ Binance API connectivity failed: {response.status}")
                    return False
                    
    except Exception as e:
        print(f"❌ Network connectivity test failed: {e}")
        return False

def print_solutions(missing_required, missing_optional):
    """Print solutions for common problems"""
    print("\n💡 Solutions:")
    
    if missing_required:
        print("🔧 Install missing required packages:")
        print(f"   pip install {' '.join(missing_required)}")
        print("   OR run: python install_core.py")
        
    if missing_optional:
        print("🔧 Install missing optional packages (for full functionality):")
        print(f"   pip install {' '.join(missing_optional)}")
        
    print("\n🚀 Quick start commands:")
    print("   1. python install_core.py    # Install core dependencies")
    print("   2. python main.py            # Start the trading system")
    print("   3. Open http://localhost:5000 in your browser")

async def main():
    """Main diagnostic function"""
    print("🔍 Trading System Diagnostics")
    print("=" * 50)
    
    all_checks_passed = True
    
    # Check Python version
    if not check_python_version():
        all_checks_passed = False
    
    # Check dependencies
    missing_required, missing_optional = check_dependencies()
    if missing_required:
        all_checks_passed = False
    
    # Check configuration
    if missing_required:
        print("\n⚠️  Skipping configuration check due to missing dependencies")
    else:
        if not check_configuration():
            all_checks_passed = False
        
        if not check_directories():
            all_checks_passed = False
            
        # Test network connectivity
        network_ok = await test_network_connectivity()
        if not network_ok:
            print("⚠️  Network issues may prevent data fetching")
    
    print("\n" + "=" * 50)
    
    if all_checks_passed:
        print("🎉 All checks passed! The system should be ready to run.")
        print("Try running: python main.py")
    else:
        print("❌ Some issues were found that may prevent the system from working.")
        print_solutions(missing_required, missing_optional)
    
    print(f"\n📅 Diagnostic completed at {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

if __name__ == "__main__":
    asyncio.run(main())