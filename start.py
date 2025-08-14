#!/usr/bin/env python3
"""
Startup script for the Crypto Trading AI System
This script provides clear error messages and guidance when the system cannot start
"""

import sys
import os
import subprocess
from datetime import datetime

def print_header():
    """Print system header"""
    print("🚀 Crypto Trading AI System")
    print("=" * 50)
    print(f"📅 Started at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

def check_dependency(module_name):
    """Check if a dependency is available"""
    try:
        __import__(module_name)
        return True
    except ImportError:
        return False

def install_dependency(package_name):
    """Try to install a dependency"""
    try:
        print(f"📦 Installing {package_name}...")
        result = subprocess.run(
            [sys.executable, '-m', 'pip', 'install', package_name],
            capture_output=True, text=True, timeout=60
        )
        if result.returncode == 0:
            print(f"✅ Successfully installed {package_name}")
            return True
        else:
            print(f"❌ Failed to install {package_name}")
            return False
    except:
        return False

def check_and_install_dependencies():
    """Check and try to install missing dependencies"""
    required_deps = [
        ('loguru', 'loguru'),
        ('aiohttp', 'aiohttp'),
        ('pandas', 'pandas'),
        ('numpy', 'numpy'),
        ('flask', 'flask'),
    ]
    
    missing_deps = []
    
    print("🔍 Checking dependencies...")
    for module_name, package_name in required_deps:
        if check_dependency(module_name):
            print(f"✅ {module_name}")
        else:
            print(f"❌ {module_name} - MISSING")
            missing_deps.append((module_name, package_name))
    
    if missing_deps:
        print(f"\n⚠️  Found {len(missing_deps)} missing dependencies")
        print("🔧 Attempting to install missing packages...")
        
        success_count = 0
        for module_name, package_name in missing_deps:
            if install_dependency(package_name):
                success_count += 1
        
        print(f"\n📊 Installation result: {success_count}/{len(missing_deps)} successful")
        
        if success_count < len(missing_deps):
            print("\n❌ Some dependencies could not be installed automatically")
            print("💡 Manual installation required:")
            print("   pip install loguru aiohttp pandas numpy flask")
            print("   OR")
            print("   pip install -r requirements.txt")
            return False
    
    return True

def main():
    """Main function"""
    print_header()
    
    # Check if system can start
    if not check_and_install_dependencies():
        print("\n❌ Cannot start system due to missing dependencies")
        print("\n🔧 To fix this issue:")
        print("   1. Install Python packages: pip install loguru aiohttp pandas numpy flask")
        print("   2. Then run: python main.py")
        print("   3. Or use the complete installer: python install_packages.py")
        return 1
    
    # Try to start the main system
    try:
        print("\n🚀 Starting trading system...")
        
        # Import and start the main system
        import asyncio
        from main import main as main_system
        
        asyncio.run(main_system())
        
    except ImportError as e:
        print(f"\n❌ System startup failed: {e}")
        print("💡 This usually means missing dependencies or configuration issues")
        print("🔧 Try running the diagnostic script: python diagnose.py")
        return 1
    except KeyboardInterrupt:
        print("\n⏹️  System stopped by user")
        return 0
    except Exception as e:
        print(f"\n❌ Unexpected error: {e}")
        print("🔧 Try running the diagnostic script: python diagnose.py")
        return 1

if __name__ == "__main__":
    sys.exit(main())