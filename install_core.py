#!/usr/bin/env python3
"""
Simple installer for core dependencies needed for the trading system to start
This installs only the essential packages to get the system running
"""

import subprocess
import sys
import time

def install_package(package_name):
    """Install a single package with timeout handling"""
    try:
        print(f"📦 Installing {package_name}...")
        result = subprocess.run(
            [sys.executable, '-m', 'pip', 'install', package_name, '--user', '--timeout', '60'],
            capture_output=True,
            text=True,
            timeout=120  # 2 minute timeout per package
        )
        
        if result.returncode == 0:
            print(f"✅ Successfully installed {package_name}")
            return True
        else:
            print(f"❌ Failed to install {package_name}")
            print(f"Error: {result.stderr}")
            return False
            
    except subprocess.TimeoutExpired:
        print(f"⏰ Timeout installing {package_name} - skipping")
        return False
    except Exception as e:
        print(f"❌ Error installing {package_name}: {e}")
        return False

def main():
    print("🚀 Installing core dependencies for Trading System...")
    print("This will install only essential packages to get the system started")
    
    # Core packages needed for basic functionality
    core_packages = [
        'loguru==0.7.2',          # Logging
        'aiohttp==3.9.1',         # HTTP client for API calls
        'pandas>=2.0.0',          # Data manipulation (use any compatible version)
        'numpy>=1.20.0',          # Mathematical operations
        'flask>=3.0.0',           # Web framework
        'requests>=2.28.0',       # HTTP requests
    ]
    
    # Optional packages (install if possible but don't fail if they don't install)
    optional_packages = [
        'python-binance>=1.0.0',  # Binance API
        'scikit-learn>=1.2.0',    # ML library
        'flask-socketio>=5.0.0',  # WebSocket support
    ]
    
    successful_installs = 0
    total_packages = len(core_packages) + len(optional_packages)
    
    print(f"\n📋 Installing {len(core_packages)} core packages...")
    for package in core_packages:
        if install_package(package):
            successful_installs += 1
        time.sleep(1)  # Brief delay between installs
    
    print(f"\n🔧 Installing {len(optional_packages)} optional packages...")
    for package in optional_packages:
        if install_package(package):
            successful_installs += 1
        time.sleep(1)
    
    print(f"\n📊 Installation Summary:")
    print(f"✅ Successfully installed: {successful_installs}/{total_packages} packages")
    
    if successful_installs >= len(core_packages):
        print("🎉 Core dependencies installed successfully!")
        print("You can now try running: python main.py")
    else:
        print("⚠️  Some core dependencies failed to install")
        print("You may need to install them manually or check your internet connection")
    
    print("\n📝 If you continue to have issues, try:")
    print("   pip install --upgrade pip")
    print("   pip install loguru pandas aiohttp flask requests")

if __name__ == "__main__":
    main()