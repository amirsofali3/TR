# Install the packages needed for the crypto trading system

# Install packages one by one to avoid timeout issues
packages = [
    'pandas==2.1.4',
    'numpy==1.24.4',
    'scikit-learn==1.3.2',
    'catboost==1.2.2',
    'flask==3.0.0',
    'flask-socketio==5.3.6',
    'flask-cors==4.0.0',
    'aiohttp==3.9.1',
    'loguru==0.7.2',
    'python-dateutil==2.8.2',
    'requests==2.31.0',
    'scipy==1.11.4',
    'joblib==1.3.2'
]

for package in packages:
    try:
        import subprocess
        result = subprocess.run(['pip', 'install', '--quiet', package], 
                              capture_output=True, text=True, timeout=300)
        if result.returncode == 0:
            print(f"✅ Installed {package}")
        else:
            print(f"❌ Failed to install {package}: {result.stderr}")
    except Exception as e:
        print(f"❌ Error installing {package}: {e}")

print("\n📦 Package installation completed!")
print("Run 'python main.py' to start the trading system.")
print("Open http://localhost:5000 in your browser to access the dashboard.")