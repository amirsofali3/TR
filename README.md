# Crypto Trading AI System

A comprehensive cryptocurrency trading AI system with advanced machine learning, 435+ technical indicators, step-wise TP/SL, and real-time web dashboard.

## 🚀 Features

### 🤖 Advanced AI Trading
- **CatBoost ML Model** with high depth and tree count for maximum accuracy
- **RFE Feature Selection** to identify the most impactful indicators
- **435+ Technical Indicators** from comprehensive encyclopedia
- **Confidence-based Execution** (only >70% confidence trades)
- **Auto-retraining** with latest market data every 6 hours

### 📊 Comprehensive Technical Analysis
- **Trend Indicators**: SMA, EMA, MACD, Ichimoku, SuperTrend, Parabolic SAR
- **Momentum Indicators**: RSI, Stochastic, CCI, Williams %R, ROC, MFI
- **Volatility Indicators**: ATR, Bollinger Bands, Donchian Channels, Keltner Channels
- **Volume Indicators**: OBV, CMF, Force Index, VWAP, Money Flow Index
- **Price Action**: Returns, patterns, support/resistance levels
- **Candlestick Patterns**: 25+ pattern recognition algorithms
- **Order Book Analysis**: Bid/ask spreads, depth analysis, imbalance calculations

### 💰 Advanced Risk Management
- **Step-wise TP/SL System**: Multiple take profit levels with trailing stops
- **Dynamic Stop Loss**: Moves to previous TP level as profits are taken
- **Position Size Management**: Based on confidence and portfolio balance
- **Portfolio Tracking**: Real-time P&L, daily performance, position monitoring

### 🌐 Real-time Web Dashboard
- **Live Market Data**: Real-time price updates and analysis
- **Indicator Status**: Visual active/inactive indicators with blinking lights
- **Model Training Progress**: Live training status and accuracy metrics
- **Portfolio Management**: Balance, positions, P&L tracking
- **Signal History**: Recent trading signals with execution status
- **TP/SL Configuration**: Per-symbol TP/SL settings

### 🔧 System Architecture
- **4-Hour Timeframe Analysis** with 1-minute price updates
- **Demo Mode**: Paper trading with real market data
- **Database Persistence**: SQLite for data storage and history
- **Async Operations**: Non-blocking data collection and analysis
- **REST API**: Full API for external integrations
- **WebSocket**: Real-time dashboard updates

## 📋 Requirements

```
pandas==2.1.4
numpy==1.24.4
scikit-learn==1.3.2
catboost==1.2.2
ta-lib==0.4.26
python-binance==1.0.19
websockets==12.0
aiohttp==3.9.1
ccxt==4.1.74
sqlalchemy==2.0.23
psycopg2-binary==2.9.9
flask==3.0.0
flask-socketio==5.3.6
flask-cors==4.0.0
gunicorn==21.2.0
plotly==5.17.0
matplotlib==3.8.2
seaborn==0.13.0
python-dotenv==1.0.0
schedule==1.2.0
pytz==2023.3
python-dateutil==2.8.2
requests==2.31.0
urllib3==2.1.0
loguru==0.7.2
pyyaml==6.0.1
configparser==6.0.0
scipy==1.11.4
statsmodels==0.14.1
numba==0.58.1
joblib==1.3.2
openpyxl==3.1.2
xlsxwriter==3.1.9
pytest==7.4.3
pytest-asyncio==0.21.1
jupyter==1.0.0
ipython==8.18.1
```

## 🛠️ Installation & Quick Start

### Option 1: Smart Startup (Recommended)
The easiest way to get started - automatically handles dependency installation:
```bash
python start.py
```

### Option 2: Manual Installation  
1. **Clone the repository**
```bash
git clone <repository-url>
cd TR
```

2. **Install core dependencies**
```bash
pip install loguru aiohttp pandas numpy flask
# OR install all dependencies
pip install -r requirements.txt
```

3. **Configure API keys** (optional for demo mode)
```bash
# Edit config/settings.py
BINANCE_API_KEY = "your_api_key_here"
BINANCE_SECRET_KEY = "your_secret_key_here"
```

4. **Start the system**
```bash
python main.py
```

5. **Open dashboard**
```
http://localhost:5000
```

### 🔍 Troubleshooting Startup Issues

**Problem**: System doesn't start analysis / "No module named 'loguru'" errors
**Solution**: Missing dependencies. Use one of these methods:
```bash
# Method 1: Smart installer
python start.py

# Method 2: Manual install
pip install loguru aiohttp pandas numpy flask

# Method 3: Diagnostics
python diagnose.py
```

**Problem**: "Failed to initialize components"
**Solution**: Run diagnostics to identify the specific issue:
```bash
python diagnose.py
```

## 📊 Technical Indicators

The system includes 435+ technical indicators organized by category:

### Core Price Data (7)
- Timestamp, Open, High, Low, Close, Volume, Symbol

### Trend Indicators (120+)
- **Moving Averages**: SMA, EMA, WMA, HMA, DEMA, TEMA, KAMA, ZLEMA, LSMA
- **MACD Variations**: Multiple timeframe combinations
- **Ichimoku Cloud**: Full suite including Tenkan, Kijun, Senkou A/B, Chikou
- **SuperTrend**: Multiple ATR periods and multipliers
- **Parabolic SAR**: Adaptive stop and reverse

### Momentum Indicators (50+)
- **RSI**: Multiple periods (2, 3, 5, 7, 9, 14, 21, 28, 50)
- **Stochastic**: %K and %D for various periods
- **CCI**: Commodity Channel Index variations
- **Williams %R**: Multiple timeframes
- **ROC**: Rate of Change calculations
- **MFI**: Money Flow Index
- **Ultimate Oscillator**: Multi-timeframe momentum
- **TSI**: True Strength Index

### Volatility Indicators (40+)
- **ATR**: Average True Range for multiple periods
- **Bollinger Bands**: Various standard deviation multipliers
- **Donchian Channels**: Price breakout levels
- **Keltner Channels**: EMA-based volatility channels
- **Realized Volatility**: Historical volatility calculations
- **Chaikin Volatility**: Volume-weighted volatility

### Volume Indicators (30+)
- **OBV**: On-Balance Volume and smoothed versions
- **CMF**: Chaikin Money Flow
- **A/D Line**: Accumulation/Distribution
- **Force Index**: Price-volume momentum
- **Volume Moving Averages**: SMA and EMA variations
- **VWAP**: Volume Weighted Average Price
- **PVI/NVI**: Positive/Negative Volume Index

### Price Action Indicators (60+)
- **Returns**: Multiple horizon calculations
- **High-Low Ranges**: Various period ranges
- **Rolling Extremes**: Support and resistance levels
- **Pivot Points**: Multiple timeframe pivots
- **Linear Regression**: Slope, intercept, R-squared
- **ZigZag Patterns**: Trend reversal identification

### Candlestick Patterns (25+)
- **Reversal Patterns**: Hammer, Shooting Star, Doji variations
- **Continuation Patterns**: Spinning tops, inside/outside bars
- **Engulfing Patterns**: Bullish and bearish engulfing
- **Star Patterns**: Morning/Evening star formations
- **Advanced Patterns**: Three white soldiers, three black crows

### Order Book Analysis (50+)
- **Level 1-10 Data**: Bid/ask prices and sizes
- **Spreads and Imbalances**: Market microstructure
- **Depth Analysis**: Cumulative order book depth
- **Order Flow**: Signed volume and flow analysis
- **Kyle's Lambda**: Market impact measurements

### Intermarket Analysis (20+)
- **Correlation Analysis**: Asset correlation matrices
- **Beta Calculations**: Market sensitivity measures
- **Price Ratios**: Cross-asset relative strength
- **Spread Analysis**: Basis and calendar spreads

## 🤖 Machine Learning Model

### CatBoost Configuration
```python
CatBoostClassifier(
    iterations=1000,           # High iteration count
    depth=10,                  # Deep trees
    learning_rate=0.1,         # Optimized learning rate
    loss_function='MultiClass', # 3-class prediction (BUY/HOLD/SELL)
    bootstrap_type='Bayesian', # Advanced sampling
    subsample=0.8,             # Feature subsampling
    reg_lambda=3.0,            # L2 regularization
    early_stopping_rounds=50   # Prevent overfitting
)
```

### Feature Selection (RFE)
- **RFE with Cross-Validation**: 3-fold CV for optimal feature count
- **Target Features**: 50 most impactful indicators
- **Feature Weights**: Active features (1.0), inactive features (0.01)
- **Dynamic Selection**: Re-runs RFE with latest 500 market candles

### Label Generation
- **Multi-class Classification**: BUY (2), HOLD (1), SELL (0)
- **Future Return Analysis**: 1-period and 3-period forward returns
- **Threshold-based Labels**: ±2% thresholds for buy/sell signals
- **Confidence Scoring**: Probability-based confidence metrics

## 💰 Risk Management

### Step-wise TP/SL System
```
Entry: $50,000
TP1: $51,000 (2%) - Close 20% of position, move SL to entry
TP2: $52,000 (4%) - Close 20% of position, move SL to TP1
TP3: $53,000 (6%) - Close 20% of position, move SL to TP2
TP4: $54,000 (8%) - Close 20% of position, move SL to TP3
TP5: $55,000 (10%) - Close remaining 20%, move SL to TP4
```

### Position Management
- **Dynamic Position Sizing**: Based on confidence and portfolio balance
- **Risk per Trade**: 2% of portfolio (adjustable)
- **Maximum Positions**: Configurable per symbol
- **Portfolio Protection**: Stop trading at maximum drawdown

### Performance Tracking
- **Real-time P&L**: Unrealized and realized profits/losses
- **Daily Performance**: Daily P&L tracking and statistics
- **Trade History**: Complete audit trail of all trades
- **Risk Metrics**: Sharpe ratio, maximum drawdown, win rate

## 🌐 Web Dashboard

### Main Dashboard Features
- **System Status**: Real-time system health monitoring
- **ML Model Progress**: Training progress and performance metrics
- **Active Indicators**: Visual indicator status with blinking lights
- **Portfolio Summary**: Balance, P&L, active positions
- **Price Monitor**: Real-time price feeds for all supported pairs
- **Signal History**: Recent trading signals with execution status

### Configuration Interface
- **TP/SL Settings**: Per-symbol take profit and stop loss configuration
- **Timeframe Selection**: Adjustable analysis timeframes
- **Confidence Threshold**: Minimum confidence for trade execution
- **Position Limits**: Maximum positions per symbol

### Real-time Updates
- **WebSocket Integration**: Live updates without page refresh
- **5-second Refresh**: Automatic data updates every 5 seconds
- **Mobile Responsive**: Works on all device types
- **Dark Mode Ready**: Professional trading interface

## 📡 API Endpoints

### System Information
```
GET /api/status          - System status and configuration
GET /api/indicators      - Indicator status and importance
GET /api/portfolio       - Portfolio summary and positions
GET /api/signals         - Recent trading signals
GET /api/prices          - Current market prices
```

### Configuration
```
GET /api/tp_sl_config/{symbol}     - Get TP/SL configuration
POST /api/tp_sl_config/{symbol}    - Update TP/SL configuration
```

### Trading Operations
```
POST /api/close_position/{id}      - Manually close position
POST /api/retrain_model            - Trigger model retraining
```

## 🔧 Configuration

### Main Settings (config/settings.py)
```python
# Trading Configuration
DEFAULT_TIMEFRAME = "4h"           # Analysis timeframe
UPDATE_INTERVAL = 60               # Update every minute
CONFIDENCE_THRESHOLD = 70.0        # Minimum execution confidence
MAX_POSITIONS = 10                 # Maximum open positions

# Supported Cryptocurrencies
SUPPORTED_PAIRS = [
    "BTCUSDT", "ETHUSDT", "ADAUSDT", "DOTUSDT", "LINKUSDT",
    "BNBUSDT", "XRPUSDT", "LTCUSDT", "BCHUSDT", "EOSUSDT"
]

# ML Configuration
ML_LOOKBACK_PERIODS = 500          # Data for RFE selection
CATBOOST_ITERATIONS = 1000         # Model complexity
CATBOOST_DEPTH = 10                # Tree depth
RFE_N_FEATURES = 50                # Selected features

# Risk Management
DEFAULT_TP_LEVELS = [0.02, 0.04, 0.06, 0.08, 0.10]  # TP percentages
DEFAULT_SL_PERCENTAGE = 0.02       # Initial SL percentage
TP_SL_TRAILING_STEP = 0.01         # Trailing step size

# Demo Mode
DEMO_MODE = True                   # Start in demo mode
DEMO_BALANCE = 10000.0             # Starting balance
```

## 📊 Data Sources

### Binance API Integration
- **Historical Data**: OHLCV data up to 365 days
- **Real-time Prices**: Live price feeds with order book data
- **Multiple Timeframes**: 1m, 5m, 15m, 1h, 4h, 1d support
- **Rate Limiting**: Built-in rate limit handling

### Data Storage
- **SQLite Database**: Persistent data storage
- **Market Data Table**: Historical OHLCV storage
- **Real-time Prices**: Current market prices
- **Position History**: Complete trading history
- **Model Metadata**: Training results and performance

## 🧪 Testing

### Basic System Test
```bash
python test_basic.py
```

### Comprehensive Test (requires dependencies)
```bash
python test_system.py
```

### Test Coverage
- ✅ Component initialization
- ✅ Data collection and storage
- ✅ Indicator calculations (435+ indicators)
- ✅ ML model training and prediction
- ✅ Risk management system
- ✅ Trading engine integration
- ✅ Web application functionality
- ✅ End-to-end workflow

## 🚀 Usage

### Demo Mode (Recommended for testing)
1. Start system: `python main.py`
2. Open dashboard: `http://localhost:5000`
3. Monitor indicator status and model training
4. Watch live signals and portfolio performance
5. Configure TP/SL settings per symbol
6. Test with paper trading

### Live Trading Mode
1. Configure API keys in `config/settings.py`
2. Set `DEMO_MODE = False`
3. Start with small position sizes
4. Monitor performance closely
5. Adjust confidence threshold based on results

## 📈 Performance Optimization

### System Performance
- **Async Operations**: Non-blocking I/O for data collection
- **Caching**: Price and indicator caching for efficiency
- **Database Indexing**: Optimized queries for historical data
- **Memory Management**: Efficient pandas operations

### ML Model Optimization
- **Feature Selection**: RFE reduces overfitting
- **Cross-validation**: 3-fold CV for robust feature selection
- **Early Stopping**: Prevents overfitting during training
- **Bayesian Bootstrap**: Improved sampling for small datasets

## 🔒 Security

### API Security
- **Environment Variables**: Secure API key storage
- **Rate Limiting**: Built-in Binance rate limit compliance
- **Error Handling**: Graceful handling of API failures
- **Demo Mode**: Safe testing without real trading

### Data Security
- **Local Storage**: All data stored locally
- **Encrypted Connections**: HTTPS for all API calls
- **Access Control**: Web interface access controls
- **Audit Trail**: Complete trading history logging

## 🐛 Troubleshooting

### Common Issues

**Installation Problems**
```bash
# Install TA-Lib separately if needed
pip install TA-Lib

# For Windows users
pip install ta-lib-binary
```

**API Connection Issues**
- Check internet connection
- Verify API keys (for live mode)
- Check Binance API status
- Review rate limiting

**Model Training Issues**
- Ensure sufficient historical data (>200 candles)
- Check indicator calculations
- Verify feature data quality
- Review training logs

**Dashboard Issues**
- Clear browser cache
- Check console for JavaScript errors
- Verify WebSocket connection
- Test API endpoints manually

## 📝 Logging

### Log Levels
```python
LOG_LEVEL = "INFO"                 # INFO, DEBUG, WARNING, ERROR
LOG_FILE = "logs/trading_system.log"
MAX_LOG_SIZE = 10                  # MB
BACKUP_COUNT = 5                   # Rotating logs
```

### Log Categories
- **System**: Startup, shutdown, component status
- **Data**: Data collection, storage, retrieval
- **ML**: Model training, prediction, performance
- **Trading**: Signal generation, position management
- **Risk**: Position updates, P&L calculations
- **Web**: API requests, WebSocket connections

## 🤝 Contributing

### Development Setup
1. Fork the repository
2. Create feature branch
3. Install development dependencies
4. Run tests before committing
5. Submit pull request

### Code Style
- Follow PEP 8 guidelines
- Use type hints where possible
- Add docstrings for functions
- Include error handling
- Write unit tests for new features

## 📄 License

This project is licensed under the MIT License - see the LICENSE file for details.

## ⚠️ Disclaimer

**Important**: This software is for educational and research purposes only. Cryptocurrency trading involves substantial risk of loss and is not suitable for all investors. The software is provided "as is" without warranty of any kind. Past performance does not guarantee future results. Always conduct thorough testing in demo mode before live trading.

## 📞 Support

For questions, issues, or suggestions:
1. Check the troubleshooting section
2. Review existing GitHub issues
3. Create a new issue with detailed information
4. Include system logs and configuration

---

**🎯 Built with advanced AI and machine learning for professional cryptocurrency trading**