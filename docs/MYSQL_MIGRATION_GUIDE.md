# MySQL Migration and Training Improvements - User Guide

This guide covers the new MySQL migration support and ML training pipeline improvements implemented in the crypto trading system.

## Overview

The system has been enhanced with:
- **Database abstraction** supporting both SQLite and MySQL
- **Automatic initial training** when the system starts
- **Fallback signals** when the ML model is not trained
- **Improved logging** and warning management
- **Migration utilities** to transfer existing data

## Database Configuration

### SQLite (Default)
The system continues to use SQLite by default with no configuration changes needed.

### MySQL Support
To use MySQL instead of SQLite, set the following environment variables:

```bash
export MYSQL_ENABLED=true
export MYSQL_HOST=localhost
export MYSQL_PORT=3306
export MYSQL_USER=your_username
export MYSQL_PASSWORD=your_password
export MYSQL_DB=trading_system
export MYSQL_CHARSET=utf8mb4
```

### Configuration in settings.py
New settings have been added to `config/settings.py`:

```python
# MySQL Configuration (MySQL migration)
MYSQL_ENABLED = False
MYSQL_HOST = "localhost"
MYSQL_PORT = 3306
MYSQL_USER = "root" 
MYSQL_PASSWORD = ""
MYSQL_DB = "trading_system"
MYSQL_CHARSET = "utf8mb4"
MYSQL_TIMEZONE = "+00:00"

# ML Training Pipeline Configuration (MySQL migration)
MIN_INITIAL_TRAIN_SAMPLES = 400  # Minimum samples required for initial training
MIN_RFE_SAMPLES = 150  # Minimum samples for RFE, otherwise use baseline features
MAX_BASELINE_FEATURES = 120  # Max features when RFE is skipped
TRAIN_RETRY_COOLDOWN_MIN = 10  # Minutes to wait before retrying failed training

# Fallback Strategy Configuration (MySQL migration) 
ALLOW_FALLBACK_EXECUTION = False  # Allow fallback signals to execute trades
```

## Database Migration

### Setting up MySQL Schema
1. Create the database and tables using the provided schema:
```bash
mysql -u root -p < docs/mysql_schema.sql
```

### Migrating Existing Data
Use the migration script to transfer data from SQLite to MySQL:

```bash
# Set MySQL environment variables first
export MYSQL_ENABLED=true
export MYSQL_HOST=localhost
export MYSQL_USER=root
export MYSQL_PASSWORD=your_password
export MYSQL_DB=trading_system

# Run migration (dry-run first)
python scripts/migrate_sqlite_to_mysql.py --dry-run --verbose

# Run actual migration
python scripts/migrate_sqlite_to_mysql.py --verbose
```

### Migration Options
- `--dry-run`: Preview changes without actually migrating data
- `--batch-size N`: Process N rows at a time (default: 1000)
- `--tables table1 table2`: Migrate specific tables only
- `--verbose`: Enable detailed logging

## Training Improvements

### Automatic Initial Training
The system now automatically attempts to train the ML model during startup:

1. **Trigger Conditions**: Model training is triggered when:
   - The ML model is not already trained
   - No training cooldown is active
   - Sufficient historical data is available (≥ MIN_INITIAL_TRAIN_SAMPLES)

2. **Training Process**:
   - Fetches historical data for the primary symbol (BTCUSDT)
   - Calculates technical indicators
   - Performs feature selection (RFE or importance-based fallback)
   - Trains the CatBoost model

3. **Sample Size Guards**:
   - Minimum 400 samples required for initial training
   - Minimum 150 samples required for RFE feature selection
   - Falls back to importance-based selection with max 120 features

### Training Cooldown
If training fails, the system implements a cooldown period (default: 10 minutes) before attempting to retrain, preventing log spam and excessive resource usage.

### Enhanced Logging
Training operations now use structured logging with `[TRAIN]` prefix:
```
[TRAIN] Preparing features and labels...
[TRAIN] Raw features: 431, samples: 500
[TRAIN] Samples after dropna: 485 (dropped 15)
[TRAIN] Class distribution - SELL: 162, HOLD: 161, BUY: 162
[TRAIN] Final dataset: 485 samples with 431 features
```

## Fallback Strategy

### When Fallback is Used
The system uses fallback signals when:
- The ML model is not trained
- Training failed and cooldown is active

### Fallback Signal Generation
The fallback strategy uses simple technical analysis:
- **SMA Crossover**: 14-period vs 50-period Simple Moving Average
- **RSI Thresholds**: Relative Strength Index for overbought/oversold conditions
- **Signal Logic**:
  - **BUY**: SMA(14) > SMA(50) and RSI < 70
  - **SELL**: SMA(14) < SMA(50) and RSI > 30
  - **HOLD**: All other conditions

### Fallback Confidence
Fallback signals generate confidence scores between 55-65% based on the strength of the technical indicators.

### Execution Control
Fallback signal execution can be controlled via the `ALLOW_FALLBACK_EXECUTION` setting:
- `False` (default): Fallback signals are generated but not executed (safe mode)
- `True`: Fallback signals can trigger actual trades

## Warning Management

### Reduced Spam
The system now reduces warning spam:
- **"Model is not trained"** warnings are logged only every 10th prediction
- **Missing features** warnings are logged once per session
- **Training failures** use cooldown to prevent repeated attempts

### Prediction Counters
The system tracks prediction warning counts and includes them in model status information.

## API Enhancements

### Enhanced Model Status Endpoint
The model status endpoint now includes additional information:
```json
{
  "is_trained": false,
  "training_progress": 0,
  "fallback_active": true,
  "last_training_time": null,
  "samples_in_last_train": 0,
  "class_distribution": {},
  "training_cooldown_active": false,
  "prediction_warning_count": 23
}
```

### Fallback Signal Indication
Analysis results now include a `fallback` field to indicate whether the signal was generated by the ML model or fallback strategy:
```json
{
  "symbol": "BTCUSDT",
  "prediction": "BUY",
  "confidence": 58.5,
  "fallback": true,
  "fallback_indicators": {
    "sma_14": 50125.30,
    "sma_50": 49875.60,
    "rsi": 65.2
  }
}
```

## Testing

### Running Tests
The system includes comprehensive tests:

```bash
# Database backend tests
python test/test_db_backend.py

# Initial training tests
python test/test_initial_training.py

# Fallback signal tests
python test/test_fallback_signal.py
```

### Test Coverage
- Database abstraction layer (SQLite/MySQL switching)
- Initial training trigger and execution
- Fallback signal generation for different market conditions
- Signal structure validation
- Configuration handling

## Troubleshooting

### Common Issues

1. **MySQL Connection Failed**
   - Verify environment variables are set correctly
   - Check MySQL server is running and accessible
   - Ensure user has appropriate permissions

2. **Training Fails Immediately**
   - Check if sufficient historical data exists (≥ 400 samples)
   - Verify indicators are calculating correctly
   - Check system logs for specific error messages

3. **Migration Script Errors**
   - Run with `--dry-run` first to identify issues
   - Check source SQLite database exists and is readable
   - Ensure MySQL database exists and user has write permissions

4. **Fallback Signals Not Appearing**
   - Verify model is not trained (`is_trained = False`)
   - Check if sufficient price data exists for SMA calculations
   - Enable debug logging to see fallback signal generation

### Debug Mode
Enable verbose logging to troubleshoot issues:
```python
# In your main script
import logging
logging.getLogger('src.database.db_manager').setLevel(logging.DEBUG)
logging.getLogger('src.ml_model.catboost_model').setLevel(logging.DEBUG)
```

## Performance Considerations

### Database Performance
- MySQL with proper indexing may perform better than SQLite for large datasets
- Consider using connection pooling for high-frequency trading
- Monitor query performance and optimize as needed

### Training Performance
- RFE feature selection can be time-consuming with large feature sets
- Consider adjusting `MIN_RFE_SAMPLES` threshold based on your hardware
- Training runs asynchronously to avoid blocking the main analysis loop

## Migration Checklist

- [ ] Set up MySQL database and user permissions
- [ ] Configure MySQL environment variables
- [ ] Run migration script with `--dry-run` to verify
- [ ] Execute full migration
- [ ] Test system startup with MySQL backend
- [ ] Verify fallback signals are working
- [ ] Monitor initial training process
- [ ] Update any custom scripts to use new database abstraction

## Follow-up Fixes (Latest Updates)

### 🔧 CatBoost Parameter Fix
- **Fixed**: Resolved CatBoost parameter conflict causing training failures
- Removed incompatible `subsample` parameter when using Bayesian bootstrap  
- Added automatic class weight computation for imbalanced datasets

### 🗄️ Auto Schema Creation
- **New**: Automatic database table creation when database is empty
- Set `AUTO_CREATE_SCHEMA=true` (default) to enable
- Configurable table names via `MYSQL_MARKET_DATA_TABLE` environment variable
- Creates core tables: market_data, real_time_prices, positions

### 🔍 Enhanced MySQL Detection  
- **Improved**: Robust credential validation with clear error messages
- Lists missing required variables: MYSQL_HOST, MYSQL_USER, MYSQL_PASSWORD, MYSQL_DB
- `AUTO_FALLBACK_DB=true` (default) enables automatic SQLite fallback
- Set `AUTO_FALLBACK_DB=false` to force MySQL-only mode

### 📊 Training Diagnostics & Status
- **New**: Comprehensive training diagnostics in status endpoint
- Added fields: `last_training_error`, `class_distribution`, `class_weights`, `sanitization`
- Feature sanitization metadata tracking (initial/final counts, dropped features)
- Training retry scheduling with `next_retry_at` timestamp

### 🔄 Retry Logic Improvements
- **Enhanced**: Configurable retry cooldown via `TRAIN_RETRY_COOLDOWN_SEC` (default: 600s)  
- Automatic retry scheduling after training failures
- Clear logging of next retry time and remaining cooldown
- Retry attempts occur during regular market analysis cycles

### 📈 New Configuration Options

```bash
# Auto Schema Creation
export AUTO_CREATE_SCHEMA=true          # Create missing DB tables

# MySQL Fallback Control  
export AUTO_FALLBACK_DB=true            # Fallback to SQLite if MySQL fails

# Configurable Table Name
export MYSQL_MARKET_DATA_TABLE=market_data  # Custom market data table name

# Training Retry Settings
export TRAIN_RETRY_COOLDOWN_SEC=600     # Retry cooldown in seconds (10 min)
```

### 🧪 Test Coverage
- **New**: 4 test files with 12+ focused test cases
- Auto schema creation validation
- MySQL misconfiguration handling
- CatBoost parameter conflict prevention  
- Status diagnostics integrity testing

## Next Steps

1. **Monitor Performance**: Compare SQLite vs MySQL performance in your environment
2. **Tune Parameters**: Adjust training thresholds based on your data characteristics
3. **Extend Fallback**: Consider adding more sophisticated fallback strategies
4. **Scale Testing**: Test with larger datasets and multiple symbols
5. **Production Deployment**: Plan rolling deployment with fallback to SQLite if needed