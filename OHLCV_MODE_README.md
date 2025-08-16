# OHLCV-Only Mode Implementation

This document describes the architectural changes implemented to transition from the 400+ indicator encyclopedia to OHLCV-only mode with 1-minute timeframe.

## Key Changes

### 1. Configuration Updates (config/settings.py)
- `DEFAULT_TIMEFRAME = "1m"` - Primary timeframe set to 1 minute
- `RFE_N_FEATURES = 25` - Reduced feature count for OHLCV-only mode
- `INITIAL_COLLECTION_DURATION_SEC = 0` - Bootstrap collection disabled
- `ENABLE_TICK_BOOTSTRAP = False` - Tick-based bootstrap bypassed
- `RFE_SELECTION_CANDLES = 1000` - Recent window for RFE selection
- `BASE_MUST_KEEP_FEATURES = ["open", "high", "low", "close", "volume"]` - Core OHLCV features
- `INDICATOR_CSV_PATH = "ohlcv_only_indicators.csv"` - OHLCV-only indicators source

### 2. Data Flow Changes
**Old Flow (Bootstrap Mode):**
1. Bootstrap tick collection (1 hour)
2. Tick aggregation to OHLC
3. 400+ indicator calculation  
4. Full dataset RFE
5. Model training

**New Flow (OHLCV-Only Mode):**
1. Skip bootstrap entirely
2. Read from preloaded `candles` table
3. Calculate OHLCV-only indicators from CSV (~200 features)
4. RFE on recent 1000 candles, select top 25
5. Train model on full historical dataset with selected features

### 3. Components Modified

#### A. Indicator Engine (src/indicators/indicator_engine.py)
- Loads from `ohlcv_only_indicators.csv` instead of large encyclopedia
- Implements `get_must_keep_features()` with BASE_MUST_KEEP_FEATURES
- Fallback to default OHLCV indicators if CSV missing
- Enhanced feature categorization (must-keep vs RFE candidates)

#### B. CatBoost Model (src/ml_model/catboost_model.py)
- RFE runs on recent window (1000 candles) for speed
- Final training uses full historical dataset
- Model preservation: keeps previous model if new one underperforms
- Enhanced logging with feature counts and performance metrics

#### C. Main Startup (main.py)
- Bypasses all bootstrap phases (4-7)
- New OHLCV-only startup flow:
  - Initialize candle data manager
  - Load full historical OHLCV data
  - Calculate indicators and train model
  - Start trading operations

#### D. Data Access (src/data_access/candle_data_manager.py)
- New module for reading from `candles` table
- Supports multiple symbols: BTCUSDT, ETHUSDT, SOLUSDT, DOGEUSDT
- Validation and safeguards for missing data
- Recent window extraction for RFE

#### E. Trading Engine (src/trading_engine/trading_engine.py)
- Updated retrain workflow for OHLCV-only mode
- Status API shows OHLCV-specific metrics
- Feature categorization in web interface

### 4. Database Requirements

The system expects a `candles` table with the following schema:

```sql
CREATE TABLE candles (
    id BIGINT AUTO_INCREMENT PRIMARY KEY,
    symbol VARCHAR(20) NOT NULL,
    timestamp BIGINT NOT NULL,
    open DECIMAL(20,8) NOT NULL,
    high DECIMAL(20,8) NOT NULL,
    low DECIMAL(20,8) NOT NULL,
    close DECIMAL(20,8) NOT NULL,
    volume DECIMAL(20,8) NOT NULL,
    datetime DATETIME NOT NULL,
    INDEX idx_symbol_timestamp (symbol, timestamp),
    UNIQUE KEY unique_symbol_timestamp (symbol, timestamp)
);
```

### 5. OHLCV Indicators CSV Structure

The `ohlcv_only_indicators.csv` file should contain:
- **Indicator**: Feature name
- **Category**: Group (e.g., Price Action, Volume, Momentum)
- **Required Inputs**: Base OHLCV columns needed
- **Must Keep (Not in RFE)**: "Yes" if feature should never be removed
- **RFE Eligible**: "Yes" if feature can be selected/removed by RFE

### 6. Web Interface Updates

#### Status API Enhancements:
- `mode: "ohlcv_only"`
- `timeframe: "1m"`
- `rfe_target_features: 25`
- `rfe_window_size: 1000`
- Bootstrap progress shows "completed" (bypassed)

#### Feature Display:
- Base OHLCV features (always present)
- Must-keep features (prerequisites)
- RFE-selected features (top 25)
- RFE candidates not selected

### 7. Retrain Workflow

**Manual Retrain Button:**
1. Fetch latest 1000 1m candles per symbol
2. Run RFE on recent window
3. Apply selected features to full historical dataset
4. Train new model with preservation logic
5. Keep new model only if performance is acceptable

### 8. Safeguards

- **Data Validation**: Check `candles` table exists and has data
- **Symbol Validation**: Warn if symbols missing, continue with available
- **Minimum Samples**: Require MIN_INITIAL_TRAIN_SAMPLES for training
- **Model Preservation**: Revert to previous model if new training fails
- **Fallback Indicators**: Use defaults if CSV missing

### 9. Testing

Run the implementation test:
```bash
python test_ohlcv_implementation.py
```

This validates:
- Configuration settings
- CSV structure and content  
- Code changes integration

### 10. Migration Notes

- **No Data Loss**: Existing tables preserved, just not used for training
- **Backward Compatible**: System can fall back to defaults if needed
- **Configurable**: CSV path and settings can be modified
- **Future-Proof**: Can expand to non-OHLCV indicators later by changing CSV

## Performance Benefits

1. **Faster Startup**: No 1-hour bootstrap collection
2. **Efficient RFE**: Runs on 1000 recent samples vs full dataset
3. **Reduced Feature Space**: ~200 OHLCV features vs 400+ encyclopedia
4. **1m Resolution**: Higher frequency trading signals
5. **Model Preservation**: Avoids degradation from poor retraining