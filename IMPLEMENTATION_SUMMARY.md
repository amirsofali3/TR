# OHLCV Classification & Symbol Encoding Fixes - Implementation Summary

## Phase 2 Fixes (Second Round) ✅

### A. Enhanced CatBoost Model Training

**1. Feature Sanitization Integration** ✅
- **Issue**: Symbol encoding and RFE selection errors due to non-numeric features
- **Solution**: 
  - Enhanced `_sanitize_features()` to store stats in `self.last_sanitization_stats`
  - Added early sanitization in `train_model()` for both X and X_recent before any processing
  - Added sanitization in `perform_rfe_selection()` before filtering rfe_features
  - Added sanitization in `select_features_by_importance()` fallback method

**2. CatBoost Train Directory Configuration** ✅
- **Issue**: RFE selection error "Can't create train working dir: catboost_info"
- **Solution**:
  - Added dedicated temporary directories: `catboost_rfe_tmp`, `catboost_importance_tmp`
  - Set `train_dir`, `allow_writing_files=True`, `verbose=False` for all temporary models
  - Used `os.makedirs(..., exist_ok=True)` for directory creation

**3. Symbol Encoding Pipeline** ✅  
- **Issue**: String symbol column causing "Cannot convert 'ETHUSDT' to float" errors
- **Solution**: Symbol sanitization now happens before any model training or RFE selection

### B. Indicator Engine Verification ✅
- **Status**: Base OHLCV features (open, high, low, close, volume, timestamp, symbol) already properly included in must-keep from Phase 1

### C. Trading Engine UI Features ✅
- **Status**: Case-insensitive OHLCV feature categorization already implemented with timestamp & symbol support

## Phase 1 Fixes (Previous Implementation)

## Issues Fixed

### 1. Must-Keep Detection Issues ✅
**Problem**: Only considered features produced by indicator functions; base OHLCV columns not always included in must_keep set.

**Solution**: 
- Added base OHLCV features (`["open", "high", "low", "close", "volume", "timestamp", "symbol"]`) to must_keep even if not computed by indicator functions
- Enhanced logic in `calculate_all_indicators()` to include original DataFrame columns in computed_indicators set when they match base features
- Added case-insensitive mapping between computed features and base features

### 2. RFE Classification Problems ✅
**Problem**: Simplistic base name extraction (`split('_')[0]`) failed for complex indicators and used case-sensitive matching.

**Solution**:
- Added `_sanitize_indicator_name()` helper for consistent case-insensitive name normalization
- Added `_build_indicator_classification_sets()` to create case-insensitive lookup sets from CSV
- Implemented sophisticated `_extract_base_indicator_name()` that handles:
  - Special characters (e.g., `Stoch_%K_5`, `Williams_%R_14`)
  - Multi-part indicators (e.g., `Bollinger_Width_10_x2.0`)
  - Progressive matching strategies for complex names
- Built classification map for each computed feature → base indicator name

### 3. Symbol Encoding Issue ✅
**Problem**: Symbol column passed as string to CatBoost causing training errors.

**Solution**: Enhanced `_sanitize_features()` with special symbol column handling:
- **Constant symbols** (≤1 unique): Drop entirely (no information value)
- **Low cardinality** (≤50 symbols): Encode to `symbol_code` using LabelEncoder
- **High cardinality** (>50 symbols): Drop to prevent overfitting
- Added comprehensive logging for symbol processing actions

### 4. DataFrame Fragmentation ✅
**Problem**: Column-by-column assignment (`features_df[col_name] = series`) caused pandas fragmentation warnings.

**Solution**:
- Build all indicators in `results` dictionary first
- Create indicators DataFrame with single `pd.DataFrame(results)` call
- Combine with original data using single `pd.concat([df, indicators_df], axis=1)`
- Eliminates iterative column insertion that causes fragmentation

### 5. UI Categorization Issues ✅
**Problem**: Case-sensitive matching in trading_engine failed to properly categorize features.

**Solution**: Updated `_get_ohlcv_features_info()` with:
- Case-insensitive matching using `sanitize_feature_name()` helper
- Inclusion of timestamp & symbol in base feature set
- Proper categorization into base_ohlcv, must_keep_other, rfe_selected, rfe_not_selected

## Key Implementation Details

### A. indicator_engine.py Changes
```python
# New helper methods
def _sanitize_indicator_name(self, name: str) -> str
def _build_indicator_classification_sets(self)  
def _extract_base_indicator_name(self, feature_name: str) -> str

# Enhanced calculate_all_indicators with:
# - Phase 1: Calculate indicators 
# - Phase 2: Build classification map with sophisticated name extraction
# - Phase 3: Classify features using case-insensitive CSV-based sets
# - Phase 4: Build DataFrame with single concat to avoid fragmentation
```

### B. catboost_model.py Changes  
```python
# Enhanced _sanitize_features with special symbol handling
if column.lower() == 'symbol':
    if unique_count <= 1:
        # Drop constant symbol
    elif unique_count <= 50:
        # Encode to symbol_code  
    else:
        # Drop high-cardinality symbol
```

### C. trading_engine.py Changes
```python
# Case-insensitive feature categorization
def sanitize_feature_name(name: str) -> str
base_features = BASE_MUST_KEEP_FEATURES + ['timestamp', 'symbol']
# Case-insensitive lookup sets for proper UI categorization
```

## Expected Behavior Changes

### Phase 2 - Before Fix:
```
[TRAIN] RFE selection failed: Can't create train working dir: catboost_info
[TRAIN] Feature importance selection failed: Bad value for num_feature ... "ETHUSDT": Cannot convert 'b'ETHUSDT'' to float
[TRAIN] Initial model training failed due to string symbol column
UI Active Indicators still shows limited features (selected_features stays empty)
```

### Phase 2 - After Fix:
```
[TRAIN] Sanitizing main training features...
[TRAIN] Main sanitization: 145 → 144 features
[TRAIN] Encoded symbol column symbol to symbol_code with 4 categories
[TRAIN] Sanitizing RFE window features...
[TRAIN] RFE sanitization: 144 → 144 features
[TRAIN] RFE selected 25 features out of 137 eligible
[TRAIN] Selected feature count equals Must keep + target RFE features (7 + 25 = 32)
UI shows Active Features > 0 and Feature Selection Summary populated
```

### Phase 1 - Before Fix:
```
[INDICATORS] Computed: 145, Must keep: 2, RFE candidates: 0
[TRAIN] No RFE-eligible features found  
[TRAIN] Bad value for num_feature ... "ETHUSDT": symbol column error
UI shows only Timestamp & Volume as active indicators
PerformanceWarning: DataFrame is highly fragmented
```

### Phase 1 - After Fix:
```
[INDICATORS] Computed: 145, Must keep: ≥7, RFE candidates: >0  
[TRAIN] RFE selected N features out of M eligible
[TRAIN] Encoded symbol column symbol to symbol_code with K categories
UI shows correct OHLCV feature categorization
No pandas fragmentation warnings
```

## Validation Steps

### Phase 2 Acceptance Criteria ✅
1. **RFE Stage Completion**: RFE stage completes (or falls back) without catboost_info directory error
2. **Symbol Encoding**: No string conversion errors in training (symbol encoded or dropped)  
3. **Feature Count**: Selected feature count equals Must keep + target RFE features (e.g., 7 + 25 = 32) or fewer if limited by available features
4. **UI Display**: UI shows Active Features > 0 and Feature Selection Summary populated
5. **Sanitization Stats**: Sanitization stats logged with counts (stored in self.last_sanitization_stats)

### Phase 1 Validation ✅
1. **Must-Keep Count**: Should be ≥7 (base OHLCV + timestamp + symbol)
2. **RFE Candidates**: Should be >0 with majority of computed indicators
3. **RFE Training**: No "No RFE-eligible features found" warnings
4. **CatBoost Training**: No symbol conversion errors 
5. **UI Display**: Correct feature categorization in active indicators panel
6. **Performance**: No pandas DataFrame fragmentation warnings

---

## Phase 3 Comprehensive Stabilization ✅

### A. Critical Row Preservation Fix ✅
**Problem**: Sanitization dropping rows (57913→57616) causing X/y length mismatch
**Solution**:
- **Non-destructive `_sanitize_features()`**: Never drops rows, only columns
- **`_impute_missing()` helper**: Forward fill → back fill → zero/median strategy  
- **Row alignment verification**: Assert len(X)==len(y) after sanitization
- **Automatic realignment**: `y = y.loc[X.index]` if mismatch detected
- **Transparent logging**: "Samples: 57913 → 57913 (MUST BE EQUAL)"

### B. Optional RFE with Robust Fallbacks ✅
**Configuration**:
- **`ENABLE_RFE = True`**: Optional RFE flag (can be disabled)
- **`RFE_FALLBACK_TOP_N = 50`**: Top N features when RFE disabled/fails
- **`MIN_ANALYSIS_CANDLES = 150`**: Live analysis data fallback

**Training Flow**:
- **`perform_full_training_flow()`**: Comprehensive wrapper with 3-tier fallback
  1. Full training with RFE (if enabled)
  2. Simplified training with all numeric features
  3. Minimal training with must-keep features only
- **`selected_features` guarantee**: Never empty, fallbacks to must_keep → all numeric
- **Stable CatBoost artifacts**: `train_dir=catboost_main` prevents catboost_info errors

### C. Enhanced Data Handling ✅
**Live Analysis Improvements**:
- **`analyze_symbol()` fallback**: Fetch `MIN_ANALYSIS_CANDLES*2` from DB if live insufficient
- **Transparent data sourcing**: Log when extended fetch successful vs live buffer

**Indicator Engine Enhancements**:
- **`get_all_feature_names()`**: Returns base OHLCV + computed indicators
- **Skip reason tracking**: Already implemented with transparency logging

### D. Improved UI Endpoints ✅
**`/api/indicators` endpoint**:
- **Baseline mode**: Show must_keep + base OHLCV as "active" when model not trained
- **Training mode**: Show `selected_features` after successful training
- **Status indicators**: `indicator_status: 'baseline'|'trained'`, `rfe_enabled`, `skipped_count`
- **Transparency**: First 5 skipped indicators with reasons

### E. Stability Guarantees
**Training Pipeline**:
✅ **Row count preservation**: Sanitization maintains exact sample count
✅ **Feature selection robustness**: Multiple fallback strategies prevent empty selection
✅ **Error recovery**: Comprehensive training flow with graceful degradation
✅ **Data sufficiency**: Automatic fallback to extended historical data
✅ **UI consistency**: Always shows meaningful active features

**Expected Behavior**:
```
[TRAIN] NON-DESTRUCTIVE feature sanitization completed:
[TRAIN]   Features: 145 → 144
[TRAIN]   Samples: 57913 → 57913 (MUST BE EQUAL)
[TRAIN] X/y alignment verified: 57913 samples
[TRAIN] RFE is enabled, performing feature selection...
[TRAIN] Selected 25 features
[TRAIN] Model training complete
UI: Active Features shows >0 indicators
```

## Files Modified

### Phase 3 Changes ✅
- `config/settings.py` - Added ENABLE_RFE, MIN_ANALYSIS_CANDLES, RFE_FALLBACK_TOP_N
- `src/ml_model/catboost_model.py` - Non-destructive sanitization, comprehensive training flow, stable train_dir
- `src/indicators/indicator_engine.py` - Added get_all_feature_names() method
- `src/trading_engine/trading_engine.py` - Enhanced analyze_symbol() with data fallback
- `src/web_app/app.py` - Improved /api/indicators endpoint with baseline/trained modes
- `IMPLEMENTATION_SUMMARY.md` - Phase 3 documentation

### Phase 2 Changes
- `src/ml_model/catboost_model.py` - Enhanced feature sanitization integration, CatBoost train_dir configuration

### Phase 1 Changes  
- `src/indicators/indicator_engine.py` - Core classification logic
- `src/ml_model/catboost_model.py` - Symbol handling in sanitization  
- `src/trading_engine/trading_engine.py` - UI categorization with case-insensitive matching

The implementation preserves existing public method signatures and includes guards for OHLCV-only mode to avoid breaking other functionality.