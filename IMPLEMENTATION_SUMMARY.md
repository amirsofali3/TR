# OHLCV Classification & Symbol Encoding Fixes - Implementation Summary

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

### Before Fix:
```
[INDICATORS] Computed: 145, Must keep: 2, RFE candidates: 0
[TRAIN] No RFE-eligible features found  
[TRAIN] Bad value for num_feature ... "ETHUSDT": symbol column error
UI shows only Timestamp & Volume as active indicators
PerformanceWarning: DataFrame is highly fragmented
```

### After Fix:
```
[INDICATORS] Computed: 145, Must keep: ≥7, RFE candidates: >0  
[TRAIN] RFE selected N features out of M eligible
[TRAIN] Encoded symbol column symbol to symbol_code with K categories
UI shows correct OHLCV feature categorization
No pandas fragmentation warnings
```

## Validation Steps

1. **Must-Keep Count**: Should be ≥7 (base OHLCV + timestamp + symbol)
2. **RFE Candidates**: Should be >0 with majority of computed indicators
3. **RFE Training**: No "No RFE-eligible features found" warnings
4. **CatBoost Training**: No symbol conversion errors 
5. **UI Display**: Correct feature categorization in active indicators panel
6. **Performance**: No pandas DataFrame fragmentation warnings

## Files Modified
- `src/indicators/indicator_engine.py` - Core classification logic
- `src/ml_model/catboost_model.py` - Symbol handling in sanitization  
- `src/trading_engine/trading_engine.py` - UI categorization with case-insensitive matching

The implementation preserves existing public method signatures and includes guards for OHLCV-only mode to avoid breaking other functionality.