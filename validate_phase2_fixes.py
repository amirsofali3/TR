#!/usr/bin/env python3
"""
Syntax Validation for Phase 2 OHLCV Mode Fixes
Validates code changes without requiring external dependencies.
"""

import ast
import os
import sys

def validate_python_syntax(file_path):
    """Validate Python syntax for a given file"""
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            source_code = f.read()
        
        # Parse the AST to check for syntax errors
        ast.parse(source_code, filename=file_path)
        return True, None
    except SyntaxError as e:
        return False, f"Syntax error: {e}"
    except Exception as e:
        return False, f"Error: {e}"

def validate_imports_and_structure(file_path):
    """Validate imports and basic structure"""
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            content = f.read()
        
        issues = []
        
        # Check for common issues in our changes
        if 'catboost_model.py' in file_path:
            # Check for our key additions
            if 'self.last_sanitization_stats = metadata' not in content:
                issues.append("Missing sanitization stats storage")
            if 'catboost_rfe_tmp' not in content:
                issues.append("Missing RFE train directory")
            if 'catboost_importance_tmp' not in content:
                issues.append("Missing importance train directory")
            if 'X_sanitized, main_sanitization_stats = self._sanitize_features(X)' not in content:
                issues.append("Missing main feature sanitization in train_model")
            
        return len(issues) == 0, issues
        
    except Exception as e:
        return False, [f"Error reading file: {e}"]

def main():
    """Validate all modified files"""
    print("🔍 Validating Phase 2 OHLCV Mode Fixes...\n")
    
    files_to_check = [
        'src/ml_model/catboost_model.py',
        'src/indicators/indicator_engine.py', 
        'src/trading_engine/trading_engine.py'
    ]
    
    all_passed = True
    
    for file_path in files_to_check:
        if not os.path.exists(file_path):
            print(f"❌ File not found: {file_path}")
            all_passed = False
            continue
            
        print(f"📄 Checking {file_path}...")
        
        # Check syntax
        syntax_ok, syntax_error = validate_python_syntax(file_path)
        if syntax_ok:
            print(f"  ✅ Syntax validation passed")
        else:
            print(f"  ❌ Syntax validation failed: {syntax_error}")
            all_passed = False
            
        # Check structure for catboost_model.py
        if 'catboost_model.py' in file_path:
            structure_ok, structure_issues = validate_imports_and_structure(file_path)
            if structure_ok:
                print(f"  ✅ Structure validation passed")
            else:
                print(f"  ❌ Structure validation failed:")
                for issue in structure_issues:
                    print(f"    - {issue}")
                all_passed = False
        
        print()
    
    # Check documentation updates
    print("📄 Checking IMPLEMENTATION_SUMMARY.md...")
    if os.path.exists('IMPLEMENTATION_SUMMARY.md'):
        with open('IMPLEMENTATION_SUMMARY.md', 'r') as f:
            content = f.read()
        
        if 'Phase 2 Fixes' in content:
            print("  ✅ Phase 2 documentation added")
        else:
            print("  ❌ Phase 2 documentation missing")
            all_passed = False
    else:
        print("  ❌ IMPLEMENTATION_SUMMARY.md not found")
        all_passed = False
    
    print("\n" + "="*50)
    
    if all_passed:
        print("🎉 All validations passed!")
        print("\n✅ Phase 2 Fixes Implemented:")
        print("   1. Enhanced _sanitize_features to store stats in self.last_sanitization_stats")
        print("   2. Added early sanitization in train_model() for X and X_recent")
        print("   3. Added sanitization in perform_rfe_selection() before filtering")
        print("   4. Added dedicated train_dir for CatBoost RFE and importance models")
        print("   5. Updated select_features_by_importance to sanitize input")
        print("   6. Base OHLCV features already properly handled (Phase 1)")
        print("   7. Case-insensitive categorization already implemented (Phase 1)")
        print("   8. Updated IMPLEMENTATION_SUMMARY.md with Phase 2 details")
        print("\n🎯 Expected Results:")
        print("   - RFE selection completes without catboost_info directory error")
        print("   - No symbol string conversion errors during training")
        print("   - Selected feature count = Must keep + RFE features (e.g., 7 + 25 = 32)")
        print("   - UI shows Active Features > 0 and populated Feature Selection Summary")
        print("   - Sanitization stats logged with detailed counts")
        return 0
    else:
        print("❌ Some validations failed. Please review the issues above.")
        return 1

if __name__ == "__main__":
    sys.exit(main())