#!/usr/bin/env python3
"""
Simple syntax check for our code changes
"""

import os
import sys
import ast

def check_syntax(filepath):
    """Check if a Python file has valid syntax"""
    try:
        with open(filepath, 'r') as f:
            content = f.read()
        
        # Parse the AST to check syntax
        ast.parse(content)
        print(f"✅ {filepath} - syntax OK")
        return True
        
    except SyntaxError as e:
        print(f"❌ {filepath} - syntax error: {e}")
        print(f"   Line {e.lineno}: {e.text}")
        return False
    except Exception as e:
        print(f"❌ {filepath} - error: {e}")
        return False

def main():
    """Check syntax of our modified files"""
    files_to_check = [
        'src/ml_model/catboost_model.py',
        'src/database/db_manager.py',
        'src/trading_engine/trading_engine.py',
    ]
    
    print("🔍 Checking syntax of modified files...")
    
    all_good = True
    for file_path in files_to_check:
        full_path = os.path.join(os.path.dirname(__file__), file_path)
        if os.path.exists(full_path):
            if not check_syntax(full_path):
                all_good = False
        else:
            print(f"⚠️  {file_path} - file not found")
            all_good = False
    
    if all_good:
        print("\n🎉 All files have valid syntax!")
        return 0
    else:
        print("\n💥 Some files have syntax errors!")
        return 1

if __name__ == "__main__":
    sys.exit(main())