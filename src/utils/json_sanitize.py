"""
JSON Serialization Utility for Trading System
Handles numpy types and other non-serializable objects safely
"""

import json
import numpy as np
import pandas as pd
from datetime import datetime, date
from decimal import Decimal
from typing import Any, Dict, List, Union


def sanitize_for_json(obj: Any) -> Any:
    """
    Recursively sanitize an object for JSON serialization.
    Converts numpy types, pandas objects, and other non-serializable types.
    """
    if obj is None:
        return None
    
    # Handle numpy types
    if isinstance(obj, np.integer):
        return int(obj)
    elif isinstance(obj, np.floating):
        return float(obj)
    elif isinstance(obj, np.ndarray):
        return obj.tolist()
    elif isinstance(obj, np.bool_):
        return bool(obj)
    
    # Handle pandas types
    elif isinstance(obj, pd.Series):
        return obj.to_list()
    elif isinstance(obj, pd.DataFrame):
        return obj.to_dict('records')
    elif isinstance(obj, pd.Timestamp):
        return obj.isoformat()
    
    # Handle datetime types
    elif isinstance(obj, (datetime, date)):
        return obj.isoformat()
    
    # Handle Decimal
    elif isinstance(obj, Decimal):
        return float(obj)
    
    # Handle dictionaries (recursively sanitize keys and values)
    elif isinstance(obj, dict):
        sanitized = {}
        for key, value in obj.items():
            # Convert non-string keys to strings
            clean_key = str(key) if not isinstance(key, str) else key
            sanitized[clean_key] = sanitize_for_json(value)
        return sanitized
    
    # Handle lists and tuples
    elif isinstance(obj, (list, tuple)):
        return [sanitize_for_json(item) for item in obj]
    
    # Handle sets
    elif isinstance(obj, set):
        return [sanitize_for_json(item) for item in obj]
    
    # Handle basic types (already JSON serializable)
    elif isinstance(obj, (str, int, float, bool)):
        return obj
    
    # For everything else, try to convert to string
    else:
        try:
            return str(obj)
        except Exception:
            return f"<non-serializable: {type(obj).__name__}>"


def safe_json_dumps(obj: Any, **kwargs) -> str:
    """
    Safely serialize an object to JSON string.
    Automatically sanitizes the object before serialization.
    """
    try:
        sanitized = sanitize_for_json(obj)
        return json.dumps(sanitized, **kwargs)
    except Exception as e:
        # Fallback: create error response
        error_obj = {
            "error": "JSON serialization failed",
            "error_type": str(type(e).__name__),
            "error_message": str(e),
            "original_type": str(type(obj).__name__)
        }
        return json.dumps(error_obj)


def safe_json_loads(json_str: str, **kwargs) -> Any:
    """
    Safely deserialize JSON string to Python object.
    """
    try:
        return json.loads(json_str, **kwargs)
    except Exception as e:
        return {
            "error": "JSON deserialization failed",
            "error_type": str(type(e).__name__),
            "error_message": str(e)
        }


class JSONEncoder(json.JSONEncoder):
    """
    Custom JSON encoder that handles numpy and pandas types automatically.
    """
    
    def default(self, obj):
        sanitized = sanitize_for_json(obj)
        if sanitized != obj:
            return sanitized
        return super().default(obj)


def make_json_safe(data: Any) -> Any:
    """
    Alias for sanitize_for_json for backward compatibility.
    """
    return sanitize_for_json(data)


# Example usage and testing
if __name__ == "__main__":
    # Test cases
    test_data = {
        "numpy_int": np.int64(123),
        "numpy_float": np.float32(12.34),
        "numpy_array": np.array([1, 2, 3]),
        "numpy_bool": np.bool_(True),
        "datetime": datetime.now(),
        "decimal": Decimal("123.45"),
        "nested_dict": {
            np.int64(42): "numpy key",
            "list": [np.float64(1.1), np.float64(2.2)],
            "set": {1, 2, 3}
        }
    }
    
    print("Original:", test_data)
    sanitized = sanitize_for_json(test_data)
    print("Sanitized:", sanitized)
    print("JSON:", safe_json_dumps(test_data, indent=2))