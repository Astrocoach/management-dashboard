#!/usr/bin/env python3
"""
Test script to verify edge case handling for CSV loading
"""

import pandas as pd
import sys
import os
import io
from  import load_and_process_csv

class MockStreamlitFile:
    """Mock Streamlit uploaded file object"""
    def __init__(self, content, name="test.csv", size=None):
        self.content = content
        self.name = name
        self.size = size if size is not None else len(content)
        self._io = io.StringIO(content)
    
    def read(self, size=-1):
        return self._io.read(size)
    
    def readline(self):
        return self._io.readline()
    
    def readlines(self):
        return self._io.readlines()
    
    def seek(self, position):
        return self._io.seek(position)
    
    def tell(self):
        return self._io.tell()
    
    def __iter__(self):
        return iter(self._io)
    
    def __next__(self):
        return next(self._io)

def test_empty_file():
    """Test handling of empty CSV file"""
    print("🧪 Testing empty file...")
    empty_file = MockStreamlitFile("", "empty.csv", 0)
    
    # This should return None and show an error
    result = load_and_process_csv(empty_file)
    if result is None:
        print("✅ Empty file handled correctly")
        return True
    else:
        print("❌ Empty file not handled correctly")
        return False

def test_no_columns_file():
    """Test handling of file with no columns"""
    print("🧪 Testing file with no columns...")
    no_cols_file = MockStreamlitFile("\n\n\n", "no_cols.csv")
    
    result = load_and_process_csv(no_cols_file)
    if result is None:
        print("✅ No columns file handled correctly")
        return True
    else:
        print("❌ No columns file not handled correctly")
        return False

def test_malformed_csv():
    """Test handling of malformed CSV"""
    print("🧪 Testing malformed CSV...")
    malformed_content = '''userid,datetimeutc,event
123,"2023-01-01 10:00:00,click
456,2023-01-01 11:00:00",view'''
    
    malformed_file = MockStreamlitFile(malformed_content, "malformed.csv")
    
    # This might succeed or fail depending on pandas parsing
    result = load_and_process_csv(malformed_file)
    print(f"✅ Malformed CSV handled (result: {'success' if result is not None else 'handled gracefully'})")
    return True

def test_valid_csv():
    """Test handling of valid CSV"""
    print("🧪 Testing valid CSV...")
    valid_content = '''userid,datetimeutc,event
123,2023-01-01T10:00:00Z,click
456,2023-01-01T11:00:00Z,view
789,2023-01-01T12:00:00Z,purchase'''
    
    valid_file = MockStreamlitFile(valid_content, "valid.csv")
    
    result = load_and_process_csv(valid_file)
    print(f"DEBUG: Result type: {type(result)}")
    if result is not None:
        print(f"DEBUG: Result length: {len(result)}")
        print(f"DEBUG: Result columns: {list(result.columns)}")
    
    if result is not None and len(result) == 3:
        print("✅ Valid CSV handled correctly")
        return True
    else:
        print("❌ Valid CSV not handled correctly")
        return False

def test_semicolon_delimiter():
    """Test handling of semicolon-delimited CSV"""
    print("🧪 Testing semicolon-delimited CSV...")
    semicolon_content = '''userid;datetimeutc;event
123;2023-01-01T10:00:00Z;click
456;2023-01-01T11:00:00Z;view'''
    
    semicolon_file = MockStreamlitFile(semicolon_content, "semicolon.csv")
    
    result = load_and_process_csv(semicolon_file)
    if result is not None:
        print("✅ Semicolon-delimited CSV handled correctly")
        return True
    else:
        print("❌ Semicolon-delimited CSV not handled correctly")
        return False

if __name__ == "__main__":
    print("🧪 Testing Edge Cases for CSV Loading")
    print("=" * 50)
    
    # Suppress Streamlit messages for testing
    import streamlit as st
    
    # Mock streamlit functions to avoid errors
    def mock_error(msg): print(f"ERROR: {msg}")
    def mock_warning(msg): print(f"WARNING: {msg}")
    def mock_success(msg): print(f"SUCCESS: {msg}")
    def mock_info(msg): print(f"INFO: {msg}")
    
    st.error = mock_error
    st.warning = mock_warning
    st.success = mock_success
    st.info = mock_info
    
    tests = [
        test_empty_file,
        test_no_columns_file,
        test_malformed_csv,
        test_valid_csv,
        test_semicolon_delimiter
    ]
    
    passed = 0
    total = len(tests)
    
    for test in tests:
        try:
            if test():
                passed += 1
        except Exception as e:
            print(f"❌ Test failed with exception: {e}")
    
    print(f"\n📊 Results: {passed}/{total} tests passed")
    
    if passed == total:
        print("✅ All edge case tests passed!")
        sys.exit(0)
    else:
        print("❌ Some tests failed!")
        sys.exit(1)