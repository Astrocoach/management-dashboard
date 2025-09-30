#!/usr/bin/env python3
"""
CSV Verification Test Script
This script verifies if the analytics.csv file is being read completely
and checks for any parsing errors or data truncation.
"""

import pandas as pd
import numpy as np
import sys
from datetime import datetime

def test_csv_reading():
    """Test CSV reading completeness and accuracy"""
    
    csv_file = "C:/Users/Khatushyamji/Downloads/management-dashboard/analytics.csv"
    
    print("=" * 60)
    print("CSV VERIFICATION TEST")
    print("=" * 60)
    
    # Test 1: Basic file reading
    print("\n1. BASIC FILE READING TEST")
    print("-" * 30)
    
    try:
        # Read with default settings
        df = pd.read_csv(csv_file)
        print(f"✅ Successfully read CSV file")
        print(f"   Rows loaded: {len(df):,}")
        print(f"   Columns: {len(df.columns)}")
        print(f"   Memory usage: {df.memory_usage(deep=True).sum() / 1024 / 1024:.2f} MB")
        
    except Exception as e:
        print(f"❌ Error reading CSV: {e}")
        return False
    
    # Test 2: Check for truncation by comparing with file line count
    print("\n2. TRUNCATION CHECK")
    print("-" * 30)
    
    try:
        # Count lines in file manually
        with open(csv_file, 'r', encoding='utf-8') as f:
            file_line_count = sum(1 for line in f)
        
        expected_rows = file_line_count - 1  # Subtract header
        actual_rows = len(df)
        
        print(f"   File lines (including header): {file_line_count:,}")
        print(f"   Expected data rows: {expected_rows:,}")
        print(f"   Actual rows loaded: {actual_rows:,}")
        
        if actual_rows == expected_rows:
            print("✅ No truncation detected - all rows loaded")
        else:
            print(f"❌ TRUNCATION DETECTED: Missing {expected_rows - actual_rows:,} rows")
            
    except Exception as e:
        print(f"❌ Error checking truncation: {e}")
    
    # Test 3: Data type analysis
    print("\n3. DATA TYPE ANALYSIS")
    print("-" * 30)
    
    print("   Column data types:")
    for col, dtype in df.dtypes.items():
        print(f"     {col}: {dtype}")
    
    # Test 4: Missing values check
    print("\n4. MISSING VALUES CHECK")
    print("-" * 30)
    
    missing_counts = df.isnull().sum()
    total_missing = missing_counts.sum()
    
    if total_missing == 0:
        print("✅ No missing values detected")
    else:
        print(f"⚠️  Found {total_missing:,} missing values:")
        for col, count in missing_counts[missing_counts > 0].items():
            percentage = (count / len(df)) * 100
            print(f"     {col}: {count:,} ({percentage:.2f}%)")
    
    # Test 5: Datetime parsing test
    print("\n5. DATETIME PARSING TEST")
    print("-" * 30)
    
    if 'datetimeutc' in df.columns:
        try:
            # Test datetime parsing
            df_test = df.copy()
            df_test['datetimeutc_parsed'] = pd.to_datetime(df_test['datetimeutc'], errors='coerce', utc=True)
            
            # Count parsing failures
            parsing_failures = df_test['datetimeutc_parsed'].isnull().sum()
            
            if parsing_failures == 0:
                print("✅ All datetime values parsed successfully")
            else:
                print(f"⚠️  {parsing_failures:,} datetime parsing failures")
                
            # Show date range
            valid_dates = df_test['datetimeutc_parsed'].dropna()
            if not valid_dates.empty:
                print(f"   Date range: {valid_dates.min()} to {valid_dates.max()}")
                
        except Exception as e:
            print(f"❌ Error testing datetime parsing: {e}")
    else:
        print("❌ 'datetimeutc' column not found")
    
    # Test 6: Data integrity checks
    print("\n6. DATA INTEGRITY CHECKS")
    print("-" * 30)
    
    # Check for duplicate rows
    duplicates = df.duplicated().sum()
    print(f"   Duplicate rows: {duplicates:,}")
    
    # Check unique values in key columns
    if 'userid' in df.columns:
        unique_users = df['userid'].nunique()
        print(f"   Unique users: {unique_users:,}")
    
    if 'analyticsid' in df.columns:
        unique_analytics_ids = df['analyticsid'].nunique()
        total_analytics_ids = len(df)
        print(f"   Unique analytics IDs: {unique_analytics_ids:,} / {total_analytics_ids:,}")
        
        if unique_analytics_ids != total_analytics_ids:
            print("⚠️  Analytics IDs are not unique!")
    
    # Test 7: Sample data verification
    print("\n7. SAMPLE DATA VERIFICATION")
    print("-" * 30)
    
    print("   First 3 rows:")
    print(df.head(3).to_string())
    
    print("\n   Last 3 rows:")
    print(df.tail(3).to_string())
    
    # Test 8: Memory and performance check
    print("\n8. MEMORY AND PERFORMANCE")
    print("-" * 30)
    
    memory_usage = df.memory_usage(deep=True)
    total_memory = memory_usage.sum() / 1024 / 1024
    
    print(f"   Total memory usage: {total_memory:.2f} MB")
    print(f"   Average memory per row: {total_memory * 1024 / len(df):.2f} KB")
    
    # Check for potential memory issues
    if total_memory > 100:
        print("⚠️  Large memory usage detected - consider chunked reading for very large files")
    
    # Test 9: JSON data parsing (if applicable)
    print("\n9. JSON DATA PARSING TEST")
    print("-" * 30)
    
    if 'analyticsdata' in df.columns:
        try:
            # Test JSON parsing on a sample
            sample_json = df['analyticsdata'].dropna().head(5)
            json_parse_errors = 0
            
            for idx, json_str in sample_json.items():
                try:
                    import json
                    json.loads(json_str)
                except:
                    json_parse_errors += 1
            
            if json_parse_errors == 0:
                print("✅ Sample JSON data parses correctly")
            else:
                print(f"⚠️  {json_parse_errors} JSON parsing errors in sample")
                
        except Exception as e:
            print(f"❌ Error testing JSON parsing: {e}")
    else:
        print("   No 'analyticsdata' column found")
    
    print("\n" + "=" * 60)
    print("CSV VERIFICATION COMPLETE")
    print("=" * 60)
    
    return True

if __name__ == "__main__":
    test_csv_reading()