#!/usr/bin/env python3
"""
Test script to verify CSV loading functionality
"""

import pandas as pd
import sys
import os

def test_csv_loading():
    """Test loading the analytics.csv file"""
    csv_path = "analytics.csv"
    
    if not os.path.exists(csv_path):
        print(f"❌ CSV file not found: {csv_path}")
        return False
    
    try:
        # Test basic CSV loading
        df = pd.read_csv(csv_path)
        print(f"✅ CSV loaded successfully!")
        print(f"   - Shape: {df.shape}")
        print(f"   - Columns: {list(df.columns)}")
        
        # Check for required columns
        required_columns = ['userid', 'datetimeutc']
        missing_columns = [col for col in required_columns if col not in df.columns]
        
        if missing_columns:
            print(f"⚠️  Missing required columns: {missing_columns}")
        else:
            print(f"✅ All required columns present")
        
        # Test datetime parsing
        if 'datetimeutc' in df.columns:
            try:
                df['datetimeutc'] = pd.to_datetime(df['datetimeutc'], errors='coerce')
                valid_dates = df['datetimeutc'].notna().sum()
                print(f"✅ Datetime parsing successful: {valid_dates}/{len(df)} valid dates")
            except Exception as e:
                print(f"❌ Datetime parsing failed: {e}")
        
        # Show sample data
        print(f"\n📊 Sample data (first 3 rows):")
        print(df.head(3).to_string())
        
        return True
        
    except Exception as e:
        print(f"❌ Error loading CSV: {e}")
        return False

if __name__ == "__main__":
    print("🧪 Testing CSV Loading Functionality")
    print("=" * 50)
    
    success = test_csv_loading()
    
    if success:
        print("\n✅ All tests passed!")
        sys.exit(0)
    else:
        print("\n❌ Tests failed!")
        sys.exit(1)