#!/usr/bin/env python3
"""
Application Loading Test
This script simulates exactly how main.py loads the CSV file
to verify data completeness and accuracy.
"""

import pandas as pd
import numpy as np
from datetime import datetime

def load_and_process_csv_test(file_path):
    """Simulate the exact loading process from main.py"""
    try:
        print("Loading CSV with pandas.read_csv()...")
        df = pd.read_csv(file_path)
        print(f"Initial load: {len(df):,} rows, {len(df.columns)} columns")
        
        # Parse datetime - ENSURE TIMEZONE NAIVE (same as main.py)
        print("\nProcessing datetime columns...")
        if 'datetimeutc' in df.columns:
            print("Found 'datetimeutc' column")
            original_count = len(df)
            
            # Show sample of original datetime values
            print("Sample original datetime values:")
            print(df['datetimeutc'].head(10).tolist())
            
            df['datetimeutc'] = pd.to_datetime(df['datetimeutc'], errors='coerce', utc=True).dt.tz_localize(None)
            
            # Check for NaT values after parsing
            nat_count = df['datetimeutc'].isna().sum()
            print(f"NaT values after datetime parsing: {nat_count:,}")
            
            if nat_count > 0:
                print("Sample rows with NaT values:")
                nat_rows = df[df['datetimeutc'].isna()].head(5)
                print(nat_rows[['analyticsid', 'userid', 'datetimeutc', 'name']].to_string())
        
        elif 'created_at' in df.columns:
            print("Found 'created_at' column, using as datetimeutc")
            df['created_at'] = pd.to_datetime(df['created_at'], errors='coerce', utc=True).dt.tz_localize(None)
            df['datetimeutc'] = df['created_at']
        
        # Extract date components - Handle NaN values properly (same as main.py)
        if 'datetimeutc' in df.columns:
            print("\nExtracting date components...")
            before_dropna = len(df)
            
            # Remove rows with invalid datetime values to prevent mixed types
            df = df.dropna(subset=['datetimeutc'])
            after_dropna = len(df)
            
            dropped_rows = before_dropna - after_dropna
            print(f"Rows dropped due to invalid datetime: {dropped_rows:,}")
            print(f"Remaining rows: {after_dropna:,}")
            
            # Extract date components only from valid datetime values
            df['date'] = df['datetimeutc'].dt.date
            df['hour'] = df['datetimeutc'].dt.hour
            df['day_of_week'] = df['datetimeutc'].dt.day_name()
            df['month'] = df['datetimeutc'].dt.month
            
            print(f"Successfully extracted date components for {len(df):,} rows")
            
            # Show date range
            print(f"Date range: {df['date'].min()} to {df['date'].max()}")
        
        return df
    except Exception as e:
        print(f"Error loading CSV: {str(e)}")
        return None

def compare_with_raw_file(file_path):
    """Compare processed data with raw file content"""
    print("\n" + "="*60)
    print("COMPARING WITH RAW FILE")
    print("="*60)
    
    # Count raw file lines
    with open(file_path, 'r', encoding='utf-8') as f:
        raw_lines = sum(1 for line in f)
    
    print(f"Raw file lines (including header): {raw_lines:,}")
    print(f"Expected data rows: {raw_lines - 1:,}")
    
    # Load with our processing
    processed_df = load_and_process_csv_test(file_path)
    
    if processed_df is not None:
        print(f"Processed data rows: {len(processed_df):,}")
        
        data_loss = (raw_lines - 1) - len(processed_df)
        if data_loss > 0:
            print(f"⚠️  Data loss: {data_loss:,} rows ({data_loss/(raw_lines-1)*100:.2f}%)")
        else:
            print("✅ No data loss detected")
        
        # Check data types in final dataframe
        print("\nFinal data types:")
        for col, dtype in processed_df.dtypes.items():
            print(f"  {col}: {dtype}")
        
        # Check for any remaining mixed types in date column
        if 'date' in processed_df.columns:
            date_types = processed_df['date'].apply(type).value_counts()
            print(f"\nDate column type distribution:")
            for dtype, count in date_types.items():
                print(f"  {dtype}: {count:,}")
        
        return processed_df
    else:
        print("❌ Failed to process data")
        return None

def detailed_datetime_analysis(file_path):
    """Analyze datetime parsing issues in detail"""
    print("\n" + "="*60)
    print("DETAILED DATETIME ANALYSIS")
    print("="*60)
    
    # Load raw data
    df_raw = pd.read_csv(file_path)
    
    # Analyze datetime column patterns
    if 'datetimeutc' in df_raw.columns:
        print("Analyzing datetime patterns...")
        
        # Sample of datetime values
        datetime_sample = df_raw['datetimeutc'].head(20)
        print("\nFirst 20 datetime values:")
        for i, dt in enumerate(datetime_sample):
            print(f"  {i+1:2d}: {dt}")
        
        # Check for different datetime formats
        print("\nChecking datetime format patterns...")
        
        # Try parsing with different methods
        methods = [
            ("Standard ISO", lambda x: pd.to_datetime(x, errors='coerce')),
            ("UTC aware", lambda x: pd.to_datetime(x, errors='coerce', utc=True)),
            ("Infer format", lambda x: pd.to_datetime(x, errors='coerce', infer_datetime_format=True)),
        ]
        
        for method_name, method_func in methods:
            try:
                parsed = method_func(df_raw['datetimeutc'])
                success_rate = (1 - parsed.isna().sum() / len(parsed)) * 100
                print(f"  {method_name}: {success_rate:.2f}% success rate")
            except Exception as e:
                print(f"  {method_name}: Failed - {e}")

if __name__ == "__main__":
    file_path = "C:/Users/Khatushyamji/Downloads/management-dashboard/analytics.csv"
    
    print("APPLICATION LOADING SIMULATION TEST")
    print("="*60)
    
    # Test 1: Simulate exact application loading
    processed_data = compare_with_raw_file(file_path)
    
    # Test 2: Detailed datetime analysis
    detailed_datetime_analysis(file_path)
    
    print("\n" + "="*60)
    print("SIMULATION TEST COMPLETE")
    print("="*60)