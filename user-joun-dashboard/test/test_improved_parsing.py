#!/usr/bin/env python3
"""
Test Improved Datetime Parsing
This script tests the improved datetime parsing logic.
"""

import pandas as pd
import numpy as np

def test_improved_parsing():
    """Test the improved datetime parsing logic"""
    
    file_path = "C:/Users/Khatushyamji/Downloads/management-dashboard/analytics.csv"
    
    print("TESTING IMPROVED DATETIME PARSING")
    print("="*50)
    
    # Load raw data
    df = pd.read_csv(file_path)
    print(f"Total rows loaded: {len(df):,}")
    
    # Apply improved parsing logic
    print("\nApplying improved parsing logic...")
    
    # First try parsing with UTC (for ISO format with timezone)
    df['datetimeutc'] = pd.to_datetime(df['datetimeutc'], errors='coerce', utc=True)
    
    # For timezone-aware values, convert to timezone-naive
    if df['datetimeutc'].dt.tz is not None:
        df['datetimeutc'] = df['datetimeutc'].dt.tz_localize(None)
    
    # Count successful parses after first attempt
    first_attempt_success = (~df['datetimeutc'].isnull()).sum()
    print(f"First attempt (UTC parsing): {first_attempt_success:,} successful")
    
    # For values that failed UTC parsing, try without timezone assumption
    failed_mask = df['datetimeutc'].isnull()
    failed_count = failed_mask.sum()
    print(f"Failed entries to retry: {failed_count:,}")
    
    if failed_mask.any():
        # Re-read the original values for failed entries
        original_df = pd.read_csv(file_path)
        failed_values = original_df.loc[failed_mask, 'datetimeutc']
        
        # Parse without timezone assumption
        parsed_without_tz = pd.to_datetime(failed_values, errors='coerce')
        
        # Count successful parses in second attempt
        second_attempt_success = (~parsed_without_tz.isnull()).sum()
        print(f"Second attempt (no timezone): {second_attempt_success:,} successful")
        
        # Update the failed entries
        df.loc[failed_mask, 'datetimeutc'] = parsed_without_tz
    
    # Final statistics
    final_success = (~df['datetimeutc'].isnull()).sum()
    final_failed = df['datetimeutc'].isnull().sum()
    success_rate = (final_success / len(df)) * 100
    
    print(f"\nFINAL RESULTS:")
    print(f"Successfully parsed: {final_success:,} ({success_rate:.2f}%)")
    print(f"Failed to parse: {final_failed:,} ({100-success_rate:.2f}%)")
    
    # Test date extraction
    print(f"\nTesting date extraction...")
    df_clean = df.dropna(subset=['datetimeutc'])
    df_clean['date'] = df_clean['datetimeutc'].dt.date
    
    date_success = (~df_clean['date'].isnull()).sum()
    print(f"Date extraction successful: {date_success:,}")
    
    # Show date range
    if date_success > 0:
        min_date = df_clean['date'].min()
        max_date = df_clean['date'].max()
        print(f"Date range: {min_date} to {max_date}")
    
    # Analyze by category
    print(f"\nAnalysis by category:")
    category_stats = df.groupby('category').agg({
        'datetimeutc': ['count', lambda x: (~x.isnull()).sum()]
    })
    category_stats.columns = ['Total', 'Parsed_Successfully']
    category_stats['Success_Rate'] = (category_stats['Parsed_Successfully'] / category_stats['Total'] * 100).round(2)
    print(category_stats)
    
    return final_success, len(df)

if __name__ == "__main__":
    success_count, total_count = test_improved_parsing()
    
    print(f"\n" + "="*50)
    if success_count > total_count * 0.95:  # More than 95% success
        print("✅ EXCELLENT: Datetime parsing is working very well!")
    elif success_count > total_count * 0.8:  # More than 80% success
        print("✅ GOOD: Datetime parsing is working well!")
    elif success_count > total_count * 0.5:  # More than 50% success
        print("⚠️  FAIR: Datetime parsing has improved but could be better")
    else:
        print("❌ POOR: Datetime parsing still needs work")