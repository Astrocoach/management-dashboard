#!/usr/bin/env python3
"""
Datetime Investigation Script
This script investigates why datetime parsing is failing for most rows.
"""

import pandas as pd
import numpy as np

def investigate_datetime_issues():
    """Investigate datetime parsing failures in detail"""
    
    file_path = "C:/Users/Khatushyamji/Downloads/management-dashboard/analytics.csv"
    
    print("DATETIME PARSING INVESTIGATION")
    print("="*60)
    
    # Load raw data
    df = pd.read_csv(file_path)
    print(f"Total rows loaded: {len(df):,}")
    
    # Analyze the datetimeutc column
    print(f"\nAnalyzing 'datetimeutc' column...")
    
    # Check for empty/null values
    null_count = df['datetimeutc'].isnull().sum()
    empty_count = (df['datetimeutc'] == '').sum()
    print(f"Null values: {null_count:,}")
    print(f"Empty strings: {empty_count:,}")
    
    # Get unique value patterns
    print(f"\nUnique datetime value patterns (first 50):")
    unique_values = df['datetimeutc'].value_counts().head(50)
    for i, (value, count) in enumerate(unique_values.items(), 1):
        print(f"  {i:2d}. '{value}' (appears {count:,} times)")
    
    # Check for specific problematic patterns
    print(f"\nChecking for problematic patterns...")
    
    # Check for dash-only values
    dash_only = (df['datetimeutc'] == '-').sum()
    print(f"Dash-only values ('-'): {dash_only:,}")
    
    # Check for very short values
    short_values = df[df['datetimeutc'].astype(str).str.len() < 10]
    print(f"Values shorter than 10 characters: {len(short_values):,}")
    if len(short_values) > 0:
        print("Sample short values:")
        for val in short_values['datetimeutc'].unique()[:10]:
            count = (df['datetimeutc'] == val).sum()
            print(f"  '{val}' (appears {count:,} times)")
    
    # Check for valid ISO format values
    print(f"\nChecking for valid datetime formats...")
    
    # Try to identify valid ISO format
    iso_pattern = df['datetimeutc'].str.contains(r'^\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}', na=False)
    iso_count = iso_pattern.sum()
    print(f"ISO format pattern matches: {iso_count:,}")
    
    # Show samples of valid ISO format
    if iso_count > 0:
        print("Sample valid ISO format values:")
        valid_iso = df[iso_pattern]['datetimeutc'].head(10)
        for val in valid_iso:
            print(f"  {val}")
    
    # Test parsing on different subsets
    print(f"\nTesting datetime parsing on different subsets...")
    
    # Test 1: Parse all values
    try:
        parsed_all = pd.to_datetime(df['datetimeutc'], errors='coerce')
        success_all = (1 - parsed_all.isnull().sum() / len(parsed_all)) * 100
        print(f"All values: {success_all:.2f}% success rate")
    except Exception as e:
        print(f"All values: Failed - {e}")
    
    # Test 2: Parse only ISO pattern matches
    if iso_count > 0:
        try:
            iso_subset = df[iso_pattern]['datetimeutc']
            parsed_iso = pd.to_datetime(iso_subset, errors='coerce')
            success_iso = (1 - parsed_iso.isnull().sum() / len(parsed_iso)) * 100
            print(f"ISO pattern subset: {success_iso:.2f}% success rate")
        except Exception as e:
            print(f"ISO pattern subset: Failed - {e}")
    
    # Test 3: Parse non-dash values
    non_dash = df[df['datetimeutc'] != '-']['datetimeutc']
    if len(non_dash) > 0:
        try:
            parsed_non_dash = pd.to_datetime(non_dash, errors='coerce')
            success_non_dash = (1 - parsed_non_dash.isnull().sum() / len(parsed_non_dash)) * 100
            print(f"Non-dash values: {success_non_dash:.2f}% success rate")
        except Exception as e:
            print(f"Non-dash values: Failed - {e}")
    
    # Analyze by category
    print(f"\nAnalyzing datetime patterns by category...")
    category_analysis = df.groupby('category').agg({
        'datetimeutc': ['count', lambda x: (x == '-').sum(), lambda x: x.str.contains(r'^\d{4}-\d{2}-\d{2}T', na=False).sum()]
    }).round(2)
    category_analysis.columns = ['Total', 'Dash_Count', 'Valid_ISO_Count']
    category_analysis['Valid_ISO_Percent'] = (category_analysis['Valid_ISO_Count'] / category_analysis['Total'] * 100).round(2)
    print(category_analysis)
    
    # Check specific rows that should have valid dates
    print(f"\nChecking payment events (should have valid dates)...")
    payment_events = df[df['category'] == 'adapty_event']
    if len(payment_events) > 0:
        print(f"Payment events count: {len(payment_events):,}")
        print("Sample payment event datetime values:")
        for val in payment_events['datetimeutc'].head(10):
            print(f"  '{val}'")
        
        # Try parsing payment event dates
        try:
            parsed_payments = pd.to_datetime(payment_events['datetimeutc'], errors='coerce')
            success_payments = (1 - parsed_payments.isnull().sum() / len(parsed_payments)) * 100
            print(f"Payment events parsing success: {success_payments:.2f}%")
        except Exception as e:
            print(f"Payment events parsing failed: {e}")
    
    # Final recommendation
    print(f"\n" + "="*60)
    print("FINDINGS AND RECOMMENDATIONS")
    print("="*60)
    
    if dash_only > len(df) * 0.8:  # If more than 80% are dashes
        print("❌ CRITICAL ISSUE: Most datetime values are '-' (dash)")
        print("   This indicates missing or invalid timestamp data")
        print("   Recommendation: Check data source and ETL process")
    elif iso_count < len(df) * 0.1:  # If less than 10% are valid ISO
        print("⚠️  WARNING: Very few valid datetime values found")
        print("   Most data will be filtered out during processing")
        print("   Recommendation: Review datetime data quality")
    else:
        print("✅ Datetime data appears to have valid entries")
        print("   Some data loss is expected due to invalid entries")

if __name__ == "__main__":
    investigate_datetime_issues()