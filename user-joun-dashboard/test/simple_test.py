#!/usr/bin/env python3
"""
Simple Test for main.py Core Functions
"""

import pandas as pd
import numpy as np
import sys
import os
from datetime import datetime, timedelta
import warnings
warnings.filterwarnings('ignore')

# Suppress Streamlit warnings
os.environ['STREAMLIT_LOGGER_LEVEL'] = 'error'

def test_user_segmentation_edge_cases():
    """Test the specific edge case that was causing the error"""
    print("🔍 Testing User Segmentation Edge Cases...")
    
    # Import the function
    try:
        from  import perform_user_segmentation
        print("✅ Successfully imported perform_user_segmentation")
    except ImportError as e:
        print(f"❌ Import failed: {e}")
        return False
    
    # Test 1: Single user (the original error case)
    print("\n📝 Test 1: Single user segmentation")
    try:
        single_user_metrics = pd.DataFrame({
            'userid': ['user1'],
            'total_sessions': [5],
            'avg_session_duration': [120.5],
            'total_events': [25],
            'days_active': [3]
        })
        
        result = perform_user_segmentation(single_user_metrics)
        
        if result is not None and 'segment_label' in result.columns:
            print(f"✅ PASS: Single user assigned to segment '{result['segment_label'].iloc[0]}'")
        else:
            print("❌ FAIL: No segment assigned")
            return False
            
    except Exception as e:
        print(f"❌ FAIL: Exception occurred: {str(e)}")
        return False
    
    # Test 2: Two users (still less than 4 clusters)
    print("\n📝 Test 2: Two users segmentation")
    try:
        two_user_metrics = pd.DataFrame({
            'userid': ['user1', 'user2'],
            'total_sessions': [5, 10],
            'avg_session_duration': [120.5, 200.0],
            'total_events': [25, 50],
            'days_active': [3, 7]
        })
        
        result = perform_user_segmentation(two_user_metrics)
        
        if result is not None and 'segment_label' in result.columns:
            segments = result['segment_label'].unique()
            print(f"✅ PASS: Two users assigned to {len(segments)} segment(s): {list(segments)}")
        else:
            print("❌ FAIL: No segments assigned")
            return False
            
    except Exception as e:
        print(f"❌ FAIL: Exception occurred: {str(e)}")
        return False
    
    # Test 3: Five users (more than 4 clusters)
    print("\n📝 Test 3: Five users segmentation")
    try:
        five_user_metrics = pd.DataFrame({
            'userid': ['user1', 'user2', 'user3', 'user4', 'user5'],
            'total_sessions': [5, 10, 15, 2, 8],
            'avg_session_duration': [120.5, 200.0, 300.0, 50.0, 150.0],
            'total_events': [25, 50, 75, 10, 40],
            'days_active': [3, 7, 10, 1, 5]
        })
        
        result = perform_user_segmentation(five_user_metrics)
        
        if result is not None and 'segment_label' in result.columns:
            segments = result['segment_label'].unique()
            print(f"✅ PASS: Five users assigned to {len(segments)} segment(s): {list(segments)}")
        else:
            print("❌ FAIL: No segments assigned")
            return False
            
    except Exception as e:
        print(f"❌ FAIL: Exception occurred: {str(e)}")
        return False
    
    # Test 4: Empty data
    print("\n📝 Test 4: Empty data segmentation")
    try:
        empty_metrics = pd.DataFrame(columns=['userid', 'total_sessions', 'avg_session_duration', 'total_events', 'days_active'])
        
        result = perform_user_segmentation(empty_metrics)
        
        if result is not None:
            print("✅ PASS: Empty data handled gracefully")
        else:
            print("❌ FAIL: Empty data not handled")
            return False
            
    except Exception as e:
        print(f"❌ FAIL: Exception occurred: {str(e)}")
        return False
    
    return True

def test_anomaly_detection_edge_cases():
    """Test anomaly detection with edge cases"""
    print("\n🔍 Testing Anomaly Detection Edge Cases...")
    
    try:
        from  import detect_anomalies
        print("✅ Successfully imported detect_anomalies")
    except ImportError as e:
        print(f"❌ Import failed: {e}")
        return False
    
    # Test 1: Single user
    print("\n📝 Test 1: Single user anomaly detection")
    try:
        single_user_data = pd.DataFrame({
            'userid': ['user1'],
            'session_id': [1],
            'datetimeutc': [datetime.now()]
        })
        
        result = detect_anomalies(single_user_data)
        print("✅ PASS: Single user anomaly detection handled")
        
    except Exception as e:
        print(f"❌ FAIL: Exception occurred: {str(e)}")
        return False
    
    # Test 2: Empty data
    print("\n📝 Test 2: Empty data anomaly detection")
    try:
        empty_data = pd.DataFrame(columns=['userid', 'session_id', 'datetimeutc'])
        
        result = detect_anomalies(empty_data)
        print("✅ PASS: Empty data anomaly detection handled")
        
    except Exception as e:
        print(f"❌ FAIL: Exception occurred: {str(e)}")
        return False
    
    return True

def main():
    print("🚀 Running Simple Tests for main.py Edge Cases")
    print("=" * 60)
    
    success1 = test_user_segmentation_edge_cases()
    success2 = test_anomaly_detection_edge_cases()
    
    print("\n" + "=" * 60)
    print("📊 FINAL RESULTS")
    print("=" * 60)
    
    if success1 and success2:
        print("🎉 ALL TESTS PASSED! main.py is ready for production!")
        return True
    else:
        print("⚠️ SOME TESTS FAILED! main.py needs attention.")
        return False

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)