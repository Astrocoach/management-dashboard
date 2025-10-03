#!/usr/bin/env python3
"""
Comprehensive Test Suite for main.py
This script tests all major functions and edge cases in the analytics dashboard.
"""

import pandas as pd
import numpy as np
import sys
import os
from datetime import datetime, timedelta
import warnings
warnings.filterwarnings('ignore')

# Add the current directory to the path to import main functions
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

# Import functions from main.py
try:
    from  main import (
        load_and_process_csv, 
        reconstruct_sessions, 
        perform_user_segmentation, 
        detect_anomalies,
        parse_payment_data,
        create_improved_sankey_diagram,
        calculate_session_metrics,
        create_funnel_chart
    )
    print("✅ Successfully imported functions from main.py")
except ImportError as e:
    print(f"❌ Error importing functions: {e}")
    sys.exit(1)

class TestDataGenerator:
    """Generate test data for various scenarios"""
    
    @staticmethod
    def create_minimal_data():
        """Create minimal dataset with 1 user"""
        return pd.DataFrame({
            'analyticsid': ['1'],
            'userid': ['user1'],
            'datetimeutc': ['2025-09-23 09:00:39'],
            'category': ['app_event'],
            'name': ['screen_view'],
            'deviceid': ['device1'],
            'data': ['{}']
        })
    
    @staticmethod
    def create_small_data():
        """Create small dataset with 3 users"""
        data = []
        base_time = datetime(2025, 9, 23, 9, 0, 0)
        
        for i in range(3):
            for j in range(5):  # 5 events per user
                data.append({
                    'analyticsid': f'{i*5 + j + 1}',
                    'userid': f'user{i+1}',
                    'datetimeutc': (base_time + timedelta(hours=i, minutes=j*10)).strftime('%Y-%m-%d %H:%M:%S'),
                    'category': 'app_event',
                    'name': f'event_{j+1}',
                    'deviceid': f'device{i+1}',
                    'data': '{}'
                })
        
        return pd.DataFrame(data)
    
    @staticmethod
    def create_payment_data():
        """Create dataset with payment events"""
        data = []
        base_time = datetime(2025, 9, 23, 9, 0, 0)
        
        # Regular app events
        for i in range(2):
            for j in range(3):
                data.append({
                    'analyticsid': f'{i*3 + j + 1}',
                    'userid': f'user{i+1}',
                    'datetimeutc': (base_time + timedelta(hours=i, minutes=j*10)).strftime('%Y-%m-%d %H:%M:%S'),
                    'category': 'app_event',
                    'name': f'event_{j+1}',
                    'deviceid': f'device{i+1}',
                    'data': '{}'
                })
        
        # Payment events
        payment_data = {
            'vendorProductId': 'premium_monthly',
            'localizedTitle': 'Premium Monthly',
            'price': {'amount': 9.99, 'currencyCode': 'USD'},
            'regionCode': 'US'
        }
        
        data.append({
            'analyticsid': '7',
            'userid': 'user1',
            'datetimeutc': '2025-09-02T09:08:07.875Z',
            'category': 'adapty_event',
            'name': 'purchase',
            'deviceid': 'device1',
            'data': str(payment_data).replace("'", '"')
        })
        
        return pd.DataFrame(data)
    
    @staticmethod
    def create_empty_data():
        """Create empty dataset"""
        return pd.DataFrame(columns=['analyticsid', 'userid', 'datetimeutc', 'category', 'name', 'deviceid', 'data'])
    
    @staticmethod
    def create_invalid_datetime_data():
        """Create dataset with invalid datetime values"""
        return pd.DataFrame({
            'analyticsid': ['1', '2', '3'],
            'userid': ['user1', 'user2', 'user3'],
            'datetimeutc': ['invalid_date', '', None],
            'category': ['app_event', 'app_event', 'app_event'],
            'name': ['event1', 'event2', 'event3'],
            'deviceid': ['device1', 'device2', 'device3'],
            'data': ['{}', '{}', '{}']
        })

class TestRunner:
    """Run comprehensive tests on main.py functions"""
    
    def __init__(self):
        self.test_results = []
        self.generator = TestDataGenerator()
    
    def log_test(self, test_name, passed, message=""):
        """Log test result"""
        status = "✅ PASS" if passed else "❌ FAIL"
        self.test_results.append({
            'test': test_name,
            'passed': passed,
            'message': message
        })
        print(f"{status}: {test_name}")
        if message:
            print(f"    {message}")
    
    def test_load_and_process_csv(self):
        """Test CSV loading and processing function"""
        print("\n🔍 Testing load_and_process_csv function...")
        
        # Test 1: Normal data processing
        try:
            # Create a temporary CSV file
            test_data = self.generator.create_small_data()
            test_file = 'temp_test.csv'
            test_data.to_csv(test_file, index=False)
            
            result = load_and_process_csv(test_file)
            
            # Clean up
            os.remove(test_file)
            
            if result is not None and not result.empty:
                self.log_test("load_and_process_csv - Normal data", True, f"Processed {len(result)} rows")
            else:
                self.log_test("load_and_process_csv - Normal data", False, "Returned empty or None")
                
        except Exception as e:
            self.log_test("load_and_process_csv - Normal data", False, f"Exception: {str(e)}")
        
        # Test 2: Invalid datetime data
        try:
            invalid_data = self.generator.create_invalid_datetime_data()
            test_file = 'temp_invalid.csv'
            invalid_data.to_csv(test_file, index=False)
            
            result = load_and_process_csv(test_file)
            
            # Clean up
            os.remove(test_file)
            
            # Should handle invalid dates gracefully
            self.log_test("load_and_process_csv - Invalid dates", True, "Handled invalid dates gracefully")
                
        except Exception as e:
            self.log_test("load_and_process_csv - Invalid dates", False, f"Exception: {str(e)}")
    
    def test_reconstruct_sessions(self):
        """Test session reconstruction function"""
        print("\n🔍 Testing reconstruct_sessions function...")
        
        # Test 1: Normal session reconstruction
        try:
            test_data = self.generator.create_small_data()
            # Add required datetime column
            test_data['datetimeutc'] = pd.to_datetime(test_data['datetimeutc'])
            
            result = reconstruct_sessions(test_data)
            
            if result is not None and 'session_id' in result.columns:
                self.log_test("reconstruct_sessions - Normal data", True, f"Added session_id to {len(result)} rows")
            else:
                self.log_test("reconstruct_sessions - Normal data", False, "Failed to add session_id")
                
        except Exception as e:
            self.log_test("reconstruct_sessions - Normal data", False, f"Exception: {str(e)}")
        
        # Test 2: Empty data
        try:
            empty_data = self.generator.create_empty_data()
            result = reconstruct_sessions(empty_data)
            
            self.log_test("reconstruct_sessions - Empty data", True, "Handled empty data gracefully")
                
        except Exception as e:
            self.log_test("reconstruct_sessions - Empty data", False, f"Exception: {str(e)}")
    
    def test_perform_user_segmentation(self):
        """Test user segmentation function"""
        print("\n🔍 Testing perform_user_segmentation function...")
        
        # Test 1: Single user (edge case that caused the original error)
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
                self.log_test("perform_user_segmentation - Single user", True, f"Assigned segment: {result['segment_label'].iloc[0]}")
            else:
                self.log_test("perform_user_segmentation - Single user", False, "Failed to assign segment")
                
        except Exception as e:
            self.log_test("perform_user_segmentation - Single user", False, f"Exception: {str(e)}")
        
        # Test 2: Multiple users
        try:
            multi_user_metrics = pd.DataFrame({
                'userid': ['user1', 'user2', 'user3', 'user4', 'user5'],
                'total_sessions': [10, 5, 15, 2, 8],
                'avg_session_duration': [200, 100, 300, 50, 150],
                'total_events': [50, 25, 75, 10, 40],
                'days_active': [7, 3, 10, 1, 5]
            })
            
            result = perform_user_segmentation(multi_user_metrics)
            
            if result is not None and 'segment_label' in result.columns:
                unique_segments = result['segment_label'].nunique()
                self.log_test("perform_user_segmentation - Multiple users", True, f"Created {unique_segments} segments")
            else:
                self.log_test("perform_user_segmentation - Multiple users", False, "Failed to create segments")
                
        except Exception as e:
            self.log_test("perform_user_segmentation - Multiple users", False, f"Exception: {str(e)}")
        
        # Test 3: Empty data
        try:
            empty_metrics = pd.DataFrame(columns=['userid', 'total_sessions', 'avg_session_duration', 'total_events', 'days_active'])
            result = perform_user_segmentation(empty_metrics)
            
            self.log_test("perform_user_segmentation - Empty data", True, "Handled empty data gracefully")
                
        except Exception as e:
            self.log_test("perform_user_segmentation - Empty data", False, f"Exception: {str(e)}")
    
    def test_detect_anomalies(self):
        """Test anomaly detection function"""
        print("\n🔍 Testing detect_anomalies function...")
        
        # Test 1: Normal data
        try:
            test_data = self.generator.create_small_data()
            test_data['session_id'] = [1, 1, 2, 2, 2, 3, 3, 4, 4, 5, 5, 6, 6, 7, 7]  # Add session IDs
            
            result = detect_anomalies(test_data)
            
            if result is not None and 'anomaly_label' in result.columns:
                self.log_test("detect_anomalies - Normal data", True, f"Analyzed {len(result)} users")
            else:
                self.log_test("detect_anomalies - Normal data", False, "Failed to detect anomalies")
                
        except Exception as e:
            self.log_test("detect_anomalies - Normal data", False, f"Exception: {str(e)}")
        
        # Test 2: Single user
        try:
            single_user_data = self.generator.create_minimal_data()
            single_user_data['session_id'] = [1]
            
            result = detect_anomalies(single_user_data)
            
            self.log_test("detect_anomalies - Single user", True, "Handled single user gracefully")
                
        except Exception as e:
            self.log_test("detect_anomalies - Single user", False, f"Exception: {str(e)}")
        
        # Test 3: Empty data
        try:
            empty_data = self.generator.create_empty_data()
            result = detect_anomalies(empty_data)
            
            self.log_test("detect_anomalies - Empty data", True, "Handled empty data gracefully")
                
        except Exception as e:
            self.log_test("detect_anomalies - Empty data", False, f"Exception: {str(e)}")
    
    def test_parse_payment_data(self):
        """Test payment data parsing function"""
        print("\n🔍 Testing parse_payment_data function...")
        
        # Test 1: Data with payment events
        try:
            payment_data = self.generator.create_payment_data()
            result = parse_payment_data(payment_data)
            
            if result is not None and not result.empty:
                self.log_test("parse_payment_data - With payments", True, f"Parsed {len(result)} payment records")
            else:
                self.log_test("parse_payment_data - With payments", False, "Failed to parse payments")
                
        except Exception as e:
            self.log_test("parse_payment_data - With payments", False, f"Exception: {str(e)}")
        
        # Test 2: Data without payment events
        try:
            no_payment_data = self.generator.create_small_data()
            result = parse_payment_data(no_payment_data)
            
            self.log_test("parse_payment_data - No payments", True, "Handled no payment data gracefully")
                
        except Exception as e:
            self.log_test("parse_payment_data - No payments", False, f"Exception: {str(e)}")
    
    def test_calculate_session_metrics(self):
        """Test session metrics calculation function"""
        print("\n🔍 Testing calculate_session_metrics function...")
        
        # Test 1: Normal session metrics
        try:
            test_data = self.generator.create_small_data()
            test_data['datetimeutc'] = pd.to_datetime(test_data['datetimeutc'])
            test_data = reconstruct_sessions(test_data)  # Add session IDs first
            
            result = calculate_session_metrics(test_data)
            
            if result is not None and not result.empty:
                self.log_test("calculate_session_metrics - Normal data", True, f"Calculated metrics for {len(result)} users")
            else:
                self.log_test("calculate_session_metrics - Normal data", False, "Failed to calculate metrics")
                
        except Exception as e:
            self.log_test("calculate_session_metrics - Normal data", False, f"Exception: {str(e)}")
        
        # Test 2: Empty data
        try:
            empty_data = self.generator.create_empty_data()
            result = calculate_session_metrics(empty_data)
            
            self.log_test("calculate_session_metrics - Empty data", True, "Handled empty data gracefully")
                
        except Exception as e:
            self.log_test("calculate_session_metrics - Empty data", False, f"Exception: {str(e)}")
    
    def run_all_tests(self):
        """Run all tests"""
        print("🚀 Starting Comprehensive Test Suite for main.py")
        print("=" * 60)
        
        self.test_load_and_process_csv()
        self.test_reconstruct_sessions()
        self.test_perform_user_segmentation()
        self.test_detect_anomalies()
        self.test_parse_payment_data()
        self.test_calculate_session_metrics()
        
        # Summary
        print("\n" + "=" * 60)
        print("📊 TEST SUMMARY")
        print("=" * 60)
        
        total_tests = len(self.test_results)
        passed_tests = sum(1 for result in self.test_results if result['passed'])
        failed_tests = total_tests - passed_tests
        
        print(f"Total Tests: {total_tests}")
        print(f"Passed: {passed_tests}")
        print(f"Failed: {failed_tests}")
        print(f"Success Rate: {(passed_tests/total_tests)*100:.1f}%")
        
        if failed_tests > 0:
            print(f"\n❌ Failed Tests:")
            for result in self.test_results:
                if not result['passed']:
                    print(f"  - {result['test']}: {result['message']}")
        
        print(f"\n{'🎉 ALL TESTS PASSED!' if failed_tests == 0 else '⚠️  SOME TESTS FAILED'}")
        
        return failed_tests == 0

if __name__ == "__main__":
    runner = TestRunner()
    success = runner.run_all_tests()
    
    if success:
        print("\n✅ main.py is ready for production!")
    else:
        print("\n❌ main.py needs attention before deployment.")
    
    sys.exit(0 if success else 1)