#!/usr/bin/env python3
"""
Comprehensive test to identify categorical ordering issues in API data loading
"""

import sys
import os
import pandas as pd
import json
import traceback
from datetime import datetime, timedelta

# Add parent directory to path to import main functions
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

try:
    from main import (
        load_payment_data, 
        load_user_tracking_data,
        parse_payment_data, 
        process_payment_api_data,
        process_user_tracking_api_data,
        fetch_api_data
    )
except ImportError as e:
    print(f"❌ Import error: {e}")
    sys.exit(1)

class CategoricalOrderingDebugger:
    """Debug categorical ordering issues in data loading"""
    
    def __init__(self):
        self.test_results = {}
        self.error_locations = []
    
    def create_test_data(self):
        """Create test data that might trigger categorical ordering issues"""
        test_data = []
        
        # Create data with mixed regions (including None values)
        regions = ['US', 'EU', 'UK', None, 'CA', 'Unknown']
        currencies = ['USD', 'EUR', 'GBP', None, 'CAD']
        platforms = ['iOS', 'Android', None]
        
        for i in range(50):
            # Create analytics data with potential categorical issues
            analytics_data = {
                'adaptyObject': {
                    'vendorProductId': f'product_{i % 5}',
                    'localizedTitle': f'Product {i % 5}',
                    'price': {
                        'amount': round(9.99 + (i % 10), 2),
                        'currencyCode': currencies[i % len(currencies)]
                    },
                    'regionCode': regions[i % len(regions)],
                    'paywallName': f'paywall_{i % 3}'
                }
            }
            
            # Create API response format data
            api_data = {
                'analyticsid': str(i + 1),
                'userid': f'user_{i % 20}',
                'deviceid': f'device_{platforms[i % len(platforms)]}_{i}',
                'appversion': '1.0.0',
                'category': 'adapty_event',
                'name': 'payment_success',
                'datetimeutc': (datetime.now() - timedelta(days=i % 30)).isoformat(),
                'appname': 'TestApp',
                'analyticsdata': json.dumps(analytics_data),
                'analytic_attr_data': [
                    {'analytic_name': 'amount', 'analytic_value': str(analytics_data['adaptyObject']['price']['amount'])},
                    {'analytic_name': 'currencyCode', 'analytic_value': analytics_data['adaptyObject']['price']['currencyCode']},
                    {'analytic_name': 'regionCode', 'analytic_value': analytics_data['adaptyObject']['regionCode']},
                    {'analytic_name': 'vendorProductId', 'analytic_value': analytics_data['adaptyObject']['vendorProductId']},
                    {'analytic_name': 'paywallName', 'analytic_value': analytics_data['adaptyObject']['paywallName']}
                ]
            }
            
            test_data.append(api_data)
        
        return pd.DataFrame(test_data)
    
    def test_function_safely(self, func_name, func, *args, **kwargs):
        """Test a function and catch categorical ordering errors"""
        print(f"\n🧪 Testing {func_name}...")
        
        try:
            result = func(*args, **kwargs)
            print(f"✅ {func_name}: SUCCESS")
            self.test_results[func_name] = {'status': 'SUCCESS', 'result': result}
            return result
            
        except Exception as e:
            error_msg = str(e)
            print(f"❌ {func_name}: FAILED")
            print(f"   Error: {error_msg}")
            
            if "'values' is not ordered" in error_msg or "categories" in error_msg.lower():
                print(f"   🎯 CATEGORICAL ORDERING ERROR DETECTED!")
                self.error_locations.append(func_name)
            
            print(f"   Traceback:")
            traceback.print_exc()
            
            self.test_results[func_name] = {
                'status': 'FAILED', 
                'error': error_msg,
                'traceback': traceback.format_exc()
            }
            return None
    
    def run_comprehensive_debug(self):
        """Run comprehensive debugging of all data loading functions"""
        print("🔍 Starting Comprehensive Categorical Ordering Debug")
        print("=" * 60)
        
        # Create test data
        test_df = self.create_test_data()
        print(f"📊 Created test data with {len(test_df)} records")
        
        # Test each function individually
        print("\n🧪 Testing individual functions...")
        
        # Test parse_payment_data
        self.test_function_safely("parse_payment_data", parse_payment_data, test_df)
        
        # Test process_payment_api_data
        self.test_function_safely("process_payment_api_data", process_payment_api_data, test_df)
        
        # Test process_user_tracking_api_data
        self.test_function_safely("process_user_tracking_api_data", process_user_tracking_api_data, test_df)
        
        # Test load_payment_data and load_user_tracking_data (these are likely where the error occurs)
        print("\n🎯 Testing data loading functions (most likely error source)...")
        try:
            # Test payment data loading
            self.test_function_safely("load_payment_data", load_payment_data, "Last 7 Days")
            
            # Test user tracking data loading
            self.test_function_safely("load_user_tracking_data", load_user_tracking_data, "Last 7 Days")
            
        except Exception as e:
            print(f"❌ Data loading setup failed: {e}")
        
        # Test DataFrame operations that might cause issues
        print("\n🧪 Testing DataFrame operations...")
        
        # Test groupby operations
        try:
            if not test_df.empty:
                # Test groupby on potentially categorical columns
                self.test_function_safely("groupby_category", lambda df: df.groupby('category').size(), test_df)
                self.test_function_safely("groupby_name", lambda df: df.groupby('name').size(), test_df)
                
        except Exception as e:
            print(f"❌ DataFrame operations test failed: {e}")
        
        # Generate report
        self.generate_debug_report()
    
    def generate_debug_report(self):
        """Generate a comprehensive debug report"""
        print("\n" + "=" * 60)
        print("📋 CATEGORICAL ORDERING DEBUG REPORT")
        print("=" * 60)
        
        total_tests = len(self.test_results)
        failed_tests = sum(1 for r in self.test_results.values() if r['status'] == 'FAILED')
        passed_tests = total_tests - failed_tests
        
        print(f"📊 Test Summary:")
        print(f"   Total Tests: {total_tests}")
        print(f"   ✅ Passed: {passed_tests}")
        print(f"   ❌ Failed: {failed_tests}")
        
        if self.error_locations:
            print(f"\n🎯 Categorical Ordering Errors Found In:")
            for location in self.error_locations:
                print(f"   - {location}")
        else:
            print(f"\n✅ No categorical ordering errors detected in tested functions")
        
        if failed_tests > 0:
            print(f"\n❌ Failed Tests Details:")
            for test_name, result in self.test_results.items():
                if result['status'] == 'FAILED':
                    print(f"\n   {test_name}:")
                    print(f"     Error: {result['error']}")
        
        # Save detailed report
        report_data = {
            'timestamp': datetime.now().isoformat(),
            'summary': {
                'total_tests': total_tests,
                'passed_tests': passed_tests,
                'failed_tests': failed_tests,
                'categorical_errors': len(self.error_locations)
            },
            'error_locations': self.error_locations,
            'detailed_results': self.test_results
        }
        
        report_file = 'test/categorical_ordering_debug_report.json'
        try:
            with open(report_file, 'w') as f:
                json.dump(report_data, f, indent=2, default=str)
            print(f"\n📁 Detailed report saved to: {report_file}")
        except Exception as e:
            print(f"❌ Failed to save report: {e}")

def main():
    """Run the categorical ordering debugger"""
    debugger = CategoricalOrderingDebugger()
    debugger.run_comprehensive_debug()

if __name__ == "__main__":
    main()