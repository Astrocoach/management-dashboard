#!/usr/bin/env python3
"""
Simple Validation Runner for Analytics Algorithms
Tests core algorithms directly with minimal dependencies
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime, timedelta
import time
import warnings
warnings.filterwarnings('ignore')

# Import the functions to test
from main import (
    reconstruct_sessions, calculate_session_metrics, 
    perform_user_segmentation, detect_anomalies
)


class SimpleAlgorithmValidator:
    """Simple validator for core algorithms"""
    
    def __init__(self):
        self.test_results = {}
        self.start_time = time.time()
    
    def generate_test_data(self, num_users=100, days_range=30):
        """Generate simple test data matching main.py expected format"""
        print("📊 Generating test data...")
        
        # Generate user events with correct column names for main.py
        events = []
        
        for user_id in range(1, num_users + 1):
            # Generate random events for each user
            num_events = np.random.randint(5, 50)
            base_date = datetime.now() - timedelta(days=days_range)
            
            for event_idx in range(num_events):
                event_date = base_date + timedelta(
                    days=np.random.randint(0, days_range),
                    hours=np.random.randint(0, 24),
                    minutes=np.random.randint(0, 60)
                )
                
                events.append({
                    'analyticsid': len(events) + 1,
                    'userid': user_id,  # Use userid as expected by main.py
                    'deviceid': f'device_{user_id}',
                    'appversion': '1.0.0',
                    'category': 'app_event',
                    'name': np.random.choice(['login', 'page_view', 'purchase', 'logout']),
                    'datetimeutc': event_date,  # Use datetimeutc as expected by main.py
                    'appname': 'TestApp',
                    'analyticsdata': '',
                    'membershipid': '',
                    'created_at': event_date,
                    'updated_at': event_date
                })
        
        df = pd.DataFrame(events)
        df['datetimeutc'] = pd.to_datetime(df['datetimeutc'])
        
        print(f"✅ Generated {len(df)} events for {num_users} users")
        return df
    
    def test_session_reconstruction(self, df):
        """Test session reconstruction algorithm"""
        print("\n🧪 Testing Session Reconstruction...")
        
        try:
            start_time = time.time()
            sessions_df = reconstruct_sessions(df, timeout_minutes=30)
            end_time = time.time()
            
            # Validate results - reconstruct_sessions returns the original df with session_id added
            assert len(sessions_df) > 0, "No sessions were reconstructed"
            assert 'session_id' in sessions_df.columns, "Missing session_id column"
            assert 'userid' in sessions_df.columns, "Missing userid column"
            
            # Check session metrics
            session_metrics = calculate_session_metrics(sessions_df)
            assert session_metrics is not None, "No session metrics calculated"
            assert isinstance(session_metrics, pd.DataFrame), "Session metrics should be a DataFrame"
            assert not session_metrics.empty, "Session metrics DataFrame should not be empty"
            
            execution_time = end_time - start_time
            
            result = {
                'status': 'PASSED',
                'sessions_count': len(sessions_df),
                'unique_users': sessions_df['userid'].nunique(),
                'unique_session_ids': sessions_df['session_id'].nunique(),
                'execution_time': execution_time,
                'metrics_rows': len(session_metrics)
            }
            
            print(f"✅ Session Reconstruction: {result['sessions_count']} events with {result['unique_session_ids']} sessions for {result['unique_users']} users")
            print(f"   Execution time: {execution_time:.3f}s")
            
            return result
            
        except Exception as e:
            print(f"❌ Session Reconstruction failed: {e}")
            return {'status': 'FAILED', 'error': str(e)}
    
    def test_user_segmentation(self, df):
        """Test user segmentation algorithm"""
        print("\n🧪 Testing User Segmentation...")
        
        try:
            # First, we need to create user metrics from the session data
            sessions_df = reconstruct_sessions(df, timeout_minutes=30)
            
            # Calculate user metrics for segmentation
            user_metrics = sessions_df.groupby('userid').agg({
                'session_id': 'nunique',  # total_sessions
                'datetimeutc': ['min', 'max'],  # for days_active calculation
                'analyticsid': 'count'  # total_events
            }).reset_index()
            
            # Flatten column names
            user_metrics.columns = ['userid', 'total_sessions', 'first_event', 'last_event', 'total_events']
            
            # Calculate days active and average session duration
            user_metrics['days_active'] = (user_metrics['last_event'] - user_metrics['first_event']).dt.days + 1
            user_metrics['avg_session_duration'] = np.random.uniform(60, 300, len(user_metrics))  # Mock duration
            
            start_time = time.time()
            segmentation_result = perform_user_segmentation(user_metrics, n_clusters=4)
            end_time = time.time()
            
            # Validate results
            assert segmentation_result is not None, "Segmentation returned None"
            assert isinstance(segmentation_result, pd.DataFrame), "Segmentation should return DataFrame"
            assert len(segmentation_result) > 0, "No user segments created"
            assert 'segment' in segmentation_result.columns, "Missing segment column"
            assert 'segment_label' in segmentation_result.columns, "Missing segment_label column"
            
            execution_time = end_time - start_time
            unique_clusters = segmentation_result['segment'].nunique()
            
            result = {
                'status': 'PASSED',
                'users_segmented': len(segmentation_result),
                'clusters_found': unique_clusters,
                'execution_time': execution_time
            }
            
            print(f"✅ User Segmentation: {result['users_segmented']} users in {result['clusters_found']} clusters")
            print(f"   Execution time: {execution_time:.3f}s")
            
            return result
            
        except Exception as e:
            print(f"❌ User Segmentation failed: {e}")
            return {'status': 'FAILED', 'error': str(e)}
    
    def test_anomaly_detection(self, df):
        """Test anomaly detection algorithm"""
        print("\n🧪 Testing Anomaly Detection...")
        
        try:
            # First reconstruct sessions for anomaly detection
            sessions_df = reconstruct_sessions(df, timeout_minutes=30)
            
            start_time = time.time()
            anomalies = detect_anomalies(sessions_df)
            end_time = time.time()
            
            # Validate results
            assert anomalies is not None, "Anomaly detection returned None"
            
            execution_time = end_time - start_time
            
            if isinstance(anomalies, pd.DataFrame):
                anomaly_count = len(anomalies)
                total_records = len(sessions_df)
                anomaly_rate = (anomaly_count / total_records) * 100 if total_records > 0 else 0
            elif isinstance(anomalies, (list, np.ndarray)):
                anomaly_count = len(anomalies)
                total_records = len(sessions_df)
                anomaly_rate = (anomaly_count / total_records) * 100 if total_records > 0 else 0
            else:
                anomaly_count = 0
                anomaly_rate = 0
                total_records = len(sessions_df)
            
            result = {
                'status': 'PASSED',
                'total_records': total_records,
                'anomalies_detected': anomaly_count,
                'anomaly_rate': anomaly_rate,
                'execution_time': execution_time
            }
            
            print(f"✅ Anomaly Detection: {result['anomalies_detected']} anomalies ({result['anomaly_rate']:.1f}%)")
            print(f"   Execution time: {execution_time:.3f}s")
            
            return result
            
        except Exception as e:
            print(f"❌ Anomaly Detection failed: {e}")
            return {'status': 'FAILED', 'error': str(e)}
    
    def test_data_processing_edge_cases(self, df):
        """Test edge cases for data processing"""
        print("\n🧪 Testing Edge Cases...")
        
        edge_case_results = {}
        
        # Test 1: Empty dataframe
        try:
            empty_df = pd.DataFrame(columns=df.columns)
            sessions = reconstruct_sessions(empty_df, timeout_minutes=30)
            edge_case_results['empty_data'] = 'PASSED' if len(sessions) == 0 else 'FAILED'
        except:
            edge_case_results['empty_data'] = 'FAILED'
        
        # Test 2: Single user
        try:
            single_user_df = df[df['user_id'] == df['user_id'].iloc[0]].copy()
            sessions = reconstruct_sessions(single_user_df, timeout_minutes=30)
            edge_case_results['single_user'] = 'PASSED' if len(sessions) > 0 else 'FAILED'
        except:
            edge_case_results['single_user'] = 'FAILED'
        
        # Test 3: Single event per user
        try:
            single_event_df = df.groupby('user_id').first().reset_index()
            sessions = reconstruct_sessions(single_event_df, timeout_minutes=30)
            edge_case_results['single_events'] = 'PASSED' if len(sessions) > 0 else 'FAILED'
        except:
            edge_case_results['single_events'] = 'FAILED'
        
        passed_edge_cases = sum(1 for result in edge_case_results.values() if result == 'PASSED')
        total_edge_cases = len(edge_case_results)
        
        print(f"✅ Edge Cases: {passed_edge_cases}/{total_edge_cases} passed")
        
        return {
            'status': 'PASSED' if passed_edge_cases == total_edge_cases else 'PARTIAL',
            'edge_cases': edge_case_results,
            'passed': passed_edge_cases,
            'total': total_edge_cases
        }
    
    def generate_validation_summary(self):
        """Generate validation summary with charts"""
        print("\n📊 Generating Validation Summary...")
        
        # Create summary visualization
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        fig.suptitle('Algorithm Validation Summary', fontsize=16, fontweight='bold')
        
        # 1. Test Results Overview
        test_names = list(self.test_results.keys())
        test_statuses = [result.get('status', 'UNKNOWN') for result in self.test_results.values()]
        
        passed_count = test_statuses.count('PASSED')
        failed_count = test_statuses.count('FAILED')
        partial_count = test_statuses.count('PARTIAL')
        
        axes[0, 0].pie([passed_count, failed_count, partial_count], 
                      labels=['Passed', 'Failed', 'Partial'],
                      colors=['lightgreen', 'lightcoral', 'lightyellow'],
                      autopct='%1.1f%%')
        axes[0, 0].set_title('Test Results Overview')
        
        # 2. Execution Times
        execution_times = []
        algorithm_names = []
        
        for test_name, result in self.test_results.items():
            if 'execution_time' in result:
                execution_times.append(result['execution_time'])
                algorithm_names.append(test_name.replace('_', ' ').title())
        
        if execution_times:
            bars = axes[0, 1].bar(algorithm_names, execution_times, color='skyblue')
            axes[0, 1].set_ylabel('Execution Time (seconds)')
            axes[0, 1].set_title('Algorithm Performance')
            axes[0, 1].tick_params(axis='x', rotation=45)
            
            # Add value labels on bars
            for bar, time_val in zip(bars, execution_times):
                axes[0, 1].text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                               f'{time_val:.3f}s', ha='center', va='bottom')
        
        # 3. Algorithm Metrics
        metrics_data = {}
        
        if 'session_reconstruction' in self.test_results:
            sr_result = self.test_results['session_reconstruction']
            if 'sessions_count' in sr_result:
                metrics_data['Sessions Created'] = sr_result['sessions_count']
        
        if 'user_segmentation' in self.test_results:
            us_result = self.test_results['user_segmentation']
            if 'clusters_found' in us_result:
                metrics_data['Clusters Found'] = us_result['clusters_found']
        
        if 'anomaly_detection' in self.test_results:
            ad_result = self.test_results['anomaly_detection']
            if 'anomalies_detected' in ad_result:
                metrics_data['Anomalies Detected'] = ad_result['anomalies_detected']
        
        if metrics_data:
            metric_names = list(metrics_data.keys())
            metric_values = list(metrics_data.values())
            
            bars = axes[1, 0].bar(metric_names, metric_values, color='lightcoral')
            axes[1, 0].set_ylabel('Count')
            axes[1, 0].set_title('Algorithm Output Metrics')
            axes[1, 0].tick_params(axis='x', rotation=45)
            
            # Add value labels
            for bar, value in zip(bars, metric_values):
                axes[1, 0].text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.5,
                               str(value), ha='center', va='bottom')
        
        # 4. Validation Summary Text
        total_execution_time = time.time() - self.start_time
        
        summary_text = f"""
        Validation Summary:
        
        Total Tests: {len(self.test_results)}
        Passed: {passed_count}
        Failed: {failed_count}
        Partial: {partial_count}
        
        Total Execution Time: {total_execution_time:.2f}s
        
        Status: {'✅ ALL PASSED' if failed_count == 0 else '⚠️ ISSUES FOUND'}
        """
        
        axes[1, 1].text(0.1, 0.5, summary_text, transform=axes[1, 1].transAxes,
                        fontsize=12, verticalalignment='center',
                        bbox=dict(boxstyle="round,pad=0.3", facecolor="lightblue", alpha=0.7))
        axes[1, 1].set_title('Validation Summary')
        axes[1, 1].axis('off')
        
        plt.tight_layout()
        
        # Save chart
        chart_path = "test/simple_validation_summary.png"
        plt.savefig(chart_path, dpi=300, bbox_inches='tight')
        print(f"📊 Validation summary saved to: {chart_path}")
        
        plt.show()
        
        return chart_path
    
    def run_comprehensive_validation(self):
        """Run all validation tests"""
        print("🚀 STARTING SIMPLE ALGORITHM VALIDATION")
        print("=" * 60)
        
        # Generate test data
        test_df = self.generate_test_data(num_users=50, days_range=30)
        
        # Run tests
        self.test_results['session_reconstruction'] = self.test_session_reconstruction(test_df)
        self.test_results['user_segmentation'] = self.test_user_segmentation(test_df)
        self.test_results['anomaly_detection'] = self.test_anomaly_detection(test_df)
        self.test_results['edge_cases'] = self.test_data_processing_edge_cases(test_df)
        
        # Generate summary
        chart_path = self.generate_validation_summary()
        
        # Print final results
        print("\n" + "=" * 60)
        print("📋 VALIDATION RESULTS SUMMARY")
        print("=" * 60)
        
        passed_tests = sum(1 for result in self.test_results.values() if result.get('status') == 'PASSED')
        total_tests = len(self.test_results)
        
        for test_name, result in self.test_results.items():
            status = result.get('status', 'UNKNOWN')
            icon = "✅" if status == 'PASSED' else "⚠️" if status == 'PARTIAL' else "❌"
            print(f"{icon} {test_name.replace('_', ' ').title()}: {status}")
            
            if 'error' in result:
                print(f"   Error: {result['error']}")
        
        print(f"\nOverall: {passed_tests}/{total_tests} tests passed")
        
        if passed_tests == total_tests:
            print("🎉 ALL ALGORITHMS VALIDATED SUCCESSFULLY!")
        else:
            print("⚠️ Some tests failed. Please review the results.")
        
        print(f"\n📊 Validation chart saved to: {chart_path}")
        
        return self.test_results


def main():
    """Main execution function"""
    validator = SimpleAlgorithmValidator()
    results = validator.run_comprehensive_validation()
    
    return results


if __name__ == "__main__":
    main()