#!/usr/bin/env python3
"""
Comprehensive Tests for Session Reconstruction Algorithm
Tests the session reconstruction and metrics calculation with known patterns and edge cases.
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime, timedelta
import unittest
from typing import Dict, List, Tuple
import warnings
warnings.filterwarnings('ignore')

# Import the functions to test
from main import reconstruct_sessions, calculate_session_metrics
from test.comprehensive_test_data_generator import AnalyticsTestDataGenerator


class TestSessionReconstruction(unittest.TestCase):
    """Test cases for session reconstruction algorithm"""
    
    def setUp(self):
        """Set up test data generator"""
        self.generator = AnalyticsTestDataGenerator(seed=42)
        self.session_timeout = 30  # 30 minutes timeout
    
    def test_single_event_session(self):
        """Test session reconstruction with single event per user"""
        print("\n🧪 Testing single event sessions...")
        
        # Create data with exactly one event per user
        events = []
        for user_id in range(1, 6):  # 5 users
            event_time = datetime.now() - timedelta(hours=user_id)
            events.append({
                'analyticsid': user_id,
                'userid': user_id,
                'deviceid': f'device_{user_id}',
                'appversion': '1.0.0',
                'category': 'app_event',
                'name': 'open_SplashScreen',
                'datetimeutc': event_time,
                'appname': 'TestApp',
                'analyticsdata': '',
                'membershipid': '',
                'created_at': event_time,
                'updated_at': event_time
            })
        
        df = pd.DataFrame(events)
        df['datetimeutc'] = pd.to_datetime(df['datetimeutc'])
        
        # Test session reconstruction
        sessions = reconstruct_sessions(df, timeout_minutes=self.session_timeout)
        
        # Validate results
        self.assertEqual(len(sessions), 5, "Should have 5 sessions (one per user)")
        
        for session in sessions:
            self.assertEqual(session['event_count'], 1, "Each session should have exactly 1 event")
            self.assertEqual(session['duration_minutes'], 0, "Single event sessions should have 0 duration")
        
        print(f"✅ Single event test passed: {len(sessions)} sessions created")
        return sessions
    
    def test_timeout_boundary_sessions(self):
        """Test session reconstruction at timeout boundaries"""
        print("\n🧪 Testing timeout boundary sessions...")
        
        base_time = datetime.now() - timedelta(hours=2)
        events = []
        event_id = 1
        
        # User 1: Events exactly at timeout boundary (should be 2 sessions)
        user1_events = [
            base_time,
            base_time + timedelta(minutes=29),  # Within timeout
            base_time + timedelta(minutes=30),  # Exactly at timeout - new session
            base_time + timedelta(minutes=31)   # Same session as previous
        ]
        
        for i, event_time in enumerate(user1_events):
            events.append({
                'analyticsid': event_id,
                'userid': 1,
                'deviceid': 'device_1',
                'appversion': '1.0.0',
                'category': 'app_event',
                'name': f'event_{i}',
                'datetimeutc': event_time,
                'appname': 'TestApp',
                'analyticsdata': '',
                'membershipid': '',
                'created_at': event_time,
                'updated_at': event_time
            })
            event_id += 1
        
        # User 2: Events just over timeout boundary (should be 2 sessions)
        user2_events = [
            base_time + timedelta(hours=1),
            base_time + timedelta(hours=1, minutes=31)  # Just over 30 min timeout
        ]
        
        for i, event_time in enumerate(user2_events):
            events.append({
                'analyticsid': event_id,
                'userid': 2,
                'deviceid': 'device_2',
                'appversion': '1.0.0',
                'category': 'app_event',
                'name': f'event_{i}',
                'datetimeutc': event_time,
                'appname': 'TestApp',
                'analyticsdata': '',
                'membershipid': '',
                'created_at': event_time,
                'updated_at': event_time
            })
            event_id += 1
        
        df = pd.DataFrame(events)
        df['datetimeutc'] = pd.to_datetime(df['datetimeutc'])
        
        # Test session reconstruction
        sessions = reconstruct_sessions(df, timeout_minutes=self.session_timeout)
        
        # Validate results
        user1_sessions = [s for s in sessions if s['userid'] == 1]
        user2_sessions = [s for s in sessions if s['userid'] == 2]
        
        self.assertEqual(len(user1_sessions), 2, "User 1 should have 2 sessions")
        self.assertEqual(len(user2_sessions), 2, "User 2 should have 2 sessions")
        
        # Check event counts
        self.assertEqual(user1_sessions[0]['event_count'], 2, "First session should have 2 events")
        self.assertEqual(user1_sessions[1]['event_count'], 2, "Second session should have 2 events")
        
        print(f"✅ Timeout boundary test passed: {len(sessions)} sessions created")
        return sessions
    
    def test_overlapping_sessions_multiple_users(self):
        """Test session reconstruction with overlapping sessions from multiple users"""
        print("\n🧪 Testing overlapping sessions from multiple users...")
        
        base_time = datetime.now() - timedelta(hours=1)
        events = []
        event_id = 1
        
        # Create overlapping sessions for 3 users
        for user_id in range(1, 4):
            # Each user has 2 sessions with 3 events each
            for session_idx in range(2):
                session_start = base_time + timedelta(minutes=user_id * 10 + session_idx * 60)
                
                for event_idx in range(3):
                    event_time = session_start + timedelta(minutes=event_idx * 5)
                    events.append({
                        'analyticsid': event_id,
                        'userid': user_id,
                        'deviceid': f'device_{user_id}',
                        'appversion': '1.0.0',
                        'category': 'app_event',
                        'name': f'event_{event_idx}',
                        'datetimeutc': event_time,
                        'appname': 'TestApp',
                        'analyticsdata': '',
                        'membershipid': '',
                        'created_at': event_time,
                        'updated_at': event_time
                    })
                    event_id += 1
        
        df = pd.DataFrame(events)
        df['datetimeutc'] = pd.to_datetime(df['datetimeutc'])
        
        # Test session reconstruction
        sessions = reconstruct_sessions(df, timeout_minutes=self.session_timeout)
        
        # Validate results
        self.assertEqual(len(sessions), 6, "Should have 6 sessions (2 per user)")
        
        for user_id in range(1, 4):
            user_sessions = [s for s in sessions if s['userid'] == user_id]
            self.assertEqual(len(user_sessions), 2, f"User {user_id} should have 2 sessions")
            
            for session in user_sessions:
                self.assertEqual(session['event_count'], 3, "Each session should have 3 events")
                self.assertGreater(session['duration_minutes'], 0, "Sessions should have positive duration")
        
        print(f"✅ Overlapping sessions test passed: {len(sessions)} sessions created")
        return sessions
    
    def test_session_metrics_calculation(self):
        """Test session metrics calculation with known data"""
        print("\n🧪 Testing session metrics calculation...")
        
        # Generate test data with known patterns
        df = self.generator.generate_user_events(num_users=20, days_range=7)
        
        # Reconstruct sessions
        sessions = reconstruct_sessions(df, timeout_minutes=self.session_timeout)
        
        # Calculate metrics
        metrics = calculate_session_metrics(sessions)
        
        # Validate metrics structure
        required_metrics = [
            'total_sessions', 'unique_users', 'avg_session_duration',
            'avg_events_per_session', 'total_events'
        ]
        
        for metric in required_metrics:
            self.assertIn(metric, metrics, f"Metric '{metric}' should be present")
            self.assertIsNotNone(metrics[metric], f"Metric '{metric}' should not be None")
        
        # Validate metric values
        self.assertGreater(metrics['total_sessions'], 0, "Should have sessions")
        self.assertGreater(metrics['unique_users'], 0, "Should have users")
        self.assertGreaterEqual(metrics['avg_session_duration'], 0, "Duration should be non-negative")
        self.assertGreater(metrics['avg_events_per_session'], 0, "Should have events per session")
        
        # Cross-validate metrics
        total_events_from_sessions = sum(s['event_count'] for s in sessions)
        self.assertEqual(metrics['total_events'], total_events_from_sessions, 
                        "Total events should match sum of session events")
        
        print(f"✅ Session metrics test passed:")
        print(f"   - Total sessions: {metrics['total_sessions']}")
        print(f"   - Unique users: {metrics['unique_users']}")
        print(f"   - Avg session duration: {metrics['avg_session_duration']:.2f} minutes")
        print(f"   - Avg events per session: {metrics['avg_events_per_session']:.2f}")
        
        return metrics, sessions
    
    def test_edge_case_empty_data(self):
        """Test session reconstruction with empty data"""
        print("\n🧪 Testing empty data edge case...")
        
        empty_df = pd.DataFrame(columns=[
            'analyticsid', 'userid', 'deviceid', 'appversion', 'category',
            'name', 'datetimeutc', 'appname', 'analyticsdata', 'membershipid',
            'created_at', 'updated_at'
        ])
        
        # Test session reconstruction
        sessions = reconstruct_sessions(empty_df, timeout_minutes=self.session_timeout)
        
        # Validate results
        self.assertEqual(len(sessions), 0, "Empty data should result in no sessions")
        
        # Test metrics calculation
        metrics = calculate_session_metrics(sessions)
        self.assertEqual(metrics['total_sessions'], 0, "Should have 0 sessions")
        self.assertEqual(metrics['unique_users'], 0, "Should have 0 users")
        
        print("✅ Empty data test passed")
        return sessions, metrics
    
    def test_session_duration_accuracy(self):
        """Test accuracy of session duration calculations"""
        print("\n🧪 Testing session duration accuracy...")
        
        base_time = datetime.now() - timedelta(hours=1)
        events = []
        
        # Create a session with known duration (exactly 15 minutes)
        session_events = [
            base_time,
            base_time + timedelta(minutes=5),
            base_time + timedelta(minutes=10),
            base_time + timedelta(minutes=15)  # Last event at 15 minutes
        ]
        
        for i, event_time in enumerate(session_events):
            events.append({
                'analyticsid': i + 1,
                'userid': 1,
                'deviceid': 'device_1',
                'appversion': '1.0.0',
                'category': 'app_event',
                'name': f'event_{i}',
                'datetimeutc': event_time,
                'appname': 'TestApp',
                'analyticsdata': '',
                'membershipid': '',
                'created_at': event_time,
                'updated_at': event_time
            })
        
        df = pd.DataFrame(events)
        df['datetimeutc'] = pd.to_datetime(df['datetimeutc'])
        
        # Test session reconstruction
        sessions = reconstruct_sessions(df, timeout_minutes=self.session_timeout)
        
        # Validate duration
        self.assertEqual(len(sessions), 1, "Should have exactly 1 session")
        self.assertEqual(sessions[0]['duration_minutes'], 15.0, 
                        "Session duration should be exactly 15 minutes")
        self.assertEqual(sessions[0]['event_count'], 4, "Should have 4 events")
        
        print(f"✅ Duration accuracy test passed: {sessions[0]['duration_minutes']} minutes")
        return sessions


class SessionReconstructionValidator:
    """Validator class for session reconstruction results"""
    
    def __init__(self):
        self.test_results = {}
    
    def run_comprehensive_validation(self):
        """Run all session reconstruction tests and generate validation report"""
        print("🔍 Running comprehensive session reconstruction validation...")
        
        test_suite = TestSessionReconstruction()
        test_suite.setUp()
        
        # Run all tests
        tests = [
            ('single_event', test_suite.test_single_event_session),
            ('timeout_boundary', test_suite.test_timeout_boundary_sessions),
            ('overlapping_sessions', test_suite.test_overlapping_sessions_multiple_users),
            ('session_metrics', test_suite.test_session_metrics_calculation),
            ('empty_data', test_suite.test_edge_case_empty_data),
            ('duration_accuracy', test_suite.test_session_duration_accuracy)
        ]
        
        for test_name, test_func in tests:
            try:
                result = test_func()
                self.test_results[test_name] = {'status': 'PASSED', 'result': result}
                print(f"✅ {test_name}: PASSED")
            except Exception as e:
                self.test_results[test_name] = {'status': 'FAILED', 'error': str(e)}
                print(f"❌ {test_name}: FAILED - {e}")
        
        return self.test_results
    
    def generate_validation_charts(self, save_path: str = None):
        """Generate visual validation charts for session reconstruction"""
        print("\n📊 Generating session reconstruction validation charts...")
        
        # Generate test data for visualization
        generator = AnalyticsTestDataGenerator(seed=42)
        df = generator.generate_user_events(num_users=50, days_range=14)
        
        # Reconstruct sessions
        sessions = reconstruct_sessions(df, timeout_minutes=30)
        metrics = calculate_session_metrics(sessions)
        
        # Create visualization
        fig, axes = plt.subplots(2, 3, figsize=(18, 12))
        fig.suptitle('Session Reconstruction Algorithm Validation', fontsize=16, fontweight='bold')
        
        # 1. Session duration distribution
        durations = [s['duration_minutes'] for s in sessions if s['duration_minutes'] > 0]
        axes[0, 0].hist(durations, bins=20, alpha=0.7, color='skyblue', edgecolor='black')
        axes[0, 0].set_title('Session Duration Distribution')
        axes[0, 0].set_xlabel('Duration (minutes)')
        axes[0, 0].set_ylabel('Frequency')
        axes[0, 0].axvline(np.mean(durations), color='red', linestyle='--', 
                          label=f'Mean: {np.mean(durations):.1f} min')
        axes[0, 0].legend()
        
        # 2. Events per session distribution
        event_counts = [s['event_count'] for s in sessions]
        axes[0, 1].hist(event_counts, bins=15, alpha=0.7, color='lightgreen', edgecolor='black')
        axes[0, 1].set_title('Events per Session Distribution')
        axes[0, 1].set_xlabel('Number of Events')
        axes[0, 1].set_ylabel('Frequency')
        axes[0, 1].axvline(np.mean(event_counts), color='red', linestyle='--',
                          label=f'Mean: {np.mean(event_counts):.1f}')
        axes[0, 1].legend()
        
        # 3. Sessions per user distribution
        user_session_counts = {}
        for session in sessions:
            user_id = session['userid']
            user_session_counts[user_id] = user_session_counts.get(user_id, 0) + 1
        
        session_counts = list(user_session_counts.values())
        axes[0, 2].hist(session_counts, bins=10, alpha=0.7, color='orange', edgecolor='black')
        axes[0, 2].set_title('Sessions per User Distribution')
        axes[0, 2].set_xlabel('Number of Sessions')
        axes[0, 2].set_ylabel('Frequency')
        axes[0, 2].axvline(np.mean(session_counts), color='red', linestyle='--',
                          label=f'Mean: {np.mean(session_counts):.1f}')
        axes[0, 2].legend()
        
        # 4. Session timeline for sample users
        sample_users = list(user_session_counts.keys())[:5]
        colors = plt.cm.Set3(np.linspace(0, 1, len(sample_users)))
        
        for i, user_id in enumerate(sample_users):
            user_sessions = [s for s in sessions if s['userid'] == user_id]
            for j, session in enumerate(user_sessions):
                start_time = session['start_time']
                duration = session['duration_minutes']
                axes[1, 0].barh(i, duration, left=start_time.hour + start_time.minute/60, 
                               height=0.6, color=colors[i], alpha=0.7,
                               label=f'User {user_id}' if j == 0 else "")
        
        axes[1, 0].set_title('Session Timeline (Sample Users)')
        axes[1, 0].set_xlabel('Hour of Day')
        axes[1, 0].set_ylabel('User')
        axes[1, 0].set_yticks(range(len(sample_users)))
        axes[1, 0].set_yticklabels([f'User {uid}' for uid in sample_users])
        axes[1, 0].legend()
        
        # 5. Test results summary
        test_status = [result['status'] for result in self.test_results.values()]
        passed_count = test_status.count('PASSED')
        failed_count = test_status.count('FAILED')
        
        axes[1, 1].pie([passed_count, failed_count], labels=['Passed', 'Failed'],
                      colors=['lightgreen', 'lightcoral'], autopct='%1.1f%%')
        axes[1, 1].set_title(f'Test Results Summary\n({passed_count}/{len(test_status)} tests passed)')
        
        # 6. Key metrics summary
        metrics_text = f"""
        Key Session Metrics:
        
        Total Sessions: {metrics['total_sessions']:,}
        Unique Users: {metrics['unique_users']:,}
        Avg Duration: {metrics['avg_session_duration']:.1f} min
        Avg Events/Session: {metrics['avg_events_per_session']:.1f}
        Total Events: {metrics['total_events']:,}
        
        Session Timeout: 30 minutes
        Test Data: 50 users, 14 days
        """
        
        axes[1, 2].text(0.1, 0.5, metrics_text, transform=axes[1, 2].transAxes,
                        fontsize=10, verticalalignment='center',
                        bbox=dict(boxstyle="round,pad=0.3", facecolor="lightblue", alpha=0.7))
        axes[1, 2].set_title('Session Reconstruction Summary')
        axes[1, 2].axis('off')
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"📊 Validation charts saved to: {save_path}")
        
        plt.show()
        
        return fig


def main():
    """Run comprehensive session reconstruction validation"""
    print("🚀 Starting Session Reconstruction Algorithm Validation")
    print("=" * 60)
    
    # Create validator
    validator = SessionReconstructionValidator()
    
    # Run validation tests
    results = validator.run_comprehensive_validation()
    
    # Generate validation charts
    chart_path = "test/session_reconstruction_validation.png"
    validator.generate_validation_charts(save_path=chart_path)
    
    # Print summary
    print("\n" + "=" * 60)
    print("📋 VALIDATION SUMMARY")
    print("=" * 60)
    
    passed_tests = sum(1 for r in results.values() if r['status'] == 'PASSED')
    total_tests = len(results)
    
    print(f"✅ Tests Passed: {passed_tests}/{total_tests}")
    
    if passed_tests == total_tests:
        print("🎉 All session reconstruction tests PASSED!")
        print("✅ Session reconstruction algorithm is working correctly")
    else:
        print("⚠️  Some tests failed. Please review the results above.")
        
        for test_name, result in results.items():
            if result['status'] == 'FAILED':
                print(f"❌ {test_name}: {result['error']}")
    
    print("\n📊 Visual validation charts generated successfully")
    print(f"📁 Charts saved to: {chart_path}")


if __name__ == "__main__":
    main()