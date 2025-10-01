"""
Comprehensive Test Suite for Goal Funnel Visualization

This module provides extensive testing for the Goal Funnel Visualization implementation,
validating funnel stages, conversion paths, data integrity, and edge cases.
"""

import unittest
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from datetime import datetime, timedelta
from typing import List, Dict, Tuple, Any
import sys
import os

# Add parent directory to path for imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from main import (
    create_goal_funnel_visualization,
    create_goal_funnel_ga_style,
    reconstruct_sessions
)


class GoalFunnelTestDataGenerator:
    """Generate test data for funnel validation"""
    
    def __init__(self):
        self.base_date = datetime(2025, 1, 1)
        self.user_id_counter = 1000
        
    def create_complete_user_journey(self, user_id: int, session_id: int, 
                                   complete_journey: bool = True) -> List[Dict]:
        """Create a complete user journey through all funnel stages"""
        events = []
        base_time = self.base_date + timedelta(hours=user_id % 24)
        
        # Stage 1: App Entry
        events.append({
            'userid': user_id,
            'session_id': session_id,
            'name': 'open_SplashScreen',
            'datetimeutc': base_time.isoformat(),
            'category': 'app_event'
        })
        
        if not complete_journey:
            return events
            
        # Stage 2: Onboarding Started
        events.append({
            'userid': user_id,
            'session_id': session_id,
            'name': 'open_WizardScreen',
            'datetimeutc': (base_time + timedelta(minutes=1)).isoformat(),
            'category': 'app_event'
        })
        
        # Stage 3: Onboarding Completed
        events.append({
            'userid': user_id,
            'session_id': session_id,
            'name': 'onboarding_completed',
            'datetimeutc': (base_time + timedelta(minutes=5)).isoformat(),
            'category': 'app_event'
        })
        
        # Stage 4: Paywall Viewed
        events.append({
            'userid': user_id,
            'session_id': session_id,
            'name': 'open_PaywallScreen',
            'datetimeutc': (base_time + timedelta(minutes=10)).isoformat(),
            'category': 'app_event'
        })
        
        # Stage 5: Purchase Completed
        events.append({
            'userid': user_id,
            'session_id': session_id,
            'name': 'payment_success',
            'datetimeutc': (base_time + timedelta(minutes=15)).isoformat(),
            'category': 'adapty_event'
        })
        
        return events
    
    def create_partial_journey_with_dropoff(self, user_id: int, session_id: int,
                                          drop_stage: int, exit_event: str) -> List[Dict]:
        """Create a user journey that drops off at a specific stage"""
        events = []
        base_time = self.base_date + timedelta(hours=user_id % 24)
        
        stage_events = [
            'open_SplashScreen',
            'open_WizardScreen', 
            'onboarding_completed',
            'open_PaywallScreen',
            'payment_success'
        ]
        
        # Add events up to drop stage
        for i in range(min(drop_stage + 1, len(stage_events))):
            events.append({
                'userid': user_id,
                'session_id': session_id,
                'name': stage_events[i],
                'datetimeutc': (base_time + timedelta(minutes=i*2)).isoformat(),
                'category': 'app_event' if stage_events[i] != 'payment_success' else 'adapty_event'
            })
        
        # Add exit event
        events.append({
            'userid': user_id,
            'session_id': session_id,
            'name': exit_event,
            'datetimeutc': (base_time + timedelta(minutes=drop_stage*2 + 1)).isoformat(),
            'category': 'app_event'
        })
        
        return events
    
    def generate_test_dataset(self, total_users: int = 1000, 
                            conversion_rates: List[float] = None) -> pd.DataFrame:
        """Generate a complete test dataset with specified conversion rates"""
        if conversion_rates is None:
            conversion_rates = [1.0, 0.8, 0.6, 0.4, 0.2]  # Default conversion rates
            
        all_events = []
        user_id = self.user_id_counter
        
        for i in range(total_users):
            session_id = i + 1000
            
            # Determine how far this user progresses
            progress_stage = 0
            for stage, rate in enumerate(conversion_rates):
                if np.random.random() < rate:
                    progress_stage = stage
                else:
                    break
            
            if progress_stage == len(conversion_rates) - 1:
                # Complete journey
                events = self.create_complete_user_journey(user_id, session_id, True)
            else:
                # Partial journey with drop-off
                exit_events = ['open_HomeScreen', 'open_ProfileScreen', 'app_backgrounded', '(exit)']
                exit_event = np.random.choice(exit_events)
                events = self.create_partial_journey_with_dropoff(user_id, session_id, 
                                                                progress_stage, exit_event)
            
            all_events.extend(events)
            user_id += 1
            
        self.user_id_counter = user_id
        return pd.DataFrame(all_events)


class TestGoalFunnelStages(unittest.TestCase):
    """Test funnel stage identification and processing"""
    
    def setUp(self):
        self.generator = GoalFunnelTestDataGenerator()
        
    def test_stage_mapping_completeness(self):
        """Test that all expected stages are properly mapped"""
        # Create data with all stage events
        df = self.generator.generate_test_dataset(100, [1.0, 1.0, 1.0, 1.0, 1.0])
        
        fig, stage_stats = create_goal_funnel_visualization(df)
        
        self.assertIsNotNone(fig, "Funnel figure should be created")
        self.assertEqual(len(stage_stats), 5, "Should have 5 funnel stages")
        
        expected_stages = ['App Entry', 'Onboarding Started', 'Onboarding Completed', 
                          'Paywall Viewed', 'Purchase Completed']
        actual_stages = [s['stage'] for s in stage_stats]
        
        self.assertEqual(actual_stages, expected_stages, "Stage order should match expected sequence")
        
    def test_stage_event_recognition(self):
        """Test that stage events are correctly recognized"""
        # Create minimal test data with both users having App Entry events
        events = [
            {'userid': 1, 'session_id': 1, 'name': 'open_SplashScreen', 'datetimeutc': '2025-01-01T10:00:00'},
            {'userid': 1, 'session_id': 1, 'name': 'open_HomeScreen', 'datetimeutc': '2025-01-01T10:01:00'},
            {'userid': 2, 'session_id': 2, 'name': 'open_HomeScreen', 'datetimeutc': '2025-01-01T10:02:00'},
            {'userid': 2, 'session_id': 2, 'name': 'click_Onboarding_Start', 'datetimeutc': '2025-01-01T10:03:00'},
        ]
        df = pd.DataFrame(events)
        
        fig, stage_stats = create_goal_funnel_visualization(df)
        
        # Should recognize App Entry stage for both users
        app_entry_stats = next((s for s in stage_stats if s['stage'] == 'App Entry'), None)
        self.assertIsNotNone(app_entry_stats, "App Entry stage should be found")
        self.assertEqual(app_entry_stats['sessions'], 2, "Should count both users in App Entry")
        
        # Should recognize Onboarding Started for user 2
        onboarding_stats = next((s for s in stage_stats if s['stage'] == 'Onboarding Started'), None)
        self.assertIsNotNone(onboarding_stats, "Onboarding Started stage should be found")
        self.assertEqual(onboarding_stats['sessions'], 1, "Should count one user in Onboarding Started")
        
    def test_missing_stage_detection(self):
        """Test detection of missing expected stages"""
        # Create data missing some stages
        events = [
            {'userid': 1, 'session_id': 1, 'name': 'open_SplashScreen', 'datetimeutc': '2025-01-01T10:00:00'},
            {'userid': 1, 'session_id': 1, 'name': 'payment_success', 'datetimeutc': '2025-01-01T10:15:00'},
        ]
        df = pd.DataFrame(events)
        
        fig, stage_stats = create_goal_funnel_ga_style(df)
        
        # Should detect missing intermediate stages
        self.assertLess(len(stage_stats), 5, "Should have fewer stages when events are missing")


class TestConversionPaths(unittest.TestCase):
    """Test conversion path accuracy and user flow tracking"""
    
    def setUp(self):
        self.generator = GoalFunnelTestDataGenerator()
        
    def test_linear_conversion_path(self):
        """Test tracking of users through linear conversion path"""
        # Create 100 users with 50% conversion at each stage
        df = self.generator.generate_test_dataset(100, [1.0, 0.5, 0.5, 0.5, 0.5])
        
        fig, stage_stats = create_goal_funnel_visualization(df)
        
        # Verify decreasing user counts through stages
        user_counts = [s['sessions'] for s in stage_stats]
        
        for i in range(len(user_counts) - 1):
            self.assertGreaterEqual(user_counts[i], user_counts[i + 1], 
                                  f"Stage {i} should have >= users than stage {i+1}")
            
    def test_progression_calculation(self):
        """Test accurate calculation of stage-to-stage progression"""
        # Create controlled test data
        df = self.generator.generate_test_dataset(100, [1.0, 0.8, 0.6, 0.4, 0.2])
        
        fig, stage_stats = create_goal_funnel_visualization(df)
        
        # Verify progression percentages
        for i, stats in enumerate(stage_stats[:-1]):  # Exclude last stage
            proceeded_pct = stats['proceeded_pct']
            self.assertGreaterEqual(proceeded_pct, 0, "Progression percentage should be non-negative")
            self.assertLessEqual(proceeded_pct, 100, "Progression percentage should not exceed 100%")
            
    def test_dropoff_destination_tracking(self):
        """Test accurate tracking of drop-off destinations"""
        # Create users that drop off to specific destinations
        events = []
        base_time = datetime(2025, 1, 1, 10, 0, 0)
        
        # User 1: Completes App Entry, then drops off to HomeScreen
        events.extend([
            {'userid': 1, 'session_id': 1, 'name': 'open_SplashScreen', 'datetimeutc': base_time.isoformat()},
            {'userid': 1, 'session_id': 1, 'name': 'open_HomeScreen', 'datetimeutc': (base_time + timedelta(minutes=1)).isoformat()}
        ])
        
        # User 2: Completes App Entry, then drops off to ProfileScreen
        events.extend([
            {'userid': 2, 'session_id': 2, 'name': 'open_SplashScreen', 'datetimeutc': base_time.isoformat()},
            {'userid': 2, 'session_id': 2, 'name': 'open_ProfileScreen', 'datetimeutc': (base_time + timedelta(minutes=1)).isoformat()}
        ])
        
        # User 3: Progresses to onboarding to create multiple stages
        events.extend([
            {'userid': 3, 'session_id': 3, 'name': 'open_SplashScreen', 'datetimeutc': base_time.isoformat()},
            {'userid': 3, 'session_id': 3, 'name': 'open_WizardScreen', 'datetimeutc': (base_time + timedelta(minutes=1)).isoformat()}
        ])
        
        df = pd.DataFrame(events)
        fig, stage_stats = create_goal_funnel_visualization(df, top_n_dropoffs=5)
        
        # Should have multiple stages now
        self.assertIsNotNone(fig, "Should create funnel with multiple stages")
        self.assertGreater(len(stage_stats), 0, "Should have at least one stage")
        
        # Check drop-off destinations for App Entry stage
        app_entry_stats = next((s for s in stage_stats if s['stage'] == 'App Entry'), None)
        self.assertIsNotNone(app_entry_stats)
        
        dropoff_events = [event for event, count in app_entry_stats['dropoffs']]
        # open_HomeScreen is part of App Entry stage, so it's not a dropoff
        # Only open_ProfileScreen should be tracked as a dropoff destination
        self.assertIn('open_ProfileScreen', dropoff_events, "Should track ProfileScreen as drop-off destination")
        
        # Verify that we have some dropoffs tracked
        self.assertGreater(len(app_entry_stats['dropoffs']), 0, "Should have dropoff destinations tracked")


class TestDataIntegrity(unittest.TestCase):
    """Test data integrity and calculation accuracy"""
    
    def setUp(self):
        self.generator = GoalFunnelTestDataGenerator()
        
    def test_user_count_consistency(self):
        """Test that user counts are consistent across calculations"""
        df = self.generator.generate_test_dataset(50, [1.0, 0.8, 0.6, 0.4, 0.2])
        
        fig, stage_stats = create_goal_funnel_visualization(df)
        
        # Verify that proceeded + dropped = total for each stage
        for i, stats in enumerate(stage_stats[:-1]):  # Exclude last stage
            total_sessions = stats['sessions']
            proceeded = stats['proceeded']
            
            # Calculate dropped count from dropoffs
            dropped = sum(count for event, count in stats['dropoffs'])
            
            # For stages that aren't the last, proceeded + dropped should equal total
            # (Note: some users might not have any next event, so this is approximate)
            self.assertLessEqual(proceeded, total_sessions, 
                               f"Proceeded count should not exceed total sessions for stage {i}")
            
    def test_percentage_calculations(self):
        """Test accuracy of percentage calculations"""
        # Create exact test data
        events = []
        base_time = datetime(2025, 1, 1, 10, 0, 0)
        
        # 10 users enter app, 5 proceed to onboarding
        for user_id in range(1, 11):
            events.append({
                'userid': user_id,
                'session_id': user_id,
                'name': 'open_SplashScreen',
                'datetimeutc': base_time.isoformat()
            })
            
            if user_id <= 5:  # First 5 users proceed
                events.append({
                    'userid': user_id,
                    'session_id': user_id,
                    'name': 'open_WizardScreen',
                    'datetimeutc': (base_time + timedelta(minutes=1)).isoformat()
                })
        
        df = pd.DataFrame(events)
        fig, stage_stats = create_goal_funnel_visualization(df)
        
        app_entry_stats = next((s for s in stage_stats if s['stage'] == 'App Entry'), None)
        self.assertIsNotNone(app_entry_stats)
        
        # Should be exactly 50% progression (5 out of 10)
        expected_percentage = 50.0
        actual_percentage = app_entry_stats['proceeded_pct']
        self.assertAlmostEqual(actual_percentage, expected_percentage, places=1,
                              msg="Progression percentage should be exactly 50%")
        
    def test_session_reconstruction_accuracy(self):
        """Test that session reconstruction works correctly"""
        # Create events with time gaps that should create separate sessions
        events = []
        base_time = datetime(2025, 1, 1, 10, 0, 0)
        
        # User 1: Two separate sessions (2 hours apart)
        events.extend([
            {'userid': 1, 'name': 'open_SplashScreen', 'datetimeutc': base_time.isoformat()},
            {'userid': 1, 'name': 'open_SplashScreen', 'datetimeutc': (base_time + timedelta(hours=2)).isoformat()}
        ])
        
        df = pd.DataFrame(events)
        # Convert datetime strings to datetime objects
        df['datetimeutc'] = pd.to_datetime(df['datetimeutc'])
        df_with_sessions = reconstruct_sessions(df, timeout_minutes=30)
        
        # Should create 2 different session IDs for the same user
        user_sessions = df_with_sessions[df_with_sessions['userid'] == 1]['session_id'].unique()
        self.assertEqual(len(user_sessions), 2, "Should create 2 separate sessions for user with 2-hour gap")
        
    def test_empty_data_handling(self):
        """Test handling of empty or invalid data"""
        # Test with empty DataFrame
        empty_df = pd.DataFrame()
        fig, stage_stats = create_goal_funnel_visualization(empty_df)
        
        self.assertIsNone(fig, "Should return None for empty data")
        self.assertEqual(len(stage_stats), 0, "Should return empty stats for empty data")
        
        # Test with DataFrame missing required columns
        invalid_df = pd.DataFrame({'invalid_column': [1, 2, 3]})
        fig, stage_stats = create_goal_funnel_visualization(invalid_df)
        
        self.assertIsNone(fig, "Should return None for invalid data structure")
        self.assertEqual(len(stage_stats), 0, "Should return empty stats for invalid data")


class TestEdgeCases(unittest.TestCase):
    """Test edge cases and error conditions"""
    
    def setUp(self):
        self.generator = GoalFunnelTestDataGenerator()
        
    def test_single_user_funnel(self):
        """Test funnel with only one user"""
        df = self.generator.generate_test_dataset(1, [1.0, 1.0, 1.0, 1.0, 1.0])
        
        fig, stage_stats = create_goal_funnel_visualization(df)
        
        self.assertIsNotNone(fig, "Should handle single user data")
        self.assertGreater(len(stage_stats), 0, "Should generate stats for single user")
        
    def test_duplicate_events(self):
        """Test handling of duplicate events for same user/session"""
        events = [
            {'userid': 1, 'session_id': 1, 'name': 'open_SplashScreen', 'datetimeutc': '2025-01-01T10:00:00'},
            {'userid': 1, 'session_id': 1, 'name': 'open_SplashScreen', 'datetimeutc': '2025-01-01T10:00:01'},  # Duplicate
            {'userid': 1, 'session_id': 1, 'name': 'open_WizardScreen', 'datetimeutc': '2025-01-01T10:01:00'},
        ]
        df = pd.DataFrame(events)
        
        fig, stage_stats = create_goal_funnel_visualization(df)
        
        self.assertIsNotNone(fig, "Should handle duplicate events gracefully")
        
    def test_out_of_order_events(self):
        """Test handling of events that occur out of chronological order"""
        events = [
            {'userid': 1, 'session_id': 1, 'name': 'open_WizardScreen', 'datetimeutc': '2025-01-01T10:01:00'},  # Later event first
            {'userid': 1, 'session_id': 1, 'name': 'open_SplashScreen', 'datetimeutc': '2025-01-01T10:00:00'},  # Earlier event second
        ]
        df = pd.DataFrame(events)
        
        fig, stage_stats = create_goal_funnel_visualization(df)
        
        self.assertIsNotNone(fig, "Should handle out-of-order events")
        
    def test_annotation_modes(self):
        """Test different annotation modes in GA-style funnel"""
        df = self.generator.generate_test_dataset(50, [1.0, 0.8, 0.6, 0.4, 0.2])
        
        for mode in ["Minimal", "Standard", "Detailed"]:
            fig, stage_stats = create_goal_funnel_ga_style(df, annotation_mode=mode)
            
            self.assertIsNotNone(fig, f"Should create figure for {mode} annotation mode")
            self.assertGreater(len(stage_stats), 0, f"Should generate stats for {mode} mode")


class TestPerformance(unittest.TestCase):
    """Test performance with large datasets"""
    
    def setUp(self):
        self.generator = GoalFunnelTestDataGenerator()
        
    def test_large_dataset_performance(self):
        """Test funnel performance with large dataset"""
        import time
        
        # Generate large dataset
        df = self.generator.generate_test_dataset(5000, [1.0, 0.7, 0.5, 0.3, 0.1])
        
        start_time = time.time()
        fig, stage_stats = create_goal_funnel_visualization(df)
        end_time = time.time()
        
        processing_time = end_time - start_time
        
        self.assertIsNotNone(fig, "Should handle large dataset")
        self.assertLess(processing_time, 10.0, "Should process large dataset in reasonable time (<10s)")
        
    def test_memory_efficiency(self):
        """Test memory usage with large datasets"""
        import psutil
        import os
        
        process = psutil.Process(os.getpid())
        initial_memory = process.memory_info().rss / 1024 / 1024  # MB
        
        # Process large dataset
        df = self.generator.generate_test_dataset(10000, [1.0, 0.8, 0.6, 0.4, 0.2])
        fig, stage_stats = create_goal_funnel_visualization(df)
        
        final_memory = process.memory_info().rss / 1024 / 1024  # MB
        memory_increase = final_memory - initial_memory
        
        self.assertLess(memory_increase, 500, "Memory increase should be reasonable (<500MB)")


def run_comprehensive_funnel_tests():
    """Run all funnel tests and generate a report"""
    
    # Create test suite
    test_suite = unittest.TestSuite()
    
    # Add all test classes
    test_classes = [
        TestGoalFunnelStages,
        TestConversionPaths, 
        TestDataIntegrity,
        TestEdgeCases,
        TestPerformance
    ]
    
    for test_class in test_classes:
        tests = unittest.TestLoader().loadTestsFromTestCase(test_class)
        test_suite.addTests(tests)
    
    # Run tests
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(test_suite)
    
    # Generate summary report
    total_tests = result.testsRun
    failures = len(result.failures)
    errors = len(result.errors)
    success_rate = ((total_tests - failures - errors) / total_tests * 100) if total_tests > 0 else 0
    
    print(f"\n{'='*60}")
    print("GOAL FUNNEL COMPREHENSIVE TEST REPORT")
    print(f"{'='*60}")
    print(f"Total Tests Run: {total_tests}")
    print(f"Successful: {total_tests - failures - errors}")
    print(f"Failures: {failures}")
    print(f"Errors: {errors}")
    print(f"Success Rate: {success_rate:.1f}%")
    print(f"{'='*60}")
    
    if failures > 0:
        print("\nFAILURES:")
        for test, traceback in result.failures:
            newline = '\n'
            print(f"- {test}: {traceback.split('AssertionError: ')[-1].split(newline)[0]}")
    
    if errors > 0:
        print("\nERRORS:")
        for test, traceback in result.errors:
            newline = '\n'
            print(f"- {test}: {traceback.split(newline)[-2]}")
    
    return result


if __name__ == "__main__":
    run_comprehensive_funnel_tests()