"""
Funnel Validation Scenarios

This module tests specific real-world scenarios that could occur in the funnel,
particularly focusing on parameter removal impact and data quality issues.
"""

import unittest
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import sys
import os

# Add parent directory to path for imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from main import create_goal_funnel_visualization, create_goal_funnel_ga_style


class TestParameterRemovalImpact(unittest.TestCase):
    """Test scenarios related to parameter removal and its impact on funnel accuracy"""
    
    def test_missing_userid_handling(self):
        """Test funnel behavior when userid is missing or null"""
        events = [
            {'userid': 1, 'name': 'open_SplashScreen', 'datetimeutc': '2025-01-01T10:00:00'},
            {'userid': None, 'name': 'open_SplashScreen', 'datetimeutc': '2025-01-01T10:01:00'},  # Missing userid
            {'userid': 2, 'name': 'open_SplashScreen', 'datetimeutc': '2025-01-01T10:02:00'},  # Valid user with App Entry event
            # Add onboarding events to create multiple stages
            {'userid': 1, 'name': 'open_WizardScreen', 'datetimeutc': '2025-01-01T10:03:00'},
            {'userid': 2, 'name': 'open_WizardScreen', 'datetimeutc': '2025-01-01T10:04:00'},
        ]
        df = pd.DataFrame(events)
        
        fig, stage_stats = create_goal_funnel_visualization(df)
        
        # Should handle missing userids gracefully
        self.assertIsNotNone(fig, "Should handle missing userids")
        
        # Should only count valid users with App Entry events
        app_entry_stats = next((s for s in stage_stats if s['stage'] == 'App Entry'), None)
        if app_entry_stats:
            self.assertEqual(app_entry_stats['sessions'], 2, "Should count both valid users with App Entry events")
    
    def test_missing_timestamp_handling(self):
        """Test funnel behavior when timestamps are missing or invalid"""
        events = [
            {'userid': 1, 'name': 'open_SplashScreen', 'datetimeutc': '2025-01-01T10:00:00'},
            {'userid': 2, 'name': 'open_SplashScreen', 'datetimeutc': None},  # Missing timestamp
            {'userid': 3, 'name': 'open_SplashScreen', 'datetimeutc': 'invalid_date'},  # Invalid timestamp
            {'userid': 4, 'name': 'open_WizardScreen', 'datetimeutc': '2025-01-01T10:02:00'},
        ]
        df = pd.DataFrame(events)
        
        fig, stage_stats = create_goal_funnel_visualization(df)
        
        # Should handle invalid timestamps gracefully
        self.assertIsNotNone(fig, "Should handle invalid timestamps")
    
    def test_missing_event_names(self):
        """Test funnel behavior when event names are missing or null"""
        events = [
            {'userid': 1, 'name': 'open_SplashScreen', 'datetimeutc': '2025-01-01T10:00:00'},
            {'userid': 2, 'name': None, 'datetimeutc': '2025-01-01T10:01:00'},  # Missing event name
            {'userid': 3, 'name': '', 'datetimeutc': '2025-01-01T10:02:00'},  # Empty event name
            {'userid': 4, 'name': 'open_WizardScreen', 'datetimeutc': '2025-01-01T10:03:00'},
        ]
        df = pd.DataFrame(events)
        
        fig, stage_stats = create_goal_funnel_visualization(df)
        
        # Should handle missing event names gracefully
        self.assertIsNotNone(fig, "Should handle missing event names")
    
    def test_parameter_removal_simulation(self):
        """Simulate the impact of removing key parameters from events"""
        # Original complete events
        complete_events = [
            {'userid': 1, 'name': 'open_SplashScreen', 'datetimeutc': '2025-01-01T10:00:00', 'category': 'app_event', 'device_id': 'device1'},
            {'userid': 1, 'name': 'open_WizardScreen', 'datetimeutc': '2025-01-01T10:01:00', 'category': 'app_event', 'device_id': 'device1'},
            {'userid': 2, 'name': 'open_SplashScreen', 'datetimeutc': '2025-01-01T10:02:00', 'category': 'app_event', 'device_id': 'device2'},
        ]
        
        # Events after parameter removal (missing category and device_id)
        reduced_events = [
            {'userid': 1, 'name': 'open_SplashScreen', 'datetimeutc': '2025-01-01T10:00:00'},
            {'userid': 1, 'name': 'open_WizardScreen', 'datetimeutc': '2025-01-01T10:01:00'},
            {'userid': 2, 'name': 'open_SplashScreen', 'datetimeutc': '2025-01-01T10:02:00'},
        ]
        
        complete_df = pd.DataFrame(complete_events)
        reduced_df = pd.DataFrame(reduced_events)
        
        # Test both versions
        fig_complete, stats_complete = create_goal_funnel_visualization(complete_df)
        fig_reduced, stats_reduced = create_goal_funnel_visualization(reduced_df)
        
        # Both should work, but may have different results
        self.assertIsNotNone(fig_complete, "Complete data should work")
        self.assertIsNotNone(fig_reduced, "Reduced data should still work")
        
        # Compare user counts - should be the same
        if len(stats_complete) > 0 and len(stats_reduced) > 0:
            complete_app_entry = next((s for s in stats_complete if s['stage'] == 'App Entry'), None)
            reduced_app_entry = next((s for s in stats_reduced if s['stage'] == 'App Entry'), None)
            
            if complete_app_entry and reduced_app_entry:
                self.assertEqual(complete_app_entry['sessions'], reduced_app_entry['sessions'],
                               "User counts should be consistent after parameter removal")


class TestDataQualityScenarios(unittest.TestCase):
    """Test scenarios with various data quality issues"""
    
    def test_duplicate_user_sessions(self):
        """Test handling of duplicate events within the same user session"""
        events = [
            {'userid': 1, 'name': 'open_SplashScreen', 'datetimeutc': '2025-01-01T10:00:00'},
            {'userid': 1, 'name': 'open_SplashScreen', 'datetimeutc': '2025-01-01T10:00:01'},  # Duplicate
            {'userid': 1, 'name': 'open_SplashScreen', 'datetimeutc': '2025-01-01T10:00:02'},  # Another duplicate
            {'userid': 1, 'name': 'open_WizardScreen', 'datetimeutc': '2025-01-01T10:01:00'},
        ]
        df = pd.DataFrame(events)
        
        fig, stage_stats = create_goal_funnel_visualization(df)
        
        self.assertIsNotNone(fig, "Should handle duplicate events")
        
        # Should count user only once in App Entry stage
        app_entry_stats = next((s for s in stage_stats if s['stage'] == 'App Entry'), None)
        if app_entry_stats:
            self.assertEqual(app_entry_stats['sessions'], 1, "Should count user only once despite duplicates")
    
    def test_time_travel_events(self):
        """Test handling of events with timestamps in the future"""
        future_date = datetime.now() + timedelta(days=30)
        events = [
            {'userid': 1, 'name': 'open_SplashScreen', 'datetimeutc': '2025-01-01T10:00:00'},
            {'userid': 2, 'name': 'open_SplashScreen', 'datetimeutc': future_date.isoformat()},  # Future event
            {'userid': 3, 'name': 'open_WizardScreen', 'datetimeutc': '2025-01-01T10:01:00'},
        ]
        df = pd.DataFrame(events)
        
        fig, stage_stats = create_goal_funnel_visualization(df)
        
        self.assertIsNotNone(fig, "Should handle future timestamps")
    
    def test_extreme_session_lengths(self):
        """Test handling of extremely long or short sessions"""
        base_time = datetime(2025, 1, 1, 10, 0, 0)
        events = [
            # Very short session (events 1 second apart)
            {'userid': 1, 'name': 'open_SplashScreen', 'datetimeutc': base_time.isoformat()},
            {'userid': 1, 'name': 'open_WizardScreen', 'datetimeutc': (base_time + timedelta(seconds=1)).isoformat()},
            {'userid': 1, 'name': 'payment_success', 'datetimeutc': (base_time + timedelta(seconds=2)).isoformat()},
            
            # Very long session (events hours apart)
            {'userid': 2, 'name': 'open_SplashScreen', 'datetimeutc': base_time.isoformat()},
            {'userid': 2, 'name': 'open_WizardScreen', 'datetimeutc': (base_time + timedelta(hours=5)).isoformat()},
            {'userid': 2, 'name': 'payment_success', 'datetimeutc': (base_time + timedelta(hours=10)).isoformat()},
        ]
        df = pd.DataFrame(events)
        
        fig, stage_stats = create_goal_funnel_visualization(df)
        
        self.assertIsNotNone(fig, "Should handle extreme session lengths")
    
    def test_unicode_and_special_characters(self):
        """Test handling of unicode and special characters in event names"""
        events = [
            {'userid': 1, 'name': 'open_SplashScreen', 'datetimeutc': '2025-01-01T10:00:00'},
            {'userid': 2, 'name': 'événement_spécial', 'datetimeutc': '2025-01-01T10:01:00'},  # Unicode
            {'userid': 3, 'name': 'event_with_emoji_🎉', 'datetimeutc': '2025-01-01T10:02:00'},  # Emoji
            {'userid': 4, 'name': 'event-with-special!@#$%^&*()chars', 'datetimeutc': '2025-01-01T10:03:00'},  # Special chars
            # Add onboarding events to create multiple stages
            {'userid': 1, 'name': 'open_WizardScreen', 'datetimeutc': '2025-01-01T10:04:00'},
            {'userid': 2, 'name': 'open_WizardScreen', 'datetimeutc': '2025-01-01T10:05:00'},
        ]
        df = pd.DataFrame(events)
        
        fig, stage_stats = create_goal_funnel_visualization(df)
        
        self.assertIsNotNone(fig, "Should handle unicode and special characters")


class TestBusinessLogicValidation(unittest.TestCase):
    """Test business logic and funnel behavior validation"""
    
    def test_impossible_conversion_paths(self):
        """Test detection of impossible conversion paths (e.g., purchase without paywall)"""
        events = [
            {'userid': 1, 'name': 'open_SplashScreen', 'datetimeutc': '2025-01-01T10:00:00'},
            {'userid': 1, 'name': 'payment_success', 'datetimeutc': '2025-01-01T10:01:00'},  # Skip intermediate steps
            {'userid': 2, 'name': 'open_SplashScreen', 'datetimeutc': '2025-01-01T10:02:00'},
            {'userid': 2, 'name': 'open_WizardScreen', 'datetimeutc': '2025-01-01T10:03:00'},
            {'userid': 2, 'name': 'open_PaywallScreen', 'datetimeutc': '2025-01-01T10:04:00'},
            {'userid': 2, 'name': 'payment_success', 'datetimeutc': '2025-01-01T10:05:00'},
        ]
        df = pd.DataFrame(events)
        
        fig, stage_stats = create_goal_funnel_visualization(df)
        
        self.assertIsNotNone(fig, "Should handle impossible conversion paths")
        
        # Both users should be counted in their respective stages
        purchase_stats = next((s for s in stage_stats if s['stage'] == 'Purchase Completed'), None)
        if purchase_stats:
            self.assertEqual(purchase_stats['sessions'], 2, "Should count both users who made purchases")
    
    def test_multiple_purchases_same_user(self):
        """Test handling of multiple purchases by the same user"""
        events = [
            {'userid': 1, 'name': 'open_SplashScreen', 'datetimeutc': '2025-01-01T10:00:00'},
            {'userid': 1, 'name': 'payment_success', 'datetimeutc': '2025-01-01T10:01:00'},
            {'userid': 1, 'name': 'payment_success', 'datetimeutc': '2025-01-01T10:02:00'},  # Second purchase
            {'userid': 1, 'name': 'payment_success', 'datetimeutc': '2025-01-01T10:03:00'},  # Third purchase
        ]
        df = pd.DataFrame(events)
        
        fig, stage_stats = create_goal_funnel_visualization(df)
        
        self.assertIsNotNone(fig, "Should handle multiple purchases")
        
        # Should count user only once in Purchase Completed stage
        purchase_stats = next((s for s in stage_stats if s['stage'] == 'Purchase Completed'), None)
        if purchase_stats:
            self.assertEqual(purchase_stats['sessions'], 1, "Should count user only once despite multiple purchases")
    
    def test_cross_session_behavior(self):
        """Test user behavior across multiple sessions"""
        base_time = datetime(2025, 1, 1, 10, 0, 0)
        events = [
            # Session 1: User starts onboarding
            {'userid': 1, 'name': 'open_SplashScreen', 'datetimeutc': base_time.isoformat()},
            {'userid': 1, 'name': 'open_WizardScreen', 'datetimeutc': (base_time + timedelta(minutes=1)).isoformat()},
            
            # Session 2: User completes onboarding and purchases (next day)
            {'userid': 1, 'name': 'open_SplashScreen', 'datetimeutc': (base_time + timedelta(days=1)).isoformat()},
            {'userid': 1, 'name': 'onboarding_completed', 'datetimeutc': (base_time + timedelta(days=1, minutes=1)).isoformat()},
            {'userid': 1, 'name': 'open_PaywallScreen', 'datetimeutc': (base_time + timedelta(days=1, minutes=2)).isoformat()},
            {'userid': 1, 'name': 'payment_success', 'datetimeutc': (base_time + timedelta(days=1, minutes=3)).isoformat()},
        ]
        df = pd.DataFrame(events)
        
        fig, stage_stats = create_goal_funnel_visualization(df)
        
        self.assertIsNotNone(fig, "Should handle cross-session behavior")


class TestFunnelVisualizationModes(unittest.TestCase):
    """Test different funnel visualization modes and their accuracy"""
    
    def test_ga_style_vs_standard_consistency(self):
        """Test that GA-style and standard funnels produce consistent results"""
        # Create test data
        events = [
            {'userid': 1, 'name': 'open_SplashScreen', 'datetimeutc': '2025-01-01T10:00:00'},
            {'userid': 1, 'name': 'open_WizardScreen', 'datetimeutc': '2025-01-01T10:01:00'},
            {'userid': 2, 'name': 'open_SplashScreen', 'datetimeutc': '2025-01-01T10:02:00'},
            {'userid': 2, 'name': 'open_HomeScreen', 'datetimeutc': '2025-01-01T10:03:00'},
        ]
        df = pd.DataFrame(events)
        
        # Test both visualization modes
        fig_standard, stats_standard = create_goal_funnel_visualization(df)
        fig_ga, stats_ga = create_goal_funnel_ga_style(df)
        
        self.assertIsNotNone(fig_standard, "Standard funnel should work")
        self.assertIsNotNone(fig_ga, "GA-style funnel should work")
        
        # Compare user counts - should be consistent
        if len(stats_standard) > 0 and len(stats_ga) > 0:
            standard_app_entry = next((s for s in stats_standard if s['stage'] == 'App Entry'), None)
            ga_app_entry = next((s for s in stats_ga if s['stage'] == 'App Entry'), None)
            
            if standard_app_entry and ga_app_entry:
                self.assertEqual(standard_app_entry['sessions'], ga_app_entry['sessions'],
                               "User counts should be consistent between visualization modes")
    
    def test_annotation_mode_impact(self):
        """Test that different annotation modes don't affect data accuracy"""
        events = [
            {'userid': 1, 'name': 'open_SplashScreen', 'datetimeutc': '2025-01-01T10:00:00'},
            {'userid': 1, 'name': 'open_WizardScreen', 'datetimeutc': '2025-01-01T10:01:00'},
            {'userid': 2, 'name': 'open_SplashScreen', 'datetimeutc': '2025-01-01T10:02:00'},
        ]
        df = pd.DataFrame(events)
        
        # Test all annotation modes
        modes = ["Minimal", "Standard", "Detailed"]
        results = {}
        
        for mode in modes:
            fig, stats = create_goal_funnel_ga_style(df, annotation_mode=mode)
            results[mode] = stats
            self.assertIsNotNone(fig, f"Should work with {mode} annotation mode")
        
        # Compare results across modes - data should be identical
        if all(len(results[mode]) > 0 for mode in modes):
            for stage_idx in range(len(results["Standard"])):
                standard_sessions = results["Standard"][stage_idx]['sessions']
                minimal_sessions = results["Minimal"][stage_idx]['sessions']
                detailed_sessions = results["Detailed"][stage_idx]['sessions']
                
                self.assertEqual(standard_sessions, minimal_sessions,
                               f"Session counts should match between Standard and Minimal modes for stage {stage_idx}")
                self.assertEqual(standard_sessions, detailed_sessions,
                               f"Session counts should match between Standard and Detailed modes for stage {stage_idx}")


def run_validation_scenarios():
    """Run all validation scenario tests"""
    
    test_suite = unittest.TestSuite()
    
    test_classes = [
        TestParameterRemovalImpact,
        TestDataQualityScenarios,
        TestBusinessLogicValidation,
        TestFunnelVisualizationModes
    ]
    
    for test_class in test_classes:
        tests = unittest.TestLoader().loadTestsFromTestCase(test_class)
        test_suite.addTests(tests)
    
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(test_suite)
    
    return result


if __name__ == "__main__":
    run_validation_scenarios()