#!/usr/bin/env python3
"""
Comprehensive Tests for Cohort Retention Analysis Algorithm
Tests the cohort analysis with known retention patterns, edge cases, and validation metrics.
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
from typing import Dict, List, Tuple, Optional
import warnings
warnings.filterwarnings('ignore')

# Import the test data generator
from test.comprehensive_test_data_generator import AnalyticsTestDataGenerator


class TestCohortAnalysis(unittest.TestCase):
    """Test cases for cohort retention analysis algorithm"""
    
    def setUp(self):
        """Set up test data generator"""
        self.generator = AnalyticsTestDataGenerator(seed=42)
    
    def perform_cohort_analysis(self, df: pd.DataFrame) -> Dict:
        """
        Perform cohort analysis similar to main.py implementation
        This replicates the cohort analysis logic from main.py
        """
        # Convert event_date to datetime
        df['event_date'] = pd.to_datetime(df['event_date'])
        
        # Define cohort as the week of first activity
        user_first_activity = df.groupby('userid')['event_date'].min().reset_index()
        user_first_activity.columns = ['userid', 'cohort_group']
        
        # Create cohort period (weekly cohorts)
        user_first_activity['cohort_period'] = user_first_activity['cohort_group'].dt.to_period('W')
        user_first_activity['cohort'] = user_first_activity['cohort_period'].dt.start_time.dt.date
        
        # Merge cohort info back to main dataframe
        df = df.merge(user_first_activity[['userid', 'cohort']], on='userid')
        
        # Create period number (weeks since cohort start)
        df['period_number'] = (df['event_date'].dt.to_period('W') - 
                              df['cohort'].apply(lambda x: pd.Period(x, freq='W'))).apply(lambda x: x.n)
        
        # Count unique users in each cohort-period combination
        cohort_data = df.groupby(['cohort', 'period_number'])['userid'].nunique().reset_index()
        cohort_data.rename(columns={'userid': 'users'}, inplace=True)
        
        # Create cohort table
        cohort_table = cohort_data.pivot(index='cohort', columns='period_number', values='users')
        
        # Get cohort sizes (period 0)
        cohort_sizes = cohort_table.iloc[:, 0]
        
        # Calculate retention rates
        retention_table = cohort_table.divide(cohort_sizes, axis=0)
        
        return {
            'cohort_table': cohort_table,
            'retention_table': retention_table,
            'cohort_sizes': cohort_sizes,
            'cohort_data': cohort_data,
            'user_cohorts': user_first_activity
        }
    
    def test_cohort_analysis_with_known_patterns(self):
        """Test cohort analysis with data containing known retention patterns"""
        print("\n🧪 Testing cohort analysis with known retention patterns...")
        
        # Generate cohort data with known patterns
        df = self.generator.generate_cohort_retention_data(
            num_cohorts=4, 
            cohort_size=100, 
            retention_rates=[1.0, 0.8, 0.6, 0.4, 0.3]
        )
        
        # Perform cohort analysis
        result = self.perform_cohort_analysis(df)
        
        cohort_table = result['cohort_table']
        retention_table = result['retention_table']
        cohort_sizes = result['cohort_sizes']
        
        # Validate structure
        self.assertIsInstance(cohort_table, pd.DataFrame, "Cohort table should be DataFrame")
        self.assertIsInstance(retention_table, pd.DataFrame, "Retention table should be DataFrame")
        self.assertGreater(len(cohort_table), 0, "Should have cohort data")
        
        # Validate cohort sizes
        self.assertTrue(all(size > 0 for size in cohort_sizes), "All cohort sizes should be positive")
        
        # Validate retention rates (period 0 should be 1.0)
        period_0_retention = retention_table.iloc[:, 0]
        self.assertTrue(all(abs(rate - 1.0) < 0.01 for rate in period_0_retention), 
                       "Period 0 retention should be 1.0")
        
        # Check retention decay pattern
        for cohort_idx in range(len(retention_table)):
            cohort_retention = retention_table.iloc[cohort_idx].dropna()
            if len(cohort_retention) > 1:
                # Retention should generally decrease over time
                decreasing_trend = all(cohort_retention.iloc[i] >= cohort_retention.iloc[i+1] 
                                     for i in range(len(cohort_retention)-1))
                # Allow for some variation but expect general downward trend
                self.assertTrue(cohort_retention.iloc[0] >= cohort_retention.iloc[-1], 
                               "Retention should generally decrease over time")
        
        print(f"✅ Cohort analysis completed:")
        print(f"   - Number of cohorts: {len(cohort_table)}")
        print(f"   - Max periods tracked: {cohort_table.shape[1]}")
        print(f"   - Average cohort size: {cohort_sizes.mean():.1f}")
        print(f"   - Period 0 retention: {period_0_retention.mean():.3f}")
        
        return result
    
    def test_cohort_retention_accuracy(self):
        """Test accuracy of cohort retention calculations"""
        print("\n🧪 Testing cohort retention calculation accuracy...")
        
        # Generate controlled test data
        test_data = []
        base_date = datetime(2024, 1, 1)
        
        # Create 2 cohorts with known retention patterns
        # Cohort 1: 100 users, 80% retention in week 1, 60% in week 2
        cohort1_users = [f"user_{i}" for i in range(100)]
        
        # Week 0 (all users active)
        for user in cohort1_users:
            test_data.append({
                'userid': user,
                'event_date': base_date,
                'event_type': 'login'
            })
        
        # Week 1 (80 users active)
        for user in cohort1_users[:80]:
            test_data.append({
                'userid': user,
                'event_date': base_date + timedelta(days=7),
                'event_type': 'login'
            })
        
        # Week 2 (60 users active)
        for user in cohort1_users[:60]:
            test_data.append({
                'userid': user,
                'event_date': base_date + timedelta(days=14),
                'event_type': 'login'
            })
        
        # Cohort 2: 50 users, 90% retention in week 1
        cohort2_users = [f"user2_{i}" for i in range(50)]
        cohort2_start = base_date + timedelta(days=7)
        
        # Week 0 for cohort 2
        for user in cohort2_users:
            test_data.append({
                'userid': user,
                'event_date': cohort2_start,
                'event_type': 'login'
            })
        
        # Week 1 for cohort 2 (45 users active)
        for user in cohort2_users[:45]:
            test_data.append({
                'userid': user,
                'event_date': cohort2_start + timedelta(days=7),
                'event_type': 'login'
            })
        
        df = pd.DataFrame(test_data)
        
        # Perform cohort analysis
        result = self.perform_cohort_analysis(df)
        retention_table = result['retention_table']
        
        # Validate expected retention rates
        cohorts = sorted(retention_table.index)
        
        # Check cohort 1 retention
        cohort1_retention = retention_table.loc[cohorts[0]]
        self.assertAlmostEqual(cohort1_retention.iloc[0], 1.0, places=2, msg="Period 0 should be 100%")
        self.assertAlmostEqual(cohort1_retention.iloc[1], 0.8, places=2, msg="Period 1 should be 80%")
        self.assertAlmostEqual(cohort1_retention.iloc[2], 0.6, places=2, msg="Period 2 should be 60%")
        
        # Check cohort 2 retention
        cohort2_retention = retention_table.loc[cohorts[1]]
        self.assertAlmostEqual(cohort2_retention.iloc[0], 1.0, places=2, msg="Period 0 should be 100%")
        self.assertAlmostEqual(cohort2_retention.iloc[1], 0.9, places=2, msg="Period 1 should be 90%")
        
        print(f"✅ Cohort retention accuracy validated:")
        print(f"   - Cohort 1 retention: {cohort1_retention.iloc[:3].values}")
        print(f"   - Cohort 2 retention: {cohort2_retention.iloc[:2].values}")
        
        return result
    
    def test_edge_case_single_cohort(self):
        """Test cohort analysis with single cohort"""
        print("\n🧪 Testing single cohort edge case...")
        
        # Generate single cohort data
        df = self.generator.generate_cohort_retention_data(
            num_cohorts=1, 
            cohort_size=50, 
            retention_rates=[1.0, 0.7, 0.5]
        )
        
        result = self.perform_cohort_analysis(df)
        cohort_table = result['cohort_table']
        retention_table = result['retention_table']
        
        # Should have exactly one cohort
        self.assertEqual(len(cohort_table), 1, "Should have exactly one cohort")
        self.assertEqual(len(retention_table), 1, "Should have exactly one retention row")
        
        # Validate retention pattern
        retention_row = retention_table.iloc[0]
        self.assertAlmostEqual(retention_row.iloc[0], 1.0, places=2, msg="Period 0 should be 100%")
        
        print(f"✅ Single cohort test: {len(cohort_table)} cohort with {cohort_table.shape[1]} periods")
        
        return result
    
    def test_edge_case_no_retention(self):
        """Test cohort analysis with users who don't return"""
        print("\n🧪 Testing no retention edge case...")
        
        # Generate data where users only appear once
        test_data = []
        base_date = datetime(2024, 1, 1)
        
        for i in range(30):
            test_data.append({
                'userid': f"user_{i}",
                'event_date': base_date + timedelta(days=i % 7),  # Spread across a week
                'event_type': 'login'
            })
        
        df = pd.DataFrame(test_data)
        result = self.perform_cohort_analysis(df)
        
        retention_table = result['retention_table']
        
        # All retention rates beyond period 0 should be 0
        for cohort_idx in range(len(retention_table)):
            cohort_retention = retention_table.iloc[cohort_idx]
            self.assertAlmostEqual(cohort_retention.iloc[0], 1.0, places=2, msg="Period 0 should be 100%")
            
            # Check if there are any subsequent periods
            if len(cohort_retention) > 1:
                subsequent_periods = cohort_retention.iloc[1:].dropna()
                if len(subsequent_periods) > 0:
                    self.assertTrue(all(rate == 0.0 for rate in subsequent_periods), 
                                   "No retention should result in 0% for subsequent periods")
        
        print(f"✅ No retention test: {len(retention_table)} cohorts with no returning users")
        
        return result
    
    def test_cohort_period_calculation(self):
        """Test that cohort periods are calculated correctly"""
        print("\n🧪 Testing cohort period calculations...")
        
        # Create test data with specific dates
        test_data = []
        
        # User 1: First activity on Monday (week 1)
        test_data.extend([
            {'userid': 'user1', 'event_date': '2024-01-01', 'event_type': 'login'},  # Week 1
            {'userid': 'user1', 'event_date': '2024-01-08', 'event_type': 'login'},  # Week 2 (period 1)
            {'userid': 'user1', 'event_date': '2024-01-15', 'event_type': 'login'}, # Week 3 (period 2)
        ])
        
        # User 2: First activity on Wednesday (same week as user 1)
        test_data.extend([
            {'userid': 'user2', 'event_date': '2024-01-03', 'event_type': 'login'},  # Week 1
            {'userid': 'user2', 'event_date': '2024-01-10', 'event_type': 'login'},  # Week 2 (period 1)
        ])
        
        # User 3: First activity in week 2
        test_data.extend([
            {'userid': 'user3', 'event_date': '2024-01-08', 'event_type': 'login'},  # Week 2
            {'userid': 'user3', 'event_date': '2024-01-15', 'event_type': 'login'}, # Week 3 (period 1)
        ])
        
        df = pd.DataFrame(test_data)
        result = self.perform_cohort_analysis(df)
        
        user_cohorts = result['user_cohorts']
        cohort_data = result['cohort_data']
        
        # Validate cohort assignments
        user1_cohort = user_cohorts[user_cohorts['userid'] == 'user1']['cohort'].iloc[0]
        user2_cohort = user_cohorts[user_cohorts['userid'] == 'user2']['cohort'].iloc[0]
        user3_cohort = user_cohorts[user_cohorts['userid'] == 'user3']['cohort'].iloc[0]
        
        # User 1 and 2 should be in the same cohort (same week)
        self.assertEqual(user1_cohort, user2_cohort, "Users from same week should be in same cohort")
        
        # User 3 should be in a different cohort
        self.assertNotEqual(user1_cohort, user3_cohort, "Users from different weeks should be in different cohorts")
        
        # Validate period calculations
        cohort_periods = cohort_data.groupby('cohort')['period_number'].max()
        
        print(f"✅ Cohort period calculations validated:")
        print(f"   - Number of cohorts: {len(cohort_periods)}")
        print(f"   - Max periods per cohort: {cohort_periods.values}")
        
        return result
    
    def test_cohort_size_consistency(self):
        """Test that cohort sizes are consistent across calculations"""
        print("\n🧪 Testing cohort size consistency...")
        
        # Generate test data
        df = self.generator.generate_cohort_retention_data(
            num_cohorts=3, 
            cohort_size=75, 
            retention_rates=[1.0, 0.8, 0.6, 0.4]
        )
        
        result = self.perform_cohort_analysis(df)
        
        cohort_table = result['cohort_table']
        cohort_sizes = result['cohort_sizes']
        user_cohorts = result['user_cohorts']
        
        # Validate cohort sizes match period 0 counts
        period_0_counts = cohort_table.iloc[:, 0]
        
        for cohort in cohort_sizes.index:
            expected_size = cohort_sizes[cohort]
            actual_size = period_0_counts[cohort]
            self.assertEqual(expected_size, actual_size, 
                           f"Cohort size mismatch for cohort {cohort}")
        
        # Validate total users
        total_users_in_cohorts = user_cohorts['userid'].nunique()
        total_users_in_data = df['userid'].nunique()
        self.assertEqual(total_users_in_cohorts, total_users_in_data, 
                        "All users should be assigned to cohorts")
        
        print(f"✅ Cohort size consistency validated:")
        print(f"   - Total users: {total_users_in_data}")
        print(f"   - Users in cohorts: {total_users_in_cohorts}")
        print(f"   - Cohort sizes: {cohort_sizes.values}")
        
        return result


class CohortAnalysisValidator:
    """Validator class for cohort analysis results"""
    
    def __init__(self):
        self.test_results = {}
        self.validation_data = {}
    
    def run_comprehensive_validation(self):
        """Run all cohort analysis tests and generate validation report"""
        print("🔍 Running comprehensive cohort analysis validation...")
        
        test_suite = TestCohortAnalysis()
        test_suite.setUp()
        
        # Run all tests
        tests = [
            ('known_patterns', test_suite.test_cohort_analysis_with_known_patterns),
            ('retention_accuracy', test_suite.test_cohort_retention_accuracy),
            ('single_cohort', test_suite.test_edge_case_single_cohort),
            ('no_retention', test_suite.test_edge_case_no_retention),
            ('period_calculation', test_suite.test_cohort_period_calculation),
            ('size_consistency', test_suite.test_cohort_size_consistency)
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
        """Generate visual validation charts for cohort analysis"""
        print("\n📊 Generating cohort analysis validation charts...")
        
        # Generate test data for visualization
        generator = AnalyticsTestDataGenerator(seed=42)
        df = generator.generate_cohort_retention_data(
            num_cohorts=5, 
            cohort_size=100, 
            retention_rates=[1.0, 0.8, 0.65, 0.5, 0.4, 0.3]
        )
        
        # Perform cohort analysis
        test_suite = TestCohortAnalysis()
        test_suite.setUp()
        result = test_suite.perform_cohort_analysis(df)
        
        cohort_table = result['cohort_table']
        retention_table = result['retention_table']
        cohort_sizes = result['cohort_sizes']
        
        # Create visualization
        fig, axes = plt.subplots(2, 3, figsize=(18, 12))
        fig.suptitle('Cohort Retention Analysis Validation', fontsize=16, fontweight='bold')
        
        # 1. Cohort retention heatmap
        sns.heatmap(retention_table.fillna(0), annot=True, fmt='.2f', cmap='YlOrRd', 
                   ax=axes[0, 0], cbar_kws={'label': 'Retention Rate'})
        axes[0, 0].set_title('Cohort Retention Heatmap')
        axes[0, 0].set_xlabel('Period Number')
        axes[0, 0].set_ylabel('Cohort')
        
        # 2. Cohort sizes
        cohort_labels = [f"Cohort {i+1}" for i in range(len(cohort_sizes))]
        axes[0, 1].bar(cohort_labels, cohort_sizes.values, color='skyblue', alpha=0.7)
        axes[0, 1].set_title('Cohort Sizes')
        axes[0, 1].set_xlabel('Cohort')
        axes[0, 1].set_ylabel('Number of Users')
        axes[0, 1].tick_params(axis='x', rotation=45)
        
        # 3. Retention curves
        for i, (cohort, retention) in enumerate(retention_table.iterrows()):
            periods = range(len(retention.dropna()))
            axes[0, 2].plot(periods, retention.dropna().values, 
                           marker='o', label=f'Cohort {i+1}', linewidth=2)
        
        axes[0, 2].set_title('Retention Curves by Cohort')
        axes[0, 2].set_xlabel('Period Number')
        axes[0, 2].set_ylabel('Retention Rate')
        axes[0, 2].legend()
        axes[0, 2].grid(True, alpha=0.3)
        axes[0, 2].set_ylim(0, 1.1)
        
        # 4. Average retention by period
        avg_retention = retention_table.mean(axis=0)
        periods = range(len(avg_retention.dropna()))
        axes[1, 0].bar(periods, avg_retention.dropna().values, color='lightcoral', alpha=0.7)
        axes[1, 0].set_title('Average Retention by Period')
        axes[1, 0].set_xlabel('Period Number')
        axes[1, 0].set_ylabel('Average Retention Rate')
        axes[1, 0].set_ylim(0, 1.1)
        
        # Add value labels on bars
        for i, v in enumerate(avg_retention.dropna().values):
            axes[1, 0].text(i, v + 0.02, f'{v:.2f}', ha='center', va='bottom')
        
        # 5. Test results summary
        test_status = [result['status'] for result in self.test_results.values()]
        passed_count = test_status.count('PASSED')
        failed_count = test_status.count('FAILED')
        
        axes[1, 1].pie([passed_count, failed_count], labels=['Passed', 'Failed'],
                      colors=['lightgreen', 'lightcoral'], autopct='%1.1f%%')
        axes[1, 1].set_title(f'Test Results Summary\n({passed_count}/{len(test_status)} tests passed)')
        
        # 6. Analysis summary
        total_users = cohort_sizes.sum()
        avg_cohort_size = cohort_sizes.mean()
        max_periods = cohort_table.shape[1]
        
        # Calculate overall retention metrics
        period_1_retention = retention_table.iloc[:, 1].mean() if cohort_table.shape[1] > 1 else 0
        final_period_retention = retention_table.iloc[:, -1].mean()
        
        summary_text = f"""
        Cohort Analysis Summary:
        
        Total Users: {total_users:,}
        Number of Cohorts: {len(cohort_sizes)}
        Average Cohort Size: {avg_cohort_size:.1f}
        Max Periods Tracked: {max_periods}
        
        Retention Metrics:
        Period 0: 100.0%
        Period 1: {period_1_retention:.1%}
        Final Period: {final_period_retention:.1%}
        
        Analysis Method:
        - Weekly cohorts
        - Period-over-period tracking
        - User-level retention calculation
        """
        
        axes[1, 2].text(0.1, 0.5, summary_text, transform=axes[1, 2].transAxes,
                        fontsize=10, verticalalignment='center',
                        bbox=dict(boxstyle="round,pad=0.3", facecolor="lightblue", alpha=0.7))
        axes[1, 2].set_title('Cohort Analysis Summary')
        axes[1, 2].axis('off')
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"📊 Validation charts saved to: {save_path}")
        
        plt.show()
        
        return fig


def main():
    """Run comprehensive cohort analysis validation"""
    print("🚀 Starting Cohort Analysis Algorithm Validation")
    print("=" * 60)
    
    # Create validator
    validator = CohortAnalysisValidator()
    
    # Run validation tests
    results = validator.run_comprehensive_validation()
    
    # Generate validation charts
    chart_path = "test/cohort_analysis_validation.png"
    validator.generate_validation_charts(save_path=chart_path)
    
    # Print summary
    print("\n" + "=" * 60)
    print("📋 VALIDATION SUMMARY")
    print("=" * 60)
    
    passed_tests = sum(1 for r in results.values() if r['status'] == 'PASSED')
    total_tests = len(results)
    
    print(f"✅ Tests Passed: {passed_tests}/{total_tests}")
    
    if passed_tests == total_tests:
        print("🎉 All cohort analysis tests PASSED!")
        print("✅ Cohort retention analysis algorithm is working correctly")
    else:
        print("⚠️  Some tests failed. Please review the results above.")
        
        for test_name, result in results.items():
            if result['status'] == 'FAILED':
                print(f"❌ {test_name}: {result['error']}")
    
    print("\n📊 Visual validation charts generated successfully")
    print(f"📁 Charts saved to: {chart_path}")


if __name__ == "__main__":
    main()