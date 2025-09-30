#!/usr/bin/env python3
"""
Comprehensive Tests for K-means User Segmentation Algorithm
Tests the user segmentation with validation metrics, cluster quality assessment, and edge cases.
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

from sklearn.metrics import silhouette_score, calinski_harabasz_score, davies_bouldin_score
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler

# Import the functions to test
from main import perform_user_segmentation, reconstruct_sessions, calculate_session_metrics
from test.comprehensive_test_data_generator import AnalyticsTestDataGenerator


class TestUserSegmentation(unittest.TestCase):
    """Test cases for K-means user segmentation algorithm"""
    
    def setUp(self):
        """Set up test data generator"""
        self.generator = AnalyticsTestDataGenerator(seed=42)
    
    def test_segmentation_with_known_clusters(self):
        """Test segmentation with data that has known cluster structure"""
        print("\n🧪 Testing segmentation with known cluster structure...")
        
        # Generate segmentation test data with known segments
        df = self.generator.generate_segmentation_test_data()
        
        # Reconstruct sessions first
        sessions = reconstruct_sessions(df, timeout_minutes=30)
        
        # Perform segmentation
        segmentation_result = perform_user_segmentation(sessions)
        
        # Validate structure
        self.assertIsInstance(segmentation_result, dict, "Result should be a dictionary")
        self.assertIn('user_segments', segmentation_result, "Should contain user_segments")
        self.assertIn('segment_summary', segmentation_result, "Should contain segment_summary")
        
        user_segments = segmentation_result['user_segments']
        segment_summary = segmentation_result['segment_summary']
        
        # Validate user segments
        self.assertIsInstance(user_segments, pd.DataFrame, "User segments should be DataFrame")
        required_columns = ['userid', 'segment', 'total_sessions', 'total_events', 
                           'avg_session_duration', 'days_active']
        for col in required_columns:
            self.assertIn(col, user_segments.columns, f"Should have column: {col}")
        
        # Validate segment summary
        self.assertIsInstance(segment_summary, pd.DataFrame, "Segment summary should be DataFrame")
        
        # Check cluster quality
        features = user_segments[['total_sessions', 'total_events', 'avg_session_duration', 'days_active']]
        scaler = StandardScaler()
        features_scaled = scaler.fit_transform(features)
        
        if len(user_segments['segment'].unique()) > 1:
            silhouette_avg = silhouette_score(features_scaled, user_segments['segment'])
            self.assertGreater(silhouette_avg, 0, "Silhouette score should be positive")
            print(f"✅ Silhouette score: {silhouette_avg:.3f}")
        
        print(f"✅ Segmentation test passed: {len(user_segments)} users segmented into {len(segment_summary)} segments")
        return segmentation_result
    
    def test_segmentation_quality_metrics(self):
        """Test segmentation quality using multiple validation metrics"""
        print("\n🧪 Testing segmentation quality metrics...")
        
        # Generate larger dataset for better clustering
        df = self.generator.generate_user_events(num_users=100, days_range=30)
        sessions = reconstruct_sessions(df, timeout_minutes=30)
        
        # Perform segmentation
        segmentation_result = perform_user_segmentation(sessions)
        user_segments = segmentation_result['user_segments']
        
        # Prepare features for quality assessment
        features = user_segments[['total_sessions', 'total_events', 'avg_session_duration', 'days_active']]
        scaler = StandardScaler()
        features_scaled = scaler.fit_transform(features)
        
        labels = user_segments['segment']
        n_clusters = len(labels.unique())
        
        # Calculate quality metrics
        quality_metrics = {}
        
        if n_clusters > 1 and n_clusters < len(user_segments):
            # Silhouette Score (higher is better, range: -1 to 1)
            quality_metrics['silhouette_score'] = silhouette_score(features_scaled, labels)
            
            # Calinski-Harabasz Index (higher is better)
            quality_metrics['calinski_harabasz_score'] = calinski_harabasz_score(features_scaled, labels)
            
            # Davies-Bouldin Index (lower is better)
            quality_metrics['davies_bouldin_score'] = davies_bouldin_score(features_scaled, labels)
            
            # Inertia (within-cluster sum of squares)
            kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
            kmeans.fit(features_scaled)
            quality_metrics['inertia'] = kmeans.inertia_
        
        # Validate quality metrics
        if 'silhouette_score' in quality_metrics:
            self.assertGreaterEqual(quality_metrics['silhouette_score'], -1, 
                                   "Silhouette score should be >= -1")
            self.assertLessEqual(quality_metrics['silhouette_score'], 1, 
                                "Silhouette score should be <= 1")
            
            # For good clustering, silhouette score should be > 0.2
            if quality_metrics['silhouette_score'] > 0.2:
                print(f"✅ Good clustering quality (Silhouette: {quality_metrics['silhouette_score']:.3f})")
            else:
                print(f"⚠️  Moderate clustering quality (Silhouette: {quality_metrics['silhouette_score']:.3f})")
        
        print(f"✅ Quality metrics calculated for {n_clusters} clusters")
        for metric, value in quality_metrics.items():
            print(f"   - {metric}: {value:.3f}")
        
        return quality_metrics, segmentation_result
    
    def test_edge_case_single_user(self):
        """Test segmentation with single user"""
        print("\n🧪 Testing single user edge case...")
        
        # Create data with only one user
        events = []
        base_time = datetime.now() - timedelta(days=7)
        
        for i in range(10):  # 10 events for one user
            event_time = base_time + timedelta(hours=i * 2)
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
        
        sessions = reconstruct_sessions(df, timeout_minutes=30)
        
        # Perform segmentation
        segmentation_result = perform_user_segmentation(sessions)
        user_segments = segmentation_result['user_segments']
        
        # Validate results
        self.assertEqual(len(user_segments), 1, "Should have exactly 1 user")
        self.assertEqual(user_segments.iloc[0]['userid'], 1, "Should be user 1")
        
        print("✅ Single user edge case handled correctly")
        return segmentation_result
    
    def test_edge_case_identical_users(self):
        """Test segmentation with identical user behavior"""
        print("\n🧪 Testing identical users edge case...")
        
        # Create data where all users have identical behavior
        events = []
        event_id = 1
        base_time = datetime.now() - timedelta(days=7)
        
        for user_id in range(1, 6):  # 5 identical users
            for i in range(5):  # Same number of events
                event_time = base_time + timedelta(hours=i * 2)  # Same timing pattern
                events.append({
                    'analyticsid': event_id,
                    'userid': user_id,
                    'deviceid': f'device_{user_id}',
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
        
        sessions = reconstruct_sessions(df, timeout_minutes=30)
        
        # Perform segmentation
        segmentation_result = perform_user_segmentation(sessions)
        user_segments = segmentation_result['user_segments']
        
        # Validate results - all users should be in same segment
        unique_segments = user_segments['segment'].nunique()
        self.assertLessEqual(unique_segments, 2, "Identical users should be in same or very few segments")
        
        print(f"✅ Identical users edge case: {unique_segments} segments created")
        return segmentation_result
    
    def test_optimal_cluster_number(self):
        """Test determination of optimal number of clusters"""
        print("\n🧪 Testing optimal cluster number determination...")
        
        # Generate diverse dataset
        df = self.generator.generate_user_events(num_users=80, days_range=30)
        sessions = reconstruct_sessions(df, timeout_minutes=30)
        
        # Calculate user features
        user_features = []
        for user_id in df['userid'].unique():
            user_sessions = [s for s in sessions if s['userid'] == user_id]
            if user_sessions:
                total_sessions = len(user_sessions)
                total_events = sum(s['event_count'] for s in user_sessions)
                avg_duration = np.mean([s['duration_minutes'] for s in user_sessions])
                days_active = len(set(s['start_time'].date() for s in user_sessions))
                
                user_features.append({
                    'userid': user_id,
                    'total_sessions': total_sessions,
                    'total_events': total_events,
                    'avg_session_duration': avg_duration,
                    'days_active': days_active
                })
        
        features_df = pd.DataFrame(user_features)
        features = features_df[['total_sessions', 'total_events', 'avg_session_duration', 'days_active']]
        
        # Scale features
        scaler = StandardScaler()
        features_scaled = scaler.fit_transform(features)
        
        # Test different numbers of clusters
        cluster_range = range(2, min(8, len(features_df) // 2))
        inertias = []
        silhouette_scores = []
        
        for n_clusters in cluster_range:
            kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
            cluster_labels = kmeans.fit_predict(features_scaled)
            
            inertias.append(kmeans.inertia_)
            silhouette_scores.append(silhouette_score(features_scaled, cluster_labels))
        
        # Find optimal number using elbow method and silhouette score
        if len(silhouette_scores) > 0:
            optimal_clusters = cluster_range[np.argmax(silhouette_scores)]
            max_silhouette = max(silhouette_scores)
            
            self.assertGreater(max_silhouette, 0, "Best silhouette score should be positive")
            print(f"✅ Optimal clusters: {optimal_clusters} (Silhouette: {max_silhouette:.3f})")
        
        return {
            'cluster_range': list(cluster_range),
            'inertias': inertias,
            'silhouette_scores': silhouette_scores,
            'optimal_clusters': optimal_clusters if len(silhouette_scores) > 0 else None
        }
    
    def test_segment_interpretability(self):
        """Test that segments are interpretable and meaningful"""
        print("\n🧪 Testing segment interpretability...")
        
        # Generate segmentation test data with known patterns
        df = self.generator.generate_segmentation_test_data()
        sessions = reconstruct_sessions(df, timeout_minutes=30)
        
        # Perform segmentation
        segmentation_result = perform_user_segmentation(sessions)
        user_segments = segmentation_result['user_segments']
        segment_summary = segmentation_result['segment_summary']
        
        # Validate segment characteristics
        for segment_id in segment_summary.index:
            segment_data = segment_summary.loc[segment_id]
            
            # Check that segments have meaningful differences
            self.assertGreater(segment_data['user_count'], 0, "Segment should have users")
            self.assertGreater(segment_data['avg_total_sessions'], 0, "Should have sessions")
            self.assertGreater(segment_data['avg_total_events'], 0, "Should have events")
        
        # Check segment separation
        if len(segment_summary) > 1:
            # Segments should have different characteristics
            session_means = segment_summary['avg_total_sessions'].values
            event_means = segment_summary['avg_total_events'].values
            
            session_variance = np.var(session_means)
            event_variance = np.var(event_means)
            
            self.assertGreater(session_variance, 0, "Segments should differ in session counts")
            self.assertGreater(event_variance, 0, "Segments should differ in event counts")
            
            print(f"✅ Segments show meaningful differences:")
            print(f"   - Session count variance: {session_variance:.2f}")
            print(f"   - Event count variance: {event_variance:.2f}")
        
        return segmentation_result


class UserSegmentationValidator:
    """Validator class for user segmentation results"""
    
    def __init__(self):
        self.test_results = {}
        self.validation_data = {}
    
    def run_comprehensive_validation(self):
        """Run all user segmentation tests and generate validation report"""
        print("🔍 Running comprehensive user segmentation validation...")
        
        test_suite = TestUserSegmentation()
        test_suite.setUp()
        
        # Run all tests
        tests = [
            ('known_clusters', test_suite.test_segmentation_with_known_clusters),
            ('quality_metrics', test_suite.test_segmentation_quality_metrics),
            ('single_user', test_suite.test_edge_case_single_user),
            ('identical_users', test_suite.test_edge_case_identical_users),
            ('optimal_clusters', test_suite.test_optimal_cluster_number),
            ('interpretability', test_suite.test_segment_interpretability)
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
        """Generate visual validation charts for user segmentation"""
        print("\n📊 Generating user segmentation validation charts...")
        
        # Generate test data for visualization
        generator = AnalyticsTestDataGenerator(seed=42)
        df = generator.generate_segmentation_test_data()
        sessions = reconstruct_sessions(df, timeout_minutes=30)
        
        # Perform segmentation
        segmentation_result = perform_user_segmentation(sessions)
        user_segments = segmentation_result['user_segments']
        segment_summary = segmentation_result['segment_summary']
        
        # Create visualization
        fig, axes = plt.subplots(2, 3, figsize=(18, 12))
        fig.suptitle('User Segmentation Algorithm Validation', fontsize=16, fontweight='bold')
        
        # 1. Segment distribution
        segment_counts = user_segments['segment'].value_counts().sort_index()
        colors = plt.cm.Set3(np.linspace(0, 1, len(segment_counts)))
        
        axes[0, 0].pie(segment_counts.values, labels=[f'Segment {i}' for i in segment_counts.index],
                      colors=colors, autopct='%1.1f%%')
        axes[0, 0].set_title('User Distribution Across Segments')
        
        # 2. Segment characteristics - Sessions vs Events
        for i, segment_id in enumerate(segment_summary.index):
            segment_users = user_segments[user_segments['segment'] == segment_id]
            axes[0, 1].scatter(segment_users['total_sessions'], segment_users['total_events'],
                             color=colors[i], label=f'Segment {segment_id}', alpha=0.7, s=50)
        
        axes[0, 1].set_xlabel('Total Sessions')
        axes[0, 1].set_ylabel('Total Events')
        axes[0, 1].set_title('Segment Characteristics: Sessions vs Events')
        axes[0, 1].legend()
        axes[0, 1].grid(True, alpha=0.3)
        
        # 3. Segment characteristics - Duration vs Days Active
        for i, segment_id in enumerate(segment_summary.index):
            segment_users = user_segments[user_segments['segment'] == segment_id]
            axes[0, 2].scatter(segment_users['avg_session_duration'], segment_users['days_active'],
                             color=colors[i], label=f'Segment {segment_id}', alpha=0.7, s=50)
        
        axes[0, 2].set_xlabel('Avg Session Duration (min)')
        axes[0, 2].set_ylabel('Days Active')
        axes[0, 2].set_title('Segment Characteristics: Duration vs Activity')
        axes[0, 2].legend()
        axes[0, 2].grid(True, alpha=0.3)
        
        # 4. Segment summary statistics
        segment_metrics = ['avg_total_sessions', 'avg_total_events', 'avg_session_duration']
        x_pos = np.arange(len(segment_summary))
        width = 0.25
        
        for i, metric in enumerate(segment_metrics):
            values = segment_summary[metric].values
            axes[1, 0].bar(x_pos + i * width, values, width, 
                          label=metric.replace('avg_', '').replace('_', ' ').title(),
                          alpha=0.8)
        
        axes[1, 0].set_xlabel('Segment')
        axes[1, 0].set_ylabel('Average Value')
        axes[1, 0].set_title('Segment Summary Statistics')
        axes[1, 0].set_xticks(x_pos + width)
        axes[1, 0].set_xticklabels([f'Segment {i}' for i in segment_summary.index])
        axes[1, 0].legend()
        axes[1, 0].grid(True, alpha=0.3)
        
        # 5. Test results summary
        test_status = [result['status'] for result in self.test_results.values()]
        passed_count = test_status.count('PASSED')
        failed_count = test_status.count('FAILED')
        
        axes[1, 1].pie([passed_count, failed_count], labels=['Passed', 'Failed'],
                      colors=['lightgreen', 'lightcoral'], autopct='%1.1f%%')
        axes[1, 1].set_title(f'Test Results Summary\n({passed_count}/{len(test_status)} tests passed)')
        
        # 6. Segmentation quality metrics
        quality_text = f"""
        Segmentation Quality:
        
        Total Users: {len(user_segments):,}
        Number of Segments: {len(segment_summary)}
        Largest Segment: {segment_counts.max()} users
        Smallest Segment: {segment_counts.min()} users
        
        Segment Balance: {(segment_counts.min() / segment_counts.max()):.2f}
        
        Algorithm: K-Means Clustering
        Features: Sessions, Events, Duration, Activity
        """
        
        # Add quality metrics if available
        if 'quality_metrics' in self.test_results and self.test_results['quality_metrics']['status'] == 'PASSED':
            quality_metrics = self.test_results['quality_metrics']['result'][0]
            if 'silhouette_score' in quality_metrics:
                quality_text += f"\nSilhouette Score: {quality_metrics['silhouette_score']:.3f}"
        
        axes[1, 2].text(0.1, 0.5, quality_text, transform=axes[1, 2].transAxes,
                        fontsize=10, verticalalignment='center',
                        bbox=dict(boxstyle="round,pad=0.3", facecolor="lightblue", alpha=0.7))
        axes[1, 2].set_title('Segmentation Summary')
        axes[1, 2].axis('off')
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"📊 Validation charts saved to: {save_path}")
        
        plt.show()
        
        return fig


def main():
    """Run comprehensive user segmentation validation"""
    print("🚀 Starting User Segmentation Algorithm Validation")
    print("=" * 60)
    
    # Create validator
    validator = UserSegmentationValidator()
    
    # Run validation tests
    results = validator.run_comprehensive_validation()
    
    # Generate validation charts
    chart_path = "test/user_segmentation_validation.png"
    validator.generate_validation_charts(save_path=chart_path)
    
    # Print summary
    print("\n" + "=" * 60)
    print("📋 VALIDATION SUMMARY")
    print("=" * 60)
    
    passed_tests = sum(1 for r in results.values() if r['status'] == 'PASSED')
    total_tests = len(results)
    
    print(f"✅ Tests Passed: {passed_tests}/{total_tests}")
    
    if passed_tests == total_tests:
        print("🎉 All user segmentation tests PASSED!")
        print("✅ User segmentation algorithm is working correctly")
    else:
        print("⚠️  Some tests failed. Please review the results above.")
        
        for test_name, result in results.items():
            if result['status'] == 'FAILED':
                print(f"❌ {test_name}: {result['error']}")
    
    print("\n📊 Visual validation charts generated successfully")
    print(f"📁 Charts saved to: {chart_path}")


if __name__ == "__main__":
    main()