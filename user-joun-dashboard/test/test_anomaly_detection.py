#!/usr/bin/env python3
"""
Comprehensive Tests for Isolation Forest Anomaly Detection Algorithm
Tests the anomaly detection with known anomalies, normal behavior patterns, and validation metrics.
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

from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score, precision_recall_curve
from sklearn.ensemble import IsolationForest

# Import the functions to test
from main import detect_anomalies, reconstruct_sessions
from test.comprehensive_test_data_generator import AnalyticsTestDataGenerator


class TestAnomalyDetection(unittest.TestCase):
    """Test cases for isolation forest anomaly detection algorithm"""
    
    def setUp(self):
        """Set up test data generator"""
        self.generator = AnalyticsTestDataGenerator(seed=42)
    
    def test_anomaly_detection_with_known_anomalies(self):
        """Test anomaly detection with data containing known anomalous users"""
        print("\n🧪 Testing anomaly detection with known anomalies...")
        
        # Generate base data
        base_df = self.generator.generate_user_events(num_users=50, days_range=30)
        
        # Add known anomalous users
        df_with_anomalies = self.generator.generate_known_anomaly_users(base_df, num_anomalies=5)
        
        # Reconstruct sessions
        sessions = reconstruct_sessions(df_with_anomalies, timeout_minutes=30)
        
        # Perform anomaly detection
        anomaly_result = detect_anomalies(sessions)
        
        # Validate structure
        self.assertIsInstance(anomaly_result, dict, "Result should be a dictionary")
        self.assertIn('anomalous_users', anomaly_result, "Should contain anomalous_users")
        self.assertIn('user_scores', anomaly_result, "Should contain user_scores")
        
        anomalous_users = anomaly_result['anomalous_users']
        user_scores = anomaly_result['user_scores']
        
        # Validate anomalous users
        self.assertIsInstance(anomalous_users, pd.DataFrame, "Anomalous users should be DataFrame")
        if len(anomalous_users) > 0:
            required_columns = ['userid', 'total_sessions', 'total_events', 'anomaly_score']
            for col in required_columns:
                self.assertIn(col, anomalous_users.columns, f"Should have column: {col}")
        
        # Validate user scores
        self.assertIsInstance(user_scores, pd.DataFrame, "User scores should be DataFrame")
        self.assertIn('userid', user_scores.columns, "Should have userid column")
        self.assertIn('anomaly_score', user_scores.columns, "Should have anomaly_score column")
        
        # Check if anomalies were detected
        num_anomalies_detected = len(anomalous_users)
        total_users = df_with_anomalies['userid'].nunique()
        
        print(f"✅ Anomaly detection completed:")
        print(f"   - Total users: {total_users}")
        print(f"   - Anomalies detected: {num_anomalies_detected}")
        print(f"   - Detection rate: {(num_anomalies_detected/total_users)*100:.1f}%")
        
        return anomaly_result
    
    def test_anomaly_detection_accuracy(self):
        """Test accuracy of anomaly detection using ground truth"""
        print("\n🧪 Testing anomaly detection accuracy...")
        
        # Generate base data (normal users)
        base_df = self.generator.generate_user_events(num_users=45, days_range=30)
        normal_user_ids = set(base_df['userid'].unique())
        
        # Add known anomalous users
        df_with_anomalies = self.generator.generate_known_anomaly_users(base_df, num_anomalies=5)
        all_user_ids = set(df_with_anomalies['userid'].unique())
        anomalous_user_ids = all_user_ids - normal_user_ids
        
        # Create ground truth labels
        ground_truth = {}
        for user_id in all_user_ids:
            ground_truth[user_id] = 1 if user_id in anomalous_user_ids else 0
        
        # Reconstruct sessions and detect anomalies
        sessions = reconstruct_sessions(df_with_anomalies, timeout_minutes=30)
        anomaly_result = detect_anomalies(sessions)
        
        user_scores = anomaly_result['user_scores']
        anomalous_users = anomaly_result['anomalous_users']
        
        # Create predictions based on detected anomalies
        predictions = {}
        for user_id in all_user_ids:
            predictions[user_id] = 1 if user_id in anomalous_users['userid'].values else 0
        
        # Calculate accuracy metrics
        y_true = [ground_truth[uid] for uid in sorted(all_user_ids)]
        y_pred = [predictions[uid] for uid in sorted(all_user_ids)]
        
        # Calculate metrics
        accuracy = sum(1 for i in range(len(y_true)) if y_true[i] == y_pred[i]) / len(y_true)
        
        true_positives = sum(1 for i in range(len(y_true)) if y_true[i] == 1 and y_pred[i] == 1)
        false_positives = sum(1 for i in range(len(y_true)) if y_true[i] == 0 and y_pred[i] == 1)
        false_negatives = sum(1 for i in range(len(y_true)) if y_true[i] == 1 and y_pred[i] == 0)
        
        precision = true_positives / (true_positives + false_positives) if (true_positives + false_positives) > 0 else 0
        recall = true_positives / (true_positives + false_negatives) if (true_positives + false_negatives) > 0 else 0
        f1_score = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
        
        # Validate performance
        self.assertGreaterEqual(accuracy, 0.6, "Accuracy should be at least 60%")
        
        print(f"✅ Anomaly detection accuracy metrics:")
        print(f"   - Accuracy: {accuracy:.3f}")
        print(f"   - Precision: {precision:.3f}")
        print(f"   - Recall: {recall:.3f}")
        print(f"   - F1-Score: {f1_score:.3f}")
        print(f"   - True Positives: {true_positives}")
        print(f"   - False Positives: {false_positives}")
        print(f"   - False Negatives: {false_negatives}")
        
        return {
            'accuracy': accuracy,
            'precision': precision,
            'recall': recall,
            'f1_score': f1_score,
            'confusion_matrix': {
                'true_positives': true_positives,
                'false_positives': false_positives,
                'false_negatives': false_negatives,
                'true_negatives': len(y_true) - true_positives - false_positives - false_negatives
            },
            'ground_truth': ground_truth,
            'predictions': predictions
        }
    
    def test_edge_case_no_anomalies(self):
        """Test anomaly detection with normal data (no anomalies)"""
        print("\n🧪 Testing no anomalies edge case...")
        
        # Generate only normal user data
        df = self.generator.generate_user_events(num_users=30, days_range=20)
        sessions = reconstruct_sessions(df, timeout_minutes=30)
        
        # Perform anomaly detection
        anomaly_result = detect_anomalies(sessions)
        anomalous_users = anomaly_result['anomalous_users']
        
        # Should detect very few or no anomalies in normal data
        anomaly_rate = len(anomalous_users) / df['userid'].nunique()
        self.assertLessEqual(anomaly_rate, 0.15, "Anomaly rate should be low for normal data (<15%)")
        
        print(f"✅ No anomalies test: {len(anomalous_users)} anomalies detected from {df['userid'].nunique()} users")
        print(f"   - Anomaly rate: {anomaly_rate*100:.1f}%")
        
        return anomaly_result
    
    def test_edge_case_all_anomalies(self):
        """Test anomaly detection with highly anomalous data"""
        print("\n🧪 Testing all anomalies edge case...")
        
        # Generate base data with few normal users
        base_df = self.generator.generate_user_events(num_users=5, days_range=30)
        
        # Add many anomalous users
        df_with_anomalies = self.generator.generate_known_anomaly_users(base_df, num_anomalies=20)
        sessions = reconstruct_sessions(df_with_anomalies, timeout_minutes=30)
        
        # Perform anomaly detection
        anomaly_result = detect_anomalies(sessions)
        anomalous_users = anomaly_result['anomalous_users']
        
        # Should detect high anomaly rate
        total_users = df_with_anomalies['userid'].nunique()
        anomaly_rate = len(anomalous_users) / total_users
        
        self.assertGreaterEqual(anomaly_rate, 0.3, "Should detect high anomaly rate when most users are anomalous")
        
        print(f"✅ All anomalies test: {len(anomalous_users)} anomalies detected from {total_users} users")
        print(f"   - Anomaly rate: {anomaly_rate*100:.1f}%")
        
        return anomaly_result
    
    def test_anomaly_score_distribution(self):
        """Test that anomaly scores are properly distributed"""
        print("\n🧪 Testing anomaly score distribution...")
        
        # Generate mixed data
        base_df = self.generator.generate_user_events(num_users=40, days_range=30)
        df_with_anomalies = self.generator.generate_known_anomaly_users(base_df, num_anomalies=10)
        
        sessions = reconstruct_sessions(df_with_anomalies, timeout_minutes=30)
        anomaly_result = detect_anomalies(sessions)
        
        user_scores = anomaly_result['user_scores']
        scores = user_scores['anomaly_score'].values
        
        # Validate score properties
        self.assertTrue(len(scores) > 0, "Should have anomaly scores")
        self.assertTrue(np.all(np.isfinite(scores)), "All scores should be finite")
        
        # Check score distribution
        score_mean = np.mean(scores)
        score_std = np.std(scores)
        score_range = np.max(scores) - np.min(scores)
        
        self.assertGreater(score_std, 0, "Scores should have variation")
        self.assertGreater(score_range, 0, "Scores should have range")
        
        print(f"✅ Anomaly score distribution:")
        print(f"   - Mean score: {score_mean:.3f}")
        print(f"   - Score std: {score_std:.3f}")
        print(f"   - Score range: {score_range:.3f}")
        print(f"   - Min score: {np.min(scores):.3f}")
        print(f"   - Max score: {np.max(scores):.3f}")
        
        return {
            'scores': scores,
            'mean': score_mean,
            'std': score_std,
            'range': score_range,
            'min': np.min(scores),
            'max': np.max(scores)
        }
    
    def test_isolation_forest_parameters(self):
        """Test isolation forest with different parameters"""
        print("\n🧪 Testing isolation forest parameters...")
        
        # Generate test data
        base_df = self.generator.generate_user_events(num_users=30, days_range=30)
        df_with_anomalies = self.generator.generate_known_anomaly_users(base_df, num_anomalies=5)
        
        sessions = reconstruct_sessions(df_with_anomalies, timeout_minutes=30)
        
        # Calculate user features for direct testing
        user_features = []
        for user_id in df_with_anomalies['userid'].unique():
            user_sessions = [s for s in sessions if s['userid'] == user_id]
            if user_sessions:
                total_sessions = len(user_sessions)
                total_events = sum(s['event_count'] for s in user_sessions)
                avg_duration = np.mean([s['duration_minutes'] for s in user_sessions])
                days_active = len(set(s['start_time'].date() for s in user_sessions))
                
                user_features.append([total_sessions, total_events, avg_duration, days_active])
        
        features_array = np.array(user_features)
        
        # Test different contamination rates
        contamination_rates = [0.1, 0.15, 0.2]
        results = {}
        
        for contamination in contamination_rates:
            iso_forest = IsolationForest(contamination=contamination, random_state=42, n_estimators=100)
            predictions = iso_forest.fit_predict(features_array)
            anomaly_count = sum(1 for p in predictions if p == -1)
            
            results[contamination] = {
                'anomaly_count': anomaly_count,
                'anomaly_rate': anomaly_count / len(predictions)
            }
        
        # Validate that different contamination rates produce different results
        anomaly_counts = [results[c]['anomaly_count'] for c in contamination_rates]
        self.assertTrue(len(set(anomaly_counts)) > 1, "Different contamination rates should produce different results")
        
        print(f"✅ Isolation forest parameter testing:")
        for contamination, result in results.items():
            print(f"   - Contamination {contamination}: {result['anomaly_count']} anomalies ({result['anomaly_rate']*100:.1f}%)")
        
        return results


class AnomalyDetectionValidator:
    """Validator class for anomaly detection results"""
    
    def __init__(self):
        self.test_results = {}
        self.validation_data = {}
    
    def run_comprehensive_validation(self):
        """Run all anomaly detection tests and generate validation report"""
        print("🔍 Running comprehensive anomaly detection validation...")
        
        test_suite = TestAnomalyDetection()
        test_suite.setUp()
        
        # Run all tests
        tests = [
            ('known_anomalies', test_suite.test_anomaly_detection_with_known_anomalies),
            ('accuracy_metrics', test_suite.test_anomaly_detection_accuracy),
            ('no_anomalies', test_suite.test_edge_case_no_anomalies),
            ('all_anomalies', test_suite.test_edge_case_all_anomalies),
            ('score_distribution', test_suite.test_anomaly_score_distribution),
            ('forest_parameters', test_suite.test_isolation_forest_parameters)
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
        """Generate visual validation charts for anomaly detection"""
        print("\n📊 Generating anomaly detection validation charts...")
        
        # Generate test data for visualization
        generator = AnalyticsTestDataGenerator(seed=42)
        base_df = generator.generate_user_events(num_users=50, days_range=30)
        df_with_anomalies = generator.generate_known_anomaly_users(base_df, num_anomalies=8)
        
        sessions = reconstruct_sessions(df_with_anomalies, timeout_minutes=30)
        anomaly_result = detect_anomalies(sessions)
        
        user_scores = anomaly_result['user_scores']
        anomalous_users = anomaly_result['anomalous_users']
        
        # Create visualization
        fig, axes = plt.subplots(2, 3, figsize=(18, 12))
        fig.suptitle('Anomaly Detection Algorithm Validation', fontsize=16, fontweight='bold')
        
        # 1. Anomaly score distribution
        scores = user_scores['anomaly_score'].values
        axes[0, 0].hist(scores, bins=20, alpha=0.7, color='skyblue', edgecolor='black')
        axes[0, 0].axvline(np.mean(scores), color='red', linestyle='--', label=f'Mean: {np.mean(scores):.3f}')
        
        # Highlight anomaly threshold if available
        if len(anomalous_users) > 0:
            threshold = anomalous_users['anomaly_score'].min()
            axes[0, 0].axvline(threshold, color='orange', linestyle='--', label=f'Threshold: {threshold:.3f}')
        
        axes[0, 0].set_title('Anomaly Score Distribution')
        axes[0, 0].set_xlabel('Anomaly Score')
        axes[0, 0].set_ylabel('Frequency')
        axes[0, 0].legend()
        axes[0, 0].grid(True, alpha=0.3)
        
        # 2. User behavior scatter plot (Sessions vs Events)
        normal_users = user_scores[~user_scores['userid'].isin(anomalous_users['userid'])]
        
        axes[0, 1].scatter(normal_users['total_sessions'], normal_users['total_events'], 
                          alpha=0.6, color='blue', label='Normal Users', s=50)
        
        if len(anomalous_users) > 0:
            axes[0, 1].scatter(anomalous_users['total_sessions'], anomalous_users['total_events'],
                              alpha=0.8, color='red', label='Anomalous Users', s=80, marker='^')
        
        axes[0, 1].set_xlabel('Total Sessions')
        axes[0, 1].set_ylabel('Total Events')
        axes[0, 1].set_title('User Behavior: Sessions vs Events')
        axes[0, 1].legend()
        axes[0, 1].grid(True, alpha=0.3)
        
        # 3. Anomaly detection results
        total_users = len(user_scores)
        anomaly_count = len(anomalous_users)
        normal_count = total_users - anomaly_count
        
        axes[0, 2].pie([normal_count, anomaly_count], 
                      labels=['Normal Users', 'Anomalous Users'],
                      colors=['lightblue', 'lightcoral'], 
                      autopct='%1.1f%%')
        axes[0, 2].set_title(f'Anomaly Detection Results\n({anomaly_count}/{total_users} anomalies)')
        
        # 4. Score vs behavior correlation
        axes[1, 0].scatter(user_scores['total_sessions'], user_scores['anomaly_score'],
                          alpha=0.6, color='purple', s=50)
        axes[1, 0].set_xlabel('Total Sessions')
        axes[1, 0].set_ylabel('Anomaly Score')
        axes[1, 0].set_title('Sessions vs Anomaly Score')
        axes[1, 0].grid(True, alpha=0.3)
        
        # 5. Test results summary
        test_status = [result['status'] for result in self.test_results.values()]
        passed_count = test_status.count('PASSED')
        failed_count = test_status.count('FAILED')
        
        axes[1, 1].pie([passed_count, failed_count], labels=['Passed', 'Failed'],
                      colors=['lightgreen', 'lightcoral'], autopct='%1.1f%%')
        axes[1, 1].set_title(f'Test Results Summary\n({passed_count}/{len(test_status)} tests passed)')
        
        # 6. Algorithm summary
        algorithm_text = f"""
        Anomaly Detection Summary:
        
        Algorithm: Isolation Forest
        Total Users Analyzed: {total_users:,}
        Anomalies Detected: {anomaly_count}
        Detection Rate: {(anomaly_count/total_users)*100:.1f}%
        
        Score Statistics:
        Mean Score: {np.mean(scores):.3f}
        Score Range: {np.max(scores) - np.min(scores):.3f}
        
        Features Used:
        - Total Sessions
        - Total Events  
        - Avg Session Duration
        - Days Active
        """
        
        # Add accuracy metrics if available
        if 'accuracy_metrics' in self.test_results and self.test_results['accuracy_metrics']['status'] == 'PASSED':
            accuracy_data = self.test_results['accuracy_metrics']['result']
            algorithm_text += f"\nAccuracy: {accuracy_data['accuracy']:.3f}"
            algorithm_text += f"\nPrecision: {accuracy_data['precision']:.3f}"
            algorithm_text += f"\nRecall: {accuracy_data['recall']:.3f}"
        
        axes[1, 2].text(0.1, 0.5, algorithm_text, transform=axes[1, 2].transAxes,
                        fontsize=10, verticalalignment='center',
                        bbox=dict(boxstyle="round,pad=0.3", facecolor="lightblue", alpha=0.7))
        axes[1, 2].set_title('Anomaly Detection Summary')
        axes[1, 2].axis('off')
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"📊 Validation charts saved to: {save_path}")
        
        plt.show()
        
        return fig


def main():
    """Run comprehensive anomaly detection validation"""
    print("🚀 Starting Anomaly Detection Algorithm Validation")
    print("=" * 60)
    
    # Create validator
    validator = AnomalyDetectionValidator()
    
    # Run validation tests
    results = validator.run_comprehensive_validation()
    
    # Generate validation charts
    chart_path = "test/anomaly_detection_validation.png"
    validator.generate_validation_charts(save_path=chart_path)
    
    # Print summary
    print("\n" + "=" * 60)
    print("📋 VALIDATION SUMMARY")
    print("=" * 60)
    
    passed_tests = sum(1 for r in results.values() if r['status'] == 'PASSED')
    total_tests = len(results)
    
    print(f"✅ Tests Passed: {passed_tests}/{total_tests}")
    
    if passed_tests == total_tests:
        print("🎉 All anomaly detection tests PASSED!")
        print("✅ Anomaly detection algorithm is working correctly")
    else:
        print("⚠️  Some tests failed. Please review the results above.")
        
        for test_name, result in results.items():
            if result['status'] == 'FAILED':
                print(f"❌ {test_name}: {result['error']}")
    
    print("\n📊 Visual validation charts generated successfully")
    print(f"📁 Charts saved to: {chart_path}")


if __name__ == "__main__":
    main()