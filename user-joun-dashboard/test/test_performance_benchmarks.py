#!/usr/bin/env python3
"""
Performance Benchmark Tests for Analytics Algorithms
Measures execution time, memory usage, and scalability of algorithms in main.py
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import time
import psutil
import gc
from datetime import datetime, timedelta
import unittest
from typing import Dict, List, Tuple, Optional
import warnings
warnings.filterwarnings('ignore')

# Import the functions to benchmark
from main import (
    reconstruct_sessions, calculate_session_metrics, 
    perform_user_segmentation, detect_anomalies
)
from test.comprehensive_test_data_generator import AnalyticsTestDataGenerator
from test.test_session_reconstruction import TestSessionReconstruction
from test.test_user_segmentation import TestUserSegmentation
from test.test_anomaly_detection import TestAnomalyDetection
from test.test_cohort_analysis import TestCohortAnalysis
from test.test_payment_parsing import TestPaymentParsing


class PerformanceBenchmark:
    """Performance benchmark utility class"""
    
    def __init__(self):
        self.process = psutil.Process()
        self.generator = AnalyticsTestDataGenerator(seed=42)
    
    def measure_performance(self, func, *args, **kwargs) -> Dict:
        """Measure execution time and memory usage of a function"""
        # Force garbage collection before measurement
        gc.collect()
        
        # Get initial memory usage
        initial_memory = self.process.memory_info().rss / 1024 / 1024  # MB
        
        # Measure execution time
        start_time = time.time()
        result = func(*args, **kwargs)
        end_time = time.time()
        
        # Get final memory usage
        final_memory = self.process.memory_info().rss / 1024 / 1024  # MB
        
        # Force garbage collection after measurement
        gc.collect()
        
        execution_time = end_time - start_time
        memory_used = final_memory - initial_memory
        
        return {
            'result': result,
            'execution_time': execution_time,
            'memory_used': memory_used,
            'initial_memory': initial_memory,
            'final_memory': final_memory
        }
    
    def benchmark_scalability(self, func, data_sizes: List[int], data_generator_func) -> Dict:
        """Benchmark function performance across different data sizes"""
        results = {}
        
        for size in data_sizes:
            print(f"  📊 Testing with {size:,} records...")
            
            # Generate test data
            test_data = data_generator_func(size)
            
            # Measure performance
            perf_result = self.measure_performance(func, test_data)
            
            results[size] = {
                'execution_time': perf_result['execution_time'],
                'memory_used': perf_result['memory_used'],
                'records_per_second': size / perf_result['execution_time'] if perf_result['execution_time'] > 0 else 0
            }
            
            print(f"    ⏱️  Time: {perf_result['execution_time']:.3f}s")
            print(f"    💾 Memory: {perf_result['memory_used']:.2f}MB")
            print(f"    🚀 Rate: {results[size]['records_per_second']:,.0f} records/sec")
        
        return results


class TestPerformanceBenchmarks(unittest.TestCase):
    """Test cases for performance benchmarks"""
    
    def setUp(self):
        """Set up benchmark utilities"""
        self.benchmark = PerformanceBenchmark()
        self.generator = AnalyticsTestDataGenerator(seed=42)
    
    def test_session_reconstruction_performance(self):
        """Benchmark session reconstruction algorithm performance"""
        print("\n🧪 Benchmarking session reconstruction performance...")
        
        data_sizes = [1000, 5000, 10000, 25000]
        
        def generate_session_data(size):
            return self.generator.generate_user_events(
                num_users=max(10, size // 100), 
                days_range=30,
                events_per_user_per_day=(1, size // max(10, size // 100) // 30)
            )
        
        def session_reconstruction_wrapper(df):
            return reconstruct_sessions(df, timeout_minutes=30)
        
        results = self.benchmark.benchmark_scalability(
            session_reconstruction_wrapper, 
            data_sizes, 
            generate_session_data
        )
        
        # Validate performance characteristics
        execution_times = [results[size]['execution_time'] for size in data_sizes]
        
        # Should scale reasonably (not exponentially)
        time_ratios = [execution_times[i+1] / execution_times[i] for i in range(len(execution_times)-1)]
        avg_time_ratio = np.mean(time_ratios)
        
        self.assertLess(avg_time_ratio, 10, "Session reconstruction should scale reasonably")
        
        print(f"✅ Session reconstruction performance:")
        print(f"   - Average scaling ratio: {avg_time_ratio:.2f}x")
        print(f"   - Best rate: {max(results[size]['records_per_second'] for size in data_sizes):,.0f} records/sec")
        
        return results
    
    def test_user_segmentation_performance(self):
        """Benchmark user segmentation algorithm performance"""
        print("\n🧪 Benchmarking user segmentation performance...")
        
        user_counts = [50, 100, 250, 500]
        
        def generate_segmentation_data(num_users):
            return self.generator.generate_user_segmentation_data(
                num_users=num_users,
                num_groups=4
            )
        
        def segmentation_wrapper(df):
            return perform_user_segmentation(df, n_clusters=4)
        
        results = self.benchmark.benchmark_scalability(
            segmentation_wrapper,
            user_counts,
            generate_segmentation_data
        )
        
        # Validate performance
        execution_times = [results[size]['execution_time'] for size in user_counts]
        
        # K-means should be relatively fast for reasonable user counts
        max_time = max(execution_times)
        self.assertLess(max_time, 30, "User segmentation should complete within 30 seconds")
        
        print(f"✅ User segmentation performance:")
        print(f"   - Max execution time: {max_time:.3f}s")
        print(f"   - Best rate: {max(results[size]['records_per_second'] for size in user_counts):,.0f} users/sec")
        
        return results
    
    def test_anomaly_detection_performance(self):
        """Benchmark anomaly detection algorithm performance"""
        print("\n🧪 Benchmarking anomaly detection performance...")
        
        user_counts = [100, 250, 500, 1000]
        
        def generate_anomaly_data(num_users):
            base_df = self.generator.generate_user_events(num_users=num_users, days_range=30)
            return self.generator.generate_known_anomaly_users(base_df, num_anomalies=max(1, num_users // 20))
        
        def anomaly_detection_wrapper(df):
            sessions = reconstruct_sessions(df, timeout_minutes=30)
            return detect_anomalies(sessions)
        
        results = self.benchmark.benchmark_scalability(
            anomaly_detection_wrapper,
            user_counts,
            generate_anomaly_data
        )
        
        # Validate performance
        execution_times = [results[size]['execution_time'] for size in user_counts]
        
        # Isolation forest should be reasonably fast
        max_time = max(execution_times)
        self.assertLess(max_time, 60, "Anomaly detection should complete within 60 seconds")
        
        print(f"✅ Anomaly detection performance:")
        print(f"   - Max execution time: {max_time:.3f}s")
        print(f"   - Best rate: {max(results[size]['records_per_second'] for size in user_counts):,.0f} users/sec")
        
        return results
    
    def test_cohort_analysis_performance(self):
        """Benchmark cohort analysis algorithm performance"""
        print("\n🧪 Benchmarking cohort analysis performance...")
        
        data_sizes = [5000, 10000, 25000, 50000]
        
        def generate_cohort_data(size):
            return self.generator.generate_cohort_retention_data(
                num_cohorts=max(4, size // 5000),
                cohort_size=max(100, size // max(4, size // 5000)),
                retention_rates=[1.0, 0.8, 0.6, 0.4, 0.3]
            )
        
        def cohort_analysis_wrapper(df):
            # Replicate cohort analysis from test_cohort_analysis.py
            test_suite = TestCohortAnalysis()
            test_suite.setUp()
            return test_suite.perform_cohort_analysis(df)
        
        results = self.benchmark.benchmark_scalability(
            cohort_analysis_wrapper,
            data_sizes,
            generate_cohort_data
        )
        
        # Validate performance
        execution_times = [results[size]['execution_time'] for size in data_sizes]
        
        # Cohort analysis should scale well with pandas operations
        max_time = max(execution_times)
        self.assertLess(max_time, 45, "Cohort analysis should complete within 45 seconds")
        
        print(f"✅ Cohort analysis performance:")
        print(f"   - Max execution time: {max_time:.3f}s")
        print(f"   - Best rate: {max(results[size]['records_per_second'] for size in data_sizes):,.0f} records/sec")
        
        return results
    
    def test_payment_parsing_performance(self):
        """Benchmark payment parsing algorithm performance"""
        print("\n🧪 Benchmarking payment parsing performance...")
        
        transaction_counts = [500, 1000, 2500, 5000]
        
        def generate_payment_data(num_transactions):
            return self.generator.generate_payment_events(
                num_users=max(50, num_transactions // 10),
                num_transactions=num_transactions
            )
        
        def payment_parsing_wrapper(df):
            test_suite = TestPaymentParsing()
            test_suite.setUp()
            return test_suite.parse_payment_data(df)
        
        results = self.benchmark.benchmark_scalability(
            payment_parsing_wrapper,
            transaction_counts,
            generate_payment_data
        )
        
        # Validate performance
        execution_times = [results[size]['execution_time'] for size in transaction_counts]
        
        # Payment parsing should be fast (mostly JSON parsing)
        max_time = max(execution_times)
        self.assertLess(max_time, 30, "Payment parsing should complete within 30 seconds")
        
        print(f"✅ Payment parsing performance:")
        print(f"   - Max execution time: {max_time:.3f}s")
        print(f"   - Best rate: {max(results[size]['records_per_second'] for size in transaction_counts):,.0f} transactions/sec")
        
        return results
    
    def test_memory_efficiency(self):
        """Test memory efficiency of algorithms"""
        print("\n🧪 Testing memory efficiency...")
        
        # Test with moderately large dataset
        df = self.generator.generate_user_events(num_users=500, days_range=30)
        
        algorithms = [
            ('Session Reconstruction', lambda: reconstruct_sessions(df, timeout_minutes=30)),
            ('User Segmentation', lambda: perform_user_segmentation(df, n_clusters=5)),
            ('Anomaly Detection', lambda: detect_anomalies(reconstruct_sessions(df, timeout_minutes=30))),
        ]
        
        memory_results = {}
        
        for name, func in algorithms:
            perf_result = self.benchmark.measure_performance(func)
            memory_results[name] = {
                'memory_used': perf_result['memory_used'],
                'execution_time': perf_result['execution_time']
            }
            
            # Memory usage should be reasonable (< 500MB for test data)
            self.assertLess(perf_result['memory_used'], 500, 
                           f"{name} should use less than 500MB memory")
        
        print(f"✅ Memory efficiency results:")
        for name, result in memory_results.items():
            print(f"   - {name}: {result['memory_used']:.2f}MB, {result['execution_time']:.3f}s")
        
        return memory_results
    
    def test_concurrent_performance(self):
        """Test performance under concurrent operations"""
        print("\n🧪 Testing concurrent performance simulation...")
        
        # Simulate multiple operations on the same data
        df = self.generator.generate_user_events(num_users=200, days_range=30)
        
        operations = []
        
        # Simulate concurrent session reconstruction calls
        for i in range(3):
            perf_result = self.benchmark.measure_performance(
                reconstruct_sessions, df, 30
            )
            operations.append(('Session Reconstruction', perf_result))
        
        # Simulate concurrent segmentation calls
        for i in range(2):
            perf_result = self.benchmark.measure_performance(
                perform_user_segmentation, df, 4
            )
            operations.append(('User Segmentation', perf_result))
        
        # Analyze performance consistency
        session_times = [op[1]['execution_time'] for op in operations if op[0] == 'Session Reconstruction']
        segmentation_times = [op[1]['execution_time'] for op in operations if op[0] == 'User Segmentation']
        
        # Performance should be consistent (coefficient of variation < 0.5)
        if len(session_times) > 1:
            session_cv = np.std(session_times) / np.mean(session_times)
            self.assertLess(session_cv, 0.5, "Session reconstruction performance should be consistent")
        
        if len(segmentation_times) > 1:
            segmentation_cv = np.std(segmentation_times) / np.mean(segmentation_times)
            self.assertLess(segmentation_cv, 0.5, "User segmentation performance should be consistent")
        
        print(f"✅ Concurrent performance:")
        print(f"   - Session reconstruction times: {[f'{t:.3f}s' for t in session_times]}")
        print(f"   - User segmentation times: {[f'{t:.3f}s' for t in segmentation_times]}")
        
        return operations


class PerformanceBenchmarkValidator:
    """Validator class for performance benchmark results"""
    
    def __init__(self):
        self.test_results = {}
        self.benchmark_data = {}
    
    def run_comprehensive_benchmarks(self):
        """Run all performance benchmark tests"""
        print("🔍 Running comprehensive performance benchmarks...")
        
        test_suite = TestPerformanceBenchmarks()
        test_suite.setUp()
        
        # Run all benchmark tests
        tests = [
            ('session_reconstruction', test_suite.test_session_reconstruction_performance),
            ('user_segmentation', test_suite.test_user_segmentation_performance),
            ('anomaly_detection', test_suite.test_anomaly_detection_performance),
            ('cohort_analysis', test_suite.test_cohort_analysis_performance),
            ('payment_parsing', test_suite.test_payment_parsing_performance),
            ('memory_efficiency', test_suite.test_memory_efficiency),
            ('concurrent_performance', test_suite.test_concurrent_performance)
        ]
        
        for test_name, test_func in tests:
            try:
                result = test_func()
                self.test_results[test_name] = {'status': 'PASSED', 'result': result}
                self.benchmark_data[test_name] = result
                print(f"✅ {test_name}: PASSED")
            except Exception as e:
                self.test_results[test_name] = {'status': 'FAILED', 'error': str(e)}
                print(f"❌ {test_name}: FAILED - {e}")
        
        return self.test_results
    
    def generate_performance_charts(self, save_path: str = None):
        """Generate visual performance benchmark charts"""
        print("\n📊 Generating performance benchmark charts...")
        
        # Create visualization
        fig, axes = plt.subplots(2, 3, figsize=(18, 12))
        fig.suptitle('Algorithm Performance Benchmarks', fontsize=16, fontweight='bold')
        
        # 1. Execution time comparison
        if 'session_reconstruction' in self.benchmark_data:
            session_data = self.benchmark_data['session_reconstruction']
            sizes = list(session_data.keys())
            times = [session_data[size]['execution_time'] for size in sizes]
            
            axes[0, 0].plot(sizes, times, marker='o', linewidth=2, label='Session Reconstruction')
        
        if 'user_segmentation' in self.benchmark_data:
            seg_data = self.benchmark_data['user_segmentation']
            sizes = list(seg_data.keys())
            times = [seg_data[size]['execution_time'] for size in sizes]
            
            axes[0, 0].plot(sizes, times, marker='s', linewidth=2, label='User Segmentation')
        
        axes[0, 0].set_xlabel('Data Size')
        axes[0, 0].set_ylabel('Execution Time (seconds)')
        axes[0, 0].set_title('Execution Time vs Data Size')
        axes[0, 0].legend()
        axes[0, 0].grid(True, alpha=0.3)
        axes[0, 0].set_xscale('log')
        axes[0, 0].set_yscale('log')
        
        # 2. Memory usage comparison
        if 'memory_efficiency' in self.benchmark_data:
            memory_data = self.benchmark_data['memory_efficiency']
            algorithms = list(memory_data.keys())
            memory_usage = [memory_data[alg]['memory_used'] for alg in algorithms]
            
            bars = axes[0, 1].bar(algorithms, memory_usage, color=['skyblue', 'lightgreen', 'lightcoral'])
            axes[0, 1].set_ylabel('Memory Usage (MB)')
            axes[0, 1].set_title('Memory Usage by Algorithm')
            axes[0, 1].tick_params(axis='x', rotation=45)
            
            # Add value labels on bars
            for bar, value in zip(bars, memory_usage):
                axes[0, 1].text(bar.get_x() + bar.get_width()/2, bar.get_height() + 1,
                               f'{value:.1f}MB', ha='center', va='bottom')
        
        # 3. Processing rate comparison
        processing_rates = {}
        
        for test_name in ['session_reconstruction', 'user_segmentation', 'anomaly_detection']:
            if test_name in self.benchmark_data:
                data = self.benchmark_data[test_name]
                max_rate = max(data[size]['records_per_second'] for size in data.keys())
                processing_rates[test_name.replace('_', ' ').title()] = max_rate
        
        if processing_rates:
            algorithms = list(processing_rates.keys())
            rates = list(processing_rates.values())
            
            bars = axes[0, 2].bar(algorithms, rates, color=['orange', 'purple', 'brown'])
            axes[0, 2].set_ylabel('Records/Second')
            axes[0, 2].set_title('Maximum Processing Rate')
            axes[0, 2].tick_params(axis='x', rotation=45)
            axes[0, 2].set_yscale('log')
            
            # Add value labels
            for bar, value in zip(bars, rates):
                axes[0, 2].text(bar.get_x() + bar.get_width()/2, bar.get_height() * 1.1,
                               f'{value:,.0f}', ha='center', va='bottom')
        
        # 4. Scalability analysis
        if 'session_reconstruction' in self.benchmark_data:
            session_data = self.benchmark_data['session_reconstruction']
            sizes = list(session_data.keys())
            times = [session_data[size]['execution_time'] for size in sizes]
            
            # Calculate scaling factor
            scaling_factors = []
            for i in range(1, len(sizes)):
                size_ratio = sizes[i] / sizes[i-1]
                time_ratio = times[i] / times[i-1]
                scaling_factors.append(time_ratio / size_ratio)
            
            axes[1, 0].plot(sizes[1:], scaling_factors, marker='o', linewidth=2, color='red')
            axes[1, 0].axhline(y=1, color='green', linestyle='--', label='Linear Scaling')
            axes[1, 0].set_xlabel('Data Size')
            axes[1, 0].set_ylabel('Scaling Factor')
            axes[1, 0].set_title('Algorithm Scalability\n(Lower is Better)')
            axes[1, 0].legend()
            axes[1, 0].grid(True, alpha=0.3)
        
        # 5. Test results summary
        test_status = [result['status'] for result in self.test_results.values()]
        passed_count = test_status.count('PASSED')
        failed_count = test_status.count('FAILED')
        
        axes[1, 1].pie([passed_count, failed_count], labels=['Passed', 'Failed'],
                      colors=['lightgreen', 'lightcoral'], autopct='%1.1f%%')
        axes[1, 1].set_title(f'Benchmark Results\n({passed_count}/{len(test_status)} tests passed)')
        
        # 6. Performance summary
        summary_text = """
        Performance Benchmark Summary:
        
        Algorithms Tested:
        - Session Reconstruction
        - User Segmentation (K-Means)
        - Anomaly Detection (Isolation Forest)
        - Cohort Analysis
        - Payment Parsing
        
        Metrics Measured:
        - Execution Time
        - Memory Usage
        - Processing Rate
        - Scalability
        - Consistency
        
        Test Conditions:
        - Various data sizes
        - Memory efficiency
        - Concurrent operations
        """
        
        axes[1, 2].text(0.1, 0.5, summary_text, transform=axes[1, 2].transAxes,
                        fontsize=10, verticalalignment='center',
                        bbox=dict(boxstyle="round,pad=0.3", facecolor="lightblue", alpha=0.7))
        axes[1, 2].set_title('Performance Summary')
        axes[1, 2].axis('off')
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"📊 Performance charts saved to: {save_path}")
        
        plt.show()
        
        return fig


def main():
    """Run comprehensive performance benchmarks"""
    print("🚀 Starting Algorithm Performance Benchmarks")
    print("=" * 60)
    
    # Create validator
    validator = PerformanceBenchmarkValidator()
    
    # Run benchmark tests
    results = validator.run_comprehensive_benchmarks()
    
    # Generate performance charts
    chart_path = "test/performance_benchmarks.png"
    validator.generate_performance_charts(save_path=chart_path)
    
    # Print summary
    print("\n" + "=" * 60)
    print("📋 PERFORMANCE BENCHMARK SUMMARY")
    print("=" * 60)
    
    passed_tests = sum(1 for r in results.values() if r['status'] == 'PASSED')
    total_tests = len(results)
    
    print(f"✅ Benchmarks Passed: {passed_tests}/{total_tests}")
    
    if passed_tests == total_tests:
        print("🎉 All performance benchmarks PASSED!")
        print("✅ All algorithms meet performance requirements")
    else:
        print("⚠️  Some benchmarks failed. Please review the results above.")
        
        for test_name, result in results.items():
            if result['status'] == 'FAILED':
                print(f"❌ {test_name}: {result['error']}")
    
    print("\n📊 Performance visualization charts generated successfully")
    print(f"📁 Charts saved to: {chart_path}")


if __name__ == "__main__":
    main()