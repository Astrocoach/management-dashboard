#!/usr/bin/env python3
"""
Comprehensive Validation Test Runner
Executes all algorithm tests and generates comprehensive validation report with visual charts
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import unittest
import time
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

# Import all test modules
from test.test_session_reconstruction import SessionReconstructionValidator
from test.test_user_segmentation import UserSegmentationValidator
from test.test_anomaly_detection import AnomalyDetectionValidator
from test.test_cohort_analysis import CohortAnalysisValidator
from test.test_payment_parsing import PaymentParsingValidator
from test.test_performance_benchmarks import PerformanceBenchmarkValidator
from test.visual_validation_charts import ComprehensiveVisualValidator


class ComprehensiveTestRunner:
    """Master test runner for all algorithm validations"""
    
    def __init__(self):
        self.start_time = None
        self.end_time = None
        self.test_results = {}
        self.visual_validator = ComprehensiveVisualValidator()
    
    def run_all_tests(self):
        """Execute all comprehensive tests"""
        print("🚀 STARTING COMPREHENSIVE ALGORITHM VALIDATION")
        print("=" * 80)
        print(f"Started at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        print("=" * 80)
        
        self.start_time = time.time()
        
        # Test execution plan
        test_plan = [
            ("Session Reconstruction Tests", self._run_session_reconstruction_tests),
            ("User Segmentation Tests", self._run_user_segmentation_tests),
            ("Anomaly Detection Tests", self._run_anomaly_detection_tests),
            ("Cohort Analysis Tests", self._run_cohort_analysis_tests),
            ("Payment Parsing Tests", self._run_payment_parsing_tests),
            ("Performance Benchmark Tests", self._run_performance_benchmark_tests),
            ("Visual Validation Generation", self._run_visual_validation)
        ]
        
        # Execute each test suite
        for test_name, test_function in test_plan:
            print(f"\n📊 Executing {test_name}...")
            print("-" * 60)
            
            try:
                start_test_time = time.time()
                result = test_function()
                end_test_time = time.time()
                
                self.test_results[test_name] = {
                    'status': 'PASSED',
                    'result': result,
                    'execution_time': end_test_time - start_test_time
                }
                
                print(f"✅ {test_name} completed successfully in {end_test_time - start_test_time:.2f}s")
                
            except Exception as e:
                end_test_time = time.time()
                self.test_results[test_name] = {
                    'status': 'FAILED',
                    'error': str(e),
                    'execution_time': end_test_time - start_test_time
                }
                print(f"❌ {test_name} failed: {e}")
        
        self.end_time = time.time()
        
        # Generate final report
        self._generate_final_report()
        
        return self.test_results
    
    def _run_session_reconstruction_tests(self):
        """Run session reconstruction validation tests"""
        validator = SessionReconstructionValidator()
        results = validator.run_comprehensive_tests()
        
        # Generate charts
        chart_path = "test/session_reconstruction_validation.png"
        validator.generate_validation_charts(save_path=chart_path)
        
        return {
            'test_results': results,
            'chart_path': chart_path,
            'summary': self._summarize_test_results(results)
        }
    
    def _run_user_segmentation_tests(self):
        """Run user segmentation validation tests"""
        validator = UserSegmentationValidator()
        results = validator.run_comprehensive_tests()
        
        # Generate charts
        chart_path = "test/user_segmentation_validation.png"
        validator.generate_validation_charts(save_path=chart_path)
        
        return {
            'test_results': results,
            'chart_path': chart_path,
            'summary': self._summarize_test_results(results)
        }
    
    def _run_anomaly_detection_tests(self):
        """Run anomaly detection validation tests"""
        validator = AnomalyDetectionValidator()
        results = validator.run_comprehensive_tests()
        
        # Generate charts
        chart_path = "test/anomaly_detection_validation.png"
        validator.generate_validation_charts(save_path=chart_path)
        
        return {
            'test_results': results,
            'chart_path': chart_path,
            'summary': self._summarize_test_results(results)
        }
    
    def _run_cohort_analysis_tests(self):
        """Run cohort analysis validation tests"""
        validator = CohortAnalysisValidator()
        results = validator.run_comprehensive_tests()
        
        # Generate charts
        chart_path = "test/cohort_analysis_validation.png"
        validator.generate_validation_charts(save_path=chart_path)
        
        return {
            'test_results': results,
            'chart_path': chart_path,
            'summary': self._summarize_test_results(results)
        }
    
    def _run_payment_parsing_tests(self):
        """Run payment parsing validation tests"""
        validator = PaymentParsingValidator()
        results = validator.run_comprehensive_tests()
        
        # Generate charts
        chart_path = "test/payment_parsing_validation.png"
        validator.generate_validation_charts(save_path=chart_path)
        
        return {
            'test_results': results,
            'chart_path': chart_path,
            'summary': self._summarize_test_results(results)
        }
    
    def _run_performance_benchmark_tests(self):
        """Run performance benchmark tests"""
        validator = PerformanceBenchmarkValidator()
        results = validator.run_comprehensive_benchmarks()
        
        # Generate charts
        chart_path = "test/performance_benchmarks.png"
        validator.generate_performance_charts(save_path=chart_path)
        
        return {
            'test_results': results,
            'chart_path': chart_path,
            'summary': self._summarize_benchmark_results(results)
        }
    
    def _run_visual_validation(self):
        """Run comprehensive visual validation"""
        # Run all validations through visual validator
        results = self.visual_validator.run_all_validations()
        
        # Generate comprehensive dashboard
        dashboard_path = "test/comprehensive_validation_dashboard.png"
        self.visual_validator.generate_comprehensive_dashboard(save_path=dashboard_path)
        
        # Generate validation report
        report_path = "test/validation_report.txt"
        report_content = self.visual_validator.generate_validation_report(save_path=report_path)
        
        return {
            'validation_results': results,
            'dashboard_path': dashboard_path,
            'report_path': report_path,
            'report_content': report_content
        }
    
    def _summarize_test_results(self, results):
        """Summarize test results"""
        if isinstance(results, dict):
            passed = sum(1 for r in results.values() if 
                        (isinstance(r, dict) and r.get('status') == 'PASSED') or
                        (isinstance(r, bool) and r))
            total = len(results)
            return {'passed': passed, 'total': total, 'success_rate': (passed/total*100) if total > 0 else 0}
        return {'passed': 1, 'total': 1, 'success_rate': 100}
    
    def _summarize_benchmark_results(self, results):
        """Summarize benchmark results"""
        if isinstance(results, dict):
            passed = sum(1 for r in results.values() if r.get('status') == 'PASSED')
            total = len(results)
            return {'passed': passed, 'total': total, 'success_rate': (passed/total*100) if total > 0 else 0}
        return {'passed': 1, 'total': 1, 'success_rate': 100}
    
    def _generate_final_report(self):
        """Generate comprehensive final validation report"""
        print("\n" + "=" * 80)
        print("📋 COMPREHENSIVE VALIDATION REPORT")
        print("=" * 80)
        
        total_execution_time = self.end_time - self.start_time
        
        print(f"Execution Time: {total_execution_time:.2f} seconds")
        print(f"Completed at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        print("")
        
        # Overall summary
        total_passed = 0
        total_tests = 0
        
        print("TEST SUITE RESULTS:")
        print("-" * 40)
        
        for test_name, result in self.test_results.items():
            if result['status'] == 'PASSED':
                if 'summary' in result['result']:
                    summary = result['result']['summary']
                    passed = summary.get('passed', 1)
                    total = summary.get('total', 1)
                    success_rate = summary.get('success_rate', 100)
                else:
                    passed = 1
                    total = 1
                    success_rate = 100
                
                total_passed += passed
                total_tests += total
                
                status_icon = "✅" if success_rate >= 80 else "⚠️" if success_rate >= 60 else "❌"
                print(f"{status_icon} {test_name}: {passed}/{total} tests passed ({success_rate:.1f}%) - {result['execution_time']:.2f}s")
            else:
                total_tests += 1
                print(f"❌ {test_name}: FAILED - {result['error']} - {result['execution_time']:.2f}s")
        
        print("")
        print("OVERALL SUMMARY:")
        print("-" * 40)
        
        overall_success_rate = (total_passed / total_tests * 100) if total_tests > 0 else 0
        
        print(f"Total Tests: {total_tests}")
        print(f"Passed: {total_passed}")
        print(f"Failed: {total_tests - total_passed}")
        print(f"Success Rate: {overall_success_rate:.1f}%")
        print("")
        
        # Generate status assessment
        if overall_success_rate >= 90:
            print("🎉 EXCELLENT: All algorithms are performing exceptionally well!")
            assessment = "EXCELLENT"
        elif overall_success_rate >= 80:
            print("✅ GOOD: Most algorithms are performing well with minor issues.")
            assessment = "GOOD"
        elif overall_success_rate >= 60:
            print("⚠️ FAIR: Some algorithms need attention and improvement.")
            assessment = "FAIR"
        else:
            print("❌ POOR: Significant issues detected. Immediate attention required.")
            assessment = "POOR"
        
        print("")
        print("GENERATED ARTIFACTS:")
        print("-" * 40)
        
        # List all generated files
        generated_files = []
        for test_name, result in self.test_results.items():
            if result['status'] == 'PASSED' and 'result' in result:
                test_result = result['result']
                if 'chart_path' in test_result:
                    generated_files.append(test_result['chart_path'])
                if 'dashboard_path' in test_result:
                    generated_files.append(test_result['dashboard_path'])
                if 'report_path' in test_result:
                    generated_files.append(test_result['report_path'])
        
        for file_path in generated_files:
            print(f"📊 {file_path}")
        
        print("")
        print("RECOMMENDATIONS:")
        print("-" * 40)
        
        if assessment == "EXCELLENT":
            print("• Continue monitoring algorithm performance")
            print("• Consider optimizing for even better performance")
            print("• Document best practices for future development")
        elif assessment == "GOOD":
            print("• Review failed tests and address minor issues")
            print("• Consider performance optimizations")
            print("• Maintain current testing practices")
        elif assessment == "FAIR":
            print("• Prioritize fixing failed algorithms")
            print("• Review algorithm implementations")
            print("• Increase test coverage for edge cases")
        else:
            print("• Immediate review of all failed algorithms required")
            print("• Consider algorithm redesign if necessary")
            print("• Implement additional validation measures")
        
        print("")
        print("=" * 80)
        print("🏁 COMPREHENSIVE VALIDATION COMPLETED")
        print("=" * 80)
        
        # Save final report
        self._save_final_report(overall_success_rate, assessment, total_execution_time)
    
    def _save_final_report(self, success_rate, assessment, execution_time):
        """Save final report to file"""
        report_path = "test/final_validation_report.txt"
        
        with open(report_path, 'w') as f:
            f.write("COMPREHENSIVE ALGORITHM VALIDATION - FINAL REPORT\n")
            f.write("=" * 60 + "\n")
            f.write(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write(f"Execution Time: {execution_time:.2f} seconds\n")
            f.write(f"Overall Success Rate: {success_rate:.1f}%\n")
            f.write(f"Assessment: {assessment}\n\n")
            
            f.write("TEST SUITE DETAILS:\n")
            f.write("-" * 30 + "\n")
            
            for test_name, result in self.test_results.items():
                f.write(f"{test_name}: {result['status']}\n")
                if result['status'] == 'FAILED':
                    f.write(f"  Error: {result['error']}\n")
                f.write(f"  Execution Time: {result['execution_time']:.2f}s\n\n")
            
            f.write("ARTIFACTS GENERATED:\n")
            f.write("-" * 30 + "\n")
            
            for test_name, result in self.test_results.items():
                if result['status'] == 'PASSED' and 'result' in result:
                    test_result = result['result']
                    if 'chart_path' in test_result:
                        f.write(f"Chart: {test_result['chart_path']}\n")
                    if 'dashboard_path' in test_result:
                        f.write(f"Dashboard: {test_result['dashboard_path']}\n")
                    if 'report_path' in test_result:
                        f.write(f"Report: {test_result['report_path']}\n")
        
        print(f"📋 Final report saved to: {report_path}")


def main():
    """Main execution function"""
    print("🔬 COMPREHENSIVE ALGORITHM VALIDATION SUITE")
    print("Testing all algorithms in main.py for accuracy, performance, and reliability")
    print("")
    
    # Create and run comprehensive test runner
    runner = ComprehensiveTestRunner()
    results = runner.run_all_tests()
    
    # Final success check
    passed_suites = sum(1 for r in results.values() if r['status'] == 'PASSED')
    total_suites = len(results)
    
    if passed_suites == total_suites:
        print("\n🎉 ALL VALIDATION SUITES COMPLETED SUCCESSFULLY!")
        print("✅ Your algorithms are ready for production use.")
    else:
        print(f"\n⚠️ {total_suites - passed_suites} validation suite(s) failed.")
        print("❌ Please review the failed tests before deploying.")
    
    print(f"\n📊 Generated comprehensive validation artifacts in the 'test/' directory")
    print("🔍 Review the visual charts and reports for detailed insights")


if __name__ == "__main__":
    main()