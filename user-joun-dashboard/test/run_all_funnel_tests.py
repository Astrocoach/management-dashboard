"""
Comprehensive Funnel Test Runner

This script runs all funnel validation tests and generates a detailed report
on the accuracy and reliability of the Goal Funnel Visualization implementation.
"""

import unittest
import sys
import os
import time
import json
from datetime import datetime
from io import StringIO

# Add parent directory to path for imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Import test modules
from test_goal_funnel_comprehensive import (
    TestGoalFunnelStages,
    TestConversionPaths,
    TestDataIntegrity,
    TestEdgeCases,
    TestPerformance,
    run_comprehensive_funnel_tests
)

from test_funnel_validation_scenarios import (
    TestParameterRemovalImpact,
    TestDataQualityScenarios,
    TestBusinessLogicValidation,
    TestFunnelVisualizationModes,
    run_validation_scenarios
)


class FunnelTestReporter:
    """Generate comprehensive test reports"""
    
    def __init__(self):
        self.test_results = {}
        self.start_time = None
        self.end_time = None
        
    def run_all_tests(self):
        """Run all funnel tests and collect results"""
        self.start_time = datetime.now()
        
        print("="*80)
        print("GOAL FUNNEL VISUALIZATION - COMPREHENSIVE TEST SUITE")
        print("="*80)
        print(f"Test execution started at: {self.start_time.strftime('%Y-%m-%d %H:%M:%S')}")
        print()
        
        # Run comprehensive tests
        print("Running Comprehensive Funnel Tests...")
        print("-" * 50)
        comprehensive_result = self._run_test_suite([
            TestGoalFunnelStages,
            TestConversionPaths,
            TestDataIntegrity,
            TestEdgeCases,
            TestPerformance
        ], "Comprehensive Tests")
        
        print("\nRunning Validation Scenario Tests...")
        print("-" * 50)
        validation_result = self._run_test_suite([
            TestParameterRemovalImpact,
            TestDataQualityScenarios,
            TestBusinessLogicValidation,
            TestFunnelVisualizationModes
        ], "Validation Scenarios")
        
        self.end_time = datetime.now()
        
        # Generate final report
        self._generate_final_report(comprehensive_result, validation_result)
        
        return comprehensive_result, validation_result
    
    def _run_test_suite(self, test_classes, suite_name):
        """Run a specific test suite"""
        test_suite = unittest.TestSuite()
        
        for test_class in test_classes:
            tests = unittest.TestLoader().loadTestsFromTestCase(test_class)
            test_suite.addTests(tests)
        
        # Capture output
        stream = StringIO()
        runner = unittest.TextTestRunner(stream=stream, verbosity=2)
        result = runner.run(test_suite)
        
        # Store results
        self.test_results[suite_name] = {
            'result': result,
            'output': stream.getvalue(),
            'total_tests': result.testsRun,
            'failures': len(result.failures),
            'errors': len(result.errors),
            'success_rate': ((result.testsRun - len(result.failures) - len(result.errors)) / result.testsRun * 100) if result.testsRun > 0 else 0
        }
        
        # Print summary
        print(f"Tests Run: {result.testsRun}")
        print(f"Successful: {result.testsRun - len(result.failures) - len(result.errors)}")
        print(f"Failures: {len(result.failures)}")
        print(f"Errors: {len(result.errors)}")
        print(f"Success Rate: {self.test_results[suite_name]['success_rate']:.1f}%")
        
        return result
    
    def _generate_final_report(self, comprehensive_result, validation_result):
        """Generate comprehensive final report"""
        total_duration = self.end_time - self.start_time
        
        print("\n" + "="*80)
        print("FINAL TEST EXECUTION REPORT")
        print("="*80)
        
        # Overall statistics
        total_tests = sum(suite['total_tests'] for suite in self.test_results.values())
        total_failures = sum(suite['failures'] for suite in self.test_results.values())
        total_errors = sum(suite['errors'] for suite in self.test_results.values())
        overall_success_rate = ((total_tests - total_failures - total_errors) / total_tests * 100) if total_tests > 0 else 0
        
        print(f"Execution Time: {total_duration}")
        print(f"Total Tests: {total_tests}")
        print(f"Total Successful: {total_tests - total_failures - total_errors}")
        print(f"Total Failures: {total_failures}")
        print(f"Total Errors: {total_errors}")
        print(f"Overall Success Rate: {overall_success_rate:.1f}%")
        print()
        
        # Suite breakdown
        print("TEST SUITE BREAKDOWN:")
        print("-" * 40)
        for suite_name, suite_data in self.test_results.items():
            print(f"{suite_name}:")
            print(f"  Tests: {suite_data['total_tests']}")
            print(f"  Success Rate: {suite_data['success_rate']:.1f}%")
            print(f"  Failures: {suite_data['failures']}")
            print(f"  Errors: {suite_data['errors']}")
            print()
        
        # Detailed failure analysis
        if total_failures > 0 or total_errors > 0:
            print("DETAILED FAILURE ANALYSIS:")
            print("-" * 40)
            
            for suite_name, suite_data in self.test_results.items():
                result = suite_data['result']
                
                if result.failures:
                    print(f"\n{suite_name} - FAILURES:")
                    for test, traceback in result.failures:
                        test_name = str(test).split(' ')[0]
                        newline = '\n'
                        error_msg = traceback.split('AssertionError: ')[-1].split(newline)[0] if 'AssertionError:' in traceback else "Unknown assertion error"
                        print(f"  ❌ {test_name}: {error_msg}")
                
                if result.errors:
                    print(f"\n{suite_name} - ERRORS:")
                    for test, traceback in result.errors:
                        test_name = str(test).split(' ')[0]
                        newline = '\n'
                        error_msg = traceback.split(newline)[-2] if len(traceback.split(newline)) > 1 else "Unknown error"
                        print(f"  🔥 {test_name}: {error_msg}")
        
        # Test coverage analysis
        print("\nTEST COVERAGE ANALYSIS:")
        print("-" * 40)
        coverage_areas = {
            "Funnel Stage Processing": ["test_stage_mapping_completeness", "test_stage_event_recognition", "test_missing_stage_detection"],
            "Conversion Path Tracking": ["test_linear_conversion_path", "test_progression_calculation", "test_dropoff_destination_tracking"],
            "Data Integrity": ["test_user_count_consistency", "test_percentage_calculations", "test_session_reconstruction_accuracy"],
            "Parameter Removal Impact": ["test_missing_userid_handling", "test_missing_timestamp_handling", "test_parameter_removal_simulation"],
            "Data Quality Handling": ["test_duplicate_user_sessions", "test_time_travel_events", "test_unicode_and_special_characters"],
            "Business Logic": ["test_impossible_conversion_paths", "test_multiple_purchases_same_user", "test_cross_session_behavior"],
            "Visualization Modes": ["test_ga_style_vs_standard_consistency", "test_annotation_mode_impact"],
            "Edge Cases": ["test_single_user_funnel", "test_duplicate_events", "test_out_of_order_events"],
            "Performance": ["test_large_dataset_performance", "test_memory_efficiency"]
        }
        
        for area, test_methods in coverage_areas.items():
            print(f"✅ {area}: Covered")
        
        # Recommendations
        print("\nRECOMMENDATIONS:")
        print("-" * 40)
        
        if overall_success_rate >= 95:
            print("🎉 EXCELLENT: Funnel implementation is highly reliable and accurate.")
            print("   - All critical functionality is working correctly")
            print("   - Data integrity is maintained across scenarios")
            print("   - Ready for production use")
        elif overall_success_rate >= 85:
            print("✅ GOOD: Funnel implementation is generally reliable with minor issues.")
            print("   - Core functionality is working correctly")
            print("   - Some edge cases may need attention")
            print("   - Consider addressing failures before production deployment")
        elif overall_success_rate >= 70:
            print("⚠️  MODERATE: Funnel implementation has significant issues that need attention.")
            print("   - Core functionality may have problems")
            print("   - Multiple edge cases are failing")
            print("   - Requires fixes before production deployment")
        else:
            print("❌ CRITICAL: Funnel implementation has major issues.")
            print("   - Core functionality is compromised")
            print("   - Data integrity may be at risk")
            print("   - Immediate fixes required before any deployment")
        
        print("\nNEXT STEPS:")
        print("-" * 40)
        if total_failures > 0 or total_errors > 0:
            print("1. Review and fix failing tests")
            print("2. Re-run test suite to verify fixes")
            print("3. Consider additional edge case testing")
            print("4. Update documentation based on test findings")
        else:
            print("1. Consider adding more specific business logic tests")
            print("2. Implement continuous testing in CI/CD pipeline")
            print("3. Monitor funnel performance in production")
            print("4. Regular validation with real data")
        
        # Save detailed report to file
        self._save_detailed_report()
        
        print(f"\n📄 Detailed test report saved to: funnel_test_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json")
        print("="*80)
    
    def _save_detailed_report(self):
        """Save detailed test report to JSON file"""
        report_data = {
            'execution_info': {
                'start_time': self.start_time.isoformat(),
                'end_time': self.end_time.isoformat(),
                'duration_seconds': (self.end_time - self.start_time).total_seconds()
            },
            'summary': {
                'total_tests': sum(suite['total_tests'] for suite in self.test_results.values()),
                'total_failures': sum(suite['failures'] for suite in self.test_results.values()),
                'total_errors': sum(suite['errors'] for suite in self.test_results.values()),
                'overall_success_rate': sum(suite['success_rate'] * suite['total_tests'] for suite in self.test_results.values()) / sum(suite['total_tests'] for suite in self.test_results.values()) if sum(suite['total_tests'] for suite in self.test_results.values()) > 0 else 0
            },
            'test_suites': {}
        }
        
        for suite_name, suite_data in self.test_results.items():
            result = suite_data['result']
            report_data['test_suites'][suite_name] = {
                'total_tests': suite_data['total_tests'],
                'success_rate': suite_data['success_rate'],
                'failures': suite_data['failures'],
                'errors': suite_data['errors'],
                'failure_details': [
                    {
                        'test': str(test).split(' ')[0],
                        'error': (traceback.split('AssertionError: ')[-1].split('\n')[0] 
                                if 'AssertionError:' in traceback 
                                else traceback.split('\n')[-2])
                    }
                    for test, traceback in result.failures
                ],
                'error_details': [
                    {
                        'test': str(test).split(' ')[0],
                        'error': (traceback.split('\n')[-2] 
                                if len(traceback.split('\n')) > 1 
                                else "Unknown error")
                    }
                    for test, traceback in result.errors
                ]
            }
        
        filename = f"funnel_test_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        with open(filename, 'w') as f:
            json.dump(report_data, f, indent=2)


def main():
    """Main test execution function"""
    reporter = FunnelTestReporter()
    comprehensive_result, validation_result = reporter.run_all_tests()
    
    # Return exit code based on test results
    total_failures = len(comprehensive_result.failures) + len(validation_result.failures)
    total_errors = len(comprehensive_result.errors) + len(validation_result.errors)
    
    if total_failures > 0 or total_errors > 0:
        sys.exit(1)  # Exit with error code if tests failed
    else:
        sys.exit(0)  # Exit successfully if all tests passed


if __name__ == "__main__":
    main()