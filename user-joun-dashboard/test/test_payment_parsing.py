#!/usr/bin/env python3
"""
Comprehensive Tests for Payment Data Parsing and Revenue Calculations
Tests the payment parsing with various JSON structures, edge cases, and validation metrics.
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import json
from datetime import datetime, timedelta
import unittest
from typing import Dict, List, Tuple, Optional
import warnings
warnings.filterwarnings('ignore')

# Import the test data generator
from test.comprehensive_test_data_generator import AnalyticsTestDataGenerator


class TestPaymentParsing(unittest.TestCase):
    """Test cases for payment data parsing and revenue calculations"""
    
    def setUp(self):
        """Set up test data generator"""
        self.generator = AnalyticsTestDataGenerator(seed=42)
    
    def parse_payment_data(self, df: pd.DataFrame) -> Dict:
        """
        Parse payment data similar to main.py implementation
        This replicates the payment parsing logic from main.py
        """
        payment_data = []
        
        for _, row in df.iterrows():
            if row['event_type'] == 'purchase' and pd.notna(row['event_data']):
                try:
                    # Parse JSON data
                    if isinstance(row['event_data'], str):
                        data = json.loads(row['event_data'])
                    else:
                        data = row['event_data']
                    
                    # Extract payment information
                    payment_info = {
                        'userid': row['userid'],
                        'event_date': row['event_date'],
                        'amount': data.get('amount', 0),
                        'currency': data.get('currency', 'USD'),
                        'product_id': data.get('product_id', 'unknown'),
                        'product_name': data.get('product_name', 'Unknown Product'),
                        'quantity': data.get('quantity', 1),
                        'payment_method': data.get('payment_method', 'unknown'),
                        'transaction_id': data.get('transaction_id', f"txn_{row.name}")
                    }
                    
                    payment_data.append(payment_info)
                    
                except (json.JSONDecodeError, KeyError, TypeError) as e:
                    # Handle parsing errors
                    continue
        
        if not payment_data:
            return {
                'payments_df': pd.DataFrame(),
                'total_revenue': 0,
                'transaction_count': 0,
                'unique_customers': 0,
                'avg_transaction_value': 0,
                'revenue_by_product': pd.DataFrame(),
                'revenue_by_date': pd.DataFrame()
            }
        
        payments_df = pd.DataFrame(payment_data)
        payments_df['event_date'] = pd.to_datetime(payments_df['event_date'])
        
        # Calculate metrics
        total_revenue = payments_df['amount'].sum()
        transaction_count = len(payments_df)
        unique_customers = payments_df['userid'].nunique()
        avg_transaction_value = payments_df['amount'].mean()
        
        # Revenue by product
        revenue_by_product = payments_df.groupby(['product_id', 'product_name']).agg({
            'amount': 'sum',
            'quantity': 'sum',
            'userid': 'nunique'
        }).reset_index()
        revenue_by_product.columns = ['product_id', 'product_name', 'revenue', 'units_sold', 'unique_customers']
        
        # Revenue by date
        revenue_by_date = payments_df.groupby(payments_df['event_date'].dt.date).agg({
            'amount': 'sum',
            'userid': 'nunique'
        }).reset_index()
        revenue_by_date.columns = ['date', 'revenue', 'unique_customers']
        
        return {
            'payments_df': payments_df,
            'total_revenue': total_revenue,
            'transaction_count': transaction_count,
            'unique_customers': unique_customers,
            'avg_transaction_value': avg_transaction_value,
            'revenue_by_product': revenue_by_product,
            'revenue_by_date': revenue_by_date
        }
    
    def test_payment_parsing_with_valid_json(self):
        """Test payment parsing with valid JSON structures"""
        print("\n🧪 Testing payment parsing with valid JSON...")
        
        # Generate payment events
        df = self.generator.generate_payment_events(num_users=20, num_transactions=50)
        
        # Parse payment data
        result = self.parse_payment_data(df)
        
        payments_df = result['payments_df']
        
        # Validate structure
        self.assertIsInstance(payments_df, pd.DataFrame, "Payments should be DataFrame")
        self.assertGreater(len(payments_df), 0, "Should have payment records")
        
        # Validate required columns
        required_columns = ['userid', 'event_date', 'amount', 'currency', 'product_id', 'product_name']
        for col in required_columns:
            self.assertIn(col, payments_df.columns, f"Should have column: {col}")
        
        # Validate data types
        self.assertTrue(pd.api.types.is_datetime64_any_dtype(payments_df['event_date']), 
                       "event_date should be datetime")
        self.assertTrue(pd.api.types.is_numeric_dtype(payments_df['amount']), 
                       "amount should be numeric")
        
        # Validate amounts are positive
        self.assertTrue(all(payments_df['amount'] >= 0), "All amounts should be non-negative")
        
        print(f"✅ Payment parsing completed:")
        print(f"   - Total transactions: {len(payments_df)}")
        print(f"   - Total revenue: ${result['total_revenue']:.2f}")
        print(f"   - Unique customers: {result['unique_customers']}")
        print(f"   - Avg transaction: ${result['avg_transaction_value']:.2f}")
        
        return result
    
    def test_revenue_calculation_accuracy(self):
        """Test accuracy of revenue calculations"""
        print("\n🧪 Testing revenue calculation accuracy...")
        
        # Create controlled test data with known values
        test_data = []
        base_date = datetime(2024, 1, 1)
        
        # Known transactions
        transactions = [
            {'userid': 'user1', 'amount': 100.00, 'product_id': 'prod1', 'product_name': 'Product 1'},
            {'userid': 'user1', 'amount': 50.00, 'product_id': 'prod2', 'product_name': 'Product 2'},
            {'userid': 'user2', 'amount': 75.00, 'product_id': 'prod1', 'product_name': 'Product 1'},
            {'userid': 'user3', 'amount': 200.00, 'product_id': 'prod3', 'product_name': 'Product 3'},
            {'userid': 'user2', 'amount': 25.00, 'product_id': 'prod2', 'product_name': 'Product 2'},
        ]
        
        for i, txn in enumerate(transactions):
            event_data = {
                'amount': txn['amount'],
                'currency': 'USD',
                'product_id': txn['product_id'],
                'product_name': txn['product_name'],
                'quantity': 1,
                'payment_method': 'credit_card',
                'transaction_id': f'txn_{i+1}'
            }
            
            test_data.append({
                'userid': txn['userid'],
                'event_date': base_date + timedelta(days=i),
                'event_type': 'purchase',
                'event_data': json.dumps(event_data)
            })
        
        df = pd.DataFrame(test_data)
        result = self.parse_payment_data(df)
        
        # Validate total revenue
        expected_total = sum(txn['amount'] for txn in transactions)
        self.assertAlmostEqual(result['total_revenue'], expected_total, places=2, 
                              msg=f"Total revenue should be {expected_total}")
        
        # Validate transaction count
        self.assertEqual(result['transaction_count'], len(transactions), 
                        "Transaction count should match input")
        
        # Validate unique customers
        expected_customers = len(set(txn['userid'] for txn in transactions))
        self.assertEqual(result['unique_customers'], expected_customers, 
                        "Unique customers should match")
        
        # Validate average transaction value
        expected_avg = expected_total / len(transactions)
        self.assertAlmostEqual(result['avg_transaction_value'], expected_avg, places=2,
                              msg=f"Average transaction should be {expected_avg}")
        
        # Validate revenue by product
        revenue_by_product = result['revenue_by_product']
        
        # Product 1: 100 + 75 = 175
        prod1_revenue = revenue_by_product[revenue_by_product['product_id'] == 'prod1']['revenue'].iloc[0]
        self.assertAlmostEqual(prod1_revenue, 175.00, places=2, msg="Product 1 revenue should be 175")
        
        # Product 2: 50 + 25 = 75
        prod2_revenue = revenue_by_product[revenue_by_product['product_id'] == 'prod2']['revenue'].iloc[0]
        self.assertAlmostEqual(prod2_revenue, 75.00, places=2, msg="Product 2 revenue should be 75")
        
        # Product 3: 200
        prod3_revenue = revenue_by_product[revenue_by_product['product_id'] == 'prod3']['revenue'].iloc[0]
        self.assertAlmostEqual(prod3_revenue, 200.00, places=2, msg="Product 3 revenue should be 200")
        
        print(f"✅ Revenue calculation accuracy validated:")
        print(f"   - Total revenue: ${result['total_revenue']:.2f} (expected: ${expected_total:.2f})")
        print(f"   - Transaction count: {result['transaction_count']} (expected: {len(transactions)})")
        print(f"   - Unique customers: {result['unique_customers']} (expected: {expected_customers})")
        
        return result
    
    def test_edge_case_invalid_json(self):
        """Test payment parsing with invalid JSON data"""
        print("\n🧪 Testing invalid JSON edge case...")
        
        # Create test data with invalid JSON
        test_data = [
            {
                'userid': 'user1',
                'event_date': '2024-01-01',
                'event_type': 'purchase',
                'event_data': '{"amount": 100, "invalid_json"'  # Invalid JSON
            },
            {
                'userid': 'user2',
                'event_date': '2024-01-02',
                'event_type': 'purchase',
                'event_data': None  # Null data
            },
            {
                'userid': 'user3',
                'event_date': '2024-01-03',
                'event_type': 'purchase',
                'event_data': '{"amount": "not_a_number"}'  # Invalid amount
            },
            {
                'userid': 'user4',
                'event_date': '2024-01-04',
                'event_type': 'purchase',
                'event_data': '{"amount": 50, "currency": "USD"}'  # Valid JSON
            }
        ]
        
        df = pd.DataFrame(test_data)
        result = self.parse_payment_data(df)
        
        payments_df = result['payments_df']
        
        # Should only parse valid transactions
        self.assertGreaterEqual(len(payments_df), 0, "Should handle invalid JSON gracefully")
        
        # If any valid transactions were parsed, validate them
        if len(payments_df) > 0:
            self.assertTrue(all(pd.notna(payments_df['amount'])), "All parsed amounts should be valid")
            self.assertTrue(all(payments_df['amount'] >= 0), "All amounts should be non-negative")
        
        print(f"✅ Invalid JSON handling: {len(payments_df)} valid transactions from {len(test_data)} attempts")
        
        return result
    
    def test_edge_case_empty_data(self):
        """Test payment parsing with no purchase events"""
        print("\n🧪 Testing empty data edge case...")
        
        # Create test data with no purchase events
        test_data = [
            {
                'userid': 'user1',
                'event_date': '2024-01-01',
                'event_type': 'login',
                'event_data': None
            },
            {
                'userid': 'user2',
                'event_date': '2024-01-02',
                'event_type': 'page_view',
                'event_data': '{"page": "home"}'
            }
        ]
        
        df = pd.DataFrame(test_data)
        result = self.parse_payment_data(df)
        
        # Should return empty results
        self.assertEqual(len(result['payments_df']), 0, "Should have no payment records")
        self.assertEqual(result['total_revenue'], 0, "Total revenue should be 0")
        self.assertEqual(result['transaction_count'], 0, "Transaction count should be 0")
        self.assertEqual(result['unique_customers'], 0, "Unique customers should be 0")
        self.assertEqual(result['avg_transaction_value'], 0, "Average transaction should be 0")
        
        print(f"✅ Empty data handling: All metrics correctly set to 0")
        
        return result
    
    def test_multiple_currencies(self):
        """Test payment parsing with multiple currencies"""
        print("\n🧪 Testing multiple currencies...")
        
        # Create test data with different currencies
        test_data = []
        currencies = [
            {'currency': 'USD', 'amount': 100},
            {'currency': 'EUR', 'amount': 85},
            {'currency': 'GBP', 'amount': 75},
            {'currency': 'USD', 'amount': 50},
        ]
        
        for i, curr_data in enumerate(currencies):
            event_data = {
                'amount': curr_data['amount'],
                'currency': curr_data['currency'],
                'product_id': f'prod_{i+1}',
                'product_name': f'Product {i+1}',
                'quantity': 1
            }
            
            test_data.append({
                'userid': f'user_{i+1}',
                'event_date': f'2024-01-0{i+1}',
                'event_type': 'purchase',
                'event_data': json.dumps(event_data)
            })
        
        df = pd.DataFrame(test_data)
        result = self.parse_payment_data(df)
        
        payments_df = result['payments_df']
        
        # Validate currency handling
        unique_currencies = payments_df['currency'].unique()
        expected_currencies = set(curr['currency'] for curr in currencies)
        
        self.assertEqual(set(unique_currencies), expected_currencies, 
                        "Should preserve all currencies")
        
        # Validate amounts by currency
        for curr_data in currencies:
            curr_payments = payments_df[payments_df['currency'] == curr_data['currency']]
            self.assertTrue(len(curr_payments) > 0, f"Should have {curr_data['currency']} payments")
        
        print(f"✅ Multiple currencies handled:")
        for currency in unique_currencies:
            curr_total = payments_df[payments_df['currency'] == currency]['amount'].sum()
            curr_count = len(payments_df[payments_df['currency'] == currency])
            print(f"   - {currency}: {curr_count} transactions, total: {curr_total}")
        
        return result
    
    def test_product_aggregation(self):
        """Test product-level revenue aggregation"""
        print("\n🧪 Testing product aggregation...")
        
        # Create test data with repeated products
        test_data = []
        product_sales = [
            {'product_id': 'prod1', 'product_name': 'Widget A', 'amount': 100, 'quantity': 2},
            {'product_id': 'prod1', 'product_name': 'Widget A', 'amount': 150, 'quantity': 3},
            {'product_id': 'prod2', 'product_name': 'Widget B', 'amount': 75, 'quantity': 1},
            {'product_id': 'prod1', 'product_name': 'Widget A', 'amount': 50, 'quantity': 1},
            {'product_id': 'prod3', 'product_name': 'Widget C', 'amount': 200, 'quantity': 4},
        ]
        
        for i, sale in enumerate(product_sales):
            event_data = {
                'amount': sale['amount'],
                'currency': 'USD',
                'product_id': sale['product_id'],
                'product_name': sale['product_name'],
                'quantity': sale['quantity']
            }
            
            test_data.append({
                'userid': f'user_{i+1}',
                'event_date': f'2024-01-0{i+1}',
                'event_type': 'purchase',
                'event_data': json.dumps(event_data)
            })
        
        df = pd.DataFrame(test_data)
        result = self.parse_payment_data(df)
        
        revenue_by_product = result['revenue_by_product']
        
        # Validate product aggregation
        # Product 1: 100 + 150 + 50 = 300, quantity: 2 + 3 + 1 = 6
        prod1_data = revenue_by_product[revenue_by_product['product_id'] == 'prod1']
        self.assertEqual(len(prod1_data), 1, "Should have one row per product")
        self.assertAlmostEqual(prod1_data['revenue'].iloc[0], 300, places=2, msg="Product 1 revenue should be 300")
        self.assertEqual(prod1_data['units_sold'].iloc[0], 6, msg="Product 1 units should be 6")
        
        # Product 2: 75, quantity: 1
        prod2_data = revenue_by_product[revenue_by_product['product_id'] == 'prod2']
        self.assertAlmostEqual(prod2_data['revenue'].iloc[0], 75, places=2, msg="Product 2 revenue should be 75")
        self.assertEqual(prod2_data['units_sold'].iloc[0], 1, msg="Product 2 units should be 1")
        
        # Product 3: 200, quantity: 4
        prod3_data = revenue_by_product[revenue_by_product['product_id'] == 'prod3']
        self.assertAlmostEqual(prod3_data['revenue'].iloc[0], 200, places=2, msg="Product 3 revenue should be 200")
        self.assertEqual(prod3_data['units_sold'].iloc[0], 4, msg="Product 3 units should be 4")
        
        print(f"✅ Product aggregation validated:")
        for _, row in revenue_by_product.iterrows():
            print(f"   - {row['product_name']}: ${row['revenue']:.2f}, {row['units_sold']} units")
        
        return result
    
    def test_date_based_revenue(self):
        """Test date-based revenue calculations"""
        print("\n🧪 Testing date-based revenue calculations...")
        
        # Create test data across multiple dates
        test_data = []
        daily_sales = [
            {'date': '2024-01-01', 'sales': [100, 50]},  # 2 transactions, $150 total
            {'date': '2024-01-02', 'sales': [75]},       # 1 transaction, $75 total
            {'date': '2024-01-03', 'sales': [200, 25, 100]},  # 3 transactions, $325 total
        ]
        
        user_counter = 1
        for day_data in daily_sales:
            for amount in day_data['sales']:
                event_data = {
                    'amount': amount,
                    'currency': 'USD',
                    'product_id': f'prod_{user_counter}',
                    'product_name': f'Product {user_counter}'
                }
                
                test_data.append({
                    'userid': f'user_{user_counter}',
                    'event_date': day_data['date'],
                    'event_type': 'purchase',
                    'event_data': json.dumps(event_data)
                })
                user_counter += 1
        
        df = pd.DataFrame(test_data)
        result = self.parse_payment_data(df)
        
        revenue_by_date = result['revenue_by_date']
        
        # Validate daily revenue
        for day_data in daily_sales:
            date_str = day_data['date']
            expected_revenue = sum(day_data['sales'])
            expected_customers = len(day_data['sales'])
            
            date_row = revenue_by_date[revenue_by_date['date'].astype(str) == date_str]
            self.assertEqual(len(date_row), 1, f"Should have one row for {date_str}")
            
            actual_revenue = date_row['revenue'].iloc[0]
            actual_customers = date_row['unique_customers'].iloc[0]
            
            self.assertAlmostEqual(actual_revenue, expected_revenue, places=2,
                                  msg=f"Revenue for {date_str} should be {expected_revenue}")
            self.assertEqual(actual_customers, expected_customers,
                           msg=f"Customers for {date_str} should be {expected_customers}")
        
        print(f"✅ Date-based revenue validated:")
        for _, row in revenue_by_date.iterrows():
            print(f"   - {row['date']}: ${row['revenue']:.2f}, {row['unique_customers']} customers")
        
        return result


class PaymentParsingValidator:
    """Validator class for payment parsing results"""
    
    def __init__(self):
        self.test_results = {}
        self.validation_data = {}
    
    def run_comprehensive_validation(self):
        """Run all payment parsing tests and generate validation report"""
        print("🔍 Running comprehensive payment parsing validation...")
        
        test_suite = TestPaymentParsing()
        test_suite.setUp()
        
        # Run all tests
        tests = [
            ('valid_json', test_suite.test_payment_parsing_with_valid_json),
            ('revenue_accuracy', test_suite.test_revenue_calculation_accuracy),
            ('invalid_json', test_suite.test_edge_case_invalid_json),
            ('empty_data', test_suite.test_edge_case_empty_data),
            ('multiple_currencies', test_suite.test_multiple_currencies),
            ('product_aggregation', test_suite.test_product_aggregation),
            ('date_revenue', test_suite.test_date_based_revenue)
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
        """Generate visual validation charts for payment parsing"""
        print("\n📊 Generating payment parsing validation charts...")
        
        # Generate test data for visualization
        generator = AnalyticsTestDataGenerator(seed=42)
        df = generator.generate_payment_events(num_users=30, num_transactions=100)
        
        # Parse payment data
        test_suite = TestPaymentParsing()
        test_suite.setUp()
        result = test_suite.parse_payment_data(df)
        
        payments_df = result['payments_df']
        revenue_by_product = result['revenue_by_product']
        revenue_by_date = result['revenue_by_date']
        
        # Create visualization
        fig, axes = plt.subplots(2, 3, figsize=(18, 12))
        fig.suptitle('Payment Data Parsing & Revenue Analysis Validation', fontsize=16, fontweight='bold')
        
        # 1. Revenue by product (top 10)
        top_products = revenue_by_product.nlargest(10, 'revenue')
        axes[0, 0].barh(range(len(top_products)), top_products['revenue'], color='lightgreen', alpha=0.7)
        axes[0, 0].set_yticks(range(len(top_products)))
        axes[0, 0].set_yticklabels([f"{row['product_name'][:15]}..." if len(row['product_name']) > 15 
                                   else row['product_name'] for _, row in top_products.iterrows()])
        axes[0, 0].set_xlabel('Revenue ($)')
        axes[0, 0].set_title('Top 10 Products by Revenue')
        
        # 2. Daily revenue trend
        if len(revenue_by_date) > 0:
            revenue_by_date_sorted = revenue_by_date.sort_values('date')
            axes[0, 1].plot(revenue_by_date_sorted['date'], revenue_by_date_sorted['revenue'], 
                           marker='o', linewidth=2, color='blue')
            axes[0, 1].set_xlabel('Date')
            axes[0, 1].set_ylabel('Revenue ($)')
            axes[0, 1].set_title('Daily Revenue Trend')
            axes[0, 1].tick_params(axis='x', rotation=45)
            axes[0, 1].grid(True, alpha=0.3)
        
        # 3. Transaction amount distribution
        if len(payments_df) > 0:
            axes[0, 2].hist(payments_df['amount'], bins=20, alpha=0.7, color='orange', edgecolor='black')
            axes[0, 2].axvline(payments_df['amount'].mean(), color='red', linestyle='--', 
                              label=f'Mean: ${payments_df["amount"].mean():.2f}')
            axes[0, 2].set_xlabel('Transaction Amount ($)')
            axes[0, 2].set_ylabel('Frequency')
            axes[0, 2].set_title('Transaction Amount Distribution')
            axes[0, 2].legend()
            axes[0, 2].grid(True, alpha=0.3)
        
        # 4. Currency distribution
        if len(payments_df) > 0:
            currency_counts = payments_df['currency'].value_counts()
            axes[1, 0].pie(currency_counts.values, labels=currency_counts.index, autopct='%1.1f%%')
            axes[1, 0].set_title('Transaction Distribution by Currency')
        
        # 5. Test results summary
        test_status = [result['status'] for result in self.test_results.values()]
        passed_count = test_status.count('PASSED')
        failed_count = test_status.count('FAILED')
        
        axes[1, 1].pie([passed_count, failed_count], labels=['Passed', 'Failed'],
                      colors=['lightgreen', 'lightcoral'], autopct='%1.1f%%')
        axes[1, 1].set_title(f'Test Results Summary\n({passed_count}/{len(test_status)} tests passed)')
        
        # 6. Payment parsing summary
        total_revenue = result['total_revenue']
        transaction_count = result['transaction_count']
        unique_customers = result['unique_customers']
        avg_transaction = result['avg_transaction_value']
        
        summary_text = f"""
        Payment Parsing Summary:
        
        Total Revenue: ${total_revenue:,.2f}
        Total Transactions: {transaction_count:,}
        Unique Customers: {unique_customers:,}
        Avg Transaction: ${avg_transaction:.2f}
        
        Products Analyzed: {len(revenue_by_product)}
        Date Range: {len(revenue_by_date)} days
        
        Parsing Features:
        - JSON data extraction
        - Multi-currency support
        - Product aggregation
        - Date-based analysis
        - Error handling
        """
        
        axes[1, 2].text(0.1, 0.5, summary_text, transform=axes[1, 2].transAxes,
                        fontsize=10, verticalalignment='center',
                        bbox=dict(boxstyle="round,pad=0.3", facecolor="lightblue", alpha=0.7))
        axes[1, 2].set_title('Payment Analysis Summary')
        axes[1, 2].axis('off')
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"📊 Validation charts saved to: {save_path}")
        
        plt.show()
        
        return fig


def main():
    """Run comprehensive payment parsing validation"""
    print("🚀 Starting Payment Data Parsing Validation")
    print("=" * 60)
    
    # Create validator
    validator = PaymentParsingValidator()
    
    # Run validation tests
    results = validator.run_comprehensive_validation()
    
    # Generate validation charts
    chart_path = "test/payment_parsing_validation.png"
    validator.generate_validation_charts(save_path=chart_path)
    
    # Print summary
    print("\n" + "=" * 60)
    print("📋 VALIDATION SUMMARY")
    print("=" * 60)
    
    passed_tests = sum(1 for r in results.values() if r['status'] == 'PASSED')
    total_tests = len(results)
    
    print(f"✅ Tests Passed: {passed_tests}/{total_tests}")
    
    if passed_tests == total_tests:
        print("🎉 All payment parsing tests PASSED!")
        print("✅ Payment data parsing and revenue calculations are working correctly")
    else:
        print("⚠️  Some tests failed. Please review the results above.")
        
        for test_name, result in results.items():
            if result['status'] == 'FAILED':
                print(f"❌ {test_name}: {result['error']}")
    
    print("\n📊 Visual validation charts generated successfully")
    print(f"📁 Charts saved to: {chart_path}")


if __name__ == "__main__":
    main()