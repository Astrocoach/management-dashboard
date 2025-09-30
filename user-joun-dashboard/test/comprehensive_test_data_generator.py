#!/usr/bin/env python3
"""
Comprehensive Test Data Generator for Analytics Dashboard
This module generates synthetic datasets with known characteristics for testing algorithms.
"""

import pandas as pd
import numpy as np
import json
from datetime import datetime, timedelta
from typing import Dict, List, Tuple, Optional
import random

class AnalyticsTestDataGenerator:
    """Generate synthetic analytics data with known patterns for testing"""
    
    def __init__(self, seed: int = 42):
        """Initialize the generator with a random seed for reproducibility"""
        np.random.seed(seed)
        random.seed(seed)
        self.seed = seed
    
    def generate_user_events(self, 
                           num_users: int = 100,
                           days_range: int = 30,
                           events_per_user_range: Tuple[int, int] = (5, 50),
                           session_timeout_minutes: int = 30) -> pd.DataFrame:
        """
        Generate synthetic user event data with known session patterns
        
        Args:
            num_users: Number of unique users
            days_range: Number of days to generate data for
            events_per_user_range: Min and max events per user
            session_timeout_minutes: Session timeout for session reconstruction testing
            
        Returns:
            DataFrame with columns: analyticsid, userid, deviceid, appversion, 
                                  category, name, datetimeutc, appname, analyticsdata, 
                                  membershipid, created_at, updated_at
        """
        
        # Event types with realistic distributions
        event_types = {
            'app_event': [
                'open_SplashScreen', 'open_HomeScreen', 'open_ProfileScreen',
                'open_SettingsScreen', 'click_Feature_A', 'click_Feature_B',
                'click_Feature_C', 'scroll_Feed', 'search_Content'
            ],
            'adapty_event': [
                'paywall_viewed', 'payment_success', 'subscription_started',
                'trial_started', 'subscription_cancelled'
            ],
            'user_action': [
                'login', 'logout', 'share_content', 'like_content', 'comment'
            ]
        }
        
        # Device types
        device_types = ['ios', 'android']
        app_versions = ['1.0.0', '1.1.0', '1.2.0', '2.0.0']
        
        events = []
        event_id = 1
        
        base_date = datetime.now() - timedelta(days=days_range)
        
        for user_id in range(1, num_users + 1):
            # Determine user behavior pattern
            user_type = np.random.choice(['casual', 'regular', 'power'], p=[0.5, 0.3, 0.2])
            
            if user_type == 'casual':
                num_events = np.random.randint(events_per_user_range[0], events_per_user_range[0] + 10)
                session_frequency = 0.3  # Low session frequency
            elif user_type == 'regular':
                num_events = np.random.randint(15, 35)
                session_frequency = 0.6  # Medium session frequency
            else:  # power user
                num_events = np.random.randint(30, events_per_user_range[1])
                session_frequency = 0.9  # High session frequency
            
            # Generate device info
            device_type = np.random.choice(device_types)
            device_id = f"device_{user_id}_{device_type}"
            app_version = np.random.choice(app_versions)
            
            # Generate events for this user
            user_events = []
            current_date = base_date + timedelta(days=np.random.randint(0, days_range//2))
            
            events_generated = 0
            while events_generated < num_events:
                # Decide if this is a new session
                if not user_events or np.random.random() < session_frequency:
                    # Start new session
                    session_start = current_date + timedelta(
                        hours=np.random.randint(8, 22),
                        minutes=np.random.randint(0, 60)
                    )
                    
                    # Generate events within this session
                    session_events = np.random.randint(1, 8)  # 1-7 events per session
                    session_events = min(session_events, num_events - events_generated)
                    
                    for i in range(session_events):
                        # Events within session are close together
                        if i == 0:
                            event_time = session_start
                        else:
                            # Add 1-10 minutes between events in same session
                            event_time = user_events[-1]['datetimeutc'] + timedelta(
                                minutes=np.random.randint(1, 10)
                            )
                        
                        # Choose event type and name
                        category = np.random.choice(list(event_types.keys()), 
                                                  p=[0.7, 0.2, 0.1])  # Most events are app_events
                        event_name = np.random.choice(event_types[category])
                        
                        # Generate analytics data (JSON for payment events)
                        analytics_data = {}
                        if category == 'adapty_event' and 'payment' in event_name:
                            analytics_data = {
                                'adaptyObject': {
                                    'vendorProductId': f'product_{np.random.randint(1, 5)}',
                                    'localizedTitle': f'Premium Feature {np.random.randint(1, 5)}',
                                    'price': {
                                        'amount': round(np.random.uniform(0.99, 29.99), 2),
                                        'currencyCode': np.random.choice(['USD', 'EUR', 'GBP'])
                                    },
                                    'regionCode': np.random.choice(['US', 'EU', 'UK', 'CA'])
                                }
                            }
                        
                        event = {
                            'analyticsid': event_id,
                            'userid': user_id,
                            'deviceid': device_id,
                            'appversion': app_version,
                            'category': category,
                            'name': event_name,
                            'datetimeutc': event_time,
                            'appname': 'TestApp',
                            'analyticsdata': json.dumps(analytics_data) if analytics_data else '',
                            'membershipid': f'member_{user_id}' if np.random.random() < 0.3 else '',
                            'created_at': event_time,
                            'updated_at': event_time
                        }
                        
                        user_events.append(event)
                        events_generated += 1
                        event_id += 1
                
                # Move to next day with some probability
                if np.random.random() < 0.3:
                    current_date += timedelta(days=1)
            
            events.extend(user_events)
        
        df = pd.DataFrame(events)
        
        # Ensure datetime columns are properly formatted
        df['datetimeutc'] = pd.to_datetime(df['datetimeutc'])
        df['created_at'] = pd.to_datetime(df['created_at'])
        df['updated_at'] = pd.to_datetime(df['updated_at'])
        
        return df.sort_values(['userid', 'datetimeutc']).reset_index(drop=True)
    
    def generate_known_anomaly_users(self, base_df: pd.DataFrame, num_anomalies: int = 5) -> pd.DataFrame:
        """
        Add known anomalous users to the dataset for anomaly detection testing
        
        Args:
            base_df: Base dataframe to add anomalies to
            num_anomalies: Number of anomalous users to add
            
        Returns:
            DataFrame with anomalous users added
        """
        
        anomaly_events = []
        max_user_id = base_df['userid'].max()
        max_event_id = base_df['analyticsid'].max()
        
        for i in range(num_anomalies):
            user_id = max_user_id + i + 1
            
            # Create different types of anomalies
            anomaly_type = np.random.choice(['high_volume', 'unusual_pattern', 'bot_like'])
            
            if anomaly_type == 'high_volume':
                # User with extremely high event count
                num_events = np.random.randint(200, 500)  # Much higher than normal
                session_count = np.random.randint(50, 100)
                
            elif anomaly_type == 'unusual_pattern':
                # User with unusual event patterns
                num_events = np.random.randint(100, 200)
                session_count = 1  # All events in one session (unusual)
                
            else:  # bot_like
                # Bot-like behavior: very regular intervals
                num_events = np.random.randint(150, 300)
                session_count = np.random.randint(30, 60)
            
            # Generate events for anomalous user
            base_time = datetime.now() - timedelta(days=15)
            
            for event_idx in range(num_events):
                max_event_id += 1
                
                if anomaly_type == 'bot_like':
                    # Very regular intervals (every 5 minutes exactly)
                    event_time = base_time + timedelta(minutes=event_idx * 5)
                else:
                    # Random times
                    event_time = base_time + timedelta(
                        hours=np.random.randint(0, 24 * 15),
                        minutes=np.random.randint(0, 60)
                    )
                
                event = {
                    'analyticsid': max_event_id,
                    'userid': user_id,
                    'deviceid': f'anomaly_device_{user_id}',
                    'appversion': '1.0.0',
                    'category': 'app_event',
                    'name': np.random.choice(['open_HomeScreen', 'click_Feature_A']),
                    'datetimeutc': event_time,
                    'appname': 'TestApp',
                    'analyticsdata': '',
                    'membershipid': '',
                    'created_at': event_time,
                    'updated_at': event_time
                }
                
                anomaly_events.append(event)
        
        anomaly_df = pd.DataFrame(anomaly_events)
        anomaly_df['datetimeutc'] = pd.to_datetime(anomaly_df['datetimeutc'])
        anomaly_df['created_at'] = pd.to_datetime(anomaly_df['created_at'])
        anomaly_df['updated_at'] = pd.to_datetime(anomaly_df['updated_at'])
        
        # Combine with base data
        combined_df = pd.concat([base_df, anomaly_df], ignore_index=True)
        return combined_df.sort_values(['userid', 'datetimeutc']).reset_index(drop=True)
    
    def generate_cohort_test_data(self, cohort_sizes: List[int] = [100, 80, 60, 40, 20]) -> pd.DataFrame:
        """
        Generate data with known cohort retention patterns for testing cohort analysis
        
        Args:
            cohort_sizes: List of cohort sizes for each week
            
        Returns:
            DataFrame with predictable cohort retention patterns
        """
        
        events = []
        event_id = 1
        user_id = 1
        
        # Generate cohorts for 5 consecutive weeks
        base_date = datetime.now() - timedelta(weeks=5)
        
        for week, cohort_size in enumerate(cohort_sizes):
            cohort_start_date = base_date + timedelta(weeks=week)
            
            # Generate users for this cohort
            for user_idx in range(cohort_size):
                current_user_id = user_id + user_idx
                
                # First event (cohort entry)
                first_event_time = cohort_start_date + timedelta(
                    days=np.random.randint(0, 7),
                    hours=np.random.randint(8, 20)
                )
                
                event = {
                    'analyticsid': event_id,
                    'userid': current_user_id,
                    'deviceid': f'device_{current_user_id}',
                    'appversion': '1.0.0',
                    'category': 'app_event',
                    'name': 'open_SplashScreen',
                    'datetimeutc': first_event_time,
                    'appname': 'TestApp',
                    'analyticsdata': '',
                    'membershipid': '',
                    'created_at': first_event_time,
                    'updated_at': first_event_time
                }
                events.append(event)
                event_id += 1
                
                # Generate retention pattern (decreasing retention over time)
                retention_probability = 1.0  # 100% in week 0
                
                for retention_week in range(1, 6):  # Check retention for 5 weeks
                    # Decrease retention probability each week
                    retention_probability *= 0.7  # 70% retention rate per week
                    
                    if np.random.random() < retention_probability:
                        # User returns this week
                        return_date = first_event_time + timedelta(weeks=retention_week)
                        return_date += timedelta(
                            days=np.random.randint(0, 7),
                            hours=np.random.randint(8, 20)
                        )
                        
                        # Generate 1-3 events for returning user
                        num_return_events = np.random.randint(1, 4)
                        for _ in range(num_return_events):
                            event = {
                                'analyticsid': event_id,
                                'userid': current_user_id,
                                'deviceid': f'device_{current_user_id}',
                                'appversion': '1.0.0',
                                'category': 'app_event',
                                'name': np.random.choice(['open_HomeScreen', 'click_Feature_A']),
                                'datetimeutc': return_date,
                                'appname': 'TestApp',
                                'analyticsdata': '',
                                'membershipid': '',
                                'created_at': return_date,
                                'updated_at': return_date
                            }
                            events.append(event)
                            event_id += 1
                            
                            # Small time increment for multiple events
                            return_date += timedelta(minutes=np.random.randint(1, 30))
            
            user_id += cohort_size
        
        df = pd.DataFrame(events)
        df['datetimeutc'] = pd.to_datetime(df['datetimeutc'])
        df['created_at'] = pd.to_datetime(df['created_at'])
        df['updated_at'] = pd.to_datetime(df['updated_at'])
        
        return df.sort_values(['userid', 'datetimeutc']).reset_index(drop=True)
    
    def generate_segmentation_test_data(self) -> pd.DataFrame:
        """
        Generate data with known user segments for testing segmentation algorithms
        
        Returns:
            DataFrame with users that should clearly segment into distinct groups
        """
        
        events = []
        event_id = 1
        
        # Define 4 clear user segments
        segments = {
            'power_users': {
                'count': 20,
                'sessions_range': (15, 25),
                'events_per_session': (8, 15),
                'session_duration_range': (300, 600),  # 5-10 minutes
                'days_active_range': (20, 30)
            },
            'converters': {
                'count': 15,
                'sessions_range': (8, 15),
                'events_per_session': (5, 10),
                'session_duration_range': (180, 360),  # 3-6 minutes
                'days_active_range': (10, 20)
            },
            'explorers': {
                'count': 25,
                'sessions_range': (5, 12),
                'events_per_session': (3, 8),
                'session_duration_range': (120, 300),  # 2-5 minutes
                'days_active_range': (5, 15)
            },
            'churners': {
                'count': 40,
                'sessions_range': (1, 5),
                'events_per_session': (1, 4),
                'session_duration_range': (30, 120),  # 0.5-2 minutes
                'days_active_range': (1, 5)
            }
        }
        
        user_id = 1
        base_date = datetime.now() - timedelta(days=30)
        
        for segment_name, segment_config in segments.items():
            for _ in range(segment_config['count']):
                num_sessions = np.random.randint(*segment_config['sessions_range'])
                days_active = np.random.randint(*segment_config['days_active_range'])
                
                # Generate sessions for this user
                session_dates = sorted([
                    base_date + timedelta(days=np.random.randint(0, days_active))
                    for _ in range(num_sessions)
                ])
                
                for session_date in session_dates:
                    session_start = session_date + timedelta(
                        hours=np.random.randint(8, 20),
                        minutes=np.random.randint(0, 60)
                    )
                    
                    events_in_session = np.random.randint(*segment_config['events_per_session'])
                    session_duration = np.random.randint(*segment_config['session_duration_range'])
                    
                    for event_idx in range(events_in_session):
                        # Distribute events across session duration
                        event_time = session_start + timedelta(
                            seconds=(session_duration / events_in_session) * event_idx +
                                   np.random.randint(0, 60)
                        )
                        
                        event = {
                            'analyticsid': event_id,
                            'userid': user_id,
                            'deviceid': f'device_{user_id}',
                            'appversion': '1.0.0',
                            'category': 'app_event',
                            'name': np.random.choice(['open_HomeScreen', 'click_Feature_A', 'scroll_Feed']),
                            'datetimeutc': event_time,
                            'appname': 'TestApp',
                            'analyticsdata': '',
                            'membershipid': '',
                            'created_at': event_time,
                            'updated_at': event_time,
                            'expected_segment': segment_name  # Ground truth for validation
                        }
                        events.append(event)
                        event_id += 1
                
                user_id += 1
        
        df = pd.DataFrame(events)
        df['datetimeutc'] = pd.to_datetime(df['datetimeutc'])
        df['created_at'] = pd.to_datetime(df['created_at'])
        df['updated_at'] = pd.to_datetime(df['updated_at'])
        
        return df.sort_values(['userid', 'datetimeutc']).reset_index(drop=True)
    
    def generate_payment_test_data(self, user_ids: List[int]) -> pd.DataFrame:
        """
        Generate payment event data for testing revenue calculations
        
        Args:
            user_ids: List of user IDs to generate payments for
            
        Returns:
            DataFrame with payment events
        """
        
        payment_events = []
        event_id = 1
        
        # Product catalog with known prices
        products = [
            {'id': 'premium_monthly', 'title': 'Premium Monthly', 'price': 9.99},
            {'id': 'premium_yearly', 'title': 'Premium Yearly', 'price': 99.99},
            {'id': 'feature_unlock', 'title': 'Feature Unlock', 'price': 2.99},
            {'id': 'premium_weekly', 'title': 'Premium Weekly', 'price': 2.99},
        ]
        
        currencies = ['USD', 'EUR', 'GBP']
        regions = ['US', 'EU', 'UK', 'CA']
        
        # Generate payments for subset of users (realistic conversion rate)
        paying_users = np.random.choice(user_ids, size=len(user_ids)//10, replace=False)
        
        base_date = datetime.now() - timedelta(days=30)
        
        for user_id in paying_users:
            # Each paying user makes 1-3 payments
            num_payments = np.random.randint(1, 4)
            
            for _ in range(num_payments):
                payment_date = base_date + timedelta(
                    days=np.random.randint(0, 30),
                    hours=np.random.randint(8, 22)
                )
                
                product = np.random.choice(products)
                currency = np.random.choice(currencies)
                region = np.random.choice(regions)
                
                # Adjust price based on currency (simplified)
                price = product['price']
                if currency == 'EUR':
                    price *= 0.85
                elif currency == 'GBP':
                    price *= 0.75
                
                analytics_data = {
                    'adaptyObject': {
                        'vendorProductId': product['id'],
                        'localizedTitle': product['title'],
                        'price': {
                            'amount': round(price, 2),
                            'currencyCode': currency
                        },
                        'regionCode': region
                    }
                }
                
                event = {
                    'analyticsid': event_id,
                    'userid': user_id,
                    'deviceid': f'device_{user_id}',
                    'appversion': '1.0.0',
                    'category': 'adapty_event',
                    'name': 'payment_success',
                    'datetimeutc': payment_date,
                    'appname': 'TestApp',
                    'analyticsdata': json.dumps(analytics_data),
                    'membershipid': f'member_{user_id}',
                    'created_at': payment_date,
                    'updated_at': payment_date
                }
                payment_events.append(event)
                event_id += 1
        
        df = pd.DataFrame(payment_events)
        df['datetimeutc'] = pd.to_datetime(df['datetimeutc'])
        df['created_at'] = pd.to_datetime(df['created_at'])
        df['updated_at'] = pd.to_datetime(df['updated_at'])
        
        return df.sort_values(['userid', 'datetimeutc']).reset_index(drop=True)


def main():
    """Example usage of the test data generator"""
    generator = AnalyticsTestDataGenerator(seed=42)
    
    print("🔧 Generating test datasets...")
    
    # Generate base user events
    base_events = generator.generate_user_events(num_users=100, days_range=30)
    print(f"✅ Generated {len(base_events)} base events for {base_events['userid'].nunique()} users")
    
    # Add anomalous users
    events_with_anomalies = generator.generate_known_anomaly_users(base_events, num_anomalies=5)
    print(f"✅ Added anomalous users. Total events: {len(events_with_anomalies)}")
    
    # Generate cohort test data
    cohort_data = generator.generate_cohort_test_data()
    print(f"✅ Generated cohort test data: {len(cohort_data)} events")
    
    # Generate segmentation test data
    segmentation_data = generator.generate_segmentation_test_data()
    print(f"✅ Generated segmentation test data: {len(segmentation_data)} events")
    
    # Generate payment data
    user_ids = base_events['userid'].unique().tolist()
    payment_data = generator.generate_payment_test_data(user_ids)
    print(f"✅ Generated payment data: {len(payment_data)} payment events")
    
    print("\n📊 Test data generation complete!")


if __name__ == "__main__":
    main()