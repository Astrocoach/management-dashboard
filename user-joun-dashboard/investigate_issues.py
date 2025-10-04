#!/usr/bin/env python3
"""
Script to investigate churn risk and payment events issues
"""

import sys
import pandas as pd
import json
from main import load_all_json_data

def investigate_issues():
    print('Loading JSON data...')
    try:
        json_data = load_all_json_data()
        df = json_data['app_events']
        adapty_df = json_data['adapty_events']
        
        print(f'App events loaded: {len(df)} rows')
        print(f'Adapty events loaded: {len(adapty_df)} rows')
        
        # Check for payment_success events
        if not df.empty:
            payment_events = df[df['name'] == 'payment_success']
            print(f'Payment success events in app_events: {len(payment_events)}')
            
        if not adapty_df.empty:
            adapty_payment_events = adapty_df[adapty_df['name'] == 'payment_success']
            print(f'Payment success events in adapty_events: {len(adapty_payment_events)}')
            
            # Check what event names we have in adapty
            print('Adapty event names:')
            event_counts = adapty_df['name'].value_counts().head(10)
            for name, count in event_counts.items():
                print(f'  {name}: {count}')
        
        # Check churn analysis data requirements
        if not df.empty:
            print('\nChecking churn analysis requirements...')
            datetime_col = 'datetimeutc'
            datetime_count = df[datetime_col].notna().sum()
            unique_users = df['userid'].nunique()
            print(f'Users with {datetime_col}: {datetime_count}')
            print(f'Unique users: {unique_users}')
            
            # Check if we can calculate last activity
            if datetime_col in df.columns:
                last_activity = df.groupby('userid')[datetime_col].max()
                print(f'Last activity calculated for {len(last_activity)} users')
                
                # Check datetime format
                sample_datetime = df[datetime_col].dropna().iloc[0] if len(df[datetime_col].dropna()) > 0 else None
                print(f'Sample datetime: {sample_datetime} (type: {type(sample_datetime)})')
                
        # Check if there are any events that could be payment-related
        print('\nChecking for payment-related events...')
        if not df.empty:
            payment_related = df[df['name'].str.contains('payment|purchase|buy|pay', case=False, na=False)]
            print(f'Payment-related events in app_events: {len(payment_related)}')
            if len(payment_related) > 0:
                print('Payment-related event names:')
                for name in payment_related['name'].unique():
                    print(f'  {name}')
                    
        if not adapty_df.empty:
            adapty_payment_related = adapty_df[adapty_df['name'].str.contains('payment|purchase|buy|pay', case=False, na=False)]
            print(f'Payment-related events in adapty_events: {len(adapty_payment_related)}')
            if len(adapty_payment_related) > 0:
                print('Adapty payment-related event names:')
                for name in adapty_payment_related['name'].unique():
                    print(f'  {name}')
                    
    except Exception as e:
        print(f'Error: {e}')
        import traceback
        traceback.print_exc()

if __name__ == '__main__':
    investigate_issues()