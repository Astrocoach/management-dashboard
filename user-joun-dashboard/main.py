import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from plotly.subplots import make_subplots
import json
from datetime import datetime, timedelta
import time
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import IsolationForest
import warnings
import os
from dotenv import load_dotenv
import hashlib
import requests
import io
from reportlab.lib.pagesizes import letter, A4
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.units import inch
from reportlab.lib import colors
warnings.filterwarnings('ignore')

# Load environment variables
load_dotenv()

# Authentication Functions
def check_credentials(email, password):
    """Check if the provided credentials are valid"""
    admin_email = os.getenv('ADMIN_EMAIL', 'admin@example.com')
    admin_password = os.getenv('ADMIN_PASSWORD', 'your_secure_password_here')
    
    return email == admin_email and password == admin_password

def login_page():
    """Display the login page"""
    # Hide sidebar and adjust main content area
    st.markdown("""
    <style>
    /* Hide sidebar */
    .css-1d391kg {display: none}
    .css-1rs6os {display: none}
    .css-17eq0hr {display: none}
    section[data-testid="stSidebar"] {display: none}
    
    /* Adjust main content area */
    .main .block-container {
        padding-top: 1rem;
        padding-left: 1rem;
        padding-right: 1rem;
        max-width: 100%;
    }
    
    /* Login form styling */
    .login-header {
        text-align: center;
        margin-bottom: 2rem;
        padding: 2rem 0;
    }
    
    .login-title {
        font-size: 3rem;
        font-weight: bold;
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        background-clip: text;
        margin-bottom: 0.5rem;
    }
    
    .login-subtitle {
        color: #666;
        font-size: 1.2rem;
    }
    
    
    </style>
    """, unsafe_allow_html=True)
    
    # Create login header at the top
    st.markdown("""
    <div class="login-header">
        <div class="login-title">🔐 Login to Dashboard</div>
        <div class="login-subtitle">Enter your credentials to access the analytics</div>
    </div>
    """, unsafe_allow_html=True)
    
    # Create centered login form
    col1, col2, col3 = st.columns([1, 1, 1])
    
    with col2:
        st.markdown('<div class="login-form-container">', unsafe_allow_html=True)
        
        with st.form("login_form"):
            email = st.text_input("📧 Email", placeholder="Enter your email")
            password = st.text_input("🔒 Password", type="password", placeholder="Enter your password")
            submit_button = st.form_submit_button("🚀 Login", use_container_width=True)
            
            if submit_button:
                if check_credentials(email, password):
                    st.session_state.authenticated = True
                    st.session_state.user_email = email
                    st.success("✅ Login successful! Redirecting...")
                    st.rerun()
                else:
                    st.error("❌ Invalid email or password. Please try again.")
        
        st.markdown('</div>', unsafe_allow_html=True)

def logout():
    """Logout function"""
    st.session_state.authenticated = False
    st.session_state.user_email = None
    st.rerun()

def check_authentication():
    """Check if user is authenticated"""
    if 'authenticated' not in st.session_state:
        st.session_state.authenticated = False
    
    if not st.session_state.authenticated:
        login_page()
        return False
    return True

# Page Configuration
st.set_page_config(
    page_title="User Analytics Dashboard",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
<style>
    .metric-card {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 20px;
        border-radius: 10px;
        color: white;
        box-shadow: 0 4px 6px rgba(0,0,0,0.1);
    }
    .metric-value {
        font-size: 32px;
        font-weight: bold;
        margin: 10px 0;
    }
    .metric-label {
        font-size: 14px;
        opacity: 0.9;
    }
    .metric-change {
        font-size: 12px;
        margin-top: 5px;
    }
    .stTabs [data-baseweb="tab-list"] {
        gap: 8px;
    }
    .stTabs [data-baseweb="tab"] {
        background-color: #f0f2f6;
        border-radius: 8px;
        padding: 10px 20px;
    }
</style>
""", unsafe_allow_html=True)

# ==================== DATA PROCESSING FUNCTIONS ====================

@st.cache_data
def load_and_process_csv(_uploaded_file):
    """Load and process uploaded CSV file"""
    try:
        # Debug information
        st.info(f"📁 Processing file: {_uploaded_file.name} ({_uploaded_file.size} bytes)")
        
        # Reset file pointer to beginning for Streamlit file uploader
        if hasattr(_uploaded_file, 'seek'):
            _uploaded_file.seek(0)
        
        # For Streamlit file uploader, we can also try reading as StringIO
        try:
            # First attempt: direct pandas read
            df = pd.read_csv(_uploaded_file)
        except pd.errors.EmptyDataError:
            st.error("The uploaded CSV file is empty or contains no data.")
            return None
        except pd.errors.ParserError as e:
            st.warning(f"Parser error with default settings: {str(e)}")
            # Try with different delimiter
            if hasattr(_uploaded_file, 'seek'):
                _uploaded_file.seek(0)
            try:
                df = pd.read_csv(_uploaded_file, delimiter=';')
                st.success("✅ Successfully parsed with semicolon delimiter")
            except Exception as e2:
                st.error(f"Error parsing CSV file with alternative delimiter: {str(e2)}. Please ensure the file is properly formatted.")
                return None
        except UnicodeDecodeError:
            # Try different encoding
            if hasattr(_uploaded_file, 'seek'):
                _uploaded_file.seek(0)
            try:
                df = pd.read_csv(_uploaded_file, encoding='latin-1')
                st.success("✅ Successfully parsed with latin-1 encoding")
            except Exception as e:
                st.error(f"Error with encoding: {str(e)}")
                return None
        except Exception as e:
            st.error(f"Error reading CSV file: {str(e)}")
            st.error(f"File type: {type(_uploaded_file)}")
            return None
        
        # Check if DataFrame is empty or has no columns
        if df.empty:
            st.error("The uploaded CSV file is empty. Please upload a file with data.")
            return None
            
        if len(df.columns) == 0:
            st.error("The uploaded CSV file has no columns. Please check the file format.")
            return None
            
        # Check if file has only one column with no header (common CSV parsing issue)
        if len(df.columns) == 1 and df.columns[0].startswith('Unnamed'):
            st.error("The CSV file appears to have formatting issues. Please ensure it has proper column headers and is comma-separated.")
            return None
        
        # Store original datetime values before processing
        original_datetime_values = None
        if 'datetimeutc' in df.columns:
            original_datetime_values = df['datetimeutc'].copy()
        
        # Parse datetime - Handle multiple formats
        if 'datetimeutc' in df.columns:
            # First try parsing with UTC (for ISO format with timezone)
            df['datetimeutc'] = pd.to_datetime(df['datetimeutc'], errors='coerce', utc=True)
            
            # For timezone-aware values, convert to timezone-naive
            if df['datetimeutc'].dt.tz is not None:
                df['datetimeutc'] = df['datetimeutc'].dt.tz_localize(None)
            
            # For values that failed UTC parsing, try without timezone assumption
            failed_mask = df['datetimeutc'].isnull()
            if failed_mask.any():
                # Use the stored original values for failed entries
                failed_values = original_datetime_values[failed_mask]
                
                # Parse without timezone assumption
                parsed_without_tz = pd.to_datetime(failed_values, errors='coerce')
                
                # Update the failed entries
                df.loc[failed_mask, 'datetimeutc'] = parsed_without_tz
                
        elif 'created_at' in df.columns:
            df['created_at'] = pd.to_datetime(df['created_at'], errors='coerce')
            df['datetimeutc'] = df['created_at']
        
        # Extract date components - Handle NaN values properly
        if 'datetimeutc' in df.columns:
            # Remove rows with invalid datetime values to prevent mixed types
            df = df.dropna(subset=['datetimeutc'])
            
            # Extract date components only from valid datetime values
            df['date'] = df['datetimeutc'].dt.date
            df['hour'] = df['datetimeutc'].dt.hour
            df['day_of_week'] = df['datetimeutc'].dt.day_name()
            df['month'] = df['datetimeutc'].dt.month
        
        # Success message
        st.success(f"✅ Successfully loaded {len(df)} rows with {len(df.columns)} columns")
        
        return df
    except Exception as e:
        st.error(f"Error loading CSV: {str(e)}")
        return None

@st.cache_data
def reconstruct_sessions(df, timeout_minutes=30):
    """Reconstruct user sessions based on time gaps"""
    if 'userid' not in df.columns or 'datetimeutc' not in df.columns:
        return df
    
    df = df.sort_values(['userid', 'datetimeutc'])
    df['time_diff'] = df.groupby('userid')['datetimeutc'].diff()
    df['new_session'] = (df['time_diff'] > pd.Timedelta(minutes=timeout_minutes)) | df['time_diff'].isna()
    df['session_id'] = df.groupby('userid')['new_session'].cumsum()
    
    return df

@st.cache_data
def calculate_session_metrics(df):
    """Calculate session-level metrics"""
    if 'session_id' not in df.columns:
        return pd.DataFrame()
    
    session_metrics = df.groupby(['userid', 'session_id']).agg({
        'datetimeutc': ['min', 'max', 'count'],
        'analyticsid': 'count'
    }).reset_index()
    
    session_metrics.columns = ['userid', 'session_id', 'session_start', 'session_end', 'event_count', 'total_events']
    session_metrics['session_duration'] = (session_metrics['session_end'] - session_metrics['session_start']).dt.total_seconds() / 60
    
    return session_metrics

@st.cache_data
def parse_payment_data(df):
    """Parse payment data from analyticsdata JSON field"""
    payment_data = []
    
    for idx, row in df.iterrows():
        if row.get('category') == 'adapty_event' and row.get('name') == 'payment_success':
            try:
                data = json.loads(row['analyticsdata']) if isinstance(row['analyticsdata'], str) else row['analyticsdata']
                adapty = data.get('adaptyObject', {})
                
                payment_data.append({
                    'userid': row['userid'],
                    'date': row['datetimeutc'],
                    'product_id': adapty.get('vendorProductId'),
                    'product_title': adapty.get('localizedTitle'),
                    'amount': adapty.get('price', {}).get('amount', 0),
                    'currency': adapty.get('price', {}).get('currencyCode'),
                    'region': adapty.get('regionCode', 'Unknown'),
                    'platform': 'iOS' if 'ios' in row.get('deviceid', '').lower() else 'Android'
                })
            except:
                continue
    
    return pd.DataFrame(payment_data)

@st.cache_data
def perform_user_segmentation(user_metrics_df, n_clusters=4):
    """Segment users using K-Means clustering"""
    if user_metrics_df.empty:
        return user_metrics_df
    
    features = ['total_sessions', 'avg_session_duration', 'total_events', 'days_active']
    
    # Handle missing values
    for col in features:
        if col not in user_metrics_df.columns:
            user_metrics_df[col] = 0
    
    X = user_metrics_df[features].fillna(0)
    
    # Check if we have enough samples for clustering
    n_samples = len(X)
    if n_samples < 2:
        # Not enough data for clustering, assign all users to a single segment
        user_metrics_df['segment'] = 0
        user_metrics_df['segment_label'] = 'All Users'
        return user_metrics_df
    
    # Adjust number of clusters based on available samples
    # K-means requires n_samples >= n_clusters
    effective_clusters = min(n_clusters, n_samples)
    
    try:
        # Standardize features
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)
        
        # K-Means clustering
        kmeans = KMeans(n_clusters=effective_clusters, random_state=42, n_init=10)
        user_metrics_df['segment'] = kmeans.fit_predict(X_scaled)
        
        # Label segments based on characteristics and available clusters
        if effective_clusters == 1:
            segment_labels = {0: 'All Users'}
        elif effective_clusters == 2:
            segment_labels = {0: 'Active Users', 1: 'Casual Users'}
        elif effective_clusters == 3:
            segment_labels = {0: 'Power Users', 1: 'Regular Users', 2: 'Casual Users'}
        else:  # 4 or more clusters
            segment_labels = {
                0: 'Power Users',
                1: 'Converters',
                2: 'Explorers',
                3: 'Churners'
            }
            # Add additional labels if more clusters
            for i in range(4, effective_clusters):
                segment_labels[i] = f'Segment {i+1}'
        
        user_metrics_df['segment_label'] = user_metrics_df['segment'].map(segment_labels)
        
    except Exception as e:
        # Fallback: assign all users to a single segment
        st.warning(f"User segmentation failed: {str(e)}. Assigning all users to single segment.")
        user_metrics_df['segment'] = 0
        user_metrics_df['segment_label'] = 'All Users'
    
    return user_metrics_df

@st.cache_data
def detect_anomalies(df):
    """Detect anomalous user behavior"""
    if df.empty:
        return pd.DataFrame(columns=['userid', 'total_events', 'total_sessions', 'anomaly', 'anomaly_label'])
    
    try:
        user_behavior = df.groupby('userid').agg({
            'analyticsid': 'count',
            'session_id': 'nunique'
        }).reset_index()
        
        user_behavior.columns = ['userid', 'total_events', 'total_sessions']
        
        # Check if we have enough samples for anomaly detection
        if len(user_behavior) < 2:
            # Not enough data for anomaly detection, mark all as normal
            user_behavior['anomaly'] = 1
            user_behavior['anomaly_label'] = 'Normal'
            return user_behavior
        
        # Isolation Forest for anomaly detection
        iso_forest = IsolationForest(contamination=0.1, random_state=42)
        user_behavior['anomaly'] = iso_forest.fit_predict(user_behavior[['total_events', 'total_sessions']])
        user_behavior['anomaly_label'] = user_behavior['anomaly'].map({1: 'Normal', -1: 'Anomaly'})
        
    except Exception as e:
        # Fallback: return empty dataframe with correct structure
        st.warning(f"Anomaly detection failed: {str(e)}. Skipping anomaly analysis.")
        return pd.DataFrame(columns=['userid', 'total_events', 'total_sessions', 'anomaly', 'anomaly_label'])
    
    return user_behavior

# ==================== JSON DATA PROCESSING FUNCTIONS ====================

@st.cache_data
def process_app_events_data(df):
    """Process app events data from JSON - similar to CSV processing"""
    try:
        if df.empty:
            return df
        
        # Parse datetime - Handle multiple formats
        if 'datetimeutc' in df.columns:
            # First try parsing with UTC (for ISO format with timezone)
            df['datetimeutc'] = pd.to_datetime(df['datetimeutc'], errors='coerce', utc=True)
            
            # For timezone-aware values, convert to timezone-naive
            if df['datetimeutc'].dt.tz is not None:
                df['datetimeutc'] = df['datetimeutc'].dt.tz_localize(None)
            
        elif 'created_at' in df.columns:
            df['created_at'] = pd.to_datetime(df['created_at'], errors='coerce')
            df['datetimeutc'] = df['created_at']
        
        # Extract date components - Handle NaN values properly
        if 'datetimeutc' in df.columns:
            # Remove rows with invalid datetime values to prevent mixed types
            df = df.dropna(subset=['datetimeutc'])
            
            # Extract date components only from valid datetime values
            df['date'] = df['datetimeutc'].dt.date
            df['hour'] = df['datetimeutc'].dt.hour
            df['day_of_week'] = df['datetimeutc'].dt.day_name()
            df['month'] = df['datetimeutc'].dt.month
        
        # Ensure userid is string for consistent correlation
        if 'userid' in df.columns:
            df['userid'] = df['userid'].astype(str)
        
        return df
        
    except Exception as e:
        st.error(f"Error processing app events data: {str(e)}")
        return df

# ==================== JSON DATA LOADING FUNCTIONS ====================

@st.cache_data
def load_app_events_json():
    """Load app events from JSON file"""
    try:
        file_path = "offline_data/app_event.json"
        if not os.path.exists(file_path):
            st.error(f"App events file not found: {file_path}")
            return pd.DataFrame()
        
        with open(file_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        # Handle nested JSON structure - extract data array
        if isinstance(data, dict) and 'data' in data:
            df = pd.DataFrame(data['data'])
        elif isinstance(data, list):
            df = pd.DataFrame(data)
        else:
            df = pd.DataFrame(data)
        
        if df.empty:
            st.warning("App events file is empty")
            return df
        
        # Process the data similar to CSV processing
        df = process_app_events_data(df)
        
        st.success(f"✅ Successfully loaded {len(df)} app events from JSON")
        return df
        
    except Exception as e:
        st.error(f"Error loading app events JSON: {str(e)}")
        return pd.DataFrame()

@st.cache_data
def load_adapty_events_json():
    """Load Adapty events from JSON file"""
    try:
        file_path = "offline_data/adapty_event.json"
        if not os.path.exists(file_path):
            st.error(f"Adapty events file not found: {file_path}")
            return pd.DataFrame()
        
        with open(file_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        # Handle nested JSON structure - extract data array
        if isinstance(data, dict) and 'data' in data:
            df = pd.DataFrame(data['data'])
        elif isinstance(data, list):
            df = pd.DataFrame(data)
        else:
            df = pd.DataFrame(data)
        
        if df.empty:
            st.warning("Adapty events file is empty")
            return df
        
        # Extract amount from analytic_attr_data before converting to JSON strings
        if 'analytic_attr_data' in df.columns:
            def extract_amount(attr_data):
                try:
                    if isinstance(attr_data, list):
                        for item in attr_data:
                            if isinstance(item, dict) and item.get('analytic_name') == 'amount':
                                return float(item.get('analytic_value', 0))
                    return 0.0
                except (ValueError, TypeError):
                    return 0.0
            
            df['amount'] = df['analytic_attr_data'].apply(extract_amount)
        
        # Extract currency from analyticsdata if available
        if 'analyticsdata' in df.columns:
            def extract_currency(analytics_data):
                try:
                    if isinstance(analytics_data, str):
                        data = json.loads(analytics_data)
                        if isinstance(data, dict) and 'adaptyObject' in data:
                            price = data['adaptyObject'].get('price', {})
                            return price.get('currencyCode', 'USD')
                    return 'USD'
                except (json.JSONDecodeError, TypeError):
                    return 'USD'
            
            df['currency'] = df['analyticsdata'].apply(extract_currency)
            
            # Extract additional product information
            def extract_product_info(analytics_data):
                try:
                    if isinstance(analytics_data, str):
                        data = json.loads(analytics_data)
                        if isinstance(data, dict) and 'adaptyObject' in data:
                            adapty_obj = data['adaptyObject']
                            return {
                                'product_id': adapty_obj.get('vendorProductId', ''),
                                'product_title': adapty_obj.get('localizedTitle', ''),
                                'region': adapty_obj.get('regionCode', 'Unknown')
                            }
                    return {'product_id': '', 'product_title': '', 'region': 'Unknown'}
                except (json.JSONDecodeError, TypeError):
                    return {'product_id': '', 'product_title': '', 'region': 'Unknown'}
            
            product_info = df['analyticsdata'].apply(extract_product_info)
            df['product_id'] = [info['product_id'] for info in product_info]
            df['product_title'] = [info['product_title'] for info in product_info]
            df['region'] = [info['region'] for info in product_info]
        
        # Handle complex data types (lists/dicts) by converting to JSON strings
        for col in df.columns:
            if df[col].dtype == 'object':
                # Check if column contains lists or dicts
                sample_values = df[col].dropna().head(5)
                if len(sample_values) > 0:
                    first_val = sample_values.iloc[0]
                    if isinstance(first_val, (list, dict)):
                        df[col] = df[col].apply(lambda x: json.dumps(x) if isinstance(x, (list, dict)) else str(x))
        
        # Process datetime
        if 'datetimeutc' in df.columns:
            df['datetimeutc'] = pd.to_datetime(df['datetimeutc'], errors='coerce')
            df = df.dropna(subset=['datetimeutc'])
            
            # Extract date components
            df['date'] = df['datetimeutc'].dt.date
            df['hour'] = df['datetimeutc'].dt.hour
            df['day_of_week'] = df['datetimeutc'].dt.day_name()
            df['month'] = df['datetimeutc'].dt.month
        
        # Ensure string columns are properly typed
        string_columns = ['category', 'name', 'appname']
        for col in string_columns:
            if col in df.columns:
                df[col] = df[col].astype(str)
        
        # Ensure userid is string for consistency
        if 'userid' in df.columns:
            df['userid'] = df['userid'].astype(str)
        
        st.success(f"✅ Successfully loaded {len(df)} Adapty events from JSON")
        return df
        
    except Exception as e:
        st.error(f"Error loading Adapty events JSON: {str(e)}")
        return pd.DataFrame()

@st.cache_data
def load_revenue_json():
    """Load revenue data from JSON file"""
    try:
        file_path = "offline_data/revenue.json"
        if not os.path.exists(file_path):
            st.error(f"Revenue file not found: {file_path}")
            return pd.DataFrame()
        
        with open(file_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        # Handle the nested date-based structure
        records = []
        if isinstance(data, dict) and 'data' in data:
            # Extract entries from each date
            for date_key, date_data in data['data'].items():
                if isinstance(date_data, dict) and 'entries' in date_data:
                    for entry in date_data['entries']:
                        # Add the date to each entry
                        entry_with_date = entry.copy()
                        entry_with_date['date'] = date_key
                        records.append(entry_with_date)
        
        if not records:
            st.warning("No revenue entries found in JSON file")
            return pd.DataFrame()
        
        df = pd.DataFrame(records)
        
        # Convert customer_user_id to string for correlation
        if 'customer_user_id' in df.columns:
            df['customer_user_id'] = df['customer_user_id'].astype(str)
        
        # Process datetime columns
        datetime_columns = ['purchase_date', 'last_updated_at']
        for col in datetime_columns:
            if col in df.columns:
                df[col] = pd.to_datetime(df[col], errors='coerce')
        
        # Ensure price_usd is numeric
        if 'price_usd' in df.columns:
            df['price_usd'] = pd.to_numeric(df['price_usd'], errors='coerce')
            # Create revenue_usd column for compatibility
            df['revenue_usd'] = df['price_usd']
        
        # Create created_at column from purchase_date for compatibility
        if 'purchase_date' in df.columns:
            df['created_at'] = df['purchase_date']
        
        st.success(f"✅ Successfully loaded {len(df)} revenue records from JSON")
        return df
        
    except Exception as e:
        st.error(f"Error loading revenue JSON: {str(e)}")
        return pd.DataFrame()

@st.cache_data
def correlate_user_revenue_data(app_events_df, adapty_events_df, revenue_df):
    """Correlate revenue events with user data using revenue.json as bridge"""
    try:
        if app_events_df.empty or revenue_df.empty:
            st.warning("Cannot correlate data: missing app events or revenue data")
            return pd.DataFrame(), pd.DataFrame()
        
        # Create user mapping from revenue data
        # revenue.json contains customer_user_id and profile_id
        available_columns = revenue_df.columns.tolist()
        profile_col = 'profile_id' if 'profile_id' in available_columns else 'adapty_profile_id'
        customer_col = 'customer_user_id'
        
        if profile_col not in available_columns or customer_col not in available_columns:
            st.warning(f"Missing required columns in revenue data. Available: {available_columns}")
            return pd.DataFrame(), pd.DataFrame()
        
        user_mapping = revenue_df[[customer_col, profile_col]].drop_duplicates()
        
        # Correlate app events with revenue data using customer_user_id
        # Convert userid in app_events to string for matching
        app_events_df['userid_str'] = app_events_df['userid'].astype(str)
        
        # Convert customer_user_id to string for consistent matching
        user_mapping[customer_col + '_str'] = user_mapping[customer_col].astype(str)
        
        # Merge app events with user mapping
        app_events_with_revenue = app_events_df.merge(
            user_mapping, 
            left_on='userid_str', 
            right_on=customer_col + '_str', 
            how='left'
        )
        
        # If we have adapty events, correlate them too
        correlated_adapty_df = pd.DataFrame()
        if not adapty_events_df.empty:
            # Adapty events should have userid that matches profile_id
            adapty_events_df['userid_str'] = adapty_events_df['userid'].astype(str)
            
            # Convert profile_id to string for consistent matching
            user_mapping[profile_col + '_str'] = user_mapping[profile_col].astype(str)
            
            correlated_adapty_df = adapty_events_df.merge(
                user_mapping,
                left_on='userid_str',
                right_on=profile_col + '_str',
                how='left'
            )
        
        # Create comprehensive user revenue summary
        user_revenue_summary = revenue_df.groupby('customer_user_id').agg({
            'revenue_usd': 'sum',
            'created_at': ['min', 'max', 'count']
        }).reset_index()
        
        user_revenue_summary.columns = [
            'customer_user_id', 'total_revenue_usd', 
            'first_purchase', 'last_purchase', 'purchase_count'
        ]
        
        # Add revenue summary to app events
        app_events_enriched = app_events_with_revenue.merge(
            user_revenue_summary,
            on='customer_user_id',
            how='left'
        )
        
        # Fill NaN revenue values with 0
        revenue_columns = ['total_revenue_usd', 'purchase_count']
        for col in revenue_columns:
            if col in app_events_enriched.columns:
                app_events_enriched[col] = app_events_enriched[col].fillna(0)
        
        st.success(f"✅ Successfully correlated {len(app_events_enriched)} app events with revenue data")
        
        return app_events_enriched, correlated_adapty_df
        
    except Exception as e:
        st.error(f"Error correlating user revenue data: {str(e)}")
        return pd.DataFrame(), pd.DataFrame()

@st.cache_data
def load_all_json_data():
    """Load and correlate all JSON data sources"""
    try:
        st.info("🔄 Loading JSON data sources...")
        
        # Load individual data sources
        app_events_df = load_app_events_json()
        adapty_events_df = load_adapty_events_json()
        revenue_df = load_revenue_json()
        
        # Correlate the data
        if not app_events_df.empty and not revenue_df.empty:
            app_events_enriched, adapty_events_correlated = correlate_user_revenue_data(
                app_events_df, adapty_events_df, revenue_df
            )
            
            return {
                'app_events': app_events_enriched,
                'adapty_events': adapty_events_correlated,
                'revenue': revenue_df,
                'app_events_raw': app_events_df,
                'adapty_events_raw': adapty_events_df
            }
        else:
            st.warning("Using raw data without correlation due to missing data")
            return {
                'app_events': app_events_df,
                'adapty_events': adapty_events_df,
                'revenue': revenue_df,
                'app_events_raw': app_events_df,
                'adapty_events_raw': adapty_events_df
            }
            
    except Exception as e:
        st.error(f"Error loading JSON data: {str(e)}")
        return {
            'app_events': pd.DataFrame(),
            'adapty_events': pd.DataFrame(),
            'revenue': pd.DataFrame(),
            'app_events_raw': pd.DataFrame(),
            'adapty_events_raw': pd.DataFrame()
        }

# ==================== VISUALIZATION FUNCTIONS ====================

def create_kpi_cards(metrics):
    """Create KPI metric cards"""
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("Daily Active Users", f"{metrics.get('dau', 0):,}", f"{metrics.get('dau_change', 0):.1f}%")
    with col2:
        st.metric("Total Revenue", f"${metrics.get('revenue', 0):,.2f}", f"{metrics.get('revenue_change', 0):.1f}%")
    with col3:
        st.metric("Conversion Rate", f"{metrics.get('conversion_rate', 0):.2f}%", f"{metrics.get('conv_change', 0):.1f}%")
    with col4:
        st.metric("Avg Session (min)", f"{metrics.get('avg_session', 0):.1f}", f"{metrics.get('session_change', 0):.1f}%")

def create_improved_sankey_diagram(df, top_n=10, min_users=50):
    """Create improved Sankey diagram focusing on major user flows"""
    
    # Sort by user and time
    df_sorted = df.sort_values(['userid', 'datetimeutc'])
    
    # Create event sequences (limit to first 3 events per user)
    df_sorted['event_rank'] = df_sorted.groupby('userid').cumcount()
    df_filtered = df_sorted[df_sorted['event_rank'] < 3]
    
    # Get transitions between consecutive events
    df_filtered['next_screen'] = df_filtered.groupby('userid')['name'].shift(-1)
    
    # Count transitions
    transitions = df_filtered.groupby(['name', 'next_screen']).size().reset_index(name='count')
    transitions = transitions.dropna()
    
    # Filter: Keep only high-volume transitions
    transitions = transitions[transitions['count'] >= min_users]
    
    # Keep only top screens by total volume
    screen_volumes = pd.concat([
        transitions.groupby('name')['count'].sum(),
        transitions.groupby('next_screen')['count'].sum()
    ]).groupby(level=0).sum().nlargest(top_n)
    
    top_screens = set(screen_volumes.index)
    
    # Filter transitions to only include top screens
    transitions = transitions[
        transitions['name'].isin(top_screens) & 
        transitions['next_screen'].isin(top_screens)
    ]
    
    # Take top transitions by count
    transitions = transitions.nlargest(15, 'count')
    
    if transitions.empty:
        st.warning("Not enough data to create user flow diagram")
        return None
    
    # Create unique labels
    all_screens = sorted(list(set(transitions['name'].unique()) | set(transitions['next_screen'].unique())))
    screen_dict = {screen: idx for idx, screen in enumerate(all_screens)}
    
    # Create colors - different colors for different stages
    node_colors = ['#667eea' if i % 2 == 0 else '#764ba2' for i in range(len(all_screens))]
    
    # Simplify screen names for readability
    simplified_labels = [
        label.replace('open_', '').replace('Screen', '').replace('click_', '') 
        for label in all_screens
    ]
    
    fig = go.Figure(data=[go.Sankey(
        node=dict(
            pad=20,
            thickness=30,
            line=dict(color="white", width=2),
            label=simplified_labels,
            color=node_colors,
            customdata=all_screens,
            hovertemplate='%{customdata}<br>%{value} users<extra></extra>'
        ),
        link=dict(
            source=[screen_dict[s] for s in transitions['name']],
            target=[screen_dict[s] for s in transitions['next_screen']],
            value=transitions['count'].tolist(),
            color='rgba(102, 126, 234, 0.3)',
            hovertemplate='%{value} users<extra></extra>'
        )
    )])
    
    fig.update_layout(
        title={
            'text': f"User Flow - Top {len(all_screens)} Screens (First 3 Events Per User)",
            'x': 0.5,
            'xanchor': 'center'
        },
        font=dict(size=12, family='Arial'),
        height=700,
        paper_bgcolor='white',
        plot_bgcolor='white'
    )
    
    return fig

def classify_user_segments(df):
    """Classify users into new vs existing based on onboarding behavior
    
    Returns:
        dict: {
            'new_users': set of user IDs who went through onboarding,
            'existing_users': set of user IDs who never completed onboarding
        }
    """
    # Users who completed onboarding (new users)
    onboarded_users = set(df[df['name'] == 'onboarding_completed']['userid'])
    
    # Users who started onboarding but never completed (also considered new)
    onboarding_started = set(df[df['name'].isin(['open_WizardScreen', 'click_Onboarding_Start'])]['userid'])
    
    # All new users (started or completed onboarding)
    new_users = onboarded_users | onboarding_started
    
    # Existing users (never went through onboarding)
    all_users = set(df['userid'])
    existing_users = all_users - new_users
    
    return {
        'new_users': new_users,
        'existing_users': existing_users,
        'onboarded_users': onboarded_users
    }

def filter_df_by_user_segment(df, user_segment, segment_type='new'):
    """Filter dataframe by user segment
    
    Args:
        df: Analytics dataframe
        user_segment: Dict from classify_user_segments()
        segment_type: 'new', 'existing', or 'all'
    
    Returns:
        Filtered dataframe
    """
    if segment_type == 'new':
        return df[df['userid'].isin(user_segment['new_users'])]
    elif segment_type == 'existing':
        return df[df['userid'].isin(user_segment['existing_users'])]
    else:  # 'all'
        return df

def create_funnel_chart(df, segment_type='all', adapty_df=None):
    """Create conversion funnel visualization with user segmentation
    
    Args:
        df: Analytics dataframe
        segment_type: 'new', 'existing', or 'all'
        adapty_df: Adapty events dataframe for payment events
    """
    # Classify users
    user_segments = classify_user_segments(df)
    
    # Filter data by segment
    filtered_df = filter_df_by_user_segment(df, user_segments, segment_type)
    
    if segment_type == 'new':
        # For new users, show the complete onboarding journey
        funnel_stages = [
            ('App Opened', ['open_SplashScreen', 'open_HomeScreen']),
            ('Onboarding Started', ['open_WizardScreen', 'click_Onboarding_Start']),
            ('Onboarding Completed', ['onboarding_completed']),
            ('Paywall Viewed', ['open_PaywallScreen', 'paywall_viewed']),
            ('Payment Success', ['payment_success'])
        ]
        title_suffix = " - New Users"
        colors = ['#667eea', '#764ba2', '#f093fb', '#4facfe', '#43e97b']
    elif segment_type == 'existing':
        # For existing users, skip onboarding stages
        funnel_stages = [
            ('App Opened', ['open_SplashScreen', 'open_HomeScreen']),
            ('Paywall Viewed', ['open_PaywallScreen', 'paywall_viewed']),
            ('Payment Success', ['payment_success'])
        ]
        title_suffix = " - Existing Users"
        colors = ['#667eea', '#4facfe', '#43e97b']
    else:
        # All users (original funnel)
        funnel_stages = [
            ('App Opened', ['open_SplashScreen', 'open_HomeScreen']),
            ('Onboarding Started', ['open_WizardScreen', 'click_Onboarding_Start']),
            ('Onboarding Completed', ['onboarding_completed']),
            ('Paywall Viewed', ['open_PaywallScreen', 'paywall_viewed']),
            ('Payment Success', ['payment_success'])
        ]
        title_suffix = " - All Users"
        colors = ['#667eea', '#764ba2', '#f093fb', '#4facfe', '#43e97b']
    
    funnel_data = []
    for stage_name, events in funnel_stages:
        if 'payment_success' in events and adapty_df is not None and not adapty_df.empty:
            # Use Adapty events for payment_success
            count = adapty_df[adapty_df['name'].isin(events)]['userid'].nunique()
        else:
            # Use app events for other stages
            count = filtered_df[filtered_df['name'].isin(events)]['userid'].nunique()
        funnel_data.append({'Stage': stage_name, 'Users': count})
    
    funnel_df = pd.DataFrame(funnel_data)
    
    fig = go.Figure(go.Funnel(
        y=funnel_df['Stage'],
        x=funnel_df['Users'],
        textinfo="value+percent initial",
        marker=dict(color=colors[:len(funnel_stages)])
    ))
    
    fig.update_layout(title_text=f"Conversion Funnel{title_suffix}", height=500)
    return fig

def create_goal_funnel_visualization(df, session_timeout_minutes=30, top_n_dropoffs=5, segment_type='all', adapty_df=None):
    """Create Goal Funnel Visualization with drop-off paths (no cart stages)
    - Dynamically builds stages based on available events
    - Uses sessions to compute progression and drop-offs
    - Supports user segmentation (new/existing/all users)
    
    Args:
        df: Analytics dataframe
        session_timeout_minutes: Session timeout in minutes
        top_n_dropoffs: Number of top drop-off events to show
        segment_type: 'new', 'existing', or 'all'
    
    Returns: (figure, stage_stats)
    stage_stats: List[dict] with keys: stage, sessions, proceeded, proceeded_pct, dropoffs(list[(event,count)])
    """
    try:
        if df is None or df.empty:
            return None, []

        # Ensure required columns
        required_cols = {'userid', 'datetimeutc', 'name'}
        if not required_cols.issubset(set(df.columns)):
            return None, []

        # Apply user segmentation filter
        user_segments = classify_user_segments(df)
        df = filter_df_by_user_segment(df, user_segments, segment_type)
        
        # Reconstruct sessions if missing
        if 'session_id' not in df.columns:
            try:
                df = reconstruct_sessions(df, timeout_minutes=session_timeout_minutes)
            except Exception:
                # Fallback: treat each user-day as a session
                df['date'] = pd.to_datetime(df['datetimeutc']).dt.date
                df['session_id'] = df.groupby(['userid', 'date']).ngroup()

        # Define stage mapping based on user segment
        if segment_type == 'new':
            # For new users, show the complete onboarding journey
            stage_map = [
                ('App Entry', ['open_SplashScreen', 'open_HomeScreen']),
                ('Onboarding Started', ['open_WizardScreen', 'click_Onboarding_Start']),
                ('Onboarding Completed', ['onboarding_completed']),
                ('Paywall Viewed', ['open_PaywallScreen', 'paywall_viewed']),
                ('Purchase Completed', ['payment_success'])
            ]
        elif segment_type == 'existing':
            # For existing users, skip onboarding stages
            stage_map = [
                ('App Entry', ['open_SplashScreen', 'open_HomeScreen']),
                ('Paywall Viewed', ['open_PaywallScreen', 'paywall_viewed']),
                ('Purchase Completed', ['payment_success'])
            ]
        else:
            # All users (original stages)
            stage_map = [
                ('App Entry', ['open_SplashScreen', 'open_HomeScreen']),
                ('Onboarding Started', ['open_WizardScreen', 'click_Onboarding_Start']),
                ('Onboarding Completed', ['onboarding_completed']),
                ('Paywall Viewed', ['open_PaywallScreen', 'paywall_viewed']),
                ('Purchase Completed', ['payment_success'])
            ]

        # Keep only stages that exist in data
        available_events = set(df['name'].unique())
        stages = [(label, events) for label, events in stage_map if any(e in available_events for e in events)]
        if len(stages) < 2:
            return None, []

        # Sort by user-session-time
        df_sorted = df.sort_values(['userid', 'session_id', 'datetimeutc']).copy()
        df_sorted['event_index'] = df_sorted.groupby(['userid', 'session_id']).cumcount()

        # Prepare accumulators
        stage_stats = []
        stage_event_sets = [set(evts) for _, evts in stages]

        # Iterate sessions
        session_groups = df_sorted.groupby(['userid', 'session_id'])
        per_stage_sessions = [0] * len(stages)
        per_stage_proceeded = [0] * len(stages)
        per_stage_dropoffs = [dict() for _ in range(len(stages))]

        for (_, _), g in session_groups:
            names = g['name'].tolist()
            # For each stage, check presence and next progression
            for i, evset in enumerate(stage_event_sets):
                # indices where stage event occurs
                indices = [idx for idx, n in enumerate(names) if n in evset]
                if not indices:
                    continue
                per_stage_sessions[i] += 1
                last_idx = max(indices)
                # progressed if any next-stage event appears after last_idx
                progressed = False
                next_event = None
                if i < len(stage_event_sets) - 1:
                    next_evset = stage_event_sets[i+1]
                    for j in range(last_idx + 1, len(names)):
                        if names[j] in next_evset:
                            progressed = True
                            break
                        # capture first next event after stage occurrence (for drop-off path)
                        if next_event is None:
                            next_event = names[j]
                # If at final stage, there's no progression check
                if progressed:
                    per_stage_proceeded[i] += 1
                else:
                    label = next_event if next_event is not None else '(exit)'
                    per_stage_dropoffs[i][label] = per_stage_dropoffs[i].get(label, 0) + 1

        # Build stats per stage
        for i, (label, _) in enumerate(stages):
            sessions_count = per_stage_sessions[i]
            proceeded_count = per_stage_proceeded[i]
            proceeded_pct = (proceeded_count / sessions_count * 100) if sessions_count > 0 else 0.0
            # Top drop-offs
            drop_dict = per_stage_dropoffs[i]
            top_dropoffs = sorted(drop_dict.items(), key=lambda x: x[1], reverse=True)[:top_n_dropoffs]
            stage_stats.append({
                'stage': label,
                'sessions': sessions_count,
                'proceeded': proceeded_count,
                'proceeded_pct': proceeded_pct,
                'dropoffs': top_dropoffs
            })

        # Create funnel figure for sessions per stage with dynamic colors based on stage performance
        def get_progression_color(proceeded_pct):
            """Return color based on stage performance (progression percentage)"""
            if proceeded_pct >= 70:
                return '#10B981'  # Green for good stage performance
            elif proceeded_pct >= 40:
                return '#F59E0B'  # Orange for moderate stage performance
            else:
                return '#EF4444'  # Red for poor stage performance
        
        # Calculate colors based on stage performance (progression rates)
        stage_colors = []
        for s in stage_stats:
            stage_colors.append(get_progression_color(s['proceeded_pct']))
        
        fig = go.Figure(go.Funnel(
            y=[s['stage'] for s in stage_stats],
            x=[s['sessions'] for s in stage_stats],
            textinfo="value+percent initial",
            marker=dict(color=stage_colors)
        ))

        # Add drop-off annotations and summaries
        for idx, s in enumerate(stage_stats):
            dropped_count = max(s['sessions'] - s['proceeded'], 0)
            dropped_pct = (dropped_count / s['sessions'] * 100) if s['sessions'] > 0 else 0
            drop_text = "\n".join([f"{name}: {count}" for name, count in s['dropoffs']]) if s['dropoffs'] else "No drop-offs"
            annot = f"Users Dropped: {dropped_count} ({dropped_pct:.1f}%)\nTop drop-offs:\n{drop_text}"
            fig.add_annotation(
                xref='paper', yref='y', x=1.02, y=s['stage'],
                text=annot, showarrow=False, align='left',
                font=dict(size=11), bordercolor='lightgray', borderwidth=1, borderpad=6,
                bgcolor='rgba(255,255,255,0.8)'
            )

        fig.update_layout(
            title_text="Goal Funnel Visualization (Drop-off Paths)",
            height=600, margin=dict(l=80, r=250, t=60, b=40),
            paper_bgcolor='white'
        )

        return fig, stage_stats
    except Exception as e:
        # Return a basic info figure on error
        info_fig = go.Figure()
        info_fig.update_layout(title_text=f"Goal Funnel Visualization Error: {str(e)}")
        return info_fig, []

def create_goal_funnel_ga_style(df, session_timeout_minutes=30, top_n_dropoffs=5, annotation_mode="Standard", segment_type='all'):
    """Create a Google Analytics-style Goal Funnel with clear exits table.
    Layout:
    - Left: clean funnel bars with sessions per stage
    - Right: per-stage exit table (top drop-offs)
    Color scheme: green for progression, soft reds for exits.
    Supports user segmentation (new/existing/all users).
    """
    fig = None
    stage_stats = []
    try:
        # Reuse computation from goal funnel, but keep output clean
        base_fig, base_stats = create_goal_funnel_visualization(df, session_timeout_minutes, top_n_dropoffs, segment_type)
        stage_stats = base_stats
        if not stage_stats:
            return None, []

        # Compute missing stage events against expected mapping to highlight gaps
        stage_map = [
            ('App Entry', ['open_SplashScreen', 'open_HomeScreen']),
            ('Onboarding Started', ['open_WizardScreen', 'click_Onboarding_Start']),
            ('Onboarding Completed', ['onboarding_completed']),
            ('Paywall Viewed', ['open_PaywallScreen', 'paywall_viewed']),
            ('Purchase Completed', ['payment_success'])
        ]
        available_events = set(df['name'].unique()) if 'name' in df.columns else set()
        missing_stages = [label for label, events in stage_map if not any(e in available_events for e in events)]

        # Build exit rows for table
        def tidy_label(name):
            if name is None:
                return ''
            name = str(name)
            if name == '(exit)':
                return '(exit)'
            return name.replace('open_', '').replace('click_', '').replace('_', ' ')[:32]

        exit_rows = []
        for s in stage_stats:
            if s['dropoffs']:
                for ev, cnt in s['dropoffs'][:top_n_dropoffs]:
                    exit_rows.append([s['stage'], tidy_label(ev), cnt])
            else:
                exit_rows.append([s['stage'], '(no drop-offs)', 0])

        # Prepare funnel data
        stages = [s['stage'] for s in stage_stats]
        stage_sessions = [s['sessions'] for s in stage_stats]

        # Create subplots: funnel (xy) + table
        fig = make_subplots(
            rows=1, cols=2, column_widths=[0.55, 0.45],
            specs=[[{"type": "xy"}, {"type": "table"}]],
            horizontal_spacing=0.12
        )

        # Funnel on the left with dynamic colors based on progression rates
        def get_progression_color_ga(proceeded_pct):
            """Return color based on progression percentage for GA-style funnel"""
            if proceeded_pct >= 70:
                return '#D1FAE5'  # Light green for good progression
            elif proceeded_pct >= 40:
                return '#FEF3C7'  # Light yellow for moderate progression
            else:
                return '#FEE2E2'  # Light red for poor progression
        
        def get_border_color_ga(proceeded_pct):
            """Return border color based on progression percentage"""
            if proceeded_pct >= 70:
                return '#10B981'  # Green border for good progression
            elif proceeded_pct >= 40:
                return '#F59E0B'  # Orange border for moderate progression
            else:
                return '#EF4444'  # Red border for poor progression
        
        # Calculate colors based on progression rates
        stage_colors_ga = []
        border_colors_ga = []
        for s in stage_stats:
            stage_colors_ga.append(get_progression_color_ga(s['proceeded_pct']))
            border_colors_ga.append(get_border_color_ga(s['proceeded_pct']))
        
        fig.add_trace(go.Funnel(
            y=stages,
            x=stage_sessions,
            textinfo="value+percent initial",
            marker=dict(color=stage_colors_ga, line=dict(color=border_colors_ga, width=2))
        ), row=1, col=1)

        # Add drop-off annotations (controlled by annotation_mode)
        if annotation_mode in ["Standard", "Detailed"]:
            for s in stage_stats:
                dropped_count = max(s['sessions'] - s['proceeded'], 0)
                dropped_pct = (dropped_count / s['sessions'] * 100) if s['sessions'] > 0 else 0
                drop_text = f"Users Dropped: {dropped_count} ({dropped_pct:.1f}%)"
                
                # Dynamic annotation colors based on drop-off rate (inverse of progression)
                if dropped_pct <= 30:  # Low drop-off (good)
                    font_color = '#065F46'  # Dark green
                    bg_color = 'rgba(209,250,229,0.6)'  # Light green
                    border_color = '#10B981'  # Green
                elif dropped_pct <= 60:  # Moderate drop-off
                    font_color = '#92400E'  # Dark orange
                    bg_color = 'rgba(254,243,199,0.6)'  # Light yellow
                    border_color = '#F59E0B'  # Orange
                else:  # High drop-off (bad)
                    font_color = '#7F1D1D'  # Dark red
                    bg_color = 'rgba(254,226,226,0.6)'  # Light red
                    border_color = '#EF4444'  # Red
                
                fig.add_annotation(
                    xref='paper', yref='y', x=0.30, y=s['stage'],
                    text=drop_text, showarrow=False,
                    font=dict(size=10, color=font_color),
                    bgcolor=bg_color, bordercolor=border_color, borderwidth=1, borderpad=3
                )

        # Add red drop count annotations per stage (only in Detailed mode)
        if annotation_mode == "Detailed":
            for s in stage_stats:
                dropped_count = max(s['sessions'] - s['proceeded'], 0)
                drop_pct = (100.0 - s['proceeded_pct']) if s['sessions'] > 0 else 0.0
                drop_text = f"Dropped: {dropped_count} ({drop_pct:.1f}%)"
                fig.add_annotation(
                    xref='paper', yref='y', x=0.37, y=s['stage'],
                    text=drop_text, showarrow=False,
                    font=dict(size=10, color='#7F1D1D'),
                    bgcolor='rgba(254,226,226,0.7)', bordercolor='#DC2626', borderwidth=1, borderpad=3
                )

        # Exit table on the right
        table_header = dict(values=['Stage', 'Exit To', 'Sessions'],
                            fill_color='#F3F4F6', align='left', font=dict(color='#111827', size=12))
        stage_col = [row[0] for row in exit_rows]
        exit_col = [row[1] for row in exit_rows]
        count_col = [row[2] for row in exit_rows]

        fig.add_trace(go.Table(
            header=table_header,
            cells=dict(
                values=[stage_col, exit_col, count_col],
                align='left',
                fill_color=[['#FFFFFF'] * len(exit_rows), ['#FEE2E2'] * len(exit_rows), ['#FECACA'] * len(exit_rows)],
                font=dict(color=['#111827', '#7F1D1D', '#7F1D1D'], size=11)
            )
        ), row=1, col=2)

        fig.update_layout(
            title_text="Goal Funnel Visualization (GA-style)",
            height=650,
            margin=dict(l=60, r=40, t=60, b=40),
            paper_bgcolor='white'
        )

        # Add a notice for missing stages/events
        if missing_stages:
            missing_text = "Missing events: " + ", ".join(missing_stages)
            fig.add_annotation(
                xref='paper', yref='paper', x=0.98, y=0.98,
                text=missing_text, showarrow=False, align='right',
                font=dict(size=12, color='#991B1B'),
                bgcolor='rgba(254,202,202,0.9)', bordercolor='#DC2626', borderwidth=1, borderpad=6
            )

        return fig, stage_stats
    except Exception as e:
        info_fig = go.Figure()
        info_fig.update_layout(title_text=f"Goal Funnel Visualization Error: {str(e)}")
        return info_fig, stage_stats

# ==================== AI SUMMARY FUNCTIONS ====================

def generate_fallback_summary(data_summary):
    """Generate a comprehensive fallback summary when AI API is unavailable"""
    try:
        if isinstance(data_summary, dict) and 'error' not in data_summary:
            total_users = data_summary.get('total_users', 0)
            total_events = data_summary.get('total_events', 0)
            revenue = data_summary.get('revenue_total', 0)
            avg_session = data_summary.get('avg_session_duration', 0)
            top_events = data_summary.get('top_events', {})
            user_segments = data_summary.get('user_segments', {})
            
            summary = f"""
# Analytics Summary Report

## Key Performance Insights
- **Total Users**: {total_users:,} users tracked in the system
- **Event Volume**: {total_events:,} total events recorded
- **Revenue Performance**: ${revenue:,.2f} total revenue generated
- **User Engagement**: {avg_session:.1f} minutes average session duration

## User Behavior Patterns
"""
            
            if top_events:
                summary += "**Most Popular Events:**\n"
                for event, count in list(top_events.items())[:5]:
                    percentage = (count / total_events * 100) if total_events > 0 else 0
                    summary += f"- {event}: {count:,} events ({percentage:.1f}%)\n"
            
            if user_segments:
                high_activity = user_segments.get('high_activity', 0)
                medium_activity = user_segments.get('medium_activity', 0)
                low_activity = user_segments.get('low_activity', 0)
                
                summary += f"""
**User Segmentation:**
- High Activity Users: {high_activity:,} ({(high_activity/total_users*100):.1f}% of total)
- Medium Activity Users: {medium_activity:,} ({(medium_activity/total_users*100):.1f}% of total)
- Low Activity Users: {low_activity:,} ({(low_activity/total_users*100):.1f}% of total)
"""
            
            summary += f"""
## Revenue Analysis
- Total Revenue: ${revenue:,.2f}
- Average Revenue per User: ${(revenue/total_users):.2f} (if all users contributed)
- Revenue per Event: ${(revenue/total_events):.4f} (if all events contributed)

## Recommendations for Growth
1. **User Engagement**: Focus on converting low-activity users to medium/high activity
2. **Event Optimization**: Analyze top-performing events to replicate success patterns
3. **Session Duration**: Work on increasing average session time through better UX
4. **Revenue Optimization**: Identify revenue-generating events and optimize conversion funnels

## Risk Areas to Monitor
1. **User Retention**: Monitor user activity levels to prevent churn
2. **Event Distribution**: Ensure balanced event distribution across user segments
3. **Revenue Concentration**: Diversify revenue sources if heavily dependent on few events
4. **Session Quality**: Address short session durations that may indicate user friction

*Note: This summary was generated using built-in analytics. For AI-powered insights, please configure your DeepSeek API key.*
"""
            return summary
        else:
            return "Unable to generate summary due to data processing errors."
            
    except Exception as e:
        return f"Error generating fallback summary: {str(e)}"

def generate_ai_summary(data_summary):
    """Generate AI-powered summary using DeepSeek API"""
    try:
        # DeepSeek API configuration
        api_key = os.getenv('DEEPSEEK_API_KEY', '')
        if not api_key:
            return "Error: DEEPSEEK_API_KEY not found in environment variables"
        
        url = "https://api.deepseek.com/v1/chat/completions"
        headers = {
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json"
        }
        
        # Extract key metrics for detailed analysis
        total_users = data_summary.get('total_users', 0)
        total_events = data_summary.get('total_events', 0)
        revenue = data_summary.get('revenue_total', 0)
        event_analysis = data_summary.get('event_analysis', {})
        session_analysis = data_summary.get('session_analysis', {})
        
        prompt = f"""
        Analyze the following Astrocoach app analytics data and provide a comprehensive business intelligence report.

        ## Core Metrics:
        - Total Users: {total_users:,}
        - Total Events: {total_events:,}
        - Date Range: {data_summary.get('date_range', 'Unknown')}
        - Revenue: ${revenue:,.2f}
        - Average Session Duration: {data_summary.get('avg_session_duration', 0):.1f} minutes

        ## Event Analysis:
        - Unique Event Types: {event_analysis.get('unique_events', 0)}
        - Most Common Event: {event_analysis.get('most_common_event', 'Unknown')} ({event_analysis.get('most_common_count', 0):,} occurrences)
        - Events per User: {event_analysis.get('events_per_user', 0)}
        - Top Events: {data_summary.get('top_events', {})}

        ## Session Data:
        {f"- Total Sessions: {session_analysis.get('total_sessions', 0):,}" if session_analysis else "- Session tracking not available"}
        {f"- Avg Sessions per User: {session_analysis.get('avg_sessions_per_user', 0)}" if session_analysis else ""}
        {f"- Users with Multiple Sessions: {session_analysis.get('users_with_multiple_sessions', 0):,}" if session_analysis else ""}

        ## User Segmentation:
        {data_summary.get('user_segments', {})}

        ## Context:
        This is Astrocoach, a mobile astrology app with user funnel: App Entry → Onboarding → Calendar/Predictions → Monetization.

        IMPORTANT FORMATTING REQUIREMENTS:
        - Use clean numbered sections (1., 2., 3., etc.) - NO asterisks or special characters before numbers
        - Provide complete analysis for ALL sections below
        - Each section must have substantial content (minimum 3-4 bullet points)
        - Use proper markdown formatting with clear headers

        Please provide a comprehensive analysis with these EXACT sections:

        ## 1. Key Performance Insights
        [Evaluate overall app health, user engagement patterns, and performance against industry benchmarks]

        ## 2. User Behavior Analysis  
        [Analyze user journey progression, engagement depth, and behavioral patterns]

        ## 3. Revenue Analysis
        [Assess monetization effectiveness, revenue opportunities, and conversion patterns]

        ## 4. Growth Recommendations
        [Provide 4-5 specific, actionable strategies for user acquisition, retention, and revenue growth]

        ## 5. Risk Areas
        [Identify 3-4 critical issues requiring immediate attention, including data quality concerns]

        ## 6. Data Quality Assessment
        [Highlight any tracking issues, anomalies, or data inconsistencies that need technical attention]

        Ensure each section is complete with specific insights and actionable recommendations. Do not leave any section empty or incomplete.
        """
        
        payload = {
            "model": "deepseek-reasoner",
            "messages": [
                {"role": "system", "content": """You are the Chief Analytics Officer and Strategic Advisor for Astrocoach, deeply invested in the company's success and growth. You don't just analyze data—you champion Astrocoach's mission and drive actionable insights that directly impact business outcomes.

## CRITICAL FORMATTING REQUIREMENTS

**MANDATORY**: Your response MUST include ALL 6 sections with substantial content:
1. Key Performance Insights (minimum 4 bullet points)
2. User Behavior Analysis (minimum 4 bullet points)  
3. Revenue Analysis (minimum 3 bullet points)
4. Growth Recommendations (minimum 5 specific strategies)
5. Risk Areas (minimum 4 critical issues)
6. Data Quality Assessment (minimum 3 technical observations)

**FORMATTING RULES**:
- Use clean numbered headers: ## 1. Section Name (NO asterisks or special characters)
- Each section must have substantial, specific content
- Use bullet points with actionable insights
- Never leave sections empty or incomplete
- Provide concrete numbers and recommendations where possible

## YOUR ROLE & MINDSET

You operate with an ownership mentality. Astrocoach's success is YOUR success. You:
- Think like a founder: every recommendation should drive measurable growth
- Balance data-driven rigor with strategic intuition
- Challenge assumptions and ask the hard questions
- Prioritize actions by ROI and implementation feasibility
- Speak with confidence backed by evidence, not corporate jargon

## CORE RESPONSIBILITIES

### 1. USER ANALYTICS MASTERY
- Dissect user behavior patterns to uncover hidden growth opportunities
- Identify friction points in user journeys and propose concrete solutions
- Segment users intelligently to enable targeted interventions
- Track and interpret cohort behavior, retention curves, and engagement metrics
- Translate complex analytics into clear, actionable stories

### 2. GROWTH STRATEGY EXPERTISE
- Design and prioritize growth experiments with clear hypotheses
- Recommend channel strategies based on CAC, LTV, and payback periods
- Identify viral loops, referral mechanics, and organic growth levers
- Benchmark against industry standards while innovating beyond them
- Build growth models that forecast impact of proposed initiatives

### 3. BUSINESS INTELLIGENCE
- Connect metrics to business outcomes (revenue, retention, satisfaction)
- Spot early warning signals in dashboards before they become problems
- Recommend operational improvements that scale with growth
- Quantify the business impact of product and marketing decisions

## YOUR COMMUNICATION STYLE

**Be Direct & Actionable**
- Start with the "so what" - why does this insight matter?
- Always include 2-3 specific next steps
- Use frameworks (AARRR, RFM, cohorts) when they add clarity
- Quantify impact: "This could increase retention by ~15%" not "This might help"

**Be Honest & Transparent**
- Call out data quality issues or limitations
- Admit when you need more information to make a recommendation
- Flag risks alongside opportunities
- If something won't work for Astrocoach, say so and explain why

**Be Strategic Yet Practical**
- Think 3 moves ahead but recommend what's achievable now
- Consider technical constraints, team bandwidth, and market timing
- Distinguish between quick wins and long-term strategic plays
- Always tie recommendations back to Astrocoach's growth objectives

## DECISION-MAKING FRAMEWORK

When analyzing any question, consider:
1. **Impact**: What's the potential upside for Astrocoach?
2. **Effort**: What resources and time are required?
3. **Confidence**: How certain are we this will work?
4. **Risk**: What could go wrong and how do we mitigate it?
5. **Measurement**: How will we know if it's working?

## OUTPUT STANDARDS

Every response should:
✓ Answer the specific question asked
✓ Provide context that makes the answer meaningful
✓ Include at least one actionable recommendation per section
✓ Quantify impact where possible (even rough estimates)
✓ Consider both short-term tactics and long-term strategy
✓ Reference relevant metrics or KPIs
✓ Acknowledge trade-offs or alternative approaches

## FORBIDDEN BEHAVIORS

Never:
✗ Give vague advice like "consider exploring" or "it might be worth testing"
✗ Ignore the practical constraints of a real business
✗ Recommend best practices without adapting them to Astrocoach
✗ Use analytics jargon without explaining what it means for the business
✗ Provide analysis without interpretation
✗ Suggest actions without explaining expected outcomes

## YOUR MISSION

Help Astrocoach make better decisions faster. Every interaction should leave the team more informed, more confident, and more equipped to drive growth. You're not just an analyst—you're a strategic partner who deeply cares about moving the needle.

Remember: Good advice is specific, timely, and grounded in both data and business reality. Great advice changes behavior and drives results.

Now, bring your full expertise to help Astrocoach succeed."""},
                {"role": "user", "content": prompt}
            ],
            "max_tokens": 8000,
            "temperature": 1
        }
        
        # Try with a reasonable timeout
        response = requests.post(url, headers=headers, json=payload, timeout=60)
        
        if response.status_code == 200:
            result = response.json()
            # For deepseek-reasoner, extract only the content, not the thinking process
            message = result['choices'][0]['message']
            if 'content' in message:
                return message['content']
            else:
                # Fallback for different response structure
                return str(message)
        elif response.status_code == 429:
            return "Error: Rate limit exceeded. Please try again in a few minutes."
        elif response.status_code == 401:
            return "Error: Invalid API key. Please check your DEEPSEEK_API_KEY."
        elif response.status_code == 403:
            return "Error: Access forbidden. Please check your API key permissions."
        else:
            return f"Error: API returned status code {response.status_code}. Response: {response.text}"
                    
    except requests.exceptions.Timeout:
        return "Error: Request timed out. The API is taking too long to respond."
    except requests.exceptions.ConnectionError:
        return "Error: Connection failed. Please check your internet connection."
    except Exception as e:
        return f"Error: Unexpected error occurred: {str(e)}"

def create_data_summary(df):
    """Create a comprehensive data summary for AI analysis"""
    try:
        # Use correct column names from your data structure
        user_col = 'userid' if 'userid' in df.columns else 'user_id'
        event_col = 'name' if 'name' in df.columns else 'event_name'
        time_col = 'datetimeutc' if 'datetimeutc' in df.columns else 'timestamp'
        
        summary = {
            "total_users": len(df[user_col].unique()) if user_col in df.columns else 0,
            "total_events": len(df),
            "date_range": f"{df[time_col].min()} to {df[time_col].max()}" if time_col in df.columns else "Unknown",
            "top_events": df[event_col].value_counts().head(10).to_dict() if event_col in df.columns else {},
            "revenue_total": df['revenue'].sum() if 'revenue' in df.columns else 0,
            "avg_session_duration": df.groupby(user_col)[time_col].apply(lambda x: (x.max() - x.min()).total_seconds() / 60).mean() if all(col in df.columns for col in [user_col, time_col]) else 0
        }
        
        # Add user segmentation insights
        if user_col in df.columns:
            user_activity = df.groupby(user_col).size()
            summary["user_segments"] = {
                "high_activity": len(user_activity[user_activity > user_activity.quantile(0.8)]),
                "medium_activity": len(user_activity[(user_activity > user_activity.quantile(0.2)) & (user_activity <= user_activity.quantile(0.8))]),
                "low_activity": len(user_activity[user_activity <= user_activity.quantile(0.2)])
            }
            
        # Add event type analysis
        if event_col in df.columns:
            event_counts = df[event_col].value_counts()
            summary["event_analysis"] = {
                "unique_events": len(event_counts),
                "most_common_event": event_counts.index[0] if len(event_counts) > 0 else "None",
                "most_common_count": int(event_counts.iloc[0]) if len(event_counts) > 0 else 0,
                "events_per_user": round(len(df) / len(df[user_col].unique()), 2) if user_col in df.columns and len(df[user_col].unique()) > 0 else 0
            }
            
        # Add session analysis if session_id exists
        if 'session_id' in df.columns and user_col in df.columns:
            session_stats = df.groupby(user_col)['session_id'].nunique()
            summary["session_analysis"] = {
                "total_sessions": df['session_id'].nunique(),
                "avg_sessions_per_user": round(session_stats.mean(), 2),
                "users_with_multiple_sessions": len(session_stats[session_stats > 1])
            }
        
        return summary
    except Exception as e:
        return {"error": f"Error creating data summary: {str(e)}"}

def generate_pdf_report(ai_summary, data_summary):
    """Generate a PDF report with AI summary and data insights"""
    try:
        buffer = io.BytesIO()
        doc = SimpleDocTemplate(buffer, pagesize=A4)
        styles = getSampleStyleSheet()
        story = []
        
        # Title
        title_style = ParagraphStyle(
            'CustomTitle',
            parent=styles['Heading1'],
            fontSize=24,
            spaceAfter=30,
            textColor=colors.HexColor('#1f2937'),
            alignment=1  # Center alignment
        )
        story.append(Paragraph("AI-Powered Analytics Summary", title_style))
        story.append(Spacer(1, 20))
        
        # Date
        date_style = ParagraphStyle(
            'DateStyle',
            parent=styles['Normal'],
            fontSize=12,
            textColor=colors.HexColor('#6b7280'),
            alignment=1
        )
        story.append(Paragraph(f"Generated on: {datetime.now().strftime('%B %d, %Y at %I:%M %p')}", date_style))
        story.append(Spacer(1, 30))
        
        # Data Overview Section
        story.append(Paragraph("Data Overview", styles['Heading2']))
        
        if isinstance(data_summary, dict) and 'error' not in data_summary:
            data_table_data = [
                ['Metric', 'Value'],
                ['Total Users', f"{data_summary.get('total_users', 0):,}"],
                ['Total Events', f"{data_summary.get('total_events', 0):,}"],
                ['Date Range', data_summary.get('date_range', 'Unknown')],
                ['Total Revenue', f"${data_summary.get('revenue_total', 0):,.2f}"],
                ['Avg Session Duration', f"{data_summary.get('avg_session_duration', 0):.1f} minutes"]
            ]
            
            data_table = Table(data_table_data, colWidths=[2.5*inch, 3*inch])
            data_table.setStyle(TableStyle([
                ('BACKGROUND', (0, 0), (-1, 0), colors.HexColor('#f3f4f6')),
                ('TEXTCOLOR', (0, 0), (-1, 0), colors.HexColor('#1f2937')),
                ('ALIGN', (0, 0), (-1, -1), 'LEFT'),
                ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
                ('FONTSIZE', (0, 0), (-1, 0), 12),
                ('BOTTOMPADDING', (0, 0), (-1, 0), 12),
                ('BACKGROUND', (0, 1), (-1, -1), colors.white),
                ('GRID', (0, 0), (-1, -1), 1, colors.HexColor('#e5e7eb'))
            ]))
            story.append(data_table)
        
        story.append(Spacer(1, 30))
        
        # AI Analysis Section
        story.append(Paragraph("AI-Powered Analysis", styles['Heading2']))
        story.append(Spacer(1, 12))
        
        # Split AI summary into paragraphs
        ai_paragraphs = ai_summary.split('\n\n')
        for paragraph in ai_paragraphs:
            if paragraph.strip():
                story.append(Paragraph(paragraph.strip(), styles['Normal']))
                story.append(Spacer(1, 12))
        
        # Build PDF
        doc.build(story)
        buffer.seek(0)
        return buffer
        
    except Exception as e:
        st.error(f"Error generating PDF: {str(e)}")
        return None

# ==================== MAIN APP ====================

def main():
    st.title("📊 Astrocoach User Analytics Dashboard")
    st.markdown("### Comprehensive User Behavior & Revenue Analytics")
    
    # Sidebar
    with st.sidebar:
        st.header("⚙️ Configuration")
        
        # JSON Data Status
        st.subheader("📊 JSON Data Sources")
        st.info("🔄 Using integrated JSON data sources")
        
        # Data refresh button
        if st.button("🔄 Refresh Data", use_container_width=True):
            st.cache_data.clear()
            st.rerun()
        
        st.divider()
        
        # Filters
        st.subheader("🔍 Filters")
        date_range = st.selectbox("Date Range", ["Last 7 Days", "Last 30 Days", "Last 90 Days", "All Time"])
        
        # User Search
        st.subheader("👤 User Search")
        user_search = st.text_input("Search by User ID")
        
        st.divider()
        
        # Session Settings
        st.subheader("⏱️ Session Settings")
        session_timeout = st.slider("Session Timeout (minutes)", 10, 60, 30)

        # Sankey Settings
        st.subheader("🔄 User Flow Settings")
        sankey_top_n = st.slider("Top Screens to Display", 5, 15, 10)
        sankey_min_users = st.slider("Min Users per Flow", 10, 200, 50)

        # Funnel Settings
        st.subheader("🔻 Funnel Settings")
        top_n_dropoffs = st.slider("Top Drop-offs per Stage", 1, 10, 5)
        annotation_mode = st.selectbox("Funnel Annotation Level", ["Minimal", "Standard", "Detailed"], index=1)
        
        st.divider()
        st.info("💡 JSON data sources are automatically loaded")
    
    # Main Content - Load JSON data
    with st.spinner("Loading and processing JSON data..."):
        json_data = load_all_json_data()
        
        # Extract data from the loaded JSON
        df = json_data['app_events']
        adapty_df = json_data['adapty_events'] 
        revenue_df = json_data['revenue']
        
        if not df.empty:
            # Reconstruct sessions
            df = reconstruct_sessions(df, session_timeout)
            
            # Calculate metrics
            session_metrics = calculate_session_metrics(df)
            
            # Use Adapty events for payment analysis (filter for payment_success events)
            if not adapty_df.empty:
                payment_df = adapty_df[adapty_df['name'] == 'payment_success'].copy()
                # If no payment_success events, try to use all adapty events with amount > 0
                if payment_df.empty and 'amount' in adapty_df.columns:
                    payment_df = adapty_df[adapty_df['amount'] > 0].copy()
            else:
                payment_df = pd.DataFrame()
            
            st.success(f"✅ Loaded {len(df):,} events from {df['userid'].nunique():,} users")
            
            # Calculate KPIs
            total_users = df['userid'].nunique()
            total_sessions = df['session_id'].nunique() if 'session_id' in df.columns else 0
            avg_session_duration = session_metrics['session_duration'].mean() if not session_metrics.empty else 0
            total_revenue = payment_df['amount'].sum() if not payment_df.empty and 'amount' in payment_df.columns else 0
            conversion_rate = (payment_df['userid'].nunique() / total_users * 100) if not payment_df.empty and 'userid' in payment_df.columns else 0
            
            kpi_metrics = {
                'dau': total_users,
                'dau_change': 12.5,
                'revenue': total_revenue,
                'revenue_change': 23.1,
                'conversion_rate': conversion_rate,
                'conv_change': 2.3,
                'avg_session': avg_session_duration,
                'session_change': -5.2
            }
            
            # Tabs
            tab_ai, tab1, tab2, tab3, tab4, tab5, tab6, tab7, tab8 = st.tabs([
                "🤖 AI Summary",
                "📈 Overview", 
                "🛤️ User Journey", 
                "⚡ Features", 
                "💰 Monetization", 
                "👥 Segmentation",
                "🔍 User Explorer",
                "🧠 Advanced Analytics",
                "📥 Export & Data"
            ])
            
            # TAB AI: AI SUMMARY
            with tab_ai:
                st.header("🤖 AI-Powered Analytics Summary")
                st.markdown("Get intelligent insights and recommendations powered by DeepSeek AI")
                
                # Create data summary for AI analysis
                data_summary = create_data_summary(df)
                
                col1, col2 = st.columns([2, 1])
                
                with col1:
                    st.subheader("📊 Data Overview")
                    
                    if isinstance(data_summary, dict) and 'error' not in data_summary:
                        # Display key metrics
                        metric_col1, metric_col2, metric_col3 = st.columns(3)
                        
                        with metric_col1:
                            st.metric("Total Users", f"{data_summary.get('total_users', 0):,}")
                            st.metric("Total Events", f"{data_summary.get('total_events', 0):,}")
                        
                        with metric_col2:
                            st.metric("Total Revenue", f"${data_summary.get('revenue_total', 0):,.2f}")
                            st.metric("Avg Session Duration", f"{data_summary.get('avg_session_duration', 0):.1f} min")
                        
                        with metric_col3:
                            if 'user_segments' in data_summary:
                                segments = data_summary['user_segments']
                                st.metric("High Activity Users", segments.get('high_activity', 0))
                                st.metric("Medium Activity Users", segments.get('medium_activity', 0))
                        
                        # Top Events Chart
                        if data_summary.get('top_events'):
                            st.subheader("🔥 Top Events")
                           
                            
                        else:
                            st.warning("No event data available for analysis")
                    
                    else:
                        st.error("Error creating data summary for AI analysis")
                    
                    with col2:
                        st.subheader("🚀 Generate AI Insights")
                        
                        if st.button("🧠 Generate AI Summary", type="primary", use_container_width=True):
                            # Create a progress container
                            progress_container = st.container()
                            
                            with progress_container:
                                # Progress bar
                                progress_bar = st.progress(0)
                                status_text = st.empty()
                                
                                # Step 1: Preparing data
                                status_text.text("🔄 Preparing analytics data...")
                                progress_bar.progress(20)
                                time.sleep(0.5)
                                
                                # Step 2: Connecting to AI
                                status_text.text("🤖 Connecting to DeepSeek AI (Think Model)...")
                                progress_bar.progress(40)
                                time.sleep(0.5)
                                
                                # Step 3: AI Processing
                                status_text.text("🧠 AI is analyzing your data (this may take 30-60 seconds)...")
                                progress_bar.progress(60)
                                
                                # Generate AI summary
                                ai_summary = generate_ai_summary(data_summary)
                                
                                # Step 4: Processing results
                                status_text.text("📊 Processing AI insights...")
                                progress_bar.progress(90)
                                time.sleep(0.3)
                                
                                if ai_summary and not ai_summary.startswith("Error"):
                                    # Step 5: Complete
                                    status_text.text("✅ Analysis completed successfully!")
                                    progress_bar.progress(100)
                                    time.sleep(0.5)
                                    
                                    st.session_state['ai_summary'] = ai_summary
                                    st.session_state['data_summary'] = data_summary
                                    
                                    # Clear progress indicators
                                    progress_container.empty()
                                    st.success("🎉 AI analysis completed! Scroll down to view insights.")
                                    st.rerun()  # Refresh to show the new insights
                                else:
                                    # Error handling
                                    status_text.text("❌ Analysis failed")
                                    progress_bar.progress(100)
                                    time.sleep(0.5)
                                    progress_container.empty()
                                    st.error(f"❌ {ai_summary}")
                                    # Clear any previous successful summary
                                    if 'ai_summary' in st.session_state:
                                        del st.session_state['ai_summary']
                        
                        # Show current status
                        if 'ai_summary' in st.session_state:
                            st.success("✅ AI insights available below")
                        else:
                            st.info("💡 Click to generate AI-powered insights")
                        
                        if st.button("📄 Download PDF Report", use_container_width=True):
                            if 'ai_summary' in st.session_state:
                                with st.spinner("Generating PDF report..."):
                                    pdf_buffer = generate_pdf_report(
                                        st.session_state['ai_summary'], 
                                        st.session_state['data_summary']
                                    )
                                    if pdf_buffer:
                                        st.download_button(
                                            label="📥 Download AI Analytics Report",
                                            data=pdf_buffer.getvalue(),
                                            file_name=f"ai_analytics_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.pdf",
                                            mime="application/pdf",
                                            use_container_width=True
                                        )
                            else:
                                st.warning("Please generate AI summary first")
                    
                    # Display AI Summary
                    if 'ai_summary' in st.session_state:
                        st.divider()
                        st.subheader("🎯 AI-Generated Insights")
                        
                        # Create expandable sections for better organization
                        with st.expander("📈 Key Performance Insights", expanded=True):
                            ai_content = st.session_state['ai_summary']
                            if "Key Performance Insights" in ai_content:
                                insights_section = ai_content.split("Key Performance Insights")[1].split("User Behavior Patterns")[0] if "User Behavior Patterns" in ai_content else ai_content.split("Key Performance Insights")[1]
                                st.markdown(insights_section.strip())
                            else:
                                st.markdown(ai_content[:500] + "..." if len(ai_content) > 500 else ai_content)
                        
                        with st.expander("👥 User Behavior Analysis"):
                            if "User Behavior Patterns" in ai_content:
                                behavior_section = ai_content.split("User Behavior Patterns")[1].split("Revenue Analysis")[0] if "Revenue Analysis" in ai_content else ai_content.split("User Behavior Patterns")[1]
                                st.markdown(behavior_section.strip())
                        
                        with st.expander("💰 Revenue Analysis"):
                            if "Revenue Analysis" in ai_content:
                                revenue_section = ai_content.split("Revenue Analysis")[1].split("Recommendations for Growth")[0] if "Recommendations for Growth" in ai_content else ai_content.split("Revenue Analysis")[1]
                                st.markdown(revenue_section.strip())
                        
                        with st.expander("🚀 Growth Recommendations"):
                            if "Recommendations for Growth" in ai_content:
                                recommendations_section = ai_content.split("Recommendations for Growth")[1].split("Risk Areas to Monitor")[0] if "Risk Areas to Monitor" in ai_content else ai_content.split("Recommendations for Growth")[1]
                                st.markdown(recommendations_section.strip())
                        
                        with st.expander("⚠️ Risk Areas"):
                            if "Risk Areas to Monitor" in ai_content:
                                risk_section = ai_content.split("Risk Areas to Monitor")[1]
                                st.markdown(risk_section.strip())
                    
                    else:
                        st.info("👆 Click 'Generate AI Summary' to get intelligent insights about your data")
                
                # TAB 1: OVERVIEW
                with tab1:
                    st.header("Executive Dashboard")
                    create_kpi_cards(kpi_metrics)
                    
                    st.divider()
                    
                    col1, col2 = st.columns(2)
                    
                    with col1:
                        st.subheader("Daily Active Users Trend")
                        if 'date' in df.columns and df['date'].notna().any():
                            try:
                                # Use only valid dates for daily users calculation
                                df_valid_dates = df.dropna(subset=['date'])
                                daily_users = df_valid_dates.groupby('date')['userid'].nunique().reset_index()
                                daily_users.columns = ['Date', 'Users']
                                fig = px.line(daily_users, x='Date', y='Users', markers=True)
                                fig.update_layout(height=400)
                                st.plotly_chart(fig, use_container_width=True)
                            except Exception as e:
                                st.error(f"Error creating daily users trend: {str(e)}")
                        else:
                            st.warning("No valid date data available for daily users trend")
                    
                    with col2:
                        st.subheader("Platform Distribution")
                        platform_data = df['deviceid'].apply(lambda x: 'iOS' if 'ios' in str(x).lower() else 'Android')
                        platform_counts = platform_data.value_counts()
                        fig = px.pie(values=platform_counts.values, names=platform_counts.index, hole=0.4)
                        fig.update_layout(height=400)
                        st.plotly_chart(fig, use_container_width=True)
                    
                    st.divider()
                    
                    col3, col4 = st.columns(2)
                    
                    with col3:
                        st.subheader("Hourly Activity Pattern")
                        hourly_activity = df.groupby('hour')['userid'].count().reset_index()
                        hourly_activity.columns = ['Hour', 'Events']
                        fig = px.bar(hourly_activity, x='Hour', y='Events', color='Events', color_continuous_scale='Viridis')
                        fig.update_layout(height=400)
                        st.plotly_chart(fig, use_container_width=True)
                    
                    with col4:
                        st.subheader("Day of Week Activity")
                        dow_order = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']
                        dow_activity = df.groupby('day_of_week')['userid'].count().reindex(dow_order).reset_index()
                        dow_activity.columns = ['Day', 'Events']
                        fig = px.bar(dow_activity, x='Day', y='Events', color='Events', color_continuous_scale='Blues')
                        fig.update_layout(height=400)
                        st.plotly_chart(fig, use_container_width=True)
                
                # TAB 2: USER JOURNEY
                with tab2:
                    st.header("User Journey Analysis")
                    
                    st.subheader("User Flow - Sankey Diagram")
                    st.info("Showing the most common paths users take through your app (first 3 events per user)")
                    sankey_fig = create_improved_sankey_diagram(df, top_n=sankey_top_n, min_users=sankey_min_users)
                    if sankey_fig:
                        st.plotly_chart(sankey_fig, use_container_width=True)
                    
                    st.divider()
                    
                    st.subheader("Conversion Funnel")
                    
                    # User segmentation info
                    user_segments = classify_user_segments(df)
                    col1, col2, col3 = st.columns(3)
                    with col1:
                        st.metric("New Users", len(user_segments['new_users']))
                    with col2:
                        st.metric("Existing Users", len(user_segments['existing_users']))
                    with col3:
                        st.metric("Total Users", len(user_segments['new_users']) + len(user_segments['existing_users']))
                    
                    # Funnel segment selector
                    segment_option = st.selectbox(
                        "Select User Segment for Funnel Analysis:",
                        options=['all', 'new', 'existing'],
                        format_func=lambda x: {
                            'all': 'All Users (Combined)',
                            'new': 'New Users (Went through onboarding)',
                            'existing': 'Existing Users (Skipped onboarding)'
                        }[x],
                        index=0
                    )
                    
                    # Display appropriate funnel based on selection
                    if segment_option == 'all':
                        # Show all three funnels side by side
                        st.write("**Comparison View - All Segments**")
                        col1, col2, col3 = st.columns(3)
                        
                        with col1:
                            st.write("**All Users**")
                            funnel_fig_all = create_funnel_chart(df, 'all', adapty_df)
                            st.plotly_chart(funnel_fig_all, use_container_width=True)
                        
                        with col2:
                            st.write("**New Users**")
                            funnel_fig_new = create_funnel_chart(df, 'new', adapty_df)
                            st.plotly_chart(funnel_fig_new, use_container_width=True)
                        
                        with col3:
                            st.write("**Existing Users**")
                            funnel_fig_existing = create_funnel_chart(df, 'existing', adapty_df)
                            st.plotly_chart(funnel_fig_existing, use_container_width=True)
                    else:
                        # Show single selected funnel
                        funnel_fig = create_funnel_chart(df, segment_option, adapty_df)
                        st.plotly_chart(funnel_fig, use_container_width=True)
                    
                    st.divider()

                    st.subheader("Goal Funnel Visualization (Drop-off Paths)")
                    
                    # Goal funnel segment selector
                    goal_segment_option = st.selectbox(
                        "Select User Segment for Goal Funnel:",
                        options=['all', 'new', 'existing'],
                        format_func=lambda x: {
                            'all': 'All Users (Combined)',
                            'new': 'New Users (Went through onboarding)',
                            'existing': 'Existing Users (Skipped onboarding)'
                        }[x],
                        index=1,  # Default to 'new' users for goal funnel
                        key='goal_funnel_segment'
                    )
                    
                    goal_fig, stage_stats = create_goal_funnel_ga_style(
                        df,
                        session_timeout_minutes=session_timeout,
                        top_n_dropoffs=top_n_dropoffs,
                        annotation_mode=annotation_mode,
                        segment_type=goal_segment_option
                    )
                    if goal_fig is not None:
                        st.plotly_chart(goal_fig, use_container_width=True)
                    else:
                        st.info("Insufficient data to build Goal Funnel Visualization.")

                    # Stage summary table
                    if stage_stats:
                        summary_df = pd.DataFrame([
                            {
                                'Stage': s['stage'],
                                'Sessions': s['sessions'],
                                'Proceeded': s['proceeded'],
                                'Proceeded %': round(s['proceeded_pct'], 1)
                            } for s in stage_stats
                        ])
                        st.dataframe(summary_df, use_container_width=True)

                        # Drop-off details per stage (compact expanders)
                        for s in stage_stats:
                            with st.expander(f"Exits from '{s['stage']}'"):
                                if s['dropoffs']:
                                    drop_df = pd.DataFrame(s['dropoffs'][:5], columns=['Exit To', 'Sessions'])
                                    st.dataframe(drop_df, use_container_width=True)
                                else:
                                    st.write("No drop-offs recorded for this stage.")
                    
                    st.subheader("Common Event Sequences")
                    # Top event sequences
                    df_sorted = df.sort_values(['userid', 'datetimeutc'])
                    df_sorted['event_sequence'] = df_sorted.groupby('userid')['name'].transform(lambda x: ' → '.join(x.head(5)))
                    top_sequences = df_sorted['event_sequence'].value_counts().head(10)
                    
                    sequence_df = pd.DataFrame({
                        'Sequence': top_sequences.index,
                        'Frequency': top_sequences.values
                    })
                    st.dataframe(sequence_df, use_container_width=True)
                
                # TAB 3: FEATURES
                with tab3:
                    st.header("Feature Analytics")
                    
                    col1, col2 = st.columns(2)
                    
                    with col1:
                        st.subheader("Top Features by Usage")
                        feature_usage = df.groupby('name').size().sort_values(ascending=False).head(15)
                        fig = px.bar(x=feature_usage.values, y=feature_usage.index, orientation='h',
                                    labels={'x': 'Usage Count', 'y': 'Feature'},
                                    color=feature_usage.values, color_continuous_scale='Teal')
                        fig.update_layout(height=500)
                        st.plotly_chart(fig, use_container_width=True)
                    
                    with col2:
                        st.subheader("Feature Engagement by User")
                        feature_users = df.groupby('name')['userid'].nunique().sort_values(ascending=False).head(15)
                        fig = px.bar(x=feature_users.values, y=feature_users.index, orientation='h',
                                    labels={'x': 'Unique Users', 'y': 'Feature'},
                                    color=feature_users.values, color_continuous_scale='Purp')
                        fig.update_layout(height=500)
                        st.plotly_chart(fig, use_container_width=True)
                    
                    st.divider()
                    
                    st.subheader("Feature Usage Heatmap")
                    pivot_data = df.pivot_table(
                        values='analyticsid',
                        index='day_of_week',
                        columns='hour',
                        aggfunc='count',
                        fill_value=0
                    )
                    
                    fig = px.imshow(pivot_data, 
                                   labels=dict(x="Hour of Day", y="Day of Week", color="Events"),
                                   aspect="auto",
                                   color_continuous_scale='YlOrRd')
                    fig.update_layout(height=400)
                    st.plotly_chart(fig, use_container_width=True)
                
                # TAB 4: MONETIZATION
                with tab4:
                    st.header("Monetization Insights")
                    
                    if not payment_df.empty:
                        col1, col2, col3, col4 = st.columns(4)
                        
                        with col1:
                            st.metric("Total Revenue", f"${payment_df['amount'].sum():,.2f}")
                        with col2:
                            st.metric("Total Transactions", f"{len(payment_df):,}")
                        with col3:
                            avg_transaction = payment_df['amount'].mean()
                            st.metric("Avg Transaction", f"${avg_transaction:.2f}")
                        with col4:
                            paying_users = payment_df['userid'].nunique()
                            st.metric("Paying Users", f"{paying_users:,}")
                        
                        st.divider()
                        
                        col1, col2 = st.columns(2)
                        
                        with col1:
                            st.subheader("Revenue by Product")
                            product_revenue = payment_df.groupby('product_title')['amount'].sum().sort_values(ascending=False)
                            fig = px.pie(values=product_revenue.values, names=product_revenue.index, hole=0.4)
                            fig.update_layout(height=400)
                            st.plotly_chart(fig, use_container_width=True)
                        
                        with col2:
                            st.subheader("Revenue by Region")
                            region_revenue = payment_df.groupby('region')['amount'].sum().sort_values(ascending=False).head(10)
                            fig = px.bar(x=region_revenue.index, y=region_revenue.values,
                                        labels={'x': 'Region', 'y': 'Revenue'},
                                        color=region_revenue.values, color_continuous_scale='Greens')
                            fig.update_layout(height=400)
                            st.plotly_chart(fig, use_container_width=True)
                        
                        st.divider()
                        
                        st.subheader("Revenue Trend")
                        if not payment_df.empty and 'date' in payment_df.columns:
                            try:
                                # Ensure proper date handling for payment data
                                payment_df_clean = payment_df.dropna(subset=['date'])
                                payment_df_clean['date'] = pd.to_datetime(payment_df_clean['date'], errors='coerce').dt.date
                                payment_df_clean = payment_df_clean.dropna(subset=['date'])
                                
                                if not payment_df_clean.empty:
                                    daily_revenue = payment_df_clean.groupby('date')['amount'].sum().reset_index()
                                    daily_revenue.columns = ['Date', 'Revenue']
                                    fig = px.line(daily_revenue, x='Date', y='Revenue', markers=True)
                                else:
                                    st.warning("No valid payment date data available")
                                    fig = None
                            except Exception as e:
                                st.error(f"Error creating revenue trend: {str(e)}")
                                fig = None
                        else:
                            st.warning("No payment data available for revenue trend")
                            fig = None
                        
                        if fig is not None:
                            fig.update_layout(height=400)
                            st.plotly_chart(fig, use_container_width=True)
                    else:
                        st.info("No payment data available. Upload payment events CSV to view monetization analytics.")
                
                # TAB 5: SEGMENTATION
                with tab5:
                    st.header("User Segmentation")
                    
                    try:
                        # User-level metrics
                        if df.empty or 'userid' not in df.columns:
                            st.warning("No user data available for segmentation analysis.")
                            user_metrics = pd.DataFrame(columns=['userid', 'total_sessions', 'total_events', 'days_active', 'avg_session_duration'])
                        else:
                            user_metrics = df.groupby('userid').agg({
                                'session_id': 'nunique',
                                'analyticsid': 'count',
                                'date': lambda x: x.nunique()
                            }).reset_index()
                            user_metrics.columns = ['userid', 'total_sessions', 'total_events', 'days_active']
                            
                            # Session duration
                            if not session_metrics.empty:
                                try:
                                    avg_duration = session_metrics.groupby('userid')['session_duration'].mean().reset_index()
                                    avg_duration.columns = ['userid', 'avg_session_duration']
                                    user_metrics = user_metrics.merge(avg_duration, on='userid', how='left')
                                except Exception as e:
                                    st.warning(f"Could not calculate session duration: {str(e)}")
                                    user_metrics['avg_session_duration'] = 0
                            else:
                                user_metrics['avg_session_duration'] = 0
                        
                        # Perform segmentation
                        user_metrics = perform_user_segmentation(user_metrics)
                        
                    except Exception as e:
                        st.error(f"Error in user segmentation: {str(e)}")
                        user_metrics = pd.DataFrame(columns=['userid', 'total_sessions', 'total_events', 'days_active', 'avg_session_duration', 'segment', 'segment_label'])
                    
                    col1, col2 = st.columns(2)
                    
                    with col1:
                        st.subheader("User Segments Distribution")
                        segment_counts = user_metrics['segment_label'].value_counts()
                        fig = px.pie(values=segment_counts.values, names=segment_counts.index, hole=0.4)
                        fig.update_layout(height=400)
                        st.plotly_chart(fig, use_container_width=True)
                    
                    with col2:
                        st.subheader("Segment Characteristics")
                        segment_stats = user_metrics.groupby('segment_label').agg({
                            'total_sessions': 'mean',
                            'total_events': 'mean',
                            'avg_session_duration': 'mean',
                            'days_active': 'mean'
                        }).round(2)
                        st.dataframe(segment_stats, use_container_width=True)
                    
                    st.divider()
                    
                    st.subheader("User Clustering Visualization")
                    fig = px.scatter(user_metrics, 
                                    x='total_sessions', 
                                    y='total_events',
                                    color='segment_label',
                                    size='days_active',
                                    hover_data=['userid', 'avg_session_duration'],
                                    labels={'total_sessions': 'Total Sessions', 
                                           'total_events': 'Total Events'})
                    fig.update_layout(height=500)
                    st.plotly_chart(fig, use_container_width=True)
                
                # TAB 6: USER EXPLORER
                with tab6:
                    st.header("Individual User Explorer")
                    
                    if user_search:
                        user_data = df[df['userid'] == int(user_search)]
                        
                        if not user_data.empty:
                            st.success(f"Found {len(user_data)} events for User ID: {user_search}")
                            
                            # User summary
                            col1, col2, col3, col4 = st.columns(4)
                            
                            with col1:
                                st.metric("Total Events", len(user_data))
                            with col2:
                                sessions = user_data['session_id'].nunique() if 'session_id' in user_data.columns else 0
                                st.metric("Sessions", sessions)
                            with col3:
                                days_active = user_data['date'].nunique()
                                st.metric("Days Active", days_active)
                            with col4:
                                user_payments = payment_df[payment_df['userid'] == int(user_search)] if not payment_df.empty else pd.DataFrame()
                                total_spent = user_payments['amount'].sum() if not user_payments.empty else 0
                                st.metric("Total Spent", f"${total_spent:.2f}")
                            
                            st.divider()
                            
                            st.subheader("User Journey Timeline")
                            timeline_data = user_data[['datetimeutc', 'name', 'category']].sort_values('datetimeutc')
                            st.dataframe(timeline_data, use_container_width=True, height=400)
                            
                            st.divider()
                            
                            col1, col2 = st.columns(2)
                            
                            with col1:
                                st.subheader("Feature Usage")
                                user_features = user_data['name'].value_counts()
                                fig = px.bar(x=user_features.values, y=user_features.index, orientation='h')
                                fig.update_layout(height=400)
                                st.plotly_chart(fig, use_container_width=True)
                            
                            with col2:
                                st.subheader("Activity by Day")
                                daily_activity = user_data.groupby('date').size()
                                fig = px.line(x=daily_activity.index, y=daily_activity.values, markers=True)
                                fig.update_layout(height=400, xaxis_title='Date', yaxis_title='Events')
                                st.plotly_chart(fig, use_container_width=True)
                            
                            if not user_payments.empty:
                                st.divider()
                                st.subheader("Payment History")
                                st.dataframe(user_payments, use_container_width=True)
                        else:
                            st.warning(f"No data found for User ID: {user_search}")
                    else:
                        st.info("Enter a User ID in the sidebar to explore individual user behavior")
                
                # TAB 7: ADVANCED ANALYTICS
                with tab7:
                    st.header("Advanced Analytics")
                    
                    st.subheader("Anomaly Detection")
                    anomaly_data = detect_anomalies(df)
                    
                    col1, col2 = st.columns(2)
                    
                    with col1:
                        anomaly_counts = anomaly_data['anomaly_label'].value_counts()
                        fig = px.pie(values=anomaly_counts.values, names=anomaly_counts.index)
                        fig.update_layout(height=400, title="User Behavior Classification")
                        st.plotly_chart(fig, use_container_width=True)
                    
                    with col2:
                        fig = px.scatter(anomaly_data, 
                                        x='total_sessions', 
                                        y='total_events',
                                        color='anomaly_label',
                                        hover_data=['userid'])
                        fig.update_layout(height=400, title="Anomaly Detection Scatter")
                        st.plotly_chart(fig, use_container_width=True)
                    
                    st.divider()
                    
                    st.subheader("Anomalous Users")
                    anomalous_users = anomaly_data[anomaly_data['anomaly'] == -1].sort_values('total_events', ascending=False)
                    st.dataframe(anomalous_users.head(20), use_container_width=True)
                    
                    st.divider()
                    
                    st.subheader("Churn Risk Analysis")
                    
                    try:
                        if user_metrics.empty or 'userid' not in user_metrics.columns:
                            st.warning("No user metrics available for churn analysis.")
                        else:
                            # Calculate days since last activity - FIXED TIMEZONE ISSUE
                            last_activity = df.groupby('userid')['datetimeutc'].max().reset_index()
                            last_activity.columns = ['userid', 'last_seen']
                            # Use timezone-naive datetime
                            current_time = pd.Timestamp.now().tz_localize(None)
                            last_activity['days_since_last_seen'] = (current_time - last_activity['last_seen']).dt.days
                            
                            # Merge with user metrics
                            churn_data = user_metrics.merge(last_activity[['userid', 'days_since_last_seen']], on='userid', how='left')
                            churn_data['churn_risk'] = churn_data['days_since_last_seen'].apply(
                                lambda x: 'High' if x > 7 else 'Medium' if x > 3 else 'Low'
                            )
                            
                            col1, col2 = st.columns(2)
                            
                            with col1:
                                churn_counts = churn_data['churn_risk'].value_counts()
                                fig = px.bar(x=churn_counts.index, y=churn_counts.values, 
                                            color=churn_counts.index,
                                            color_discrete_map={'Low': 'green', 'Medium': 'orange', 'High': 'red'})
                                fig.update_layout(height=400, title="Churn Risk Distribution", showlegend=False)
                                st.plotly_chart(fig, use_container_width=True)
                            
                            with col2:
                                high_risk_users = churn_data[churn_data['churn_risk'] == 'High'].nlargest(10, 'total_events')
                                st.write("High Risk Users (Top 10 by Activity)")
                                if not high_risk_users.empty:
                                    st.dataframe(high_risk_users[['userid', 'total_sessions', 'total_events', 'days_since_last_seen']], 
                                               use_container_width=True)
                                else:
                                    st.info("No high-risk users found.")
                    except Exception as e:
                        st.error(f"Error in churn risk analysis: {str(e)}")
                        st.info("Please check that user data is properly loaded.")
                    
                    st.divider()
                    
                    st.subheader("Cohort Retention Analysis")
                    
                    # Check if we have valid date data for cohort analysis
                    if 'date' in df.columns and not df['date'].empty and df['date'].notna().any():
                        try:
                            # Create cohorts based on first activity date
                            # Ensure we only work with valid dates
                            df_valid_dates = df.dropna(subset=['date'])
                            
                            if not df_valid_dates.empty:
                                first_activity = df_valid_dates.groupby('userid')['date'].min().reset_index()
                                first_activity.columns = ['userid', 'cohort_date']
                                first_activity['cohort_period'] = pd.to_datetime(first_activity['cohort_date']).dt.to_period('W')
                                first_activity['cohort'] = first_activity['cohort_period'].dt.start_time.dt.date
                                
                                # Calculate retention by cohort
                                cohort_users = first_activity.merge(df_valid_dates[['userid', 'date']], on='userid')
                                cohort_users['cohort_str'] = cohort_users['cohort'].astype(str)
                                cohort_users['period'] = ((pd.to_datetime(cohort_users['date']) - 
                                                          pd.to_datetime(cohort_users['cohort'])).dt.days / 7).astype(int)
                                
                                retention = cohort_users.groupby(['cohort_str', 'period'])['userid'].nunique().reset_index()
                                retention.rename(columns={'cohort_str': 'cohort'}, inplace=True)
                            else:
                                st.warning("No valid date data available for cohort analysis")
                                retention = pd.DataFrame()
                        except Exception as e:
                            st.error(f"Error in cohort analysis: {str(e)}")
                            retention = pd.DataFrame()
                    else:
                        st.warning("No date data available for cohort analysis")
                        retention = pd.DataFrame()
                    
                    # Only create pivot table and visualization if we have retention data
                    if not retention.empty:
                        try:
                            retention_pivot = retention.pivot(index='cohort', columns='period', values='userid')
                            
                            # Calculate retention percentage
                            if 0 in retention_pivot.columns:
                                retention_pct = retention_pivot.div(retention_pivot[0], axis=0) * 100
                                
                                fig = px.imshow(retention_pct.iloc[-10:].fillna(0), 
                                               labels=dict(x="Weeks Since First Activity", y="Cohort", color="Retention %"),
                                               aspect="auto",
                                               color_continuous_scale='RdYlGn',
                                               text_auto='.0f')
                                fig.update_layout(height=400, title="Weekly Cohort Retention (%)")
                                st.plotly_chart(fig, use_container_width=True)
                            else:
                                st.warning("Insufficient data for retention percentage calculation")
                        except Exception as e:
                            st.error(f"Error creating retention visualization: {str(e)}")
                    else:
                        st.info("No cohort data available to display retention analysis")
                
                # TAB 8: EXPORT & DATA
                with tab8:
                    st.header("Export & Raw Data")
                    
                    st.subheader("Download Processed Data")
                    
                    col1, col2, col3 = st.columns(3)
                    
                    with col1:
                        csv_events = df.to_csv(index=False).encode('utf-8')
                        st.download_button(
                            label="Download Events Data",
                            data=csv_events,
                            file_name=f"processed_events_{datetime.now().strftime('%Y%m%d')}.csv",
                            mime="text/csv",
                        )
                    
                    with col2:
                        if not session_metrics.empty:
                            csv_sessions = session_metrics.to_csv(index=False).encode('utf-8')
                            st.download_button(
                                label="Download Session Metrics",
                                data=csv_sessions,
                                file_name=f"session_metrics_{datetime.now().strftime('%Y%m%d')}.csv",
                                mime="text/csv",
                            )
                    
                    with col3:
                        if not user_metrics.empty:
                            csv_users = user_metrics.to_csv(index=False).encode('utf-8')
                            st.download_button(
                                label="Download User Segments",
                                data=csv_users,
                                file_name=f"user_segments_{datetime.now().strftime('%Y%m%d')}.csv",
                                mime="text/csv",
                            )
                    
                    st.divider()
                    
                    st.subheader("Data Summary")
                    
                    col1, col2 = st.columns(2)
                    
                    with col1:
                        st.write("**Events Data Overview**")
                        st.write(f"- Total Records: {len(df):,}")
                        st.write(f"- Unique Users: {df['userid'].nunique():,}")
                        st.write(f"- Date Range: {df['date'].min()} to {df['date'].max()}")
                        st.write(f"- Event Types: {df['name'].nunique():,}")
                        st.write(f"- Platforms: {df['deviceid'].apply(lambda x: 'iOS' if 'ios' in str(x).lower() else 'Android').nunique()}")
                    
                    with col2:
                        st.write("**Payment Data Overview**")
                        if not payment_df.empty:
                            st.write(f"- Total Transactions: {len(payment_df):,}")
                            st.write(f"- Paying Users: {payment_df['userid'].nunique():,}")
                            st.write(f"- Total Revenue: ${payment_df['amount'].sum():,.2f}")
                            st.write(f"- Avg Transaction: ${payment_df['amount'].mean():.2f}")
                            st.write(f"- Regions: {payment_df['region'].nunique()}")
                        else:
                            st.write("No payment data loaded")
                    
                    st.divider()
                    
                    st.subheader("Raw Data Preview")
                    
                    data_view = st.selectbox("Select Data to View", 
                                            ["Events Data", "Session Metrics", "User Metrics", "Payment Data"])
                    
                    if data_view == "Events Data":
                        st.dataframe(df.head(1000), use_container_width=True, height=400)
                    elif data_view == "Session Metrics":
                        if not session_metrics.empty:
                            st.dataframe(session_metrics.head(1000), use_container_width=True, height=400)
                        else:
                            st.info("No session metrics available")
                    elif data_view == "User Metrics":
                        if not user_metrics.empty:
                            st.dataframe(user_metrics.head(1000), use_container_width=True, height=400)
                        else:
                            st.info("No user metrics available")
                    elif data_view == "Payment Data":
                        if not payment_df.empty:
                            st.dataframe(payment_df.head(1000), use_container_width=True, height=400)
                        else:
                            st.info("No payment data available")
                    
                    st.divider()
                    
                    st.subheader("Data Quality Report")
                    
                    col1, col2 = st.columns(2)
                    
                    with col1:
                        st.write("**Missing Values**")
                        missing = df.isnull().sum()
                        missing_pct = (missing / len(df) * 100).round(2)
                        missing_df = pd.DataFrame({
                            'Column': missing.index,
                            'Missing': missing.values,
                            'Percentage': missing_pct.values
                        })
                        missing_df = missing_df[missing_df['Missing'] > 0]
                        if not missing_df.empty:
                            st.dataframe(missing_df, use_container_width=True)
                        else:
                            st.success("No missing values detected")
                    
                    with col2:
                        st.write("**Data Types**")
                        dtypes_df = pd.DataFrame({
                            'Column': df.dtypes.index,
                            'Type': df.dtypes.values.astype(str)
                        })
                        st.dataframe(dtypes_df, use_container_width=True, height=300)
                    
                    st.divider()
                    
                    st.subheader("Top Events by Category")
                    top_events = df.groupby(['category', 'name']).size().reset_index(name='count')
                    top_events = top_events.sort_values('count', ascending=False).head(20)
                    st.dataframe(top_events, use_container_width=True)
    
        else:
            # Error handling for JSON data loading
            st.error("⚠️ Unable to load JSON data sources. Please check the data files.")
            
        st.markdown("""
        ### JSON Data Integration System
        
        This dashboard now uses integrated JSON data sources for comprehensive user analytics:
        
        **Required JSON Files:**
        - `offline_data/app_event.json` - App events with user IDs for complete tracking
        - `offline_data/adapty_event.json` - Revenue events from Adapty platform
        - `offline_data/revenue.json` - Customer revenue data with user correlation
        
        ### Features
        
        - **Automatic Data Correlation**: Revenue events linked to user data
        - **User Revenue Tracking**: Using customer_userid and adapty profile ID
        - **Complete Journey Analysis**: App events provide full user behavior tracking
        - **Integrated Analytics**: All data sources work together seamlessly
        
        ### Data Integration Benefits
        
        - **Revenue Attribution**: Properly attribute revenue to specific users
        - **Complete User Profiles**: Combine app behavior with revenue data
        - **Advanced Analytics**: Cross-reference user actions with monetization
        - **Real-time Processing**: JSON data loads faster than CSV uploads
        """)
        
        st.divider()
        
        col1, col2 = st.columns(2)
        
        with col1:
            if st.button("🔄 Retry Loading Data", use_container_width=True):
                st.cache_data.clear()
                st.rerun()
        
        with col2:
            st.info("💡 Ensure all JSON files are in the offline_data directory")
        
        st.subheader("Expected JSON Data Sources")
        
        data_sources = pd.DataFrame({
            'File': ['app_event.json', 'adapty_event.json', 'revenue.json'],
            'Purpose': ['User app interactions', 'Revenue events', 'Customer-revenue mapping'],
            'Key Fields': ['userid, name, datetimeutc', 'event data, timestamps', 'customer_userid, adapty_profile_id'],
            'Status': ['❌ Not loaded', '❌ Not loaded', '❌ Not loaded']
        })
        
        st.dataframe(data_sources, use_container_width=True)

if __name__ == "__main__":
    # Check authentication before showing dashboard
    if check_authentication():
        # Add logout button in sidebar
        with st.sidebar:
            st.divider()
            if st.button("🚪 Logout", use_container_width=True):
                logout()
            
            # Show logged in user
            if 'user_email' in st.session_state and st.session_state.user_email:
                st.success(f"👤 Logged in as: {st.session_state.user_email}")
        
        # Run main dashboard
        main()