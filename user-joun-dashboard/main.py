import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import json
from datetime import datetime, timedelta
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import IsolationForest
import warnings
warnings.filterwarnings('ignore')

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

def create_funnel_chart(df):
    """Create conversion funnel visualization"""
    funnel_stages = [
        ('App Opened', ['open_SplashScreen', 'open_HomeScreen']),
        ('Onboarding Started', ['open_WizardScreen', 'click_Onboarding_Start']),
        ('Onboarding Completed', ['onboarding_completed']),
        ('Paywall Viewed', ['open_PaywallScreen', 'paywall_viewed']),
        ('Payment Success', ['payment_success'])
    ]
    
    funnel_data = []
    for stage_name, events in funnel_stages:
        count = df[df['name'].isin(events)]['userid'].nunique()
        funnel_data.append({'Stage': stage_name, 'Users': count})
    
    funnel_df = pd.DataFrame(funnel_data)
    
    fig = go.Figure(go.Funnel(
        y=funnel_df['Stage'],
        x=funnel_df['Users'],
        textinfo="value+percent initial",
        marker=dict(color=['#667eea', '#764ba2', '#f093fb', '#4facfe', '#43e97b'])
    ))
    
    fig.update_layout(title_text="Conversion Funnel", height=500)
    return fig

# ==================== MAIN APP ====================

def main():
    st.title("📊 User Analytics Dashboard")
    st.markdown("### Comprehensive User Behavior & Revenue Analytics")
    
    # Sidebar
    with st.sidebar:
        st.header("⚙️ Configuration")
        
        # File Upload
        st.subheader("📁 Data Upload")
        app_events_file = st.file_uploader("Upload App Events CSV", type=['csv'], key='app_events')
        payment_events_file = st.file_uploader("Upload Payment Events CSV (Optional)", type=['csv'], key='payments')
        
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
        
        st.divider()
        st.info("💡 Upload your CSV files to begin analysis")
    
    # Main Content
    if app_events_file is not None:
        # Load data
        with st.spinner("Loading and processing data..."):
            df = load_and_process_csv(app_events_file)
            
            if df is not None:
                # Reconstruct sessions
                df = reconstruct_sessions(df, session_timeout)
                
                # Calculate metrics
                session_metrics = calculate_session_metrics(df)
                
                # Load payment data if available
                payment_df = pd.DataFrame()
                if payment_events_file is not None:
                    payment_raw = load_and_process_csv(payment_events_file)
                    if payment_raw is not None:
                        payment_df = parse_payment_data(payment_raw)
                
                st.success(f"✅ Loaded {len(df):,} events from {df['userid'].nunique():,} users")
                
                # Calculate KPIs
                total_users = df['userid'].nunique()
                total_sessions = df['session_id'].nunique() if 'session_id' in df.columns else 0
                avg_session_duration = session_metrics['session_duration'].mean() if not session_metrics.empty else 0
                total_revenue = payment_df['amount'].sum() if not payment_df.empty else 0
                conversion_rate = (payment_df['userid'].nunique() / total_users * 100) if not payment_df.empty else 0
                
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
                tab1, tab2, tab3, tab4, tab5, tab6, tab7, tab8 = st.tabs([
                    "📈 Overview", 
                    "🛤️ User Journey", 
                    "⚡ Features", 
                    "💰 Monetization", 
                    "👥 Segmentation",
                    "🔍 User Explorer",
                    "🧠 Advanced Analytics",
                    "📥 Export & Data"
                ])
                
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
                    funnel_fig = create_funnel_chart(df)
                    st.plotly_chart(funnel_fig, use_container_width=True)
                    
                    st.divider()
                    
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
                        st.dataframe(high_risk_users[['userid', 'total_sessions', 'total_events', 'days_since_last_seen']], 
                                   use_container_width=True)
                    
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
        # Welcome screen
        st.info("Welcome to the Analytics Dashboard! Upload your CSV files to begin analysis.")
        
        st.markdown("""
        ### Getting Started
        
        1. **Upload Data**: Use the sidebar to upload your app events CSV file
        2. **Optional**: Upload payment events CSV for monetization analytics
        3. **Configure**: Set your preferred session timeout and filters
        4. **Explore**: Navigate through the tabs to view different analytics
        
        ### Features
        
        - **Overview**: Executive dashboard with key metrics and trends
        - **User Journey**: Flow analysis and conversion funnels
        - **Features**: Feature usage and engagement analytics
        - **Monetization**: Revenue analytics and product performance
        - **Segmentation**: User clustering and behavior patterns
        - **User Explorer**: Deep dive into individual user behavior
        - **Advanced**: Anomaly detection and churn prediction
        - **Export**: Download processed data and reports
        
        ### Data Requirements
        
        Your CSV should contain these columns:
        - `userid`: User identifier
        - `datetimeutc`: Event timestamp
        - `name`: Event name
        - `category`: Event category
        - `deviceid`: Device identifier
        - `analyticsdata`: JSON data (for payment events)
        """)
        
        st.divider()
        
        st.subheader("Sample Data Format")
        
        sample_data = pd.DataFrame({
            'analyticsid': [1, 2, 3],
            'userid': [12345, 12345, 67890],
            'deviceid': ['abc123-ios', 'abc123-ios', 'def456-android'],
            'category': ['app_event', 'app_event', 'adapty_event'],
            'name': ['open_HomeScreen', 'click_Feature', 'payment_success'],
            'datetimeutc': ['2025-09-01 10:30:00', '2025-09-01 10:31:00', '2025-09-01 11:00:00']
        })
        
        st.dataframe(sample_data, use_container_width=True)

if __name__ == "__main__":
    main()