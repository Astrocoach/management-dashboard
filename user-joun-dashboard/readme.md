# User Analytics Dashboard 📊

A comprehensive Streamlit-based analytics dashboard for tracking user behavior, engagement, and monetization metrics. This enterprise-level tracking system provides deep insights into user journeys, feature usage, segmentation, and revenue analytics.

## Features

### 🎯 Core Capabilities

- **Executive Dashboard**: Real-time KPIs and trend analysis
- **User Journey Analysis**: Interactive Sankey diagrams and conversion funnels
- **Feature Analytics**: Usage patterns and engagement metrics
- **Monetization Insights**: Revenue tracking and product performance
- **User Segmentation**: ML-powered clustering and behavior patterns
- **Individual User Explorer**: Deep dive into specific user behaviors
- **Advanced Analytics**: Anomaly detection and churn prediction
- **Data Export**: Download processed data and comprehensive reports

### 🔥 Key Highlights

- **Smart Session Reconstruction**: Automatically groups events into sessions based on configurable timeouts
- **Machine Learning Integration**: K-Means clustering for user segmentation and Isolation Forest for anomaly detection
- **Interactive Visualizations**: Plotly-powered charts with drill-down capabilities
- **Cohort Analysis**: Track user retention across weekly cohorts
- **Churn Risk Prediction**: Identify at-risk users before they leave
- **Multi-Platform Support**: iOS and Android tracking

## Installation

### Prerequisites

- Python 3.8 or higher
- pip package manager

### Setup

1. **Clone or download the repository**

```bash
cd management-dashboard
```

2. **Install dependencies**

```bash
pip install -r requirements.txt
```

3. **Run the application**

```bash
streamlit run main.py
```

4. **Access the dashboard**

Open your browser and navigate to `http://localhost:8501`

## Usage

### Data Upload

1. **Prepare Your CSV Files**:
   - **App Events CSV** (Required): Contains user interaction events
   - **Payment Events CSV** (Optional): Contains transaction data

2. **Required Columns** for App Events:
   - `analyticsid`: Unique event identifier
   - `userid`: User identifier
   - `deviceid`: Device identifier (should contain 'ios' or 'android')
   - `category`: Event category (e.g., 'app_event', 'adapty_event')
   - `name`: Event name (e.g., 'open_HomeScreen', 'click_Button')
   - `datetimeutc`: Timestamp in UTC format

3. **Optional Columns** for Payment Events:
   - `analyticsdata`: JSON string containing payment details

### CSV Format Example

```csv
analyticsid,userid,deviceid,category,name,datetimeutc
1,12345,device-ios-123,app_event,open_SplashScreen,2025-09-01 10:00:00
2,12345,device-ios-123,app_event,open_HomeScreen,2025-09-01 10:00:15
3,12345,device-ios-123,app_event,click_Feature_Button,2025-09-01 10:01:30
```

### Dashboard Navigation

#### 📈 Overview Tab
- Daily active users trend
- Platform distribution (iOS vs Android)
- Hourly and daily activity patterns
- Key performance indicators

#### 🛤️ User Journey Tab
- **Improved Sankey Diagram**: Clean, readable user flow visualization
  - Shows top screens and major user paths
  - Configurable filters for screen count and minimum users
  - Simplified screen names for clarity
- **Conversion Funnel**: Track users through key milestones
- **Common Event Sequences**: Identify popular user patterns

#### ⚡ Features Tab
- Top features by usage and unique users
- Feature engagement heatmap by time and day
- Identify popular and underutilized features

#### 💰 Monetization Tab
- Total revenue and transaction metrics
- Revenue breakdown by product and region
- Daily revenue trends
- Average transaction value

#### 👥 Segmentation Tab
- Automated user clustering (Power Users, Converters, Explorers, Churners)
- Segment characteristics and distribution
- Visual clustering representation

#### 🔍 User Explorer Tab
- Search for individual users by ID
- View complete user timeline
- Feature usage breakdown
- Payment history

#### 🧠 Advanced Analytics Tab
- **Anomaly Detection**: Identify unusual user behavior
- **Churn Risk Analysis**: Predict at-risk users
- **Cohort Retention**: Track user retention over time

#### 📥 Export & Data Tab
- Download processed data as CSV
- View raw data and data quality reports
- Access comprehensive data summaries

## Configuration Options

### Session Settings
- **Session Timeout**: Adjust the time gap (10-60 minutes) that defines a new session
- Default: 30 minutes

### User Flow Settings
- **Top Screens**: Control how many screens appear in the Sankey diagram (5-15)
- **Min Users per Flow**: Set minimum users threshold for displayed paths (10-200)

### Date Range Filters
- Last 7 Days
- Last 30 Days
- Last 90 Days
- All Time

## Key Fixes in This Version

### 1. ✅ Timezone Issue Resolved
**Problem**: `TypeError: Cannot subtract tz-naive and tz-aware datetime-like objects`

**Solution**: All datetime objects are now converted to timezone-naive format using:
```python
df['datetimeutc'] = pd.to_datetime(df['datetimeutc'], errors='coerce', utc=True).dt.tz_localize(None)
current_time = pd.Timestamp.now().tz_localize(None)
```

### 2. ✅ Improved Sankey Diagram
**Problems**: 
- Too cluttered and hard to read
- Overlapping labels
- Too many flows

**Solutions**:
- Limited to top N screens (configurable, default 10)
- Shows only first 3 events per user for clarity
- Filters out low-volume transitions
- Simplified screen names (removes 'open_', 'Screen', 'click_' prefixes)
- Added user-configurable parameters in sidebar
- Better color scheme and hover information
- Larger, more readable diagram (700px height)

### 3. ✅ Enhanced Error Handling
- Graceful handling of missing columns
- Better error messages
- Fallback options for empty datasets

## Technical Architecture

### Data Processing Pipeline

```
CSV Upload → Data Validation → Datetime Normalization
           ↓
Session Reconstruction → Event Grouping → Metrics Calculation
           ↓
Machine Learning → Segmentation & Anomaly Detection
           ↓
Visualization → Interactive Dashboard
```

### Machine Learning Components

1. **K-Means Clustering**: Segments users into 4 groups based on:
   - Total sessions
   - Average session duration
   - Total events
   - Days active

2. **Isolation Forest**: Detects anomalous user behavior based on:
   - Event frequency
   - Session patterns

### Performance Optimizations

- **Caching**: `@st.cache_data` decorator for expensive operations
- **Lazy Loading**: Data processed only when needed
- **Efficient Aggregations**: Pandas groupby operations
- **Limited Data Display**: Top N results and pagination

## Troubleshooting

### Common Issues

**Issue**: Dashboard doesn't load data
- **Solution**: Check CSV file format matches requirements
- Ensure datetime column exists and is properly formatted

**Issue**: Sankey diagram is empty
- **Solution**: Lower the "Min Users per Flow" threshold in sidebar
- Ensure you have enough data (at least 100 events)

**Issue**: Memory errors with large files
- **Solution**: Process data in chunks or filter by date range
- Consider sampling large datasets

**Issue**: Segmentation not working
- **Solution**: Ensure you have at least 20 users
- Check that all required metrics columns exist

## Best Practices

1. **Data Preparation**:
   - Clean your data before upload
   - Ensure consistent datetime formats
   - Remove duplicate events

2. **Performance**:
   - Start with smaller date ranges for initial exploration
   - Use filters to focus on specific user segments
   - Export large datasets for offline analysis

3. **Analysis**:
   - Compare metrics across different time periods
   - Segment users before analyzing behavior
   - Cross-reference anomalies with external events

## Roadmap

- [ ] Real-time data streaming support
- [ ] Automated report generation
- [ ] Custom event funnel builder
- [ ] A/B test analysis module
- [ ] Email alerts for churn risks
- [ ] Database integration (PostgreSQL, MongoDB)
- [ ] Multi-tenant support
- [ ] Custom dashboard themes

## Dependencies

- **streamlit**: Web application framework
- **pandas**: Data manipulation and analysis
- **numpy**: Numerical computing
- **plotly**: Interactive visualizations
- **scikit-learn**: Machine learning algorithms
- **openpyxl**: Excel file support

## Contributing

Contributions are welcome! Please follow these guidelines:
1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Test thoroughly
5. Submit a pull request

## License

This project is available for use and modification as needed for your analytics requirements.

## Support

For issues, questions, or feature requests:
- Check the Troubleshooting section
- Review the code comments
- Test with sample data first

## Version History

### v1.0.0 
- ✅ Fixed timezone compatibility issues
- ✅ Completely redesigned Sankey diagram
- ✅ Added configurable user flow parameters
- ✅ Improved error handling
- ✅ Enhanced documentation

### v1.5.0 (Current)
- ✅ **Google Analytics Style Funnels**: Fully customizable multi-step conversion funnels with drop-off visualization
- ✅ **User Authentication**: Secure login system with session management and role-based access
- ✅ **Advanced Funnel Analytics**: 
  - Time-to-convert metrics between funnel steps
  - Segment-based funnel comparison (by platform, country, user type)
  - Cohort-based funnel tracking over time
  - Export funnel data with detailed drop-off reasons
- ✅ **Enhanced User Journey**: 
  - Loop detection and removal in user flows
  - Improved Sankey diagram with conversion percentages
  - Top conversion paths identification
- ✅ **Improved Performance**: 
  - Lazy loading for large datasets
  - Optimized memory usage for 1M+ events
  - Background data processing
- ✅ **Integration Ready**: 
  - REST API endpoints for external data sources
  - Webhook support for real-time event streaming



---

**Built with ❤️ using Streamlit and Python**