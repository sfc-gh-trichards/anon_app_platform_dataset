import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import numpy as np
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

# Set page config
st.set_page_config(
    page_title="Anonymous Dataset Analytics Dashboard",
    layout="wide"
)

@st.cache_data
def load_data():
    """Load all CSV files with caching for better performance"""
    try:
        # Load company day fact data
        company_df = pd.read_csv('anon_company_day_fact.csv')
        company_df['view_date'] = pd.to_datetime(company_df['view_date'])
        
        # Load user day fact data
        user_df = pd.read_csv('anon_user_day_fact.csv')
        user_df['view_date'] = pd.to_datetime(user_df['view_date'])
        
        # Load views data
        views_df = pd.read_csv('anon_views.csv')
        views_df['view_date'] = pd.to_datetime(views_df['view_date'])
        views_df['view_time'] = pd.to_datetime(views_df['view_time'])
        
        return company_df, user_df, views_df
    except Exception as e:
        st.error(f"Error loading data: {e}")
        return None, None, None

def display_summary_metrics(company_df, user_df, views_df):
    """Display high-level summary metrics"""
    st.title("Anonymous Dataset Analytics Dashboard")
    
    # Create metrics columns
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric(
            label="Total Company Records",
            value=f"{len(company_df):,}",
            delta=f"{company_df['company_id'].nunique():,} unique companies"
        )
    
    with col2:
        st.metric(
            label="Total User Records",
            value=f"{len(user_df):,}",
            delta=f"{user_df['user_id'].nunique():,} unique users"
        )
    
    with col3:
        st.metric(
            label="Total Views",
            value=f"{len(views_df):,}",
            delta=f"{views_df['app_id'].nunique():,} unique apps"
        )
    
    with col4:
        total_spent = company_df['total_amount_spent'].sum()
        st.metric(
            label="Total Amount Spent",
            value=f"${total_spent:,.2f}",
            delta=f"${company_df['total_amount_spent'].mean():.2f} avg per record"
        )

def display_company_analytics(company_df):
    """Display company day fact analytics"""
    st.header("Company Day Fact Analytics")
    
    # Summary statistics
    col1, col2 = st.columns([2, 1])
    
    with col1:
        # Time series of total amount spent
        daily_spend = company_df.groupby('view_date')['total_amount_spent'].sum().reset_index()
        fig_spend = px.line(daily_spend, x='view_date', y='total_amount_spent',
                           title='Daily Total Amount Spent Over Time',
                           labels={'total_amount_spent': 'Total Amount Spent ($)', 'view_date': 'Date'})
        fig_spend.update_layout(height=400)
        st.plotly_chart(fig_spend, use_container_width=True)
    
    with col2:
        # Summary stats
        st.markdown("### Summary Statistics")
        stats_data = {
            'Metric': ['Total Records', 'Unique Companies', 'Date Range', 'Avg Daily Spend', 'Total Spend'],
            'Value': [
                f"{len(company_df):,}",
                f"{company_df['company_id'].nunique():,}",
                f"{company_df['view_date'].min().strftime('%Y-%m-%d')} to {company_df['view_date'].max().strftime('%Y-%m-%d')}",
                f"${company_df['total_amount_spent'].mean():.2f}",
                f"${company_df['total_amount_spent'].sum():,.2f}"
            ]
        }
        stats_df = pd.DataFrame(stats_data)
        st.dataframe(stats_df, hide_index=True)
    
    # Distribution charts
    col1, col2 = st.columns(2)
    
    with col1:
        # Distribution of total amount spent
        fig_spend_dist = px.histogram(company_df, x='total_amount_spent', nbins=50,
                                     title='Distribution of Total Amount Spent',
                                     labels={'total_amount_spent': 'Amount Spent ($)', 'count': 'Frequency'})
        fig_spend_dist.update_layout(height=400)
        st.plotly_chart(fig_spend_dist, use_container_width=True)
    
    with col2:
        # Distribution of view count
        fig_view_dist = px.histogram(company_df, x='view_count', nbins=30,
                                    title='Distribution of View Count',
                                    labels={'view_count': 'View Count', 'count': 'Frequency'})
        fig_view_dist.update_layout(height=400)
        st.plotly_chart(fig_view_dist, use_container_width=True)
    
    # Correlation heatmap
    numeric_cols = ['total_amount_spent', 'total_view_time', 'app_count', 'view_count', 'user_count']
    correlation_matrix = company_df[numeric_cols].corr()
    
    fig_corr = px.imshow(correlation_matrix, 
                        title='Correlation Matrix - Company Metrics',
                        color_continuous_scale='RdBu',
                        aspect='auto')
    fig_corr.update_layout(height=400)
    st.plotly_chart(fig_corr, use_container_width=True)

def display_user_analytics(user_df):
    """Display user day fact analytics"""
    st.header("User Day Fact Analytics")
    
    # Summary statistics
    col1, col2 = st.columns([2, 1])
    
    with col1:
        # Time series of user activity
        daily_users = user_df.groupby('view_date')['user_id'].nunique().reset_index()
        fig_users = px.line(daily_users, x='view_date', y='user_id',
                           title='Daily Active Users Over Time',
                           labels={'user_id': 'Active Users', 'view_date': 'Date'})
        fig_users.update_layout(height=400)
        st.plotly_chart(fig_users, use_container_width=True)
    
    with col2:
        # Summary stats
        st.markdown("### Summary Statistics")
        stats_data = {
            'Metric': ['Total Records', 'Unique Users', 'Unique Companies', 'Date Range', 'Avg User Spend'],
            'Value': [
                f"{len(user_df):,}",
                f"{user_df['user_id'].nunique():,}",
                f"{user_df['company_id'].nunique():,}",
                f"{user_df['view_date'].min().strftime('%Y-%m-%d')} to {user_df['view_date'].max().strftime('%Y-%m-%d')}",
                f"${user_df['total_amount_spent'].mean():.4f}"
            ]
        }
        stats_df = pd.DataFrame(stats_data)
        st.dataframe(stats_df, hide_index=True)
    
    # Distribution charts
    col1, col2 = st.columns(2)
    
    with col1:
        # Distribution of total view time
        fig_time_dist = px.histogram(user_df, x='total_view_time', nbins=50,
                                    title='Distribution of Total View Time',
                                    labels={'total_view_time': 'View Time (seconds)', 'count': 'Frequency'})
        fig_time_dist.update_layout(height=400)
        st.plotly_chart(fig_time_dist, use_container_width=True)
    
    with col2:
        # Distribution of app count
        fig_app_dist = px.histogram(user_df, x='app_count', nbins=20,
                                   title='Distribution of App Count',
                                   labels={'app_count': 'Number of Apps', 'count': 'Frequency'})
        fig_app_dist.update_layout(height=400)
        st.plotly_chart(fig_app_dist, use_container_width=True)
    
    # Top users by activity
    st.markdown("### Top 10 Most Active Users")
    top_users = user_df.groupby('user_id').agg({
        'total_amount_spent': 'sum',
        'total_view_time': 'sum',
        'view_count': 'sum',
        'app_count': 'sum'
    }).sort_values('total_view_time', ascending=False).head(10)
    
    fig_top_users = px.bar(top_users, y='total_view_time', 
                          title='Top 10 Users by Total View Time',
                          labels={'total_view_time': 'Total View Time (seconds)', 'user_id': 'User ID'})
    fig_top_users.update_layout(height=400)
    st.plotly_chart(fig_top_users, use_container_width=True)

def display_views_analytics(views_df):
    """Display views analytics"""
    st.header("Views Analytics")
    
    # Summary statistics
    col1, col2 = st.columns([2, 1])
    
    with col1:
        # Time series of daily views
        daily_views = views_df.groupby('view_date').size().reset_index(name='view_count')
        fig_views = px.line(daily_views, x='view_date', y='view_count',
                           title='Daily View Count Over Time',
                           labels={'view_count': 'Number of Views', 'view_date': 'Date'})
        fig_views.update_layout(height=400)
        st.plotly_chart(fig_views, use_container_width=True)
    
    with col2:
        # Summary stats
        st.markdown("### Summary Statistics")
        stats_data = {
            'Metric': ['Total Views', 'Unique Users', 'Unique Apps', 'Unique Companies', 'Date Range'],
            'Value': [
                f"{len(views_df):,}",
                f"{views_df['user_id'].nunique():,}",
                f"{views_df['app_id'].nunique():,}",
                f"{views_df['company_id'].nunique():,}",
                f"{views_df['view_date'].min().strftime('%Y-%m-%d')} to {views_df['view_date'].max().strftime('%Y-%m-%d')}"
            ]
        }
        stats_df = pd.DataFrame(stats_data)
        st.dataframe(stats_df, hide_index=True)
    
    # Distribution charts
    col1, col2 = st.columns(2)
    
    with col1:
        # Distribution of total view time
        fig_view_time_dist = px.histogram(views_df, x='total_view_time', nbins=50,
                                         title='Distribution of View Time per Session',
                                         labels={'total_view_time': 'View Time (seconds)', 'count': 'Frequency'})
        fig_view_time_dist.update_layout(height=400)
        st.plotly_chart(fig_view_time_dist, use_container_width=True)
    
    with col2:
        # Distribution of amount spent
        fig_amount_dist = px.histogram(views_df, x='total_amount_spent', nbins=50,
                                      title='Distribution of Amount Spent per View',
                                      labels={'total_amount_spent': 'Amount Spent ($)', 'count': 'Frequency'})
        fig_amount_dist.update_layout(height=400)
        st.plotly_chart(fig_amount_dist, use_container_width=True)
    
    # Hourly activity pattern
    views_df['hour'] = views_df['view_time'].dt.hour
    hourly_activity = views_df['hour'].value_counts().sort_index()
    
    fig_hourly = px.bar(x=hourly_activity.index, y=hourly_activity.values,
                       title='Hourly Activity Pattern',
                       labels={'x': 'Hour of Day', 'y': 'Number of Views'})
    fig_hourly.update_layout(height=400)
    st.plotly_chart(fig_hourly, use_container_width=True)

def main():
    """Main function to run the Streamlit app"""
    st.sidebar.title("Dashboard Navigation")
    
    # Load data
    with st.spinner("Loading data..."):
        company_df, user_df, views_df = load_data()
    
    if company_df is None or user_df is None or views_df is None:
        st.error("Failed to load data. Please check your CSV files.")
        return
    
    # Navigation
    page = st.sidebar.selectbox(
        "Choose a section:",
        ["Overview", "Company Analytics", "User Analytics", "Views Analytics"]
    )
    
    if page == "Overview":
        display_summary_metrics(company_df, user_df, views_df)
        
        # Quick insights
        st.header("Quick Insights")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("### Data Overview")
            overview_data = {
                'Dataset': ['Company Day Fact', 'User Day Fact', 'Views'],
                'Records': [f"{len(company_df):,}", f"{len(user_df):,}", f"{len(views_df):,}"],
                'Date Range': [
                    f"{company_df['view_date'].min().strftime('%Y-%m-%d')} to {company_df['view_date'].max().strftime('%Y-%m-%d')}",
                    f"{user_df['view_date'].min().strftime('%Y-%m-%d')} to {user_df['view_date'].max().strftime('%Y-%m-%d')}",
                    f"{views_df['view_date'].min().strftime('%Y-%m-%d')} to {views_df['view_date'].max().strftime('%Y-%m-%d')}"
                ]
            }
            overview_df = pd.DataFrame(overview_data)
            st.dataframe(overview_df, hide_index=True)
        
        with col2:
            st.markdown("### Key Metrics")
            key_metrics = {
                'Metric': ['Total Amount Spent', 'Total View Time (hours)', 'Unique Companies', 'Unique Users', 'Unique Apps'],
                'Value': [
                    f"${company_df['total_amount_spent'].sum():,.2f}",
                    f"{views_df['total_view_time'].sum() / 3600:,.1f}",
                    f"{company_df['company_id'].nunique():,}",
                    f"{user_df['user_id'].nunique():,}",
                    f"{views_df['app_id'].nunique():,}"
                ]
            }
            metrics_df = pd.DataFrame(key_metrics)
            st.dataframe(metrics_df, hide_index=True)
    
    elif page == "Company Analytics":
        display_company_analytics(company_df)
    
    elif page == "User Analytics":
        display_user_analytics(user_df)
    
    elif page == "Views Analytics":
        display_views_analytics(views_df)
    
    # Footer
    st.markdown("---")
    st.markdown("Dashboard created with Streamlit and Plotly Express")

if __name__ == "__main__":
    main()
