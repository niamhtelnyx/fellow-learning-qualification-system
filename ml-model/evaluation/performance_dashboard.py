#!/usr/bin/env python3
"""
Performance Monitoring Dashboard for Fellow Learning Qualification System
Streamlit dashboard to monitor model performance, data insights, and learning trends
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import json
import sqlite3
from datetime import datetime, timedelta
from pathlib import Path
import logging

# Configure page
st.set_page_config(
    page_title="Fellow Learning Dashboard",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded"
)

logger = logging.getLogger(__name__)

class DashboardData:
    """Data loader and processor for dashboard"""
    
    def __init__(self):
        self.base_dir = Path(__file__).parent.parent
        self.db_path = self.base_dir / "data" / "fellow_data.db"
        self.models_dir = self.base_dir / "ml-model" / "models"
        self.enrichment_file = self.base_dir.parent / "fellow-enrichment-progress.csv"
        
    def load_fellow_data(self, days_back: int = 30) -> pd.DataFrame:
        """Load Fellow call data"""
        if not self.db_path.exists():
            return pd.DataFrame()
        
        try:
            conn = sqlite3.connect(self.db_path)
            cutoff_date = (datetime.now() - timedelta(days=days_back)).strftime('%Y-%m-%d')
            
            query = """
            SELECT * FROM meetings 
            WHERE date >= ? 
            ORDER BY date DESC
            """
            
            df = pd.read_sql_query(query, conn, params=(cutoff_date,))
            conn.close()
            
            if not df.empty:
                df['date'] = pd.to_datetime(df['date'])
            
            return df
            
        except Exception as e:
            st.error(f"Error loading Fellow data: {e}")
            return pd.DataFrame()
    
    def load_enrichment_data(self) -> pd.DataFrame:
        """Load company enrichment data"""
        if self.enrichment_file.exists():
            try:
                return pd.read_csv(self.enrichment_file)
            except Exception as e:
                st.error(f"Error loading enrichment data: {e}")
        return pd.DataFrame()
    
    def load_model_performance_history(self) -> Dict:
        """Load model performance history"""
        log_file = self.models_dir / "continuous_learning_log.json"
        
        if log_file.exists():
            try:
                with open(log_file, 'r') as f:
                    return json.load(f)
            except Exception as e:
                st.error(f"Error loading performance history: {e}")
        
        return {'cycles': []}
    
    def get_model_versions(self) -> List[str]:
        """Get list of available model versions"""
        if not self.models_dir.exists():
            return []
        
        versions = []
        for version_dir in self.models_dir.iterdir():
            if version_dir.is_dir() and version_dir.name.startswith('v_'):
                versions.append(version_dir.name.replace('v_', ''))
        
        return sorted(versions, reverse=True)
    
    def load_model_metadata(self, version: str) -> Dict:
        """Load model metadata for specific version"""
        version_dir = self.models_dir / f"v_{version}"
        metadata_file = version_dir / "metadata.json"
        
        if metadata_file.exists():
            try:
                with open(metadata_file, 'r') as f:
                    return json.load(f)
            except Exception as e:
                st.error(f"Error loading model metadata: {e}")
        
        return {}

@st.cache_data(ttl=300)  # Cache for 5 minutes
def load_dashboard_data(days_back: int = 30):
    """Load all dashboard data with caching"""
    data_loader = DashboardData()
    
    return {
        'fellow_data': data_loader.load_fellow_data(days_back),
        'enrichment_data': data_loader.load_enrichment_data(),
        'performance_history': data_loader.load_model_performance_history(),
        'model_versions': data_loader.get_model_versions()
    }

def display_overview_metrics(fellow_data: pd.DataFrame, enrichment_data: pd.DataFrame):
    """Display overview metrics"""
    
    col1, col2, col3, col4 = st.columns(4)
    
    # Fellow calls metrics
    with col1:
        st.metric(
            label="📞 Total Calls",
            value=len(fellow_data),
            delta=f"{len(fellow_data[fellow_data['date'] >= datetime.now() - timedelta(days=7)])} this week"
        )
    
    # Average sentiment score
    with col2:
        avg_sentiment = fellow_data['sentiment_score'].mean() if not fellow_data.empty else 0
        st.metric(
            label="⭐ Avg Sentiment",
            value=f"{avg_sentiment:.1f}",
            delta=f"Target: 7.0+"
        )
    
    # High-value prospects
    with col3:
        high_value = len(fellow_data[fellow_data['sentiment_score'] >= 7]) if not fellow_data.empty else 0
        high_value_pct = (high_value / len(fellow_data) * 100) if not fellow_data.empty else 0
        st.metric(
            label="🎯 High Value",
            value=f"{high_value} ({high_value_pct:.1f}%)",
            delta="Score ≥ 7"
        )
    
    # Companies enriched
    with col4:
        enriched_count = len(enrichment_data)
        st.metric(
            label="🏢 Enriched Companies",
            value=enriched_count,
            delta=f"Ready for ML"
        )

def display_call_trends(fellow_data: pd.DataFrame):
    """Display call volume and sentiment trends"""
    
    if fellow_data.empty:
        st.warning("No Fellow call data available")
        return
    
    st.subheader("📈 Call Volume & Sentiment Trends")
    
    # Prepare daily aggregated data
    daily_stats = fellow_data.groupby(fellow_data['date'].dt.date).agg({
        'id': 'count',
        'sentiment_score': ['mean', 'std'],
        'follow_up_scheduled': 'sum'
    }).reset_index()
    
    daily_stats.columns = ['date', 'call_count', 'avg_sentiment', 'sentiment_std', 'follow_ups']
    daily_stats['date'] = pd.to_datetime(daily_stats['date'])
    
    # Create subplot figure
    fig = make_subplots(
        rows=2, cols=1,
        subplot_titles=("Daily Call Volume", "Average Sentiment Score"),
        vertical_spacing=0.1
    )
    
    # Call volume
    fig.add_trace(
        go.Scatter(
            x=daily_stats['date'],
            y=daily_stats['call_count'],
            mode='lines+markers',
            name='Calls per Day',
            line=dict(color='#1f77b4')
        ),
        row=1, col=1
    )
    
    # Sentiment trend
    fig.add_trace(
        go.Scatter(
            x=daily_stats['date'],
            y=daily_stats['avg_sentiment'],
            mode='lines+markers',
            name='Avg Sentiment',
            line=dict(color='#ff7f0e')
        ),
        row=2, col=1
    )
    
    # Add target line for sentiment
    fig.add_hline(y=7, line_dash="dash", line_color="red", 
                  annotation_text="Target", row=2, col=1)
    
    fig.update_layout(
        height=500,
        showlegend=True,
        title_text="Call Activity Trends"
    )
    
    st.plotly_chart(fig, use_container_width=True)

def display_qualification_analysis(fellow_data: pd.DataFrame, enrichment_data: pd.DataFrame):
    """Display qualification score analysis"""
    
    st.subheader("🎯 Qualification Analysis")
    
    if enrichment_data.empty:
        st.warning("No enrichment data available for qualification analysis")
        return
    
    col1, col2 = st.columns([2, 1])
    
    with col1:
        # Score distribution
        fig = px.histogram(
            enrichment_data,
            x='Score',
            nbins=20,
            title="Qualification Score Distribution",
            labels={'Score': 'Qualification Score', 'count': 'Number of Companies'}
        )
        
        # Add routing thresholds
        fig.add_vline(x=85, line_dash="dash", line_color="green", 
                     annotation_text="AE Handoff (85+)")
        fig.add_vline(x=70, line_dash="dash", line_color="orange", 
                     annotation_text="Nurture (70+)")
        
        st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        # Routing breakdown
        routing_counts = enrichment_data['Routing'].value_counts()
        
        fig = px.pie(
            values=routing_counts.values,
            names=routing_counts.index,
            title="Routing Breakdown"
        )
        
        st.plotly_chart(fig, use_container_width=True)
    
    # Top scoring companies
    st.subheader("🏆 Top Scoring Companies")
    
    top_companies = enrichment_data.nlargest(10, 'Score')[
        ['Company', 'Industry', 'Score', 'AI Signals', 'Routing']
    ]
    
    st.dataframe(top_companies, use_container_width=True)

def display_model_performance(performance_history: Dict):
    """Display model performance metrics over time"""
    
    st.subheader("🤖 Model Performance Tracking")
    
    cycles = performance_history.get('cycles', [])
    
    if not cycles:
        st.warning("No model performance history available")
        return
    
    # Extract performance data
    performance_data = []
    for cycle in cycles:
        if cycle.get('performance_after'):
            perf = cycle['performance_after']
            performance_data.append({
                'cycle_date': pd.to_datetime(cycle['cycle_start']).date(),
                'accuracy': perf.get('accuracy', 0),
                'precision': perf.get('precision', 0),
                'recall': perf.get('recall', 0),
                'f1_score': perf.get('f1_score', 0),
                'auc_roc': perf.get('auc_roc', 0),
                'model_version': perf.get('model_version', 'unknown'),
                'test_samples': perf.get('test_samples', 0)
            })
    
    if not performance_data:
        st.warning("No model performance data available")
        return
    
    df = pd.DataFrame(performance_data)
    
    # Performance metrics over time
    fig = px.line(
        df,
        x='cycle_date',
        y=['accuracy', 'precision', 'recall', 'f1_score', 'auc_roc'],
        title="Model Performance Over Time",
        labels={'value': 'Score', 'cycle_date': 'Date', 'variable': 'Metric'}
    )
    
    # Add target lines
    fig.add_hline(y=0.85, line_dash="dash", line_color="green", 
                 annotation_text="Target: 85%")
    fig.add_hline(y=0.75, line_dash="dash", line_color="orange", 
                 annotation_text="Warning: 75%")
    
    st.plotly_chart(fig, use_container_width=True)
    
    # Latest performance summary
    if performance_data:
        latest = performance_data[-1]
        
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.metric(
                "Accuracy",
                f"{latest['accuracy']:.3f}",
                delta=f"Target: 0.850"
            )
        
        with col2:
            st.metric(
                "Precision",
                f"{latest['precision']:.3f}",
                delta=f"Voice AI Focus"
            )
        
        with col3:
            st.metric(
                "Recall",
                f"{latest['recall']:.3f}",
                delta=f"Coverage"
            )
        
        with col4:
            st.metric(
                "Test Samples",
                latest['test_samples'],
                delta=f"Model: {latest['model_version']}"
            )

def display_learning_insights(fellow_data: pd.DataFrame, enrichment_data: pd.DataFrame):
    """Display insights and recommendations"""
    
    st.subheader("💡 Learning Insights & Recommendations")
    
    insights = []
    
    # Analyze call patterns
    if not fellow_data.empty:
        high_sentiment_calls = fellow_data[fellow_data['sentiment_score'] >= 8]
        high_sentiment_rate = len(high_sentiment_calls) / len(fellow_data)
        
        if high_sentiment_rate < 0.2:
            insights.append({
                'type': 'warning',
                'title': 'Low High-Value Call Rate',
                'message': f"Only {high_sentiment_rate:.1%} of calls are high sentiment (8+). Consider improving lead qualification."
            })
        
        # Follow-up rate analysis
        follow_up_rate = fellow_data['follow_up_scheduled'].mean() if 'follow_up_scheduled' in fellow_data.columns else 0
        if follow_up_rate < 0.4:
            insights.append({
                'type': 'info',
                'title': 'Follow-up Opportunity',
                'message': f"Follow-up rate is {follow_up_rate:.1%}. Consider training on converting more calls to next steps."
            })
    
    # Analyze enrichment patterns
    if not enrichment_data.empty:
        voice_ai_companies = enrichment_data[enrichment_data['AI Signals'].str.contains('Voice AI', na=False)]
        voice_ai_rate = len(voice_ai_companies) / len(enrichment_data)
        
        if voice_ai_rate > 0.3:
            insights.append({
                'type': 'success',
                'title': 'Strong Voice AI Pipeline',
                'message': f"{voice_ai_rate:.1%} of enriched companies show Voice AI signals. Good market targeting."
            })
        
        # High-scoring companies
        high_score_rate = len(enrichment_data[enrichment_data['Score'] >= 85]) / len(enrichment_data)
        if high_score_rate < 0.2:
            insights.append({
                'type': 'warning',
                'title': 'Low Qualification Rate',
                'message': f"Only {high_score_rate:.1%} score 85+. Model may need retraining with more positive examples."
            })
    
    # Model recommendations
    insights.append({
        'type': 'info',
        'title': 'Model Training Recommendation',
        'message': "Retrain models weekly with new Fellow outcomes to improve accuracy and adapt to market changes."
    })
    
    # Display insights
    for insight in insights:
        if insight['type'] == 'success':
            st.success(f"✅ **{insight['title']}**: {insight['message']}")
        elif insight['type'] == 'warning':
            st.warning(f"⚠️ **{insight['title']}**: {insight['message']}")
        else:
            st.info(f"ℹ️ **{insight['title']}**: {insight['message']}")

def main():
    """Main dashboard application"""
    
    # Dashboard header
    st.title("🤖 Fellow Learning Qualification Dashboard")
    st.markdown("Monitor ML model performance and qualification insights")
    
    # Sidebar controls
    with st.sidebar:
        st.header("⚙️ Controls")
        
        # Data refresh
        if st.button("🔄 Refresh Data"):
            st.cache_data.clear()
            st.rerun()
        
        # Date range selection
        days_back = st.selectbox(
            "📅 Data Range",
            options=[7, 14, 30, 60, 90],
            index=2,
            format_func=lambda x: f"Last {x} days"
        )
        
        # Auto-refresh toggle
        auto_refresh = st.checkbox("🔄 Auto-refresh (60s)", value=False)
        
        if auto_refresh:
            import time
            time.sleep(60)
            st.rerun()
    
    # Load data
    data = load_dashboard_data(days_back)
    
    # Display overview metrics
    display_overview_metrics(data['fellow_data'], data['enrichment_data'])
    
    # Create tabs for different views
    tab1, tab2, tab3, tab4 = st.tabs(["📊 Trends", "🎯 Qualification", "🤖 Model Performance", "💡 Insights"])
    
    with tab1:
        display_call_trends(data['fellow_data'])
    
    with tab2:
        display_qualification_analysis(data['fellow_data'], data['enrichment_data'])
    
    with tab3:
        display_model_performance(data['performance_history'])
    
    with tab4:
        display_learning_insights(data['fellow_data'], data['enrichment_data'])
    
    # Footer
    st.markdown("---")
    st.markdown(
        """
        **Fellow Learning Qualification System** | 
        Real-time ML model monitoring and insights | 
        Last updated: """ + datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    )

if __name__ == "__main__":
    main()