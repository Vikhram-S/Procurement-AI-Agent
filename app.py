"""
Procurement Optimization AI Agent - Streamlit Application
Main application for hospital procurement optimization using AI
"""

import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import numpy as np
from datetime import datetime, timedelta
import json
import os

# Import our custom modules
from agent.procurement_agent import ProcurementAgent
from agent.llm_interface import get_available_models
from data.sample_data import SampleDataGenerator
from data.database import ProcurementDatabase
from models.demand_forecaster import DemandForecaster
from models.cost_optimizer import CostOptimizer

# Page configuration
st.set_page_config(
    page_title="Procurement Optimization AI Agent",
    page_icon="🏥",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for beautiful UI
st.markdown("""
<style>
    .main-header {
        font-size: 3rem;
        font-weight: bold;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 2rem;
        background: linear-gradient(90deg, #1f77b4, #ff7f0e);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
    }
    
    .metric-card {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 1.5rem;
        border-radius: 10px;
        color: white;
        margin: 1rem 0;
    }
    
    .success-card {
        background: linear-gradient(135deg, #11998e 0%, #38ef7d 100%);
        padding: 1rem;
        border-radius: 10px;
        color: white;
        margin: 1rem 0;
    }
    
    .warning-card {
        background: linear-gradient(135deg, #f093fb 0%, #f5576c 100%);
        padding: 1rem;
        border-radius: 10px;
        color: white;
        margin: 1rem 0;
    }
    
    .info-card {
        background: linear-gradient(135deg, #4facfe 0%, #00f2fe 100%);
        padding: 1rem;
        border-radius: 10px;
        color: white;
        margin: 1rem 0;
    }
    
    .sidebar .sidebar-content {
        background: linear-gradient(180deg, #667eea 0%, #764ba2 100%);
    }
</style>
""", unsafe_allow_html=True)

# Initialize session state
if 'agent' not in st.session_state:
    st.session_state.agent = None
if 'data_loaded' not in st.session_state:
    st.session_state.data_loaded = False
if 'analysis_results' not in st.session_state:
    st.session_state.analysis_results = None

def main():
    """Main application function"""
    
    # Header
    st.markdown('<h1 class="main-header">🏥 Procurement Optimization AI Agent</h1>', unsafe_allow_html=True)
    st.markdown("### Intelligent AI-powered procurement optimization for healthcare institutions")
    
    # Sidebar
    with st.sidebar:
        st.markdown("## 🤖 AI Configuration")
        
        # Model selection
        model_type = st.selectbox(
            "LLM Type",
            ["ollama", "huggingface"],
            help="Choose the type of local LLM to use"
        )
        
        available_models = get_available_models()
        model_name = st.selectbox(
            "Model",
            available_models.get(model_type, ["mistral:7b"]),
            help="Select the specific model to use"
        )
        
        # Initialize agent
        if st.button("🚀 Initialize AI Agent", type="primary"):
            with st.spinner("Initializing AI Agent..."):
                try:
                    st.session_state.agent = ProcurementAgent(model_type, model_name)
                    st.success("AI Agent initialized successfully!")
                except Exception as e:
                    st.error(f"Error initializing agent: {e}")
        
        st.markdown("---")
        
        # Data management
        st.markdown("## 📊 Data Management")
        
        if st.button("📁 Load Sample Data"):
            with st.spinner("Loading sample data..."):
                try:
                    generator = SampleDataGenerator()
                    sample_data = generator.generate_complete_dataset(100)
                    st.session_state.sample_data = sample_data
                    st.session_state.data_loaded = True
                    st.success("Sample data loaded successfully!")
                except Exception as e:
                    st.error(f"Error loading sample data: {e}")
        
        if st.button("🗄️ Initialize Database"):
            with st.spinner("Setting up database..."):
                try:
                    db = ProcurementDatabase()
                    db.add_sample_data()
                    st.success("Database initialized with sample data!")
                except Exception as e:
                    st.error(f"Error initializing database: {e}")
    
    # Main content
    if st.session_state.agent is None:
        st.info("👈 Please initialize the AI Agent from the sidebar to begin analysis.")
        return
    
    # Tabs for different functionalities
    tab1, tab2, tab3, tab4, tab5 = st.tabs([
        "📊 Dashboard", 
        "🤖 AI Analysis", 
        "📈 Demand Forecasting", 
        "💰 Cost Optimization", 
        "📋 Reports"
    ])
    
    with tab1:
        show_dashboard()
    
    with tab2:
        show_ai_analysis()
    
    with tab3:
        show_demand_forecasting()
    
    with tab4:
        show_cost_optimization()
    
    with tab5:
        show_reports()

def show_dashboard():
    """Show the main dashboard"""
    st.markdown("## 📊 Procurement Dashboard")
    
    if not st.session_state.data_loaded:
        st.info("Please load sample data from the sidebar to view the dashboard.")
        return
    
    # Get data
    data = st.session_state.sample_data
    purchase_orders = data['purchase_orders']
    vendors = data['vendors']
    
    # Key metrics
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.markdown(f"""
        <div class="metric-card">
            <h3>Total Spend</h3>
            <h2>${purchase_orders['total_amount'].sum():,.2f}</h2>
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        st.markdown(f"""
        <div class="metric-card">
            <h3>Total Orders</h3>
            <h2>{len(purchase_orders)}</h2>
        </div>
        """, unsafe_allow_html=True)
    
    with col3:
        st.markdown(f"""
        <div class="metric-card">
            <h3>Active Vendors</h3>
            <h2>{len(vendors)}</h2>
        </div>
        """, unsafe_allow_html=True)
    
    with col4:
        avg_order = purchase_orders['total_amount'].mean()
        st.markdown(f"""
        <div class="metric-card">
            <h3>Avg Order Value</h3>
            <h2>${avg_order:,.2f}</h2>
        </div>
        """, unsafe_allow_html=True)
    
    # Charts
    col1, col2 = st.columns(2)
    
    with col1:
        # Spending by category
        category_spend = purchase_orders.groupby('category')['total_amount'].sum().reset_index()
        fig1 = px.pie(
            category_spend, 
            values='total_amount', 
            names='category',
            title="Spending by Category"
        )
        st.plotly_chart(fig1, use_container_width=True)
    
    with col2:
        # Monthly spending trend
        purchase_orders['month'] = pd.to_datetime(purchase_orders['order_date']).dt.to_period('M')
        monthly_spend = purchase_orders.groupby('month')['total_amount'].sum().reset_index()
        monthly_spend['month'] = monthly_spend['month'].astype(str)
        
        fig2 = px.line(
            monthly_spend,
            x='month',
            y='total_amount',
            title="Monthly Spending Trend"
        )
        st.plotly_chart(fig2, use_container_width=True)
    
    # Vendor performance table
    st.markdown("## 🏢 Vendor Performance")
    vendor_performance = purchase_orders.groupby('vendor_name').agg({
        'total_amount': 'sum',
        'order_id': 'count'
    }).reset_index()
    vendor_performance.columns = ['Vendor', 'Total Spend', 'Order Count']
    vendor_performance = vendor_performance.sort_values('Total Spend', ascending=False)
    
    st.dataframe(vendor_performance, use_container_width=True)

def show_ai_analysis():
    """Show AI analysis results"""
    st.markdown("## 🤖 AI-Powered Analysis")
    
    if not st.session_state.data_loaded:
        st.info("Please load sample data from the sidebar to perform AI analysis.")
        return
    
    # Analysis options
    col1, col2 = st.columns(2)
    
    with col1:
        analysis_type = st.selectbox(
            "Analysis Type",
            ["Comprehensive Analysis", "Trend Analysis", "Vendor Analysis", "Cost Analysis"]
        )
    
    with col2:
        if st.button("🔍 Run AI Analysis", type="primary"):
            with st.spinner("Running AI analysis..."):
                try:
                    data = st.session_state.sample_data
                    results = st.session_state.agent.analyze_procurement_data(
                        data['purchase_orders'],
                        data['vendors'],
                        data['historical_data']
                    )
                    st.session_state.analysis_results = results
                    st.success("AI analysis completed!")
                except Exception as e:
                    st.error(f"Error during analysis: {e}")
    
    # Display results
    if st.session_state.analysis_results:
        results = st.session_state.analysis_results
        
        # Summary metrics
        st.markdown("### 📊 Analysis Summary")
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.markdown(f"""
            <div class="info-card">
                <h4>Model Used</h4>
                <p>{results.get('model_info', {}).get('name', 'Unknown')}</p>
            </div>
            """, unsafe_allow_html=True)
        
        with col2:
            st.markdown(f"""
            <div class="success-card">
                <h4>Analysis Status</h4>
                <p>✅ Completed</p>
            </div>
            """, unsafe_allow_html=True)
        
        with col3:
            st.markdown(f"""
            <div class="info-card">
                <h4>Timestamp</h4>
                <p>{results.get('timestamp', 'Unknown')}</p>
            </div>
            """, unsafe_allow_html=True)
        
        # AI Insights
        st.markdown("### 🧠 AI Insights")
        
        if 'workflow_analysis' in results and 'analysis_results' in results['workflow_analysis']:
            analysis_data = results['workflow_analysis']['analysis_results']
            
            # Display key insights
            if 'trends' in analysis_data:
                st.markdown("#### 📈 Key Trends")
                trends = analysis_data['trends']
                for key, value in trends.items():
                    st.write(f"**{key.replace('_', ' ').title()}:** {value}")
            
            if 'cost_optimization' in analysis_data:
                st.markdown("#### 💰 Cost Optimization Opportunities")
                cost_opt = analysis_data['cost_optimization']
                for key, value in cost_opt.items():
                    st.write(f"**{key.replace('_', ' ').title()}:** {value}")
        
        # Recommendations
        if 'workflow_analysis' in results and 'recommendations' in results['workflow_analysis']:
            st.markdown("### 🎯 AI Recommendations")
            recommendations = results['workflow_analysis']['recommendations']
            
            for category, actions in recommendations.items():
                st.markdown(f"#### {category.replace('_', ' ').title()}")
                if isinstance(actions, list):
                    for action in actions:
                        st.write(f"• {action}")
                else:
                    st.write(f"• {actions}")

def show_demand_forecasting():
    """Show demand forecasting analysis"""
    st.markdown("## 📈 Demand Forecasting")
    
    if not st.session_state.data_loaded:
        st.info("Please load sample data from the sidebar to perform demand forecasting.")
        return
    
    # Forecasting parameters
    col1, col2 = st.columns(2)
    
    with col1:
        forecast_days = st.slider("Forecast Period (Days)", 7, 90, 30)
    
    with col2:
        if st.button("🔮 Generate Forecast", type="primary"):
            with st.spinner("Generating demand forecast..."):
                try:
                    data = st.session_state.sample_data
                    forecaster = DemandForecaster()
                    
                    # Train models
                    training_results = forecaster.train_models(data['purchase_orders'])
                    
                    # Generate forecast
                    forecast_results = forecaster.forecast_demand(data['purchase_orders'], forecast_days)
                    
                    st.session_state.forecast_results = forecast_results
                    st.session_state.training_results = training_results
                    st.success("Demand forecast generated!")
                except Exception as e:
                    st.error(f"Error generating forecast: {e}")
    
    # Display results
    if hasattr(st.session_state, 'forecast_results'):
        results = st.session_state.forecast_results
        
        if 'error' not in results:
            # Forecast chart
            st.markdown("### 📊 Demand Forecast")
            
            forecast_df = pd.DataFrame(results['forecasts'])
            forecast_df['date'] = pd.to_datetime(forecast_df['date'])
            
            fig = px.line(
                forecast_df,
                x='date',
                y='predicted_spend',
                title=f"Demand Forecast - Next {forecast_days} Days",
                labels={'predicted_spend': 'Predicted Spend ($)', 'date': 'Date'}
            )
            fig.add_scatter(
                x=forecast_df['date'],
                y=forecast_df['predicted_spend'] * (1 - forecast_df['confidence']),
                mode='lines',
                line=dict(dash='dash', color='red'),
                name='Lower Bound'
            )
            fig.add_scatter(
                x=forecast_df['date'],
                y=forecast_df['predicted_spend'] * (1 + forecast_df['confidence']),
                mode='lines',
                line=dict(dash='dash', color='red'),
                name='Upper Bound',
                fill='tonexty'
            )
            
            st.plotly_chart(fig, use_container_width=True)
            
            # Forecast summary
            col1, col2, col3 = st.columns(3)
            
            with col1:
                st.metric(
                    "Total Predicted Spend",
                    f"${results['total_predicted_spend']:,.2f}"
                )
            
            with col2:
                st.metric(
                    "Model Used",
                    results['model_used']
                )
            
            with col3:
                st.metric(
                    "Forecast Period",
                    results['forecast_period']
                )
            
            # Seasonality analysis
            if hasattr(st.session_state, 'training_results'):
                st.markdown("### 📅 Seasonality Analysis")
                data = st.session_state.sample_data
                seasonality = forecaster.analyze_seasonality(data['purchase_orders'])
                
                if 'error' not in seasonality:
                    col1, col2 = st.columns(2)
                    
                    with col1:
                        monthly_data = pd.DataFrame(list(seasonality['monthly_patterns'].items()), 
                                                  columns=['Month', 'Average Spend'])
                        fig_monthly = px.bar(monthly_data, x='Month', y='Average Spend', 
                                           title="Monthly Spending Patterns")
                        st.plotly_chart(fig_monthly, use_container_width=True)
                    
                    with col2:
                        dow_data = pd.DataFrame(list(seasonality['day_of_week_patterns'].items()), 
                                              columns=['Day of Week', 'Average Spend'])
                        fig_dow = px.bar(dow_data, x='Day of Week', y='Average Spend', 
                                       title="Day of Week Spending Patterns")
                        st.plotly_chart(fig_dow, use_container_width=True)

def show_cost_optimization():
    """Show cost optimization analysis"""
    st.markdown("## 💰 Cost Optimization")
    
    if not st.session_state.data_loaded:
        st.info("Please load sample data from the sidebar to perform cost optimization analysis.")
        return
    
    if st.button("🔍 Analyze Cost Opportunities", type="primary"):
        with st.spinner("Analyzing cost optimization opportunities..."):
            try:
                data = st.session_state.sample_data
                optimizer = CostOptimizer()
                
                opportunities = optimizer.analyze_cost_opportunities(
                    data['purchase_orders'],
                    data['vendors'],
                    data['items']
                )
                
                st.session_state.cost_opportunities = opportunities
                st.success("Cost optimization analysis completed!")
            except Exception as e:
                st.error(f"Error during cost optimization analysis: {e}")
    
    # Display results
    if hasattr(st.session_state, 'cost_opportunities'):
        opportunities = st.session_state.cost_opportunities
        
        # Total savings
        total_savings = opportunities.get('total_potential_savings', 0)
        st.markdown(f"""
        <div class="success-card">
            <h3>💰 Total Potential Savings</h3>
            <h2>${total_savings:,.2f}</h2>
        </div>
        """, unsafe_allow_html=True)
        
        # Individual strategies
        st.markdown("### 🎯 Optimization Strategies")
        
        for strategy, result in opportunities.items():
            if strategy == 'total_potential_savings':
                continue
                
            if isinstance(result, dict) and 'error' not in result:
                with st.expander(f"📊 {strategy.replace('_', ' ').title()}"):
                    col1, col2 = st.columns(2)
                    
                    with col1:
                        st.metric(
                            "Potential Savings",
                            f"${result.get('potential_savings', 0):,.2f}"
                        )
                    
                    with col2:
                        if 'opportunities' in result:
                            st.metric(
                                "Opportunities",
                                len(result['opportunities'])
                            )
                    
                    st.write(f"**Recommendation:** {result.get('recommendation', 'N/A')}")
                    
                    # Show detailed opportunities if available
                    if 'opportunities' in result and result['opportunities']:
                        st.markdown("#### Detailed Opportunities")
                        opp_df = pd.DataFrame(result['opportunities'])
                        st.dataframe(opp_df, use_container_width=True)
            
            elif isinstance(result, dict) and 'error' in result:
                with st.expander(f"❌ {strategy.replace('_', ' ').title()}"):
                    st.error(result['error'])

def show_reports():
    """Show comprehensive reports"""
    st.markdown("## 📋 Comprehensive Reports")
    
    # Report generation options
    col1, col2 = st.columns(2)
    
    with col1:
        report_type = st.selectbox(
            "Report Type",
            ["Executive Summary", "Detailed Analysis", "Cost Optimization", "Vendor Performance", "Demand Forecast"]
        )
    
    with col2:
        if st.button("📄 Generate Report", type="primary"):
            with st.spinner("Generating report..."):
                try:
                    report = generate_comprehensive_report(report_type)
                    st.session_state.current_report = report
                    st.success("Report generated successfully!")
                except Exception as e:
                    st.error(f"Error generating report: {e}")
    
    # Display report
    if hasattr(st.session_state, 'current_report'):
        st.markdown("### 📊 Generated Report")
        
        # Download button
        report_text = st.session_state.current_report
        st.download_button(
            label="📥 Download Report",
            data=report_text,
            file_name=f"procurement_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt",
            mime="text/plain"
        )
        
        # Display report
        st.text_area("Report Content", report_text, height=600)

def generate_comprehensive_report(report_type: str) -> str:
    """Generate comprehensive report based on type"""
    
    report = f"=== PROCUREMENT OPTIMIZATION REPORT ===\n"
    report += f"Report Type: {report_type}\n"
    report += f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n"
    
    if report_type == "Executive Summary":
        report += generate_executive_summary()
    elif report_type == "Detailed Analysis":
        report += generate_detailed_analysis()
    elif report_type == "Cost Optimization":
        report += generate_cost_optimization_report()
    elif report_type == "Vendor Performance":
        report += generate_vendor_performance_report()
    elif report_type == "Demand Forecast":
        report += generate_demand_forecast_report()
    
    return report

def generate_executive_summary() -> str:
    """Generate executive summary report"""
    if not st.session_state.data_loaded:
        return "No data available for report generation."
    
    data = st.session_state.sample_data
    purchase_orders = data['purchase_orders']
    
    summary = "EXECUTIVE SUMMARY\n"
    summary += "=" * 50 + "\n\n"
    
    summary += f"Total Procurement Spend: ${purchase_orders['total_amount'].sum():,.2f}\n"
    summary += f"Total Purchase Orders: {len(purchase_orders)}\n"
    summary += f"Active Vendors: {len(data['vendors'])}\n"
    summary += f"Average Order Value: ${purchase_orders['total_amount'].mean():,.2f}\n\n"
    
    if hasattr(st.session_state, 'cost_opportunities'):
        total_savings = st.session_state.cost_opportunities.get('total_potential_savings', 0)
        summary += f"Potential Cost Savings: ${total_savings:,.2f}\n"
        summary += f"Savings Percentage: {(total_savings / purchase_orders['total_amount'].sum() * 100):.1f}%\n\n"
    
    summary += "KEY RECOMMENDATIONS:\n"
    summary += "1. Implement bulk purchasing strategies\n"
    summary += "2. Consolidate vendor relationships\n"
    summary += "3. Negotiate better pricing terms\n"
    summary += "4. Optimize inventory management\n"
    summary += "5. Establish long-term contracts\n\n"
    
    return summary

def generate_detailed_analysis() -> str:
    """Generate detailed analysis report"""
    if not st.session_state.data_loaded:
        return "No data available for report generation."
    
    data = st.session_state.sample_data
    purchase_orders = data['purchase_orders']
    
    analysis = "DETAILED ANALYSIS\n"
    analysis += "=" * 50 + "\n\n"
    
    # Spending analysis
    analysis += "SPENDING ANALYSIS:\n"
    analysis += "-" * 20 + "\n"
    analysis += f"Total Spend: ${purchase_orders['total_amount'].sum():,.2f}\n"
    analysis += f"Number of Orders: {len(purchase_orders)}\n"
    analysis += f"Average Order Value: ${purchase_orders['total_amount'].mean():,.2f}\n"
    analysis += f"Median Order Value: ${purchase_orders['total_amount'].median():,.2f}\n"
    analysis += f"Standard Deviation: ${purchase_orders['total_amount'].std():,.2f}\n\n"
    
    # Category analysis
    analysis += "CATEGORY ANALYSIS:\n"
    analysis += "-" * 20 + "\n"
    category_spend = purchase_orders.groupby('category')['total_amount'].sum().sort_values(ascending=False)
    for category, spend in category_spend.items():
        analysis += f"{category}: ${spend:,.2f}\n"
    analysis += "\n"
    
    # Vendor analysis
    analysis += "VENDOR ANALYSIS:\n"
    analysis += "-" * 20 + "\n"
    vendor_spend = purchase_orders.groupby('vendor_name')['total_amount'].sum().sort_values(ascending=False)
    for vendor, spend in vendor_spend.head(10).items():
        analysis += f"{vendor}: ${spend:,.2f}\n"
    analysis += "\n"
    
    return analysis

def generate_cost_optimization_report() -> str:
    """Generate cost optimization report"""
    if not hasattr(st.session_state, 'cost_opportunities'):
        return "No cost optimization analysis available."
    
    opportunities = st.session_state.cost_opportunities
    
    report = "COST OPTIMIZATION REPORT\n"
    report += "=" * 50 + "\n\n"
    
    total_savings = opportunities.get('total_potential_savings', 0)
    report += f"TOTAL POTENTIAL SAVINGS: ${total_savings:,.2f}\n\n"
    
    for strategy, result in opportunities.items():
        if strategy == 'total_potential_savings':
            continue
            
        if isinstance(result, dict) and 'error' not in result:
            report += f"{strategy.replace('_', ' ').upper()}:\n"
            report += "-" * 30 + "\n"
            report += f"Potential Savings: ${result.get('potential_savings', 0):,.2f}\n"
            report += f"Recommendation: {result.get('recommendation', 'N/A')}\n"
            
            if 'opportunities' in result:
                report += f"Number of Opportunities: {len(result['opportunities'])}\n"
                for i, opp in enumerate(result['opportunities'][:5], 1):  # Show top 5
                    report += f"  {i}. {opp}\n"
            report += "\n"
    
    return report

def generate_vendor_performance_report() -> str:
    """Generate vendor performance report"""
    if not st.session_state.data_loaded:
        return "No data available for report generation."
    
    data = st.session_state.sample_data
    purchase_orders = data['purchase_orders']
    vendors = data['vendors']
    
    report = "VENDOR PERFORMANCE REPORT\n"
    report += "=" * 50 + "\n\n"
    
    # Vendor spending analysis
    vendor_performance = purchase_orders.groupby('vendor_name').agg({
        'total_amount': ['sum', 'count', 'mean'],
        'order_date': ['min', 'max']
    }).reset_index()
    
    vendor_performance.columns = ['vendor_name', 'total_spend', 'order_count', 'avg_order_value', 'first_order', 'last_order']
    vendor_performance = vendor_performance.sort_values('total_spend', ascending=False)
    
    report += "TOP VENDORS BY SPEND:\n"
    report += "-" * 30 + "\n"
    for _, vendor in vendor_performance.head(10).iterrows():
        report += f"{vendor['vendor_name']}:\n"
        report += f"  Total Spend: ${vendor['total_spend']:,.2f}\n"
        report += f"  Order Count: {vendor['order_count']}\n"
        report += f"  Average Order: ${vendor['avg_order_value']:,.2f}\n"
        report += f"  First Order: {vendor['first_order']}\n"
        report += f"  Last Order: {vendor['last_order']}\n\n"
    
    return report

def generate_demand_forecast_report() -> str:
    """Generate demand forecast report"""
    if not hasattr(st.session_state, 'forecast_results'):
        return "No demand forecast available."
    
    results = st.session_state.forecast_results
    
    report = "DEMAND FORECAST REPORT\n"
    report += "=" * 50 + "\n\n"
    
    if 'error' not in results:
        report += f"Model Used: {results.get('model_used', 'Unknown')}\n"
        report += f"Forecast Period: {results.get('forecast_period', 'Unknown')}\n"
        report += f"Total Predicted Spend: ${results.get('total_predicted_spend', 0):,.2f}\n\n"
        
        report += "FORECAST DETAILS:\n"
        report += "-" * 20 + "\n"
        
        forecasts = results.get('forecasts', [])
        for i, forecast in enumerate(forecasts[:10], 1):  # Show first 10 days
            report += f"Day {i}: ${forecast.get('predicted_spend', 0):,.2f} (Confidence: {forecast.get('confidence', 0):.1%})\n"
        
        if len(forecasts) > 10:
            report += f"... and {len(forecasts) - 10} more days\n"
    else:
        report += f"Error: {results['error']}\n"
    
    return report

if __name__ == "__main__":
    main()
