"""
Visualization utilities for Procurement Optimization AI
Provides chart creation and data visualization functions
"""

import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import pandas as pd
import numpy as np
from typing import Dict, List, Optional
from datetime import datetime, timedelta


def create_dashboard_charts(purchase_orders: pd.DataFrame, vendors: pd.DataFrame) -> Dict:
    """
    Create comprehensive dashboard charts
    
    Args:
        purchase_orders: Purchase order data
        vendors: Vendor data
        
    Returns:
        Dictionary containing all charts
    """
    charts = {}
    
    # 1. Spending by Category (Pie Chart)
    category_spend = purchase_orders.groupby('category')['total_amount'].sum().reset_index()
    charts['category_spend'] = px.pie(
        category_spend,
        values='total_amount',
        names='category',
        title="Spending by Category",
        color_discrete_sequence=px.colors.qualitative.Set3
    )
    
    # 2. Monthly Spending Trend (Line Chart)
    purchase_orders['month'] = pd.to_datetime(purchase_orders['order_date']).dt.to_period('M')
    monthly_spend = purchase_orders.groupby('month')['total_amount'].sum().reset_index()
    monthly_spend['month'] = monthly_spend['month'].astype(str)
    
    charts['monthly_trend'] = px.line(
        monthly_spend,
        x='month',
        y='total_amount',
        title="Monthly Spending Trend",
        labels={'total_amount': 'Total Spend ($)', 'month': 'Month'},
        line_shape='linear',
        markers=True
    )
    
    # 3. Vendor Performance (Bar Chart)
    vendor_performance = purchase_orders.groupby('vendor_name')['total_amount'].sum().sort_values(ascending=False).head(10)
    charts['vendor_performance'] = px.bar(
        x=vendor_performance.index,
        y=vendor_performance.values,
        title="Top 10 Vendors by Spend",
        labels={'x': 'Vendor', 'y': 'Total Spend ($)'},
        color=vendor_performance.values,
        color_continuous_scale='viridis'
    )
    
    # 4. Order Status Distribution (Donut Chart)
    status_counts = purchase_orders['status'].value_counts()
    charts['order_status'] = px.pie(
        values=status_counts.values,
        names=status_counts.index,
        title="Order Status Distribution",
        hole=0.4,
        color_discrete_sequence=px.colors.qualitative.Pastel
    )
    
    # 5. Daily Spending Pattern (Area Chart)
    purchase_orders['date'] = pd.to_datetime(purchase_orders['order_date']).dt.date
    daily_spend = purchase_orders.groupby('date')['total_amount'].sum().reset_index()
    daily_spend['date'] = pd.to_datetime(daily_spend['date'])
    
    charts['daily_pattern'] = px.area(
        daily_spend,
        x='date',
        y='total_amount',
        title="Daily Spending Pattern",
        labels={'total_amount': 'Daily Spend ($)', 'date': 'Date'},
        fill='tonexty'
    )
    
    return charts


def create_cost_optimization_charts(opportunities: Dict) -> Dict:
    """
    Create charts for cost optimization analysis
    
    Args:
        opportunities: Cost optimization opportunities
        
    Returns:
        Dictionary containing optimization charts
    """
    charts = {}
    
    # 1. Savings by Strategy (Bar Chart)
    strategy_savings = {}
    for strategy, result in opportunities.items():
        if strategy != 'total_potential_savings' and isinstance(result, dict) and 'potential_savings' in result:
            strategy_savings[strategy.replace('_', ' ').title()] = result['potential_savings']
    
    if strategy_savings:
        charts['strategy_savings'] = px.bar(
            x=list(strategy_savings.keys()),
            y=list(strategy_savings.values()),
            title="Potential Savings by Strategy",
            labels={'x': 'Strategy', 'y': 'Potential Savings ($)'},
            color=list(strategy_savings.values()),
            color_continuous_scale='RdYlGn'
        )
    
    # 2. Savings Distribution (Pie Chart)
    if strategy_savings:
        charts['savings_distribution'] = px.pie(
            values=list(strategy_savings.values()),
            names=list(strategy_savings.keys()),
            title="Savings Distribution by Strategy",
            color_discrete_sequence=px.colors.qualitative.Set2
        )
    
    return charts


def create_demand_forecast_chart(forecast_data: List[Dict], historical_data: Optional[pd.DataFrame] = None) -> go.Figure:
    """
    Create demand forecast chart with confidence intervals
    
    Args:
        forecast_data: List of forecast data points
        historical_data: Optional historical data for comparison
        
    Returns:
        Plotly figure with forecast
    """
    fig = go.Figure()
    
    # Add historical data if available
    if historical_data is not None and not historical_data.empty:
        fig.add_trace(go.Scatter(
            x=historical_data['date'],
            y=historical_data['amount'],
            mode='lines+markers',
            name='Historical Data',
            line=dict(color='blue', width=2),
            marker=dict(size=4)
        ))
    
    # Add forecast data
    forecast_df = pd.DataFrame(forecast_data)
    forecast_df['date'] = pd.to_datetime(forecast_df['date'])
    
    # Main forecast line
    fig.add_trace(go.Scatter(
        x=forecast_df['date'],
        y=forecast_df['predicted_spend'],
        mode='lines+markers',
        name='Forecast',
        line=dict(color='red', width=3),
        marker=dict(size=6)
    ))
    
    # Confidence intervals
    fig.add_trace(go.Scatter(
        x=forecast_df['date'],
        y=forecast_df['predicted_spend'] * (1 + forecast_df['confidence']),
        mode='lines',
        line=dict(width=0),
        showlegend=False,
        fillcolor='rgba(255,0,0,0.2)',
        fill='tonexty'
    ))
    
    fig.add_trace(go.Scatter(
        x=forecast_df['date'],
        y=forecast_df['predicted_spend'] * (1 - forecast_df['confidence']),
        mode='lines',
        line=dict(width=0),
        fillcolor='rgba(255,0,0,0.2)',
        fill='tonexty',
        name='Confidence Interval'
    ))
    
    fig.update_layout(
        title="Demand Forecast with Confidence Intervals",
        xaxis_title="Date",
        yaxis_title="Predicted Spend ($)",
        hovermode='x unified',
        template='plotly_white'
    )
    
    return fig


def create_vendor_analysis_charts(vendors: pd.DataFrame, purchase_orders: pd.DataFrame) -> Dict:
    """
    Create vendor analysis charts
    
    Args:
        vendors: Vendor data
        purchase_orders: Purchase order data
        
    Returns:
        Dictionary containing vendor analysis charts
    """
    charts = {}
    
    # 1. Vendor Rating Distribution
    charts['vendor_ratings'] = px.histogram(
        vendors,
        x='rating',
        title="Vendor Rating Distribution",
        labels={'rating': 'Rating', 'count': 'Number of Vendors'},
        nbins=10,
        color_discrete_sequence=['lightblue']
    )
    
    # 2. Vendor Performance Scatter Plot
    vendor_performance = purchase_orders.groupby('vendor_name').agg({
        'total_amount': 'sum',
        'order_id': 'count'
    }).reset_index()
    vendor_performance.columns = ['vendor_name', 'total_spend', 'order_count']
    
    # Merge with vendor ratings
    vendor_performance = vendor_performance.merge(
        vendors[['name', 'rating']],
        left_on='vendor_name',
        right_on='name',
        how='left'
    )
    
    charts['vendor_scatter'] = px.scatter(
        vendor_performance,
        x='total_spend',
        y='rating',
        size='order_count',
        title="Vendor Performance: Spend vs Rating",
        labels={'total_spend': 'Total Spend ($)', 'rating': 'Rating', 'order_count': 'Order Count'},
        hover_data=['vendor_name']
    )
    
    # 3. Vendor Category Analysis
    if 'category' in vendors.columns:
        category_counts = vendors['category'].value_counts()
        charts['vendor_categories'] = px.bar(
            x=category_counts.index,
            y=category_counts.values,
            title="Vendors by Category",
            labels={'x': 'Category', 'y': 'Number of Vendors'},
            color=category_counts.values,
            color_continuous_scale='plasma'
        )
    
    return charts


def create_seasonality_charts(seasonality_data: Dict) -> Dict:
    """
    Create seasonality analysis charts
    
    Args:
        seasonality_data: Seasonality analysis results
        
    Returns:
        Dictionary containing seasonality charts
    """
    charts = {}
    
    # 1. Monthly Patterns
    if 'monthly_patterns' in seasonality_data:
        monthly_data = pd.DataFrame(list(seasonality_data['monthly_patterns'].items()),
                                  columns=['Month', 'Average Spend'])
        charts['monthly_patterns'] = px.bar(
            monthly_data,
            x='Month',
            y='Average Spend',
            title="Monthly Spending Patterns",
            labels={'Average Spend': 'Average Spend ($)', 'Month': 'Month'},
            color='Average Spend',
            color_continuous_scale='viridis'
        )
    
    # 2. Day of Week Patterns
    if 'day_of_week_patterns' in seasonality_data:
        dow_data = pd.DataFrame(list(seasonality_data['day_of_week_patterns'].items()),
                              columns=['Day of Week', 'Average Spend'])
        charts['day_of_week_patterns'] = px.bar(
            dow_data,
            x='Day of Week',
            y='Average Spend',
            title="Day of Week Spending Patterns",
            labels={'Average Spend': 'Average Spend ($)', 'Day of Week': 'Day of Week'},
            color='Average Spend',
            color_continuous_scale='plasma'
        )
    
    # 3. Quarterly Patterns
    if 'quarterly_patterns' in seasonality_data:
        quarterly_data = pd.DataFrame(list(seasonality_data['quarterly_patterns'].items()),
                                    columns=['Quarter', 'Average Spend'])
        charts['quarterly_patterns'] = px.bar(
            quarterly_data,
            x='Quarter',
            y='Average Spend',
            title="Quarterly Spending Patterns",
            labels={'Average Spend': 'Average Spend ($)', 'Quarter': 'Quarter'},
            color='Average Spend',
            color_continuous_scale='inferno'
        )
    
    return charts


def create_comparison_chart(data1: pd.DataFrame, data2: pd.DataFrame, 
                          label1: str, label2: str, metric: str) -> go.Figure:
    """
    Create comparison chart between two datasets
    
    Args:
        data1: First dataset
        data2: Second dataset
        label1: Label for first dataset
        label2: Label for second dataset
        metric: Metric to compare
        
    Returns:
        Plotly figure with comparison
    """
    fig = go.Figure()
    
    # Add first dataset
    fig.add_trace(go.Scatter(
        x=data1.index,
        y=data1[metric],
        mode='lines+markers',
        name=label1,
        line=dict(color='blue', width=2)
    ))
    
    # Add second dataset
    fig.add_trace(go.Scatter(
        x=data2.index,
        y=data2[metric],
        mode='lines+markers',
        name=label2,
        line=dict(color='red', width=2)
    ))
    
    fig.update_layout(
        title=f"Comparison: {label1} vs {label2}",
        xaxis_title="Time",
        yaxis_title=metric,
        hovermode='x unified',
        template='plotly_white'
    )
    
    return fig


def create_summary_dashboard(purchase_orders: pd.DataFrame, vendors: pd.DataFrame, 
                           opportunities: Optional[Dict] = None) -> go.Figure:
    """
    Create a comprehensive summary dashboard
    
    Args:
        purchase_orders: Purchase order data
        vendors: Vendor data
        opportunities: Optional cost optimization opportunities
        
    Returns:
        Plotly figure with subplots
    """
    # Create subplots
    fig = make_subplots(
        rows=2, cols=2,
        subplot_titles=('Spending by Category', 'Monthly Trend', 'Vendor Performance', 'Cost Optimization'),
        specs=[[{"type": "pie"}, {"type": "scatter"}],
               [{"type": "bar"}, {"type": "bar"}]]
    )
    
    # 1. Spending by Category (Pie Chart)
    category_spend = purchase_orders.groupby('category')['total_amount'].sum()
    fig.add_trace(
        go.Pie(labels=category_spend.index, values=category_spend.values, name="Categories"),
        row=1, col=1
    )
    
    # 2. Monthly Trend (Line Chart)
    purchase_orders['month'] = pd.to_datetime(purchase_orders['order_date']).dt.to_period('M')
    monthly_spend = purchase_orders.groupby('month')['total_amount'].sum()
    fig.add_trace(
        go.Scatter(x=monthly_spend.index.astype(str), y=monthly_spend.values, name="Monthly Spend"),
        row=1, col=2
    )
    
    # 3. Vendor Performance (Bar Chart)
    vendor_performance = purchase_orders.groupby('vendor_name')['total_amount'].sum().sort_values(ascending=False).head(5)
    fig.add_trace(
        go.Bar(x=vendor_performance.index, y=vendor_performance.values, name="Top Vendors"),
        row=2, col=1
    )
    
    # 4. Cost Optimization (Bar Chart)
    if opportunities:
        strategy_savings = {}
        for strategy, result in opportunities.items():
            if strategy != 'total_potential_savings' and isinstance(result, dict) and 'potential_savings' in result:
                strategy_savings[strategy.replace('_', ' ').title()] = result['potential_savings']
        
        if strategy_savings:
            fig.add_trace(
                go.Bar(x=list(strategy_savings.keys()), y=list(strategy_savings.values()), name="Savings"),
                row=2, col=2
            )
    
    fig.update_layout(height=800, title_text="Procurement Summary Dashboard")
    return fig
