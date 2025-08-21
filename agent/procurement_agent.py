"""
Main Procurement Optimization AI Agent
Orchestrates the complete analysis workflow using LangChain and LangGraph
"""

from typing import Dict, Any, List, Optional
import pandas as pd
from datetime import datetime
import json

from .llm_interface import LLMInterface
from .graph_workflow import ProcurementWorkflow


class ProcurementAgent:
    """Main AI agent for procurement optimization analysis"""
    
    def __init__(self, model_type: str = "ollama", model_name: str = "mistral:7b"):
        """
        Initialize the procurement agent
        
        Args:
            model_type: Type of LLM to use ("ollama" or "huggingface")
            model_name: Specific model name
        """
        self.llm_interface = LLMInterface(model_type, model_name)
        self.workflow = ProcurementWorkflow(self.llm_interface)
        self.analysis_history = []
        
    def analyze_procurement_data(self, 
                               purchase_orders: pd.DataFrame,
                               vendor_data: pd.DataFrame,
                               historical_data: Optional[pd.DataFrame] = None) -> Dict[str, Any]:
        """
        Analyze procurement data and provide optimization recommendations
        
        Args:
            purchase_orders: DataFrame with purchase order data
            vendor_data: DataFrame with vendor information
            historical_data: Optional historical data for trend analysis
            
        Returns:
            Complete analysis results
        """
        # Generate data summary
        data_summary = self._generate_data_summary(purchase_orders, vendor_data, historical_data)
        
        # Run the LangGraph workflow
        workflow_results = self.workflow.run_analysis(data_summary)
        
        # Perform additional ML-based analysis
        ml_analysis = self._perform_ml_analysis(purchase_orders, vendor_data)
        
        # Combine results
        complete_analysis = {
            "workflow_analysis": workflow_results,
            "ml_analysis": ml_analysis,
            "data_summary": data_summary,
            "timestamp": datetime.now().isoformat(),
            "model_info": {
                "type": self.llm_interface.model_type,
                "name": self.llm_interface.model_name
            }
        }
        
        # Store in history
        self.analysis_history.append(complete_analysis)
        
        return complete_analysis
    
    def _generate_data_summary(self, 
                             purchase_orders: pd.DataFrame,
                             vendor_data: pd.DataFrame,
                             historical_data: Optional[pd.DataFrame] = None) -> str:
        """Generate a comprehensive summary of the procurement data"""
        
        summary_parts = []
        
        # Purchase orders summary
        if not purchase_orders.empty:
            po_summary = f"""
            Purchase Orders Analysis:
            - Total orders: {len(purchase_orders)}
            - Total spend: ${purchase_orders.get('amount', pd.Series([0])).sum():,.2f}
            - Average order value: ${purchase_orders.get('amount', pd.Series([0])).mean():,.2f}
            - Date range: {purchase_orders.get('order_date', pd.Series()).min()} to {purchase_orders.get('order_date', pd.Series()).max()}
            - Categories: {purchase_orders.get('category', pd.Series()).nunique()} unique categories
            """
            summary_parts.append(po_summary)
        
        # Vendor summary
        if not vendor_data.empty:
            vendor_summary = f"""
            Vendor Analysis:
            - Total vendors: {len(vendor_data)}
            - Active vendors: {len(vendor_data[vendor_data.get('status', '') == 'active'])}
            - Average vendor rating: {vendor_data.get('rating', pd.Series([0])).mean():.2f}/5
            - Top vendors by spend: {self._get_top_vendors(purchase_orders, vendor_data)}
            """
            summary_parts.append(vendor_summary)
        
        # Historical trends
        if historical_data is not None and not historical_data.empty:
            hist_summary = f"""
            Historical Trends:
            - Historical data points: {len(historical_data)}
            - Trend period: {historical_data.get('date', pd.Series()).min()} to {historical_data.get('date', pd.Series()).max()}
            - Growth rate: {self._calculate_growth_rate(historical_data):.1f}%
            """
            summary_parts.append(hist_summary)
        
        return "\n".join(summary_parts)
    
    def _get_top_vendors(self, purchase_orders: pd.DataFrame, vendor_data: pd.DataFrame) -> str:
        """Get top vendors by spend"""
        try:
            if 'vendor_id' in purchase_orders.columns and 'amount' in purchase_orders.columns:
                vendor_spend = purchase_orders.groupby('vendor_id')['amount'].sum().sort_values(ascending=False)
                top_vendors = vendor_spend.head(3)
                
                # Get vendor names
                vendor_names = []
                for vendor_id in top_vendors.index:
                    vendor_name = vendor_data[vendor_data.get('vendor_id', '') == vendor_id].get('name', 'Unknown').iloc[0]
                    vendor_names.append(f"{vendor_name} (${top_vendors[vendor_id]:,.2f})")
                
                return ", ".join(vendor_names)
        except:
            pass
        
        return "Data not available"
    
    def _calculate_growth_rate(self, historical_data: pd.DataFrame) -> float:
        """Calculate growth rate from historical data"""
        try:
            if 'amount' in historical_data.columns and 'date' in historical_data.columns:
                # Group by month and calculate total spend
                historical_data['month'] = pd.to_datetime(historical_data['date']).dt.to_period('M')
                monthly_spend = historical_data.groupby('month')['amount'].sum()
                
                if len(monthly_spend) >= 2:
                    first_month = monthly_spend.iloc[0]
                    last_month = monthly_spend.iloc[-1]
                    growth_rate = ((last_month - first_month) / first_month) * 100
                    return growth_rate
        except:
            pass
        
        return 0.0
    
    def _perform_ml_analysis(self, purchase_orders: pd.DataFrame, vendor_data: pd.DataFrame) -> Dict[str, Any]:
        """Perform machine learning-based analysis"""
        
        ml_results = {
            "demand_forecasting": self._forecast_demand(purchase_orders),
            "vendor_clustering": self._cluster_vendors(vendor_data),
            "anomaly_detection": self._detect_anomalies(purchase_orders),
            "cost_prediction": self._predict_costs(purchase_orders)
        }
        
        return ml_results
    
    def _forecast_demand(self, purchase_orders: pd.DataFrame) -> Dict[str, Any]:
        """Forecast future demand based on historical data"""
        try:
            if 'order_date' in purchase_orders.columns and 'amount' in purchase_orders.columns:
                # Simple time series analysis
                purchase_orders['month'] = pd.to_datetime(purchase_orders['order_date']).dt.to_period('M')
                monthly_demand = purchase_orders.groupby('month')['amount'].sum()
                
                # Calculate trend
                if len(monthly_demand) >= 3:
                    trend = (monthly_demand.iloc[-1] - monthly_demand.iloc[0]) / len(monthly_demand)
                    forecast_next_month = monthly_demand.iloc[-1] + trend
                    
                    return {
                        "current_monthly_demand": monthly_demand.iloc[-1],
                        "forecast_next_month": forecast_next_month,
                        "trend": trend,
                        "confidence": "Medium"
                    }
        except:
            pass
        
        return {"error": "Insufficient data for demand forecasting"}
    
    def _cluster_vendors(self, vendor_data: pd.DataFrame) -> Dict[str, Any]:
        """Cluster vendors based on performance metrics"""
        try:
            if not vendor_data.empty:
                # Simple clustering based on rating and performance
                high_performers = vendor_data[vendor_data.get('rating', 0) >= 4.0]
                medium_performers = vendor_data[(vendor_data.get('rating', 0) >= 3.0) & (vendor_data.get('rating', 0) < 4.0)]
                low_performers = vendor_data[vendor_data.get('rating', 0) < 3.0]
                
                return {
                    "high_performers": len(high_performers),
                    "medium_performers": len(medium_performers),
                    "low_performers": len(low_performers),
                    "recommendations": {
                        "expand_high_performers": len(high_performers) < 5,
                        "improve_low_performers": len(low_performers) > 3
                    }
                }
        except:
            pass
        
        return {"error": "Unable to cluster vendors"}
    
    def _detect_anomalies(self, purchase_orders: pd.DataFrame) -> Dict[str, Any]:
        """Detect anomalies in purchase orders"""
        try:
            if 'amount' in purchase_orders.columns:
                # Simple anomaly detection using IQR method
                Q1 = purchase_orders['amount'].quantile(0.25)
                Q3 = purchase_orders['amount'].quantile(0.75)
                IQR = Q3 - Q1
                
                lower_bound = Q1 - 1.5 * IQR
                upper_bound = Q3 + 1.5 * IQR
                
                anomalies = purchase_orders[
                    (purchase_orders['amount'] < lower_bound) | 
                    (purchase_orders['amount'] > upper_bound)
                ]
                
                return {
                    "anomaly_count": len(anomalies),
                    "anomaly_percentage": (len(anomalies) / len(purchase_orders)) * 100,
                    "threshold_lower": lower_bound,
                    "threshold_upper": upper_bound
                }
        except:
            pass
        
        return {"error": "Unable to detect anomalies"}
    
    def _predict_costs(self, purchase_orders: pd.DataFrame) -> Dict[str, Any]:
        """Predict future costs based on historical patterns"""
        try:
            if 'amount' in purchase_orders.columns and 'order_date' in purchase_orders.columns:
                # Simple cost prediction
                purchase_orders['month'] = pd.to_datetime(purchase_orders['order_date']).dt.to_period('M')
                monthly_costs = purchase_orders.groupby('month')['amount'].sum()
                
                if len(monthly_costs) >= 3:
                    avg_monthly_cost = monthly_costs.mean()
                    cost_volatility = monthly_costs.std()
                    
                    return {
                        "predicted_monthly_cost": avg_monthly_cost,
                        "cost_volatility": cost_volatility,
                        "confidence_interval": f"${avg_monthly_cost - cost_volatility:,.2f} - ${avg_monthly_cost + cost_volatility:,.2f}"
                    }
        except:
            pass
        
        return {"error": "Unable to predict costs"}
    
    def get_analysis_history(self) -> List[Dict[str, Any]]:
        """Get the history of all analyses performed"""
        return self.analysis_history
    
    def export_analysis(self, analysis_id: int = -1, format: str = "json") -> str:
        """
        Export analysis results
        
        Args:
            analysis_id: Index of analysis to export (-1 for latest)
            format: Export format ("json" or "csv")
            
        Returns:
            Exported data as string
        """
        if not self.analysis_history:
            return "No analysis history available"
        
        analysis = self.analysis_history[analysis_id]
        
        if format.lower() == "json":
            return json.dumps(analysis, indent=2)
        elif format.lower() == "csv":
            # Convert to CSV format (simplified)
            return self._convert_to_csv(analysis)
        else:
            return "Unsupported format"
    
    def _convert_to_csv(self, analysis: Dict[str, Any]) -> str:
        """Convert analysis results to CSV format"""
        # This is a simplified CSV conversion
        csv_lines = ["Metric,Value"]
        
        # Add basic metrics
        if "data_summary" in analysis:
            csv_lines.append("Data Summary,Available")
        
        if "workflow_analysis" in analysis:
            csv_lines.append("Workflow Analysis,Completed")
        
        if "ml_analysis" in analysis:
            csv_lines.append("ML Analysis,Completed")
        
        return "\n".join(csv_lines)
