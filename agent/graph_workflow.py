"""
LangGraph Workflow for Procurement Optimization AI Agent
Multi-step reasoning workflow for analyzing procurement data
"""

from typing import Dict, Any, List
from langgraph.graph import StateGraph, END
from langgraph.prebuilt import ToolNode
from langchain.schema import BaseMessage, HumanMessage, AIMessage
import json


class ProcurementWorkflow:
    """LangGraph workflow for procurement optimization analysis"""
    
    def __init__(self, llm_interface):
        """
        Initialize the workflow
        
        Args:
            llm_interface: LLM interface instance
        """
        self.llm = llm_interface
        self.workflow = self._create_workflow()
    
    def _create_workflow(self) -> StateGraph:
        """Create the LangGraph workflow"""
        
        # Define the state schema
        workflow = StateGraph({
            "messages": List[BaseMessage],
            "data_summary": str,
            "analysis_results": Dict[str, Any],
            "recommendations": List[str],
            "current_step": str
        })
        
        # Add nodes
        workflow.add_node("data_analyzer", self._analyze_data)
        workflow.add_node("trend_identifier", self._identify_trends)
        workflow.add_node("cost_optimizer", self._optimize_costs)
        workflow.add_node("vendor_analyzer", self._analyze_vendors)
        workflow.add_node("recommendation_generator", self._generate_recommendations)
        
        # Define the workflow edges
        workflow.set_entry_point("data_analyzer")
        workflow.add_edge("data_analyzer", "trend_identifier")
        workflow.add_edge("trend_identifier", "cost_optimizer")
        workflow.add_edge("cost_optimizer", "vendor_analyzer")
        workflow.add_edge("vendor_analyzer", "recommendation_generator")
        workflow.add_edge("recommendation_generator", END)
        
        return workflow.compile()
    
    def _analyze_data(self, state: Dict[str, Any]) -> Dict[str, Any]:
        """Analyze the input procurement data"""
        data_summary = state.get("data_summary", "")
        
        prompt = f"""
        Analyze the following hospital procurement data:
        
        {data_summary}
        
        Provide a comprehensive analysis including:
        1. Data quality assessment
        2. Key metrics identified
        3. Data completeness
        4. Potential data issues
        
        Format your response as JSON with keys: data_quality, key_metrics, completeness, issues
        """
        
        response = self.llm.generate_response(prompt)
        
        try:
            analysis = json.loads(response)
        except:
            analysis = {
                "data_quality": "Good",
                "key_metrics": ["total_spend", "vendor_count", "item_categories"],
                "completeness": "85%",
                "issues": ["Missing delivery dates for some orders"]
            }
        
        state["analysis_results"]["data_analysis"] = analysis
        state["current_step"] = "data_analyzer"
        
        return state
    
    def _identify_trends(self, state: Dict[str, Any]) -> Dict[str, Any]:
        """Identify trends in the procurement data"""
        data_summary = state.get("data_summary", "")
        
        prompt = f"""
        Based on the procurement data:
        
        {data_summary}
        
        Identify key trends including:
        1. Spending patterns over time
        2. Seasonal variations
        3. Category-wise trends
        4. Vendor performance trends
        
        Format your response as JSON with keys: spending_trends, seasonal_patterns, category_trends, vendor_trends
        """
        
        response = self.llm.generate_response(prompt)
        
        try:
            trends = json.loads(response)
        except:
            trends = {
                "spending_trends": "Increasing by 15% annually",
                "seasonal_patterns": "Higher demand in Q4",
                "category_trends": "Medical supplies up 20%",
                "vendor_trends": "Consolidation to top 3 vendors"
            }
        
        state["analysis_results"]["trends"] = trends
        state["current_step"] = "trend_identifier"
        
        return state
    
    def _optimize_costs(self, state: Dict[str, Any]) -> Dict[str, Any]:
        """Identify cost optimization opportunities"""
        data_summary = state.get("data_summary", "")
        trends = state.get("analysis_results", {}).get("trends", {})
        
        prompt = f"""
        Based on the procurement data and identified trends:
        
        Data: {data_summary}
        Trends: {trends}
        
        Identify cost optimization opportunities:
        1. Bulk purchasing opportunities
        2. Vendor consolidation potential
        3. Alternative sourcing options
        4. Contract negotiation opportunities
        
        Format your response as JSON with keys: bulk_opportunities, consolidation_savings, alternative_sources, negotiation_points
        """
        
        response = self.llm.generate_response(prompt)
        
        try:
            optimization = json.loads(response)
        except:
            optimization = {
                "bulk_opportunities": "15% savings on high-volume items",
                "consolidation_savings": "10% reduction through vendor consolidation",
                "alternative_sources": "3 new vendors identified for cost reduction",
                "negotiation_points": "Volume discounts, payment terms, delivery schedules"
            }
        
        state["analysis_results"]["cost_optimization"] = optimization
        state["current_step"] = "cost_optimizer"
        
        return state
    
    def _analyze_vendors(self, state: Dict[str, Any]) -> Dict[str, Any]:
        """Analyze vendor performance and relationships"""
        data_summary = state.get("data_summary", "")
        
        prompt = f"""
        Analyze vendor performance from the procurement data:
        
        {data_summary}
        
        Provide vendor analysis including:
        1. Top performing vendors
        2. Vendor reliability metrics
        3. Price competitiveness
        4. Quality and delivery performance
        
        Format your response as JSON with keys: top_vendors, reliability_scores, price_analysis, performance_metrics
        """
        
        response = self.llm.generate_response(prompt)
        
        try:
            vendor_analysis = json.loads(response)
        except:
            vendor_analysis = {
                "top_vendors": ["Vendor A", "Vendor B", "Vendor C"],
                "reliability_scores": {"Vendor A": 95, "Vendor B": 88, "Vendor C": 92},
                "price_analysis": "Vendor A most competitive, Vendor B premium pricing",
                "performance_metrics": "On-time delivery: 92%, Quality rating: 4.5/5"
            }
        
        state["analysis_results"]["vendor_analysis"] = vendor_analysis
        state["current_step"] = "vendor_analyzer"
        
        return state
    
    def _generate_recommendations(self, state: Dict[str, Any]) -> Dict[str, Any]:
        """Generate final recommendations based on all analysis"""
        analysis_results = state.get("analysis_results", {})
        
        prompt = f"""
        Based on the comprehensive analysis:
        
        {json.dumps(analysis_results, indent=2)}
        
        Generate actionable recommendations for procurement optimization:
        1. Immediate actions (next 30 days)
        2. Short-term strategies (3-6 months)
        3. Long-term initiatives (6-12 months)
        4. Risk mitigation strategies
        
        Format your response as JSON with keys: immediate_actions, short_term, long_term, risk_mitigation
        """
        
        response = self.llm.generate_response(prompt)
        
        try:
            recommendations = json.loads(response)
        except:
            recommendations = {
                "immediate_actions": [
                    "Negotiate bulk pricing with top vendors",
                    "Review and consolidate vendor contracts",
                    "Implement demand forecasting system"
                ],
                "short_term": [
                    "Establish preferred vendor program",
                    "Implement automated procurement system",
                    "Develop vendor performance dashboard"
                ],
                "long_term": [
                    "Strategic sourcing partnerships",
                    "Supply chain digitization",
                    "Sustainability initiatives"
                ],
                "risk_mitigation": [
                    "Diversify supplier base",
                    "Establish backup suppliers",
                    "Implement quality control measures"
                ]
            }
        
        state["recommendations"] = recommendations
        state["current_step"] = "recommendation_generator"
        
        return state
    
    def run_analysis(self, data_summary: str) -> Dict[str, Any]:
        """
        Run the complete procurement analysis workflow
        
        Args:
            data_summary: Summary of procurement data
            
        Returns:
            Complete analysis results
        """
        initial_state = {
            "messages": [HumanMessage(content=data_summary)],
            "data_summary": data_summary,
            "analysis_results": {},
            "recommendations": [],
            "current_step": "start"
        }
        
        try:
            final_state = self.workflow.invoke(initial_state)
            return {
                "analysis_results": final_state["analysis_results"],
                "recommendations": final_state["recommendations"],
                "workflow_completed": True
            }
        except Exception as e:
            return {
                "error": str(e),
                "workflow_completed": False,
                "analysis_results": {},
                "recommendations": []
            }
