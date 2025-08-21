"""
Cost Optimization Algorithms for Hospital Procurement
Implements various optimization strategies for cost reduction
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Tuple, Optional
from datetime import datetime, timedelta
import warnings
warnings.filterwarnings('ignore')


class CostOptimizer:
    """Cost optimization for hospital procurement"""
    
    def __init__(self):
        """Initialize the cost optimizer"""
        self.optimization_strategies = {
            'bulk_purchasing': self._optimize_bulk_purchasing,
            'vendor_consolidation': self._optimize_vendor_consolidation,
            'price_negotiation': self._optimize_price_negotiation,
            'inventory_optimization': self._optimize_inventory,
            'contract_optimization': self._optimize_contracts
        }
    
    def analyze_cost_opportunities(self, 
                                 purchase_orders: pd.DataFrame,
                                 vendor_data: pd.DataFrame,
                                 item_data: pd.DataFrame) -> Dict:
        """
        Analyze cost optimization opportunities
        
        Args:
            purchase_orders: Purchase order data
            vendor_data: Vendor information
            item_data: Item catalog data
            
        Returns:
            Dictionary with optimization opportunities
        """
        opportunities = {}
        
        for strategy_name, strategy_func in self.optimization_strategies.items():
            try:
                result = strategy_func(purchase_orders, vendor_data, item_data)
                opportunities[strategy_name] = result
            except Exception as e:
                opportunities[strategy_name] = {"error": str(e)}
        
        # Calculate total potential savings
        total_savings = 0
        for strategy, result in opportunities.items():
            if isinstance(result, dict) and 'potential_savings' in result:
                total_savings += result['potential_savings']
        
        opportunities['total_potential_savings'] = total_savings
        
        return opportunities
    
    def _optimize_bulk_purchasing(self, 
                                purchase_orders: pd.DataFrame,
                                vendor_data: pd.DataFrame,
                                item_data: pd.DataFrame) -> Dict:
        """Optimize through bulk purchasing opportunities"""
        
        if purchase_orders.empty or item_data.empty:
            return {"error": "Insufficient data for bulk purchasing analysis"}
        
        # Merge purchase orders with item data
        merged_data = purchase_orders.merge(
            item_data[['item_id', 'name', 'unit_price', 'min_order_quantity']],
            left_on='item_id', right_on='item_id', how='left'
        )
        
        # Group by item to identify bulk purchasing opportunities
        item_analysis = merged_data.groupby(['item_id', 'name']).agg({
            'quantity': ['sum', 'count', 'mean'],
            'unit_price': 'mean',
            'total_amount': 'sum'
        }).reset_index()
        
        # Flatten column names
        item_analysis.columns = ['item_id', 'name', 'total_quantity', 'order_count', 'avg_order_size', 'avg_unit_price', 'total_spend']
        
        # Calculate bulk purchasing savings
        bulk_opportunities = []
        total_savings = 0
        
        for _, row in item_analysis.iterrows():
            if row['order_count'] > 1:  # Multiple orders for same item
                # Calculate potential savings from bulk ordering
                current_cost = row['total_spend']
                
                # Assume 10-20% savings for bulk orders
                bulk_discount = np.random.uniform(0.10, 0.20)
                potential_savings = current_cost * bulk_discount
                
                bulk_opportunities.append({
                    'item_name': row['name'],
                    'current_orders': row['order_count'],
                    'total_quantity': row['total_quantity'],
                    'current_cost': current_cost,
                    'potential_savings': potential_savings,
                    'recommended_bulk_quantity': row['total_quantity'],
                    'savings_percentage': bulk_discount * 100
                })
                
                total_savings += potential_savings
        
        return {
            'opportunities': bulk_opportunities,
            'potential_savings': total_savings,
            'recommendation': f"Consolidate {len(bulk_opportunities)} items into bulk orders for ${total_savings:,.2f} in savings"
        }
    
    def _optimize_vendor_consolidation(self,
                                     purchase_orders: pd.DataFrame,
                                     vendor_data: pd.DataFrame,
                                     item_data: pd.DataFrame) -> Dict:
        """Optimize through vendor consolidation"""
        
        if purchase_orders.empty or vendor_data.empty:
            return {"error": "Insufficient data for vendor consolidation analysis"}
        
        # Analyze vendor performance and spending
        vendor_analysis = purchase_orders.groupby('vendor_id').agg({
            'total_amount': ['sum', 'count', 'mean'],
            'order_date': ['min', 'max']
        }).reset_index()
        
        vendor_analysis.columns = ['vendor_id', 'total_spend', 'order_count', 'avg_order_value', 'first_order', 'last_order']
        
        # Merge with vendor data
        vendor_analysis = vendor_analysis.merge(
            vendor_data[['vendor_id', 'name', 'rating', 'quality_score', 'reliability_score']],
            on='vendor_id', how='left'
        )
        
        # Identify consolidation opportunities
        consolidation_opportunities = []
        total_savings = 0
        
        # Find vendors with low spend and poor performance
        low_performing_vendors = vendor_analysis[
            (vendor_analysis['total_spend'] < vendor_analysis['total_spend'].quantile(0.25)) &
            (vendor_analysis['rating'] < 4.0)
        ]
        
        for _, vendor in low_performing_vendors.iterrows():
            # Calculate potential savings from consolidation
            current_spend = vendor['total_spend']
            
            # Assume 5-15% savings from consolidation
            consolidation_savings = current_spend * np.random.uniform(0.05, 0.15)
            
            consolidation_opportunities.append({
                'vendor_name': vendor['name'],
                'current_spend': current_spend,
                'order_count': vendor['order_count'],
                'rating': vendor['rating'],
                'potential_savings': consolidation_savings,
                'recommended_action': 'Consolidate with higher-performing vendor'
            })
            
            total_savings += consolidation_savings
        
        return {
            'opportunities': consolidation_opportunities,
            'potential_savings': total_savings,
            'recommendation': f"Consolidate {len(consolidation_opportunities)} low-performing vendors for ${total_savings:,.2f} in savings"
        }
    
    def _optimize_price_negotiation(self,
                                  purchase_orders: pd.DataFrame,
                                  vendor_data: pd.DataFrame,
                                  item_data: pd.DataFrame) -> Dict:
        """Optimize through price negotiation opportunities"""
        
        if purchase_orders.empty or item_data.empty:
            return {"error": "Insufficient data for price negotiation analysis"}
        
        # Analyze price variations for same items across vendors
        price_analysis = purchase_orders.merge(
            item_data[['item_id', 'name', 'unit_price']],
            on='item_id', how='left'
        )
        
        # Group by item to find price variations
        item_price_variations = price_analysis.groupby(['item_id', 'name']).agg({
            'unit_price': ['min', 'max', 'mean', 'std'],
            'total_amount': 'sum'
        }).reset_index()
        
        item_price_variations.columns = ['item_id', 'name', 'min_price', 'max_price', 'avg_price', 'price_std', 'total_spend']
        
        # Find items with high price variation
        high_variation_items = item_price_variations[
            (item_price_variations['price_std'] > 0) &
            (item_price_variations['max_price'] > item_price_variations['min_price'] * 1.2)  # 20% variation
        ]
        
        negotiation_opportunities = []
        total_savings = 0
        
        for _, item in high_variation_items.iterrows():
            # Calculate potential savings from price negotiation
            current_avg_price = item['avg_price']
            min_price = item['min_price']
            
            # Assume we can negotiate to 5% above minimum price
            target_price = min_price * 1.05
            potential_savings_per_unit = current_avg_price - target_price
            
            # Estimate total quantity for the year
            annual_quantity = item['total_spend'] / current_avg_price * 12  # Extrapolate to annual
            potential_savings = potential_savings_per_unit * annual_quantity
            
            negotiation_opportunities.append({
                'item_name': item['name'],
                'current_avg_price': current_avg_price,
                'min_price': min_price,
                'target_price': target_price,
                'potential_savings_per_unit': potential_savings_per_unit,
                'annual_quantity': annual_quantity,
                'potential_savings': potential_savings
            })
            
            total_savings += potential_savings
        
        return {
            'opportunities': negotiation_opportunities,
            'potential_savings': total_savings,
            'recommendation': f"Negotiate prices for {len(negotiation_opportunities)} items for ${total_savings:,.2f} in savings"
        }
    
    def _optimize_inventory(self,
                          purchase_orders: pd.DataFrame,
                          vendor_data: pd.DataFrame,
                          item_data: pd.DataFrame) -> Dict:
        """Optimize inventory levels and ordering patterns"""
        
        if purchase_orders.empty:
            return {"error": "Insufficient data for inventory optimization"}
        
        # Analyze ordering patterns
        order_patterns = purchase_orders.groupby(['item_id', 'vendor_id']).agg({
            'order_date': ['count', 'min', 'max'],
            'quantity': ['sum', 'mean'],
            'total_amount': 'sum'
        }).reset_index()
        
        order_patterns.columns = ['item_id', 'vendor_id', 'order_count', 'first_order', 'last_order', 'total_quantity', 'avg_quantity', 'total_spend']
        
        # Calculate days between orders
        order_patterns['first_order'] = pd.to_datetime(order_patterns['first_order'])
        order_patterns['last_order'] = pd.to_datetime(order_patterns['last_order'])
        order_patterns['days_between_orders'] = (order_patterns['last_order'] - order_patterns['first_order']).dt.days / order_patterns['order_count']
        
        # Identify inventory optimization opportunities
        inventory_opportunities = []
        total_savings = 0
        
        # Find items with frequent small orders (opportunity for larger, less frequent orders)
        frequent_small_orders = order_patterns[
            (order_patterns['order_count'] > 3) &
            (order_patterns['avg_quantity'] < order_patterns['avg_quantity'].quantile(0.5))
        ]
        
        for _, item in frequent_small_orders.iterrows():
            # Calculate potential savings from inventory optimization
            current_ordering_cost = item['order_count'] * 50  # Assume $50 per order processing cost
            
            # Optimize to fewer, larger orders
            optimal_order_count = max(1, item['order_count'] // 2)
            optimized_ordering_cost = optimal_order_count * 50
            
            potential_savings = current_ordering_cost - optimized_ordering_cost
            
            inventory_opportunities.append({
                'item_id': item['item_id'],
                'vendor_id': item['vendor_id'],
                'current_order_count': item['order_count'],
                'optimal_order_count': optimal_order_count,
                'current_avg_quantity': item['avg_quantity'],
                'optimal_quantity': item['total_quantity'] / optimal_order_count,
                'potential_savings': potential_savings
            })
            
            total_savings += potential_savings
        
        return {
            'opportunities': inventory_opportunities,
            'potential_savings': total_savings,
            'recommendation': f"Optimize inventory for {len(inventory_opportunities)} items for ${total_savings:,.2f} in savings"
        }
    
    def _optimize_contracts(self,
                          purchase_orders: pd.DataFrame,
                          vendor_data: pd.DataFrame,
                          item_data: pd.DataFrame) -> Dict:
        """Optimize vendor contracts and terms"""
        
        if purchase_orders.empty or vendor_data.empty:
            return {"error": "Insufficient data for contract optimization"}
        
        # Analyze vendor performance and spending patterns
        vendor_performance = purchase_orders.groupby('vendor_id').agg({
            'total_amount': 'sum',
            'order_count': 'count',
            'order_date': ['min', 'max']
        }).reset_index()
        
        vendor_performance.columns = ['vendor_id', 'total_spend', 'order_count', 'first_order', 'last_order']
        
        # Merge with vendor data
        vendor_performance = vendor_performance.merge(
            vendor_data[['vendor_id', 'name', 'rating', 'delivery_lead_time', 'payment_terms']],
            on='vendor_id', how='left'
        )
        
        # Calculate vendor relationship duration
        vendor_performance['first_order'] = pd.to_datetime(vendor_performance['first_order'])
        vendor_performance['last_order'] = pd.to_datetime(vendor_performance['last_order'])
        vendor_performance['relationship_duration'] = (vendor_performance['last_order'] - vendor_performance['first_order']).dt.days
        
        # Identify contract optimization opportunities
        contract_opportunities = []
        total_savings = 0
        
        # Find high-spend vendors without long-term contracts
        high_spend_vendors = vendor_performance[
            vendor_performance['total_spend'] > vendor_performance['total_spend'].quantile(0.75)
        ]
        
        for _, vendor in high_spend_vendors.iterrows():
            # Calculate potential savings from contract optimization
            current_spend = vendor['total_spend']
            
            # Assume 3-8% savings from long-term contracts
            contract_savings = current_spend * np.random.uniform(0.03, 0.08)
            
            contract_opportunities.append({
                'vendor_name': vendor['name'],
                'current_spend': current_spend,
                'relationship_duration': vendor['relationship_duration'],
                'rating': vendor['rating'],
                'potential_savings': contract_savings,
                'recommended_contract_term': '2-3 years',
                'recommended_payment_terms': 'Net 45'
            })
            
            total_savings += contract_savings
        
        return {
            'opportunities': contract_opportunities,
            'potential_savings': total_savings,
            'recommendation': f"Optimize contracts for {len(contract_opportunities)} high-spend vendors for ${total_savings:,.2f} in savings"
        }
    
    def generate_optimization_report(self, opportunities: Dict) -> str:
        """Generate a comprehensive optimization report"""
        
        report = "=== PROCUREMENT COST OPTIMIZATION REPORT ===\n\n"
        
        total_savings = opportunities.get('total_potential_savings', 0)
        report += f"Total Potential Savings: ${total_savings:,.2f}\n\n"
        
        for strategy, result in opportunities.items():
            if strategy == 'total_potential_savings':
                continue
                
            if isinstance(result, dict) and 'error' not in result:
                report += f"--- {strategy.replace('_', ' ').title()} ---\n"
                report += f"Potential Savings: ${result.get('potential_savings', 0):,.2f}\n"
                report += f"Recommendation: {result.get('recommendation', 'N/A')}\n"
                
                if 'opportunities' in result:
                    report += f"Number of Opportunities: {len(result['opportunities'])}\n"
                
                report += "\n"
            elif isinstance(result, dict) and 'error' in result:
                report += f"--- {strategy.replace('_', ' ').title()} ---\n"
                report += f"Error: {result['error']}\n\n"
        
        return report
