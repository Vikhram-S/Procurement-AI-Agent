"""
Demand Forecasting Models for Hospital Procurement
Uses scikit-learn and XGBoost for time series forecasting
"""

import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import xgboost as xgb
from datetime import datetime, timedelta
import warnings
warnings.filterwarnings('ignore')


class DemandForecaster:
    """Demand forecasting for hospital procurement items"""
    
    def __init__(self):
        """Initialize the demand forecaster"""
        self.models = {
            'random_forest': RandomForestRegressor(n_estimators=100, random_state=42),
            'linear_regression': LinearRegression(),
            'xgboost': xgb.XGBRegressor(n_estimators=100, random_state=42)
        }
        self.scaler = StandardScaler()
        self.best_model = None
        self.feature_columns = []
        self.is_trained = False
    
    def prepare_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Prepare features for demand forecasting
        
        Args:
            df: DataFrame with purchase order data
            
        Returns:
            DataFrame with engineered features
        """
        if df.empty:
            return pd.DataFrame()
        
        # Convert date columns to datetime
        if 'order_date' in df.columns:
            df['order_date'] = pd.to_datetime(df['order_date'])
        
        # Group by date and calculate daily demand
        daily_demand = df.groupby('order_date').agg({
            'total_amount': 'sum',
            'order_id': 'count'
        }).reset_index()
        
        daily_demand.columns = ['date', 'total_spend', 'order_count']
        
        # Add time-based features
        daily_demand['year'] = daily_demand['date'].dt.year
        daily_demand['month'] = daily_demand['date'].dt.month
        daily_demand['day_of_week'] = daily_demand['date'].dt.dayofweek
        daily_demand['day_of_year'] = daily_demand['date'].dt.dayofyear
        daily_demand['quarter'] = daily_demand['date'].dt.quarter
        
        # Add lag features
        daily_demand['spend_lag_1'] = daily_demand['total_spend'].shift(1)
        daily_demand['spend_lag_7'] = daily_demand['total_spend'].shift(7)
        daily_demand['spend_lag_30'] = daily_demand['total_spend'].shift(30)
        
        daily_demand['order_count_lag_1'] = daily_demand['order_count'].shift(1)
        daily_demand['order_count_lag_7'] = daily_demand['order_count'].shift(7)
        
        # Add rolling averages
        daily_demand['spend_ma_7'] = daily_demand['total_spend'].rolling(window=7).mean()
        daily_demand['spend_ma_30'] = daily_demand['total_spend'].rolling(window=30).mean()
        daily_demand['order_count_ma_7'] = daily_demand['order_count'].rolling(window=7).mean()
        
        # Add seasonal features
        daily_demand['is_weekend'] = daily_demand['day_of_week'].isin([5, 6]).astype(int)
        daily_demand['is_month_end'] = daily_demand['date'].dt.is_month_end.astype(int)
        daily_demand['is_quarter_end'] = daily_demand['date'].dt.is_quarter_end.astype(int)
        
        # Remove rows with NaN values
        daily_demand = daily_demand.dropna()
        
        return daily_demand
    
    def train_models(self, df: pd.DataFrame, target_column: str = 'total_spend') -> dict:
        """
        Train demand forecasting models
        
        Args:
            df: DataFrame with purchase order data
            target_column: Column to predict
            
        Returns:
            Dictionary with model performance metrics
        """
        # Prepare features
        feature_df = self.prepare_features(df)
        
        if feature_df.empty:
            return {"error": "Insufficient data for training"}
        
        # Define feature columns
        self.feature_columns = [
            'year', 'month', 'day_of_week', 'day_of_year', 'quarter',
            'spend_lag_1', 'spend_lag_7', 'spend_lag_30',
            'order_count_lag_1', 'order_count_lag_7',
            'spend_ma_7', 'spend_ma_30', 'order_count_ma_7',
            'is_weekend', 'is_month_end', 'is_quarter_end'
        ]
        
        # Prepare X and y
        X = feature_df[self.feature_columns]
        y = feature_df[target_column]
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42
        )
        
        # Scale features
        X_train_scaled = self.scaler.fit_transform(X_train)
        X_test_scaled = self.scaler.transform(X_test)
        
        # Train models and evaluate
        results = {}
        best_score = float('-inf')
        
        for name, model in self.models.items():
            try:
                # Train model
                if name == 'xgboost':
                    model.fit(X_train_scaled, y_train)
                else:
                    model.fit(X_train_scaled, y_train)
                
                # Make predictions
                y_pred = model.predict(X_test_scaled)
                
                # Calculate metrics
                mae = mean_absolute_error(y_test, y_pred)
                mse = mean_squared_error(y_test, y_pred)
                rmse = np.sqrt(mse)
                r2 = r2_score(y_test, y_pred)
                
                results[name] = {
                    'mae': mae,
                    'mse': mse,
                    'rmse': rmse,
                    'r2': r2,
                    'model': model
                }
                
                # Track best model
                if r2 > best_score:
                    best_score = r2
                    self.best_model = model
                
            except Exception as e:
                results[name] = {"error": str(e)}
        
        self.is_trained = True
        return results
    
    def forecast_demand(self, df: pd.DataFrame, days_ahead: int = 30) -> dict:
        """
        Forecast demand for the next N days
        
        Args:
            df: Historical purchase order data
            days_ahead: Number of days to forecast
            
        Returns:
            Dictionary with forecast results
        """
        if not self.is_trained:
            return {"error": "Models not trained. Call train_models() first."}
        
        # Prepare features
        feature_df = self.prepare_features(df)
        
        if feature_df.empty:
            return {"error": "Insufficient data for forecasting"}
        
        # Get the last available data point
        last_date = feature_df['date'].max()
        last_features = feature_df[self.feature_columns].iloc[-1:].values
        
        # Generate future dates
        future_dates = pd.date_range(
            start=last_date + timedelta(days=1),
            periods=days_ahead,
            freq='D'
        )
        
        # Generate forecasts
        forecasts = []
        current_features = last_features.copy()
        
        for i, future_date in enumerate(future_dates):
            # Update features for the future date
            future_features = self._update_features_for_date(
                current_features[0], future_date, feature_df
            )
            
            # Scale features
            future_features_scaled = self.scaler.transform([future_features])
            
            # Make prediction
            prediction = self.best_model.predict(future_features_scaled)[0]
            
            forecasts.append({
                'date': future_date,
                'predicted_spend': max(0, prediction),  # Ensure non-negative
                'confidence': self._calculate_confidence(i, days_ahead)
            })
            
            # Update current features for next iteration
            current_features[0] = future_features
        
        return {
            'forecasts': forecasts,
            'model_used': type(self.best_model).__name__,
            'forecast_period': f"{future_dates[0].strftime('%Y-%m-%d')} to {future_dates[-1].strftime('%Y-%m-%d')}",
            'total_predicted_spend': sum(f['predicted_spend'] for f in forecasts)
        }
    
    def _update_features_for_date(self, features: np.ndarray, date: datetime, historical_df: pd.DataFrame) -> np.ndarray:
        """Update features for a specific future date"""
        # Create a copy of features
        new_features = features.copy()
        
        # Update time-based features
        new_features[0] = date.year  # year
        new_features[1] = date.month  # month
        new_features[2] = date.weekday()  # day_of_week
        new_features[3] = date.timetuple().tm_yday  # day_of_year
        new_features[4] = (date.month - 1) // 3 + 1  # quarter
        
        # Update seasonal features
        new_features[13] = 1 if date.weekday() in [5, 6] else 0  # is_weekend
        new_features[14] = 1 if date.day == pd.Timestamp(date).days_in_month else 0  # is_month_end
        new_features[15] = 1 if date.month in [3, 6, 9, 12] and date.day == pd.Timestamp(date).days_in_month else 0  # is_quarter_end
        
        return new_features
    
    def _calculate_confidence(self, day_index: int, total_days: int) -> float:
        """Calculate confidence level for forecast (decreases over time)"""
        # Confidence decreases as we forecast further into the future
        base_confidence = 0.85
        decay_rate = 0.02  # 2% decrease per day
        confidence = base_confidence * (1 - decay_rate * day_index)
        return max(0.3, confidence)  # Minimum 30% confidence
    
    def get_feature_importance(self) -> dict:
        """Get feature importance from the best model"""
        if not self.is_trained or self.best_model is None:
            return {"error": "Models not trained"}
        
        try:
            if hasattr(self.best_model, 'feature_importances_'):
                importance = self.best_model.feature_importances_
            elif hasattr(self.best_model, 'coef_'):
                importance = np.abs(self.best_model.coef_)
            else:
                return {"error": "Feature importance not available for this model"}
            
            # Create feature importance dictionary
            feature_importance = dict(zip(self.feature_columns, importance))
            
            # Sort by importance
            sorted_importance = dict(sorted(
                feature_importance.items(),
                key=lambda x: x[1],
                reverse=True
            ))
            
            return sorted_importance
            
        except Exception as e:
            return {"error": f"Error getting feature importance: {e}"}
    
    def analyze_seasonality(self, df: pd.DataFrame) -> dict:
        """Analyze seasonal patterns in the data"""
        feature_df = self.prepare_features(df)
        
        if feature_df.empty:
            return {"error": "Insufficient data for seasonality analysis"}
        
        # Monthly patterns
        monthly_avg = feature_df.groupby('month')['total_spend'].mean()
        
        # Day of week patterns
        dow_avg = feature_df.groupby('day_of_week')['total_spend'].mean()
        
        # Quarterly patterns
        quarterly_avg = feature_df.groupby('quarter')['total_spend'].mean()
        
        return {
            'monthly_patterns': monthly_avg.to_dict(),
            'day_of_week_patterns': dow_avg.to_dict(),
            'quarterly_patterns': quarterly_avg.to_dict(),
            'peak_month': monthly_avg.idxmax(),
            'peak_day': dow_avg.idxmax(),
            'peak_quarter': quarterly_avg.idxmax()
        }
