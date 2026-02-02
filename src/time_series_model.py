"""
Time Series Model Module
CRISP-DM Phase 4: Modeling - Demand Forecasting
"""

import pandas as pd
import numpy as np
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.tsa.statespace.sarimax import SARIMAX
import joblib
import os
import warnings
warnings.filterwarnings('ignore')

# Prophet import (optional - may not be installed)
try:
    from prophet import Prophet
    PROPHET_AVAILABLE = True
except ImportError:
    PROPHET_AVAILABLE = False


class DemandForecaster:
    """Time series model for booking demand forecasting."""
    
    def __init__(self, model_type: str = 'arima'):
        """
        Initialize forecaster.
        
        Args:
            model_type: 'arima', 'sarima', or 'prophet'
        """
        self.model_type = model_type
        self.model = None
        self.history = None
        
    def train(self, df: pd.DataFrame):
        """
        Train the time series model.
        
        Args:
            df: DataFrame with columns 'ds' (date) and 'y' (value)
        """
        self.history = df.copy()
        
        if self.model_type == 'arima':
            self._train_arima(df)
        elif self.model_type == 'sarima':
            self._train_sarima(df)
        elif self.model_type == 'prophet':
            self._train_prophet(df)
        else:
            raise ValueError(f"Unknown model type: {self.model_type}")
        
        return self
    
    def _train_arima(self, df: pd.DataFrame):
        """Train ARIMA model."""
        # Set date as index
        ts = df.set_index('ds')['y']
        
        # Fit ARIMA(1,1,1) - simple but effective
        self.model = ARIMA(ts, order=(1, 1, 1))
        self.fitted_model = self.model.fit()
        print(f"ARIMA AIC: {self.fitted_model.aic:.2f}")
    
    def _train_sarima(self, df: pd.DataFrame):
        """Train SARIMA model with weekly seasonality."""
        ts = df.set_index('ds')['y']
        
        # SARIMA with weekly seasonality (s=7)
        self.model = SARIMAX(
            ts, 
            order=(1, 1, 1),
            seasonal_order=(1, 1, 1, 7)
        )
        self.fitted_model = self.model.fit(disp=False)
        print(f"SARIMA AIC: {self.fitted_model.aic:.2f}")
    
    def _train_prophet(self, df: pd.DataFrame):
        """Train Prophet model."""
        if not PROPHET_AVAILABLE:
            raise ImportError("Prophet is not installed. Install with: pip install prophet")
        
        self.model = Prophet(
            yearly_seasonality=True,
            weekly_seasonality=True,
            daily_seasonality=False
        )
        self.model.fit(df[['ds', 'y']])
        print("Prophet model trained")
    
    def predict(self, periods: int = 30) -> pd.DataFrame:
        """
        Forecast future demand.
        
        Args:
            periods: Number of days to forecast
            
        Returns:
            DataFrame with forecast
        """
        if self.model_type in ['arima', 'sarima']:
            return self._predict_arima(periods)
        elif self.model_type == 'prophet':
            return self._predict_prophet(periods)
    
    def _predict_arima(self, periods: int) -> pd.DataFrame:
        """Generate ARIMA/SARIMA forecast."""
        forecast = self.fitted_model.forecast(steps=periods)
        
        # Create date range
        last_date = self.history['ds'].max()
        future_dates = pd.date_range(
            start=last_date + pd.Timedelta(days=1),
            periods=periods,
            freq='D'
        )
        
        return pd.DataFrame({
            'ds': future_dates,
            'yhat': forecast.values,
            'yhat_lower': forecast.values * 0.8,  # Simple confidence interval
            'yhat_upper': forecast.values * 1.2
        })
    
    def _predict_prophet(self, periods: int) -> pd.DataFrame:
        """Generate Prophet forecast."""
        future = self.model.make_future_dataframe(periods=periods)
        forecast = self.model.predict(future)
        
        # Return only future predictions
        return forecast[['ds', 'yhat', 'yhat_lower', 'yhat_upper']].tail(periods)
    
    def evaluate(self, test_df: pd.DataFrame) -> dict:
        """
        Evaluate forecast accuracy.
        
        Args:
            test_df: DataFrame with actual values
            
        Returns:
            Dictionary of metrics
        """
        predictions = self.predict(len(test_df))
        
        y_true = test_df['y'].values
        y_pred = predictions['yhat'].values[:len(y_true)]
        
        # Calculate metrics
        mae = np.mean(np.abs(y_true - y_pred))
        rmse = np.sqrt(np.mean((y_true - y_pred) ** 2))
        mape = np.mean(np.abs((y_true - y_pred) / y_true)) * 100
        
        metrics = {
            'mae': mae,
            'rmse': rmse,
            'mape': mape
        }
        
        print("\n=== Time Series Evaluation ===")
        print(f"MAE:  {mae:.2f}")
        print(f"RMSE: {rmse:.2f}")
        print(f"MAPE: {mape:.2f}%")
        
        return metrics
    
    def save(self, filepath: str):
        """Save model to disk."""
        import json
        from prophet.serialize import model_to_json
        
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        
        if self.model_type in ['arima', 'sarima']:
            joblib.dump({
                'model_type': self.model_type,
                'fitted_model': self.fitted_model,
                'history': self.history.to_dict() if self.history is not None else None
            }, filepath)
        elif self.model_type == 'prophet':
            # Use Prophet's native JSON serialization for compatibility
            joblib.dump({
                'model_type': self.model_type,
                'model_json': model_to_json(self.model),
                'history': self.history.to_dict() if self.history is not None else None
            }, filepath)
        
        print(f"Model saved to {filepath}")
    
    @classmethod
    def load(cls, filepath: str) -> 'DemandForecaster':
        """Load model from disk."""
        import json
        from prophet.serialize import model_from_json
        import pandas as pd
        
        data = joblib.load(filepath)
        
        instance = cls(model_type=data['model_type'])
        
        # Handle both old and new serialization formats
        if 'history' in data and data['history'] is not None:
            if isinstance(data['history'], dict):
                instance.history = pd.DataFrame(data['history'])
            else:
                instance.history = data['history']
        
        if data['model_type'] in ['arima', 'sarima']:
            instance.fitted_model = data['fitted_model']
        else:
            # Handle both old (model object) and new (JSON) formats
            if 'model_json' in data:
                instance.model = model_from_json(data['model_json'])
            else:
                instance.model = data['model']
        
        return instance


if __name__ == "__main__":
    # Example usage
    from preprocessing import load_data, clean_data, engineer_features, prepare_time_series_data
    
    # Load and prepare data
    df = load_data("../data/raw/hotel_bookings.csv")
    df = clean_data(df)
    df = engineer_features(df)
    ts_data = prepare_time_series_data(df)
    
    # Split into train/test
    train_size = int(len(ts_data) * 0.8)
    train_df = ts_data.iloc[:train_size]
    test_df = ts_data.iloc[train_size:]
    
    print(f"Training on {len(train_df)} days, testing on {len(test_df)} days")
    
    # Train ARIMA model
    forecaster = DemandForecaster(model_type='arima')
    forecaster.train(train_df)
    
    # Evaluate
    metrics = forecaster.evaluate(test_df)
    
    # Forecast next 30 days
    forecast = forecaster.predict(periods=30)
    print("\nNext 30 Days Forecast:")
    print(forecast.head(10))
    
    # Save model
    forecaster.save("../models/demand_model.pkl")
