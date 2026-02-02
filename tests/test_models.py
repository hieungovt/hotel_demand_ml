"""
Unit tests for ML models.
"""

import pytest
import os
import sys
import pandas as pd
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from classification_model import CancellationClassifier
from time_series_model import DemandForecaster


MODELS_DIR = os.path.join(os.path.dirname(__file__), '..', 'models')


class TestCancellationClassifier:
    """Tests for the CancellationClassifier."""
    
    @pytest.fixture
    def classifier(self):
        """Load the trained classifier."""
        model_path = os.path.join(MODELS_DIR, 'cancellation_model.pkl')
        if os.path.exists(model_path):
            return CancellationClassifier.load(model_path)
        pytest.skip("Classification model not found")
    
    def test_model_loads_successfully(self, classifier):
        """Model should load without errors."""
        assert classifier is not None
    
    def test_model_has_feature_names(self, classifier):
        """Model should have feature names defined."""
        assert hasattr(classifier, 'feature_names')
        assert len(classifier.feature_names) > 0
    
    def test_predict_returns_probability(self, classifier):
        """Predict should return probability between 0 and 1."""
        # Create sample input with all required features
        sample = pd.DataFrame([{col: 0 for col in classifier.feature_names}])
        sample['lead_time'] = 30
        sample['adr'] = 100
        
        prob = classifier.predict_proba(sample)
        assert 0.0 <= prob[0] <= 1.0
    
    def test_predict_handles_dataframe(self, classifier):
        """Model should accept DataFrame input."""
        sample = pd.DataFrame([{col: 0 for col in classifier.feature_names}])
        
        # Should not raise an error
        prob = classifier.predict_proba(sample)
        assert prob is not None
    
    def test_predict_batch(self, classifier):
        """Model should handle batch predictions."""
        samples = pd.DataFrame([
            {col: i for col in classifier.feature_names}
            for i in range(5)
        ])
        
        probs = classifier.predict_proba(samples)
        assert len(probs) == 5


class TestDemandForecaster:
    """Tests for the DemandForecaster."""
    
    @pytest.fixture
    def forecaster(self):
        """Load the trained forecaster."""
        model_path = os.path.join(MODELS_DIR, 'demand_model.pkl')
        if os.path.exists(model_path):
            return DemandForecaster.load(model_path)
        pytest.skip("Demand model not found")
    
    def test_model_loads_successfully(self, forecaster):
        """Model should load without errors."""
        assert forecaster is not None
    
    def test_model_has_type(self, forecaster):
        """Model should have model_type attribute."""
        assert hasattr(forecaster, 'model_type')
    
    def test_predict_returns_dataframe(self, forecaster):
        """Predict should return a DataFrame."""
        forecast = forecaster.predict(periods=7)
        assert isinstance(forecast, pd.DataFrame)
    
    def test_predict_returns_correct_periods(self, forecaster):
        """Predict should return requested number of periods."""
        periods = 14
        forecast = forecaster.predict(periods=periods)
        assert len(forecast) == periods
    
    def test_forecast_has_required_columns(self, forecaster):
        """Forecast should have ds, yhat, yhat_lower, yhat_upper."""
        forecast = forecaster.predict(periods=7)
        
        required_cols = ['ds', 'yhat', 'yhat_lower', 'yhat_upper']
        for col in required_cols:
            assert col in forecast.columns
    
    def test_forecast_dates_are_sequential(self, forecaster):
        """Forecast dates should be sequential."""
        forecast = forecaster.predict(periods=7)
        
        dates = pd.to_datetime(forecast['ds'])
        for i in range(1, len(dates)):
            diff = (dates.iloc[i] - dates.iloc[i-1]).days
            assert diff == 1
    
    def test_forecast_bounds_are_valid(self, forecaster):
        """Lower bound <= prediction <= upper bound."""
        forecast = forecaster.predict(periods=7)
        
        assert all(forecast['yhat_lower'] <= forecast['yhat'])
        assert all(forecast['yhat'] <= forecast['yhat_upper'])
