"""
Integration tests for the FastAPI endpoints.
"""

import pytest


class TestHealthEndpoint:
    """Tests for the /health endpoint."""
    
    def test_health_check_returns_200(self, client):
        """Health endpoint should return 200 status."""
        response = client.get("/health")
        assert response.status_code == 200
    
    def test_health_check_response_structure(self, client):
        """Health endpoint should return correct structure."""
        response = client.get("/health")
        data = response.json()
        
        assert "status" in data
        assert "models_loaded" in data
        assert "classification" in data["models_loaded"]
        assert "time_series" in data["models_loaded"]
    
    def test_health_check_models_loaded(self, client):
        """Health endpoint should show models as loaded."""
        response = client.get("/health")
        data = response.json()
        
        assert data["status"] == "healthy"
        # Models should be loaded if pkl files exist
        assert isinstance(data["models_loaded"]["classification"], bool)
        assert isinstance(data["models_loaded"]["time_series"], bool)


class TestCancellationPrediction:
    """Tests for the /predict/cancellation endpoint."""
    
    def test_predict_cancellation_returns_200(self, client, sample_booking):
        """Cancellation prediction should return 200."""
        response = client.post("/predict/cancellation", json=sample_booking)
        assert response.status_code == 200
    
    def test_predict_cancellation_response_structure(self, client, sample_booking):
        """Cancellation prediction should return correct structure."""
        response = client.post("/predict/cancellation", json=sample_booking)
        data = response.json()
        
        assert "will_cancel" in data
        assert "cancellation_probability" in data
        assert "confidence" in data
    
    def test_predict_cancellation_probability_range(self, client, sample_booking):
        """Probability should be between 0 and 1."""
        response = client.post("/predict/cancellation", json=sample_booking)
        data = response.json()
        
        prob = data["cancellation_probability"]
        assert 0.0 <= prob <= 1.0
    
    def test_predict_cancellation_confidence_values(self, client, sample_booking):
        """Confidence should be low, medium, or high."""
        response = client.post("/predict/cancellation", json=sample_booking)
        data = response.json()
        
        assert data["confidence"] in ["low", "medium", "high"]
    
    def test_predict_cancellation_boolean_result(self, client, sample_booking):
        """will_cancel should be a boolean."""
        response = client.post("/predict/cancellation", json=sample_booking)
        data = response.json()
        
        assert isinstance(data["will_cancel"], bool)
    
    def test_predict_cancellation_invalid_input(self, client):
        """Should return 422 for invalid input."""
        response = client.post("/predict/cancellation", json={"invalid": "data"})
        assert response.status_code == 422
    
    def test_predict_cancellation_missing_required_fields(self, client):
        """Should return 422 for missing required fields."""
        incomplete_booking = {"lead_time": 45}  # Missing other required fields
        response = client.post("/predict/cancellation", json=incomplete_booking)
        assert response.status_code == 422


class TestDemandForecast:
    """Tests for the /predict/demand endpoint."""
    
    def test_predict_demand_returns_200(self, client):
        """Demand forecast should return 200."""
        response = client.get("/predict/demand?days=7")
        assert response.status_code == 200
    
    def test_predict_demand_default_days(self, client):
        """Demand forecast should work with default days parameter."""
        response = client.get("/predict/demand")
        assert response.status_code == 200
    
    def test_predict_demand_response_structure(self, client):
        """Demand forecast should return correct structure."""
        response = client.get("/predict/demand?days=7")
        data = response.json()
        
        assert "forecasts" in data
        assert "model_type" in data
        assert isinstance(data["forecasts"], list)
    
    def test_predict_demand_forecast_count(self, client):
        """Should return correct number of forecasts."""
        days = 7
        response = client.get(f"/predict/demand?days={days}")
        data = response.json()
        
        assert len(data["forecasts"]) == days
    
    def test_predict_demand_forecast_fields(self, client):
        """Each forecast should have required fields."""
        response = client.get("/predict/demand?days=3")
        data = response.json()
        
        for forecast in data["forecasts"]:
            assert "date" in forecast
            assert "predicted_bookings" in forecast
            assert "lower_bound" in forecast
            assert "upper_bound" in forecast
    
    def test_predict_demand_bounds_logic(self, client):
        """Lower bound should be <= prediction <= upper bound."""
        response = client.get("/predict/demand?days=7")
        data = response.json()
        
        for forecast in data["forecasts"]:
            assert forecast["lower_bound"] <= forecast["predicted_bookings"]
            assert forecast["predicted_bookings"] <= forecast["upper_bound"]
    
    def test_predict_demand_invalid_days_too_high(self, client):
        """Should return 400 for days > 90."""
        response = client.get("/predict/demand?days=100")
        assert response.status_code == 400
    
    def test_predict_demand_invalid_days_too_low(self, client):
        """Should return 400 for days < 1."""
        response = client.get("/predict/demand?days=0")
        assert response.status_code == 400


class TestBatchPrediction:
    """Tests for batch cancellation prediction."""
    
    def test_batch_prediction_returns_200(self, client, sample_booking):
        """Batch prediction should return 200."""
        response = client.post(
            "/predict/cancellation/batch",
            json=[sample_booking, sample_booking]
        )
        assert response.status_code == 200
    
    def test_batch_prediction_response_count(self, client, sample_booking):
        """Should return predictions for all bookings."""
        bookings = [sample_booking, sample_booking, sample_booking]
        response = client.post("/predict/cancellation/batch", json=bookings)
        data = response.json()
        
        assert len(data["predictions"]) == len(bookings)
    
    def test_batch_prediction_empty_list(self, client):
        """Should handle empty list."""
        response = client.post("/predict/cancellation/batch", json=[])
        assert response.status_code == 200
        assert response.json()["predictions"] == []
