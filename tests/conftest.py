"""
Pytest fixtures for Hotel ML Project tests.
"""

import pytest
import sys
import os

# Add project root to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from fastapi.testclient import TestClient
from api.main import app


@pytest.fixture
def client():
    """Create a test client for the FastAPI app."""
    with TestClient(app) as test_client:
        yield test_client


@pytest.fixture
def sample_booking():
    """Sample booking data for cancellation prediction."""
    return {
        "lead_time": 45,
        "arrival_date_week_number": 27,
        "arrival_month_num": 7,
        "stays_in_weekend_nights": 2,
        "stays_in_week_nights": 5,
        "adults": 2,
        "children": 1,
        "babies": 0,
        "is_repeated_guest": 0,
        "previous_cancellations": 0,
        "previous_bookings_not_canceled": 0,
        "booking_changes": 0,
        "days_in_waiting_list": 0,
        "adr": 120.50,
        "required_car_parking_spaces": 1,
        "total_of_special_requests": 2,
        "hotel": 1,
        "meal": 1,
        "market_segment": 2,
        "distribution_channel": 1,
        "deposit_type": 0,
        "customer_type": 1,
        "season": 2,
    }


@pytest.fixture
def high_risk_booking():
    """High-risk booking likely to cancel."""
    return {
        "lead_time": 300,  # Very long lead time
        "arrival_date_week_number": 1,
        "arrival_month_num": 1,
        "stays_in_weekend_nights": 0,
        "stays_in_week_nights": 1,
        "adults": 1,
        "children": 0,
        "babies": 0,
        "is_repeated_guest": 0,
        "previous_cancellations": 3,  # Has cancelled before
        "previous_bookings_not_canceled": 0,
        "booking_changes": 0,
        "days_in_waiting_list": 50,  # Long wait
        "adr": 50.0,
        "required_car_parking_spaces": 0,
        "total_of_special_requests": 0,
        "hotel": 1,
        "meal": 0,
        "market_segment": 0,
        "distribution_channel": 0,
        "deposit_type": 0,
        "customer_type": 0,
        "season": 0,
    }
