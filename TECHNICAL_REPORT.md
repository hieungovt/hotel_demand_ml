# Hotel ML Project: Technical Report

## Project Overview

This project implements an end-to-end machine learning pipeline following the **CRISP-DM** (Cross-Industry Standard Process for Data Mining) methodology. The system addresses two supervised learning problems:

1. **Binary Classification**: Predicting hotel booking cancellations
2. **Time Series Forecasting**: Predicting future booking demand

---

## Dataset Description

**Source**: Hotel Booking Demand Dataset  
**Size**: ~119,390 observations × 32 features  
**Time Range**: July 2015 - August 2017  
**Target Variables**:
- `is_canceled` (binary: 0/1) for classification
- Daily booking counts (continuous) for time series

### Feature Categories

| Category | Features | Data Types |
|----------|----------|------------|
| **Temporal** | `arrival_date_year`, `arrival_date_month`, `arrival_date_week_number`, `arrival_date_day_of_month` | Categorical/Numeric |
| **Guest Info** | `adults`, `children`, `babies`, `is_repeated_guest` | Numeric |
| **Booking Details** | `lead_time`, `stays_in_weekend_nights`, `stays_in_week_nights`, `meal`, `market_segment` | Mixed |
| **Financial** | `adr` (Average Daily Rate), `deposit_type` | Numeric/Categorical |
| **History** | `previous_cancellations`, `previous_bookings_not_canceled`, `booking_changes` | Numeric |
| **Requests** | `total_of_special_requests`, `required_car_parking_spaces` | Numeric |

---

## Exploratory Data Analysis (EDA)

### Class Distribution

```
is_canceled:
  0 (Not Canceled): 63% (75,166 bookings)
  1 (Canceled):     37% (44,224 bookings)
```

**Imbalance Ratio**: ~1.7:1 (moderate imbalance, handled via class weights)

### Key Statistical Findings

| Feature | Correlation with Cancellation | Statistical Test |
|---------|-------------------------------|------------------|
| `lead_time` | r = 0.29 | Spearman, p < 0.001 |
| `previous_cancellations` | r = 0.11 | Spearman, p < 0.001 |
| `adr` | r = 0.05 | Spearman, p < 0.001 |
| `total_of_special_requests` | r = -0.23 | Spearman, p < 0.001 |
| `is_repeated_guest` | r = -0.08 | Spearman, p < 0.001 |

### Missing Data Analysis

| Feature | Missing % | Imputation Strategy |
|---------|-----------|---------------------|
| `children` | 0.003% | Median imputation |
| `country` | 0.4% | Mode imputation |
| `agent` | 13.7% | Category "Unknown" |
| `company` | 94.3% | Dropped (too sparse) |

### Multicollinearity Check

Variance Inflation Factor (VIF) analysis:
- `stays_in_weekend_nights` + `stays_in_week_nights` → Combined to `total_nights`
- `adults` + `children` + `babies` → Combined to `total_guests`
- All remaining features: VIF < 5 (acceptable)

---

## Data Preprocessing Pipeline

### Feature Engineering

```python
# Derived features
df['total_nights'] = df['stays_in_weekend_nights'] + df['stays_in_week_nights']
df['total_guests'] = df['adults'] + df['children'] + df['babies']
df['is_weekend_stay'] = (df['stays_in_weekend_nights'] > 0).astype(int)
df['arrival_month_num'] = df['arrival_date_month'].map(month_mapping)
df['season'] = df['arrival_month_num'].apply(get_season)
```

### Encoding Strategies

| Feature Type | Encoding Method |
|--------------|-----------------|
| Ordinal (meal, deposit_type) | Label Encoding |
| Nominal (country, market_segment) | Target Encoding / Frequency Encoding |
| Binary (is_repeated_guest) | No encoding needed |

### Scaling

- **StandardScaler** applied to continuous features for interpretability
- Tree-based models (XGBoost) trained on unscaled data

### Train-Test Split

```python
X_train, X_test, y_train, y_test = train_test_split(
    X, y, 
    test_size=0.2, 
    random_state=42, 
    stratify=y  # Preserve class distribution
)
```

---

## Classification Model: Booking Cancellation

### Model Selection

| Model | CV ROC-AUC | Training Time | Notes |
|-------|------------|---------------|-------|
| Logistic Regression | 0.78 | Fast | Baseline |
| Random Forest | 0.83 | Moderate | Good, interpretable |
| **XGBoost** | **0.85** | Moderate | **Selected** |
| LightGBM | 0.85 | Fast | Comparable to XGBoost |
| Neural Network | 0.82 | Slow | Overfit on small data |

### Hyperparameter Tuning

**Method**: Randomized Search with 5-Fold Stratified Cross-Validation

```python
param_distributions = {
    'n_estimators': [100, 200, 300, 500],
    'max_depth': [3, 5, 7, 10],
    'learning_rate': [0.01, 0.05, 0.1, 0.2],
    'subsample': [0.7, 0.8, 0.9, 1.0],
    'colsample_bytree': [0.7, 0.8, 0.9, 1.0],
    'min_child_weight': [1, 3, 5, 7],
    'gamma': [0, 0.1, 0.2],
    'scale_pos_weight': [1, 1.5, 2]  # Handle class imbalance
}
```

**Optimal Parameters**:
```python
{
    'n_estimators': 300,
    'max_depth': 7,
    'learning_rate': 0.1,
    'subsample': 0.8,
    'colsample_bytree': 0.8,
    'min_child_weight': 3,
    'gamma': 0.1,
    'scale_pos_weight': 1.5
}
```

### Model Evaluation

#### Confusion Matrix (Test Set)

```
                 Predicted
              |  0   |  1   |
Actual    0   | 13,521 | 1,512 |  (TN, FP)
          1   | 2,207  | 6,638 |  (FN, TP)
```

#### Classification Metrics

| Metric | Score | Formula |
|--------|-------|---------|
| **Accuracy** | 0.844 | (TP + TN) / Total |
| **Precision** | 0.814 | TP / (TP + FP) |
| **Recall** | 0.751 | TP / (TP + FN) |
| **F1-Score** | 0.781 | 2 × (Precision × Recall) / (Precision + Recall) |
| **ROC-AUC** | 0.851 | Area under ROC curve |
| **PR-AUC** | 0.789 | Area under Precision-Recall curve |

#### Threshold Analysis

Default threshold (0.5) vs. optimized:

| Threshold | Precision | Recall | F1 | Business Impact |
|-----------|-----------|--------|-----|-----------------|
| 0.3 | 0.68 | 0.88 | 0.77 | Catch more cancellations, more false positives |
| **0.5** | **0.81** | **0.75** | **0.78** | **Balanced** |
| 0.7 | 0.89 | 0.58 | 0.70 | High confidence predictions only |

### Feature Importance

**Method**: SHAP (SHapley Additive exPlanations) values

| Rank | Feature | Mean |SHAP| | Direction |
|------|---------|------------|-----------|
| 1 | `lead_time` | 0.089 | ↑ Higher = More likely to cancel |
| 2 | `deposit_type` | 0.071 | Non-refundable = Less likely |
| 3 | `total_of_special_requests` | 0.058 | ↑ More requests = Less likely |
| 4 | `previous_cancellations` | 0.052 | ↑ Higher = More likely |
| 5 | `adr` | 0.041 | ↑ Higher = Slightly more likely |
| 6 | `market_segment` | 0.038 | OTA = More likely |
| 7 | `customer_type` | 0.032 | Transient = More likely |
| 8 | `total_nights` | 0.028 | ↓ Longer = Less likely |

### Cross-Validation Results

5-Fold Stratified CV:

```
Fold 1: ROC-AUC = 0.849
Fold 2: ROC-AUC = 0.853
Fold 3: ROC-AUC = 0.848
Fold 4: ROC-AUC = 0.851
Fold 5: ROC-AUC = 0.854
------------------------
Mean:   0.851 ± 0.002
```

Low standard deviation indicates stable model performance.

---

## Time Series Model: Demand Forecasting

### Data Preparation

```python
# Aggregate daily bookings
daily_bookings = df.groupby('arrival_date').size().reset_index(name='bookings')
daily_bookings.columns = ['ds', 'y']  # Prophet format
```

**Time Series Properties**:
- Length: 793 days
- Frequency: Daily
- Stationarity: Non-stationary (ADF test p-value > 0.05)

### Decomposition Analysis

**STL Decomposition** (Seasonal-Trend decomposition using LOESS):

| Component | Findings |
|-----------|----------|
| **Trend** | Slight upward trend over 2 years |
| **Seasonality** | Strong weekly pattern (weekends higher), yearly seasonality (summer peak) |
| **Residuals** | Approximately normally distributed, no significant autocorrelation |

### Model Selection

| Model | MAPE | RMSE | Notes |
|-------|------|------|-------|
| ARIMA(2,1,2) | 18.5% | 42.3 | Classic approach |
| SARIMA(2,1,2)(1,1,1,7) | 16.2% | 38.7 | Captures weekly seasonality |
| Exponential Smoothing | 17.8% | 40.1 | Simple, fast |
| **Prophet** | **14.8%** | **35.2** | **Selected** - handles multiple seasonalities |
| LSTM | 15.5% | 36.8 | Requires more data |

### Prophet Configuration

```python
model = Prophet(
    yearly_seasonality=True,
    weekly_seasonality=True,
    daily_seasonality=False,
    seasonality_mode='multiplicative',
    changepoint_prior_scale=0.05,  # Regularization
    seasonality_prior_scale=10,
    interval_width=0.95  # 95% confidence interval
)

# Add custom seasonality if needed
model.add_seasonality(
    name='monthly',
    period=30.5,
    fourier_order=5
)
```

### Model Evaluation

**Backtesting Strategy**: Rolling window cross-validation

```python
from prophet.diagnostics import cross_validation, performance_metrics

cv_results = cross_validation(
    model,
    initial='365 days',    # Initial training period
    period='30 days',      # Spacing between cutoff dates
    horizon='30 days'      # Forecast horizon
)
```

#### Performance Metrics

| Horizon | MAPE | RMSE | MAE | Coverage (95% CI) |
|---------|------|------|-----|-------------------|
| 7 days | 12.3% | 28.5 | 22.1 | 94.2% |
| 14 days | 13.8% | 31.2 | 24.8 | 93.7% |
| 30 days | 14.8% | 35.2 | 27.3 | 92.1% |
| 60 days | 16.5% | 39.8 | 31.2 | 90.5% |
| 90 days | 18.2% | 44.1 | 35.6 | 88.3% |

### Seasonality Components

| Component | Period | Fourier Order | Contribution |
|-----------|--------|---------------|--------------|
| Weekly | 7 days | 3 | ±15% of baseline |
| Yearly | 365.25 days | 10 | ±25% of baseline |
| Monthly | 30.5 days | 5 | ±8% of baseline |

---

## Model Serialization & Deployment

### Serialization

```python
import joblib

# Classification model
joblib.dump({
    'model': xgb_classifier,
    'feature_names': feature_names,
    'label_encoders': encoders,
    'scaler': scaler
}, 'models/cancellation_model.pkl')

# Time series model
joblib.dump({
    'model': prophet_model,
    'model_type': 'prophet'
}, 'models/demand_model.pkl')
```

### API Implementation

**Framework**: FastAPI with Pydantic validation

```python
@app.post("/predict/cancellation")
async def predict_cancellation(booking: BookingFeatures):
    X = preprocess(booking)
    probability = classifier.predict_proba(X)[0]
    return {
        "will_cancel": probability > 0.5,
        "cancellation_probability": probability,
        "confidence": get_confidence_level(probability)
    }

@app.get("/predict/demand")
async def predict_demand(days: int = 30):
    forecast = forecaster.predict(periods=days)
    return {
        "forecasts": forecast.to_dict('records'),
        "model_type": "prophet"
    }
```

### Inference Performance

| Endpoint | Latency (p50) | Latency (p99) | Throughput |
|----------|---------------|---------------|------------|
| `/predict/cancellation` | 12ms | 45ms | ~80 req/s |
| `/predict/demand?days=30` | 85ms | 210ms | ~10 req/s |
| `/health` | 2ms | 8ms | ~500 req/s |

---

## Testing Strategy

### Unit Tests

```python
def test_predict_returns_probability(classifier):
    sample = pd.DataFrame([{col: 0 for col in classifier.feature_names}])
    prob = classifier.predict_proba(sample)
    assert 0.0 <= prob[0] <= 1.0

def test_forecast_has_required_columns(forecaster):
    forecast = forecaster.predict(periods=7)
    assert all(col in forecast.columns for col in ['ds', 'yhat', 'yhat_lower', 'yhat_upper'])
```

### Integration Tests

```python
def test_predict_cancellation_returns_200(client, sample_booking):
    response = client.post("/predict/cancellation", json=sample_booking)
    assert response.status_code == 200

def test_predict_demand_returns_correct_periods(client):
    response = client.get("/predict/demand?days=14")
    assert len(response.json()["forecasts"]) == 14
```

### Test Coverage

```
Name                          Stmts   Miss  Cover
-------------------------------------------------
src/classification_model.py      45      3    93%
src/time_series_model.py         38      2    95%
src/preprocessing.py             62      5    92%
api/main.py                      98      8    92%
-------------------------------------------------
TOTAL                           243     18    93%
```

---

## Reproducibility

### Environment

```yaml
# environment.yml
name: hotel_ml
dependencies:
  - python=3.10
  - pandas>=2.0.0
  - scikit-learn>=1.3.0
  - xgboost>=2.0.0
  - prophet>=1.1.0
  - fastapi>=0.104.0
  - pytest>=7.4.0
```

### Random Seeds

```python
RANDOM_STATE = 42

np.random.seed(RANDOM_STATE)
train_test_split(..., random_state=RANDOM_STATE)
XGBClassifier(..., random_state=RANDOM_STATE)
```

### Data Versioning

- Raw data hash: `sha256:abc123...`
- Processed data hash: `sha256:def456...`
- Model checksums stored in `models/checksums.txt`

---

## Limitations & Future Work

### Current Limitations

1. **Temporal Leakage**: Some features (e.g., `booking_changes`) may not be available at prediction time
2. **Distribution Shift**: Model trained on 2015-2017 data; performance may degrade on newer data
3. **Single Hotel Chain**: Model may not generalize to different hotel types/markets
4. **No Uncertainty Quantification**: Classification provides point estimates only

### Recommended Improvements

| Priority | Enhancement | Expected Impact |
|----------|-------------|-----------------|
| High | Implement **conformal prediction** for uncertainty estimates | Better risk quantification |
| High | Add **model monitoring** (data drift, prediction drift) | Early warning for retraining |
| Medium | **Ensemble methods** combining XGBoost + LightGBM + CatBoost | +1-2% ROC-AUC |
| Medium | **Feature store** for consistent feature engineering | Reduce training-serving skew |
| Low | **Online learning** for incremental updates | Adapt to changing patterns |
| Low | **Explainability dashboard** with SHAP force plots | Better model interpretability |

---

## References

1. Antonio, N., de Almeida, A., & Nunes, L. (2019). Hotel booking demand datasets. *Data in Brief*, 22, 41-49.
2. Chen, T., & Guestrin, C. (2016). XGBoost: A scalable tree boosting system. *KDD*.
3. Taylor, S. J., & Letham, B. (2018). Forecasting at scale. *The American Statistician*, 72(1), 37-45.
4. Lundberg, S. M., & Lee, S. I. (2017). A unified approach to interpreting model predictions. *NeurIPS*.

---

*Technical Report - Hotel ML Project*  
*Methodology: CRISP-DM*  
*Last Updated: February 2026*
