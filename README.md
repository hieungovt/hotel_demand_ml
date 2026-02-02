# Hotel ML Project - CRISP-DM

[![CI/CD](https://github.com/YOUR_USERNAME/hotel_ml_project/actions/workflows/ci.yml/badge.svg)](https://github.com/YOUR_USERNAME/hotel_ml_project/actions/workflows/ci.yml)
[![Python 3.10](https://img.shields.io/badge/python-3.10-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

**Production-ready ML system for hotel booking cancellation prediction and demand forecasting.**

Built following the CRISP-DM methodology with FastAPI, Docker, and CI/CD.

---

## 🎯 Features

- **Cancellation Prediction**: XGBoost classifier predicts booking cancellation probability
- **Demand Forecasting**: Prophet time series model forecasts future bookings
- **REST API**: FastAPI with automatic OpenAPI documentation
- **Containerized**: Docker support for easy deployment
- **CI/CD**: GitHub Actions for automated testing and deployment
- **Cloud Ready**: One-click deploy to Render (free tier)

---

## 🚀 Quick Start

### Local Development

```bash
# Clone the repository
git clone https://github.com/YOUR_USERNAME/hotel_ml_project.git
cd hotel_ml_project

# Create conda environment
conda env create -f environment.yml
conda activate hotel_ml

# Or use pip
pip install -r requirements.txt

# Run the API
cd api
uvicorn main:app --reload
```

### Docker

```bash
# Build and run
docker-compose up --build

# Or build manually
docker build -t hotel-ml-api .
docker run -p 8000:8000 hotel-ml-api
```

---

## 📁 Project Structure

```
hotel_ml_project/
├── api/                    # FastAPI application
│   └── main.py             # API endpoints
├── data/
│   ├── raw/                # Original dataset
│   └── processed/          # Cleaned data
├── models/                 # Trained ML models
│   ├── cancellation_model.pkl
│   └── demand_model.pkl
├── notebooks/              # Jupyter notebooks (CRISP-DM)
│   ├── 01_eda.ipynb        # Exploratory Data Analysis
│   ├── 02_classification.ipynb  # Cancellation model
│   └── 03_time_series.ipynb     # Demand forecasting
├── src/                    # Source code
│   ├── preprocessing.py
│   ├── classification_model.py
│   └── time_series_model.py
├── tests/                  # Pytest tests
├── .github/workflows/      # CI/CD pipeline
├── Dockerfile
├── docker-compose.yml
├── render.yaml             # Render deployment config
└── requirements.txt
```

---

## 🔌 API Endpoints

| Method | Endpoint | Description |
|--------|----------|-------------|
| GET | `/health` | Health check & model status |
| POST | `/predict/cancellation` | Predict single booking cancellation |
| POST | `/predict/cancellation/batch` | Batch cancellation predictions |
| GET | `/predict/demand?days=30` | Forecast demand for N days |

### Example: Cancellation Prediction

```bash
curl -X POST "http://localhost:8000/predict/cancellation" \
  -H "Content-Type: application/json" \
  -d '{
    "lead_time": 45,
    "arrival_date_week_number": 27,
    "arrival_month_num": 7,
    "stays_in_weekend_nights": 2,
    "stays_in_week_nights": 5,
    "adults": 2,
    "children": 1,
    "babies": 0,
    "adr": 120.50,
    "hotel": 1,
    "meal": 1,
    "market_segment": 2,
    "distribution_channel": 1,
    "deposit_type": 0,
    "customer_type": 1,
    "season": 2
  }'
```

Response:
```json
{
  "will_cancel": false,
  "cancellation_probability": 0.23,
  "confidence": "high"
}
```

### Interactive Docs

Once running, visit:
- **Swagger UI**: http://localhost:8000/docs
- **ReDoc**: http://localhost:8000/redoc

---

## ☁️ Deploy to Render (Free Tier)

### Option 1: One-Click Deploy

[![Deploy to Render](https://render.com/images/deploy-to-render-button.svg)](https://render.com/deploy?repo=https://github.com/YOUR_USERNAME/hotel_ml_project)

### Option 2: Manual Setup

1. **Push to GitHub**
   ```bash
   git init
   git add .
   git commit -m "Initial commit"
   git remote add origin https://github.com/YOUR_USERNAME/hotel_ml_project.git
   git push -u origin main
   ```

2. **Create Render Account**
   - Go to [render.com](https://render.com) and sign up (free)

3. **Create New Web Service**
   - Click "New" → "Web Service"
   - Connect your GitHub repository
   - Select "Docker" as environment
   - Choose "Free" plan
   - Click "Create Web Service"

4. **Set Environment Variables** (in Render dashboard)
   - `PORT`: `8000`

5. **Your API will be live at**:
   `https://hotel-ml-api.onrender.com`

### Auto-Deploy with GitHub Actions

Add your Render deploy hook as a GitHub secret:
1. In Render dashboard → Your Service → Settings → Deploy Hook
2. In GitHub → Settings → Secrets → Add `RENDER_DEPLOY_HOOK`

---

## 🧪 Testing

```bash
# Install test dependencies
pip install pytest pytest-cov httpx

# Run tests
pytest tests/ -v

# With coverage
pytest tests/ -v --cov=src --cov=api
```

---

## 📊 Model Performance

| Model | Metric | Score |
|-------|--------|-------|
| Cancellation Classifier | ROC-AUC | ~0.85 |
| Cancellation Classifier | F1-Score | ~0.78 |
| Demand Forecaster | MAPE | ~15% |

---

## 🛠️ Tech Stack

- **ML**: XGBoost, Prophet, scikit-learn
- **API**: FastAPI, Pydantic, Uvicorn
- **Data**: Pandas, NumPy
- **Containerization**: Docker, Docker Compose
- **CI/CD**: GitHub Actions
- **Cloud**: Render (free tier)

---

## 📝 License

MIT License - feel free to use this project for learning or as a template.

---

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Commit changes (`git commit -m 'Add amazing feature'`)
4. Push to branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request
