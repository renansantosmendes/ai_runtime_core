# Fetal Health Classification API

REST API for predicting fetal health status using machine learning models.

## Features

- **Single Predictions**: Make predictions for individual fetal health records
- **Batch Predictions**: Process multiple records efficiently
- **Multiple Models**: Support for Decision Tree and Gradient Boosting models
- **Auto-documentation**: Interactive API documentation with Swagger UI
- **Health Checks**: Monitor API and model status
- **CORS Support**: Cross-origin resource sharing enabled

## Installation

### Prerequisites

- Python 3.10 or higher
- pip package manager

### Setup

1. Navigate to the API directory:
```bash
cd ia_solutions/api
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

## Running the API

### Development Mode

Start the API server with auto-reload:

```bash
uvicorn main:app --reload --host 0.0.0.0 --port 8000
```

### Production Mode

Start the API server in production:

```bash
uvicorn main:app --host 0.0.0.0 --port 8000 --workers 4
```

The API will be available at: `http://localhost:8000`

## API Documentation

Once the server is running, access the interactive documentation:

- **Swagger UI**: http://localhost:8000/docs
- **ReDoc**: http://localhost:8000/redoc

## API Endpoints

### Health Check

**GET /** or **GET /health**

Check if the API is running and models are loaded.

**Response:**
```json
{
  "status": "healthy",
  "message": "All systems operational",
  "models_loaded": ["decision_tree", "gradient_boosting"]
}
```

### List Models

**GET /models**

Get information about all available models.

**Response:**
```json
[
  {
    "name": "gradient_boosting",
    "type": "GradientBoostingClassifier",
    "loaded": true,
    "file_path": "ia_solutions/models/gradient_boosting_model.pkl"
  },
  {
    "name": "decision_tree",
    "type": "DecisionTreeClassifier",
    "loaded": true,
    "file_path": "ia_solutions/models/decision_tree_model.pkl"
  }
]
```

### Single Prediction

**POST /predict**

Make a prediction for a single fetal health record.

**Request Body:**
```json
{
  "features": {
    "baseline_value": 120.0,
    "accelerations": 0.0,
    "fetal_movement": 0.0,
    "uterine_contractions": 0.0,
    "light_decelerations": 0.0,
    "severe_decelerations": 0.0,
    "prolongued_decelerations": 0.0,
    "abnormal_short_term_variability": 73.0,
    "mean_value_of_short_term_variability": 0.5,
    "percentage_of_time_with_abnormal_long_term_variability": 43.0,
    "mean_value_of_long_term_variability": 2.4,
    "histogram_width": 64.0,
    "histogram_min": 62.0,
    "histogram_max": 126.0,
    "histogram_number_of_peaks": 2.0,
    "histogram_number_of_zeroes": 0.0,
    "histogram_mode": 120.0,
    "histogram_mean": 137.0,
    "histogram_median": 121.0,
    "histogram_variance": 73.0,
    "histogram_tendency": 1.0
  },
  "model_name": "gradient_boosting"
}
```

**Response:**
```json
{
  "prediction_code": 1.0,
  "health_status": "Normal",
  "model_used": "gradient_boosting",
  "confidence": 0.95
}
```

### Batch Predictions

**POST /predict/batch**

Make predictions for multiple fetal health records.

**Request Body:**
```json
{
  "features_list": [
    {
      "baseline_value": 120.0,
      "accelerations": 0.0,
      ...
    },
    {
      "baseline_value": 132.0,
      "accelerations": 0.006,
      ...
    }
  ],
  "model_name": "gradient_boosting"
}
```

**Response:**
```json
{
  "predictions": [
    {
      "prediction_code": 1.0,
      "health_status": "Normal",
      "model_used": "gradient_boosting",
      "confidence": 0.95
    },
    {
      "prediction_code": 2.0,
      "health_status": "Suspect",
      "model_used": "gradient_boosting",
      "confidence": 0.87
    }
  ]
}
```

## Health Status Categories

The API classifies fetal health into three categories:

- **Normal (1.0)**: Healthy fetal condition
- **Suspect (2.0)**: Requires medical attention
- **Pathological (3.0)**: Requires immediate medical intervention

## Available Models

- **decision_tree**: Decision Tree Classifier
- **gradient_boosting**: Gradient Boosting Classifier (default)

## Error Handling

The API returns appropriate HTTP status codes:

- **200**: Successful request
- **400**: Bad request (invalid input)
- **404**: Resource not found
- **500**: Internal server error
- **503**: Service unavailable (models not loaded)

## Example Usage

### Using cURL

```bash
# Health check
curl http://localhost:8000/health

# Make a prediction
curl -X POST http://localhost:8000/predict \
  -H "Content-Type: application/json" \
  -d '{
    "features": {
      "baseline_value": 120.0,
      "accelerations": 0.0,
      "fetal_movement": 0.0,
      "uterine_contractions": 0.0,
      "light_decelerations": 0.0,
      "severe_decelerations": 0.0,
      "prolongued_decelerations": 0.0,
      "abnormal_short_term_variability": 73.0,
      "mean_value_of_short_term_variability": 0.5,
      "percentage_of_time_with_abnormal_long_term_variability": 43.0,
      "mean_value_of_long_term_variability": 2.4,
      "histogram_width": 64.0,
      "histogram_min": 62.0,
      "histogram_max": 126.0,
      "histogram_number_of_peaks": 2.0,
      "histogram_number_of_zeroes": 0.0,
      "histogram_mode": 120.0,
      "histogram_mean": 137.0,
      "histogram_median": 121.0,
      "histogram_variance": 73.0,
      "histogram_tendency": 1.0
    },
    "model_name": "gradient_boosting"
  }'
```

### Using Python

```python
import requests

# API base URL
BASE_URL = "http://localhost:8000"

# Health check
response = requests.get(f"{BASE_URL}/health")
print(response.json())

# Make a prediction
features = {
    "baseline_value": 120.0,
    "accelerations": 0.0,
    "fetal_movement": 0.0,
    "uterine_contractions": 0.0,
    "light_decelerations": 0.0,
    "severe_decelerations": 0.0,
    "prolongued_decelerations": 0.0,
    "abnormal_short_term_variability": 73.0,
    "mean_value_of_short_term_variability": 0.5,
    "percentage_of_time_with_abnormal_long_term_variability": 43.0,
    "mean_value_of_long_term_variability": 2.4,
    "histogram_width": 64.0,
    "histogram_min": 62.0,
    "histogram_max": 126.0,
    "histogram_number_of_peaks": 2.0,
    "histogram_number_of_zeroes": 0.0,
    "histogram_mode": 120.0,
    "histogram_mean": 137.0,
    "histogram_median": 121.0,
    "histogram_variance": 73.0,
    "histogram_tendency": 1.0
}

response = requests.post(
    f"{BASE_URL}/predict",
    json={"features": features, "model_name": "gradient_boosting"}
)
print(response.json())
```

## Project Structure

```
ia_solutions/api/
├── __init__.py          # Package initialization
├── main.py              # FastAPI application and endpoints
├── models.py            # Model manager and ML logic
├── schemas.py           # Pydantic models for validation
├── requirements.txt     # Python dependencies
└── README.md           # This file
```

## Development

### Running Tests

```bash
pytest
```

### Code Quality

The code follows:
- PEP 8 style guidelines
- Clean Code principles
- Type hints for better IDE support
- Comprehensive docstrings

## License

This project is part of PUC Lectures 2025.

## Support

For issues or questions, please refer to the project documentation or contact the development team.
