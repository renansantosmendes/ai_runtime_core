"""
Example script to test the Fetal Health Classification API.

This script demonstrates how to interact with the API endpoints.
"""

import requests
import json


# API Configuration
BASE_URL = "http://localhost:8000"


def print_section(title: str):
    """Print a formatted section title."""
    print("\n" + "="*80)
    print(f" {title}")
    print("="*80 + "\n")


def test_health_check():
    """Test the health check endpoint."""
    print_section("Testing Health Check Endpoint")
    
    response = requests.get(f"{BASE_URL}/health")
    print(f"Status Code: {response.status_code}")
    print(f"Response: {json.dumps(response.json(), indent=2)}")


def test_list_models():
    """Test the list models endpoint."""
    print_section("Testing List Models Endpoint")
    
    response = requests.get(f"{BASE_URL}/models")
    print(f"Status Code: {response.status_code}")
    print(f"Response: {json.dumps(response.json(), indent=2)}")


def test_single_prediction():
    """Test single prediction endpoint."""
    print_section("Testing Single Prediction Endpoint")
    
    # Example features (Normal case)
    payload = {
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
    
    response = requests.post(f"{BASE_URL}/predict", json=payload)
    print(f"Status Code: {response.status_code}")
    print(f"Response: {json.dumps(response.json(), indent=2)}")


def test_batch_prediction():
    """Test batch prediction endpoint."""
    print_section("Testing Batch Prediction Endpoint")
    
    # Example features for batch prediction
    payload = {
        "features_list": [
            {
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
            {
                "baseline_value": 132.0,
                "accelerations": 0.006,
                "fetal_movement": 0.0,
                "uterine_contractions": 0.006,
                "light_decelerations": 0.003,
                "severe_decelerations": 0.0,
                "prolongued_decelerations": 0.0,
                "abnormal_short_term_variability": 17.0,
                "mean_value_of_short_term_variability": 2.1,
                "percentage_of_time_with_abnormal_long_term_variability": 0.0,
                "mean_value_of_long_term_variability": 10.4,
                "histogram_width": 130.0,
                "histogram_min": 68.0,
                "histogram_max": 198.0,
                "histogram_number_of_peaks": 6.0,
                "histogram_number_of_zeroes": 1.0,
                "histogram_mode": 141.0,
                "histogram_mean": 136.0,
                "histogram_median": 140.0,
                "histogram_variance": 12.0,
                "histogram_tendency": 0.0
            }
        ],
        "model_name": "gradient_boosting"
    }
    
    response = requests.post(f"{BASE_URL}/predict/batch", json=payload)
    print(f"Status Code: {response.status_code}")
    print(f"Response: {json.dumps(response.json(), indent=2)}")


def test_different_models():
    """Test predictions with different models."""
    print_section("Testing Different Models")
    
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
    
    for model_name in ["decision_tree", "gradient_boosting"]:
        print(f"\nTesting with {model_name}:")
        payload = {"features": features, "model_name": model_name}
        response = requests.post(f"{BASE_URL}/predict", json=payload)
        print(f"  Status Code: {response.status_code}")
        print(f"  Prediction: {response.json()['health_status']}")
        if response.json().get('confidence'):
            print(f"  Confidence: {response.json()['confidence']:.2%}")


def main():
    """Run all tests."""
    print("\n" + "#"*80)
    print("# Fetal Health Classification API - Test Suite")
    print("#"*80)
    
    try:
        # Test all endpoints
        test_health_check()
        test_list_models()
        test_single_prediction()
        test_batch_prediction()
        test_different_models()
        
        print_section("All Tests Completed Successfully!")
        
    except requests.exceptions.ConnectionError:
        print("\n[ERROR] Could not connect to the API.")
        print("Make sure the API is running at:", BASE_URL)
        print("\nStart the API with:")
        print("  uvicorn main:app --reload")
    except Exception as e:
        print(f"\n[ERROR] An error occurred: {str(e)}")


if __name__ == "__main__":
    main()
