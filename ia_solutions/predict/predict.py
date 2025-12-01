"""
Module for making predictions using trained fetal health classification models.

This script loads a trained model and makes predictions on new data,
demonstrating how to use the saved models for inference.
"""

import os
import pickle

import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler


# Constants
MODELS_DIR = 'ia_solutions/models'
DECISION_TREE_MODEL_PATH = os.path.join(MODELS_DIR, 'decision_tree_model.pkl')
GRADIENT_BOOSTING_MODEL_PATH = os.path.join(MODELS_DIR, 'gradient_boosting_model.pkl')
DATA_URL = 'https://raw.githubusercontent.com/renansantosmendes/lectures-cdas-2023/master/fetal_health_reduced.csv'


def load_model(model_path: str):
    """
    Load a trained model from disk.
    
    Args:
        model_path: Path to the saved model file
        
    Returns:
        Loaded model object
        
    Raises:
        FileNotFoundError: If model file doesn't exist
    """
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model file not found: {model_path}")
    
    with open(model_path, 'rb') as file:
        model = pickle.load(file)
    
    print(f"[OK] Model loaded successfully from: {model_path}")
    return model


def load_sample_data(url: str, n_samples: int = 5) -> pd.DataFrame:
    """
    Load sample data for prediction demonstration.
    
    Args:
        url: URL of the CSV file containing the data
        n_samples: Number of samples to load for prediction
        
    Returns:
        DataFrame with sample data
    """
    print(f"Loading {n_samples} sample records for prediction...")
    data = pd.read_csv(url)
    
    # Get features only (exclude target column)
    features = data.iloc[:, :-1]
    
    # Select random samples
    sample_data = features.sample(n=n_samples, random_state=42)
    
    print(f"[OK] Loaded {len(sample_data)} samples")
    return sample_data


def preprocess_features(features: pd.DataFrame) -> pd.DataFrame:
    """
    Preprocess features by applying standardization.
    
    Args:
        features: Raw features DataFrame
        
    Returns:
        Preprocessed features DataFrame
    """
    print("Preprocessing features...")
    scaler = StandardScaler()
    scaled_features_array = scaler.fit_transform(features)
    scaled_features = pd.DataFrame(
        scaled_features_array,
        columns=features.columns,
        index=features.index
    )
    print("[OK] Features preprocessed")
    return scaled_features


def make_predictions(model, features: pd.DataFrame) -> np.ndarray:
    """
    Make predictions using the loaded model.
    
    Args:
        model: Trained model
        features: Preprocessed features for prediction
        
    Returns:
        Array of predictions
    """
    print("Making predictions...")
    predictions = model.predict(features)
    print("[OK] Predictions completed")
    return predictions


def interpret_predictions(predictions: np.ndarray) -> list[str]:
    """
    Interpret numerical predictions into human-readable labels.
    
    Args:
        predictions: Array of numerical predictions
        
    Returns:
        List of interpreted labels
    """
    # Fetal health classification mapping
    health_mapping = {
        1.0: "Normal",
        2.0: "Suspect",
        3.0: "Pathological"
    }
    
    return [health_mapping.get(pred, "Unknown") for pred in predictions]


def display_predictions(
    features: pd.DataFrame,
    predictions: np.ndarray,
    labels: list[str]
) -> None:
    """
    Display predictions in a formatted table.
    
    Args:
        features: Original features DataFrame
        predictions: Numerical predictions
        labels: Interpreted prediction labels
    """
    print("\n" + "="*80)
    print("PREDICTION RESULTS")
    print("="*80)
    
    results_df = pd.DataFrame({
        'Sample_Index': features.index,
        'Prediction_Code': predictions,
        'Fetal_Health_Status': labels
    })
    
    print(results_df.to_string(index=False))
    print("="*80)
    
    # Summary statistics
    print("\nSummary:")
    for label in set(labels):
        count = labels.count(label)
        percentage = (count / len(labels)) * 100
        print(f"  {label}: {count} samples ({percentage:.1f}%)")


def main():
    """Main function to execute the prediction pipeline."""
    print("="*80)
    print("FETAL HEALTH PREDICTION SYSTEM")
    print("="*80)
    print()
    
    # Choose which model to use
    model_choice = "gradient_boosting"  # Options: "decision_tree" or "gradient_boosting"
    
    if model_choice == "decision_tree":
        model_path = DECISION_TREE_MODEL_PATH
        print("Using: Decision Tree Classifier")
    else:
        model_path = GRADIENT_BOOSTING_MODEL_PATH
        print("Using: Gradient Boosting Classifier")
    
    print()
    
    # Load the trained model
    model = load_model(model_path)
    print()
    
    # Load sample data
    sample_features = load_sample_data(DATA_URL, n_samples=10)
    print()
    
    # Preprocess features
    preprocessed_features = preprocess_features(sample_features)
    print()
    
    # Make predictions
    predictions = make_predictions(model, preprocessed_features)
    print()
    
    # Interpret predictions
    prediction_labels = interpret_predictions(predictions)
    
    # Display results
    display_predictions(sample_features, predictions, prediction_labels)
    
    print("\n[OK] Prediction pipeline completed successfully!")


if __name__ == "__main__":
    main()
