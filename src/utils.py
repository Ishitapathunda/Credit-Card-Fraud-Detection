"""
Utility functions for credit card fraud detection project.
"""

import numpy as np
import pandas as pd
import joblib
import os


def load_data(filepath):
    """
    Load the credit card fraud detection dataset.
    
    Args:
        filepath: Path to the CSV file
        
    Returns:
        DataFrame with the loaded data
    """
    if not os.path.exists(filepath):
        raise FileNotFoundError(f"Dataset not found at {filepath}. Please download from Kaggle.")
    
    df = pd.read_csv(filepath)
    return df


def save_model(model, filepath):
    """
    Save a trained model using joblib.
    
    Args:
        model: Trained model object
        filepath: Path where to save the model
    """
    os.makedirs(os.path.dirname(filepath), exist_ok=True)
    joblib.dump(model, filepath)
    print(f"Model saved to {filepath}")


def load_model(filepath):
    """
    Load a saved model using joblib.
    
    Args:
        filepath: Path to the saved model
        
    Returns:
        Loaded model object
    """
    if not os.path.exists(filepath):
        raise FileNotFoundError(f"Model not found at {filepath}")
    
    model = joblib.load(filepath)
    return model


def get_feature_names():
    """
    Get the list of feature names (excluding Time, Amount, and Class).
    
    Returns:
        List of feature names
    """
    return [f'V{i}' for i in range(1, 29)]


def print_separator(title=""):
    """
    Print a separator line for better output formatting.
    
    Args:
        title: Optional title to print
    """
    if title:
        print(f"\n{'='*60}")
        print(f"  {title}")
        print(f"{'='*60}\n")
    else:
        print(f"{'='*60}\n")

