"""
Main training script for Credit Card Fraud Detection.
Run this script to train all models and save the best one.
"""

import joblib
import os
from src.preprocessing import preprocess_data
from src.train_models import train_all_models
from src.utils import print_separator, save_model

if __name__ == "__main__":
    print_separator("CREDIT CARD FRAUD DETECTION - MODEL TRAINING")
    
    # Dataset path
    dataset_path = 'data/creditcard.csv'
    
    print(f"Dataset path: {dataset_path}")
    print("\nStarting preprocessing...")
    
    # Preprocess data
    X_train, X_val, X_test, y_train, y_val, y_test, scaler = preprocess_data(
        filepath=dataset_path,
        test_size=0.2,
        val_size=0.1,
        random_state=42,
        remove_outliers=True,
        outlier_method='zscore',
        use_smote=True
    )
    
    print("\nStarting model training...")
    
    # Train all models
    best_model, all_models, metrics_dict, test_results = train_all_models(
        X_train, y_train, X_val, y_val, X_test, y_test,
        save_best=True,
        model_dir='models'
    )
    
    # Save scaler for inference
    os.makedirs('models', exist_ok=True)
    scaler_path = 'models/scaler.pkl'
    joblib.dump(scaler, scaler_path)
    print(f"✅ Scaler saved to {scaler_path}")
    
    print_separator("TRAINING COMPLETE")
    print("\n✅ All models trained successfully!")
    print("✅ Best model saved to models/final_model.pkl")
    print("✅ Scaler saved to models/scaler.pkl")
    print("✅ Evaluation plots saved to models/ directory")
    print("\nYou can now run the Streamlit app:")
    print("  streamlit run app/streamlit_app.py")

