"""
Train and compare multiple ML models for credit card fraud detection.
"""

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from sklearn.metrics import (
    confusion_matrix, classification_report, 
    precision_score, recall_score, f1_score, 
    roc_auc_score, roc_curve
)
import matplotlib.pyplot as plt
import joblib
import os
from src.evaluate import evaluate_model
from src.utils import save_model, print_separator


def train_logistic_regression(X_train, y_train, X_val, y_val, random_state=42):
    """
    Train Logistic Regression model.
    
    Args:
        X_train: Training features
        y_train: Training labels
        X_val: Validation features
        y_val: Validation labels
        random_state: Random seed
        
    Returns:
        Trained model and evaluation metrics
    """
    print_separator("Training Logistic Regression")
    
    model = LogisticRegression(
        random_state=random_state,
        max_iter=1000,
        class_weight='balanced',
        solver='lbfgs'
    )
    
    model.fit(X_train, y_train)
    
    # Evaluate
    y_pred = model.predict(X_val)
    y_pred_proba = model.predict_proba(X_val)[:, 1]
    
    metrics = evaluate_model(y_val, y_pred, y_pred_proba, "Logistic Regression")
    
    return model, metrics


def train_random_forest(X_train, y_train, X_val, y_val, random_state=42):
    """
    Train Random Forest model.
    
    Args:
        X_train: Training features
        y_train: Training labels
        X_val: Validation features
        y_val: Validation labels
        random_state: Random seed
        
    Returns:
        Trained model and evaluation metrics
    """
    print_separator("Training Random Forest")
    
    model = RandomForestClassifier(
        n_estimators=100,
        max_depth=10,
        min_samples_split=5,
        min_samples_leaf=2,
        class_weight='balanced',
        random_state=random_state,
        n_jobs=-1
    )
    
    model.fit(X_train, y_train)
    
    # Evaluate
    y_pred = model.predict(X_val)
    y_pred_proba = model.predict_proba(X_val)[:, 1]
    
    metrics = evaluate_model(y_val, y_pred, y_pred_proba, "Random Forest")
    
    return model, metrics


def train_xgboost(X_train, y_train, X_val, y_val, random_state=42):
    """
    Train XGBoost model.
    
    Args:
        X_train: Training features
        y_train: Training labels
        X_val: Validation features
        y_val: Validation labels
        random_state: Random seed
        
    Returns:
        Trained model and evaluation metrics
    """
    print_separator("Training XGBoost")
    
    model = XGBClassifier(
        n_estimators=100,
        max_depth=6,
        learning_rate=0.1,
        subsample=0.8,
        colsample_bytree=0.8,
        scale_pos_weight=10,  # Handle class imbalance
        random_state=random_state,
        eval_metric='auc',
        use_label_encoder=False
    )
    
    model.fit(
        X_train, y_train,
        eval_set=[(X_val, y_val)],
        verbose=False
    )
    
    # Evaluate
    y_pred = model.predict(X_val)
    y_pred_proba = model.predict_proba(X_val)[:, 1]
    
    metrics = evaluate_model(y_val, y_pred, y_pred_proba, "XGBoost")
    
    return model, metrics


def compare_models(models_dict, X_test, y_test):
    """
    Compare all models on test set and generate ROC curves.
    
    Args:
        models_dict: Dictionary with model names as keys and models as values
        X_test: Test features
        y_test: Test labels
        
    Returns:
        Dictionary with test metrics for all models
    """
    print_separator("Model Comparison on Test Set")
    
    results = {}
    plt.figure(figsize=(10, 8))
    
    for name, model in models_dict.items():
        y_pred = model.predict(X_test)
        y_pred_proba = model.predict_proba(X_test)[:, 1]
        
        # Calculate metrics
        precision = precision_score(y_test, y_pred)
        recall = recall_score(y_test, y_pred)
        f1 = f1_score(y_test, y_pred)
        auc = roc_auc_score(y_test, y_pred_proba)
        
        results[name] = {
            'precision': precision,
            'recall': recall,
            'f1': f1,
            'auc': auc
        }
        
        # ROC curve
        fpr, tpr, _ = roc_curve(y_test, y_pred_proba)
        plt.plot(fpr, tpr, label=f'{name} (AUC = {auc:.4f})', linewidth=2)
        
        print(f"\n{name}:")
        print(f"  Precision: {precision:.4f}")
        print(f"  Recall: {recall:.4f}")
        print(f"  F1-Score: {f1:.4f}")
        print(f"  AUC-ROC: {auc:.4f}")
    
    plt.plot([0, 1], [0, 1], 'k--', label='Random Classifier')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate', fontsize=12)
    plt.ylabel('True Positive Rate', fontsize=12)
    plt.title('ROC Curves - Model Comparison', fontsize=14, fontweight='bold')
    plt.legend(loc='lower right', fontsize=10)
    plt.grid(alpha=0.3)
    
    os.makedirs('models', exist_ok=True)
    plt.savefig('models/roc_curves_comparison.png', dpi=300, bbox_inches='tight')
    print("\nROC curves saved to models/roc_curves_comparison.png")
    plt.close()
    
    return results


def train_all_models(X_train, y_train, X_val, y_val, X_test, y_test, 
                     save_best=True, model_dir='models'):
    """
    Train all models and select the best one.
    
    Args:
        X_train: Training features
        y_train: Training labels
        X_val: Validation features
        y_val: Validation labels
        X_test: Test features
        y_test: Test labels
        save_best: Whether to save the best model
        model_dir: Directory to save models
        
    Returns:
        Best model and all trained models
    """
    print_separator("TRAINING ALL MODELS")
    
    models = {}
    metrics_dict = {}
    
    # Train Logistic Regression
    lr_model, lr_metrics = train_logistic_regression(X_train, y_train, X_val, y_val)
    models['Logistic Regression'] = lr_model
    metrics_dict['Logistic Regression'] = lr_metrics
    
    # Train Random Forest
    rf_model, rf_metrics = train_random_forest(X_train, y_train, X_val, y_val)
    models['Random Forest'] = rf_model
    metrics_dict['Random Forest'] = rf_metrics
    
    # Train XGBoost
    xgb_model, xgb_metrics = train_xgboost(X_train, y_train, X_val, y_val)
    models['XGBoost'] = xgb_model
    metrics_dict['XGBoost'] = xgb_metrics
    
    # Compare on test set
    test_results = compare_models(models, X_test, y_test)
    
    # Select best model based on AUC
    best_model_name = max(test_results.keys(), key=lambda x: test_results[x]['auc'])
    best_model = models[best_model_name]
    
    print_separator("BEST MODEL SELECTION")
    print(f"Best Model: {best_model_name}")
    print(f"AUC-ROC: {test_results[best_model_name]['auc']:.4f}")
    print(f"Precision: {test_results[best_model_name]['precision']:.4f}")
    print(f"Recall: {test_results[best_model_name]['recall']:.4f}")
    print(f"F1-Score: {test_results[best_model_name]['f1']:.4f}")
    
    # Save best model
    if save_best:
        os.makedirs(model_dir, exist_ok=True)
        model_path = os.path.join(model_dir, 'final_model.pkl')
        save_model(best_model, model_path)
        print(f"\nBest model saved to {model_path}")
    
    return best_model, models, metrics_dict, test_results

