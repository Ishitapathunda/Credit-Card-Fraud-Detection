"""
Data preprocessing pipeline for credit card fraud detection.
"""

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from imblearn.over_sampling import SMOTE
import warnings
warnings.filterwarnings('ignore')


def remove_outliers_iqr(df, column='Amount', factor=1.5):
    """
    Remove outliers using IQR method.
    
    Args:
        df: DataFrame
        column: Column name to check for outliers
        factor: IQR factor (default 1.5)
        
    Returns:
        DataFrame with outliers removed
    """
    Q1 = df[column].quantile(0.25)
    Q3 = df[column].quantile(0.75)
    IQR = Q3 - Q1
    lower_bound = Q1 - factor * IQR
    upper_bound = Q3 + factor * IQR
    
    # Remove outliers
    df_clean = df[(df[column] >= lower_bound) & (df[column] <= upper_bound)].copy()
    
    print(f"Removed {len(df) - len(df_clean)} outliers from '{column}' column")
    print(f"Original size: {len(df)}, Cleaned size: {len(df_clean)}")
    
    return df_clean


def remove_outliers_zscore(df, column='Amount', threshold=3):
    """
    Remove outliers using Z-score method.
    
    Args:
        df: DataFrame
        column: Column name to check for outliers
        threshold: Z-score threshold (default 3)
        
    Returns:
        DataFrame with outliers removed
    """
    z_scores = np.abs((df[column] - df[column].mean()) / df[column].std())
    df_clean = df[z_scores < threshold].copy()
    
    print(f"Removed {len(df) - len(df_clean)} outliers from '{column}' column (Z-score > {threshold})")
    print(f"Original size: {len(df)}, Cleaned size: {len(df_clean)}")
    
    return df_clean


def preprocess_data(filepath, test_size=0.2, val_size=0.1, random_state=42, 
                    remove_outliers=True, outlier_method='zscore', use_smote=True):
    """
    Complete preprocessing pipeline.
    
    Args:
        filepath: Path to the CSV file
        test_size: Proportion of data for test set
        val_size: Proportion of training data for validation set
        random_state: Random seed
        remove_outliers: Whether to remove outliers
        outlier_method: Method to use ('zscore' or 'iqr')
        use_smote: Whether to apply SMOTE oversampling
        
    Returns:
        Tuple of (X_train, X_val, X_test, y_train, y_val, y_test, scaler)
    """
    print("="*60)
    print("DATA PREPROCESSING PIPELINE")
    print("="*60)
    
    # Load data
    print("\n1. Loading data...")
    df = pd.read_csv(filepath)
    print(f"   Dataset shape: {df.shape}")
    print(f"   Fraud cases: {df['Class'].sum()} ({df['Class'].mean()*100:.2f}%)")
    
    # Separate features and target
    X = df.drop(['Class', 'Time'], axis=1)
    y = df['Class']
    
    # Remove outliers from Amount column
    if remove_outliers:
        print("\n2. Removing outliers...")
        if outlier_method == 'zscore':
            df_clean = remove_outliers_zscore(df, column='Amount', threshold=3)
        else:
            df_clean = remove_outliers_iqr(df, column='Amount', factor=1.5)
        
        X = df_clean.drop(['Class', 'Time'], axis=1)
        y = df_clean['Class']
        print(f"   After outlier removal: {X.shape[0]} samples")
    
    # Train/Validation/Test split
    print("\n3. Splitting data...")
    X_temp, X_test, y_temp, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=y
    )
    
    val_size_adjusted = val_size / (1 - test_size)
    X_train, X_val, y_train, y_val = train_test_split(
        X_temp, y_temp, test_size=val_size_adjusted, random_state=random_state, stratify=y_temp
    )
    
    print(f"   Train set: {X_train.shape[0]} samples")
    print(f"   Validation set: {X_val.shape[0]} samples")
    print(f"   Test set: {X_test.shape[0]} samples")
    print(f"   Train fraud rate: {y_train.mean()*100:.2f}%")
    print(f"   Val fraud rate: {y_val.mean()*100:.2f}%")
    print(f"   Test fraud rate: {y_test.mean()*100:.2f}%")
    
    # Feature scaling
    print("\n4. Scaling features...")
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_val_scaled = scaler.transform(X_val)
    X_test_scaled = scaler.transform(X_test)
    
    # Convert back to DataFrame for better handling
    X_train = pd.DataFrame(X_train_scaled, columns=X_train.columns, index=X_train.index)
    X_val = pd.DataFrame(X_val_scaled, columns=X_val.columns, index=X_val.index)
    X_test = pd.DataFrame(X_test_scaled, columns=X_test.columns, index=X_test.index)
    
    print("   Features scaled using StandardScaler")
    
    # Apply SMOTE for oversampling
    if use_smote:
        print("\n5. Applying SMOTE oversampling...")
        print(f"   Before SMOTE - Train set: {X_train.shape[0]} samples, Fraud: {y_train.sum()}")
        
        smote = SMOTE(random_state=random_state, sampling_strategy=0.1)  # 10% fraud rate
        X_train, y_train = smote.fit_resample(X_train, y_train)
        
        print(f"   After SMOTE - Train set: {X_train.shape[0]} samples, Fraud: {y_train.sum()}")
        print(f"   New fraud rate: {y_train.mean()*100:.2f}%")
    
    print("\n" + "="*60)
    print("PREPROCESSING COMPLETE")
    print("="*60 + "\n")
    
    return X_train, X_val, X_test, y_train, y_val, y_test, scaler

