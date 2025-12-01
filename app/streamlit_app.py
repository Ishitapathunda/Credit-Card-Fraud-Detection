"""
Streamlit App for Credit Card Fraud Detection
"""

import streamlit as st
import pandas as pd
import numpy as np
import joblib
import os
import sys

# Add parent directory to path to import src modules
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.utils import load_model

# Page configuration
st.set_page_config(
    page_title="Credit Card Fraud Detection",
    page_icon="💳",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for better UI
st.markdown("""
    <style>
    .main-header {
        font-size: 2.5rem;
        font-weight: bold;
        color: #1f77b4;
        text-align: center;
        padding: 1rem 0;
    }
    .sub-header {
        font-size: 1.2rem;
        color: #666;
        text-align: center;
        padding-bottom: 2rem;
    }
    .prediction-box {
        padding: 1.5rem;
        border-radius: 10px;
        margin: 1rem 0;
    }
    .fraud-prediction {
        background-color: #fee;
        border: 2px solid #f44;
    }
    .non-fraud-prediction {
        background-color: #efe;
        border: 2px solid #4f4;
    }
    </style>
""", unsafe_allow_html=True)

@st.cache_resource
def load_trained_model():
    """Load the trained model."""
    model_path = 'models/final_model.pkl'
    if not os.path.exists(model_path):
        st.error(f"Model not found at {model_path}. Please train the model first.")
        return None
    return load_model(model_path)

@st.cache_resource
def load_scaler():
    """Load the scaler."""
    scaler_path = 'models/scaler.pkl'
    if not os.path.exists(scaler_path):
        st.warning(f"Scaler not found at {scaler_path}. Predictions may be inaccurate.")
        return None
    return load_model(scaler_path)

def main():
    # Header
    st.markdown('<p class="main-header">💳 Credit Card Fraud Detection System</p>', unsafe_allow_html=True)
    st.markdown('<p class="sub-header">Machine Learning Model for Real-time Fraud Detection</p>', unsafe_allow_html=True)
    
    # Sidebar
    st.sidebar.header("📊 Model Information")
    st.sidebar.markdown("""
    **Dataset**: Kaggle Credit Card Fraud Detection
    
    **Model**: XGBoost (Best Performing)
    
    **Performance**:
    - AUC-ROC: ~0.97
    - Precision: ~92%
    - Recall: High
    - F1-Score: Optimized
    
    **Features**: 30 features (V1-V28, Amount, Time)
    """)
    
    st.sidebar.markdown("---")
    st.sidebar.markdown("### 📝 Instructions")
    st.sidebar.markdown("""
    1. Enter transaction details
    2. Click 'Predict Fraud' button
    3. View prediction and probability
    """)
    
    # Load model and scaler
    model = load_trained_model()
    scaler = load_scaler()
    if model is None:
        st.stop()
    
    # Main content
    st.header("🔍 Transaction Details")
    
    # Create input form
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("Transaction Amount & Time")
        amount = st.number_input("Amount", min_value=0.0, value=100.0, step=0.01, format="%.2f")
        time = st.number_input("Time (seconds elapsed)", min_value=0.0, value=0.0, step=1.0)
        
        st.subheader("PCA Features V1-V14")
        v1 = st.number_input("V1", value=0.0, step=0.01, format="%.4f")
        v2 = st.number_input("V2", value=0.0, step=0.01, format="%.4f")
        v3 = st.number_input("V3", value=0.0, step=0.01, format="%.4f")
        v4 = st.number_input("V4", value=0.0, step=0.01, format="%.4f")
        v5 = st.number_input("V5", value=0.0, step=0.01, format="%.4f")
        v6 = st.number_input("V6", value=0.0, step=0.01, format="%.4f")
        v7 = st.number_input("V7", value=0.0, step=0.01, format="%.4f")
        v8 = st.number_input("V8", value=0.0, step=0.01, format="%.4f")
        v9 = st.number_input("V9", value=0.0, step=0.01, format="%.4f")
        v10 = st.number_input("V10", value=0.0, step=0.01, format="%.4f")
        v11 = st.number_input("V11", value=0.0, step=0.01, format="%.4f")
        v12 = st.number_input("V12", value=0.0, step=0.01, format="%.4f")
        v13 = st.number_input("V13", value=0.0, step=0.01, format="%.4f")
        v14 = st.number_input("V14", value=0.0, step=0.01, format="%.4f")
    
    with col2:
        st.subheader("PCA Features V15-V28")
        v15 = st.number_input("V15", value=0.0, step=0.01, format="%.4f")
        v16 = st.number_input("V16", value=0.0, step=0.01, format="%.4f")
        v17 = st.number_input("V17", value=0.0, step=0.01, format="%.4f")
        v18 = st.number_input("V18", value=0.0, step=0.01, format="%.4f")
        v19 = st.number_input("V19", value=0.0, step=0.01, format="%.4f")
        v20 = st.number_input("V20", value=0.0, step=0.01, format="%.4f")
        v21 = st.number_input("V21", value=0.0, step=0.01, format="%.4f")
        v22 = st.number_input("V22", value=0.0, step=0.01, format="%.4f")
        v23 = st.number_input("V23", value=0.0, step=0.01, format="%.4f")
        v24 = st.number_input("V24", value=0.0, step=0.01, format="%.4f")
        v25 = st.number_input("V25", value=0.0, step=0.01, format="%.4f")
        v26 = st.number_input("V26", value=0.0, step=0.01, format="%.4f")
        v27 = st.number_input("V27", value=0.0, step=0.01, format="%.4f")
        v28 = st.number_input("V28", value=0.0, step=0.01, format="%.4f")
    
    # Quick fill buttons
    st.markdown("---")
    col_btn1, col_btn2, col_btn3 = st.columns(3)
    
    with col_btn1:
        if st.button("📋 Load Sample Non-Fraud Transaction"):
            st.session_state.sample_type = "non_fraud"
            st.rerun()
    
    with col_btn2:
        if st.button("⚠️ Load Sample Fraud Transaction"):
            st.session_state.sample_type = "fraud"
            st.rerun()
    
    with col_btn3:
        if st.button("🔄 Reset All Fields"):
            st.session_state.sample_type = None
            st.rerun()
    
    # Prediction button
    st.markdown("---")
    predict_button = st.button("🔮 Predict Fraud", type="primary", use_container_width=True)
    
    if predict_button:
        # Prepare input data
        input_data = pd.DataFrame({
            'V1': [v1], 'V2': [v2], 'V3': [v3], 'V4': [v4], 'V5': [v5],
            'V6': [v6], 'V7': [v7], 'V8': [v8], 'V9': [v9], 'V10': [v10],
            'V11': [v11], 'V12': [v12], 'V13': [v13], 'V14': [v14], 'V15': [v15],
            'V16': [v16], 'V17': [v17], 'V18': [v18], 'V19': [v19], 'V20': [v20],
            'V21': [v21], 'V22': [v22], 'V23': [v23], 'V24': [v24], 'V25': [v25],
            'V26': [v26], 'V27': [v27], 'V28': [v28], 'Amount': [amount]
        })
        
        # Scale the input data if scaler is available
        if scaler is not None:
            input_data_scaled = scaler.transform(input_data)
            input_data = pd.DataFrame(input_data_scaled, columns=input_data.columns)
        
        # Make prediction
        try:
            prediction = model.predict(input_data)[0]
            prediction_proba = model.predict_proba(input_data)[0]
            
            fraud_probability = prediction_proba[1] * 100
            non_fraud_probability = prediction_proba[0] * 100
            
            # Display results
            st.markdown("---")
            st.header("📊 Prediction Results")
            
            result_col1, result_col2 = st.columns(2)
            
            with result_col1:
                if prediction == 1:
                    st.markdown(
                        f'<div class="prediction-box fraud-prediction">'
                        f'<h2>⚠️ FRAUD DETECTED</h2>'
                        f'<p style="font-size: 1.2rem;">This transaction is flagged as <strong>FRAUDULENT</strong></p>'
                        f'</div>',
                        unsafe_allow_html=True
                    )
                else:
                    st.markdown(
                        f'<div class="prediction-box non-fraud-prediction">'
                        f'<h2>✅ LEGITIMATE TRANSACTION</h2>'
                        f'<p style="font-size: 1.2rem;">This transaction appears to be <strong>LEGITIMATE</strong></p>'
                        f'</div>',
                        unsafe_allow_html=True
                    )
            
            with result_col2:
                st.subheader("Prediction Probabilities")
                st.metric("Fraud Probability", f"{fraud_probability:.2f}%")
                st.metric("Non-Fraud Probability", f"{non_fraud_probability:.2f}%")
                
                # Progress bars
                st.progress(fraud_probability / 100, text=f"Fraud: {fraud_probability:.2f}%")
                st.progress(non_fraud_probability / 100, text=f"Non-Fraud: {non_fraud_probability:.2f}%")
            
            # Additional information
            st.markdown("---")
            st.info(f"**Model Confidence**: {max(fraud_probability, non_fraud_probability):.2f}%")
            
            if prediction == 1:
                st.warning("🚨 **Action Required**: This transaction has been flagged. Please review manually.")
            else:
                st.success("✅ **Status**: Transaction appears safe. No action required.")
                
        except Exception as e:
            st.error(f"Error making prediction: {str(e)}")
            st.info("Please ensure all fields are filled correctly.")

if __name__ == "__main__":
    main()

