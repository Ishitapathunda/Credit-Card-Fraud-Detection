<<<<<<< HEAD
# Credit Card Fraud Detection

A complete end-to-end machine learning project for detecting fraudulent credit card transactions using the Kaggle Credit Card Fraud Detection dataset.

## 📊 Dataset

- **Source**: [Kaggle - Credit Card Fraud Detection](https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud)
- **Size**: 284,807 transactions
- **Fraud Rate**: 0.17% (highly imbalanced)
- **Features**: 30 features (Time, Amount, V1-V28 PCA-transformed features)
- **Target**: Binary classification (0 = Non-Fraud, 1 = Fraud)

## 🎯 Project Goals

- Achieve **AUC-ROC ≈ 0.97**
- Achieve **Precision ≈ 92%** on fraud class
- Build a production-ready Streamlit application
- Comprehensive EDA and model comparison

## 🏗️ Project Structure

```
credit-card-fraud/
├── data/
│   └── creditcard.csv          # Dataset (download from Kaggle)
├── notebooks/
│   └── eda.ipynb               # Exploratory Data Analysis
├── models/
│   ├── final_model.pkl         # Trained best model
│   └── *.png                   # Evaluation plots
├── src/
│   ├── preprocessing.py        # Data preprocessing pipeline
│   ├── train_models.py         # Model training and comparison
│   ├── evaluate.py             # Model evaluation utilities
│   └── utils.py               # Helper functions
├── app/
│   └── streamlit_app.py        # Streamlit web application
├── requirements.txt            # Python dependencies
└── README.md                   # This file
```

## 🚀 Setup Instructions

### 1. Clone/Download the Project

```bash
cd "credit-card-fraud"
```

### 2. Create Virtual Environment (Recommended)

```bash
python -m venv venv

# Windows
venv\Scripts\activate

# Linux/Mac
source venv/bin/activate
```

### 3. Install Dependencies

```bash
pip install -r requirements.txt
```

### 4. Download Dataset

1. Go to [Kaggle Credit Card Fraud Detection](https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud)
2. Download `creditcard.csv`
3. Place it in the `data/` directory

## 📈 Usage

### Step 1: Exploratory Data Analysis

Open and run the Jupyter notebook:

```bash
jupyter notebook notebooks/eda.ipynb
```

This will generate:
- Class distribution visualizations
- Correlation matrices
- Feature distribution plots
- Time-based analysis
- Amount analysis

### Step 2: Train Models

Create a training script or run in Python:

```python
from src.preprocessing import preprocess_data
from src.train_models import train_all_models

# Preprocess data
X_train, X_val, X_test, y_train, y_val, y_test, scaler = preprocess_data(
    'data/creditcard.csv',
    test_size=0.2,
    val_size=0.1,
    remove_outliers=True,
    use_smote=True
)

# Train all models
best_model, all_models, metrics, test_results = train_all_models(
    X_train, y_train, X_val, y_val, X_test, y_test,
    save_best=True
)
```

Or create a `train.py` file in the root directory:

```python
# train.py
from src.preprocessing import preprocess_data
from src.train_models import train_all_models

if __name__ == "__main__":
    print("Starting model training...")
    
    # Preprocess
    X_train, X_val, X_test, y_train, y_val, y_test, scaler = preprocess_data(
        'data/creditcard.csv'
    )
    
    # Train
    best_model, all_models, metrics, test_results = train_all_models(
        X_train, y_train, X_val, y_val, X_test, y_test
    )
    
    print("\nTraining complete!")
```

Run it:

```bash
python train.py
```

### Step 3: Run Streamlit App

```bash
streamlit run app/streamlit_app.py
```

The app will open in your browser at `http://localhost:8501`

## 🔧 Model Pipeline

### Preprocessing Steps

1. **Load Data**: Read CSV file
2. **Outlier Removal**: Remove outliers from `Amount` column using Z-score (threshold=3) or IQR
3. **Train/Val/Test Split**: 70% train, 10% validation, 20% test (stratified)
4. **Feature Scaling**: StandardScaler for all features
5. **SMOTE Oversampling**: Balance the training set (10% fraud rate)

### Models Trained

1. **Logistic Regression**
   - Class weights: balanced
   - Solver: lbfgs
   - Max iterations: 1000

2. **Random Forest**
   - N estimators: 100
   - Max depth: 10
   - Class weights: balanced

3. **XGBoost** (Best Performing)
   - N estimators: 100
   - Max depth: 6
   - Learning rate: 0.1
   - Scale pos weight: 10

### Evaluation Metrics

- Confusion Matrix
- Classification Report
- Precision, Recall, F1-Score
- ROC Curve
- AUC-ROC Score

## 📊 Expected Results

After training, you should see:

- **Best Model**: XGBoost (typically)
- **AUC-ROC**: ~0.97
- **Precision (Fraud)**: ~92%
- **Recall (Fraud)**: High
- **F1-Score**: Optimized

All evaluation plots are saved in the `models/` directory:
- `confusion_matrix_*.png`
- `roc_curve_*.png`
- `roc_curves_comparison.png`

## 🎨 Streamlit App Features

- **Interactive Input**: All 30 features can be entered manually
- **Real-time Prediction**: Instant fraud detection
- **Probability Display**: Shows confidence scores
- **Sample Data**: Quick load buttons for testing
- **Modern UI**: Clean, professional interface

## 📝 Key Features

### EDA Notebook Includes:
- ✅ Class distribution analysis
- ✅ Correlation matrix
- ✅ Amount vs Class boxplots
- ✅ Time-based behavior patterns
- ✅ Feature importance graphs
- ✅ Distribution visualizations

### Preprocessing:
- ✅ StandardScaler for feature scaling
- ✅ Train/validation/test split
- ✅ SMOTE oversampling
- ✅ Outlier removal (IQR/Z-score)

### Models:
- ✅ Logistic Regression
- ✅ Random Forest
- ✅ XGBoost
- ✅ Model comparison
- ✅ Best model selection

### Evaluation:
- ✅ Confusion Matrix
- ✅ Classification Report
- ✅ ROC Curves
- ✅ AUC-ROC scores
- ✅ Precision, Recall, F1

## 🛠️ Troubleshooting

### Issue: Model file not found
**Solution**: Run the training script first to generate `models/final_model.pkl`

### Issue: Dataset not found
**Solution**: Download `creditcard.csv` from Kaggle and place in `data/` directory

### Issue: Import errors
**Solution**: Ensure you're in the project root directory and all dependencies are installed

### Issue: SMOTE errors
**Solution**: Make sure `imbalanced-learn` is installed: `pip install imbalanced-learn`

## 📚 Dependencies

- pandas: Data manipulation
- numpy: Numerical operations
- scikit-learn: Machine learning algorithms
- imbalanced-learn: SMOTE oversampling
- xgboost: Gradient boosting
- matplotlib: Plotting
- seaborn: Statistical visualizations
- streamlit: Web application
- joblib: Model serialization

## 📄 License

This project is for educational purposes. The dataset is from Kaggle.

## 👤 Author

Machine Learning Engineer - Credit Card Fraud Detection Project

## 🙏 Acknowledgments

- Kaggle for providing the dataset
- Open source ML community for tools and libraries

---

**Note**: This is a complete end-to-end project. Make sure to download the dataset from Kaggle before running the training script.

=======
#  Credit Card Fraud Detection

A comprehensive **binary classification project** using the Kaggle Credit Card Fraud Dataset (280,000+ transactions) to detect fraudulent activity. Features an end-to-end pipeline, handling data imbalance with SMOTE, model comparison, evaluation, and a Streamlit deployment for real-time predictions.

---

##  Features
-  Handles highly imbalanced data via SMOTE for better minority detection  
-  Compares models like Logistic Regression, Random Forest, and XGBoost  
-  Achieves high performance — AUC-ROC 0.97 and 92% precision  
-  Conducts thorough EDA and feature engineering to uncover insightful patterns  
-  Deploys as an interactive app using Streamlit for easy demonstrations  

---

##  Tech Stack
- **Data & Analysis:** Python, Pandas, NumPy  
- **Modeling:** Scikit-learn, XGBoost  
- **Visualization:** Matplotlib, Seaborn  
- **UI/Deployment:** Streamlit, Jupyter Notebook  
- **Tools:** GitHub, VS Code, pip / virtualenv  

---

##  Project Structure
/data → Dataset files and preprocessing scripts
/notebooks → EDA and modeling step-by-step analysis
/models → Trained model artifacts (optional)
/app → Streamlit app for real-time predictions
README.md → Project overview and instructions


---

##  How to Run Locally


git clone https://github.com/Ishitapathunda/Credit-Card-Fraud-Detection.git
cd Credit-Card-Fraud-Detection 


# Install dependencies
pip install -r requirements.txt

# Run the Jupyter notebook for EDA and model training
jupyter notebook notebooks/Fraud_Detection.ipynb

# Launch the interactive Streamlit app
streamlit run app/app.py


Then open your browser at:
http://localhost:8501

Future Enhancements

Integrate real-time data streams via REST APIs or Kafka

Add ensemble methods or deep learning models for enhanced accuracy

Packaged deployment using Docker and cloud hosting for scalability

Conclusion

This project demonstrates strong competencies in handling imbalanced datasets, ML model evaluation, and deployment of data-driven web applications. With solid performance metrics and an interactive interface, it’s a great showcase of applied Data Science for practical use cases.

Author

Ishita Pathunda
>>>>>>> 2b193f6de201eab92b861efa7c6f1341a123c4b3
#   C r e d i t - C a r d - F r a u d - D e t e c t i o n  
 #   C r e d i t - C a r d - F r a u d - D e t e c t i o n  
 