# 📘 Credit Card Fraud Detection (Machine Learning + Streamlit App)

A complete end-to-end **Fraud Detection System** built using the Kaggle dataset (284,807 transactions).  
This project includes **EDA, preprocessing, model training (LR, RF, XGBoost), evaluation, model saving**, and a **Streamlit web app** for real-time predictions.

---

## 🚀 Project Features

### ✔ Exploratory Data Analysis (EDA)
- Class imbalance visualization (fraud = 0.17%)  
- Correlation heatmap  
- Boxplots: Amount vs Class  
- Time-based fraud behavior  
- Feature importance visualization  

### ✔ Preprocessing
- Outlier removal using IQR  
- StandardScaler for feature scaling  
- Train/Validation/Test split  
- **SMOTE oversampling** for imbalanced data  
- Clean handling of numeric PCA features  

### ✔ Machine Learning Models
Trained & evaluated:
- **Logistic Regression**
- **Random Forest**
- **XGBoost**

Evaluation includes:
- Confusion Matrix  
- Classification Report  
- ROC Curve  
- Precision, Recall, F1-Score  
- AUC Score  

### ✔ Performance Achieved
- **AUC ≈ 0.97**  
- **Precision (Fraud) ≈ 92%**  
- Best performing model → **XGBoost**

### ✔ Streamlit App
- Sidebar navigation  
- Input fields for all 30 transaction features  
- Real-time prediction + fraud probability  
- Clean UI with scaling applied internally  
- Displays scaled input 

---

## 📁 Project Structure
credit-card-fraud-detection/
├── data/
│ └── creditcard.csv (NOT included — download from Kaggle)
├── notebooks/
│ └── eda.ipynb
├── models/
│ ├── final_model.pkl
│ └── scaler.pkl
├── src/
│ ├── preprocessing.py
│ ├── train_models.py
│ ├── evaluate.py
│ └── utils.py
├── app/
│ └── streamlit_app.py
├── requirements.txt
└── README.md

---

## 📥 Dataset

The dataset is **NOT included** in this repository because it exceeds GitHub's upload limit.

Download it from Kaggle here:

🔗 https://www.kaggle.com/mlg-ulb/creditcardfraud

Place the file at:

data/creditcard.csv

---

## 🛠 Installation & Setup

### 1️⃣ Create a virtual environment
```bash
python -m venv venv
```

Activate:

Windows:
venv\Scripts\activate

Mac/Linux:
source venv/bin/activate

2️⃣ Install dependencies
pip install -r requirements.txt

3️⃣ Train Machine Learning Models
python -m src.train_models

This will generate:

models/final_model.pkl

models/scaler.pkl

🌐 Run Streamlit App
streamlit run app/streamlit_app.py

The app runs at:
http://localhost:8501

📊 Model Evaluation Outputs

Evaluation plots saved in the reports/ folder:

<model>_confusion.png

<model>_roc.png

Metrics:

ROC AUC

Precision, Recall

F1 Score

Classification Report

🎯 Technologies Used

Python

pandas, numpy

scikit-learn

imbalanced-learn (SMOTE)

XGBoost

matplotlib, seaborn

Streamlit

joblib

🧠 What I Learned

Handling highly imbalanced fraud datasets

Using SMOTE for minority oversampling

Model comparison & evaluation

ROC-AUC as a key metric in fraud detection

Deploying ML models with Streamlit

🚀 Future Improvements

Add SHAP interpretability

Add real API using FastAPI

Deploy ML inference pipeline on Streamlit Cloud

Add advanced anomaly detection techniques

👩‍💻 Author

Ishita Pathunda
Machine Learning & Full Stack Developer
