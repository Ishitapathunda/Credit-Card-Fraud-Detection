# Quick Start Guide

## 🚀 Get Started in 3 Steps

### Step 1: Install Dependencies
```bash
pip install -r requirements.txt
```

### Step 2: Download Dataset
1. Visit: https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud
2. Download `creditcard.csv`
3. Place it in: `data/creditcard.csv`

### Step 3: Train Models
```bash
python train.py
```

This will:
- Preprocess the data
- Train Logistic Regression, Random Forest, and XGBoost
- Save the best model to `models/final_model.pkl`
- Generate evaluation plots

### Step 4: Run Streamlit App
```bash
streamlit run app/streamlit_app.py
```

The app will open in your browser!

## 📊 Explore the Data

Open the EDA notebook:
```bash
jupyter notebook notebooks/eda.ipynb
```

## 📁 Project Structure

```
credit-card-fraud/
├── data/              # Place creditcard.csv here
├── notebooks/         # EDA notebook
├── models/            # Trained models and plots
├── src/               # Source code
│   ├── preprocessing.py
│   ├── train_models.py
│   ├── evaluate.py
│   └── utils.py
├── app/               # Streamlit app
│   └── streamlit_app.py
├── train.py           # Main training script
├── requirements.txt   # Dependencies
└── README.md          # Full documentation
```

## ⚡ Expected Results

After training, you should achieve:
- **AUC-ROC**: ~0.97
- **Precision (Fraud)**: ~92%
- **Best Model**: XGBoost (typically)

## 🎯 Next Steps

1. Run EDA notebook to understand the data
2. Train models using `train.py`
3. Deploy using Streamlit app
4. Experiment with hyperparameters for better results

---

For detailed documentation, see [README.md](README.md)

