# 📊 Customer Churn Prediction Using Machine Learning

[![Python](https://img.shields.io/badge/Python-3.8%2B-3776AB?style=for-the-badge&logo=python&logoColor=white)](https://www.python.org/)
[![Scikit-Learn](https://img.shields.io/badge/Scikit--Learn-F7931E?style=for-the-badge&logo=scikit-learn&logoColor=white)](https://scikit-learn.org/)
[![XGBoost](https://img.shields.io/badge/XGBoost-EB1923?style=for-the-badge&logo=xgboost&logoColor=white)](https://xgboost.ai/)
[![Pandas](https://img.shields.io/badge/Pandas-150458?style=for-the-badge&logo=pandas&logoColor=white)](https://pandas.pydata.org/)
[![Jupyter](https://img.shields.io/badge/Jupyter-Notebook-F37626?style=for-the-badge&logo=jupyter&logoColor=white)](https://jupyter.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg?style=for-the-badge)](LICENSE)

An end-to-end Machine Learning pipeline designed to predict customer churn in the telecommunications industry using the **Telco Customer Churn** dataset. By identifying at-risk customers prior to cancellation, businesses can implement proactive retention strategies, optimize customer lifetime value (CLV), and reduce revenue loss.

---

## 📌 Table of Contents
- [Project Overview](#-project-overview)
- [Key Features & Highlights](#-key-features--highlights)
- [Dataset Summary](#-dataset-summary)
- [Architecture & Workflow](#-architecture--workflow)
- [Model Evaluation & Benchmarking](#-model-evaluation--benchmarking)
- [Key Business Insights](#-key-business-insights)
- [Repository Structure](#-repository-structure)
- [Getting Started](#-getting-started)
- [Sample Inference](#-sample-inference)
- [Future Enhancements](#-future-enhancements)
- [Author](#-author)

---

## 🔍 Project Overview

Customer churn occurs when customers terminate their contracts or discontinue services. In subscription and telecom industries, customer acquisition is substantially more expensive than customer retention.

This project delivers:
1. **Thorough Data Cleansing & Exploration:** Handling missing/blank strings, data coercion, distribution analysis, and correlation mapping.
2. **Class Imbalance Resolution with SMOTE:** Balancing the churn ratio strictly on training data to prevent synthetic data leakage into test sets.
3. **Multi-Model Cross-Validation:** Systematic benchmarking across **Decision Tree**, **Random Forest**, and **XGBoost**.
4. **Interpretable Retention Insights:** Clear identification of churn drivers such as contract tenure, monthly billing, and service bundling.

---

## ✨ Key Features & Highlights

- **Dataset Size:** 7,043 customers and 21 demographic & account attributes.
- **Leakage-Free SMOTE Pipeline:** Minority class oversampled exclusively within the training partition.
- **Stratified 5-Fold Cross-Validation:** Robust cross-validation showing **Random Forest** achieving **~84%** accuracy.
- **Pickle Model Serialization:** Saving trained weights and LabelEncoder mappings for production deployment.

---

## 📂 Dataset Summary

The dataset utilized is the popular **Telco Customer Churn Dataset** (`WA_Fn-UseC_-Telco-Customer-Churn.csv`):

| Category | Feature Names | Description |
| :--- | :--- | :--- |
| **Target Variable** | `Churn` | Whether the customer churned (`Yes` / `No`) |
| **Demographics** | `gender`, `SeniorCitizen`, `Partner`, `Dependents` | Customer demographic background |
| **Account Details** | `tenure`, `Contract`, `PaperlessBilling`, `PaymentMethod` | Account tenure (months), contract type, billing format |
| **Financials** | `MonthlyCharges`, `TotalCharges` | Current monthly payment & accumulated total payments |
| **Subscribed Services**| `PhoneService`, `MultipleLines`, `InternetService`, `OnlineSecurity`, `OnlineBackup`, `DeviceProtection`, `TechSupport`, `StreamingTV`, `StreamingMovies` | Network and add-on subscriptions |

### 🛠️ Data Preprocessing Steps:
- **Identifier Removal:** `customerID` dropped as it carries no predictive signal.
- **Handling Empty Total Charges:** Coerced 11 whitespace entries in `TotalCharges` into `0.0` and converted from string to numeric float.
- **Categorical Encoding:** Label encoding applied across all categorical features, with individual encoders preserved for inference pipelines.
- **Train / Test Split:** Stratified 80% training (5,634 rows) and 20% testing (1,409 rows).

---

## 🏗️ Architecture & Workflow

```text
       Raw Telco Dataset (7,043 rows)
                     │
                     ▼
       Data Cleaning & Transformation
       (Drop IDs, Handle Blanks, Cast Types)
                     │
                     ▼
          Categorical Encoding
       (Preserve Encoders in encoders.pkl)
                     │
                     ▼
        80/20 Train / Test Split
        ├── Train: 5,634 samples
        └── Test:  1,409 samples
                     │
                     ▼
             SMOTE Resampling
      (Balanced to 4,138 per class: 8,276 total)
                     │
                     ▼
        5-Fold Cross-Validation & Model Training
        ├── Decision Tree Classifier
        ├── Random Forest Classifier (Top Performer)
        └── XGBoost Classifier
                     │
                     ▼
             Test Evaluation & Metrics
         (Accuracy, Confusion Matrix, Classification Report)
                     │
                     ▼
        Export Model (`customer_churn_model.pkl`) & Run Sample Prediction
```

---

## 📈 Model Evaluation & Benchmarking

### 1. 5-Fold Cross-Validation Results

| Model | 5-Fold CV Accuracy | Performance Rank |
| :--- | :---: | :---: |
| 🌲 **Random Forest Classifier** | **84% (0.84)** | 🥇 Best Model |
| ⚡ **XGBoost Classifier** | **83% (0.83)** | 🥈 Runner-Up |
| 🌿 **Decision Tree Classifier** | **78% (0.78)** | 🥉 Baseline |

### 2. Test Set Evaluation Metrics (Random Forest)

- **Test Accuracy:** `77.7% (~78%)`
- **Confusion Matrix:**
  ```text
  [[877  159]    -->  877 True Negatives, 159 False Positives
   [155  218]]   -->  155 False Negatives, 218 True Positives
  ```

- **Detailed Classification Report:**

| Class | Precision | Recall | F1-Score | Support |
| :--- | :---: | :---: | :---: | :---: |
| **0 (Retained / No)** | `0.85` | `0.85` | `0.85` | 1,036 |
| **1 (Churned / Yes)** | `0.58` | `0.58` | `0.58` | 373 |
| **Macro Average** | `0.71` | `0.72` | `0.71` | 1,409 |
| **Weighted Average** | `0.78` | `0.78` | `0.78` | 1,409 |

---

## 💡 Key Business Insights

1. **Contract Structure:** Customers on **Month-to-Month** contracts exhibit the highest propensity to churn compared to long-term 1-year and 2-year commitments.
2. **Tenure Impact:** Customers within their initial 1–12 months of service demonstrate the highest churn vulnerability; retention incentives during the onboarding phase yield the highest ROI.
3. **Payment Methods:** Customers utilizing **Electronic Check** payment show higher churn rates compared to automated bank transfers or credit cards.
4. **Tech Support & Security:** Subscribing to add-on services like `OnlineSecurity` and `TechSupport` strongly correlates with lower churn risk.

---

## 📁 Repository Structure

```text
Customer-Churn-Prediction/
├── Cust_Chrn_Pred.ipynb                 # Interactive Jupyter notebook with complete code and EDA
├── WA_Fn-UseC_-Telco-Customer-Churn.csv # Telco customer dataset (7,043 rows)
├── requirements.txt                     # Dependencies & environment specifications
├── .gitignore                           # Git ignore rules for virtual environments & checkpoints
└── README.md                            # Comprehensive project documentation
```

---

## 🚀 Getting Started

### Option 1: Google Colab (Zero-Setup)
1. Open [Google Colab](https://colab.research.google.com/).
2. Upload `Cust_Chrn_Pred.ipynb` and `WA_Fn-UseC_-Telco-Customer-Churn.csv`.
3. Execute the notebook cells sequentially.

### Option 2: Local Setup

1. **Clone the repository:**
   ```bash
   git clone https://github.com/Charu-2207/Customer-Churn-Prediction.git
   cd Customer-Churn-Prediction
   ```

2. **Create and activate a virtual environment (optional but recommended):**
   ```bash
   python -m venv venv
   # On Windows:
   venv\Scripts\activate
   # On macOS/Linux:
   source venv/bin/activate
   ```

3. **Install the dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

4. **Launch Jupyter Notebook:**
   ```bash
   jupyter notebook Cust_Chrn_Pred.ipynb
   ```

---

## 🧪 Sample Inference

To test predictions on a sample customer record:

```python
import pickle
import pandas as pd

# Load saved model bundle and encoders
with open("customer_churn_model.pkl", "rb") as f:
    model_bundle = pickle.load(f)
    model = model_bundle["model"]

with open("encoders.pkl", "rb") as f:
    encoders = pickle.load(f)

# Customer profile input
customer_sample = {
    "gender": "Female",
    "SeniorCitizen": 0,
    "Partner": "Yes",
    "Dependents": "No",
    "tenure": 1,
    "PhoneService": "No",
    "MultipleLines": "No phone service",
    "InternetService": "DSL",
    "OnlineSecurity": "No",
    "OnlineBackup": "Yes",
    "DeviceProtection": "No",
    "TechSupport": "No",
    "StreamingTV": "No",
    "StreamingMovies": "No",
    "Contract": "Month-to-month",
    "PaperlessBilling": "Yes",
    "PaymentMethod": "Electronic check",
    "MonthlyCharges": 29.85,
    "TotalCharges": 29.85
}

df_input = pd.DataFrame([customer_sample])

# Apply encoders
for col, encoder in encoders.items():
    if col in df_input.columns:
        df_input[col] = encoder.transform(df_input[col])

# Predict
prediction = model.predict(df_input)
probability = model.predict_proba(df_input)

print("Result:", "Churn" if prediction[0] == 1 else "No Churn")
print(f"Probabilities - Retention: {probability[0][0]:.2f}, Churn: {probability[0][1]:.2f}")
```

---

## 🔮 Future Enhancements

- [ ] Deploy an interactive **Streamlit** or **FastAPI** web application for real-time customer churn scoring.
- [ ] Implement advanced hyperparameter tuning using **Optuna** / **GridSearchCV**.
- [ ] Incorporate **SHAP (SHapley Additive exPlanations)** values for localized feature explainability.
- [ ] Evaluate threshold tuning to maximize recall for high-value customer retention.

---

## 👩‍💻 Author

**Charu Nethira S**  
*Aspiring Data Analyst & Computer Science Engineer*  
- 🌐 [Portfolio](https://github.com/Charu-2207)  
- 💼 [GitHub Profile](https://github.com/Charu-2207)  

---

## 📜 License

This project is licensed under the MIT License - feel free to use and adapt it for learning and personal projects.
