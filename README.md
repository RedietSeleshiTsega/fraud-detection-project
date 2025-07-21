# 🛡️ Fraud Detection System – Adey Innovations Inc.

This project is focused on developing robust fraud detection models for e-commerce and banking transactions. It is part of a two-week challenge under Adey Innovations Inc., aimed at improving financial transaction security and customer trust.

## 🔍 Objective

Detect fraudulent transactions from two datasets:
- `Fraud_Data.csv` – e-commerce transaction data
- `creditcard.csv` – bank credit transaction data

### Key Challenges:
- Handling imbalanced datasets
- Mapping IP addresses to geolocation (country)
- Creating time-based and frequency-based features

---

## 🧪 Task 1 – Data Analysis & Preprocessing

✔️ Cleaned and explored both datasets  
✔️ Handled missing values and duplicates  
✔️ Converted timestamps and created new features (e.g. `time_since_signup`)  
✔️ Mapped IP addresses to country using `IpAddress_to_Country.csv`  
✔️ Handled class imbalance with SMOTE  
✔️ Scaled numeric features and encoded categoricals  

---

## 📊 Tech Stack

- Python (Pandas, NumPy, Matplotlib, Seaborn)
- Scikit-learn
- imbalanced-learn (SMOTE)
- Jupyter Notebook

---

## 🚀 Upcoming Tasks

Next steps include building ML models (Logistic Regression and Gradient Boosting), model evaluation using imbalanced metrics, and explainability analysis.

---

## 📁 Directory Structure

fraud-detection-project/
├── data/
├── notebooks/
├── outputs/
├── README.md
├── requirements.txt
└── .gitignore


---

## 👩‍💻 Author

Rediet Seleshi, Data Scientist @ Adey Innovations Inc.
