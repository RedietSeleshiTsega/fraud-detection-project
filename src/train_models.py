import pandas as pd
import joblib
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from xgboost import XGBClassifier
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score, f1_score, precision_recall_curve, average_precision_score

from imblearn.over_sampling import SMOTE
from sklearn.preprocessing import StandardScaler

def load_data(path):
    return pd.read_csv(path)

def prepare_data(df, target_col, scale=True):
    X = df.drop(columns=[target_col])
    y = df[target_col]

  
    if scale:
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)
        return X_scaled, y, scaler
    return X, y, None

def evaluate_model(model, X_test, y_test, model_name):
    y_pred = model.predict(X_test)
    y_proba = model.predict_proba(X_test)[:, 1]

    print(f"====== {model_name} ======")
    print("Confusion Matrix:\n", confusion_matrix(y_test, y_pred))
    print("Classification Report:\n", classification_report(y_test, y_pred))
    print("ROC-AUC Score:", roc_auc_score(y_test, y_proba))
    print("F1 Score:", f1_score(y_test, y_pred))
    print("Average Precision (PR AUC):", average_precision_score(y_test, y_proba))
    print()

def train_models(data_path, target_col, model_output_prefix):
    df = load_data(data_path)

    X, y, scaler = prepare_data(df, target_col)


    smote = SMOTE(random_state=42)
    X_resampled, y_resampled = smote.fit_resample(X, y)

    X_train, X_test, y_train, y_test = train_test_split(X_resampled, y_resampled, test_size=0.2, random_state=42)


    lr = LogisticRegression(max_iter=1000)
    lr.fit(X_train, y_train)
    evaluate_model(lr, X_test, y_test, "Logistic Regression")
    joblib.dump(lr, f"models/saved_models/{model_output_prefix}_logreg.pkl")

 
    xgb = XGBClassifier(use_label_encoder=False, eval_metric='logloss')
    xgb.fit(X_train, y_train)
    evaluate_model(xgb, X_test, y_test, "XGBoost Classifier")
    joblib.dump(xgb, f"models/saved_models/{model_output_prefix}_xgboost.pkl")


    if scaler:
        joblib.dump(scaler, f"models/saved_models/{model_output_prefix}_scaler.pkl")

if __name__ == "__main__":
    train_models("data/processed/cleaned_fraud_data.csv", "class", "fraud")
    train_models("data/processed/cleaned_creditcard.csv", "Class", "credit")
