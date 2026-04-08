import numpy as np
import joblib
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from sklearn.neural_network import MLPClassifier
import os

def train_models():
    print("Loading preprocessed data...")
    X_train = np.load('data/X_train.npy')
    X_test = np.load('data/X_test.npy')
    y_train = np.load('data/y_train.npy')
    y_test = np.load('data/y_test.npy')
    
    os.makedirs('models/trained', exist_ok=True)
    
    # 1. Logistic Regression (Baseline)
    print("Training Logistic Regression...")
    lr_model = LogisticRegression(max_iter=1000)
    lr_model.fit(X_train, y_train)
    joblib.dump(lr_model, 'models/trained/logistic_regression.pkl')
    
    # 2. Random Forest
    print("Training Random Forest...")
    rf_model = RandomForestClassifier(n_estimators=100, random_state=42)
    rf_model.fit(X_train, y_train)
    joblib.dump(rf_model, 'models/trained/random_forest.pkl')
    
    # 3. XGBoost
    print("Training XGBoost...")
    xgb_model = XGBClassifier(n_estimators=100, learning_rate=0.1, random_state=42, use_label_encoder=False, eval_metric='logloss')
    xgb_model.fit(X_train, y_train)
    # Save using joblib for simplicity or native xgb save (we use joblib)
    joblib.dump(xgb_model, 'models/trained/xgboost.pkl')
    
    # 4. MLP Classifier (Neural Network substitution for LSTM)
    print("Training MLP (Neural Network)...")
    mlp_model = MLPClassifier(hidden_layer_sizes=(32, 16), activation='relu', max_iter=200, random_state=42, early_stopping=True)
    mlp_model.fit(X_train, y_train)
    joblib.dump(mlp_model, 'models/trained/mlp_model.pkl')
    
    print("All models trained and saved successfully.")

if __name__ == "__main__":
    train_models()
