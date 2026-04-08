import numpy as np
import pandas as pd
import joblib
import json
import shap
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, mean_absolute_error, mean_squared_error
import os

def load_models():
    models = {
        'LogisticRegression': joblib.load('models/trained/logistic_regression.pkl'),
        'RandomForest': joblib.load('models/trained/random_forest.pkl'),
        'XGBoost': joblib.load('models/trained/xgboost.pkl'),
        'MLP': joblib.load('models/trained/mlp_model.pkl')
    }
    return models

def evaluate():
    print("Loading test data...")
    X_test = np.load('data/X_test.npy')
    y_test = np.load('data/y_test.npy')
    feature_names = joblib.load('models/feature_names.pkl')
    
    models = load_models()
    results = {}
    
    for name, model in models.items():
        print(f"Evaluating {name}...")
        y_pred = model.predict(X_test)
        y_pred_probs = model.predict_proba(X_test)[:, 1] if hasattr(model, "predict_proba") else y_pred
            
        acc = accuracy_score(y_test, y_pred)
        prec = precision_score(y_test, y_pred, zero_division=0)
        rec = recall_score(y_test, y_pred, zero_division=0)
        f1 = f1_score(y_test, y_pred, zero_division=0)
        roc = roc_auc_score(y_test, y_pred_probs)
        
        # Pseudo time-series metric (interpreting proba as trend score vs actual)
        rmse = np.sqrt(mean_squared_error(y_test, y_pred_probs))
        mae = mean_absolute_error(y_test, y_pred_probs)
        
        results[name] = {
            'Accuracy': acc,
            'Precision': prec,
            'Recall': rec,
            'F1-Score': f1,
            'ROC-AUC': roc,
            'RMSE': rmse,
            'MAE': mae
        }
        
    os.makedirs('evaluation/metrics', exist_ok=True)
    with open('evaluation/metrics/results.json', 'w') as f:
        json.dump(results, f, indent=4)
        
    print("Evaluation complete. Results saved.")
    
    print("Generating SHAP explainability for XGBoost...")
    xgb_model = models['XGBoost']
    # Sample background for SHAP to speed up
    explainer = shap.Explainer(xgb_model, X_test[:100], feature_names=feature_names)
    shap_values = explainer(X_test[:500])
    
    plt.figure(figsize=(10, 6))
    shap.summary_plot(shap_values, X_test[:500], feature_names=feature_names, show=False)
    plt.tight_layout()
    plt.savefig('evaluation/metrics/shap_summary.png')
    plt.close()
    
    print("SHAP values generated and saved.")

if __name__ == "__main__":
    evaluate()
