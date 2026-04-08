import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
import joblib
import os

def load_and_preprocess_data(file_path='data/synthetic_antibiotic_data.csv', is_inference=False):
    df = pd.read_csv(file_path)
    
    # Target variables
    # 1. Resistance_Status -> 1 (Resistant), 0 (Sensitive)
    df['Resistance_Target'] = df['Resistance_Status'].map({'Resistant': 1, 'Sensitive': 0})
    
    # Handle dates
    df['Treatment_Date'] = pd.to_datetime(df['Treatment_Date'])
    df = df.sort_values(by='Treatment_Date').reset_index(drop=True)
    
    df['Month'] = df['Treatment_Date'].dt.month
    df['DayOfWeek'] = df['Treatment_Date'].dt.dayofweek
    df['Year'] = df['Treatment_Date'].dt.year
    
    # Missing value imputation
    # For Age, use median; for Dosage, use mode
    if 'Age' in df.columns:
        df['Age'] = df['Age'].fillna(df['Age'].median())
    if 'Dosage_mg' in df.columns:
        df['Dosage_mg'] = df['Dosage_mg'].fillna(df['Dosage_mg'].mode()[0])
        
    # Feature Engineering (Historical/Rolling) - Ensure no data leakage (shift is needed)
    # 1. Region Resistance Rate (30 days rolling)
    # We create a dataframe of dates and regions
    daily_region_res = df.groupby(['Treatment_Date', 'Region'])['Resistance_Target'].mean().reset_index()
    daily_region_res.set_index('Treatment_Date', inplace=True)
    
    # Calculate rolling 30d stats
    rolling_res = daily_region_res.groupby('Region')['Resistance_Target'].rolling('30D').mean().shift(1).reset_index()
    rolling_res.rename(columns={'Resistance_Target': 'Rolling_Region_Res_Rate_30d'}, inplace=True)
    
    df = pd.merge(df, rolling_res, on=['Treatment_Date', 'Region'], how='left')
    df['Rolling_Region_Res_Rate_30d'] = df['Rolling_Region_Res_Rate_30d'].fillna(df['Rolling_Region_Res_Rate_30d'].mean())
    df['Rolling_Region_Res_Rate_30d'] = df['Rolling_Region_Res_Rate_30d'].fillna(0) # fallback
    
    # 2. Antibiotic usage count per hospital (30 days rolling)
    daily_hosp_abx = df.assign(Usage=1).groupby(['Treatment_Date', 'Hospital_ID', 'Antibiotic'])['Usage'].sum().reset_index()
    daily_hosp_abx.set_index('Treatment_Date', inplace=True)
    rolling_abx = daily_hosp_abx.groupby(['Hospital_ID', 'Antibiotic'])['Usage'].rolling('30D').sum().shift(1).reset_index()
    rolling_abx.rename(columns={'Usage': 'Rolling_Hosp_Abx_Usage_30d'}, inplace=True)
    
    df = pd.merge(df, rolling_abx, on=['Treatment_Date', 'Hospital_ID', 'Antibiotic'], how='left')
    df['Rolling_Hosp_Abx_Usage_30d'] = df['Rolling_Hosp_Abx_Usage_30d'].fillna(0)
    
    # Encode categorical variables
    categorical_cols = ['Gender', 'Region', 'Hospital_ID', 'Infection_Type', 'Bacterial_Strain', 'Antibiotic']
    
    encoders = {}
    
    if is_inference:
        encoders = joblib.load('models/label_encoders.pkl')
        for col in categorical_cols:
            le = encoders[col]
            # handle unseen labels gracefully or assume known
            df[col] = df[col].astype(str)
            # Find unknown classes and set to a default (first class)
            df.loc[~df[col].isin(le.classes_), col] = le.classes_[0]
            df[col] = le.transform(df[col])
    else:
        os.makedirs('models', exist_ok=True)
        for col in categorical_cols:
            le = LabelEncoder()
            df[col] = le.fit_transform(df[col].astype(str))
            encoders[col] = le
        joblib.dump(encoders, 'models/label_encoders.pkl')

    # Drop columns not for modeling
    features_to_drop = ['Patient_ID', 'Treatment_Date', 'Resistance_Status', 'Treatment_Outcome', 'Resistance_Target']
    X = df.drop(columns=features_to_drop)
    y = df['Resistance_Target']
    
    # Train-test split (Chronological, last 20% is test)
    if not is_inference:
        # Since it's sorted by date
        split_idx = int(len(df) * 0.8)
        X_train, X_test = X.iloc[:split_idx], X.iloc[split_idx:]
        y_train, y_test = y.iloc[:split_idx], y.iloc[split_idx:]
        
        # Scale features
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)
        
        joblib.dump(scaler, 'models/scaler.pkl')
        
        # Save splits
        np.save('data/X_train.npy', X_train_scaled)
        np.save('data/X_test.npy', X_test_scaled)
        np.save('data/y_train.npy', y_train.values)
        np.save('data/y_test.npy', y_test.values)
        
        # Save feature names for SHAP
        joblib.dump(X.columns.tolist(), 'models/feature_names.pkl')
        
        print("Data preprocessed and splits saved to data/ directory.")
        return X_train_scaled, X_test_scaled, y_train.values, y_test.values
    else:
        scaler = joblib.load('models/scaler.pkl')
        X_scaled = scaler.transform(X)
        return X_scaled, df

if __name__ == "__main__":
    load_and_preprocess_data()
