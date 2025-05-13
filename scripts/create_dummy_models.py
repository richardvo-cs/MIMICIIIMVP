#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Create dummy prediction models for testing the MIMIC-III API.
"""

import os
import pickle
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler

# Create dummy feature list
feature_list = [
    'age', 'gender', 'weight', 'height', 'bmi', 
    'heart_rate', 'resp_rate', 'temperature', 'systolic_bp', 'diastolic_bp',
    'spo2', 'glucose', 'wbc', 'hgb', 'platelet', 'sodium', 'potassium',
    'chloride', 'bicarbonate', 'bun', 'creatinine', 'alt', 'ast', 'albumin',
    'total_bilirubin', 'lactate', 'troponin', 'inr', 'ptt', 'ph', 'pao2', 'paco2'
]

def create_model_directory(model_name):
    """Create a directory for a model if it doesn't exist."""
    model_dir = os.path.join('models', model_name)
    os.makedirs(model_dir, exist_ok=True)
    return model_dir

def save_model(model, model_dir, filename='model.pkl'):
    """Save a model to disk."""
    model_path = os.path.join(model_dir, filename)
    with open(model_path, 'wb') as f:
        pickle.dump(model, f)
    print(f"Saved {filename} to {model_dir}")
    return model_path

def create_dummy_classification_model(model_name, random_state=42):
    """Create and save a dummy classification model."""
    model_dir = create_model_directory(model_name)
    
    # Create a dummy logistic regression model with better parameters
    model = LogisticRegression(
        random_state=random_state,
        max_iter=1000,
        class_weight='balanced',
        C=0.1  # Regularization strength
    )
    
    # Generate more realistic feature data
    np.random.seed(random_state)
    n_samples = 10000  # More samples for better training
    X = np.zeros((n_samples, len(feature_list)))
    
    # Generate features with realistic ranges and correlations
    for i, feature in enumerate(feature_list):
        if feature == 'age':
            X[:, i] = np.random.normal(65, 15, n_samples)  # Age: mean 65, std 15
        elif feature == 'gender':
            X[:, i] = np.random.binomial(1, 0.5, n_samples)  # Gender: 50/50 split
        elif feature == 'weight':
            X[:, i] = np.random.normal(80, 20, n_samples)  # Weight in kg
        elif feature == 'height':
            X[:, i] = np.random.normal(170, 10, n_samples)  # Height in cm
        elif feature == 'bmi':
            # BMI calculated from weight and height
            X[:, i] = X[:, feature_list.index('weight')] / (X[:, feature_list.index('height')]/100)**2
        elif feature == 'heart_rate':
            X[:, i] = np.random.normal(80, 20, n_samples)  # Heart rate: 60-100
        elif feature == 'resp_rate':
            X[:, i] = np.random.normal(16, 4, n_samples)  # Respiratory rate: 12-20
        elif feature == 'temperature':
            X[:, i] = np.random.normal(37, 1, n_samples)  # Temperature in Celsius
        elif feature == 'systolic_bp':
            X[:, i] = np.random.normal(120, 20, n_samples)  # Systolic BP: 90-140
        elif feature == 'diastolic_bp':
            X[:, i] = np.random.normal(80, 10, n_samples)  # Diastolic BP: 60-90
        elif feature == 'spo2':
            X[:, i] = np.random.normal(98, 2, n_samples)  # SpO2: 94-100
        elif feature == 'glucose':
            X[:, i] = np.random.normal(100, 20, n_samples)  # Glucose: 70-130
        elif feature == 'wbc':
            X[:, i] = np.random.normal(7, 2, n_samples)  # WBC: 4-10
        elif feature == 'hgb':
            X[:, i] = np.random.normal(13, 2, n_samples)  # Hemoglobin: 12-16
        elif feature == 'platelet':
            X[:, i] = np.random.normal(250, 50, n_samples)  # Platelets: 150-450
        elif feature == 'sodium':
            X[:, i] = np.random.normal(140, 3, n_samples)  # Sodium: 135-145
        elif feature == 'potassium':
            X[:, i] = np.random.normal(4, 0.5, n_samples)  # Potassium: 3.5-5.0
        elif feature == 'chloride':
            X[:, i] = np.random.normal(100, 3, n_samples)  # Chloride: 96-106
        elif feature == 'bicarbonate':
            X[:, i] = np.random.normal(24, 2, n_samples)  # Bicarbonate: 22-26
        elif feature == 'bun':
            X[:, i] = np.random.normal(15, 5, n_samples)  # BUN: 7-20
        elif feature == 'creatinine':
            X[:, i] = np.random.normal(1, 0.3, n_samples)  # Creatinine: 0.6-1.2
        elif feature == 'alt':
            X[:, i] = np.random.normal(30, 10, n_samples)  # ALT: 7-56
        elif feature == 'ast':
            X[:, i] = np.random.normal(30, 10, n_samples)  # AST: 8-48
        elif feature == 'albumin':
            X[:, i] = np.random.normal(4, 0.5, n_samples)  # Albumin: 3.4-5.4
        elif feature == 'total_bilirubin':
            X[:, i] = np.random.normal(0.7, 0.3, n_samples)  # Bilirubin: 0.2-1.2
        elif feature == 'lactate':
            X[:, i] = np.random.normal(1, 0.5, n_samples)  # Lactate: 0.5-2.2
        elif feature == 'troponin':
            X[:, i] = np.random.normal(0.01, 0.01, n_samples)  # Troponin: <0.04
        elif feature == 'inr':
            X[:, i] = np.random.normal(1, 0.2, n_samples)  # INR: 0.8-1.2
        elif feature == 'ptt':
            X[:, i] = np.random.normal(30, 5, n_samples)  # PTT: 25-35
        elif feature == 'ph':
            X[:, i] = np.random.normal(7.4, 0.05, n_samples)  # pH: 7.35-7.45
        elif feature == 'pao2':
            X[:, i] = np.random.normal(95, 5, n_samples)  # PaO2: 75-100
        elif feature == 'paco2':
            X[:, i] = np.random.normal(40, 5, n_samples)  # PaCO2: 35-45
    
    # Generate outcomes with more realistic risk calculations
    # Higher risk for older patients with abnormal vitals and labs
    risk_score = (
        (X[:, feature_list.index('age')] > 75) * 2 +  # Age > 75
        (np.abs(X[:, feature_list.index('heart_rate')] - 80) > 20) * 1.5 +  # Abnormal heart rate
        (np.abs(X[:, feature_list.index('systolic_bp')] - 120) > 30) * 1.5 +  # Abnormal BP
        (X[:, feature_list.index('temperature')] > 38) * 2 +  # Fever
        (X[:, feature_list.index('spo2')] < 90) * 2 +  # Low oxygen
        (X[:, feature_list.index('wbc')] > 12) * 1.5 +  # High WBC
        (X[:, feature_list.index('creatinine')] > 1.5) * 2 +  # High creatinine
        (X[:, feature_list.index('lactate')] > 2) * 2  # High lactate
    )
    
    # Use sigmoid function to get probabilities between 0 and 1
    probabilities = 1 / (1 + np.exp(-risk_score/4))  # Scaled risk score
    y = (probabilities > 0.5).astype(int)
    
    # Ensure reasonable class distribution (10-20% positive)
    while np.mean(y) < 0.1 or np.mean(y) > 0.2:
        probabilities = 1 / (1 + np.exp(-risk_score/4 + np.random.normal(0, 0.5)))
        y = (probabilities > 0.5).astype(int)
    
    # Fit the model
    model.fit(X, y)
    
    # Create and save the scaler
    scaler = StandardScaler()
    scaler.fit(X)
    save_model(scaler, model_dir, 'scaler.pkl')
    
    # Save the model
    save_model(model, model_dir)
    
    # Save the feature list
    feature_path = os.path.join(model_dir, 'feature_list.pkl')
    with open(feature_path, 'wb') as f:
        pickle.dump(feature_list, f)
    print(f"Saved feature list to {feature_path}")
    
    return model, scaler, feature_list

def create_dummy_regression_model(model_name, random_state=42):
    """Create and save a dummy regression model."""
    model_dir = create_model_directory(model_name)
    
    # Create a dummy linear regression model
    model = LinearRegression()
    
    # Generate more realistic feature data
    np.random.seed(random_state)
    n_samples = 1000
    X = np.zeros((n_samples, len(feature_list)))
    
    # Generate features with realistic ranges and correlations
    for i, feature in enumerate(feature_list):
        if feature == 'age':
            X[:, i] = np.random.normal(65, 15, n_samples)  # Age: mean 65, std 15
        elif feature == 'gender':
            X[:, i] = np.random.binomial(1, 0.5, n_samples)  # Gender: 50/50 split
        elif feature == 'weight':
            X[:, i] = np.random.normal(80, 20, n_samples)  # Weight in kg
        elif feature == 'height':
            X[:, i] = np.random.normal(170, 10, n_samples)  # Height in cm
        elif feature == 'bmi':
            # BMI calculated from weight and height
            X[:, i] = X[:, feature_list.index('weight')] / (X[:, feature_list.index('height')]/100)**2
        elif feature == 'heart_rate':
            X[:, i] = np.random.normal(80, 20, n_samples)  # Heart rate: 60-100
        elif feature == 'resp_rate':
            X[:, i] = np.random.normal(16, 4, n_samples)  # Respiratory rate: 12-20
        elif feature == 'temperature':
            X[:, i] = np.random.normal(37, 1, n_samples)  # Temperature in Celsius
        elif feature == 'systolic_bp':
            X[:, i] = np.random.normal(120, 20, n_samples)  # Systolic BP: 90-140
        elif feature == 'diastolic_bp':
            X[:, i] = np.random.normal(80, 10, n_samples)  # Diastolic BP: 60-90
        else:
            X[:, i] = np.random.normal(50, 10, n_samples)  # Other vitals and labs
    
    # Generate LOS with some relationship to features
    # Longer LOS for older patients with abnormal vitals
    los_base = (
        (X[:, feature_list.index('age')] > 75) * 3 +  # Age > 75: +3 days
        (np.abs(X[:, feature_list.index('heart_rate')] - 80) > 20) * 2 +  # Abnormal heart rate: +2 days
        (np.abs(X[:, feature_list.index('systolic_bp')] - 120) > 30) * 2 +  # Abnormal BP: +2 days
        (X[:, feature_list.index('temperature')] > 38) * 3  # Fever: +3 days
    )
    y = los_base + np.random.normal(5, 2, n_samples)  # Add base LOS and random variation
    y = np.maximum(y, 1)  # Ensure minimum LOS of 1 day
    
    # Fit the model
    model.fit(X, y)
    
    # Create and save the scaler
    scaler = StandardScaler()
    scaler.fit(X)
    save_model(scaler, model_dir, 'scaler.pkl')
    
    # Save the model
    save_model(model, model_dir)
    
    # Save the feature list
    feature_path = os.path.join(model_dir, 'feature_list.pkl')
    with open(feature_path, 'wb') as f:
        pickle.dump(feature_list, f)
    print(f"Saved feature_list.pkl to {model_dir}")
    
    return model_dir

def create_dummy_sample_data(output_path='data/uploads/sample_patient_data.csv', n_samples=10):
    """Create a dummy patient dataset for testing."""
    np.random.seed(42)
    
    # Generate random data for all features
    data = {}
    for feature in feature_list:
        if feature == 'gender':
            # Change gender to numeric (0/1) instead of categorical (M/F)
            data[feature] = np.random.randint(0, 2, n_samples)
        elif feature in ['age', 'weight', 'height', 'bmi']:
            data[feature] = np.random.randint(20, 90, n_samples)
        else:
            data[feature] = np.random.rand(n_samples) * 100
    
    # Add patient identifiers
    data['subject_id'] = np.arange(1000, 1000 + n_samples)
    data['hadm_id'] = np.arange(2000, 2000 + n_samples)
    data['icustay_id'] = np.arange(3000, 3000 + n_samples)
    
    # Create DataFrame and save
    df = pd.DataFrame(data)
    
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    df.to_csv(output_path, index=False)
    print(f"Created sample patient data with {n_samples} patients: {output_path}")
    
    return df

if __name__ == "__main__":
    print("Creating dummy models for testing...")
    
    # Create classification models
    create_dummy_classification_model('mortality_prediction')
    create_dummy_classification_model('readmission_prediction')
    create_dummy_classification_model('sepsis_prediction')
    
    # Create regression model
    create_dummy_regression_model('los_prediction')
    
    # Create sample patient data
    create_dummy_sample_data()
    
    print("Done! Dummy models and sample data created successfully.") 