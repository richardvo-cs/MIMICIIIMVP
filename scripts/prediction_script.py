#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
MIMIC-III Clinical Prediction Script
Functions for loading patient data and making clinical predictions using trained models.
"""

import os
import logging
import pickle
import pandas as pd
import numpy as np
from pathlib import Path
from sklearn.preprocessing import StandardScaler
import json

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def load_patient_data(file_path):
    """
    Load patient data from CSV or pickle file.
    
    Args:
        file_path (str): Path to the data file.
        
    Returns:
        pandas.DataFrame: Loaded patient data.
    """
    try:
        print(f"Attempting to load patient data from {file_path}")
        print(f"File exists: {os.path.exists(file_path)}")
        
        if file_path.endswith('.csv'):
            data = pd.read_csv(file_path)
        elif file_path.endswith('.pkl'):
            data = pd.read_pickle(file_path)
        else:
            logger.error(f"Unsupported file format: {file_path}")
            return None
        
        print(f"Successfully loaded data with shape: {data.shape}")
        print(f"Column names: {data.columns.tolist()}")
        logger.info(f"Successfully loaded patient data from {file_path}, {len(data)} patients")
        return data
    except Exception as e:
        logger.error(f"Error loading patient data: {e}")
        import traceback
        print(f"Detailed error: {traceback.format_exc()}")
        return None

def load_model(model_dir, model_filename="model.pkl"):
    """
    Load a trained model from disk.
    
    Args:
        model_dir (str): Directory containing the model file.
        model_filename (str): Name of the model file (default: "model.pkl").
        
    Returns:
        object: Loaded model or None if loading fails.
    """
    try:
        model_path = os.path.join(model_dir, model_filename)
        print(f"Attempting to load model from {model_path}")
        print(f"File exists: {os.path.exists(model_path)}")
        
        if not os.path.exists(model_path):
            logger.error(f"Model file not found: {model_path}")
            return None
        
        with open(model_path, 'rb') as f:
            model = pickle.load(f)
        
        print(f"Successfully loaded model: {type(model).__name__}")
        logger.info(f"Successfully loaded model from {model_path}")
        return model
    except Exception as e:
        logger.error(f"Error loading model: {e}")
        import traceback
        print(f"Detailed error: {traceback.format_exc()}")
        return None

def load_scaler(model_dir, scaler_filename="scaler.pkl"):
    """
    Load a trained feature scaler from disk.
    
    Args:
        model_dir (str): Directory containing the scaler file.
        scaler_filename (str): Name of the scaler file (default: "scaler.pkl").
        
    Returns:
        StandardScaler: Loaded scaler or None if loading fails.
    """
    try:
        scaler_path = os.path.join(model_dir, scaler_filename)
        print(f"Attempting to load scaler from {scaler_path}")
        print(f"File exists: {os.path.exists(scaler_path)}")
        
        if not os.path.exists(scaler_path):
            logger.warning(f"Scaler file not found: {scaler_path}, will use default scaler")
            print("Creating new default StandardScaler")
            return StandardScaler()
        
        with open(scaler_path, 'rb') as f:
            scaler = pickle.load(f)
        
        print(f"Successfully loaded scaler: {type(scaler).__name__}")
        logger.info(f"Successfully loaded scaler from {scaler_path}")
        return scaler
    except Exception as e:
        logger.error(f"Error loading scaler: {e}")
        import traceback
        print(f"Detailed error: {traceback.format_exc()}")
        return StandardScaler()

def load_feature_list(model_dir, feature_filename="feature_list.pkl"):
    """
    Load the list of features required by the model.
    
    Args:
        model_dir (str): Directory containing the feature list file.
        feature_filename (str): Name of the feature list file (default: "feature_list.pkl").
        
    Returns:
        list: List of feature names or None if loading fails.
    """
    try:
        feature_path = os.path.join(model_dir, feature_filename)
        print(f"Attempting to load feature list from {feature_path}")
        print(f"File exists: {os.path.exists(feature_path)}")
        
        if not os.path.exists(feature_path):
            logger.warning(f"Feature list file not found: {feature_path}")
            return None
        
        with open(feature_path, 'rb') as f:
            feature_list = pickle.load(f)
        
        print(f"Successfully loaded feature list with {len(feature_list)} features")
        print(f"First few features: {feature_list[:5] if len(feature_list) > 5 else feature_list}")
        logger.info(f"Successfully loaded feature list from {feature_path} ({len(feature_list)} features)")
        return feature_list
    except Exception as e:
        logger.error(f"Error loading feature list: {e}")
        import traceback
        print(f"Detailed error: {traceback.format_exc()}")
        return None

def load_thresholds(config_path="config/thresholds.json"):
    """Load prediction thresholds from config file."""
    try:
        with open(config_path, 'r') as f:
            thresholds = json.load(f)
        return thresholds
    except Exception as e:
        print(f"Error loading thresholds from {config_path}: {str(e)}")
        # Return default thresholds if config cannot be loaded
        return {
            "mortality": {"prediction_threshold": 0.5},
            "sepsis": {"prediction_threshold": 0.5},
            "readmission": {"prediction_threshold": 0.5}
        }

def preprocess_features(df, model_dir):
    """Preprocess features according to model requirements."""
    try:
        # Load feature list
        feature_list = load_feature_list(model_dir)
        if feature_list is None:
            raise ValueError("Feature list not found")
        
        # Create a mapping of similar feature names
        feature_mapping = {
            # Vital signs
            'heart_rate': ['hr', 'heartrate'],
            'resp_rate': ['respiratory_rate', 'rr'],
            'temperature': ['temp'],
            'systolic_bp': ['sbp', 'systolic'],
            'diastolic_bp': ['dbp', 'diastolic'],
            'spo2': ['oxygen_saturation', 'o2_sat'],
            
            # Lab values
            'wbc': ['white_blood_cells', 'white_blood_cell_count'],
            'hgb': ['hemoglobin', 'hb'],
            'platelet': ['platelets', 'plt'],
            'sodium': ['na'],
            'potassium': ['k'],
            'chloride': ['cl'],
            'bicarbonate': ['hco3', 'bicarb'],
            'bun': ['blood_urea_nitrogen', 'urea'],
            'creatinine': ['cr'],
            'alt': ['alanine_aminotransferase', 'sgot'],
            'ast': ['aspartate_aminotransferase', 'sgpt'],
            'albumin': ['alb'],
            'total_bilirubin': ['bilirubin', 'tbili'],
            'lactate': ['lactic_acid'],
            'troponin': ['troponin_t', 'troponin_i'],
            'inr': ['international_normalized_ratio'],
            'ptt': ['partial_thromboplastin_time'],
            'ph': ['blood_ph'],
            'pao2': ['arterial_oxygen', 'p_o2'],
            'paco2': ['arterial_carbon_dioxide', 'p_co2'],
            
            # Demographics
            'gender': ['sex'],
            'weight': ['wt'],
            'height': ['ht'],
            'bmi': ['body_mass_index'],
            
            # Additional features from test data
            'gcs_score': ['glasgow_coma_scale', 'gcs'],
            'mech_vent': ['mechanical_ventilation', 'vent'],
            'vasopressor_use': ['vasopressors'],
            'urine_output': ['urine', 'uo']
        }
        
        # Reverse mapping for easier lookup
        reverse_mapping = {}
        for standard_name, variations in feature_mapping.items():
            for var in variations:
                reverse_mapping[var] = standard_name
        
        # Rename columns based on mapping
        df = df.rename(columns=reverse_mapping)
        
        # Extract patient IDs if present
        patient_ids = None
        if 'patient_id' in df.columns:
            patient_ids = df['patient_id']
            df = df.drop('patient_id', axis=1)
        
        # Select only the features needed
        available_features = [f for f in feature_list if f in df.columns]
        missing_features = [f for f in feature_list if f not in df.columns]
        
        if missing_features:
            print(f"Warning: Missing features: {missing_features}")
            print("Imputing missing features with realistic values")
            
            # Input missing features with realistic values
            for feature in missing_features:
                if feature == 'age':
                    df[feature] = 65  # Average age
                elif feature == 'gender':
                    df[feature] = 0  # Default to male
                elif feature == 'weight':
                    df[feature] = 80  # Average weight in kg
                elif feature == 'height':
                    df[feature] = 170  # Average height in cm
                elif feature == 'bmi':
                    df[feature] = 25  # Average BMI
                elif feature == 'heart_rate':
                    df[feature] = 80  # Normal heart rate
                elif feature == 'resp_rate':
                    df[feature] = 16  # Normal respiratory rate
                elif feature == 'temperature':
                    df[feature] = 37  # Normal temperature
                elif feature == 'systolic_bp':
                    df[feature] = 120  # Normal systolic BP
                elif feature == 'diastolic_bp':
                    df[feature] = 80  # Normal diastolic BP
                elif feature == 'spo2':
                    df[feature] = 98  # Normal SpO2
                elif feature == 'glucose':
                    df[feature] = 100  # Normal glucose
                elif feature == 'wbc':
                    df[feature] = 7  # Normal WBC
                elif feature == 'hgb':
                    df[feature] = 13  # Normal hemoglobin
                elif feature == 'platelet':
                    df[feature] = 250  # Normal platelet count
                elif feature == 'sodium':
                    df[feature] = 140  # Normal sodium
                elif feature == 'potassium':
                    df[feature] = 4  # Normal potassium
                elif feature == 'chloride':
                    df[feature] = 100  # Normal chloride
                elif feature == 'bicarbonate':
                    df[feature] = 24  # Normal bicarbonate
                elif feature == 'bun':
                    df[feature] = 15  # Normal BUN
                elif feature == 'creatinine':
                    df[feature] = 1  # Normal creatinine
                elif feature == 'alt':
                    df[feature] = 30  # Normal ALT
                elif feature == 'ast':
                    df[feature] = 30  # Normal AST
                elif feature == 'albumin':
                    df[feature] = 4  # Normal albumin
                elif feature == 'total_bilirubin':
                    df[feature] = 0.7  # Normal bilirubin
                elif feature == 'lactate':
                    df[feature] = 1  # Normal lactate
                elif feature == 'troponin':
                    df[feature] = 0.01  # Normal troponin
                elif feature == 'inr':
                    df[feature] = 1  # Normal INR
                elif feature == 'ptt':
                    df[feature] = 30  # Normal PTT
                elif feature == 'ph':
                    df[feature] = 7.4  # Normal pH
                elif feature == 'pao2':
                    df[feature] = 95  # Normal PaO2
                elif feature == 'paco2':
                    df[feature] = 40  # Normal PaCO2
        
        # Ensure all features are in the correct order
        df = df[feature_list]
        
        # Load and apply scaler
        scaler = load_scaler(model_dir)
        if scaler is None:
            raise ValueError("Scaler not found")
        
        X = scaler.transform(df)
        
        return X, patient_ids
        
    except Exception as e:
        print(f"Error in preprocessing features: {str(e)}")
        raise

def predict_mortality(data, model_dir):
    """Predict in-hospital mortality for patients."""
    try:
        # Load model
        model = load_model(model_dir)
        if model is None:
            raise ValueError("Model not found")
        
        # Load threshold from config
        thresholds = load_thresholds()
        mortality_threshold = thresholds["mortality"]["prediction_threshold"]
        
        # Preprocess features
        X, patient_ids = preprocess_features(data, model_dir)
        if X is None:
            raise ValueError("Feature preprocessing failed")
        
        # Make predictions
        probabilities = model.predict_proba(X)[:, 1]
        predictions = (probabilities >= mortality_threshold).astype(int)
        
        # Create results DataFrame
        results = pd.DataFrame({
            'patient_id': patient_ids if patient_ids is not None else range(len(X)),
            'mortality_probability': probabilities,
            'mortality_prediction': predictions,
            'threshold_used': mortality_threshold
        })
        
        return results
        
    except Exception as e:
        print(f"Error in mortality prediction: {str(e)}")
        raise

##def predict_los(data, model_dir):
    """Predict length of stay for patients."""
    try:
        # Load model
        model = load_model(model_dir)
        if model is None:
            raise ValueError("Model not found")
        
        # Preprocess features
        X, patient_ids = preprocess_features(data, model_dir)
        if X is None:
            raise ValueError("Feature preprocessing failed")
        
        # Make predictions
        predictions = model.predict(X)
        
        # Create results DataFrame
        results = pd.DataFrame({
            'patient_id': patient_ids if patient_ids is not None else range(len(X)),
            'los_prediction': predictions
        })
        
        return results
        
    except Exception as e:
        print(f"Error in length of stay prediction: {str(e)}")
        raise

##def predict_readmission(data, model_dir):
    """Predict 30-day readmission for patients."""
    try:
        # Load model
        model = load_model(model_dir)
        if model is None:
            raise ValueError("Model not found")
        
        # Load threshold from config
        thresholds = load_thresholds()
        readmission_threshold = thresholds["readmission"]["prediction_threshold"]
        
        # Preprocess features
        X, patient_ids = preprocess_features(data, model_dir)
        if X is None:
            raise ValueError("Feature preprocessing failed")
        
        # Make predictions
        probabilities = model.predict_proba(X)[:, 1]
        predictions = (probabilities >= readmission_threshold).astype(int)
        
        # Create results DataFrame
        results = pd.DataFrame({
            'patient_id': patient_ids if patient_ids is not None else range(len(X)),
            'readmission_probability': probabilities,
            'readmission_prediction': predictions,
            'threshold_used': readmission_threshold
        })
        
        return results
        
    except Exception as e:
        print(f"Error in readmission prediction: {str(e)}")
        raise

##def predict_sepsis(data, model_dir):
    """Predict sepsis for patients."""
    try:
        # Load model
        model = load_model(model_dir)
        if model is None:
            raise ValueError("Model not found")
        
        # Load threshold from config
        thresholds = load_thresholds()
        sepsis_threshold = thresholds["sepsis"]["prediction_threshold"]
        
        # Preprocess features
        X, patient_ids = preprocess_features(data, model_dir)
        if X is None:
            raise ValueError("Feature preprocessing failed")
        
        # Make predictions
        probabilities = model.predict_proba(X)[:, 1]
        predictions = (probabilities >= sepsis_threshold).astype(int)
        
        # Create results DataFrame
        results = pd.DataFrame({
            'patient_id': patient_ids if patient_ids is not None else range(len(X)),
            'sepsis_probability': probabilities,
            'sepsis_prediction': predictions,
            'threshold_used': sepsis_threshold
        })
        
        return results
        
    except Exception as e:
        print(f"Error in sepsis prediction: {str(e)}")
        raise

if __name__ == "__main__":
    # Example usage
    import sys
    
    if len(sys.argv) < 3:
        print("Usage: python prediction_script.py <input_file> <output_file>")
        sys.exit(1)
    
    input_file = sys.argv[1]
    output_file = sys.argv[2]
    
    # Load data
    data = load_patient_data(input_file)
    if data is None:
        print(f"Failed to load data from {input_file}")
        sys.exit(1)
    
    # Make mortality predictions
    try:
        results = predict_mortality(data, "models/mortality")
        print(f"Successfully made mortality predictions for {len(results)} patients")
    except Exception as e:
        print(f"Error making mortality predictions: {e}")
        sys.exit(1)
    
    # Save results
    try:
        if output_file.endswith('.csv'):
            results.to_csv(output_file, index=False)
        elif output_file.endswith('.pkl'):
            results.to_pickle(output_file)
        else:
            results.to_csv(output_file, index=False)  # Default to CSV
        
        print(f"Successfully saved predictions to {output_file}, {len(results)} patients")
    except Exception as e:
        print(f"Error saving results: {e}")
        sys.exit(1) 