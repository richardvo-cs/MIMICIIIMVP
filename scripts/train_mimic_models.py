#!/usr/bin/env python3
"""
MIMIC-III Model Training Script

This script trains prediction models using the MIMIC-III database data.
The model includes:
1. Mortality prediction
"""

import os
import sys
import logging
import pickle
import pandas as pd
import numpy as np
from pathlib import Path
from datetime import datetime
from joblib import dump, load
from sqlalchemy import create_engine, text
import configparser
import matplotlib.pyplot as plt
import seaborn as sns

# Machine learning libraries
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    roc_auc_score, accuracy_score, f1_score, precision_score, recall_score,
    confusion_matrix, classification_report, roc_curve, precision_recall_curve
)
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
import xgboost as xgb

# Setup logging
logging.basicConfig(level=logging.INFO,
                   format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Ensure the models directory exists
Path("models/mortality").mkdir(parents=True, exist_ok=True)
Path("data/features").mkdir(parents=True, exist_ok=True)
# Ensure output directory for evaluation exists
Path("model_evaluation").mkdir(parents=True, exist_ok=True)

def read_db_config(config_file='database.ini'):
    """Read database connection parameters from config file."""
    
    if not os.path.exists(config_file):
        logger.error(f"Config file {config_file} not found")
        return None
    
    config = configparser.ConfigParser()
    config.read(config_file)
    
    if 'postgresql' not in config:
        logger.error("Section 'postgresql' not found in config file")
        return None
    
    return {
        'host': config['postgresql'].get('host', 'localhost'),
        'database': config['postgresql'].get('database', 'mimiciii'),
        'user': config['postgresql'].get('user', 'mimicuser'),
        'password': config['postgresql'].get('password', 'password'),
        'port': config['postgresql'].get('port', '5432')
    }

def get_db_connection():
    """Create a database connection."""
    db_config = read_db_config()
    if not db_config:
        logger.error("Failed to read database configuration")
        return None
    
    connection_string = f"postgresql://{db_config['user']}:{db_config['password']}@{db_config['host']}:{db_config['port']}/{db_config['database']}"
    
    try:
        engine = create_engine(connection_string)
        return engine
    except Exception as e:
        logger.error(f"Error connecting to database: {e}")
        return None

def extract_features_for_mortality():
    """Extract features from MIMIC-III for mortality prediction."""
    logger.info("Extracting features for mortality prediction model...")
    
    engine = get_db_connection()
    if not engine:
        return None
    
    # SQL query to extract features for mortality prediction
    # This query joins patient data with ICU stays and extracts relevant features
    query = """
    SELECT 
        p.subject_id,
        p.gender,
        EXTRACT(EPOCH FROM (p.dod - p.dob))/60/60/24/365.242 as age_at_death,
        EXTRACT(EPOCH FROM (adm.admittime - p.dob))/60/60/24/365.242 as age_at_admission,
        adm.admission_type,
        adm.hospital_expire_flag,
        icu.first_careunit,
        icu.los as icu_length_of_stay,
        vs.heart_rate_mean,
        vs.heart_rate_max,
        vs.heart_rate_min,
        vs.resp_rate_mean,
        vs.resp_rate_max,
        vs.resp_rate_min,
        vs.temperature_mean,
        vs.temperature_max,
        vs.temperature_min,
        vs.sbp_mean,
        vs.sbp_max,
        vs.sbp_min,
        vs.spo2_mean,
        vs.spo2_min,
        lab.wbc_max,
        lab.wbc_min,
        lab.creatinine_max,
        lab.lactate_max,
        COUNT(DISTINCT diag.icd9_code) as diagnosis_count
    FROM 
        patients p
    JOIN 
        admissions adm ON p.subject_id = adm.subject_id
    JOIN 
        icustays icu ON adm.hadm_id = icu.hadm_id
    LEFT JOIN (
        -- Vital signs subquery (simplified for this example)
        SELECT 
            c.icustay_id,
            AVG(CASE WHEN c.itemid IN (211, 220045) THEN c.valuenum ELSE NULL END) as heart_rate_mean,
            MAX(CASE WHEN c.itemid IN (211, 220045) THEN c.valuenum ELSE NULL END) as heart_rate_max,
            MIN(CASE WHEN c.itemid IN (211, 220045) THEN c.valuenum ELSE NULL END) as heart_rate_min,
            AVG(CASE WHEN c.itemid IN (615, 618, 220210) THEN c.valuenum ELSE NULL END) as resp_rate_mean,
            MAX(CASE WHEN c.itemid IN (615, 618, 220210) THEN c.valuenum ELSE NULL END) as resp_rate_max,
            MIN(CASE WHEN c.itemid IN (615, 618, 220210) THEN c.valuenum ELSE NULL END) as resp_rate_min,
            AVG(CASE WHEN c.itemid IN (676, 678, 223761) THEN c.valuenum ELSE NULL END) as temperature_mean,
            MAX(CASE WHEN c.itemid IN (676, 678, 223761) THEN c.valuenum ELSE NULL END) as temperature_max,
            MIN(CASE WHEN c.itemid IN (676, 678, 223761) THEN c.valuenum ELSE NULL END) as temperature_min,
            AVG(CASE WHEN c.itemid IN (51, 442, 455, 6701, 220179, 220050) THEN c.valuenum ELSE NULL END) as sbp_mean,
            MAX(CASE WHEN c.itemid IN (51, 442, 455, 6701, 220179, 220050) THEN c.valuenum ELSE NULL END) as sbp_max,
            MIN(CASE WHEN c.itemid IN (51, 442, 455, 6701, 220179, 220050) THEN c.valuenum ELSE NULL END) as sbp_min,
            AVG(CASE WHEN c.itemid IN (646, 220277) THEN c.valuenum ELSE NULL END) as spo2_mean,
            MIN(CASE WHEN c.itemid IN (646, 220277) THEN c.valuenum ELSE NULL END) as spo2_min
        FROM 
            chartevents c
        WHERE 
            c.valuenum IS NOT NULL
        GROUP BY 
            c.icustay_id
    ) vs ON icu.icustay_id = vs.icustay_id
    LEFT JOIN (
        -- Lab tests subquery (simplified)
        SELECT 
            adm.hadm_id,
            MAX(CASE WHEN l.itemid = 51301 THEN l.valuenum ELSE NULL END) as wbc_max,
            MIN(CASE WHEN l.itemid = 51301 THEN l.valuenum ELSE NULL END) as wbc_min,
            MAX(CASE WHEN l.itemid = 50912 THEN l.valuenum ELSE NULL END) as creatinine_max,
            MAX(CASE WHEN l.itemid = 50813 THEN l.valuenum ELSE NULL END) as lactate_max
        FROM 
            labevents l
        JOIN 
            admissions adm ON l.hadm_id = adm.hadm_id
        WHERE 
            l.valuenum IS NOT NULL
        GROUP BY 
            adm.hadm_id
    ) lab ON adm.hadm_id = lab.hadm_id
    LEFT JOIN 
        diagnoses_icd diag ON adm.hadm_id = diag.hadm_id
    GROUP BY 
        p.subject_id, p.gender, p.dob, p.dod, adm.admittime, adm.admission_type, 
        adm.hospital_expire_flag, icu.first_careunit, icu.los,
        vs.heart_rate_mean, vs.heart_rate_max, vs.heart_rate_min,
        vs.resp_rate_mean, vs.resp_rate_max, vs.resp_rate_min,
        vs.temperature_mean, vs.temperature_max, vs.temperature_min,
        vs.sbp_mean, vs.sbp_max, vs.sbp_min, vs.spo2_mean, vs.spo2_min,
        lab.wbc_max, lab.wbc_min, lab.creatinine_max, lab.lactate_max
    """
    
    try:
        logger.info("Executing query for mortality features")
        df = pd.read_sql(query, engine)
        logger.info(f"Extracted {len(df)} records for mortality prediction")
        
        # Save the extracted features
        feature_path = Path("data/features/mortality_features.pkl")
        df.to_pickle(feature_path)
        logger.info(f"Features saved to {feature_path}")
        
        return df
    except Exception as e:
        logger.error(f"Error extracting mortality features: {e}")
        return None

def preprocess_data(df, target_column, test_size=0.25, random_state=42):
    """
    Preprocess data for model training.
    
    Args:
        df (pandas.DataFrame): Input features dataframe
        target_column (str): Name of the target column
        test_size (float): Proportion of data to use for testing
        random_state (int): Random seed for reproducibility
        
    Returns:
        tuple: (X_train, X_test, y_train, y_test, feature_names)
    """
    logger.info(f"Preprocessing data for {target_column} prediction")
    
    # Drop rows with NaN in the target column
    df = df.dropna(subset=[target_column])
    
    # Get target
    y = df[target_column]
    
    # Remove target and non-predictive columns
    X = df.drop(columns=[target_column, 'subject_id', 'age_at_death'], errors='ignore')
    
    # Categorical to numeric
    X['gender'] = X['gender'].map({'F': 0, 'M': 1})
    if 'admission_type' in X.columns:
        X['admission_type'] = X['admission_type'].astype('category').cat.codes
    if 'first_careunit' in X.columns:
        X['first_careunit'] = X['first_careunit'].astype('category').cat.codes
    
    # Handle missing values
    X = X.fillna(X.median())
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=y
    )
    
    logger.info(f"Training data shape: {X_train.shape}")
    logger.info(f"Test data shape: {X_test.shape}")
    logger.info(f"Target column distribution: {y.value_counts(normalize=True)}")
    
    return X_train, X_test, y_train, y_test, X.columns.tolist()

def train_mortality_model(features_df=None):
    """Train in-hospital mortality prediction model."""
    logger.info("Training mortality prediction model")
    
    if features_df is None:
        # Try to load from file
        feature_path = Path("data/features/mortality_features.pkl")
        if feature_path.exists():
            features_df = pd.read_pickle(feature_path)
        else:
            features_df = extract_features_for_mortality()
    
    if features_df is None or len(features_df) == 0:
        logger.error("No features available for mortality prediction")
        return None
    
    # Preprocess data
    X_train, X_test, y_train, y_test, feature_names = preprocess_data(
        features_df, target_column='hospital_expire_flag'
    )
    
    # Define model pipeline with Random Forest
    pipeline = Pipeline([
        ('scaler', StandardScaler()),
        ('classifier', RandomForestClassifier(random_state=42))
    ])
    
    # Define hyperparameter search space
    param_grid = {
        'classifier__n_estimators': [100, 200],
        'classifier__max_depth': [None, 10, 20],
        'classifier__min_samples_split': [2, 5, 10]
    }
    
    # Train model with grid search
    grid_search = GridSearchCV(
        pipeline, param_grid, cv=5, scoring='roc_auc', n_jobs=-1, verbose=1
    )
    grid_search.fit(X_train, y_train)
    
    # Get best model
    best_model = grid_search.best_estimator_
    logger.info(f"Best parameters: {grid_search.best_params_}")
    
    # Evaluate model
    y_pred_proba = best_model.predict_proba(X_test)[:, 1]
    y_pred = best_model.predict(X_test)
    
    auc_score = roc_auc_score(y_test, y_pred_proba)
    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred)
    recall = recall_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)
    
    logger.info(f"Mortality model performance - AUC: {auc_score:.4f}, Accuracy: {accuracy:.4f}")
    logger.info(f"Precision: {precision:.4f}, Recall: {recall:.4f}, F1: {f1:.4f}")
    
    # Save model and feature list
    model_path = Path("models/mortality/random_forest_model.joblib")
    feature_list_path = Path("models/mortality/feature_list.pkl")
    
    dump(best_model, model_path)
    with open(feature_list_path, 'wb') as f:
        pickle.dump(feature_names, f)
    
    logger.info(f"Mortality model saved to {model_path}")
    logger.info(f"Feature list saved to {feature_list_path}")
    
    # Return model, test data, and predictions for evaluation
    return best_model, X_test, y_test, y_pred, y_pred_proba, feature_names

def evaluate_model(model, X_test, y_test, y_pred, y_pred_proba, feature_names):
    """Evaluate model performance and generate visualizations."""
    logger.info("Evaluating model and generating visualizations...")
    
    # Set up output directory
    output_dir = Path("model_evaluation")
    
    # Print performance metrics
    auc_score = roc_auc_score(y_test, y_pred_proba)
    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred)
    recall = recall_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)
    
    logger.info("\n===== Model Performance =====")
    logger.info(f"AUC: {auc_score:.4f}")
    logger.info(f"Accuracy: {accuracy:.4f}")
    logger.info(f"Precision: {precision:.4f}")
    logger.info(f"Recall: {recall:.4f}")
    logger.info(f"F1 Score: {f1:.4f}")
    
    # Generate classification report
    logger.info("\n===== Classification Report =====")
    logger.info(classification_report(y_test, y_pred))
    
    # Plot ROC Curve
    fpr, tpr, _ = roc_curve(y_test, y_pred_proba)
    plt.figure(figsize=(8, 6))
    plt.plot(fpr, tpr, label=f'ROC Curve (AUC = {auc_score:.4f})')
    plt.plot([0, 1], [0, 1], 'k--')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver Operating Characteristic Curve')
    plt.legend(loc='lower right')
    plt.grid(True)
    plt.savefig(output_dir / 'roc_curve.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    # Plot Precision-Recall Curve
    precision_curve, recall_curve, _ = precision_recall_curve(y_test, y_pred_proba)
    plt.figure(figsize=(8, 6))
    plt.plot(recall_curve, precision_curve, label='Precision-Recall Curve')
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.title('Precision-Recall Curve')
    plt.legend(loc='upper right')
    plt.grid(True)
    plt.savefig(output_dir / 'precision_recall_curve.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    # Plot Confusion Matrix
    cm = confusion_matrix(y_test, y_pred)
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', cbar=False)
    plt.xlabel('Predicted')
    plt.ylabel('Actual')
    plt.title('Confusion Matrix')
    plt.savefig(output_dir / 'confusion_matrix.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    # Plot Feature Importance
    if hasattr(model, 'named_steps') and hasattr(model.named_steps['classifier'], 'feature_importances_'):
        importance = model.named_steps['classifier'].feature_importances_
        indices = np.argsort(importance)[::-1]
        
        # Take top 10 features
        top_n = min(10, len(feature_names))
        
        plt.figure(figsize=(10, 6))
        plt.barh(range(top_n), importance[indices][:top_n], align='center')
        plt.yticks(range(top_n), [feature_names[i] for i in indices][:top_n])
        plt.xlabel('Importance')
        plt.title(f'Top {top_n} Important Features')
        
        # Print top features
        logger.info(f"\n===== Top {top_n} Important Features =====")
        for i in range(top_n):
            logger.info(f"{i+1}. {feature_names[indices[i]]} ({importance[indices[i]]:.4f})")
        
        plt.savefig(output_dir / 'feature_importance.png', dpi=300, bbox_inches='tight')
        plt.close()
    
    logger.info(f"Evaluation complete. Results saved to {output_dir}/")

def main():
    """Main function to train mortality prediction model."""
    logger.info("Starting MIMIC-III model training")
    
    # Extract features for mortality prediction
    try:
        mortality_features = extract_features_for_mortality()
        logger.info("Feature extraction completed")
    except Exception as e:
        logger.error(f"Error extracting features: {e}")
        mortality_features = None
    
    # Train mortality prediction model
    try:
        model_results = train_mortality_model(mortality_features)
        if model_results:
            model, X_test, y_test, y_pred, y_pred_proba, feature_names = model_results
            logger.info("Mortality prediction model training completed")
            
            # Evaluate model and generate visualizations
            evaluate_model(model, X_test, y_test, y_pred, y_pred_proba, feature_names)
        else:
            logger.error("Mortality prediction model training failed")
    except Exception as e:
        logger.error(f"Error training mortality model: {e}")
        import traceback
        traceback.print_exc()
    
    logger.info("MIMIC-III model training completed")

if __name__ == "__main__":
    main() 