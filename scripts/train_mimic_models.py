#!/usr/bin/env python3
"""
MIMIC-III Model Training Script

This script trains prediction models using the MIMIC-III database data.
The models include:
1. Mortality prediction
2. Sepsis prediction
3. Readmission prediction
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

# Machine learning libraries
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score, accuracy_score, f1_score, precision_score, recall_score
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
import xgboost as xgb

# Setup logging
logging.basicConfig(level=logging.INFO,
                   format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Ensure the models directory exists
Path("models/mortality").mkdir(parents=True, exist_ok=True)
Path("models/sepsis").mkdir(parents=True, exist_ok=True)
Path("models/readmission").mkdir(parents=True, exist_ok=True)
Path("data/features").mkdir(parents=True, exist_ok=True)

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

def extract_features_for_sepsis():
    """Extract features from MIMIC-III for sepsis prediction."""
    logger.info("Extracting features for sepsis prediction model...")
    
    engine = get_db_connection()
    if not engine:
        return None
    
    # SQL query for sepsis prediction
    # This query identifies sepsis cases using ICD codes and extracts relevant features
    query = """
    SELECT 
        p.subject_id,
        p.gender,
        EXTRACT(EPOCH FROM (adm.admittime - p.dob))/60/60/24/365.242 as age_at_admission,
        adm.admission_type,
        icu.first_careunit,
        icu.los as icu_length_of_stay,
        vs.heart_rate_mean,
        vs.heart_rate_max,
        vs.resp_rate_mean,
        vs.resp_rate_max,
        vs.temperature_mean,
        vs.temperature_max,
        vs.sbp_mean,
        vs.sbp_min,
        vs.spo2_mean,
        vs.spo2_min,
        lab.wbc_max,
        lab.wbc_min,
        lab.creatinine_max,
        lab.lactate_max,
        lab.platelet_min,
        lab.bilirubin_max,
        CASE WHEN sepsis.sepsis_present = 1 THEN 1 ELSE 0 END as sepsis
    FROM 
        patients p
    JOIN 
        admissions adm ON p.subject_id = adm.subject_id
    JOIN 
        icustays icu ON adm.hadm_id = icu.hadm_id
    LEFT JOIN (
        -- Vital signs subquery
        SELECT 
            c.icustay_id,
            AVG(CASE WHEN c.itemid IN (211, 220045) THEN c.valuenum ELSE NULL END) as heart_rate_mean,
            MAX(CASE WHEN c.itemid IN (211, 220045) THEN c.valuenum ELSE NULL END) as heart_rate_max,
            AVG(CASE WHEN c.itemid IN (615, 618, 220210) THEN c.valuenum ELSE NULL END) as resp_rate_mean,
            MAX(CASE WHEN c.itemid IN (615, 618, 220210) THEN c.valuenum ELSE NULL END) as resp_rate_max,
            AVG(CASE WHEN c.itemid IN (676, 678, 223761) THEN c.valuenum ELSE NULL END) as temperature_mean,
            MAX(CASE WHEN c.itemid IN (676, 678, 223761) THEN c.valuenum ELSE NULL END) as temperature_max,
            AVG(CASE WHEN c.itemid IN (51, 442, 455, 6701, 220179, 220050) THEN c.valuenum ELSE NULL END) as sbp_mean,
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
        -- Lab tests subquery
        SELECT 
            adm.hadm_id,
            MAX(CASE WHEN l.itemid = 51301 THEN l.valuenum ELSE NULL END) as wbc_max,
            MIN(CASE WHEN l.itemid = 51301 THEN l.valuenum ELSE NULL END) as wbc_min,
            MAX(CASE WHEN l.itemid = 50912 THEN l.valuenum ELSE NULL END) as creatinine_max,
            MAX(CASE WHEN l.itemid = 50813 THEN l.valuenum ELSE NULL END) as lactate_max,
            MIN(CASE WHEN l.itemid = 51265 THEN l.valuenum ELSE NULL END) as platelet_min,
            MAX(CASE WHEN l.itemid = 50885 THEN l.valuenum ELSE NULL END) as bilirubin_max
        FROM 
            labevents l
        JOIN 
            admissions adm ON l.hadm_id = adm.hadm_id
        WHERE 
            l.valuenum IS NOT NULL
        GROUP BY 
            adm.hadm_id
    ) lab ON adm.hadm_id = lab.hadm_id
    LEFT JOIN (
        -- Sepsis identification using ICD-9 codes
        SELECT 
            d.hadm_id,
            MAX(CASE WHEN d.icd9_code IN ('99591', '99592', '78552') THEN 1 ELSE 0 END) as sepsis_present
        FROM 
            diagnoses_icd d
        GROUP BY 
            d.hadm_id
    ) sepsis ON adm.hadm_id = sepsis.hadm_id
    """
    
    try:
        logger.info("Executing query for sepsis features")
        df = pd.read_sql(query, engine)
        logger.info(f"Extracted {len(df)} records for sepsis prediction")
        
        # Save the extracted features
        feature_path = Path("data/features/sepsis_features.pkl")
        df.to_pickle(feature_path)
        logger.info(f"Features saved to {feature_path}")
        
        return df
    except Exception as e:
        logger.error(f"Error extracting sepsis features: {e}")
        return None

def extract_features_for_readmission():
    """Extract features from MIMIC-III for readmission prediction."""
    logger.info("Extracting features for readmission prediction model...")
    
    engine = get_db_connection()
    if not engine:
        return None
    
    # SQL query for readmission prediction
    query = """
    WITH readmissions AS (
        SELECT 
            adm1.subject_id,
            adm1.hadm_id as index_admission_id,
            MIN(adm2.hadm_id) as readmission_id,
            MIN(adm2.admittime) as readmission_time,
            CASE 
                WHEN MIN(adm2.admittime) IS NOT NULL AND
                     EXTRACT(EPOCH FROM (MIN(adm2.admittime) - adm1.dischtime))/60/60/24 <= 30
                THEN 1
                ELSE 0
            END as readmitted_30days
        FROM 
            admissions adm1
        LEFT JOIN 
            admissions adm2 ON adm1.subject_id = adm2.subject_id AND
                              adm1.hadm_id <> adm2.hadm_id AND
                              adm2.admittime > adm1.dischtime
        GROUP BY 
            adm1.subject_id, adm1.hadm_id, adm1.dischtime
    )
    SELECT 
        p.subject_id,
        p.gender,
        EXTRACT(EPOCH FROM (adm.admittime - p.dob))/60/60/24/365.242 as age_at_admission,
        adm.admission_type,
        adm.discharge_location,
        icu.first_careunit,
        icu.los as icu_length_of_stay,
        EXTRACT(EPOCH FROM (adm.dischtime - adm.admittime))/60/60/24 as hospital_los,
        vs.heart_rate_mean,
        vs.sbp_mean,
        vs.spo2_mean,
        lab.wbc_max,
        lab.creatinine_max,
        COUNT(DISTINCT diag.icd9_code) as diagnosis_count,
        COUNT(DISTINCT proc.icd9_code) as procedure_count,
        COUNT(DISTINCT med.drug) as medication_count,
        r.readmitted_30days
    FROM 
        patients p
    JOIN 
        admissions adm ON p.subject_id = adm.subject_id
    JOIN 
        readmissions r ON adm.hadm_id = r.index_admission_id
    LEFT JOIN 
        icustays icu ON adm.hadm_id = icu.hadm_id
    LEFT JOIN (
        -- Vital signs subquery
        SELECT 
            c.icustay_id,
            AVG(CASE WHEN c.itemid IN (211, 220045) THEN c.valuenum ELSE NULL END) as heart_rate_mean,
            AVG(CASE WHEN c.itemid IN (51, 442, 455, 6701, 220179, 220050) THEN c.valuenum ELSE NULL END) as sbp_mean,
            AVG(CASE WHEN c.itemid IN (646, 220277) THEN c.valuenum ELSE NULL END) as spo2_mean
        FROM 
            chartevents c
        WHERE 
            c.valuenum IS NOT NULL
        GROUP BY 
            c.icustay_id
    ) vs ON icu.icustay_id = vs.icustay_id
    LEFT JOIN (
        -- Lab tests subquery
        SELECT 
            adm.hadm_id,
            MAX(CASE WHEN l.itemid = 51301 THEN l.valuenum ELSE NULL END) as wbc_max,
            MAX(CASE WHEN l.itemid = 50912 THEN l.valuenum ELSE NULL END) as creatinine_max
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
    LEFT JOIN 
        procedures_icd proc ON adm.hadm_id = proc.hadm_id
    LEFT JOIN (
        SELECT hadm_id, drug
        FROM prescriptions
    ) med ON adm.hadm_id = med.hadm_id
    GROUP BY 
        p.subject_id, p.gender, p.dob, adm.admittime, adm.dischtime, adm.admission_type, 
        adm.discharge_location, icu.first_careunit, icu.los, vs.heart_rate_mean, vs.sbp_mean, 
        vs.spo2_mean, lab.wbc_max, lab.creatinine_max, r.readmitted_30days
    """
    
    try:
        logger.info("Executing query for readmission features")
        df = pd.read_sql(query, engine)
        logger.info(f"Extracted {len(df)} records for readmission prediction")
        
        # Save the extracted features
        feature_path = Path("data/features/readmission_features.pkl")
        df.to_pickle(feature_path)
        logger.info(f"Features saved to {feature_path}")
        
        return df
    except Exception as e:
        logger.error(f"Error extracting readmission features: {e}")
        return None

def preprocess_data(df, target_column, test_size=0.25, random_state=42):
    """Preprocess data for model training."""
    logger.info(f"Preprocessing data for {target_column} prediction")
    
    # Drop rows with missing target
    df = df.dropna(subset=[target_column])
    logger.info(f"Dataset shape after dropping rows with missing target: {df.shape}")
    
    # Handle missing values
    # For numeric columns, fill with median
    numeric_cols = df.select_dtypes(include=['number']).columns
    for col in numeric_cols:
        if df[col].isnull().sum() > 0:
            df[col] = df[col].fillna(df[col].median())
    
    # For categorical columns, fill with most frequent value
    categorical_cols = df.select_dtypes(include=['object']).columns
    for col in categorical_cols:
        if df[col].isnull().sum() > 0:
            df[col] = df[col].fillna(df[col].mode()[0])
    
    # One-hot encode categorical features
    df = pd.get_dummies(df, columns=[col for col in categorical_cols if col != target_column])
    
    # Split features and target
    X = df.drop(columns=[target_column, 'subject_id'])
    y = df[target_column]
    
    # Feature list for model interpretation
    feature_names = X.columns.tolist()
    
    # Train-test split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=y
    )
    
    logger.info(f"Training set: {X_train.shape}, Test set: {X_test.shape}")
    
    return X_train, X_test, y_train, y_test, feature_names

def train_mortality_model(features_df=None):
    """Train mortality prediction model."""
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
    
    # Define model pipeline
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
    f1 = f1_score(y_test, y_pred)
    
    logger.info(f"Mortality model performance - AUC: {auc_score:.4f}, Accuracy: {accuracy:.4f}, F1: {f1:.4f}")
    
    # Save model and feature list
    model_path = Path("models/mortality/random_forest_model.joblib")
    feature_list_path = Path("models/mortality/feature_list.pkl")
    
    dump(best_model, model_path)
    with open(feature_list_path, 'wb') as f:
        pickle.dump(feature_names, f)
    
    logger.info(f"Mortality model saved to {model_path}")
    logger.info(f"Feature list saved to {feature_list_path}")
    
    return best_model

def train_sepsis_model(features_df=None):
    """Train sepsis prediction model."""
    logger.info("Training sepsis prediction model")
    
    if features_df is None:
        # Try to load from file
        feature_path = Path("data/features/sepsis_features.pkl")
        if feature_path.exists():
            features_df = pd.read_pickle(feature_path)
        else:
            features_df = extract_features_for_sepsis()
    
    if features_df is None or len(features_df) == 0:
        logger.error("No features available for sepsis prediction")
        return None
    
    # Preprocess data
    X_train, X_test, y_train, y_test, feature_names = preprocess_data(
        features_df, target_column='sepsis'
    )
    
    # Define model pipeline with XGBoost
    pipeline = Pipeline([
        ('scaler', StandardScaler()),
        ('classifier', xgb.XGBClassifier(random_state=42, use_label_encoder=False, eval_metric='logloss'))
    ])
    
    # Define hyperparameter search space
    param_grid = {
        'classifier__n_estimators': [100, 200],
        'classifier__max_depth': [3, 5, 7],
        'classifier__learning_rate': [0.01, 0.1, 0.2]
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
    
    logger.info(f"Sepsis model performance - AUC: {auc_score:.4f}, Accuracy: {accuracy:.4f}")
    logger.info(f"Precision: {precision:.4f}, Recall: {recall:.4f}")
    
    # Save model and feature list
    model_path = Path("models/sepsis/xgboost_model.joblib")
    feature_list_path = Path("models/sepsis/feature_list.pkl")
    
    dump(best_model, model_path)
    with open(feature_list_path, 'wb') as f:
        pickle.dump(feature_names, f)
    
    logger.info(f"Sepsis model saved to {model_path}")
    logger.info(f"Feature list saved to {feature_list_path}")
    
    return best_model

def train_readmission_model(features_df=None):
    """Train readmission prediction model."""
    logger.info("Training readmission prediction model")
    
    if features_df is None:
        # Try to load from file
        feature_path = Path("data/features/readmission_features.pkl")
        if feature_path.exists():
            features_df = pd.read_pickle(feature_path)
        else:
            features_df = extract_features_for_readmission()
    
    if features_df is None or len(features_df) == 0:
        logger.error("No features available for readmission prediction")
        return None
    
    # Preprocess data
    X_train, X_test, y_train, y_test, feature_names = preprocess_data(
        features_df, target_column='readmitted_30days'
    )
    
    # Define model pipeline with Gradient Boosting
    pipeline = Pipeline([
        ('scaler', StandardScaler()),
        ('classifier', GradientBoostingClassifier(random_state=42))
    ])
    
    # Define hyperparameter search space
    param_grid = {
        'classifier__n_estimators': [100, 200],
        'classifier__max_depth': [3, 5],
        'classifier__learning_rate': [0.01, 0.1]
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
    f1 = f1_score(y_test, y_pred)
    
    logger.info(f"Readmission model performance - AUC: {auc_score:.4f}, Accuracy: {accuracy:.4f}, F1: {f1:.4f}")
    
    # Save model and feature list
    model_path = Path("models/readmission/gradient_boosting_model.joblib")
    feature_list_path = Path("models/readmission/feature_list.pkl")
    
    dump(best_model, model_path)
    with open(feature_list_path, 'wb') as f:
        pickle.dump(feature_names, f)
    
    logger.info(f"Readmission model saved to {model_path}")
    logger.info(f"Feature list saved to {feature_list_path}")
    
    return best_model

def main():
    """Main function to train all models."""
    logger.info("Starting MIMIC-III model training")
    
    # Extract features
    mortality_features = extract_features_for_mortality()
    sepsis_features = extract_features_for_sepsis()
    readmission_features = extract_features_for_readmission()
    
    # Train models
    try:
        mortality_model = train_mortality_model(mortality_features)
        logger.info("Mortality prediction model training completed")
    except Exception as e:
        logger.error(f"Error training mortality model: {e}")
    
    try:
        sepsis_model = train_sepsis_model(sepsis_features)
        logger.info("Sepsis prediction model training completed")
    except Exception as e:
        logger.error(f"Error training sepsis model: {e}")
    
    try:
        readmission_model = train_readmission_model(readmission_features)
        logger.info("Readmission prediction model training completed")
    except Exception as e:
        logger.error(f"Error training readmission model: {e}")
    
    logger.info("MIMIC-III model training completed")

if __name__ == "__main__":
    main() 