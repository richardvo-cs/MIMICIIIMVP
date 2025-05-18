#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
MIMIC-III Data Preprocessing Script
This script extracts, joins, and preprocesses data from the MIMIC-III database.
"""

import os
import pandas as pd
import numpy as np
from sqlalchemy import create_engine
from tqdm import tqdm

def create_db_connection(user='mimicuser', password='password', host='localhost', 
                        port=5432, db='mimiciii'):
    """Create a database connection to the PostgreSQL database."""
    conn_str = f'postgresql://{user}:{password}@{host}:{port}/{db}'
    engine = create_engine(conn_str)
    return engine

def extract_patient_data(engine):
    """Extract patient demographics from patients table."""
    query = """
    SELECT subject_id, gender, dob, dod, 
           EXTRACT(EPOCH FROM dod - dob) / 86400 / 365.242 AS age_at_death
    FROM patients
    """
    return pd.read_sql(query, engine)

def extract_admissions_data(engine):
    """Extract admission data from admissions table."""
    query = """
    SELECT subject_id, hadm_id, admittime, dischtime, 
           deathtime, admission_type, insurance, language, 
           religion, marital_status, ethnicity, diagnosis
    FROM admissions
    """
    return pd.read_sql(query, engine)

def extract_icu_stays(engine):
    """Extract ICU stay data."""
    query = """
    SELECT subject_id, hadm_id, icustay_id, intime, outtime, 
           EXTRACT(EPOCH FROM outtime - intime) / 86400 AS los_days,
           first_careunit, last_careunit, first_wardid, last_wardid
    FROM icustays
    """
    return pd.read_sql(query, engine)

def extract_vitals(engine, icustay_ids=None):
    """Extract vital signs from the chartevents table."""
    vital_item_ids = {
        'heart_rate': [211, 220045],  # Heart rate
        'sbp': [51, 442, 455, 6701, 220179, 220050],  # Systolic BP
        'dbp': [8368, 8440, 8441, 8555, 220051, 220180],  # Diastolic BP
        'resp_rate': [615, 618, 220210, 224690],  # Respiratory rate
        'temp': [223761, 678, 223762, 676],  # Temperature
        'spo2': [646, 220277]  # Oxygen saturation
    }
    
    all_item_ids = [item for sublist in vital_item_ids.values() for item in sublist]
    item_ids_str = ','.join(str(id) for id in all_item_ids)
    
    if icustay_ids:
        icustay_filter = f"AND icustay_id IN ({','.join(map(str, icustay_ids))})"
    else:
        icustay_filter = ""
    
    query = f"""
    SELECT c.subject_id, c.hadm_id, c.icustay_id, c.charttime, 
           c.itemid, c.valuenum
    FROM chartevents c
    WHERE c.itemid IN ({item_ids_str}) 
          AND c.valuenum IS NOT NULL
          {icustay_filter}
    """
    
    print("Extracting vital signs...")
    vitals = pd.read_sql(query, engine)
    
    # Map itemids to vital sign names
    vitals['vital_type'] = 'unknown'
    for vital_name, item_ids in vital_item_ids.items():
        vitals.loc[vitals['itemid'].isin(item_ids), 'vital_type'] = vital_name
    
    return vitals

def extract_labs(engine, hadm_ids=None):
    """Extract laboratory results from the labevents table."""
    lab_item_ids = {
        'wbc': [51300, 51301],  # White blood cell count
        'hgb': [50811, 51222],  # Hemoglobin
        'platelet': [51265],  # Platelet count
        'sodium': [50824, 50983],  # Sodium
        'potassium': [50822, 50971],  # Potassium
        'bicarbonate': [50803, 50882],  # Bicarbonate
        'bun': [51006],  # Blood urea nitrogen
        'creatinine': [50912]  # Creatinine
    }
    
    all_item_ids = [item for sublist in lab_item_ids.values() for item in sublist]
    item_ids_str = ','.join(str(id) for id in all_item_ids)
    
    if hadm_ids:
        hadm_filter = f"AND hadm_id IN ({','.join(map(str, hadm_ids))})"
    else:
        hadm_filter = ""
    
    query = f"""
    SELECT subject_id, hadm_id, charttime, itemid, valuenum
    FROM labevents
    WHERE itemid IN ({item_ids_str}) 
          AND valuenum IS NOT NULL
          {hadm_filter}
    """
    
    print("Extracting lab results...")
    labs = pd.read_sql(query, engine)
    
    # Map itemids to lab test names
    labs['lab_type'] = 'unknown'
    for lab_name, item_ids in lab_item_ids.items():
        labs.loc[labs['itemid'].isin(item_ids), 'lab_type'] = lab_name
    
    return labs

def filter_impossible_values(vitals_df, labs_df):
    """Remove physiologically impossible values."""
    # Filter vitals
    # Heart rate between 20 and 300
    vitals_df.loc[(vitals_df['vital_type'] == 'heart_rate') & 
                  ((vitals_df['valuenum'] < 20) | (vitals_df['valuenum'] > 300)), 'valuenum'] = np.nan
    
    # SBP between 40 and 300
    vitals_df.loc[(vitals_df['vital_type'] == 'sbp') & 
                  ((vitals_df['valuenum'] < 40) | (vitals_df['valuenum'] > 300)), 'valuenum'] = np.nan
    
    # DBP between 20 and 200
    vitals_df.loc[(vitals_df['vital_type'] == 'dbp') & 
                  ((vitals_df['valuenum'] < 20) | (vitals_df['valuenum'] > 200)), 'valuenum'] = np.nan
    
    # Respiratory rate between 4 and 60
    vitals_df.loc[(vitals_df['vital_type'] == 'resp_rate') & 
                  ((vitals_df['valuenum'] < 4) | (vitals_df['valuenum'] > 60)), 'valuenum'] = np.nan
    
    # Temperature between 25 and 45 (Celsius)
    vitals_df.loc[(vitals_df['vital_type'] == 'temp') & 
                  ((vitals_df['valuenum'] < 25) | (vitals_df['valuenum'] > 45)), 'valuenum'] = np.nan
    
    # SpO2 between 0 and 100
    vitals_df.loc[(vitals_df['vital_type'] == 'spo2') & 
                  ((vitals_df['valuenum'] < 0) | (vitals_df['valuenum'] > 100)), 'valuenum'] = np.nan
    
    # Filter labs (example thresholds - adjust as needed)
    # WBC between 0 and 100
    labs_df.loc[(labs_df['lab_type'] == 'wbc') & 
                ((labs_df['valuenum'] < 0) | (labs_df['valuenum'] > 100)), 'valuenum'] = np.nan
    
    # Hemoglobin between 1 and 30
    labs_df.loc[(labs_df['lab_type'] == 'hgb') & 
                ((labs_df['valuenum'] < 1) | (labs_df['valuenum'] > 30)), 'valuenum'] = np.nan
    
    # Remove rows with NaN values after filtering
    vitals_df = vitals_df.dropna(subset=['valuenum'])
    labs_df = labs_df.dropna(subset=['valuenum'])
    
    return vitals_df, labs_df

def create_patient_timeline(engine, output_dir='../../data/processed'):
    """
    Create a timeline of patient data with vitals and lab results.
    """
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    # Extract data
    print("Extracting patient data...")
    patients = extract_patient_data(engine)
    
    print("Extracting admissions data...")
    admissions = extract_admissions_data(engine)
    
    print("Extracting ICU stays...")
    icu_stays = extract_icu_stays(engine)
    
    # Join patients, admissions, and ICU stays
    print("Joining patient data...")
    patient_timeline = pd.merge(
        icu_stays, 
        admissions,
        on=['subject_id', 'hadm_id'],
        how='inner'
    )
    
    patient_timeline = pd.merge(
        patient_timeline,
        patients,
        on='subject_id',
        how='inner'
    )
    
    # Calculate age at admission
    patient_timeline['admittime'] = pd.to_datetime(patient_timeline['admittime'])
    patient_timeline['dob'] = pd.to_datetime(patient_timeline['dob'])
    patient_timeline['age'] = (patient_timeline['admittime'] - patient_timeline['dob']).dt.days / 365.242
    
    # Extract vitals and labs for these patients
    vitals = extract_vitals(engine, patient_timeline['icustay_id'].tolist())
    labs = extract_labs(engine, patient_timeline['hadm_id'].tolist())
    
    # Filter out impossible values
    vitals, labs = filter_impossible_values(vitals, labs)
    
    # Save processed data
    print("Saving processed data...")
    patient_timeline.to_csv(os.path.join(output_dir, 'patient_timeline.csv'), index=False)
    vitals.to_csv(os.path.join(output_dir, 'vitals.csv'), index=False)
    labs.to_csv(os.path.join(output_dir, 'labs.csv'), index=False)
    
    return patient_timeline, vitals, labs

if __name__ == "__main__":
    # Create database connection
    engine = create_db_connection()
    
    # Process data
    patient_timeline, vitals, labs = create_patient_timeline(engine)
    
    print(f"Processed data for {patient_timeline['subject_id'].nunique()} patients")
    print(f"Extracted {len(vitals)} vital sign measurements")
    print(f"Extracted {len(labs)} lab results") 