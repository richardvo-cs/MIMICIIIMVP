#!/usr/bin/env python3
"""
MIMIC-III Feature Extraction Module

This module provides functions to extract features from the MIMIC-III database.
"""

import os
import sys
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import pickle
from sklearn.preprocessing import StandardScaler

# Import database connection
from mimic_db_connect import connect


def get_icu_cohort(conn, min_age=18, max_age=None, min_los=1, gender=None, limit=None):
    """
    Get a basic cohort of ICU patients with filtering.
    (This is a simplified version of get_basic_icu_cohort from cohort_selection.py)
    
    Returns:
    --------
    pandas.DataFrame
        Dataframe containing patient cohort information
    """
    query = """
    SELECT i.subject_id, i.hadm_id, i.icustay_id, 
           p.gender, p.dob, a.admittime, a.dischtime, 
           i.intime, i.outtime, i.los,
           EXTRACT(EPOCH FROM (a.dischtime - a.admittime))/86400 as hospital_los,
           EXTRACT(EPOCH FROM (a.admittime - p.dob))/86400/365.242 as age,
           a.hospital_expire_flag
    FROM "ICUSTAYS" i
    INNER JOIN "ADMISSIONS" a ON i.hadm_id = a.hadm_id
    INNER JOIN "PATIENTS" p ON i.subject_id = p.subject_id
    WHERE 1=1
    """
    
    # Add filters
    if min_age is not None:
        query += f" AND EXTRACT(EPOCH FROM (a.admittime - p.dob))/86400/365.242 >= {min_age}"
        
    if max_age is not None:
        query += f" AND EXTRACT(EPOCH FROM (a.admittime - p.dob))/86400/365.242 <= {max_age}"
        
    if min_los is not None:
        query += f" AND i.los >= {min_los}"
        
    if gender is not None:
        query += f" AND UPPER(p.gender) = '{gender.upper()}'"
    
    # Order by subject_id, intime
    query += " ORDER BY i.subject_id, i.intime"
    
    # Add limit if specified
    if limit is not None:
        query += f" LIMIT {limit}"
    
    # Execute query
    try:
        cohort_df = pd.read_sql_query(query, conn)
        print(f"Retrieved {len(cohort_df)} ICU stays for feature extraction.")
        return cohort_df
    except Exception as e:
        print(f"Error retrieving ICU cohort: {e}")
        return None


def get_vital_signs(conn, icustay_id):
    """
    Extract vital sign statistics for a specific ICU stay.
    
    Parameters:
    -----------
    conn : psycopg2.connection
        Database connection object
    icustay_id : int
        The ICU stay ID to get vital signs for
    
    Returns:
    --------
    pandas.DataFrame
        Dataframe containing vital sign features
    """
    # Define common vital sign ItemIDs
    vital_sign_items = {
        'Heart Rate': [211, 220045],
        'Systolic BP': [51, 442, 455, 6701, 220179, 220050],
        'Diastolic BP': [8368, 8440, 8441, 8555, 220180, 220051],
        'Mean BP': [456, 52, 6702, 443, 220052, 220181, 225312],
        'Respiratory Rate': [615, 618, 220210, 224690],
        'Temperature': [223761, 678, 223762, 676],
        'SpO2': [646, 220277],
        'GCS': [198, 220739]
    }
    
    # Initialize results dictionary
    results = {'icustay_id': icustay_id}
    
    # For each vital sign type
    for vital_name, item_ids in vital_sign_items.items():
        # Convert item_ids list to string for SQL IN clause
        item_ids_str = ','.join(map(str, item_ids))
        
        # Query to get statistics for this vital sign
        query = f"""
        SELECT 
            AVG(valuenum) as {vital_name}_mean,
            MIN(valuenum) as {vital_name}_min,
            MAX(valuenum) as {vital_name}_max,
            PERCENTILE_CONT(0.5) WITHIN GROUP (ORDER BY valuenum) as {vital_name}_median,
            STDDEV(valuenum) as {vital_name}_std
        FROM "CHARTEVENTS"
        WHERE icustay_id = {icustay_id}
        AND itemid IN ({item_ids_str})
        AND valuenum IS NOT NULL
        AND valuenum > 0
        """
        
        try:
            # Execute query and get results
            df = pd.read_sql_query(query, conn)
            
            # Add results to dictionary
            for col in df.columns:
                results[col] = df[col].values[0] if not df[col].isnull().all() else np.nan
                
        except Exception as e:
            print(f"Error retrieving {vital_name} for ICU stay {icustay_id}: {e}")
            # Add NaN values for all columns for this vital sign
            for suffix in ['_mean', '_min', '_max', '_median', '_std']:
                results[vital_name + suffix] = np.nan
    
    # Return as DataFrame row
    return pd.DataFrame([results])


def get_lab_results(conn, hadm_id):
    """
    Extract lab result statistics for a specific hospital admission.
    
    Parameters:
    -----------
    conn : psycopg2.connection
        Database connection object
    hadm_id : int
        The hospital admission ID to get lab results for
    
    Returns:
    --------
    pandas.DataFrame
        Dataframe containing lab result features
    """
    # Define common lab test ItemIDs
    lab_items = {
        'WBC': [51300, 51301],
        'Hemoglobin': [50811, 51222],
        'Platelet': [51265],
        'Sodium': [50824, 50983],
        'Potassium': [50822, 50971],
        'Bicarbonate': [50803, 50882],
        'BUN': [51006],
        'Creatinine': [50912],
        'Glucose': [50809, 50931],
        'Calcium': [50893],
        'Magnesium': [50960],
        'Phosphate': [50970],
        'Bilirubin': [50885],
        'AST': [50878],
        'ALT': [50861],
        'Albumin': [50862],
        'Lactate': [50813]
    }
    
    # Initialize results dictionary
    results = {'hadm_id': hadm_id}
    
    # For each lab test
    for lab_name, item_ids in lab_items.items():
        # Convert item_ids list to string for SQL IN clause
        item_ids_str = ','.join(map(str, item_ids))
        
        # Query to get statistics for this lab test
        query = f"""
        SELECT 
            AVG(valuenum) as {lab_name}_mean,
            MIN(valuenum) as {lab_name}_min,
            MAX(valuenum) as {lab_name}_max,
            PERCENTILE_CONT(0.5) WITHIN GROUP (ORDER BY valuenum) as {lab_name}_median,
            STDDEV(valuenum) as {lab_name}_std
        FROM "LABEVENTS"
        WHERE hadm_id = {hadm_id}
        AND itemid IN ({item_ids_str})
        AND valuenum IS NOT NULL
        AND valuenum > 0
        """
        
        try:
            # Execute query and get results
            df = pd.read_sql_query(query, conn)
            
            # Add results to dictionary
            for col in df.columns:
                results[col] = df[col].values[0] if not df[col].isnull().all() else np.nan
                
        except Exception as e:
            print(f"Error retrieving {lab_name} for hospital admission {hadm_id}: {e}")
            # Add NaN values for all columns for this lab test
            for suffix in ['_mean', '_min', '_max', '_median', '_std']:
                results[lab_name + suffix] = np.nan
    
    # Return as DataFrame row
    return pd.DataFrame([results])


def get_demographics(patient_row):
    """
    Extract demographic features from patient information.
    
    Parameters:
    -----------
    patient_row : pandas.Series
        A row from the cohort dataframe containing patient information
    
    Returns:
    --------
    dict
        Dictionary containing demographic features
    """
    features = {}
    
    # Add basic demographic features
    features['age'] = patient_row['age']
    features['gender_m'] = 1 if patient_row['gender'] == 'M' else 0
    features['gender_f'] = 1 if patient_row['gender'] == 'F' else 0
    
    # LOS features
    features['los_icu'] = patient_row['los']
    features['los_hospital'] = patient_row['hospital_los']
    
    # Mortality features
    features['hospital_expire_flag'] = patient_row['hospital_expire_flag']
    
    return features


def get_medications(conn, hadm_id):
    """
    Extract medication features for a specific hospital admission.
    
    Parameters:
    -----------
    conn : psycopg2.connection
        Database connection object
    hadm_id : int
        The hospital admission ID to get medications for
    
    Returns:
    --------
    dict
        Dictionary containing medication features
    """
    # Define medication categories to extract
    med_categories = {
        'has_antibiotic': [
            'Amoxicillin', 'Azithromycin', 'Cefazolin', 'Cefepime', 'Ceftriaxone',
            'Ciprofloxacin', 'Clindamycin', 'Doxycycline', 'Levofloxacin', 'Metronidazole',
            'Piperacillin', 'Tazobactam', 'Vancomycin'
        ],
        'has_vasoactive': [
            'Dobutamine', 'Dopamine', 'Epinephrine', 'Norepinephrine', 'Vasopressin',
            'Phenylephrine'
        ],
        'has_sedative': [
            'Dexmedetomidine', 'Fentanyl', 'Lorazepam', 'Midazolam', 'Propofol'
        ],
        'has_analgesic': [
            'Acetaminophen', 'Fentanyl', 'Hydromorphone', 'Morphine', 'Oxycodone'
        ],
        'has_antihypertensive': [
            'Amlodipine', 'Atenolol', 'Captopril', 'Clonidine', 'Enalapril',
            'Hydralazine', 'Lisinopril', 'Metoprolol', 'Nicardipine', 'Nifedipine'
        ]
    }
    
    features = {}
    
    # Initialize all medication features to 0
    for category in med_categories.keys():
        features[category] = 0
        features[f'{category}_count'] = 0
    
    # Total number of distinct medications
    query = f"""
    SELECT COUNT(DISTINCT drug) as med_count
    FROM "PRESCRIPTIONS"
    WHERE hadm_id = {hadm_id}
    """
    
    try:
        df = pd.read_sql_query(query, conn)
        features['med_count'] = df['med_count'].values[0] if not df.empty else 0
    except Exception as e:
        print(f"Error retrieving medication count for hospital admission {hadm_id}: {e}")
        features['med_count'] = 0
    
    # For each medication category
    for category, medications in med_categories.items():
        # Create SQL condition to search for each medication
        like_conditions = []
        for med in medications:
            like_conditions.append(f"UPPER(drug) LIKE '%{med.upper()}%'")
        
        like_condition_str = " OR ".join(like_conditions)
        
        # Query to check if any of these medications exist and count them
        query = f"""
        SELECT COUNT(DISTINCT drug) as category_count
        FROM "PRESCRIPTIONS"
        WHERE hadm_id = {hadm_id}
        AND ({like_condition_str})
        """
        
        try:
            df = pd.read_sql_query(query, conn)
            count = df['category_count'].values[0] if not df.empty else 0
            features[f'{category}_count'] = count
            features[category] = 1 if count > 0 else 0
        except Exception as e:
            print(f"Error retrieving {category} for hospital admission {hadm_id}: {e}")
            features[f'{category}_count'] = 0
            features[category] = 0
    
    return features


def get_procedures(conn, hadm_id):
    """
    Extract procedure features for a specific hospital admission.
    
    Parameters:
    -----------
    conn : psycopg2.connection
        Database connection object
    hadm_id : int
        The hospital admission ID to get procedures for
    
    Returns:
    --------
    dict
        Dictionary containing procedure features
    """
    # Define procedure categories to check for
    proc_categories = {
        'has_mechanical_ventilation': ['9670', '9671', '9672'],
        'has_dialysis': ['3995', '5498'],
        'has_surgery': ['0', '1', '2', '3', '4'],  # First digit for major surgeries
        'has_central_line': ['3891', '3892', '3893'],
        'has_arterial_line': ['3891', '3894']
    }
    
    features = {}
    
    # Initialize all procedure features to 0
    for category in proc_categories.keys():
        features[category] = 0
    
    # Total number of distinct procedures
    query = f"""
    SELECT COUNT(DISTINCT icd9_code) as proc_count
    FROM "PROCEDURES_ICD"
    WHERE hadm_id = {hadm_id}
    """
    
    try:
        df = pd.read_sql_query(query, conn)
        features['proc_count'] = df['proc_count'].values[0] if not df.empty else 0
    except Exception as e:
        print(f"Error retrieving procedure count for hospital admission {hadm_id}: {e}")
        features['proc_count'] = 0
    
    # For each procedure category
    for category, proc_codes in proc_categories.items():
        # Create SQL condition
        if any(len(code) == 1 for code in proc_codes):
            # For categories that check first digit
            like_conditions = []
            for code in proc_codes:
                if len(code) == 1:
                    like_conditions.append(f"icd9_code LIKE '{code}%'")
                else:
                    like_conditions.append(f"icd9_code = '{code}'")
            
            condition_str = " OR ".join(like_conditions)
        else:
            # For exact code matching
            codes_str = "','".join(proc_codes)
            condition_str = f"icd9_code IN ('{codes_str}')"
        
        # Query to check if any of these procedures exist
        query = f"""
        SELECT 1
        FROM "PROCEDURES_ICD"
        WHERE hadm_id = {hadm_id}
        AND ({condition_str})
        LIMIT 1
        """
        
        try:
            df = pd.read_sql_query(query, conn)
            features[category] = 1 if not df.empty else 0
        except Exception as e:
            print(f"Error retrieving {category} for hospital admission {hadm_id}: {e}")
            features[category] = 0
    
    return features


def get_diagnoses(conn, hadm_id):
    """
    Extract diagnosis features for a specific hospital admission.
    
    Parameters:
    -----------
    conn : psycopg2.connection
        Database connection object
    hadm_id : int
        The hospital admission ID to get diagnoses for
    
    Returns:
    --------
    dict
        Dictionary containing diagnosis features
    """
    # Define diagnosis categories to check for (based on ICD-9 codes)
    diag_categories = {
        'has_sepsis': ['99591', '99592', '78552'],
        'has_ami': ['410', '41000', '41001', '41002'],
        'has_chf': ['39891', '4280', '4281', '42820', '42821', '42822', '42830', '42831', '42832', '42833', '42840', '42841', '42842', '42843'],
        'has_cad': ['41400', '41401', '41402', '41403', '41404', '41405', '41406', '41407', '4142', '4143', '4144'],
        'has_copd': ['490', '4910', '4911', '49120', '49121', '4918', '4919', '4920', '4928', '494', '496'],
        'has_asthma': ['493'],
        'has_arf': ['5184', '5185', '5186', '5187', '51881', '51882', '51883', '51884'],
        'has_pneumonia': ['480', '481', '482', '483', '484', '485', '486', '4870', '4871', '4878'],
        'has_diabetes': ['25000', '25001', '25002', '25003', '25010', '25011', '25012', '25013', '25020', '25021', '25022', '25023', '25030', '25031', '25032', '25033', '25040', '25041', '25042', '25043', '25050', '25051', '25052', '25053', '25060', '25061', '25062', '25063', '25070', '25071', '25072', '25073', '25080', '25081', '25082', '25083', '25090', '25091', '25092', '25093'],
        'has_crf': ['5851', '5852', '5853', '5854', '5855', '5856', '5859']
    }
    
    features = {}
    
    # Initialize all diagnosis features to 0
    for category in diag_categories.keys():
        features[category] = 0
    
    # Total number of distinct diagnoses
    query = f"""
    SELECT COUNT(DISTINCT icd9_code) as diag_count
    FROM "DIAGNOSES_ICD"
    WHERE hadm_id = {hadm_id}
    """
    
    try:
        df = pd.read_sql_query(query, conn)
        features['diag_count'] = df['diag_count'].values[0] if not df.empty else 0
    except Exception as e:
        print(f"Error retrieving diagnosis count for hospital admission {hadm_id}: {e}")
        features['diag_count'] = 0
    
    # For each diagnosis category
    for category, diag_codes in diag_categories.items():
        # Create SQL condition
        like_conditions = []
        for code in diag_codes:
            if len(code) < 5:  # For 3-digit codes like '410', we want to match all subcategories
                like_conditions.append(f"icd9_code LIKE '{code}%'")
            else:
                like_conditions.append(f"icd9_code = '{code}'")
        
        like_condition_str = " OR ".join(like_conditions)
        
        # Query to check if any of these diagnoses exist
        query = f"""
        SELECT 1
        FROM "DIAGNOSES_ICD"
        WHERE hadm_id = {hadm_id}
        AND ({like_condition_str})
        LIMIT 1
        """
        
        try:
            df = pd.read_sql_query(query, conn)
            features[category] = 1 if not df.empty else 0
        except Exception as e:
            print(f"Error retrieving {category} for hospital admission {hadm_id}: {e}")
            features[category] = 0
    
    return features


def extract_features(conn, cohort_df, output_file='data/features/icu_features.pkl', normalize=True):
    """
    Extract features for a cohort of patients.
    
    Parameters:
    -----------
    conn : psycopg2.connection
        Database connection object
    cohort_df : pandas.DataFrame
        Dataframe containing cohort information
    output_file : str, optional
        Path to save the features to, default 'data/features/icu_features.pkl'
    normalize : bool, optional
        Whether to normalize numeric features, default True
    
    Returns:
    --------
    pandas.DataFrame
        Dataframe containing extracted features
    """
    # List to store feature dictionaries for each patient
    all_features = []
    
    # Process each patient in the cohort
    total_patients = len(cohort_df)
    for i, (_, patient_row) in enumerate(cohort_df.iterrows()):
        print(f"Processing patient {i+1}/{total_patients} (subject_id: {patient_row['subject_id']}, icustay_id: {patient_row['icustay_id']})")
        
        # Initialize with identifiers
        patient_features = {
            'subject_id': patient_row['subject_id'],
            'hadm_id': patient_row['hadm_id'],
            'icustay_id': patient_row['icustay_id']
        }
        
        # Get demographic features
        patient_features.update(get_demographics(patient_row))
        
        # Get vital sign features
        try:
            vital_features = get_vital_signs(conn, patient_row['icustay_id'])
            # Drop the icustay_id column since we already have it
            vital_features = vital_features.drop(columns=['icustay_id'], errors='ignore')
            patient_features.update(vital_features.iloc[0].to_dict())
        except Exception as e:
            print(f"Error processing vital signs for patient {patient_row['subject_id']}: {e}")
        
        # Get lab result features
        try:
            lab_features = get_lab_results(conn, patient_row['hadm_id'])
            # Drop the hadm_id column since we already have it
            lab_features = lab_features.drop(columns=['hadm_id'], errors='ignore')
            patient_features.update(lab_features.iloc[0].to_dict())
        except Exception as e:
            print(f"Error processing lab results for patient {patient_row['subject_id']}: {e}")
        
        # Get medication features
        try:
            medication_features = get_medications(conn, patient_row['hadm_id'])
            patient_features.update(medication_features)
        except Exception as e:
            print(f"Error processing medications for patient {patient_row['subject_id']}: {e}")
        
        # Get procedure features
        try:
            procedure_features = get_procedures(conn, patient_row['hadm_id'])
            patient_features.update(procedure_features)
        except Exception as e:
            print(f"Error processing procedures for patient {patient_row['subject_id']}: {e}")
        
        # Get diagnosis features
        try:
            diagnosis_features = get_diagnoses(conn, patient_row['hadm_id'])
            patient_features.update(diagnosis_features)
        except Exception as e:
            print(f"Error processing diagnoses for patient {patient_row['subject_id']}: {e}")
        
        # Add to the list of all features
        all_features.append(patient_features)
    
    # Convert to DataFrame
    features_df = pd.DataFrame(all_features)
    
    # Normalize numeric features if requested
    if normalize:
        print("Normalizing numeric features...")
        
        # Identify numeric columns for normalization (excluding binary/categorical and ID columns)
        id_cols = ['subject_id', 'hadm_id', 'icustay_id']
        binary_cols = [col for col in features_df.columns if features_df[col].isin([0, 1]).all()]
        
        numeric_cols = features_df.select_dtypes(include=[np.number]).columns.tolist()
        numeric_cols = [col for col in numeric_cols if col not in id_cols and col not in binary_cols]
        
        # Fill NaN values with mean for normalization
        for col in numeric_cols:
            features_df[col] = features_df[col].fillna(features_df[col].mean())
        
        # Normalize
        scaler = StandardScaler()
        features_df[numeric_cols] = scaler.fit_transform(features_df[numeric_cols])
    
    # Save features if output file is specified
    if output_file:
        try:
            # Create directory if it doesn't exist
            os.makedirs(os.path.dirname(os.path.abspath(output_file)), exist_ok=True)
            
            # Save based on file extension
            if output_file.endswith('.pkl'):
                with open(output_file, 'wb') as f:
                    pickle.dump(features_df, f)
            elif output_file.endswith('.csv'):
                features_df.to_csv(output_file, index=False)
            else:
                print(f"Unsupported file format: {output_file}")
            
            print(f"Saved features to {output_file}")
        except Exception as e:
            print(f"Error saving features: {e}")
    
    # Print feature statistics
    print(f"\nExtracted {len(features_df.columns)} features for {len(features_df)} patients.")
    print(f"Feature columns: {', '.join(features_df.columns)}")
    
    return features_df


def load_features(input_file):
    """
    Load features from a file.
    
    Parameters:
    -----------
    input_file : str
        Path to the features file to load (supports .pkl, .csv)
    
    Returns:
    --------
    pandas.DataFrame
        The loaded features dataframe, or None if loading failed
    """
    try:
        if input_file.endswith('.pkl'):
            with open(input_file, 'rb') as f:
                features_df = pickle.load(f)
        elif input_file.endswith('.csv'):
            features_df = pd.read_csv(input_file)
        else:
            print(f"Unsupported file format: {input_file}")
            return None
        
        print(f"Loaded features from {input_file} with {len(features_df)} patients and {len(features_df.columns)} features.")
        return features_df
    except Exception as e:
        print(f"Error loading features: {e}")
        return None


if __name__ == "__main__":
    # This script can be run standalone to extract features
    
    # Connect to the database
    conn = connect()
    
    if conn is None:
        print("Failed to connect to the database. Exiting.")
        sys.exit(1)
    
    # Get a cohort of patients
    print("Getting ICU patient cohort...")
    cohort_df = get_icu_cohort(conn, min_age=18, min_los=1, limit=100)
    
    if cohort_df is None or len(cohort_df) == 0:
        print("No patients found in cohort. Exiting.")
        conn.close()
        sys.exit(1)
    
    # Extract features
    print("\nExtracting features for the cohort...")
    features_df = extract_features(conn, cohort_df, output_file='data/features/icu_features.pkl', normalize=True)
    
    # Close the database connection
    conn.close()
    print("\nDatabase connection closed.") 