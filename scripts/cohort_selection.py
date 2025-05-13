#!/usr/bin/env python3
"""
MIMIC-III Cohort Selection Module

This module provides functions to select and filter patient cohorts from the MIMIC-III database.
"""

import os
import sys
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import pickle

# Function to get a cohort of ICU patients with basic filtering
def get_basic_icu_cohort(conn, min_age=18, max_age=None, min_los=1, gender=None, limit=None):
    """
    Get a basic cohort of ICU patients with filtering.
    
    Parameters:
    -----------
    conn : psycopg2.connection
        Database connection object
    min_age : int, optional
        Minimum patient age in years, default 18
    max_age : int, optional
        Maximum patient age in years, default None (no upper limit)
    min_los : float, optional
        Minimum length of stay in days, default 1
    gender : str, optional
        Filter by gender ('M' or 'F'), default None (no filter)
    limit : int, optional
        Limit the number of patients returned, default None (no limit)
    
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
    FROM icustays i
    INNER JOIN admissions a ON i.hadm_id = a.hadm_id
    INNER JOIN patients p ON i.subject_id = p.subject_id
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
        print(f"Retrieved {len(cohort_df)} ICU stays.")
        return cohort_df
    except Exception as e:
        print(f"Error retrieving ICU cohort: {e}")
        return None


# Function to get first ICU stays only
def get_first_icu_stays(conn, min_age=18, max_age=None, min_los=1, gender=None, limit=None):
    """
    Get a cohort of first ICU stays only for each patient.
    
    Parameters:
    -----------
    conn : psycopg2.connection
        Database connection object
    min_age : int, optional
        Minimum patient age in years, default 18
    max_age : int, optional
        Maximum patient age in years, default None (no upper limit)
    min_los : float, optional
        Minimum length of stay in days, default 1
    gender : str, optional
        Filter by gender ('M' or 'F'), default None (no filter)
    limit : int, optional
        Limit the number of patients returned, default None (no limit)
    
    Returns:
    --------
    pandas.DataFrame
        Dataframe containing patient cohort information
    """
    query = """
    WITH first_icu AS (
        SELECT subject_id, MIN(intime) as first_intime
        FROM icustays
        GROUP BY subject_id
    )
    SELECT i.subject_id, i.hadm_id, i.icustay_id, 
           p.gender, p.dob, a.admittime, a.dischtime, 
           i.intime, i.outtime, i.los,
           EXTRACT(EPOCH FROM (a.dischtime - a.admittime))/86400 as hospital_los,
           EXTRACT(EPOCH FROM (a.admittime - p.dob))/86400/365.242 as age,
           a.hospital_expire_flag
    FROM icustays i
    INNER JOIN admissions a ON i.hadm_id = a.hadm_id
    INNER JOIN patients p ON i.subject_id = p.subject_id
    INNER JOIN first_icu f ON i.subject_id = f.subject_id AND i.intime = f.first_intime
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
        print(f"Retrieved {len(cohort_df)} first ICU stays.")
        return cohort_df
    except Exception as e:
        print(f"Error retrieving first ICU stays: {e}")
        return None


# Function to get a cohort of patients with sepsis
def get_sepsis_cohort(conn, min_age=18, max_age=None, min_los=1, gender=None, limit=None):
    """
    Get a cohort of ICU patients with sepsis based on ICD-9 codes.
    
    Parameters:
    -----------
    conn : psycopg2.connection
        Database connection object
    min_age : int, optional
        Minimum patient age in years, default 18
    max_age : int, optional
        Maximum patient age in years, default None (no upper limit)
    min_los : float, optional
        Minimum length of stay in days, default 1
    gender : str, optional
        Filter by gender ('M' or 'F'), default None (no filter)
    limit : int, optional
        Limit the number of patients returned, default None (no limit)
    
    Returns:
    --------
    pandas.DataFrame
        Dataframe containing sepsis patient cohort information
    """
    query = """
    WITH sepsis_admissions AS (
        SELECT DISTINCT hadm_id
        FROM diagnoses_icd
        WHERE icd9_code IN ('99591', '99592') -- Sepsis and severe sepsis
           OR icd9_code LIKE '038%'           -- Septicemia
    )
    SELECT i.subject_id, i.hadm_id, i.icustay_id, 
           p.gender, p.dob, a.admittime, a.dischtime, 
           i.intime, i.outtime, i.los,
           EXTRACT(EPOCH FROM (a.dischtime - a.admittime))/86400 as hospital_los,
           EXTRACT(EPOCH FROM (a.admittime - p.dob))/86400/365.242 as age,
           a.hospital_expire_flag
    FROM icustays i
    INNER JOIN admissions a ON i.hadm_id = a.hadm_id
    INNER JOIN patients p ON i.subject_id = p.subject_id
    INNER JOIN sepsis_admissions s ON i.hadm_id = s.hadm_id
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
        print(f"Retrieved {len(cohort_df)} ICU stays with sepsis diagnosis.")
        return cohort_df
    except Exception as e:
        print(f"Error retrieving sepsis cohort: {e}")
        return None


# Function to get a cohort of patients on mechanical ventilation
def get_ventilation_cohort(conn, min_age=18, max_age=None, min_vent_hours=24, gender=None, limit=None):
    """
    Get a cohort of ICU patients who received mechanical ventilation.
    
    Parameters:
    -----------
    conn : psycopg2.connection
        Database connection object
    min_age : int, optional
        Minimum patient age in years, default 18
    max_age : int, optional
        Maximum patient age in years, default None (no upper limit)
    min_vent_hours : float, optional
        Minimum hours on ventilation, default 24
    gender : str, optional
        Filter by gender ('M' or 'F'), default None (no filter)
    limit : int, optional
        Limit the number of patients returned, default None (no limit)
    
    Returns:
    --------
    pandas.DataFrame
        Dataframe containing mechanical ventilation patient cohort information
    """
    query = """
    WITH ventilation_episodes AS (
        -- Get mechanical ventilation events from procedures_icd table
        SELECT DISTINCT hadm_id
        FROM procedures_icd p
        JOIN d_icd_procedures d ON p.icd9_code = d.icd9_code
        WHERE 
            d.long_title LIKE '%mechanical ventilation%'
            OR d.long_title LIKE '%intubation%'
            OR p.icd9_code IN ('9670', '9671', '9672') -- Continuous mechanical ventilation
    )
    SELECT i.subject_id, i.hadm_id, i.icustay_id, 
           p.gender, p.dob, a.admittime, a.dischtime, 
           i.intime, i.outtime, i.los,
           EXTRACT(EPOCH FROM (a.dischtime - a.admittime))/86400 as hospital_los,
           EXTRACT(EPOCH FROM (a.admittime - p.dob))/86400/365.242 as age,
           a.hospital_expire_flag
    FROM icustays i
    INNER JOIN admissions a ON i.hadm_id = a.hadm_id
    INNER JOIN patients p ON i.subject_id = p.subject_id
    INNER JOIN ventilation_episodes v ON i.hadm_id = v.hadm_id
    WHERE 1=1
    """
    
    # Add filters
    if min_age is not None:
        query += f" AND EXTRACT(EPOCH FROM (a.admittime - p.dob))/86400/365.242 >= {min_age}"
        
    if max_age is not None:
        query += f" AND EXTRACT(EPOCH FROM (a.admittime - p.dob))/86400/365.242 <= {max_age}"
    
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
        print(f"Retrieved {len(cohort_df)} ICU stays with mechanical ventilation.")
        return cohort_df
    except Exception as e:
        print(f"Error retrieving mechanical ventilation cohort: {e}")
        return None


# Function to get a mortality prediction cohort
def get_mortality_cohort(conn, min_age=18, max_age=None, min_los=1, 
                        exclusion_window_days=30, include_readmissions=False, limit=None):
    """
    Get a cohort for mortality prediction, with options to exclude readmissions.
    
    Parameters:
    -----------
    conn : psycopg2.connection
        Database connection object
    min_age : int, optional
        Minimum patient age in years, default 18
    max_age : int, optional
        Maximum patient age in years, default None (no upper limit)
    min_los : float, optional
        Minimum length of stay in days, default 1
    exclusion_window_days : int, optional
        Exclude readmissions within this many days, default 30
    include_readmissions : bool, optional
        Whether to include readmissions, default False
    limit : int, optional
        Limit the number of patients returned, default None (no limit)
    
    Returns:
    --------
    pandas.DataFrame
        Dataframe containing patient cohort information for mortality prediction
    """
    if include_readmissions:
        query = """
        SELECT i.subject_id, i.hadm_id, i.icustay_id, 
               p.gender, p.dob, a.admittime, a.dischtime, 
               i.intime, i.outtime, i.los,
               EXTRACT(EPOCH FROM (a.dischtime - a.admittime))/86400 as hospital_los,
               EXTRACT(EPOCH FROM (a.admittime - p.dob))/86400/365.242 as age,
               a.hospital_expire_flag,
               p.expire_flag as death_flag,
               CASE WHEN p.dod IS NOT NULL AND p.dod <= a.dischtime + INTERVAL '30 day' THEN 1 ELSE 0 END as day_30_mort
        FROM icustays i
        INNER JOIN admissions a ON i.hadm_id = a.hadm_id
        INNER JOIN patients p ON i.subject_id = p.subject_id
        WHERE 1=1
        """
    else:
        # Exclude readmissions within the exclusion window
        query = f"""
        WITH ordered_admissions AS (
            SELECT 
                a.subject_id, 
                a.hadm_id, 
                a.admittime,
                ROW_NUMBER() OVER (PARTITION BY a.subject_id ORDER BY a.admittime) as admit_order,
                LAG(a.admittime) OVER (PARTITION BY a.subject_id ORDER BY a.admittime) as prev_admittime
            FROM 
                admissions a
        ),
        eligible_admissions AS (
            SELECT 
                o.subject_id, 
                o.hadm_id
            FROM 
                ordered_admissions o
            WHERE 
                o.admit_order = 1 OR 
                (o.prev_admittime IS NULL OR o.admittime > o.prev_admittime + INTERVAL '{exclusion_window_days} day')
        )
        SELECT i.subject_id, i.hadm_id, i.icustay_id, 
               p.gender, p.dob, a.admittime, a.dischtime, 
               i.intime, i.outtime, i.los,
               EXTRACT(EPOCH FROM (a.dischtime - a.admittime))/86400 as hospital_los,
               EXTRACT(EPOCH FROM (a.admittime - p.dob))/86400/365.242 as age,
               a.hospital_expire_flag,
               p.expire_flag as death_flag,
               CASE WHEN p.dod IS NOT NULL AND p.dod <= a.dischtime + INTERVAL '30 day' THEN 1 ELSE 0 END as day_30_mort
        FROM icustays i
        INNER JOIN admissions a ON i.hadm_id = a.hadm_id
        INNER JOIN patients p ON i.subject_id = p.subject_id
        INNER JOIN eligible_admissions e ON i.hadm_id = e.hadm_id
        WHERE 1=1
        """
    
    # Add filters
    if min_age is not None:
        query += f" AND EXTRACT(EPOCH FROM (a.admittime - p.dob))/86400/365.242 >= {min_age}"
        
    if max_age is not None:
        query += f" AND EXTRACT(EPOCH FROM (a.admittime - p.dob))/86400/365.242 <= {max_age}"
        
    if min_los is not None:
        query += f" AND i.los >= {min_los}"
    
    # Order by subject_id, intime
    query += " ORDER BY i.subject_id, i.intime"
    
    # Add limit if specified
    if limit is not None:
        query += f" LIMIT {limit}"
    
    # Execute query
    try:
        cohort_df = pd.read_sql_query(query, conn)
        print(f"Retrieved {len(cohort_df)} ICU stays for mortality prediction cohort.")
        return cohort_df
    except Exception as e:
        print(f"Error retrieving mortality prediction cohort: {e}")
        return None


# Function to get a cohort of patients for length of stay prediction
def get_los_prediction_cohort(conn, min_age=18, max_age=None, min_los=1, 
                            include_readmissions=False, limit=None):
    """
    Get a cohort for length of stay prediction.
    
    Parameters:
    -----------
    conn : psycopg2.connection
        Database connection object
    min_age : int, optional
        Minimum patient age in years, default 18
    max_age : int, optional
        Maximum patient age in years, default None (no upper limit)
    min_los : float, optional
        Minimum length of stay in days, default 1
    include_readmissions : bool, optional
        Whether to include readmissions, default False
    limit : int, optional
        Limit the number of patients returned, default None (no limit)
    
    Returns:
    --------
    pandas.DataFrame
        Dataframe containing patient cohort information for LOS prediction
    """
    # Create los buckets for classification
    query_base = """
    SELECT 
        i.subject_id, i.hadm_id, i.icustay_id, 
        p.gender, p.dob, a.admittime, a.dischtime, 
        i.intime, i.outtime, i.los,
        EXTRACT(EPOCH FROM (a.dischtime - a.admittime))/86400 as hospital_los,
        EXTRACT(EPOCH FROM (a.admittime - p.dob))/86400/365.242 as age,
        a.hospital_expire_flag,
        CASE 
            WHEN i.los < 2 THEN '0-2'
            WHEN i.los < 3 THEN '2-3'
            WHEN i.los < 5 THEN '3-5'
            WHEN i.los < 7 THEN '5-7'
            WHEN i.los < 14 THEN '7-14'
            ELSE '14+'
        END as los_bucket
    """
    
    if include_readmissions:
        query = f"""
        {query_base}
        FROM icustays i
        INNER JOIN admissions a ON i.hadm_id = a.hadm_id
        INNER JOIN patients p ON i.subject_id = p.subject_id
        WHERE 1=1
        """
    else:
        # Only include first ICU stay for each patient
        query = f"""
        WITH first_icu AS (
            SELECT subject_id, MIN(intime) as first_intime
            FROM icustays
            GROUP BY subject_id
        )
        {query_base}
        FROM icustays i
        INNER JOIN admissions a ON i.hadm_id = a.hadm_id
        INNER JOIN patients p ON i.subject_id = p.subject_id
        INNER JOIN first_icu f ON i.subject_id = f.subject_id AND i.intime = f.first_intime
        WHERE 1=1
        """
    
    # Add filters
    if min_age is not None:
        query += f" AND EXTRACT(EPOCH FROM (a.admittime - p.dob))/86400/365.242 >= {min_age}"
        
    if max_age is not None:
        query += f" AND EXTRACT(EPOCH FROM (a.admittime - p.dob))/86400/365.242 <= {max_age}"
        
    if min_los is not None:
        query += f" AND i.los >= {min_los}"
    
    # Order by subject_id, intime
    query += " ORDER BY i.subject_id, i.intime"
    
    # Add limit if specified
    if limit is not None:
        query += f" LIMIT {limit}"
    
    # Execute query
    try:
        cohort_df = pd.read_sql_query(query, conn)
        print(f"Retrieved {len(cohort_df)} ICU stays for LOS prediction cohort.")
        return cohort_df
    except Exception as e:
        print(f"Error retrieving LOS prediction cohort: {e}")
        return None


# Function to get a cohort of patients with specific diagnoses
def get_diagnosis_cohort(conn, icd9_codes, min_age=18, max_age=None, min_los=1, limit=None):
    """
    Get a cohort of patients with specific ICD-9 diagnoses.
    
    Parameters:
    -----------
    conn : psycopg2.connection
        Database connection object
    icd9_codes : list
        List of ICD-9 codes or code patterns (with %)
    min_age : int, optional
        Minimum patient age in years, default 18
    max_age : int, optional
        Maximum patient age in years, default None (no upper limit)
    min_los : float, optional
        Minimum length of stay in days, default 1
    limit : int, optional
        Limit the number of patients returned, default None (no limit)
    
    Returns:
    --------
    pandas.DataFrame
        Dataframe containing patient cohort information with the specified diagnoses
    """
    # Create SQL condition for ICD-9 codes
    icd9_conditions = []
    for code in icd9_codes:
        if '%' in code:
            icd9_conditions.append(f"icd9_code LIKE '{code}'")
        else:
            icd9_conditions.append(f"icd9_code = '{code}'")
    
    icd9_condition_str = " OR ".join(icd9_conditions)
    
    query = f"""
    WITH diagnosis_admissions AS (
        SELECT DISTINCT hadm_id
        FROM diagnoses_icd
        WHERE {icd9_condition_str}
    )
    SELECT i.subject_id, i.hadm_id, i.icustay_id, 
           p.gender, p.dob, a.admittime, a.dischtime, 
           i.intime, i.outtime, i.los,
           EXTRACT(EPOCH FROM (a.dischtime - a.admittime))/86400 as hospital_los,
           EXTRACT(EPOCH FROM (a.admittime - p.dob))/86400/365.242 as age,
           a.hospital_expire_flag
    FROM icustays i
    INNER JOIN admissions a ON i.hadm_id = a.hadm_id
    INNER JOIN patients p ON i.subject_id = p.subject_id
    INNER JOIN diagnosis_admissions d ON i.hadm_id = d.hadm_id
    WHERE 1=1
    """
    
    # Add filters
    if min_age is not None:
        query += f" AND EXTRACT(EPOCH FROM (a.admittime - p.dob))/86400/365.242 >= {min_age}"
        
    if max_age is not None:
        query += f" AND EXTRACT(EPOCH FROM (a.admittime - p.dob))/86400/365.242 <= {max_age}"
        
    if min_los is not None:
        query += f" AND i.los >= {min_los}"
    
    # Order by subject_id, intime
    query += " ORDER BY i.subject_id, i.intime"
    
    # Add limit if specified
    if limit is not None:
        query += f" LIMIT {limit}"
    
    # Execute query
    try:
        cohort_df = pd.read_sql_query(query, conn)
        print(f"Retrieved {len(cohort_df)} ICU stays with specified diagnoses.")
        return cohort_df
    except Exception as e:
        print(f"Error retrieving diagnosis cohort: {e}")
        return None


# Function to save a cohort to file
def save_cohort(cohort_df, output_file):
    """
    Save a cohort dataframe to a file.
    
    Parameters:
    -----------
    cohort_df : pandas.DataFrame
        The cohort dataframe to save
    output_file : str
        Path to save the cohort file (supports .pkl, .csv)
    
    Returns:
    --------
    bool
        True if successful, False otherwise
    """
    try:
        # Create directory if it doesn't exist
        os.makedirs(os.path.dirname(os.path.abspath(output_file)), exist_ok=True)
        
        # Save based on file extension
        if output_file.endswith('.pkl'):
            with open(output_file, 'wb') as f:
                pickle.dump(cohort_df, f)
        elif output_file.endswith('.csv'):
            cohort_df.to_csv(output_file, index=False)
        else:
            print(f"Unsupported file format: {output_file}")
            return False
        
        print(f"Saved cohort to {output_file}")
        return True
    except Exception as e:
        print(f"Error saving cohort: {e}")
        return False


# Function to load a cohort from file
def load_cohort(input_file):
    """
    Load a cohort dataframe from a file.
    
    Parameters:
    -----------
    input_file : str
        Path to the cohort file to load (supports .pkl, .csv)
    
    Returns:
    --------
    pandas.DataFrame
        The loaded cohort dataframe, or None if loading failed
    """
    try:
        if input_file.endswith('.pkl'):
            with open(input_file, 'rb') as f:
                cohort_df = pickle.load(f)
        elif input_file.endswith('.csv'):
            cohort_df = pd.read_csv(input_file)
        else:
            print(f"Unsupported file format: {input_file}")
            return None
        
        print(f"Loaded cohort from {input_file} with {len(cohort_df)} records")
        return cohort_df
    except Exception as e:
        print(f"Error loading cohort: {e}")
        return None


if __name__ == "__main__":
    # This script can be run standalone to create cohorts
    from mimic_db_connect import connect
    
    # Connect to the database
    conn = connect()
    
    if conn is None:
        print("Failed to connect to the database. Exiting.")
        sys.exit(1)
    
    # Create example cohorts
    print("\n1. Creating a basic ICU cohort...")
    basic_cohort = get_basic_icu_cohort(conn, min_age=18, min_los=1, limit=100)
    if basic_cohort is not None:
        save_cohort(basic_cohort, 'data/cohorts/basic_icu_cohort.pkl')
    
    print("\n2. Creating a first ICU stays cohort...")
    first_icu_cohort = get_first_icu_stays(conn, min_age=18, min_los=1, limit=100)
    if first_icu_cohort is not None:
        save_cohort(first_icu_cohort, 'data/cohorts/first_icu_cohort.pkl')
    
    print("\n3. Creating a sepsis cohort...")
    sepsis_cohort = get_sepsis_cohort(conn, min_age=18, min_los=1, limit=100)
    if sepsis_cohort is not None:
        save_cohort(sepsis_cohort, 'data/cohorts/sepsis_cohort.pkl')
    
    print("\n4. Creating a mortality prediction cohort...")
    mortality_cohort = get_mortality_cohort(conn, min_age=18, min_los=1, limit=100)
    if mortality_cohort is not None:
        save_cohort(mortality_cohort, 'data/cohorts/mortality_cohort.pkl')
    
    # Close the database connection
    conn.close()
    print("\nDatabase connection closed.") 