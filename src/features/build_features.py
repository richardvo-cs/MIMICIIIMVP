#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
MIMIC-III Feature Engineering Script
This script builds features from the preprocessed MIMIC-III data.
"""

import os
import pandas as pd
import numpy as np
from tqdm import tqdm

def load_preprocessed_data(input_dir='../../data/processed'):
    """Load preprocessed data files."""
    patient_timeline = pd.read_csv(os.path.join(input_dir, 'patient_timeline.csv'))
    vitals = pd.read_csv(os.path.join(input_dir, 'vitals.csv'))
    labs = pd.read_csv(os.path.join(input_dir, 'labs.csv'))
    
    # Convert datetime columns
    datetime_cols = ['intime', 'outtime', 'admittime', 'dischtime', 'deathtime', 'dob', 'dod', 'charttime']
    for df in [patient_timeline, vitals, labs]:
        for col in datetime_cols:
            if col in df.columns:
                df[col] = pd.to_datetime(df[col])
    
    return patient_timeline, vitals, labs

def pivot_and_resample_vitals(vitals_df, time_window='1H'):
    """
    Pivot vitals data to wide format and resample to regular time intervals.
    
    Args:
        vitals_df: DataFrame containing vital sign measurements
        time_window: Resampling time window (default: 1 hour)
    
    Returns:
        DataFrame with pivoted and resampled vital signs
    """
    # Convert to datetime
    vitals_df['charttime'] = pd.to_datetime(vitals_df['charttime'])
    
    # Create a pivot table
    vital_pivot = vitals_df.pivot_table(
        index=['subject_id', 'hadm_id', 'icustay_id', 'charttime'],
        columns='vital_type',
        values='valuenum',
        aggfunc='mean'
    ).reset_index()
    
    # Group by patient stay
    grouped = vital_pivot.groupby(['subject_id', 'hadm_id', 'icustay_id'])
    
    # Resample for each patient stay
    resampled_dfs = []
    
    for (subject_id, hadm_id, icustay_id), group in tqdm(grouped, desc="Resampling vital signs"):
        # Set charttime as index
        group = group.set_index('charttime')
        
        # Sort by time
        group = group.sort_index()
        
        # Resample to regular time intervals
        resampled = group.resample(time_window).mean()
        
        # Forward fill missing values (carry forward last observation)
        resampled = resampled.ffill()
        
        # Add back the identifiers
        resampled['subject_id'] = subject_id
        resampled['hadm_id'] = hadm_id
        resampled['icustay_id'] = icustay_id
        
        resampled_dfs.append(resampled.reset_index())
    
    # Combine all resampled DataFrames
    resampled_vitals = pd.concat(resampled_dfs, ignore_index=True)
    
    return resampled_vitals

def pivot_and_resample_labs(labs_df, time_window='1H'):
    """
    Pivot lab data to wide format and resample to regular time intervals.
    
    Args:
        labs_df: DataFrame containing laboratory test results
        time_window: Resampling time window (default: 1 hour)
    
    Returns:
        DataFrame with pivoted and resampled lab results
    """
    # Convert to datetime
    labs_df['charttime'] = pd.to_datetime(labs_df['charttime'])
    
    # Create a pivot table
    lab_pivot = labs_df.pivot_table(
        index=['subject_id', 'hadm_id', 'charttime'],
        columns='lab_type',
        values='valuenum',
        aggfunc='mean'
    ).reset_index()
    
    # Group by patient stay
    grouped = lab_pivot.groupby(['subject_id', 'hadm_id'])
    
    # Resample for each patient stay
    resampled_dfs = []
    
    for (subject_id, hadm_id), group in tqdm(grouped, desc="Resampling lab results"):
        # Set charttime as index
        group = group.set_index('charttime')
        
        # Sort by time
        group = group.sort_index()
        
        # Resample to regular time intervals
        resampled = group.resample(time_window).mean()
        
        # Forward fill missing values (carry forward last observation)
        resampled = resampled.ffill()
        
        # Add back the identifiers
        resampled['subject_id'] = subject_id
        resampled['hadm_id'] = hadm_id
        
        resampled_dfs.append(resampled.reset_index())
    
    # Combine all resampled DataFrames
    resampled_labs = pd.concat(resampled_dfs, ignore_index=True)
    
    return resampled_labs

def calculate_derived_metrics(vitals_df, labs_df):
    """
    Calculate derived clinical metrics such as MEWS score components.
    
    Modified Early Warning Score (MEWS) components:
    - Systolic BP: <70 (3), 70-80 (2), 81-100 (1), 101-199 (0), ≥200 (2)
    - Heart rate: <40 (2), 40-50 (1), 51-100 (0), 101-110 (1), 111-129 (2), ≥130 (3)
    - Respiratory rate: <9 (2), 9-14 (0), 15-20 (1), 21-29 (2), ≥30 (3)
    - Temperature: <35.0 (2), 35.0-38.4 (0), ≥38.5 (2)
    
    Returns:
        DataFrame with derived metrics added
    """
    # Create a copy of the vitals DataFrame
    vitals_with_metrics = vitals_df.copy()
    
    # Calculate MEWS components
    
    # Systolic BP score
    vitals_with_metrics['sbp_score'] = 0
    vitals_with_metrics.loc[vitals_with_metrics['sbp'] < 70, 'sbp_score'] = 3
    vitals_with_metrics.loc[(vitals_with_metrics['sbp'] >= 70) & (vitals_with_metrics['sbp'] <= 80), 'sbp_score'] = 2
    vitals_with_metrics.loc[(vitals_with_metrics['sbp'] >= 81) & (vitals_with_metrics['sbp'] <= 100), 'sbp_score'] = 1
    vitals_with_metrics.loc[vitals_with_metrics['sbp'] >= 200, 'sbp_score'] = 2
    
    # Heart rate score
    vitals_with_metrics['hr_score'] = 0
    vitals_with_metrics.loc[vitals_with_metrics['heart_rate'] < 40, 'hr_score'] = 2
    vitals_with_metrics.loc[(vitals_with_metrics['heart_rate'] >= 40) & (vitals_with_metrics['heart_rate'] <= 50), 'hr_score'] = 1
    vitals_with_metrics.loc[(vitals_with_metrics['heart_rate'] >= 101) & (vitals_with_metrics['heart_rate'] <= 110), 'hr_score'] = 1
    vitals_with_metrics.loc[(vitals_with_metrics['heart_rate'] >= 111) & (vitals_with_metrics['heart_rate'] <= 129), 'hr_score'] = 2
    vitals_with_metrics.loc[vitals_with_metrics['heart_rate'] >= 130, 'hr_score'] = 3
    
    # Respiratory rate score
    vitals_with_metrics['resp_score'] = 0
    vitals_with_metrics.loc[vitals_with_metrics['resp_rate'] < 9, 'resp_score'] = 2
    vitals_with_metrics.loc[(vitals_with_metrics['resp_rate'] >= 15) & (vitals_with_metrics['resp_rate'] <= 20), 'resp_score'] = 1
    vitals_with_metrics.loc[(vitals_with_metrics['resp_rate'] >= 21) & (vitals_with_metrics['resp_rate'] <= 29), 'resp_score'] = 2
    vitals_with_metrics.loc[vitals_with_metrics['resp_rate'] >= 30, 'resp_score'] = 3
    
    # Temperature score
    vitals_with_metrics['temp_score'] = 0
    vitals_with_metrics.loc[vitals_with_metrics['temp'] < 35.0, 'temp_score'] = 2
    vitals_with_metrics.loc[vitals_with_metrics['temp'] >= 38.5, 'temp_score'] = 2
    
    # Calculate total MEWS score
    score_columns = ['sbp_score', 'hr_score', 'resp_score', 'temp_score']
    vitals_with_metrics['mews_score'] = vitals_with_metrics[score_columns].sum(axis=1)
    
    # You can add more derived metrics here as needed
    
    return vitals_with_metrics

def create_rolling_features(df, feature_cols, windows=[1, 6, 12, 24]):
    """
    Create rolling window features (mean, min, max, std) for specified columns.
    
    Args:
        df: DataFrame with time-series data
        feature_cols: List of columns to create rolling features for
        windows: List of window sizes in hours
        
    Returns:
        DataFrame with rolling features added
    """
    # Make a copy of the input DataFrame
    result_df = df.copy()
    
    # Ensure the DataFrame is sorted by time for each patient
    result_df = result_df.sort_values(['subject_id', 'hadm_id', 'icustay_id', 'charttime'])
    
    # Group by patient stay
    grouped = result_df.groupby(['subject_id', 'hadm_id', 'icustay_id'])
    
    # Create a list to store DataFrames with rolling features
    dfs_with_rolling = []
    
    # Process each patient stay
    for name, group in tqdm(grouped, desc="Creating rolling features"):
        # Set index to charttime for rolling operations
        group = group.set_index('charttime')
        
        # Create rolling features for each window and feature
        for window in windows:
            for col in feature_cols:
                if col in group.columns:
                    # Define the rolling window in hours
                    rolling_window = f"{window}H"
                    
                    # Calculate rolling statistics
                    group[f"{col}_mean_{window}h"] = group[col].rolling(rolling_window).mean()
                    group[f"{col}_min_{window}h"] = group[col].rolling(rolling_window).min()
                    group[f"{col}_max_{window}h"] = group[col].rolling(rolling_window).max()
                    group[f"{col}_std_{window}h"] = group[col].rolling(rolling_window).std()
        
        # Reset index to get charttime back as a column
        group = group.reset_index()
        dfs_with_rolling.append(group)
    
    # Combine all DataFrames
    result_df = pd.concat(dfs_with_rolling, ignore_index=True)
    
    return result_df

def create_time_until_event(patient_df, event_time_col, current_time_col='charttime'):
    """
    Calculate time until a specific event (e.g., death, discharge).
    
    Args:
        patient_df: DataFrame with patient data
        event_time_col: Column name containing the event timestamp
        current_time_col: Column name containing the current timestamp
        
    Returns:
        Series with time until event in hours
    """
    # Calculate time difference in hours
    time_until_event = (patient_df[event_time_col] - patient_df[current_time_col]).dt.total_seconds() / 3600
    
    return time_until_event

def define_outcome_labels(patient_df, vitals_df, time_window=24):
    """
    Define outcome labels for prediction.
    
    Args:
        patient_df: DataFrame with patient data
        vitals_df: DataFrame with vital signs
        time_window: Time window in hours for prediction (default: 24)
        
    Returns:
        DataFrame with outcome labels
    """
    # Merge patient data with vitals
    merged_df = pd.merge(
        vitals_df,
        patient_df[['subject_id', 'hadm_id', 'icustay_id', 'deathtime', 'dischtime']],
        on=['subject_id', 'hadm_id', 'icustay_id'],
        how='left'
    )
    
    # Convert timestamp columns to datetime
    for col in ['charttime', 'deathtime', 'dischtime']:
        if col in merged_df.columns:
            merged_df[col] = pd.to_datetime(merged_df[col])
    
    # Calculate time until death
    merged_df['time_until_death'] = np.nan
    death_mask = ~merged_df['deathtime'].isna()
    merged_df.loc[death_mask, 'time_until_death'] = create_time_until_event(
        merged_df[death_mask], 'deathtime', 'charttime'
    )
    
    # Calculate time until discharge
    merged_df['time_until_discharge'] = np.nan
    discharge_mask = ~merged_df['dischtime'].isna()
    merged_df.loc[discharge_mask, 'time_until_discharge'] = create_time_until_event(
        merged_df[discharge_mask], 'dischtime', 'charttime'
    )
    
    # Define mortality outcome
    # Label = 1 if death within the next `time_window` hours
    merged_df['mortality_outcome'] = 0
    merged_df.loc[
        (merged_df['time_until_death'] <= time_window) & 
        (merged_df['time_until_death'] > 0),
        'mortality_outcome'
    ] = 1
    
    # You can define additional outcomes here
    
    return merged_df

def build_feature_matrix(output_dir='../../data/features'):
    """
    Build the complete feature matrix for modeling.
    
    Args:
        output_dir: Directory to save the feature matrix
        
    Returns:
        DataFrame with features and labels for modeling
    """
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    # Load preprocessed data
    patient_timeline, vitals, labs = load_preprocessed_data()
    
    # Resample vitals and labs to regular time intervals
    print("Resampling vital signs and lab results...")
    resampled_vitals = pivot_and_resample_vitals(vitals)
    resampled_labs = pivot_and_resample_labs(labs)
    
    # Calculate derived metrics
    print("Calculating derived metrics...")
    vitals_with_metrics = calculate_derived_metrics(resampled_vitals, resampled_labs)
    
    # Define vital sign columns for creating rolling features
    vital_cols = ['heart_rate', 'sbp', 'dbp', 'resp_rate', 'temp', 'spo2', 'mews_score']
    
    # Create rolling features for vital signs
    print("Creating rolling features...")
    vitals_with_rolling = create_rolling_features(vitals_with_metrics, vital_cols)
    
    # Define outcome labels
    print("Defining outcome labels...")
    labeled_data = define_outcome_labels(patient_timeline, vitals_with_rolling)
    
    # Merge with lab results
    print("Merging with lab results...")
    resampled_labs = resampled_labs.drop(columns=['subject_id', 'hadm_id'], errors='ignore')
    feature_matrix = pd.merge(
        labeled_data,
        resampled_labs,
        left_on=['subject_id', 'hadm_id', 'charttime'],
        right_on=['subject_id', 'hadm_id', 'charttime'],
        how='left'
    )
    
    # Add demographic features
    print("Adding demographic features...")
    demographics = patient_timeline[['subject_id', 'hadm_id', 'icustay_id', 'age', 'gender', 'ethnicity']]
    feature_matrix = pd.merge(
        feature_matrix,
        demographics,
        on=['subject_id', 'hadm_id', 'icustay_id'],
        how='left'
    )
    
    # Fill missing values
    print("Handling missing values...")
    # For lab results, use forward fill first (carrying forward last known value)
    feature_matrix = feature_matrix.groupby(['subject_id', 'hadm_id', 'icustay_id']).ffill()
    
    # For remaining NaNs, use median imputation
    numeric_cols = feature_matrix.select_dtypes(include=['float64', 'int64']).columns
    for col in numeric_cols:
        if feature_matrix[col].isnull().sum() > 0:
            median_value = feature_matrix[col].median()
            feature_matrix[col].fillna(median_value, inplace=True)
    
    # Save the feature matrix
    print("Saving feature matrix...")
    feature_matrix.to_csv(os.path.join(output_dir, 'feature_matrix.csv'), index=False)
    
    return feature_matrix

if __name__ == "__main__":
    # Build feature matrix
    feature_matrix = build_feature_matrix()
    
    print(f"Created feature matrix with {feature_matrix.shape[0]} rows and {feature_matrix.shape[1]} columns")
    print(f"Number of positive mortality outcomes: {feature_matrix['mortality_outcome'].sum()}") 