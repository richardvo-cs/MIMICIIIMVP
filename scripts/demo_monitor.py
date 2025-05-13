#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
MIMIC-III Demo Monitoring System
Simulates real-time patient monitoring with randomized data.
Uses trained ML models for predictions.
"""

import os
import sys
import time
import random
import logging
import requests
import pandas as pd
import numpy as np
import json
import tempfile
import traceback
from datetime import datetime, timedelta
from pathlib import Path
import threading

# Add src directory to path for imports
sys.path.append(os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'src'))

# Try to import database connector
try:
    from utils.db_connector import get_db_cursor, test_connection
    HAS_DB_CONNECTION = test_connection()
except ImportError:
    HAS_DB_CONNECTION = False

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("monitor_logs.txt"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# API endpoints
BASE_URL = "http://localhost:8000"  # API server URL
PREDICT_ENDPOINT = f"{BASE_URL}/api/predict"

# Data directory
DATA_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'data/results')

# Risk thresholds
HIGH_RISK_THRESHOLD = 0.7  # Threshold for high risk
ALERT_THRESHOLD = 0.9      # Threshold for alerting

def load_thresholds(config_path="config/thresholds.json"):
    """Load prediction thresholds from config file."""
    try:
        with open(config_path, 'r') as f:
            thresholds = json.load(f)
        return thresholds
    except Exception as e:
        logger.error(f"Error loading thresholds from {config_path}: {e}")
        # Return default thresholds if config cannot be loaded
        return {
            "mortality": {"prediction_threshold": 0.5},
            "sepsis": {"prediction_threshold": 0.5},
            "readmission": {"prediction_threshold": 0.5}
        }

class PatientMonitor:
    def __init__(self, scenario, api_url, interval_seconds, max_intervals):
        self.scenario = scenario
        self.api_url = api_url
        self.interval_seconds = interval_seconds
        self.max_intervals = max_intervals
        self.history = []
        self.high_risk_count = 0
        self.high_risk_consecutive = 0
        self.last_predictions = None
        self.time_series_data = {
            'patient_id': str(scenario),
            'age': None,
            'gender': None,
            'timestamps': [],
            'heart_rate': [],
            'respiratory_rate': [],
            'temperature': [],
            'systolic_bp': [],
            'oxygen_saturation': [],
            'wbc': [],
            'lactate': [],
            'creatinine': [],
            'mortality_risk': []  # Changed from 'mortality' to 'mortality_risk'
        }
        # Initialize heart rate pattern tracking
        self.pattern_time = 0
        self.in_spike = False
        self.spike_duration = 0
        self.normal_duration = 0
        
        # Load thresholds from config
        self.thresholds = load_thresholds()
        
        # Load or create patient scenario
        if scenario:
            self.scenario = scenario
        else:
            self.scenario = self.generate_initial_data()
        
        # Create data directory if it doesn't exist
        os.makedirs(DATA_DIR, exist_ok=True)
        
    def generate_initial_data(self):
        """Generate or retrieve initial patient data."""
        # If a specific scenario is selected, use that data
        if self.scenario:
            return self.scenario
        
        # Generate random but clinically plausible initial values
        # Could be extended to sample from actual MIMIC database for more realism
        gender = random.choice(['M', 'F'])
        age = random.randint(18, 95) if gender == 'M' else random.randint(18, 98)
        
        # Generate vitals based on age
        if age > 70:
            # Elderly patients might have different baseline vitals
            heart_rate = random.randint(60, 90)
            respiratory_rate = random.randint(14, 22)
            temperature = round(random.uniform(36.2, 37.8), 1)
            systolic_bp = random.randint(110, 160)
            oxygen_saturation = random.randint(90, 98)
        else:
            heart_rate = random.randint(55, 85)
            respiratory_rate = random.randint(12, 20)
            temperature = round(random.uniform(36.5, 37.5), 1)
            systolic_bp = random.randint(100, 140)
            oxygen_saturation = random.randint(95, 100)
        
        # Lab values
        wbc = round(random.uniform(4.0, 11.0), 1)  # white blood cell count
        lactate = round(random.uniform(0.5, 2.0), 1)
        creatinine = round(random.uniform(0.6, 1.3), 1)
        
        # Patient identifiers
        # Use a real patient ID from the MIMIC database if available
        try:
            from sqlalchemy import create_engine, text
            import configparser
            
            # Read database config
            config = configparser.ConfigParser()
            if os.path.exists('database.ini'):
                config.read('database.ini')
                db_config = {
                    'host': config['postgresql'].get('host', 'localhost'),
                    'database': config['postgresql'].get('database', 'mimiciii'),
                    'user': config['postgresql'].get('user', 'mimicuser'),
                    'password': config['postgresql'].get('password', 'password'),
                    'port': config['postgresql'].get('port', '5432')
                }
                
                # Connect to database
                connection_string = f"postgresql://{db_config['user']}:{db_config['password']}@{db_config['host']}:{db_config['port']}/{db_config['database']}"
                engine = create_engine(connection_string)
                
                # Query for a random patient ID
                with engine.connect() as conn:
                    result = conn.execute(text(f"SELECT subject_id, gender FROM patients WHERE gender = '{gender}' ORDER BY RANDOM() LIMIT 1"))
                    row = result.fetchone()
                    if row:
                        patient_id = row[0]
                        gender = row[1]
                        logger.info(f"Using real patient ID from MIMIC-III: {patient_id}")
                    else:
                        patient_id = random.randint(10000, 99999)
            else:
                patient_id = random.randint(10000, 99999)
        except Exception as e:
            logger.warning(f"Error connecting to database to get real patient ID: {e}")
            patient_id = random.randint(10000, 99999)
        
        # Return the initial data
        return {
            'patient_id': patient_id,
            'gender': gender,
            'age': age,
            'heart_rate': heart_rate,
            'respiratory_rate': respiratory_rate,
            'temperature': temperature,
            'systolic_bp': systolic_bp,
            'oxygen_saturation': oxygen_saturation,
            'wbc': wbc,
            'lactate': lactate,
            'creatinine': creatinine
        }

    def get_random_patient_from_db(self):
        """Retrieve a random patient from the MIMIC-III database with their admission data."""
        
        with get_db_cursor() as cur:
            # Get a random patient with their first ICU stay and basic demographics
            cur.execute("""
                WITH random_patient AS (
                    SELECT p.subject_id, p.gender, 
                           EXTRACT(EPOCH FROM (a.admittime - p.dob))/86400/365.242 as age,
                           a.hadm_id, i.icustay_id, a.admittime,
                           ROW_NUMBER() OVER (ORDER BY RANDOM()) as rn
                    FROM patients p
                    JOIN admissions a ON p.subject_id = a.subject_id
                    JOIN icustays i ON a.hadm_id = i.hadm_id
                    WHERE EXTRACT(EPOCH FROM (a.admittime - p.dob))/86400/365.242 BETWEEN 18 AND 89
                    AND i.los >= 1
                    LIMIT 100
                )
                SELECT * FROM random_patient WHERE rn = 1
            """)
            
            patient_row = cur.fetchone()
            
            if not patient_row:
                logger.warning("No patient found in database")
                return None
            
            # Get the most recent vitals for this patient's ICU stay
            cur.execute("""
                WITH vital_items AS (
                    -- Map of itemids to vital signs
                    SELECT 
                        CASE 
                            WHEN itemid IN (211, 220045) THEN 'heart_rate'
                            WHEN itemid IN (615, 618, 220210) THEN 'respiratory_rate'
                            WHEN itemid IN (676, 223762) THEN 'temperature'
                            WHEN itemid IN (51, 442, 455, 6701, 220179, 220050) THEN 'systolic_bp'
                            WHEN itemid IN (646, 220277) THEN 'oxygen_saturation'
                        END AS vital_name,
                        itemid
                    FROM d_items
                    WHERE itemid IN 
                        -- Heart rate
                        (211, 220045, 
                        -- Respiratory rate
                        615, 618, 220210,
                        -- Temperature
                        676, 223762,
                        -- Systolic BP
                        51, 442, 455, 6701, 220179, 220050,
                        -- SpO2
                        646, 220277)
                )
                SELECT 
                    c.charttime,
                    v.vital_name,
                    c.valuenum
                FROM chartevents c
                JOIN vital_items v ON c.itemid = v.itemid
                WHERE c.icustay_id = %s
                AND c.valuenum IS NOT NULL
                AND c.error IS DISTINCT FROM 1
                ORDER BY c.charttime
                LIMIT 100
            """, (patient_row['icustay_id'],))
            
            vitals_data = cur.fetchall()
            
            # Get lab values
            cur.execute("""
                WITH lab_items AS (
                    -- Map of itemids to lab tests
                    SELECT 
                        CASE 
                            WHEN itemid IN (51300, 51301) THEN 'wbc'
                            WHEN itemid IN (50813) THEN 'lactate'
                            WHEN itemid IN (50912) THEN 'creatinine'
                        END AS lab_name,
                        itemid
                    FROM d_labitems
                    WHERE itemid IN 
                        -- WBC
                        (51300, 51301, 
                        -- Lactate
                        50813,
                        -- Creatinine
                        50912)
                )
                SELECT 
                    l.charttime,
                    lab.lab_name,
                    l.valuenum
                FROM labevents l
                JOIN lab_items lab ON l.itemid = lab.itemid
                WHERE l.hadm_id = %s
                AND l.valuenum IS NOT NULL
                ORDER BY l.charttime
                LIMIT 100
            """, (patient_row['hadm_id'],))
            
            lab_data = cur.fetchall()
            
            # Process vital signs - get most recent reading for each vital
            vitals = {}
            for row in vitals_data:
                vital_name = row['vital_name']
                value = row['valuenum']
                
                # Store most recent value
                if vital_name not in vitals or row['charttime'] > vitals[vital_name]['time']:
                    vitals[vital_name] = {'value': value, 'time': row['charttime']}
            
            # Process lab values - get most recent reading for each lab
            labs = {}
            for row in lab_data:
                lab_name = row['lab_name']
                value = row['valuenum']
                
                # Store most recent value
                if lab_name not in labs or row['charttime'] > labs[lab_name]['time']:
                    labs[lab_name] = {'value': value, 'time': row['charttime']}
            
            # Combine patient demographics with vitals and labs
            patient_data = {
                "patient_id": patient_row['subject_id'],
                "gender": patient_row['gender'],
                "age": int(patient_row['age']),
                "admission_time": patient_row['admittime'].strftime("%Y-%m-%d %H:%M:%S"),
            }
            
            # Add vitals with defaults if missing
            patient_data["heart_rate"] = vitals.get('heart_rate', {}).get('value', random.randint(60, 100))
            patient_data["respiratory_rate"] = vitals.get('respiratory_rate', {}).get('value', random.randint(12, 20))
            patient_data["temperature"] = vitals.get('temperature', {}).get('value', round(random.uniform(36.5, 37.5), 1))
            patient_data["systolic_bp"] = vitals.get('systolic_bp', {}).get('value', random.randint(100, 140))
            patient_data["oxygen_saturation"] = vitals.get('oxygen_saturation', {}).get('value', random.randint(95, 100))
            
            # Add labs with defaults if missing
            patient_data["wbc"] = labs.get('wbc', {}).get('value', round(random.uniform(4.5, 11.0), 1))
            patient_data["lactate"] = labs.get('lactate', {}).get('value', round(random.uniform(0.5, 2.0), 1))
            patient_data["creatinine"] = labs.get('creatinine', {}).get('value', round(random.uniform(0.6, 1.2), 2))
            
            return patient_data
    
    def generate_random_update(self, base_values):
        """Generate random updates to patient data with varying heart rate patterns"""
        update = base_values.copy()
        
        # Current values
        current_hr = update.get('heart_rate', 90)
        current_bp = update.get('systolic_bp', 120)
        current_wbc = update.get('wbc', 7.5)
        current_lactate = update.get('lactate', 1.5)
        current_temp = update.get('temperature', 37.0)
        current_spo2 = update.get('spo2', 98)
        current_resp = update.get('resp_rate', 16)
        current_creat = update.get('creatinine', 1.0)

        # Create a pattern of heart rate changes
        if self.in_spike:
            # During spike period (60 seconds)
            update['heart_rate'] = random.randint(140, 170)  # High heart rate
            self.spike_duration += 5
            
            # Correlated changes during high heart rate
            update['systolic_bp'] = random.randint(150, 180)  # High BP
            update['wbc'] = max(12, min(25, current_wbc + random.uniform(2, 4)))
            update['lactate'] = max(2.5, min(8, current_lactate + random.uniform(0.5, 1.5)))
            update['temperature'] = max(37.8, min(40.5, current_temp + random.uniform(0.2, 0.8)))
            update['spo2'] = max(85, min(94, current_spo2 - random.randint(2, 6)))
            update['resp_rate'] = max(20, min(40, current_resp + random.randint(2, 6)))
            
            if self.spike_duration >= 60:
                self.in_spike = False
                self.spike_duration = 0
                self.normal_duration = 0
        else:
            # Normal period (30 seconds)
            update['heart_rate'] = random.randint(70, 100)  # Normal heart rate
            update['systolic_bp'] = random.randint(110, 130)
            update['wbc'] = max(4, min(11, current_wbc + random.uniform(-0.5, 0.5)))
            update['lactate'] = max(0.5, min(2.0, current_lactate + random.uniform(-0.2, 0.2)))
            update['temperature'] = max(36.5, min(37.5, current_temp + random.uniform(-0.2, 0.2)))
            update['spo2'] = max(95, min(100, current_spo2 + random.randint(-1, 1)))
            update['resp_rate'] = max(12, min(20, current_resp + random.randint(-2, 2)))
            
            self.normal_duration += 5
            
            if self.normal_duration >= 30:
                self.in_spike = True
                self.normal_duration = 0
                self.spike_duration = 0
        
        # Random changes to creatinine (less volatile)
        update['creatinine'] = max(0.3, min(5, current_creat + random.uniform(-0.1, 0.1)))
        
        return update
    
    def save_time_series_data(self):
        """Save time series data in a format expected by the visualization"""
        data_file = "data/results/patient_time_series.json"
        os.makedirs(os.path.dirname(data_file), exist_ok=True)
        
        # Ensure all expected keys exist in time_series_data before saving
        expected_keys = ['patient_id', 'age', 'gender', 'timestamps', 'heart_rate', 
                        'respiratory_rate', 'temperature', 'systolic_bp', 
                        'oxygen_saturation', 'wbc', 'lactate', 'creatinine', 
                        'mortality_risk']
        for key in expected_keys:
            if key not in self.time_series_data:
                self.time_series_data[key] = [] # Initialize as empty list if missing
                logger.warning(f"Key '{key}' was missing from time_series_data, initialized.")
            # Ensure age/gender are not lists if they got appended mistakenly
            elif key in ['age', 'gender', 'patient_id'] and isinstance(self.time_series_data[key], list):
                 self.time_series_data[key] = self.scenario.get(key) 
        
        # Ensure all values are properly formatted for JSON (especially float values)
        if len(self.time_series_data['mortality_risk']) > 0:
            # Double-check mortality values before saving
            for i in range(len(self.time_series_data['mortality_risk'])):
                # Convert any non-float values to float
                try:
                    self.time_series_data['mortality_risk'][i] = float(self.time_series_data['mortality_risk'][i])
                except (ValueError, TypeError):
                    # If conversion fails, set a safe default
                    self.time_series_data['mortality_risk'][i] = 0.1
                    logger.warning(f"Fixed invalid mortality value at index {i}")
        
        # Log last values before saving
        if len(self.time_series_data['mortality_risk']) > 0:
            idx = -1  # Last index
            logger.info(f"Final values before save - HR: {self.time_series_data['heart_rate'][idx]}, " +
                        f"Mortality: {self.time_series_data['mortality_risk'][idx]:.2f}")

        with open(data_file, 'w') as f:
            json.dump(self.time_series_data, f)
        
        logger.info(f"Time series data saved to {data_file}")
    
    def make_prediction(self, current_values):
        """Make mortality risk predictions."""
        if self.api_url:
            try:
                import requests
                # Call the prediction API
                response = requests.post(
                    f"{self.api_url}/api/predict",
                    json=current_values,
                    timeout=5
                )
                if response.status_code == 200:
                    predictions = response.json()
                    logger.info(f"API Response: {predictions}")  # Log the full response
                    mortality_risk = predictions.get('predictions', {}).get('mortality', 0.1)
                    logger.info(f"Extracted mortality risk: {mortality_risk}")
                    return {'mortality': mortality_risk}
                else:
                    logger.error(f"API error: {response.status_code} - {response.text}")
            except Exception as e:
                logger.error(f"Error calling prediction API: {e}")
                logger.error(traceback.format_exc())
        
        # Fallback to local prediction if API fails or is not available
        # Simple rules-based prediction
        mortality_risk = 0.05  # baseline risk
        
        # Adjust mortality risk based on key factors
        if current_values.get('age', 0) > 80:
            mortality_risk += 0.2
        if current_values.get('temperature', 37) > 39.5 or current_values.get('temperature', 37) < 36:
            mortality_risk += 0.15
        if current_values.get('heart_rate', 80) > 120 or current_values.get('heart_rate', 80) < 50:
            mortality_risk += 0.15
        if current_values.get('systolic_bp', 120) < 90:
            mortality_risk += 0.2
        if current_values.get('oxygen_saturation', 98) < 92:
            mortality_risk += 0.2
        
        logger.info(f"Fallback prediction - Mortality risk: {mortality_risk}")
        return {'mortality': min(0.95, mortality_risk)}
    
    def check_high_risk(self, predictions):
        """Check if predictions indicate high risk using custom thresholds."""
        is_high_risk = False
        risk_status = {}
        
        # Define alert thresholds (higher than prediction thresholds)
        alert_thresholds = {
            'mortality': min(0.9, self.thresholds["mortality"]["prediction_threshold"] + 0.3)
        }
        
        # Define high risk thresholds (same as prediction thresholds)
        high_risk_thresholds = {
            'mortality': self.thresholds["mortality"]["prediction_threshold"]
        }
        
        for risk_type, probability in predictions.items():
            if probability is None: continue # Skip if prediction failed
            if risk_type in alert_thresholds:
                if probability >= alert_thresholds[risk_type]:
                    risk_status[risk_type] = "ALERT"
                    is_high_risk = True
                elif probability >= high_risk_thresholds[risk_type]:
                    risk_status[risk_type] = "HIGH"
                    is_high_risk = True
                else:
                    risk_status[risk_type] = "NORMAL"
        
        if is_high_risk:
            self.high_risk_consecutive += 1
        else:
            self.high_risk_consecutive = 0
        
        return is_high_risk, risk_status
    
    def monitor(self, interval_seconds=5, max_intervals=120): # Increase max_intervals for longer run
        """Monitor patient and make predictions"""
        current_data = self.scenario.copy()
        
        for interval in range(max_intervals):
            # Generate random update based on spike/normal cycle
            current_data = self.generate_random_update(current_data)
            timestamp = datetime.now().strftime("%H:%M:%S")
            # Add timestamp to data for potential API use (though API might not need it)
            current_data_for_api = current_data.copy()
            current_data_for_api['timestamp'] = timestamp
            
            # Update time series data dictionary
            self.time_series_data['timestamps'].append(timestamp)
            self.time_series_data['heart_rate'].append(current_data.get('heart_rate'))
            self.time_series_data['respiratory_rate'].append(current_data.get('resp_rate'))
            self.time_series_data['temperature'].append(current_data.get('temperature'))
            self.time_series_data['systolic_bp'].append(current_data.get('systolic_bp'))
            self.time_series_data['oxygen_saturation'].append(current_data.get('spo2'))
            self.time_series_data['wbc'].append(current_data.get('wbc'))
            self.time_series_data['lactate'].append(current_data.get('lactate'))
            self.time_series_data['creatinine'].append(current_data.get('creatinine'))
            
            # Make predictions using the API (or fallback to simulated)
            predictions = self.make_prediction(current_data_for_api)
            self.last_predictions = predictions
            
            # Update time series data with predictions
            mort_risk = predictions.get('mortality', 0.1)
            self.time_series_data['mortality_risk'].append(mort_risk)  # Changed from 'mortality' to 'mortality_risk'
            
            # Log the actual values being saved
            logger.info(f"Saved to time_series_data - HR: {current_data.get('heart_rate')}, " +
                        f"Mortality: {mort_risk:.2f}")
            
            # Check for high risk
            is_high_risk, risk_status = self.check_high_risk(predictions)
            
            # Keep only the last N data points (e.g., 60 points for 5 min history at 5s interval)
            max_history = 60 
            if len(self.time_series_data['timestamps']) > max_history:
                for key in self.time_series_data:
                    if isinstance(self.time_series_data[key], list):
                        self.time_series_data[key] = self.time_series_data[key][-max_history:]
            
            # Save to file for visualization
            self.save_time_series_data()
            
            # Log the results
            status_str = ", ".join([f"{k}={v}" for k, v in risk_status.items() if v != 'NORMAL'])
            if not status_str: status_str = "All Normal"
            logger.info(f"Interval {interval + 1}: " +
                       f"HR={current_data.get('heart_rate', 'N/A'):.0f}, " +
                       f"Temp={current_data.get('temperature', 'N/A'):.1f}, " +
                       f"BP={current_data.get('systolic_bp', 'N/A'):.0f}, " +
                       f"SPO2={current_data.get('spo2', 'N/A'):.0f}, " +
                       f"Mortality={predictions.get('mortality', -1):.2f}")
                       # f"Sepsis={predictions.get('sepsis', -1):.3f}, " +
                       # f"Readmission={predictions.get('readmission', -1):.3f}")
            
            # Alert if high risk for 2 consecutive intervals
            if self.high_risk_consecutive >= 2:
                logger.warning(f"ALERT: Patient {self.scenario.get('patient_id', 'Unknown')} high risk ({self.high_risk_consecutive} intervals)! Risks: {risk_status}")
                # Generate intervention plan only if needed
                # self.generate_intervention_plan(predictions, current_data)
            
            # Wait for next interval
            time.sleep(interval_seconds)
    
    # generate_intervention_plan can be kept or removed if not used
    def generate_intervention_plan(self, predictions, current_data):
         """Generate an intervention plan based on high-risk predictions"""
         # ... (Implementation can be added back if needed)
         pass 

    def start_monitoring(self):
        """Start monitoring patient vital signs and making predictions."""
        try:
            # Generate or retrieve initial patient data
            self.scenario = self.generate_initial_data()
            
            # Initialize time series data dictionary
            self.time_series_data = {
                'patient_id': str(self.scenario['patient_id']),
                'age': self.scenario['age'],
                'gender': self.scenario['gender'],
                'timestamps': [],
                'heart_rate': [],
                'respiratory_rate': [],
                'temperature': [],
                'systolic_bp': [],
                'oxygen_saturation': [],
                'wbc': [],
                'lactate': [],
                'creatinine': [],
                'mortality_risk': []  # Changed from 'mortality' to 'mortality_risk'
            }
            
            # Set current values from initial data
            self.current_data = self.scenario.copy()
            
            # Record current timestamp
            current_time = datetime.now()
            self.time_series_data['timestamps'].append(current_time.isoformat())
            
            # Record initial values
            for vital in ['heart_rate', 'respiratory_rate', 'temperature', 
                         'systolic_bp', 'oxygen_saturation', 'wbc', 
                         'lactate', 'creatinine']:
                self.time_series_data[vital].append(self.current_data[vital])
            
            # Get initial predictions
            predictions = self.make_prediction(self.current_data)
            self.time_series_data['mortality_risk'].append(predictions.get('mortality', 0.05))
            
            # Start continuous monitoring in a separate thread
            self.monitoring_thread = threading.Thread(target=self.monitor)
            self.monitoring_thread.daemon = True
            self.monitoring_thread.start()
            
        except Exception as e:
            logger.error(f"Error starting monitoring: {e}")
            logger.error(traceback.format_exc())
            raise

def create_sample_patient():
    """Create a sample patient with minimal initial data for monitoring."""
    # Only include keys that are used for initial state or display
    return {
        'subject_id': random.randint(1000, 9999),
        'age': random.randint(40, 80),
        'gender': random.choice([0, 1]),
        'heart_rate': random.randint(70, 100),  
        'resp_rate': random.randint(12, 20),
        'temperature': random.uniform(36.5, 37.5),
        'systolic_bp': random.randint(110, 130),
        'spo2': random.randint(95, 100),
        'wbc': random.uniform(5, 10),
        'lactate': random.uniform(0.8, 1.8),
        'creatinine': random.uniform(0.7, 1.2)
        # Other features will be added with defaults in generate_random_update
    }

def main():
    """Run the demo monitor"""
    try:
        # Initialize and setup patient monitor
        monitor = PatientMonitor(
            scenario=None,  # Use random scenarios
            api_url=BASE_URL,
            interval_seconds=5,  # 5 second intervals
            max_intervals=100    # Run for 100 intervals
        )
        
        logger.info(f"Starting monitoring for Patient {monitor.time_series_data.get('patient_id', 'Unknown')}...")
        # Monitor patient
        monitor.start_monitoring()
        
        # Wait for the monitoring thread to finish
        monitor.monitoring_thread.join()
        
    except KeyboardInterrupt:
        logger.info("Monitoring stopped by user.")
    except Exception as e:
        logger.error(f"Error in main monitoring function: {e}")
        logger.error(traceback.format_exc())

if __name__ == "__main__":
    main() 