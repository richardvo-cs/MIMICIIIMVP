#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
MIMIC-III Clinical Prediction API
Provides a Flask API for making clinical predictions using trained models.
"""

import os
import sys
import json
import time
import logging
import tempfile
import pandas as pd
import numpy as np
from pathlib import Path
from datetime import datetime, timedelta
from werkzeug.utils import secure_filename
from flask import Flask, request, jsonify, make_response, send_from_directory
from flask_cors import CORS
import pickle
import traceback
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from joblib import load

# Configure logging
logging.basicConfig(
    level=logging.WARNING,  # Change to WARNING to reduce noise
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Initialize Flask app
app = Flask(__name__)
CORS(app)

# --- Configuration ---
# Determine base directory relative to this script's location
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__))) 
app.config['UPLOAD_FOLDER'] = os.path.join(BASE_DIR, 'data/uploads')
app.config['MODELS_DIR'] = os.path.join(BASE_DIR, 'models')
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # 16MB max upload
app.config['ALLOWED_EXTENSIONS'] = {'csv', 'pkl', 'json'} # Allow csv, pkl, and json
# Set debug to False for stability, can be True for development
app.config['DEBUG'] = False 

# Ensure upload folder exists
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)

def allowed_file(filename):
    """Check if a filename has an allowed extension."""
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in app.config['ALLOWED_EXTENSIONS']

def load_patient_data(file_path):
    """Load patient data from CSV or PKL file."""
    try:
        if file_path.endswith('.csv'):
            df = pd.read_csv(file_path)
        elif file_path.endswith('.pkl'):
            df = pd.read_pickle(file_path)
        else:
            return None
        logger.info(f"Successfully loaded data from {file_path}, shape: {df.shape}")
        return df
    except Exception as e:
        logger.error(f"Error loading data from {file_path}: {e}")
        return None

def get_available_models():
    """ Scan the models directory and return info about valid models found. """
    models = []
    models_dir = app.config['MODELS_DIR']
    if not os.path.isdir(models_dir):
        logger.error(f"Models directory not found: {models_dir}")
        return []
        
    for item in os.listdir(models_dir):
        model_subdir = os.path.join(models_dir, item)
        if os.path.isdir(model_subdir):
            model_path = os.path.join(model_subdir, 'model.pkl')
            feature_path = os.path.join(model_subdir, 'feature_list.pkl')
            # Check if essential files exist
            if os.path.exists(model_path) and os.path.exists(feature_path):
                 models.append({'name': item, 'path': model_subdir})
                 logger.info(f"Found valid model: {item}")
            else:
                 logger.warning(f"Skipping directory {item}: missing model.pkl or feature_list.pkl")
    return models

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

class PredictionModel:
    """Base class for prediction models."""
    
    def __init__(self, model_path, features_path=None):
        self.model = None
        self.features = None
        self.model_path = model_path
        self.features_path = features_path
        self.load_model()
    
    def load_model(self):
        """Load the trained model and feature list."""
        try:
            # Try loading with joblib first
            self.model = load(self.model_path)
            logger.info(f"Loaded model from {self.model_path} using joblib")
            
            # Load feature list if available
            if self.features_path and os.path.exists(self.features_path):
                with open(self.features_path, 'rb') as f:
                    import pickle
                    self.features = pickle.load(f)
                logger.info(f"Loaded feature list from {self.features_path}")
            
        except Exception as e:
            logger.error(f"Error loading model: {e}")
            logger.warning("Using dummy model as fallback")
            # Create a dummy model for fallback
            self.model = self._create_dummy_model()
    
    def _create_dummy_model(self):
        """Create a dummy model if trained model can't be loaded."""
        # This is a fallback method - it should be overridden by subclasses
        from sklearn.ensemble import RandomForestClassifier
        return RandomForestClassifier(n_estimators=10, random_state=42)
    
    def predict(self, data):
        """Make predictions with the model."""
        # This should be implemented by subclasses
        raise NotImplementedError("Subclasses must implement predict method")

class MortalityModel(PredictionModel):
    """Model for predicting mortality risk."""
    
    def __init__(self):
        model_path = "models/mortality/random_forest_model.joblib"
        features_path = "models/mortality/feature_list.pkl"
        self.thresholds = load_thresholds()
        self.prediction_threshold = self.thresholds["mortality"]["prediction_threshold"]
        super().__init__(model_path, features_path)
    
    def _create_dummy_model(self):
        """Create a dummy mortality prediction model."""
        from sklearn.ensemble import RandomForestClassifier
        return RandomForestClassifier(n_estimators=10, random_state=42)
    
    def predict(self, data):
        """Predict mortality risk."""
        # Basic Approach
        if self.features is None or isinstance(self.model, RandomForestClassifier):
            # Simple rule-based prediction for dummy model
            risk = 0.05  # baseline risk
            
            # Increase risk based on key factors
            if data.get('age', 0) > 80:
                risk += 0.3
            if data.get('temperature', 98.6) > 102:
                risk += 0.2
            if data.get('heart_rate', 80) > 120 or data.get('heart_rate', 80) < 50:
                risk += 0.2
            if data.get('systolic_bp', 120) < 90:
                risk += 0.3
            if data.get('oxygen_saturation', 98) < 90:
                risk += 0.3
            
            return min(risk, 0.95)  # cap at 0.95
        
        # Format data correctly
        try:
            # Prepare features in the correct order
            features = {}
            for feature in self.features:
                # Map input data to model features
                if feature in data:
                    features[feature] = data[feature]
                else:
                    # Handle missing features with reasonable defaults
                    if 'heart_rate' in feature:
                        features[feature] = data.get('heart_rate', 80)
                    elif 'temperature' in feature:
                        features[feature] = data.get('temperature', 37)
                    elif 'sbp' in feature or 'systolic' in feature:
                        features[feature] = data.get('systolic_bp', 120)
                    elif 'spo2' in feature or 'oxygen' in feature:
                        features[feature] = data.get('oxygen_saturation', 98)
                    else:
                        features[feature] = 0
            
            # Create a DataFrame with the expected features
            import pandas as pd
            df = pd.DataFrame([features])
            
            # Make prediction
            probability = self.model.predict_proba(df)[0][1]
            return float(probability)
            
        except Exception as e:
            logger.error(f"Error making mortality prediction: {e}")
            # Fallback to simple prediction
            return 0.2
    
    def predict_binary(self, data):
        """Predict binary mortality outcome using custom threshold."""
        probability = self.predict(data)
        return 1 if probability >= self.prediction_threshold else 0

@app.route('/health', methods=['GET'])
def health_check():
    """Health check endpoint."""
    logger.info("Health check requested")
    return jsonify({
        'status': 'healthy',
        'timestamp': datetime.now().isoformat(),
        'version': '1.2-fileupload' # Indicate version
    })

#  Prediction Endpoint 
@app.route('/api/predict', methods=['POST'])
def predict():
    """Endpoint for making predictions with trained models."""
    try:
        data = request.json
        logger.warning(f"Received prediction request for patient {data.get('patient_id')}")
        
        # Initialize results dictionary
        results = {
            'timestamp': datetime.now().isoformat(),
            'predictions': {},
            'thresholds_used': {}
        }
        
        # Mortality prediction 
        mortality_model = MortalityModel()
        mortality_risk = mortality_model.predict(data)
        logger.info(f"Raw mortality prediction: {mortality_risk}")
        
        results['predictions']['mortality'] = mortality_risk
        results['predictions']['mortality_binary'] = 1 if mortality_risk >= mortality_model.prediction_threshold else 0
        results['thresholds_used']['mortality'] = mortality_model.prediction_threshold
        
        logger.warning(f"Prediction completed for patient {data.get('patient_id')}: {results['predictions']}")
        return jsonify(results)
        
    except Exception as e:
        logger.error(f"Error in prediction: {str(e)}")
        return jsonify({'error': str(e)}), 500

#Other Endpoints 

@app.route('/api/available_models', methods=['GET'])
def available_models_endpoint():
    """Return a list of available prediction models found."""
    logger.info("Available models endpoint requested")
    models_found = get_available_models()
    # Extract just the names for the response
    model_names = [model['name'] for model in models_found]
    return jsonify({'available_models': model_names})

@app.route('/api/time_series_data', methods=['GET'])
def get_time_series_data():
    """Returns the patient time series data stored in a JSON file."""
    try:
        file_path = 'data/results/patient_time_series.json'
        if not os.path.exists(file_path):
            return jsonify({'error': 'Time series data not found'}), 404
            
        with open(file_path, 'r') as f:
            data = json.load(f)
        return jsonify(data)
    except Exception as e:
        logger.error(f"Error reading time series data: {str(e)}")
        return jsonify({'error': str(e)}), 500


@app.errorhandler(404)
def not_found(error):
    """Handle 404 errors."""
    logger.warning(f"404 Not Found error: {request.url}")
    return jsonify({'error': 'Not found'}), 404

@app.errorhandler(500)
def server_error(error):
    """Handle 500 errors."""
    logger.error(f"500 Server error: {error}")
    # Include traceback in logs but not necessarily response
    logger.error(traceback.format_exc()) 
    return jsonify({'error': 'Internal server error'}), 500

if __name__ == '__main__':
    PORT = 8000  # API server port
    logger.info(f"Starting MIMIC-III Clinical Prediction API (File Upload) on port {PORT}")
    print(f"Starting MIMIC-III Clinical Prediction API (File Upload) on port {PORT}")
    print(f"* Health check: http://localhost:{PORT}/health")
    print(f"* Available models: http://localhost:{PORT}/api/available_models")
    print(f"* Make predictions: POST file to http://localhost:{PORT}/api/predict")
    print(f"* Models directory: {app.config['MODELS_DIR']}")
    print(f"* Upload folder: {app.config['UPLOAD_FOLDER']}")
    app.run(host='0.0.0.0', port=PORT, debug=False) 