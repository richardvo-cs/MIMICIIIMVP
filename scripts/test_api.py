#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
MIMIC-III API Test Script
A simple script to test the clinical prediction API.
"""

import os
import sys
import json
import time
import argparse
import requests
import pandas as pd
import logging
from datetime import datetime

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# API endpoints
BASE_URL = "http://localhost:8000"
HEALTH_ENDPOINT = f"{BASE_URL}/health"
AVAILABLE_MODELS_ENDPOINT = f"{BASE_URL}/api/available_models"
PREDICT_ENDPOINT = f"{BASE_URL}/api/predict"

def check_api_health():
    """Check if the API is healthy"""
    try:
        response = requests.get(HEALTH_ENDPOINT)
        if response.status_code == 200:
            logger.info(f"API Health Status: {response.json()}")
            return True
        else:
            logger.error(f"API health check failed with status code {response.status_code}")
            return False
    except Exception as e:
        logger.error(f"Error checking API health: {e}")
        return False

def get_available_models():
    """Get the list of available models from the API"""
    try:
        response = requests.get(AVAILABLE_MODELS_ENDPOINT)
        if response.status_code == 200:
            logger.info(f"Available Models: {response.json()['available_models']}")
            return response.json()['available_models']
        else:
            logger.error(f"Failed to get available models, status code: {response.status_code}")
            return []
    except Exception as e:
        logger.error(f"Error getting available models: {e}")
        return []

def send_prediction_request(file_path, model_type="all"):
    """Send a prediction request to the API"""
    try:
        start_time = time.time()
        
        with open(file_path, 'rb') as file:
            files = {'file': file}
            response = requests.post(f"{PREDICT_ENDPOINT}?model_type={model_type}", files=files)
        
        elapsed_time = (time.time() - start_time) * 1000  # Convert to milliseconds
        
        if response.status_code == 200:
            results = response.json()
            num_patients = len(results.get('mortality_predictions', {}).get('predictions', [])) if 'mortality_predictions' in results else 0
            logger.info(f"Processed {num_patients} patients in {elapsed_time:.0f} ms")
            return results
        else:
            logger.error(f"Prediction request failed with status code {response.status_code}: {response.text}")
            return None
    except Exception as e:
        logger.error(f"Error sending prediction request: {e}")
        return None

def save_results(results, output_dir="data/results"):
    """Save prediction results to files"""
    try:
        # Create output directory if it doesn't exist
        os.makedirs(output_dir, exist_ok=True)
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # Save each prediction type to a separate file
        if results.get('los_predictions'):
            los_df = pd.DataFrame(results['los_predictions']['predictions'])
            los_df.to_csv(f"{output_dir}/los_predictions_{timestamp}.csv", index=False)
            logger.info(f"Saved LOS predictions to {output_dir}/los_predictions_{timestamp}.csv")
        
        if results.get('mortality_predictions'):
            mortality_df = pd.DataFrame(results['mortality_predictions']['predictions'])
            mortality_df.to_csv(f"{output_dir}/mortality_predictions_{timestamp}.csv", index=False)
            logger.info(f"Saved mortality predictions to {output_dir}/mortality_predictions_{timestamp}.csv")
        
        if results.get('readmission_predictions'):
            readmission_df = pd.DataFrame(results['readmission_predictions']['predictions'])
            readmission_df.to_csv(f"{output_dir}/readmission_predictions_{timestamp}.csv", index=False)
            logger.info(f"Saved readmission predictions to {output_dir}/readmission_predictions_{timestamp}.csv")
        
        if results.get('sepsis_predictions'):
            sepsis_df = pd.DataFrame(results['sepsis_predictions']['predictions'])
            sepsis_df.to_csv(f"{output_dir}/sepsis_predictions_{timestamp}.csv", index=False)
            logger.info(f"Saved sepsis predictions to {output_dir}/sepsis_predictions_{timestamp}.csv")
        
        # Save all results as JSON
        with open(f"{output_dir}/all_predictions_{timestamp}.json", 'w') as f:
            json.dump(results, f, indent=2)
        logger.info(f"Saved all predictions to {output_dir}/all_predictions_{timestamp}.json")
        
    except Exception as e:
        logger.error(f"Error saving results: {e}")

def main():
    parser = argparse.ArgumentParser(description="Test the MIMIC-III Clinical Prediction API")
    parser.add_argument("--file", required=True, help="Path to the patient data file")
    parser.add_argument("--model", default="all", help="Model type to use for prediction")
    parser.add_argument("--output", default="data/results", help="Directory to save prediction results")
    args = parser.parse_args()
    
    # Check if file exists
    if not os.path.exists(args.file):
        logger.error(f"File {args.file} does not exist")
        sys.exit(1)
    
    # Check if API is healthy
    if not check_api_health():
        logger.error("API is not healthy, exiting")
        sys.exit(1)
    
    # Get available models
    available_models = get_available_models()
    if not available_models and args.model != "all":
        logger.error(f"No available models found or model {args.model} not available")
        sys.exit(1)
    
    # Send prediction request
    results = send_prediction_request(args.file, args.model)
    if results:
        # Print prediction results
        if 'los_predictions' in results:
            logger.info("LOS Predictions:")
            for pred in results['los_predictions']['predictions'][:5]:  # Print first 5 predictions
                logger.info(f"  {pred}")
            if len(results['los_predictions']['predictions']) > 5:
                logger.info(f"  ... and {len(results['los_predictions']['predictions']) - 5} more")
        
        if 'mortality_predictions' in results:
            logger.info("Mortality Predictions:")
            for pred in results['mortality_predictions']['predictions'][:5]:
                logger.info(f"  {pred}")
            if len(results['mortality_predictions']['predictions']) > 5:
                logger.info(f"  ... and {len(results['mortality_predictions']['predictions']) - 5} more")
        
        if 'readmission_predictions' in results:
            logger.info("Readmission Predictions:")
            for pred in results['readmission_predictions']['predictions'][:5]:
                logger.info(f"  {pred}")
            if len(results['readmission_predictions']['predictions']) > 5:
                logger.info(f"  ... and {len(results['readmission_predictions']['predictions']) - 5} more")
        
        if 'sepsis_predictions' in results:
            logger.info("Sepsis Predictions:")
            for pred in results['sepsis_predictions']['predictions'][:5]:
                logger.info(f"  {pred}")
            if len(results['sepsis_predictions']['predictions']) > 5:
                logger.info(f"  ... and {len(results['sepsis_predictions']['predictions']) - 5} more")
        
        # Save results
        save_results(results, args.output)
    else:
        logger.error("Failed to get predictions")
        sys.exit(1)

if __name__ == "__main__":
    main() 