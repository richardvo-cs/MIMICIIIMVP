# MIMIC-III Clinical Prediction System (Lightweight)

This is a lightweight version of the MIMIC-III Clinical Prediction System. This repository contains the dashboard and scripts without the large data files.

## Overview

The MIMIC-III Clinical Prediction System provides real-time monitoring and visualization of patient vital signs with mortality risk prediction. This lightweight repository includes:

- Interactive dashboard with safe destroy checks
- Separated vital signs visualizations (heart rate, temperature, blood pressure, SpO2)
- Real-time update capabilities (2-second intervals)
- Support scripts for API, data generation, and visualization

## Included Components

- **Interactive Dashboard (`data/results/interactive_dashboard.html`)**
  - Real-time monitoring with separate visualizations
  - Mortality risk visualization
  - Mobile-friendly responsive design
  - Anti-caching measures

- **Scripts**
  - `api.py`: REST API for data and predictions
  - `demo_monitor.py`: Patient data simulation
  - `serve_visualizations.py`: Dashboard web server
  - `prediction_script.py`: Core prediction logic

## Getting Started

### Prerequisites

- Python 3.8+
- Required Python packages (listed in requirements.txt)

### Setup

1. Clone this repository
2. Install dependencies: `pip install -r requirements.txt`
3. Create the necessary directories if they don't exist:
   ```
   mkdir -p data/results data/uploads models/mortality models/readmission models/sepsis
   ```

### Running the System

1. Start the API server: `python scripts/api.py`
2. Start the demo monitor: `python scripts/demo_monitor.py`
3. Start the visualization server: `python scripts/serve_visualizations.py`
4. Access the dashboard at: http://localhost:8080/interactive_dashboard.html

## Data Generation and Models

### Demo Mode with Synthetic Data

This system includes a powerful `demo_monitor.py` script that generates realistic synthetic patient data. You **do not need** real MIMIC-III data to see the dashboard in action. The demo monitor:

- Creates simulated patients with realistic vital signs (HR, temp, BP, SpO2)
- Generates patterns of clinical deterioration (spike patterns in vital signs)
- Simulates critical events (elevated heart rate, decreased oxygen saturation)
- Produces mortality risk scores using API predictions

The demo has been tested and produces realistic clinical scenarios, as seen in these sample patients:
- Patient ID 63745: Mortality risk 0.60-0.68, showing critical HR patterns (140-160 bpm)
- Patient ID 29809: Mortality risk 0.57-0.67, with gradual deterioration in vital signs
- Patient ID 7543: Mortality risk 0.60-0.67, showing fever patterns (37.5-40.0°C)

### Using Real MIMIC-III Data

While the demo mode works out-of-the-box, this repository is missing the following from the full version:

1. **Complete MIMIC-III Dataset** (62GB):
   - Raw data files (4.3GB compressed CHARTEVENTS.csv.gz)
   - Extracted CSV files (35GB for CHARTEVENTS.csv alone)

2. **Trained Models**:
   - Mortality prediction: Random Forest (64MB)
   - Sepsis prediction: XGBoost (268KB)
   - Readmission prediction: Gradient Boosting

3. **Feature Data**:
   - Pre-computed features in PKL format
   - Processed training and test datasets

To use real data instead of the demo, you would need:
1. MIMIC-III database access (requires credentialing through PhysioNet)
2. Trained models (can be recreated using scripts)
3. The preprocessing pipeline to transform raw data into features

## License

This project is licensed under the MIT License.
