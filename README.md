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

## Note on Data Files

This repository does not include the large data files and trained models from the original project. To use this system with real data, you would need to:

1. Obtain access to the MIMIC-III database
2. Process the data according to the project's requirements
3. Train the prediction models using the provided scripts

## License

This project is licensed under the MIT License.
