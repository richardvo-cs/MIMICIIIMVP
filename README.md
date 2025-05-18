# MIMIC-III Clinical Prediction System

This project provides tools for analysis, visualization, and predictive modeling using the MIMIC-III critical care database. The system includes a real-time patient monitoring dashboard with mortality risk prediction.

## Features

- **Interactive Patient Monitoring Dashboard**
  - Real-time monitoring of vital signs in separate visualizations
  - Heart rate, temperature, blood pressure, and oxygen saturation tracking
  - Mortality risk predictions with alerts for high-risk patients
  - Responsive, user-friendly interface with anti-caching measures

- **Real-time Data Generation**
  - Simulated patient monitoring data for demonstration
  - Customizable vital sign patterns with realistic clinical changes
  - Support for both normal and critical patient scenarios

- **Clinical Prediction API**
  - REST API for making mortality predictions
  - Machine learning models trained on MIMIC-III data
  - JSON-based data exchange format

- **Visualization Server**
  - Lightweight HTTP server for dashboard delivery
  - No-cache headers for real-time updates
  - Support for multiple dashboard types

## Getting Started

### Prerequisites

- Python 3.8+
- pip package manager

### Installation

1. Clone the repository:
   ```bash
   git clone https://github.com/yourusername/MIMICIII.git
   cd MIMICIII
   ```

2. Install the required packages:
   ```bash
   pip install -r requirements.txt
   ```

### Running the System

1. Start the API server:
   ```bash
   python scripts/api.py
   ```

2. Start the demo monitor (data generator):
   ```bash
   python scripts/demo_monitor.py
   ```

3. Start the visualization server:
   ```bash
   python scripts/serve_visualizations.py
   ```

4. Access the dashboard at: http://localhost:8080/interactive_dashboard.html

## Dashboard Features

The interactive dashboard provides:

- Separate visualizations for each vital sign
- Color-coded charts with appropriate clinical ranges
- Real-time updates every 2 seconds
- Mortality risk calculation and visualization
- Mobile-friendly responsive design

## API Endpoints

- `GET /health`: Health check endpoint
- `GET /api/available_models`: List available prediction models
- `POST /api/predict`: Submit patient data for prediction
- `GET /api/time_series_data`: Get the latest patient time series data

## Project Structure

- `data/`: Contains data files and results
- `models/`: Contains trained machine learning models
- `scripts/`: Contains Python scripts for running components
- `src/`: Source code for utilities and libraries

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Acknowledgments

- MIMIC-III database (Johnson AEW, Pollard TJ, Shen L, Lehman L, Feng M, Ghassemi M, Moody B, Szolovits P, Celi LA, and Mark RG. MIMIC-III, a freely accessible critical care database. Scientific Data (2016).)

## System Components

- **API Server** (port 8000): Handles prediction requests and data storage
- **Demo Monitor**: Simulates patient data generation
- **Visualization Server** (port 8080): Serves the interactive dashboard

## Directory Structure

```
MIMICIII/
├── data/
│   ├── results/        # For visualization outputs
│   └── uploads/        # For temporary storage
├── models/             # For ML models
├── scripts/
│   ├── api.py         # API server
│   ├── demo_monitor.py # Patient data simulation
│   ├── serve_visualizations.py # Dashboard server
│   └── prediction_script.py # Core prediction logic
├── README.md
└── requirements.txt
```

## Requirements

- Python 3.8+
- Flask
- scikit-learn
- pandas
- numpy
- joblib

## Notes

- The system uses simulated patient data for demonstration
- Mortality risk predictions are based on vital signs
- The dashboard updates every 5 seconds 