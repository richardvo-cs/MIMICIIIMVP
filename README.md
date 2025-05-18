# MIMIC-III Clinical Prediction System

This project provides tools for analysis, visualization, and predictive modeling using the MIMIC-III critical care database.

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

## System Components

- **API Server** (port 8000): Handles prediction requests and data storage
- **Demo Monitor**: Simulates patient data generation
- **Visualization Server** (port 8080): Serves the interactive dashboard

## Requirements

- Python 3.8+
- Flask
- scikit-learn
- pandas
- numpy
- joblib
