#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
MIMIC-III Visualization Server
Serves the HTML visualizations for the MIMIC-III dataset.
"""

import os
import sys
import http.server
import socketserver
import webbrowser
from pathlib import Path
import json
from datetime import datetime

PORT = 8080
RESULTS_DIR = 'data/results'

class Handler(http.server.SimpleHTTPRequestHandler):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, directory=RESULTS_DIR, **kwargs)
    
    def end_headers(self):
        # Add headers to prevent caching
        self.send_header('Access-Control-Allow-Origin', '*')
        self.send_header('Cache-Control', 'no-cache, no-store, must-revalidate')
        self.send_header('Pragma', 'no-cache')
        self.send_header('Expires', '0')
        http.server.SimpleHTTPRequestHandler.end_headers(self)
    
    def do_GET(self):
        """Handle GET requests, with special treatment for JSON data files"""
        # Special handling for the time series data file to ensure it's always fresh
        if self.path.endswith('patient_time_series.json') or '?' in self.path and self.path.split('?')[0].endswith('patient_time_series.json'):
            # Remove query parameters if present
            clean_path = self.path.split('?')[0] if '?' in self.path else self.path
            file_path = os.path.join(RESULTS_DIR, clean_path.lstrip('/'))
            
            try:
                with open(file_path, 'rb') as f:
                    content = f.read()
                
                self.send_response(200)
                self.send_header('Content-Type', 'application/json')
                self.send_header('Content-Length', str(len(content)))
                # Add strong cache prevention headers
                self.send_header('Cache-Control', 'no-cache, no-store, must-revalidate, max-age=0')
                self.send_header('Pragma', 'no-cache')
                self.send_header('Expires', '-1')
                self.end_headers()
                self.wfile.write(content)
                return
            except Exception as e:
                print(f"Error serving {file_path}: {e}")
                # Fall through to default handler
        
        # Use default handler for all other files
        return http.server.SimpleHTTPRequestHandler.do_GET(self)

def create_interactive_dashboard():
    """Creates the interactive dashboard HTML file."""
    html_content = """
    <!DOCTYPE html>
    <html>
    <head>
        <title>Patient Monitoring Dashboard</title>
        <script src="https://cdn.jsdelivr.net/npm/chart.js"></script>
        <meta http-equiv="Cache-Control" content="no-cache, no-store, must-revalidate" />
        <meta http-equiv="Pragma" content="no-cache" />
        <meta http-equiv="Expires" content="0" />
        <style>
            body { font-family: Arial, sans-serif; margin: 20px; }
            .chart-container { 
                margin-bottom: 20px;
                width: 100%;
                max-width: 1200px;
                margin-left: auto;
                margin-right: auto;
            }
            .alert { color: red; font-weight: bold; }
            .chart-grid {
                display: grid;
                grid-template-columns: repeat(2, 1fr);
                gap: 20px;
                margin-top: 20px;
            }
            .chart-wrapper {
                background: #f8f9fa;
                padding: 15px;
                border-radius: 8px;
                box-shadow: 0 2px 4px rgba(0,0,0,0.1);
            }
            .chart-title {
                text-align: center;
                margin-bottom: 10px;
                color: #333;
                font-size: 1.2em;
            }
        </style>
    </head>
    <body>
        <h1>Patient Monitoring Dashboard</h1>
        <div id="patientDetails"></div>
        <div id="riskIndicators"></div>
        <div class="chart-container">
            <canvas id="mortalityChart"></canvas>
        </div>
        <div class="chart-grid">
            <div class="chart-wrapper">
                <div class="chart-title">Heart Rate</div>
                <canvas id="heartRateChart"></canvas>
            </div>
            <div class="chart-wrapper">
                <div class="chart-title">Temperature</div>
                <canvas id="temperatureChart"></canvas>
            </div>
            <div class="chart-wrapper">
                <div class="chart-title">Blood Pressure</div>
                <canvas id="bloodPressureChart"></canvas>
            </div>
            <div class="chart-wrapper">
                <div class="chart-title">Oxygen Saturation</div>
                <canvas id="oxygenChart"></canvas>
            </div>
        </div>
        <script>
            let currentPatient = null;
            const MAX_DATAPOINTS = 100;
            
            async function loadPatientData() {
                try {
                    const response = await fetch('http://localhost:8000/api/time_series_data?ts=' + new Date().getTime());
                    if (!response.ok) {
                        if (response.status === 404) {
                            document.getElementById('patientDetails').innerHTML = '<p>Waiting for monitoring data...</p>';
                            document.getElementById('riskIndicators').innerHTML = '';
                            return;
                        }
                        throw new Error(`HTTP error: ${response.status}`);
                    }
                    const data = await response.json();
                    
                    if (!data || !data.timestamps || data.timestamps.length === 0) {
                        document.getElementById('patientDetails').innerHTML = '<p>Waiting for monitoring data...</p>';
                        document.getElementById('riskIndicators').innerHTML = '';
                        return;
                    }

                    // Ensure all required arrays exist and have the same length
                    const requiredArrays = ['timestamps', 'heart_rate', 'temperature', 'systolic_bp', 'oxygen_saturation', 'mortality_risk'];
                    const arrayLength = data.timestamps.length;
                    
                    for (const key of requiredArrays) {
                        if (!data[key] || !Array.isArray(data[key]) || data[key].length !== arrayLength) {
                            console.error(`Missing or invalid data for ${key}`);
                            document.getElementById('patientDetails').innerHTML = '<p>Error: Invalid data format</p>';
                            document.getElementById('riskIndicators').innerHTML = '';
                            return;
                        }
                    }
                    
                    currentPatient = data;
                    
                    const dataLength = currentPatient.timestamps.length;
                    if (dataLength > MAX_DATAPOINTS) {
                        const startIndex = dataLength - MAX_DATAPOINTS;
                        for (const key in currentPatient) {
                            if (Array.isArray(currentPatient[key])) {
                                currentPatient[key] = currentPatient[key].slice(startIndex);
                            }
                        }
                    }

                    updateDashboard();
                } catch (error) {
                    console.error('Error loading patient data:', error);
                    document.getElementById('patientDetails').innerHTML = `<p style="color: red;">Error loading data: ${error.message}</p>`;
                    document.getElementById('riskIndicators').innerHTML = '';
                }
            }

            function updateDashboard() {
                updatePatientDetails();
                updateRiskIndicators();
                updateMortalityChart();
                updateHeartRateChart();
                updateTemperatureChart();
                updateBloodPressureChart();
                updateOxygenChart();
            }

            function updatePatientDetails() {
                const latest = currentPatient.timestamps.length - 1;
                const details = `
                    <h2>Patient Details</h2>
                    <p>Patient ID: ${currentPatient.patient_id}</p>
                    <p>Latest Update: ${currentPatient.timestamps[latest]}</p>
                    <p>Heart Rate: ${currentPatient.heart_rate[latest]} bpm</p>
                    <p>Temperature: ${currentPatient.temperature[latest]}°C</p>
                    <p>Blood Pressure: ${currentPatient.systolic_bp[latest]} mmHg</p>
                    <p>SpO2: ${currentPatient.oxygen_saturation[latest]}%</p>
                `;
                document.getElementById('patientDetails').innerHTML = details;
            }

            function updateRiskIndicators() {
                const latest = currentPatient.timestamps.length - 1;
                const mortalityRisk = currentPatient.mortality_risk[latest];
                const riskLevel = mortalityRisk > 0.3 ? 'HIGH' : 'LOW';
                const riskClass = mortalityRisk > 0.3 ? 'alert' : '';
                
                const indicators = `
                    <h2>Risk Indicators</h2>
                    <p>Mortality Risk: <span class="${riskClass}">${(mortalityRisk * 100).toFixed(1)}% (${riskLevel})</span></p>
                `;
                document.getElementById('riskIndicators').innerHTML = indicators;
            }

            function updateMortalityChart() {
                const ctx = document.getElementById('mortalityChart').getContext('2d');
                if (window.mortalityChart && typeof window.mortalityChart.destroy === 'function') {
                    window.mortalityChart.destroy();
                }
                
                window.mortalityChart = new Chart(ctx, {
                    type: 'line',
                    data: {
                        labels: currentPatient.timestamps,
                        datasets: [{
                            label: 'Mortality Risk',
                            data: currentPatient.mortality_risk,
                            borderColor: 'red',
                            backgroundColor: 'rgba(255, 0, 0, 0.1)',
                            tension: 0.1,
                            fill: true
                        }]
                    },
                    options: {
                        responsive: true,
                        scales: {
                            y: {
                                beginAtZero: true,
                                max: 1.0
                            }
                        }
                    }
                });
            }

            function updateHeartRateChart() {
                const ctx = document.getElementById('heartRateChart').getContext('2d');
                if (window.heartRateChart && typeof window.heartRateChart.destroy === 'function') {
                    window.heartRateChart.destroy();
                }
                
                window.heartRateChart = new Chart(ctx, {
                    type: 'line',
                    data: {
                        labels: currentPatient.timestamps,
                        datasets: [{
                            label: 'Heart Rate',
                            data: currentPatient.heart_rate,
                            borderColor: 'rgb(54, 162, 235)',
                            backgroundColor: 'rgba(54, 162, 235, 0.1)',
                            tension: 0.1,
                            fill: true
                        }]
                    },
                    options: {
                        responsive: true,
                        scales: {
                            y: {
                                beginAtZero: false,
                                suggestedMin: 40,
                                suggestedMax: 160,
                                title: {
                                    display: true,
                                    text: 'BPM'
                                }
                            }
                        }
                    }
                });
            }

            function updateTemperatureChart() {
                const ctx = document.getElementById('temperatureChart').getContext('2d');
                if (window.temperatureChart && typeof window.temperatureChart.destroy === 'function') {
                    window.temperatureChart.destroy();
                }
                
                window.temperatureChart = new Chart(ctx, {
                    type: 'line',
                    data: {
                        labels: currentPatient.timestamps,
                        datasets: [{
                            label: 'Temperature',
                            data: currentPatient.temperature,
                            borderColor: 'rgb(255, 159, 64)',
                            backgroundColor: 'rgba(255, 159, 64, 0.1)',
                            tension: 0.1,
                            fill: true
                        }]
                    },
                    options: {
                        responsive: true,
                        scales: {
                            y: {
                                beginAtZero: false,
                                suggestedMin: 36,
                                suggestedMax: 40,
                                title: {
                                    display: true,
                                    text: '°C'
                                }
                            }
                        }
                    }
                });
            }

            function updateBloodPressureChart() {
                const ctx = document.getElementById('bloodPressureChart').getContext('2d');
                if (window.bloodPressureChart && typeof window.bloodPressureChart.destroy === 'function') {
                    window.bloodPressureChart.destroy();
                }
                
                window.bloodPressureChart = new Chart(ctx, {
                    type: 'line',
                    data: {
                        labels: currentPatient.timestamps,
                        datasets: [{
                            label: 'Systolic BP',
                            data: currentPatient.systolic_bp,
                            borderColor: 'rgb(75, 192, 192)',
                            backgroundColor: 'rgba(75, 192, 192, 0.1)',
                            tension: 0.1,
                            fill: true
                        }]
                    },
                    options: {
                        responsive: true,
                        scales: {
                            y: {
                                beginAtZero: false,
                                suggestedMin: 60,
                                suggestedMax: 200,
                                title: {
                                    display: true,
                                    text: 'mmHg'
                                }
                            }
                        }
                    }
                });
            }

            function updateOxygenChart() {
                const ctx = document.getElementById('oxygenChart').getContext('2d');
                if (window.oxygenChart && typeof window.oxygenChart.destroy === 'function') {
                    window.oxygenChart.destroy();
                }
                
                window.oxygenChart = new Chart(ctx, {
                    type: 'line',
                    data: {
                        labels: currentPatient.timestamps,
                        datasets: [{
                            label: 'SpO2',
                            data: currentPatient.oxygen_saturation,
                            borderColor: 'rgb(153, 102, 255)',
                            backgroundColor: 'rgba(153, 102, 255, 0.1)',
                            tension: 0.1,
                            fill: true
                        }]
                    },
                    options: {
                        responsive: true,
                        scales: {
                            y: {
                                beginAtZero: false,
                                suggestedMin: 80,
                                suggestedMax: 100,
                                title: {
                                    display: true,
                                    text: '%'
                                }
                            }
                        }
                    }
                });
            }

            // Load data every 2 seconds
            setInterval(loadPatientData, 2000);
            loadPatientData();
        </script>
    </body>
    </html>
    """
    
    os.makedirs(RESULTS_DIR, exist_ok=True)
    with open(os.path.join(RESULTS_DIR, 'interactive_dashboard.html'), 'w') as f:
        f.write(html_content)

def main():
    # Ensure the results directory exists
    Path(RESULTS_DIR).mkdir(parents=True, exist_ok=True)
    
    # Generate the HTML files
    # create_time_series_plot() # Keep this commented out or remove if not needed
    create_interactive_dashboard() 
    
    # Start the server
    with socketserver.TCPServer(("", PORT), Handler) as httpd:
        print(f"Starting server at http://localhost:{PORT}")
        print(f"Serving files from: {os.path.abspath(RESULTS_DIR)}")
        print(f"Interactive dashboard available at: http://localhost:{PORT}/interactive_dashboard.html")
        # Automatically open the interactive dashboard in the browser
        try:
            webbrowser.open_new_tab(f'http://localhost:{PORT}/interactive_dashboard.html')
        except Exception as e:
             print(f"Could not open browser automatically: {e}")
        httpd.serve_forever()

if __name__ == "__main__":
    main() 