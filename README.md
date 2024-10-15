# Energy Management System

## Overview
This project provides a comprehensive solution for energy management by collecting data, predicting future usage, and optimizing energy consumption.

## Components
1. **Data Collection**: Simulates energy consumption data using sensors.
2. **Prediction Model**: AI-based linear regression model to forecast energy usage.
3. **Optimization**: Adjusts energy consumption based on predictions.
4. **Visualization**: Plots energy data for better insights.

## How to Run
1. Generate data: `python scripts/data_collector.py`
2. Train model: `python scripts/predictor.py`
3. Optimize energy usage: `python scripts/optimizer.py`
4. Visualize data: `python scripts/visualize.py`

## Requirements
- Python 3.x
- Pandas
- scikit-learn
- matplotlib