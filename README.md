pip install pandas scikit-learn matplotlib
import pandas as pd
import random
from datetime import datetime, timedelta

def generate_energy_data(days=30):
    data = []
    start_time = datetime.now() - timedelta(days=days)
    
    for i in range(days * 24):
        timestamp = start_time + timedelta(hours=i)
        energy_usage = round(random.uniform(10, 50), 2)  # Simulate energy usage in kWh
        data.append([timestamp, energy_usage])
    
    df = pd.DataFrame(data, columns=["Timestamp", "Energy_Usage"])
    df.to_csv("data/energy_usage.csv", index=False)
    print("Energy data generated successfully.")

if __name__ == "__main__":
    generate_energy_data()
#AI predictor model
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
import joblib

def train_model():
    data = pd.read_csv("data/energy_usage.csv")
    data['Hour'] = pd.to_datetime(data['Timestamp']).dt.hour

    X = data[['Hour']]
    y = data['Energy_Usage']

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    model = LinearRegression()
    model.fit(X_train, y_train)

    print(f"Model trained with R² score: {model.score(X_test, y_test):.2f}")
    joblib.dump(model, "models/energy_predictor.pkl")

if __name__ == "__main__":
    train_model()
#optimizer.py
import pandas as pd
import joblib

def optimize_energy(hour):
    model = joblib.load("models/energy_predictor.pkl")
    predicted_usage = model.predict([[hour]])[0]

    # Simulate optimization by suggesting adjustments
    if predicted_usage > 40:
        action = "Reduce usage - Shift non-essential tasks to off-peak hours."
    elif 20 <= predicted_usage <= 40:
        action = "Maintain usage - Energy usage is within the optimal range."
    else:
        action = "Increase usage - Consider using devices now to balance load."

    return f"Predicted usage for hour {hour}: {predicted_usage:.2f} kWh. {action}"

if __name__ == "__main__":
    hour = int(input("Enter current hour (0-23): "))
    print(optimize_energy(hour))
#visualize.py
import pandas as pd
import matplotlib.pyplot as plt

def plot_energy_data():
    data = pd.read_csv("data/energy_usage.csv")
    data['Timestamp'] = pd.to_datetime(data['Timestamp'])
    plt.figure(figsize=(10, 5))

    plt.plot(data['Timestamp'], data['Energy_Usage'], label='Energy Usage (kWh)')
    plt.xlabel('Time')
    plt.ylabel('Energy Usage (kWh)')
    plt.title('Energy Consumption Over Time')
    plt.legend()
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    plot_energy_data()
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