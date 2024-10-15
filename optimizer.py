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