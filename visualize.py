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