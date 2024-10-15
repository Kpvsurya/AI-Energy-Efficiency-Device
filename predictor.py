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