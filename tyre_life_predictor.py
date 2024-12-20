import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score
import os

class TyreLifePredictor:
    def __init__(self, data_file):
        file_path = os.path.abspath(data_file)
        self.data = pd.read_csv(file_path)
        self.model = RandomForestRegressor(random_state=42)

    def preprocess_data(self):
        self.data['TyreType'] = self.data['TyreType'].astype('category').cat.codes
        X = self.data[['Lap', 'TyreType', 'WearRate', 'TrackTemperature']]
        y = self.data['TyreLife']
        return train_test_split(X, y, test_size=0.2, random_state=42)

    def train_model(self, X_train, y_train):
        self.model.fit(X_train, y_train)

    def predict(self, X):
        return self.model.predict(X)

    def evaluate_model(self, X_test, y_test):
        predictions = self.predict(X_test)
        mse = mean_squared_error(y_test, predictions)
        r2 = r2_score(y_test, predictions)
        return mse, r2

if __name__ == "__main__":
    data_file = "Tyre-Life-Predictor/example_tyre_life.csv"
    print(f"Expected file path: {os.path.abspath(data_file)}")

    predictor = TyreLifePredictor(data_file)
    X_train, X_test, y_train, y_test = predictor.preprocess_data()

    predictor.train_model(X_train, y_train)

    mse, r2 = predictor.evaluate_model(X_test, y_test)
    print(f"Model Performance:\nMean Squared Error: {mse:.2f}\nR² Score: {r2:.2f}")

    example_data = pd.DataFrame({
        'Lap': [5, 10],
        'TyreType': [0, 1],  # 0: Soft, 1: Medium (encoded as in preprocessing)
        'WearRate': [2.5, 1.8],
        'TrackTemperature': [30, 35]
    })
    predictions = predictor.predict(example_data)
    for i, pred in enumerate(predictions):
        print(f"Predicted tyre life for example {i + 1}: {pred:.2f} laps")
