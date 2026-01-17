# One-command run: python main.py

from src.data_loader import load_all_data
from src.feature_engineering import create_features
from src.prediction import predict_2026

data = load_all_data()
features = create_features(data)
predictions = predict_2026(features)
predictions.to_csv("data/processed/2026_predictions.csv")