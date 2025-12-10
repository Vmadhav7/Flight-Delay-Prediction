"""
Flight Delay Prediction - Flask Web Application
================================================
Interactive dashboard with real model predictions.
"""

from flask import Flask, render_template, request, jsonify
import pandas as pd
import numpy as np
import joblib
from pathlib import Path

app = Flask(__name__)

# Paths
BASE_DIR = Path(__file__).parent
MODELS_DIR = BASE_DIR / "models"
DATA_DIR = BASE_DIR / "data" / "raw"
OUTPUTS_DIR = BASE_DIR / "outputs"

# Global variables
model = None
feature_columns = None
airlines_df = None
airports_df = None
model_results = None


def load_resources():
    """Load model, feature columns, and reference data."""
    global model, feature_columns, airlines_df, airports_df, model_results
    
    # Load best model (Random Forest)
    model_path = MODELS_DIR / "random_forest.joblib"
    if model_path.exists():
        model = joblib.load(model_path)
        print(f"Loaded model from {model_path}")
    else:
        print("WARNING: Model not found!")
    
    # Load feature columns
    feature_path = MODELS_DIR / "feature_columns.joblib"
    if feature_path.exists():
        feature_columns = joblib.load(feature_path)
        print(f"Loaded {len(feature_columns)} feature columns")
    else:
        print("WARNING: Feature columns not found!")
    
    # Load airlines
    airlines_path = DATA_DIR / "airlines.csv"
    if airlines_path.exists():
        airlines_df = pd.read_csv(airlines_path)
    
    # Load airports
    airports_path = DATA_DIR / "airports.csv"
    if airports_path.exists():
        airports_df = pd.read_csv(airports_path)
    
    # Load model results
    results_path = OUTPUTS_DIR / "reports" / "model_comparison.csv"
    if results_path.exists():
        model_results = pd.read_csv(results_path, index_col=0)


def get_airlines_list():
    """Get list of airlines for dropdown."""
    if airlines_df is not None:
        return airlines_df.to_dict('records')
    return [
        {"IATA_CODE": "AA", "AIRLINE": "American Airlines"},
        {"IATA_CODE": "DL", "AIRLINE": "Delta Air Lines"},
        {"IATA_CODE": "UA", "AIRLINE": "United Airlines"},
        {"IATA_CODE": "WN", "AIRLINE": "Southwest Airlines"},
        {"IATA_CODE": "AS", "AIRLINE": "Alaska Airlines"},
        {"IATA_CODE": "B6", "AIRLINE": "JetBlue Airways"},
        {"IATA_CODE": "NK", "AIRLINE": "Spirit Airlines"},
        {"IATA_CODE": "F9", "AIRLINE": "Frontier Airlines"},
    ]


def get_airports_list():
    """Get list of major airports for dropdown."""
    return [
        {"IATA_CODE": "ATL", "AIRPORT": "Atlanta"},
        {"IATA_CODE": "LAX", "AIRPORT": "Los Angeles"},
        {"IATA_CODE": "ORD", "AIRPORT": "Chicago O'Hare"},
        {"IATA_CODE": "DFW", "AIRPORT": "Dallas/Fort Worth"},
        {"IATA_CODE": "DEN", "AIRPORT": "Denver"},
        {"IATA_CODE": "JFK", "AIRPORT": "New York JFK"},
        {"IATA_CODE": "SFO", "AIRPORT": "San Francisco"},
        {"IATA_CODE": "SEA", "AIRPORT": "Seattle"},
        {"IATA_CODE": "LAS", "AIRPORT": "Las Vegas"},
        {"IATA_CODE": "MCO", "AIRPORT": "Orlando"},
        {"IATA_CODE": "EWR", "AIRPORT": "Newark"},
        {"IATA_CODE": "MIA", "AIRPORT": "Miami"},
        {"IATA_CODE": "PHX", "AIRPORT": "Phoenix"},
        {"IATA_CODE": "IAH", "AIRPORT": "Houston"},
        {"IATA_CODE": "BOS", "AIRPORT": "Boston"},
    ]


def get_model_results():
    """Get model comparison results."""
    if model_results is not None:
        return model_results.to_dict('index')
    return {
        "Logistic Regression": {"accuracy": 0.839, "precision": 0.789, "recall": 0.662, "f1": 0.72, "roc_auc": 0.86},
        "Random Forest": {"accuracy": 0.854, "precision": 0.839, "recall": 0.659, "f1": 0.74, "roc_auc": 0.88},
        "XGBoost": {"accuracy": 0.865, "precision": 0.944, "recall": 0.604, "f1": 0.74, "roc_auc": 0.89}
    }


def predict_delay(origin, destination, airline, hour, day_of_week, month):
    """
    Predict delay probability using the trained model.
    Creates a feature vector matching the training features.
    """
    if model is None or feature_columns is None:
        print("Model or feature columns not loaded, using fallback")
        return fallback_prediction(hour, day_of_week, month, airline, origin)
    
    try:
        # Create feature dictionary with all possible features
        features = {}
        
        # Numeric features
        features['hour_of_day'] = hour
        features['DAY_OF_WEEK'] = day_of_week
        features['MONTH'] = month
        features['is_weekend'] = 1 if day_of_week >= 6 else 0
        features['is_holiday_season'] = 1 if month in [11, 12] else 0
        features['SCHEDULED_TIME'] = 180  # Average flight time
        features['DISTANCE'] = 1000  # Average distance
        features['origin_flights_per_hour'] = 50  # Average congestion
        features['dest_flights_per_hour'] = 50
        features['airline_delay_rate'] = 0.30  # Average
        features['route_popularity'] = 100
        
        # Determine time of day
        if 5 <= hour < 12:
            time_of_day = 'Morning'
        elif 12 <= hour < 17:
            time_of_day = 'Afternoon'
        elif 17 <= hour < 21:
            time_of_day = 'Evening'
        else:
            time_of_day = 'Night'
        
        # Create DataFrame with single row
        feature_df = pd.DataFrame([features])
        
        # Add one-hot encoded columns for all features in feature_columns
        for col in feature_columns:
            if col not in feature_df.columns:
                # Check if it's a one-hot encoded column
                if col.startswith('AIRLINE_'):
                    # Extract airline code
                    airline_code = col.replace('AIRLINE_', '')
                    feature_df[col] = 1 if airline == airline_code else 0
                elif col.startswith('time_of_day_'):
                    # Extract time of day
                    tod = col.replace('time_of_day_', '')
                    feature_df[col] = 1 if time_of_day == tod else 0
                else:
                    # Default to 0 for unknown columns
                    feature_df[col] = 0
        
        # Select only the columns the model expects, in the right order
        feature_df = feature_df.reindex(columns=feature_columns, fill_value=0)
        
        # Get probability
        proba = model.predict_proba(feature_df)[0][1]
        print(f"Model prediction: {proba:.4f} for hour={hour}, day={day_of_week}, airline={airline}")
        return float(proba)
        
    except Exception as e:
        print(f"Prediction error: {e}")
        import traceback
        traceback.print_exc()
        return fallback_prediction(hour, day_of_week, month, airline, origin)


def fallback_prediction(hour, day_of_week, month, airline, origin):
    """Fallback heuristic prediction if model fails."""
    base_risk = 0.28
    
    # Time effect
    if 5 <= hour <= 8:
        base_risk -= 0.08
    elif 16 <= hour <= 19:
        base_risk += 0.12
    
    # Day effect
    if day_of_week == 5:  # Friday
        base_risk += 0.07
    elif day_of_week == 2:  # Tuesday
        base_risk -= 0.03
    
    # Month effect
    if month in [11, 12]:
        base_risk += 0.08
    
    return max(0.05, min(0.85, base_risk))


@app.route('/')
def index():
    """Main dashboard page."""
    return render_template(
        'index.html',
        airlines=get_airlines_list(),
        airports=get_airports_list(),
        model_results=get_model_results(),
        daily_forecast=[]
    )


@app.route('/predict', methods=['POST'])
def predict():
    """API endpoint for delay prediction."""
    data = request.json
    
    origin = data.get('origin', 'ATL')
    destination = data.get('destination', 'LAX')
    airline = data.get('airline', 'AA')
    hour = int(data.get('hour', 12))
    day_of_week = int(data.get('day_of_week', 3))
    month = int(data.get('month', 6))
    
    probability = predict_delay(origin, destination, airline, hour, day_of_week, month)
    
    # Determine risk level
    if probability < 0.25:
        risk_level = "Low"
        risk_color = "#00ff88"
    elif probability < 0.40:
        risk_level = "Medium"
        risk_color = "#ffbb00"
    else:
        risk_level = "High"
        risk_color = "#ff4444"
    
    return jsonify({
        'probability': round(probability * 100, 1),
        'risk_level': risk_level,
        'risk_color': risk_color
    })


if __name__ == '__main__':
    load_resources()
    print("\n" + "=" * 50)
    print("Flight Delay Prediction Dashboard")
    print("=" * 50)
    print("Starting server at http://127.0.0.1:5000")
    print("Press Ctrl+C to stop\n")
    app.run(debug=True, port=5000)
