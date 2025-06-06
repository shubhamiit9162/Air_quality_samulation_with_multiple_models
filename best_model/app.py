from flask import Flask, request, jsonify, render_template
import joblib
import pandas as pd
import numpy as np
from datetime import datetime
import os
import json

app = Flask(__name__)

class PM25Predictor:
    def __init__(self):
        self.model = None
        self.scaler = None
        self.feature_columns = None
        self.model_loaded = False
        
    def load_latest_model(self):
        """Load the latest trained model and scaler"""
        try:
            models_dir = "models"
            if not os.path.exists(models_dir):
                return False
                
            # Find the latest model files
            model_files = [f for f in os.listdir(models_dir) if f.startswith('pm25_linear_regression_')]
            scaler_files = [f for f in os.listdir(models_dir) if f.startswith('pm25_scaler_')]
            
            if not model_files or not scaler_files:
                return False
                
            # Get the latest files (assuming timestamp in filename)
            latest_model = sorted(model_files)[-1]
            latest_scaler = sorted(scaler_files)[-1]
            
            model_path = os.path.join(models_dir, latest_model)
            scaler_path = os.path.join(models_dir, latest_scaler)
            
            # Load model and scaler
            self.model = joblib.load(model_path)
            self.scaler = joblib.load(scaler_path)
            
            print(f"Model loaded: {model_path}")
            print(f"Scaler loaded: {scaler_path}")
            
            self.model_loaded = True
            return True
            
        except Exception as e:
            print(f"Error loading model: {str(e)}")
            return False
    
    def create_features(self, temperature, humidity, datetime_str=None):
        """Create features similar to training data"""
        if datetime_str is None:
            datetime_str = datetime.now().isoformat()
            
        dt = pd.to_datetime(datetime_str)
        
        # Basic features
        features = {
            'temperature': temperature,
            'humidity': humidity,
            'hour': dt.hour,
            'minute': dt.minute,
            'day_of_week': dt.dayofweek,
            'is_weekend': 1 if dt.dayofweek >= 5 else 0,
            'hour_sin': np.sin(2 * np.pi * dt.hour / 24),
            'hour_cos': np.cos(2 * np.pi * dt.hour / 24),
            'minute_sin': np.sin(2 * np.pi * dt.minute / 60),
            'minute_cos': np.cos(2 * np.pi * dt.minute / 60),
            'temp_humidity_ratio': temperature / (humidity + 1e-6),
            'temp_squared': temperature ** 2,
            'humidity_squared': humidity ** 2,
            'temp_humidity_product': temperature * humidity
        }
        
        # For lag and rolling features, we'll use current values as approximation
        # In a real scenario, you'd maintain a history buffer
        for lag in [1, 2, 3, 5, 10]:
            features[f'pm2.5_lag_{lag}'] = 50.0  # Default PM2.5 value
            features[f'temp_lag_{lag}'] = temperature
            features[f'humidity_lag_{lag}'] = humidity
        
        for window in [5, 10, 30]:
            features[f'pm2.5_rolling_mean_{window}'] = 50.0
            features[f'pm2.5_rolling_std_{window}'] = 10.0
            features[f'temp_rolling_mean_{window}'] = temperature
            features[f'humidity_rolling_mean_{window}'] = humidity
        
        # Difference features (using zero as default)
        features['pm2.5_diff'] = 0.0
        features['temp_diff'] = 0.0
        features['humidity_diff'] = 0.0
        features['pm2.5_ma_diff_5_10'] = 0.0
        
        return features
    
    def predict(self, temperature, humidity, datetime_str=None):
        """Make PM2.5 prediction"""
        if not self.model_loaded:
            return None, "Model not loaded"
            
        try:
            # Create features
            features = self.create_features(temperature, humidity, datetime_str)
            
            # Convert to DataFrame
            feature_df = pd.DataFrame([features])
            
            # Scale features
            features_scaled = self.scaler.transform(feature_df)
            
            # Make prediction
            prediction = self.model.predict(features_scaled)[0]
            
            return max(0, prediction), None  # Ensure non-negative prediction
            
        except Exception as e:
            return None, f"Prediction error: {str(e)}"

# Initialize predictor
predictor = PM25Predictor()

@app.route('/')
def dashboard():
    """Main dashboard page"""
    return render_template('dashboard.html')

@app.route('/api/predict', methods=['POST'])
def predict_pm25():
    """API endpoint for PM2.5 prediction"""
    try:
        data = request.get_json()
        
        # Validate input
        if not data:
            return jsonify({'error': 'No data provided'}), 400
            
        temperature = data.get('temperature')
        humidity = data.get('humidity')
        datetime_str = data.get('datetime')
        
        if temperature is None or humidity is None:
            return jsonify({'error': 'Temperature and humidity are required'}), 400
            
        # Make prediction
        prediction, error = predictor.predict(temperature, humidity, datetime_str)
        
        if error:
            return jsonify({'error': error}), 500
            
        # Determine air quality level
        if prediction <= 12:
            quality = "Good"
            color = "green"
        elif prediction <= 35:
            quality = "Moderate"
            color = "yellow"
        elif prediction <= 55:
            quality = "Unhealthy for Sensitive Groups"
            color = "orange"
        elif prediction <= 150:
            quality = "Unhealthy"
            color = "red"
        else:
            quality = "Very Unhealthy"
            color = "purple"
        
        response = {
            'pm25_prediction': round(prediction, 2),
            'air_quality': quality,
            'color': color,
            'timestamp': datetime.now().isoformat(),
            'input': {
                'temperature': temperature,
                'humidity': humidity,
                'datetime': datetime_str or datetime.now().isoformat()
            }
        }
        
        return jsonify(response)
        
    except Exception as e:
        return jsonify({'error': f'Server error: {str(e)}'}), 500

@app.route('/api/batch_predict', methods=['POST'])
def batch_predict():
    """API endpoint for batch predictions"""
    try:
        data = request.get_json()
        
        if not data or 'predictions' not in data:
            return jsonify({'error': 'Invalid data format'}), 400
            
        results = []
        
        for item in data['predictions']:
            temperature = item.get('temperature')
            humidity = item.get('humidity')
            datetime_str = item.get('datetime')
            
            if temperature is None or humidity is None:
                results.append({'error': 'Missing temperature or humidity'})
                continue
                
            prediction, error = predictor.predict(temperature, humidity, datetime_str)
            
            if error:
                results.append({'error': error})
            else:
                results.append({
                    'pm25_prediction': round(prediction, 2),
                    'temperature': temperature,
                    'humidity': humidity,
                    'datetime': datetime_str or datetime.now().isoformat()
                })
        
        return jsonify({'results': results})
        
    except Exception as e:
        return jsonify({'error': f'Server error: {str(e)}'}), 500

@app.route('/api/model_info')
def model_info():
    """Get information about the loaded model"""
    if not predictor.model_loaded:
        return jsonify({'error': 'Model not loaded'}), 500
        
    return jsonify({
        'model_loaded': True,
        'model_type': 'Linear Regression',
        'features_count': len(predictor.scaler.feature_names_in_) if hasattr(predictor.scaler, 'feature_names_in_') else 'Unknown',
        'status': 'Ready for predictions'
    })

@app.route('/api/health')
def health_check():
    """Health check endpoint"""
    return jsonify({
        'status': 'healthy',
        'model_loaded': predictor.model_loaded,
        'timestamp': datetime.now().isoformat()
    })

if __name__ == '__main__':
    print("Starting PM2.5 Prediction API...")
    
    # Load the model
    if predictor.load_latest_model():
        print("✅ Model loaded successfully!")
    else:
        print("⚠️  Warning: Could not load model. Please train the model first using main.py")
    
    # Create templates directory if it doesn't exist
    os.makedirs('templates', exist_ok=True)
    
    print("Server starting on http://localhost:5000")
    app.run(debug=True, host='0.0.0.0', port=5000)