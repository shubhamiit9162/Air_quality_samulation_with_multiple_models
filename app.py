from flask import Flask, render_template, request, jsonify, send_from_directory
from flask_cors import CORS
import os
import pandas as pd
import numpy as np
import json
from datetime import datetime, timedelta
import sqlite3
from werkzeug.utils import secure_filename
import traceback
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
import pickle
import warnings
warnings.filterwarnings('ignore')

app = Flask(__name__)
CORS(app)

# Configuration
app.config['SECRET_KEY'] = '12345678900987654321'
app.config['UPLOAD_FOLDER'] = 'uploads'
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # 16MB max file size

# Ensure upload directory exists
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)
os.makedirs('models', exist_ok=True)
os.makedirs('data', exist_ok=True)

# Global variables to store current model and data
current_model = None
current_scaler = None
current_data = None
feature_names = None

def init_db():
    """Initialize SQLite database for storing predictions history"""
    conn = sqlite3.connect('predictions.db')
    cursor = conn.cursor()
    
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS predictions (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            timestamp DATETIME DEFAULT CURRENT_TIMESTAMP,
            model_name TEXT,
            actual_value REAL,
            predicted_value REAL,
            features TEXT,
            accuracy REAL
        )
    ''')
    
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS model_performance (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            timestamp DATETIME DEFAULT CURRENT_TIMESTAMP,
            model_name TEXT,
            accuracy REAL,
            rmse REAL,
            mae REAL,
            r2_score REAL,
            training_time REAL,
            data_points INTEGER
        )
    ''')
    
    conn.commit()
    conn.close()

def load_and_preprocess_data(file_path):
    """Load and preprocess data similar to main.py"""
    try:
        data = pd.read_csv(file_path)
        
        if 'datetime' not in data.columns:
            return None, "CSV must contain 'datetime' column"
        
        data['datetime'] = pd.to_datetime(data['datetime'])
        data = data.sort_values('datetime')
        data.set_index('datetime', inplace=True)
        
        # Resample to 2-minute intervals
        data = data.resample('2min').mean()
        data.ffill(inplace=True)
        
        # Filter valid data
        data = data[(data['pm2.5'] > 0) & (data['temperature'] > 0) & (data['humidity'] > 0)]
        
        return data, None
    except Exception as e:
        return None, str(e)

def create_features(data):
    """Create advanced features similar to main.py"""
    enhanced_data = data.copy()
    
    # Time-based features
    enhanced_data['hour'] = enhanced_data.index.hour
    enhanced_data['minute'] = enhanced_data.index.minute
    enhanced_data['day_of_week'] = enhanced_data.index.dayofweek
    enhanced_data['is_weekend'] = (enhanced_data.index.dayofweek >= 5).astype(int)
    
    # Cyclical features
    enhanced_data['hour_sin'] = np.sin(2 * np.pi * enhanced_data['hour'] / 24)
    enhanced_data['hour_cos'] = np.cos(2 * np.pi * enhanced_data['hour'] / 24)
    enhanced_data['minute_sin'] = np.sin(2 * np.pi * enhanced_data['minute'] / 60)
    enhanced_data['minute_cos'] = np.cos(2 * np.pi * enhanced_data['minute'] / 60)
    
    # Interaction features
    enhanced_data['temp_humidity_ratio'] = enhanced_data['temperature'] / (enhanced_data['humidity'] + 1e-6)
    enhanced_data['temp_squared'] = enhanced_data['temperature'] ** 2
    enhanced_data['humidity_squared'] = enhanced_data['humidity'] ** 2
    enhanced_data['temp_humidity_product'] = enhanced_data['temperature'] * enhanced_data['humidity']
    
    # Lag features
    for lag in [1, 2, 3, 5, 10]:
        enhanced_data[f'pm2.5_lag_{lag}'] = enhanced_data['pm2.5'].shift(lag)
        enhanced_data[f'temp_lag_{lag}'] = enhanced_data['temperature'].shift(lag)
        enhanced_data[f'humidity_lag_{lag}'] = enhanced_data['humidity'].shift(lag)
    
    # Rolling features
    for window in [5, 10, 30]:
        enhanced_data[f'pm2.5_rolling_mean_{window}'] = enhanced_data['pm2.5'].rolling(window=window).mean()
        enhanced_data[f'pm2.5_rolling_std_{window}'] = enhanced_data['pm2.5'].rolling(window=window).std()
        enhanced_data[f'temp_rolling_mean_{window}'] = enhanced_data['temperature'].rolling(window=window).mean()
        enhanced_data[f'humidity_rolling_mean_{window}'] = enhanced_data['humidity'].rolling(window=window).mean()
    
    # Difference features
    enhanced_data['pm2.5_diff'] = enhanced_data['pm2.5'].diff()
    enhanced_data['temp_diff'] = enhanced_data['temperature'].diff()
    enhanced_data['humidity_diff'] = enhanced_data['humidity'].diff()
    
    enhanced_data.dropna(inplace=True)
    return enhanced_data

@app.route('/')
def dashboard():
    """Main dashboard route"""
    return render_template('dashboard.html')

@app.route('/api/upload', methods=['POST'])
def upload_file():
    """Handle file upload and initial processing"""
    global current_data, current_model, current_scaler, feature_names
    
    try:
        if 'file' not in request.files:
            return jsonify({'error': 'No file uploaded'}), 400
        
        file = request.files['file']
        if file.filename == '':
            return jsonify({'error': 'No file selected'}), 400
        
        if file and file.filename.endswith('.csv'):
            filename = secure_filename(file.filename)
            filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            file.save(filepath)
            
            # Process the uploaded file
            data, error = load_and_preprocess_data(filepath)
            if error:
                return jsonify({'error': error}), 400
            
            # Create features
            enhanced_data = create_features(data)
            current_data = enhanced_data
            
            # Prepare basic statistics
            stats = {
                'total_points': len(enhanced_data),
                'date_range': {
                    'start': enhanced_data.index.min().isoformat(),
                    'end': enhanced_data.index.max().isoformat()
                },
                'columns': list(enhanced_data.columns),
                'pm25_stats': {
                    'mean': float(enhanced_data['pm2.5'].mean()),
                    'std': float(enhanced_data['pm2.5'].std()),
                    'min': float(enhanced_data['pm2.5'].min()),
                    'max': float(enhanced_data['pm2.5'].max())
                }
            }
            
            return jsonify({
                'message': 'File uploaded and processed successfully',
                'stats': stats
            })
        
        return jsonify({'error': 'Please upload a CSV file'}), 400
        
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/api/train_model', methods=['POST'])
def train_model():
    """Train machine learning model"""
    global current_model, current_scaler, current_data, feature_names
    
    try:
        if current_data is None:
            return jsonify({'error': 'No data available. Please upload a CSV file first.'}), 400
        
        data = request.get_json()
        model_type = data.get('model_type', 'RandomForest')
        
        # Prepare features and target
        feature_cols = [col for col in current_data.columns if col != 'pm2.5']
        X = current_data[feature_cols]
        y = current_data['pm2.5']
        
        # Split data (80% train, 20% test)
        split_point = int(len(current_data) * 0.8)
        X_train, X_test = X.iloc[:split_point], X.iloc[split_point:]
        y_train, y_test = y.iloc[:split_point], y.iloc[split_point:]
        
        # Scale features
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)
        
        # Train model
        if model_type == 'RandomForest':
            model = RandomForestRegressor(
                n_estimators=200,
                max_depth=15,
                min_samples_split=5,
                random_state=42,
                n_jobs=-1
            )
            model.fit(X_train, y_train)
            predictions = model.predict(X_test)
        else:
            # Default to Random Forest
            model = RandomForestRegressor(n_estimators=100, random_state=42)
            model.fit(X_train_scaled, y_train)
            predictions = model.predict(X_test_scaled)
        
        # Calculate metrics
        mse = mean_squared_error(y_test, predictions)
        rmse = np.sqrt(mse)
        mae = mean_absolute_error(y_test, predictions)
        r2 = r2_score(y_test, predictions)
        accuracy = max(0, (1 - rmse / np.mean(y_test)) * 100)
        
        # Store model and scaler
        current_model = model
        current_scaler = scaler
        feature_names = feature_cols
        
        # Save model to disk
        model_path = f'models/{model_type.lower()}_model.pkl'
        scaler_path = f'models/{model_type.lower()}_scaler.pkl'
        
        with open(model_path, 'wb') as f:
            pickle.dump(model, f)
        with open(scaler_path, 'wb') as f:
            pickle.dump(scaler, f)
        
        # Store performance in database
        conn = sqlite3.connect('predictions.db')
        cursor = conn.cursor()
        cursor.execute('''
            INSERT INTO model_performance 
            (model_name, accuracy, rmse, mae, r2_score, data_points)
            VALUES (?, ?, ?, ?, ?, ?)
        ''', (model_type, accuracy, rmse, mae, r2, len(current_data)))
        conn.commit()
        conn.close()
        
        return jsonify({
            'message': 'Model trained successfully',
            'performance': {
                'accuracy': accuracy,
                'rmse': rmse,
                'mae': mae,
                'r2_score': r2,
                'mse': mse
            },
            'model_type': model_type
        })
        
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/api/predict', methods=['POST'])
def predict():
    """Make predictions using trained model"""
    global current_model, current_scaler, feature_names
    
    try:
        if current_model is None:
            return jsonify({'error': 'No trained model available'}), 400
        
        data = request.get_json()
        
        # Extract features from request
        features = {}
        for feature in feature_names:
            if feature in data:
                features[feature] = data[feature]
            else:
                # Use default values for missing features
                features[feature] = 0
        
        # Create feature array
        feature_array = np.array([list(features.values())])
        
        # Scale features if scaler is available
        if current_scaler is not None:
            feature_array = current_scaler.transform(feature_array)
        
        # Make prediction
        prediction = current_model.predict(feature_array)[0]
        
        return jsonify({
            'prediction': float(prediction),
            'features_used': features
        })
        
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/api/realtime_data')
def get_realtime_data():
    """Get recent data for real-time visualization"""
    global current_data
    
    try:
        if current_data is None:
            return jsonify({'error': 'No data available'}), 400
        
        # Get last 100 data points
        recent_data = current_data.tail(100)
        
        data_points = []
        for idx, row in recent_data.iterrows():
            data_points.append({
                'timestamp': idx.isoformat(),
                'pm25': float(row['pm2.5']),
                'temperature': float(row['temperature']),
                'humidity': float(row['humidity'])
            })
        
        return jsonify({
            'data': data_points,
            'stats': {
                'avg_pm25': float(recent_data['pm2.5'].mean()),
                'max_pm25': float(recent_data['pm2.5'].max()),
                'min_pm25': float(recent_data['pm2.5'].min())
            }
        })
        
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/api/model_performance')
def get_model_performance():
    """Get historical model performance data"""
    try:
        conn = sqlite3.connect('predictions.db')
        cursor = conn.cursor()
        
        cursor.execute('''
            SELECT model_name, accuracy, rmse, mae, r2_score, timestamp, data_points
            FROM model_performance
            ORDER BY timestamp DESC
            LIMIT 10
        ''')
        
        rows = cursor.fetchall()
        conn.close()
        
        performance_data = []
        for row in rows:
            performance_data.append({
                'model_name': row[0],
                'accuracy': row[1],
                'rmse': row[2],
                'mae': row[3],
                'r2_score': row[4],
                'timestamp': row[5],
                'data_points': row[6]
            })
        
        return jsonify({'performance_history': performance_data})
        
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/api/feature_importance')
def get_feature_importance():
    """Get feature importance from trained model"""
    global current_model, feature_names
    
    try:
        if current_model is None or not hasattr(current_model, 'feature_importances_'):
            return jsonify({'error': 'No model with feature importance available'}), 400
        
        importance_data = []
        for i, importance in enumerate(current_model.feature_importances_):
            importance_data.append({
                'feature': feature_names[i],
                'importance': float(importance)
            })
        
        # Sort by importance
        importance_data.sort(key=lambda x: x['importance'], reverse=True)
        
        return jsonify({'feature_importance': importance_data[:20]})  # Top 20 features
        
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/api/data_summary')
def get_data_summary():
    """Get comprehensive data summary"""
    global current_data
    
    try:
        if current_data is None:
            return jsonify({'error': 'No data available'}), 400
        
        summary = {
            'total_records': len(current_data),
            'date_range': {
                'start': current_data.index.min().isoformat(),
                'end': current_data.index.max().isoformat()
            },
            'pm25_distribution': {
                'mean': float(current_data['pm2.5'].mean()),
                'median': float(current_data['pm2.5'].median()),
                'std': float(current_data['pm2.5'].std()),
                'min': float(current_data['pm2.5'].min()),
                'max': float(current_data['pm2.5'].max()),
                'q25': float(current_data['pm2.5'].quantile(0.25)),
                'q75': float(current_data['pm2.5'].quantile(0.75))
            },
            'correlations': {}
        }
        
        # Calculate correlations with PM2.5
        numeric_cols = current_data.select_dtypes(include=[np.number]).columns
        for col in numeric_cols:
            if col != 'pm2.5':
                corr = current_data['pm2.5'].corr(current_data[col])
                if not np.isnan(corr):
                    summary['correlations'][col] = float(corr)
        
        return jsonify(summary)
        
    except Exception as e:
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    # Initialize database
    init_db()
    
    # Run the Flask app
    app.run(debug=True, host='0.0.0.0', port=5000)