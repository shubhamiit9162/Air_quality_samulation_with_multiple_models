import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
from sklearn.preprocessing import StandardScaler
import warnings
import time
import joblib
from datetime import datetime

warnings.filterwarnings('ignore')

def load_and_preprocess(csv_path):
    """Load and preprocess data with enhanced feature engineering"""
    data = pd.read_csv(csv_path)
     
    data['datetime'] = pd.to_datetime(data['datetime'])
    
    data = data.sort_values('datetime')
     
    data.set_index('datetime', inplace=True)
    
    data = data.resample('2min').mean()
    
    data.ffill(inplace=True)
    
    data = data[(data['pm2.5'] > 0) & (data['temperature'] > 0) & (data['humidity'] > 0)]
    
    return data

def create_advanced_features(data):
    """Create advanced features for better prediction"""
    
    enhanced_data = data.copy()
    
    enhanced_data['hour'] = enhanced_data.index.hour
    enhanced_data['minute'] = enhanced_data.index.minute
    enhanced_data['day_of_week'] = enhanced_data.index.dayofweek
    enhanced_data['is_weekend'] = (enhanced_data.index.dayofweek >= 5).astype(int)
    
    enhanced_data['hour_sin'] = np.sin(2 * np.pi * enhanced_data['hour'] / 24)
    enhanced_data['hour_cos'] = np.cos(2 * np.pi * enhanced_data['hour'] / 24)
    enhanced_data['minute_sin'] = np.sin(2 * np.pi * enhanced_data['minute'] / 60)
    enhanced_data['minute_cos'] = np.cos(2 * np.pi * enhanced_data['minute'] / 60)
    
    enhanced_data['temp_humidity_ratio'] = enhanced_data['temperature'] / (enhanced_data['humidity'] + 1e-6)
    enhanced_data['temp_squared'] = enhanced_data['temperature'] ** 2
    enhanced_data['humidity_squared'] = enhanced_data['humidity'] ** 2
    enhanced_data['temp_humidity_product'] = enhanced_data['temperature'] * enhanced_data['humidity']
    
    for lag in [1, 2, 3, 5, 10]:
        enhanced_data[f'pm2.5_lag_{lag}'] = enhanced_data['pm2.5'].shift(lag)
        enhanced_data[f'temp_lag_{lag}'] = enhanced_data['temperature'].shift(lag)
        enhanced_data[f'humidity_lag_{lag}'] = enhanced_data['humidity'].shift(lag)
    
    for window in [5, 10, 30]:
        enhanced_data[f'pm2.5_rolling_mean_{window}'] = enhanced_data['pm2.5'].rolling(window=window).mean()
        enhanced_data[f'pm2.5_rolling_std_{window}'] = enhanced_data['pm2.5'].rolling(window=window).std()
        enhanced_data[f'temp_rolling_mean_{window}'] = enhanced_data['temperature'].rolling(window=window).mean()
        enhanced_data[f'humidity_rolling_mean_{window}'] = enhanced_data['humidity'].rolling(window=window).mean()
    
    enhanced_data['pm2.5_diff'] = enhanced_data['pm2.5'].diff()
    enhanced_data['temp_diff'] = enhanced_data['temperature'].diff()
    enhanced_data['humidity_diff'] = enhanced_data['humidity'].diff()
    
    enhanced_data['pm2.5_ma_diff_5_10'] = (enhanced_data['pm2.5_rolling_mean_5'] - 
                                          enhanced_data['pm2.5_rolling_mean_10'])
    
    enhanced_data.dropna(inplace=True)
    
    return enhanced_data

def split_data_flexible(data):
    """Flexible data splitting that works with any amount of data"""
    unique_dates = data.index.normalize().unique()
    print(f"Available days in dataset: {len(unique_dates)}")
    
    if len(unique_dates) >= 3:
        print("Using day-based split (2 days training, 1 day testing)")
        day1 = unique_dates[0]
        day2 = unique_dates[1]
        day3 = unique_dates[2]

        train_data = data[(data.index.normalize() == day1) | (data.index.normalize() == day2)]
        test_data = data[data.index.normalize() == day3]
        
        return train_data, test_data, "day-based"
    
    elif len(unique_dates) >= 2:
        print("Using day-based split (1 day training, 1 day testing)")
        day1 = unique_dates[0]
        day2 = unique_dates[1]

        train_data = data[data.index.normalize() == day1]
        test_data = data[data.index.normalize() == day2]
        
        return train_data, test_data, "day-based"
    
    else:
        print("Using temporal split (85% training, 15% testing)")
        split_point = int(len(data) * 0.85)
        train_data = data.iloc[:split_point]
        test_data = data.iloc[split_point:]
        
        return train_data, test_data, "temporal"

def prepare_features_targets(train_data, test_data):
    """Prepare features and targets with all engineered features"""
    feature_cols = [col for col in train_data.columns if col != 'pm2.5']
    
    X_train = train_data[feature_cols]
    y_train = train_data['pm2.5']
    
    X_test = test_data[feature_cols]
    y_test = test_data['pm2.5']
    
    print(f"Number of features: {len(feature_cols)}")
    print(f"Features: {feature_cols[:10]}..." if len(feature_cols) > 10 else f"Features: {feature_cols}")
    
    return X_train, y_train, X_test, y_test

def train_linear_regression(X_train, y_train):
    """Train Linear Regression model with scaling"""
    print("Training Linear Regression...")
    start_time = time.time()
    
    # Scale the features
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    
    # Train the model
    lr_model = LinearRegression()
    lr_model.fit(X_train_scaled, y_train)
    
    end_time = time.time()
    training_time = end_time - start_time
    
    print(f"Linear Regression training completed in {training_time:.4f} seconds")
    
    # Save the model and scaler
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    os.makedirs("models", exist_ok=True)
    
    model_path = f"models/pm25_linear_regression_{timestamp}.pkl"
    scaler_path = f"models/pm25_scaler_{timestamp}.pkl"
    
    joblib.dump(lr_model, model_path)
    joblib.dump(scaler, scaler_path)
    
    print(f"✅ Model saved to: {model_path}")
    print(f"✅ Scaler saved to: {scaler_path}")
    
    return lr_model, scaler, training_time

def predict_and_evaluate(model, scaler, X_test, y_test):
    """Predict and evaluate Linear Regression model"""
    start_time = time.time()
    
    # Scale test features
    X_test_scaled = scaler.transform(X_test)
    
    # Make predictions
    predictions = model.predict(X_test_scaled)
    
    end_time = time.time()
    prediction_time = end_time - start_time
    
    # Calculate metrics
    mse = mean_squared_error(y_test, predictions)
    rmse = np.sqrt(mse)
    mae = mean_absolute_error(y_test, predictions)
    r2 = r2_score(y_test, predictions)
    
    accuracy = max(0, (1 - rmse / np.mean(y_test)) * 100)
    
    results = {
        'predictions': predictions,
        'mse': mse,
        'rmse': rmse,
        'mae': mae,
        'r2': r2,
        'accuracy': accuracy,
        'prediction_time': prediction_time
    }
    
    return results

def plot_results(test_data, predictions, accuracy, split_type):
    """Plot Linear Regression results"""
    plt.figure(figsize=(14, 8))
    
    # Main prediction plot
    plt.subplot(2, 1, 1)
    plt.plot(test_data.index, test_data['pm2.5'], label='Actual PM2.5', linewidth=2, color='black')
    plt.plot(test_data.index, predictions, 
            label=f'Linear Regression (Acc: {accuracy:.1f}%)', 
            linestyle='--', alpha=0.8, color='red')
    
    plt.title(f'PM2.5 Prediction - Linear Regression ({split_type} split)')
    plt.xlabel('Datetime')
    plt.ylabel('PM2.5 Level')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.xticks(rotation=45)
    
    # Residual plot
    plt.subplot(2, 1, 2)
    residuals = test_data['pm2.5'] - predictions
    plt.scatter(predictions, residuals, alpha=0.6)
    plt.axhline(y=0, color='red', linestyle='--')
    plt.title('Residual Plot - Linear Regression')
    plt.xlabel('Predicted PM2.5')
    plt.ylabel('Residuals')
    plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.show()

def analyze_data(data):
    """Analyze the dataset to understand its structure"""
    print("\n=== Data Analysis ===")
    print(f"Total data points: {len(data)}")
    print(f"Date range: {data.index.min()} to {data.index.max()}")
    print(f"Duration: {(data.index.max() - data.index.min()).days + 1} days")
    print(f"Missing values: {data.isnull().sum().sum()}")
    
    print("\nBasic Statistics:")
    print(data.describe())
    
    print(f"\nAvailable columns: {list(data.columns)}")

def print_performance_summary(results, training_time):
    """Print Linear Regression performance summary"""
    print("\n" + "="*60)
    print("LINEAR REGRESSION MODEL PERFORMANCE SUMMARY")
    print("="*60)
    
    print(f"Accuracy: {results['accuracy']:.2f}%")
    print(f"R² Score: {results['r2']:.4f}")
    print(f"RMSE: {results['rmse']:.4f}")
    print(f"MAE: {results['mae']:.4f}")
    print(f"MSE: {results['mse']:.4f}")
    print(f"Training Time: {training_time:.4f} seconds")
    print(f"Prediction Time: {results['prediction_time']:.6f} seconds")
    
    if results['accuracy'] >= 85:
        print("\n🎉 TARGET ACHIEVED! Accuracy is 85% or above!")
    else:
        print(f"\n⚠️  Accuracy is {results['accuracy']:.2f}%. Consider more data or feature engineering.")
    
    print("="*60)

def load_saved_model(model_path, scaler_path):
    """Load a previously saved model and scaler"""
    model = joblib.load(model_path)
    scaler = joblib.load(scaler_path)
    
    print(f"Model loaded from: {model_path}")
    print(f"Scaler loaded from: {scaler_path}")
    
    return model, scaler    

def main():
    print("Linear Regression PM2.5 Prediction Model")
    print("Current Working Directory:", os.getcwd())
    csv_path = 'data/UK0010_PM46_2986N_7790E_2024-09-27.csv'
    
    try:
        print("Loading and preprocessing data...")
        data = load_and_preprocess(csv_path)
        
        analyze_data(data)
        
        if len(data) == 0:
            print("Error: No valid data found after preprocessing!")
            return
        
        print("\nCreating advanced features...")
        enhanced_data = create_advanced_features(data)
        print(f"Enhanced data shape: {enhanced_data.shape}")
        
        print("\nSplitting data...")
        train_data, test_data, split_type = split_data_flexible(enhanced_data)
        
        print(f"Training data: {len(train_data)} points")
        print(f"Testing data: {len(test_data)} points")
        
        if len(train_data) == 0 or len(test_data) == 0:
            print("Error: Insufficient data for training or testing!")
            return
        
        print("Preparing features and target variables...")
        X_train, y_train, X_test, y_test = prepare_features_targets(train_data, test_data)
        
        print("Training Linear Regression model...")
        model, scaler, training_time = train_linear_regression(X_train, y_train)
        
        print("Predicting and evaluating...")
        results = predict_and_evaluate(model, scaler, X_test, y_test)
        
        # Print performance summary
        print_performance_summary(results, training_time)
        
        print("\nPlotting results...")
        plot_results(test_data, results['predictions'], results['accuracy'], split_type)
        
        print("\n" + "="*50)
        print("MODEL SUCCESSFULLY SAVED!")
        print("You can now use the saved model for future predictions.")
        print("="*50)
        
    except FileNotFoundError:
        print(f"Error: Could not find the CSV file at {csv_path}")
        print("Please check if the file exists and the path is correct.")
    except Exception as e:
        print(f"An error occurred: {str(e)}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()