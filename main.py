import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVR
import warnings
import time
import psutil
import tracemalloc
from memory_profiler import profile
warnings.filterwarnings('ignore')

try:
    from tensorflow.keras.models import Sequential
    from tensorflow.keras.layers import Dense, Dropout
    from tensorflow.keras.optimizers import Adam
    TENSORFLOW_AVAILABLE = True
except ImportError:
    TENSORFLOW_AVAILABLE = False
    print("TensorFlow not available. ANN model will be skipped.")

def get_memory_usage():
    """Get current memory usage in MB"""
    process = psutil.Process()
    return process.memory_info().rss / 1024 / 1024  # Convert bytes to MB

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

def train_models(X_train, y_train):
    """Train multiple models with improved configurations and track time/memory"""
    models = {}
    performance_metrics = {}
    
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    
    # Linear Regression
    print("Training Linear Regression...")
    start_time = time.time()
    start_memory = get_memory_usage()
    tracemalloc.start()
    
    lr_model = LinearRegression()
    lr_model.fit(X_train_scaled, y_train)
    
    end_time = time.time()
    end_memory = get_memory_usage()
    current, peak = tracemalloc.get_traced_memory()
    tracemalloc.stop()
    
    models['Linear Regression'] = {'model': lr_model, 'scaler': scaler, 'scaled': True}
    performance_metrics['Linear Regression'] = {
        'training_time': end_time - start_time,
        'memory_used': end_memory - start_memory,
        'peak_memory': peak / 1024 / 1024  # Convert to MB
    }
    
    # Random Forest
    print("Training Random Forest...")
    start_time = time.time()
    start_memory = get_memory_usage()
    tracemalloc.start()
    
    rf_model = RandomForestRegressor(
        n_estimators=200, 
        max_depth=15, 
        min_samples_split=5,
        min_samples_leaf=2,
        random_state=42,
        n_jobs=-1
    )
    rf_model.fit(X_train, y_train)
    
    end_time = time.time()
    end_memory = get_memory_usage()
    current, peak = tracemalloc.get_traced_memory()
    tracemalloc.stop()
    
    models['Random Forest'] = {'model': rf_model, 'scaler': None, 'scaled': False}
    performance_metrics['Random Forest'] = {
        'training_time': end_time - start_time,
        'memory_used': end_memory - start_memory,
        'peak_memory': peak / 1024 / 1024
    }
    
    # Gradient Boosting
    print("Training Gradient Boosting...")
    start_time = time.time()
    start_memory = get_memory_usage()
    tracemalloc.start()
    
    gb_model = GradientBoostingRegressor(
        n_estimators=200,
        learning_rate=0.1,
        max_depth=8,
        min_samples_split=5,
        min_samples_leaf=2,
        random_state=42
    )
    gb_model.fit(X_train, y_train)
    
    end_time = time.time()
    end_memory = get_memory_usage()
    current, peak = tracemalloc.get_traced_memory()
    tracemalloc.stop()
    
    models['Gradient Boosting'] = {'model': gb_model, 'scaler': None, 'scaled': False}
    performance_metrics['Gradient Boosting'] = {
        'training_time': end_time - start_time,
        'memory_used': end_memory - start_memory,
        'peak_memory': peak / 1024 / 1024
    }
    
    # SVR
    print("Training SVR...")
    start_time = time.time()
    start_memory = get_memory_usage()
    tracemalloc.start()
    
    svr_model = SVR(kernel='rbf', C=100, gamma='scale', epsilon=0.1)
    svr_model.fit(X_train_scaled, y_train)
    
    end_time = time.time()
    end_memory = get_memory_usage()
    current, peak = tracemalloc.get_traced_memory()
    tracemalloc.stop()
    
    models['SVR'] = {'model': svr_model, 'scaler': scaler, 'scaled': True}
    performance_metrics['SVR'] = {
        'training_time': end_time - start_time,
        'memory_used': end_memory - start_memory,
        'peak_memory': peak / 1024 / 1024
    }
    
    # ANN
    if TENSORFLOW_AVAILABLE:
        print("Training ANN...")
        start_time = time.time()
        start_memory = get_memory_usage()
        tracemalloc.start()
        
        ann_model = Sequential([
            Dense(128, input_dim=X_train.shape[1], activation='relu'),
            Dropout(0.3),
            Dense(64, activation='relu'),
            Dropout(0.2),
            Dense(32, activation='relu'),
            Dropout(0.1),
            Dense(16, activation='relu'),
            Dense(1)
        ])
        
        ann_model.compile(
            optimizer=Adam(learning_rate=0.001),
            loss='mean_squared_error',
            metrics=['mae']
        )
        
        ann_model.fit(
            X_train_scaled, y_train,
            epochs=200,
            batch_size=32,
            validation_split=0.2,
            verbose=0
        )
        
        end_time = time.time()
        end_memory = get_memory_usage()
        current, peak = tracemalloc.get_traced_memory()
        tracemalloc.stop()
        
        models['ANN'] = {'model': ann_model, 'scaler': scaler, 'scaled': True}
        performance_metrics['ANN'] = {
            'training_time': end_time - start_time,
            'memory_used': end_memory - start_memory,
            'peak_memory': peak / 1024 / 1024
        }
    
    return models, performance_metrics

def predict_and_evaluate(models, X_test, y_test):
    """Predict and evaluate all models with time tracking"""
    results = {}
    prediction_times = {}
    
    for model_name, model_info in models.items():
        model = model_info['model']
        scaler = model_info['scaler']
        needs_scaling = model_info['scaled']
        
        # Track prediction time
        start_time = time.time()
        
        if needs_scaling and scaler is not None:
            X_test_processed = scaler.transform(X_test)
        else:
            X_test_processed = X_test
        
        if model_name == 'ANN' and TENSORFLOW_AVAILABLE:
            predictions = model.predict(X_test_processed, verbose=0).flatten()
        else:
            predictions = model.predict(X_test_processed)
        
        end_time = time.time()
        prediction_times[model_name] = end_time - start_time
        
        mse = mean_squared_error(y_test, predictions)
        rmse = np.sqrt(mse)
        mae = mean_absolute_error(y_test, predictions)
        r2 = r2_score(y_test, predictions)
        
        accuracy = max(0, (1 - rmse / np.mean(y_test)) * 100)
        
        results[model_name] = {
            'predictions': predictions,
            'mse': mse,
            'rmse': rmse,
            'mae': mae,
            'r2': r2,
            'accuracy': accuracy,
            'prediction_time': prediction_times[model_name]
        }
    
    return results

def plot_results(test_data, results, split_type):
    """Plot results with improved visualization"""
    plt.figure(figsize=(16, 10))
    
    
    plt.subplot(2, 1, 1)
    plt.plot(test_data.index, test_data['pm2.5'], label='Actual PM2.5', linewidth=2, color='black')
    
    colors = ['red', 'blue', 'green', 'orange', 'purple']
    for i, (model_name, result) in enumerate(results.items()):
        plt.plot(test_data.index, result['predictions'], 
                label=f'{model_name} (Acc: {result["accuracy"]:.1f}%)', 
                linestyle='--', alpha=0.8, color=colors[i % len(colors)])
    
    plt.title(f'PM2.5 Prediction Comparison ({split_type} split)')
    plt.xlabel('Datetime')
    plt.ylabel('PM2.5 Level')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.xticks(rotation=45)
    
    
    best_model = max(results, key=lambda k: results[k]['accuracy'])
    plt.subplot(2, 1, 2)
    residuals = test_data['pm2.5'] - results[best_model]['predictions']
    plt.scatter(results[best_model]['predictions'], residuals, alpha=0.6)
    plt.axhline(y=0, color='red', linestyle='--')
    plt.title(f'Residual Plot - {best_model}')
    plt.xlabel('Predicted PM2.5')
    plt.ylabel('Residuals')
    plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.show()

def plot_performance_comparison(performance_metrics, results):
    """Plot performance comparison including time and memory"""
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 10))
    
    models = list(performance_metrics.keys())
    
    # Training Time
    training_times = [performance_metrics[model]['training_time'] for model in models]
    ax1.bar(models, training_times, color='skyblue', alpha=0.7)
    ax1.set_title('Training Time Comparison')
    ax1.set_ylabel('Time (seconds)')
    ax1.tick_params(axis='x', rotation=45)
    
    # Memory Usage
    memory_usage = [performance_metrics[model]['memory_used'] for model in models]
    ax2.bar(models, memory_usage, color='lightgreen', alpha=0.7)
    ax2.set_title('Memory Usage Comparison')
    ax2.set_ylabel('Memory (MB)')
    ax2.tick_params(axis='x', rotation=45)
    
    # Prediction Time
    prediction_times = [results[model]['prediction_time'] for model in models]
    ax3.bar(models, prediction_times, color='orange', alpha=0.7)
    ax3.set_title('Prediction Time Comparison')
    ax3.set_ylabel('Time (seconds)')
    ax3.tick_params(axis='x', rotation=45)
    
    # Accuracy
    accuracies = [results[model]['accuracy'] for model in models]
    ax4.bar(models, accuracies, color='lightcoral', alpha=0.7)
    ax4.set_title('Accuracy Comparison')
    ax4.set_ylabel('Accuracy (%)')
    ax4.tick_params(axis='x', rotation=45)
    
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

def print_feature_importance(models, feature_names):
    """Print feature importance for tree-based models"""
    print("\n=== Feature Importance ===")
    
    for model_name, model_info in models.items():
        if hasattr(model_info['model'], 'feature_importances_'):
            importance = model_info['model'].feature_importances_
            feature_imp = pd.DataFrame({
                'feature': feature_names,
                'importance': importance
            }).sort_values('importance', ascending=False)
            
            print(f"\n{model_name} - Top 10 Important Features:")
            print(feature_imp.head(10).to_string(index=False))

def print_performance_summary(results, performance_metrics):
    """Print comprehensive performance summary"""
    print("\n" + "="*80)
    print("COMPREHENSIVE MODEL PERFORMANCE SUMMARY")
    print("="*80)
    
    # Create a comprehensive comparison table
    summary_data = []
    for model_name in results.keys():
        summary_data.append({
            'Model': model_name,
            'Accuracy (%)': f"{results[model_name]['accuracy']:.2f}",
            'R² Score': f"{results[model_name]['r2']:.4f}",
            'RMSE': f"{results[model_name]['rmse']:.4f}",
            'MAE': f"{results[model_name]['mae']:.4f}",
            'Training Time (s)': f"{performance_metrics[model_name]['training_time']:.4f}",
            'Prediction Time (s)': f"{results[model_name]['prediction_time']:.6f}",
            'Memory Used (MB)': f"{performance_metrics[model_name]['memory_used']:.2f}",
            'Peak Memory (MB)': f"{performance_metrics[model_name]['peak_memory']:.2f}"
        })
    
    summary_df = pd.DataFrame(summary_data)
    print(summary_df.to_string(index=False))
    
    # Best performing models in different categories
    best_accuracy = max(results, key=lambda k: results[k]['accuracy'])
    fastest_training = min(performance_metrics, key=lambda k: performance_metrics[k]['training_time'])
    fastest_prediction = min(results, key=lambda k: results[k]['prediction_time'])
    most_memory_efficient = min(performance_metrics, key=lambda k: performance_metrics[k]['memory_used'])
    
    print(f"\n{'='*80}")
    print("CATEGORY WINNERS:")
    print(f"Best Accuracy: {best_accuracy} ({results[best_accuracy]['accuracy']:.2f}%)")
    print(f"Fastest Training: {fastest_training} ({performance_metrics[fastest_training]['training_time']:.4f}s)")
    print(f"Fastest Prediction: {fastest_prediction} ({results[fastest_prediction]['prediction_time']:.6f}s)")
    print(f"Most Memory Efficient: {most_memory_efficient} ({performance_metrics[most_memory_efficient]['memory_used']:.2f} MB)")
    print(f"{'='*80}")

def main():
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
        
        print("Training models...")
        models, performance_metrics = train_models(X_train, y_train)
        
        print("Predicting and evaluating models...")
        results = predict_and_evaluate(models, X_test, y_test)
        
        # Print comprehensive performance summary
        print_performance_summary(results, performance_metrics)
        
        best_model = max(results, key=lambda k: results[k]['accuracy'])
        best_accuracy = results[best_model]['accuracy']
        
        if best_accuracy >= 85:
            print("\n🎉 TARGET ACHIEVED! Accuracy is 85% or above!")
        else:
            print(f"\n⚠️  Accuracy is {best_accuracy:.2f}%. Consider more data or feature engineering.")
        
        # Print feature importance
        print_feature_importance(models, X_train.columns.tolist())
        
        print("\nPlotting results...")
        plot_results(test_data, results, split_type)
        
        print("\nPlotting performance comparison...")
        plot_performance_comparison(performance_metrics, results)
        
    except FileNotFoundError:
        print(f"Error: Could not find the CSV file at {csv_path}")
        print("Please check if the file exists and the path is correct.")
    except Exception as e:
        print(f"An error occurred: {str(e)}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()