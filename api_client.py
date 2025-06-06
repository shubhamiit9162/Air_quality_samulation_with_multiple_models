import requests
import json
import pandas as pd
from typing import Dict, List, Optional, Union
import time
from datetime import datetime
import logging

class PM25DashboardClient:
    """
    Client for interacting with the PM2.5 Prediction Dashboard API
    """
    
    def __init__(self, base_url: str = "http://localhost:5000"):
        """
        Initialize the API client
        
        Args:
            base_url: Base URL of the Flask API server
        """
        self.base_url = base_url.rstrip('/')
        self.session = requests.Session()
        
        # Setup logging
        logging.basicConfig(level=logging.INFO)
        self.logger = logging.getLogger(__name__)
    
    def _make_request(self, method: str, endpoint: str, **kwargs) -> requests.Response:
        """
        Make HTTP request to the API
        
        Args:
            method: HTTP method (GET, POST, etc.)
            endpoint: API endpoint
            **kwargs: Additional arguments for requests
        
        Returns:
            Response object
        """
        url = f"{self.base_url}{endpoint}"
        
        try:
            response = self.session.request(method, url, **kwargs)
            response.raise_for_status()
            return response
        except requests.exceptions.RequestException as e:
            self.logger.error(f"API request failed: {e}")
            raise
    
    def upload_csv_file(self, file_path: str) -> Dict:
        """
        Upload CSV file to the server for processing
        
        Args:
            file_path: Path to the CSV file
        
        Returns:
            Response data containing file processing results
        """
        try:
            with open(file_path, 'rb') as f:
                files = {'file': f}
                response = self._make_request('POST', '/api/upload', files=files)
                
            result = response.json()
            self.logger.info(f"File uploaded successfully: {result.get('message', '')}")
            return result
            
        except FileNotFoundError:
            self.logger.error(f"File not found: {file_path}")
            raise
        except Exception as e:
            self.logger.error(f"Upload failed: {e}")
            raise
    
    def train_model(self, model_type: str = "RandomForest") -> Dict:
        """
        Train machine learning model
        
        Args:
            model_type: Type of model to train (RandomForest, GradientBoosting, etc.)
        
        Returns:
            Training results and model performance metrics
        """
        data = {"model_type": model_type}
        
        try:
            response = self._make_request('POST', '/api/train_model', json=data)
            result = response.json()
            
            self.logger.info(f"Model trained successfully: {model_type}")
            self.logger.info(f"Accuracy: {result['performance']['accuracy']:.2f}%")
            
            return result
            
        except Exception as e:
            self.logger.error(f"Model training failed: {e}")
            raise
    
    def make_prediction(self, features: Dict[str, float]) -> Dict:
        """
        Make prediction using trained model
        
        Args:
            features: Dictionary of feature values
        
        Returns:
            Prediction result
        """
        try:
            response = self._make_request('POST', '/api/predict', json=features)
            result = response.json()
            
            self.logger.info(f"Prediction made: {result['prediction']:.2f}")
            return result
            
        except Exception as e:
            self.logger.error(f"Prediction failed: {e}")
            raise
    
    def get_realtime_data(self) -> Dict:
        """
        Get real-time data for visualization
        
        Returns:
            Recent data points and statistics
        """
        try:
            response = self._make_request('GET', '/api/realtime_data')
            return response.json()
            
        except Exception as e:
            self.logger.error(f"Failed to get real-time data: {e}")
            raise
    
    def get_model_performance(self) -> Dict:
        """
        Get historical model performance data
        
        Returns:
            Performance history of trained models
        """
        try:
            response = self._make_request('GET', '/api/model_performance')
            return response.json()
            
        except Exception as e:
            self.logger.error(f"Failed to get model performance: {e}")
            raise
    
    def get_feature_importance(self) -> Dict:
        """
        Get feature importance from trained model
        
        Returns:
            Feature importance rankings
        """
        try:
            response = self._make_request('GET', '/api/feature_importance')
            return response.json()
            
        except Exception as e:
            self.logger.error(f"Failed to get feature importance: {e}")
            raise
    
    def get_data_summary(self) -> Dict:
        """
        Get comprehensive data summary
        
        Returns:
            Data statistics and summary information
        """
        try:
            response = self._make_request('GET', '/api/data_summary')
            return response.json()
            
        except Exception as e:
            self.logger.error(f"Failed to get data summary: {e}")
            raise
    
    def health_check(self) -> bool:
        """
        Check if the API server is running
        
        Returns:
            True if server is healthy, False otherwise
        """
        try:
            response = self._make_request('GET', '/')
            return response.status_code == 200
        except:
            return False

class PM25DataAnalyzer:
    """
    Helper class for analyzing PM2.5 data and interacting with the dashboard
    """
    
    def __init__(self, api_client: PM25DashboardClient):
        """
        Initialize the analyzer with an API client
        
        Args:
            api_client: Instance of PM25DashboardClient
        """
        self.client = api_client
        self.logger = logging.getLogger(__name__)
    
    def full_analysis_pipeline(self, csv_paths: List[str], model_types: List[str] = None) -> Dict:
        """
        Run complete analysis pipeline
        
        Args:
            csv_paths: List of paths to CSV files
            model_types: List of model types to train
        
        Returns:
            Complete analysis results
        """
        if model_types is None:
            model_types = ["RandomForest"]
        
        results = {
            'upload_result': None,
            'training_results': {},
            'data_summary': None,
            'feature_importance': None
        }
        
        try:
            self.logger.info("Starting full analysis pipeline...")
            
            # Upload each CSV file
            for csv_path in csv_paths:
                results['upload_result'] = self.client.upload_csv_file(csv_path)
            
            # Get data summary
            results['data_summary'] = self.client.get_data_summary()
            
            # Train models
            for model_type in model_types:
                self.logger.info(f"Training {model_type} model...")
                results['training_results'][model_type] = self.client.train_model(model_type)
                time.sleep(1)  # Brief pause between model training
            
            # Get feature importance
            try:
                results['feature_importance'] = self.client.get_feature_importance()
            except:
                self.logger.warning("Could not retrieve feature importance")
            
            self.logger.info("Analysis pipeline completed successfully!")
            return results
            
        except Exception as e:
            self.logger.error(f"Analysis pipeline failed: {e}")
            raise
    
    def batch_predictions(self, feature_sets: List[Dict]) -> List[Dict]:
        """
        Make batch predictions
        
        Args:
            feature_sets: List of feature dictionaries
        
        Returns:
            List of prediction results
        """
        predictions = []
        
        for i, features in enumerate(feature_sets):
            try:
                result = self.client.make_prediction(features)
                result['batch_index'] = i
                predictions.append(result)
                
            except Exception as e:
                self.logger.error(f"Prediction {i} failed: {e}")
                predictions.append({
                    'batch_index': i,
                    'error': str(e)
                })
        
        return predictions
    
    def generate_prediction_scenarios(self, base_features: Dict, 
                                    variable_ranges: Dict) -> List[Dict]:
        """
        Generate prediction scenarios by varying specific features
        
        Args:
            base_features: Base feature set
            variable_ranges: Dictionary of features to vary and their ranges
        
        Returns:
            List of feature scenarios
        """
        import itertools
        
        scenarios = []
        
        # Create all combinations of variable ranges
        keys = list(variable_ranges.keys())
        values = list(variable_ranges.values())
        
        for combination in itertools.product(*values):
            scenario = base_features.copy()
            
            for i, key in enumerate(keys):
                scenario[key] = combination[i]
            
            scenarios.append(scenario)
        
        return scenarios
    
    def monitor_realtime_data(self, duration_minutes: int = 5, 
                            interval_seconds: int = 30) -> List[Dict]:
        """
        Monitor real-time data for specified duration
        
        Args:
            duration_minutes: How long to monitor (minutes)
            interval_seconds: Interval between data fetches (seconds)
        
        Returns:
            List of data snapshots
        """
        snapshots = []
        end_time = time.time() + (duration_minutes * 60)
        
        self.logger.info(f"Starting real-time monitoring for {duration_minutes} minutes...")
        
        while time.time() < end_time:
            try:
                data = self.client.get_realtime_data()
                data['snapshot_time'] = datetime.now().isoformat()
                snapshots.append(data)
                
                self.logger.info(f"Data snapshot captured: "
                               f"Avg PM2.5: {data['stats']['avg_pm25']:.2f}")
                
                time.sleep(interval_seconds)
                
            except Exception as e:
                self.logger.error(f"Failed to capture snapshot: {e}")
                break
        
        self.logger.info(f"Monitoring completed. Captured {len(snapshots)} snapshots.")
        return snapshots

# Example usage and testing functions
def example_usage():
    """
    Example of how to use the API client
    """
    # Initialize client
    client = PM25DashboardClient("http://localhost:5000")
    analyzer = PM25DataAnalyzer(client)
    
    # Check if server is running
    if not client.health_check():
        print("Server is not running. Please start the Flask app first.")
        return
    
    # Example 1: Full analysis pipeline
    try:
        csv_paths = [
            "data/UK0010_PM46_2986N_7790E_2024-09-27.csv",
            "data/UK0010_PM46_2986N_7790E_2024-09-28.csv",
            "data/UK0010_PM46_2986N_7790E_2024-09-29.csv"
        ]
        
        results = analyzer.full_analysis_pipeline(csv_paths, ["RandomForest", "Linear Regression", "ANN"])
        
        print("Analysis Results:")
        print(f"Data points: {results['upload_result']['stats']['total_points']}")
        
        for model, result in results['training_results'].items():
            print(f"{model} Accuracy: {result['performance']['accuracy']:.2f}%")
    
    except Exception as e:
        print(f"Analysis failed: {e}")
    
    # Example 2: Make single prediction
    try:
        features = {
            'temperature': 25.0,
            'humidity': 60.0,
            'hour': 14,
            'is_weekend': 0
        }
        
        prediction = client.make_prediction(features)
        print(f"Predicted PM2.5: {prediction['prediction']:.2f}")
    
    except Exception as e:
        print(f"Prediction failed: {e}")

if __name__ == "__main__":
    example_usage()

