import requests
import json
from datetime import datetime

API_BASE_URL = "http://localhost:5000"

def single_prediction_example():
    """Example of making a single PM2.5 prediction"""
    print("=== Single Prediction Example ===")
    
    data = {
        "temperature": 25.5,
        "humidity": 65.2,
        "datetime": datetime.now().isoformat()  
    }
    
    try:
  
        response = requests.post(f"{API_BASE_URL}/api/predict", json=data)
        
        if response.status_code == 200:
            result = response.json()
            print("✅ Prediction successful!")
            print(f"PM2.5 Prediction: {result['pm25_prediction']} μg/m³")
            print(f"Air Quality: {result['air_quality']}")
            print(f"Color: {result['color']}")
            print(f"Timestamp: {result['timestamp']}")
        else:
            print(f"❌ Error: {response.json().get('error', 'Unknown error')}")
            
    except requests.exceptions.ConnectionError:
        print("❌ Connection Error: Make sure Flask API is running on localhost:5000")
    except Exception as e:
        print(f"❌ Error: {str(e)}")


def batch_prediction_example():
    """Example of making multiple predictions at once"""
    print("\n=== Batch Prediction Example ===")
    

    data = {
        "predictions": [
            {"temperature": 20.0, "humidity": 50.0},
            {"temperature": 25.0, "humidity": 60.0},
            {"temperature": 30.0, "humidity": 70.0},
            {"temperature": 15.0, "humidity": 40.0}
        ]
    }
    
    try:
        response = requests.post(f"{API_BASE_URL}/api/batch_predict", json=data)
        
        if response.status_code == 200:
            results = response.json()['results']
            print("✅ Batch prediction successful!")
            
            for i, result in enumerate(results):
                if 'error' in result:
                    print(f"Prediction {i+1}: Error - {result['error']}")
                else:
                    print(f"Prediction {i+1}: {result['pm25_prediction']} μg/m³ "
                          f"(T: {result['temperature']}°C, H: {result['humidity']}%)")
        else:
            print(f" Error: {response.json().get('error', 'Unknown error')}")
            
    except Exception as e:
        print(f" Error: {str(e)}")


def model_info_example():
    """Get information about the loaded model"""
    print("\n=== Model Info Example ===")
    
    try:
        response = requests.get(f"{API_BASE_URL}/api/model_info")
        
        if response.status_code == 200:
            info = response.json()
            print(" Model info retrieved!")
            print(f"Model Type: {info.get('model_type', 'Unknown')}")
            print(f"Features Count: {info.get('features_count', 'Unknown')}")
            print(f"Status: {info.get('status', 'Unknown')}")
        else:
            print(f" Error: {response.json().get('error', 'Unknown error')}")
            
    except Exception as e:
        print(f" Error: {str(e)}")



def health_check_example():
    """Check if API is healthy and model is loaded"""
    print("\n=== Health Check Example ===")
    
    try:
        response = requests.get(f"{API_BASE_URL}/api/health")
        
        if response.status_code == 200:
            health = response.json()
            print(" API is healthy!")
            print(f"Status: {health['status']}")
            print(f"Model Loaded: {health['model_loaded']}")
            print(f"Timestamp: {health['timestamp']}")
        else:
            print("❌ API is not healthy")
            
    except Exception as e:
        print(f"❌ Error: {str(e)}")


def javascript_example():
    """Show JavaScript example for web applications"""
    print("\n=== JavaScript Example ===")
    
    js_code = '''
// JavaScript example for calling the API from a web page

async function predictPM25(temperature, humidity) {
    try {
        const response = await fetch('http://localhost:5000/api/predict', {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json',
            },
            body: JSON.stringify({
                temperature: temperature,
                humidity: humidity,
                datetime: new Date().toISOString()
            })
        });
        
        if (response.ok) {
            const data = await response.json();
            console.log('PM2.5 Prediction:', data.pm25_prediction);
            console.log('Air Quality:', data.air_quality);
            return data;
        } else {
            const error = await response.json();
            console.error('Error:', error.error);
        }
    } catch (error) {
        console.error('Network error:', error);
    }
}

// Usage example
predictPM25(25.5, 65.2);
    '''
    
    print(js_code)


def curl_examples():
    """Show curl command examples"""
    print("\n=== cURL Command Examples ===")
    
    commands = [
        "# Single Prediction",
        'curl -X POST http://localhost:5000/api/predict \\',
        '  -H "Content-Type: application/json" \\',
        '  -d \'{"temperature": 25.5, "humidity": 65.2}\'',
        "",
        "# Health Check",
        'curl http://localhost:5000/api/health',
        "",
        "# Model Info",
        'curl http://localhost:5000/api/model_info',
        "",
        "# Batch Prediction",
        'curl -X POST http://localhost:5000/api/batch_predict \\',
        '  -H "Content-Type: application/json" \\',
        '  -d \'{"predictions": [{"temperature": 20, "humidity": 50}, {"temperature": 25, "humidity": 60}]}\''
    ]
    
    for cmd in commands:
        print(cmd)



def continuous_monitoring_example():
    """Example of continuous monitoring with the API"""
    print("\n=== Continuous Monitoring Example ===")
    
    import time
    import random
    
    print("Starting continuous monitoring (5 predictions)...")
    
    for i in range(5):
       
        temperature = round(random.uniform(15, 35), 1)
        humidity = round(random.uniform(30, 90), 1)
        
        data = {
            "temperature": temperature,
            "humidity": humidity
        }
        
        try:
            response = requests.post(f"{API_BASE_URL}/api/predict", json=data)
            
            if response.status_code == 200:
                result = response.json()
                print(f"Reading {i+1}: T={temperature}°C, H={humidity}% → "
                      f"PM2.5={result['pm25_prediction']}μg/m³ ({result['air_quality']})")
            else:
                print(f"Reading {i+1}: Error - {response.json().get('error')}")
                
        except Exception as e:
            print(f"Reading {i+1}: Error - {str(e)}")
        
        time.sleep(2)  

def data_logging_example():
    """Example of logging predictions to a file"""
    print("\n=== Data Logging Example ===")
    
    import csv
    from datetime import datetime
    
 
    sample_data = [
        {"temperature": 22.5, "humidity": 55.0},
        {"temperature": 26.0, "humidity": 70.0},
        {"temperature": 18.5, "humidity": 45.0}
    ]
    
 
    log_filename = f"pm25_predictions_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"
    
    with open(log_filename, 'w', newline='') as csvfile:
        fieldnames = ['timestamp', 'temperature', 'humidity', 'pm25_prediction', 'air_quality']
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()
        
        for data in sample_data:
            try:
                response = requests.post(f"{API_BASE_URL}/api/predict", json=data)
                
                if response.status_code == 200:
                    result = response.json()
                    
                  
                    writer.writerow({
                        'timestamp': result['timestamp'],
                        'temperature': data['temperature'],
                        'humidity': data['humidity'],
                        'pm25_prediction': result['pm25_prediction'],
                        'air_quality': result['air_quality']
                    })
                    
                    print(f"Logged: T={data['temperature']}°C, H={data['humidity']}% → "
                          f"PM2.5={result['pm25_prediction']}μg/m³")
                    
            except Exception as e:
                print(f"Error logging data: {str(e)}")
    
    print(f"✅ Data logged to: {log_filename}")



def error_handling_example():
    """Example of proper error handling"""
    print("\n=== Error Handling Example ===")
    
  
    invalid_data_sets = [
        {},  
        {"temperature": "invalid"},  
        {"humidity": 65.2},  
        {"temperature": 25.5},  
        {"temperature": 25.5, "humidity": 150}  
    ]
    
    for i, data in enumerate(invalid_data_sets):
        print(f"\nTest {i+1}: {data}")
        
        try:
            response = requests.post(f"{API_BASE_URL}/api/predict", json=data, timeout=5)
            
            if response.status_code == 200:
                result = response.json()
                print(f"✅ Success: {result['pm25_prediction']} μg/m³")
            else:
                error_info = response.json()
                print(f"❌ API Error: {error_info.get('error', 'Unknown error')}")
                
        except requests.exceptions.Timeout:
            print("❌ Timeout Error: Request took too long")
        except requests.exceptions.ConnectionError:
            print("❌ Connection Error: Cannot connect to API")
        except requests.exceptions.RequestException as e:
            print(f"❌ Request Error: {str(e)}")
        except json.JSONDecodeError:
            print("❌ JSON Error: Invalid response format")
        except Exception as e:
            print(f"❌ Unexpected Error: {str(e)}")



def performance_testing_example():
    """Example of testing API performance"""
    print("\n=== Performance Testing Example ===")
    
    import time
    
    # Test data
    test_data = {"temperature": 25.0, "humidity": 60.0}
    num_requests = 10
    
    print(f"Testing API performance with {num_requests} requests...")
    
    response_times = []
    successful_requests = 0
    
    for i in range(num_requests):
        start_time = time.time()
        
        try:
            response = requests.post(f"{API_BASE_URL}/api/predict", json=test_data)
            end_time = time.time()
            
            response_time = end_time - start_time
            response_times.append(response_time)
            
            if response.status_code == 200:
                successful_requests += 1
                print(f"Request {i+1}: {response_time:.3f}s ✅")
            else:
                print(f"Request {i+1}: {response_time:.3f}s ❌")
                
        except Exception as e:
            end_time = time.time()
            response_time = end_time - start_time
            print(f"Request {i+1}: {response_time:.3f}s ❌ ({str(e)})")
    
    # Calculate statistics
    if response_times:
        avg_time = sum(response_times) / len(response_times)
        min_time = min(response_times)
        max_time = max(response_times)
        success_rate = (successful_requests / num_requests) * 100
        
        print(f"\n📊 Performance Results:")
        print(f"Success Rate: {success_rate:.1f}%")
        print(f"Average Response Time: {avg_time:.3f}s")
        print(f"Min Response Time: {min_time:.3f}s")
        print(f"Max Response Time: {max_time:.3f}s")


if __name__ == "__main__":
    print("🚀 PM2.5 Prediction API Client Examples")
    print("=" * 50)
    
    # Run all examples
    try:
        health_check_example()
        model_info_example()
        single_prediction_example()
        batch_prediction_example()
        continuous_monitoring_example()
        data_logging_example()
        error_handling_example()
        performance_testing_example()
        javascript_example()
        curl_examples()
        
    except KeyboardInterrupt:
        print("\n\n Examples interrupted by user")
    except Exception as e:
        print(f"\n\n Unexpected error: {str(e)}")
    
    print("\n" + "=" * 50)
    print("✅ All examples completed!")
    print("\nTo use these examples:")
    print("1. First run: python main.py (to train and save the model)")
    print("2. Then run: python app.py (to start the Flask API)")
    print("3. Finally run: python api_client_examples.py (to test the API)")
    print("4. Or visit: http://localhost:5000 (for the web dashboard)")