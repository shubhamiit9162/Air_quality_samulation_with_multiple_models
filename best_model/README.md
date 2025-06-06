PM2.5 Prediction System - Linear Regression Model
A comprehensive pipeline for predicting PM2.5 levels using Linear Regression as a benchmark model.

The data processing and modeling pipeline involves several key components to ensure accuracy and robustness:

1. Filtering Criteria: PM2.5 > 0, Temperature > 0, Humidity > 0 to eliminate invalid readings
2. Intelligent Data Splitting Strategy:
   - 3+ days: 2 days training, 1 day testing
   - 2 days: 1:1 split
   - 1 day: 80% training, 20% testing temporal split
3. Linear Regression Model: Simple benchmark for PM2.5 prediction
4. Adaptive data handling for datasets of any size
5. Modern ML practices with clean, warning-free code
6. Comprehensive analysis through data exploration
7. Robust error handling for missing files or invalid data
8. Visual insights using time-series plots

Requirements: pandas, numpy, scikit-learn, matplotlib

Usage:

- Place your CSV file inside the data/ folder
- Update the CSV filename below if needed
- Run this script in VS Code or any Python environment

Data Flow Pipeline:
Raw CSV Data → Data Loading → Preprocessing → Feature Engineering → Model Training → Prediction → Evaluation → Visualization
