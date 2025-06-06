PM2.5 Prediction

The data processing and modeling pipeline involves several key components to ensure accuracy and robustness. First, the filtering criteria include conditions such as PM2.5 > 0, Temperature > 0, and Humidity > 0, which help eliminate invalid or negative readings and maintain the integrity of the input data. The intelligent data splitting strategy adapts to the available data: for datasets spanning 3 or more days, it uses two days for training and one for testing; for 2-day data, it uses a 1:1 split; and for single-day datasets, it applies an 80% training and 20% testing temporal split. The model architecture includes three approaches: a basic Linear Regression model as a simple benchmark, a Random Forest with 100 trees to capture non-linear relationships, and an Artificial Neural Network (ANN) with a 2→64→32→1 architecture using ReLU activation for more complex pattern recognition. Key features of the system include adaptive data handling that accommodates datasets of any size, adherence to modern machine learning practices with clean, warning-free code, and comprehensive analysis through data exploration and model comparison. It also incorporates robust error handling to deal with missing files or invalid data and provides visual insights using time-series plots for intuitive interpretation.
Requirements:

- pandas
- numpy
- scikit-learn
- tensorflow
- matplotlib

Usage:

- Place your CSV file inside the data/ folder.
- Update the CSV filename below if needed.
- Run this script in VS Code or any Python environment

Data Flow Pipeline

Raw CSV Data → Data Loading → Preprocessing → Feature Engineering → Model Training → Prediction → Evaluation → Visualization
# Air_quality_samulation_with_multiple_models
