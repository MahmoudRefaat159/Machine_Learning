1. Data Loading & Preprocessing
Loads car sales data from CSV and prediction data from Excel.

Uses LabelEncoder to convert categorical text (like car models) into numbers.

Removes outliers using the IQR method to clean the data.

2. Model Training
Splits data into training (80%) and testing (20%) sets.

Trains a RandomForestRegressor to predict car prices based on features like model, engine size, mileage, etc.

3. Making Predictions
Prepares new input data (from Excel) for prediction by applying the same encoding used in training.

Predicts prices and saves results to a new Excel file.

Prints R² scores to show model accuracy.

Key Points
Ensures consistency by using the same encoders for new data.

Handles outliers to improve prediction quality.

Simple but effective for price estimation using car features.