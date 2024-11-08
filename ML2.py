"""
Regression Analysis:(Any one)
A. Predict the price of the Uber ride from a given pickup point to the agreed drop-off
location. Perform following tasks:
1. Pre-process the dataset.
2. Identify outliers.
3. Check the correlation.
4. Implement linear regression and ridge, Lasso regression models.
5. Evaluate the models and compare their respective scores like R2, RMSE, etc.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.metrics import r2_score, mean_squared_error
from geopy.distance import geodesic
from sklearn.preprocessing import RobustScaler

# Load and clean the dataset
df = pd.read_csv("/content/ML2.csv").drop(['Unnamed: 0', 'key'], axis=1)
df['dropoff_longitude'] = df['dropoff_longitude'].fillna(df['dropoff_longitude'].median())
df['dropoff_latitude'] = df['dropoff_latitude'].fillna(df['dropoff_latitude'].mean())

# Convert datetime and extract features
df['pickup_datetime'] = pd.to_datetime(df['pickup_datetime'], errors='coerce')
df['year'] = df.pickup_datetime.dt.year
df['month'] = df.pickup_datetime.dt.month
df['dayofweek'] = df.pickup_datetime.dt.dayofweek
df['hour'] = df.pickup_datetime.dt.hour

# Remove rows with invalid latitude and longitude values
df = df[(df['pickup_latitude'].between(-90, 90)) & (df['dropoff_latitude'].between(-90, 90))]
df = df[(df['pickup_longitude'].between(-180, 180)) & (df['dropoff_longitude'].between(-180, 180))]

# Calculate distance traveled using geopy
def calculate_distance(row):
    start = (row['pickup_latitude'], row['pickup_longitude'])
    end = (row['dropoff_latitude'], row['dropoff_longitude'])
    return geodesic(start, end).kilometers

df['dist_travel_km'] = df.apply(calculate_distance, axis=1)

# Remove outliers based on the distance
df = df[(df['dist_travel_km'] >= 1) & (df['dist_travel_km'] <= 130)]

# Features and target variable
features = ['pickup_longitude', 'pickup_latitude', 'dropoff_longitude', 'dropoff_latitude',
            'passenger_count', 'hour', 'month', 'year', 'dayofweek', 'dist_travel_km']
X = df[features]
y = df['fare_amount']

# Check for missing values in the features and handle them
if X.isnull().any().any():
    print("Missing values found in features, filling NaNs...")
    X = X.fillna(X.median())  # Fill missing values with the median for each column

# Scale the data using RobustScaler
scaler = RobustScaler()
X_scaled = scaler.fit_transform(X)

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=100)

# Initialize Linear Regression Model
regression = LinearRegression()
regression.fit(X_train, y_train)
prediction = regression.predict(X_test)

# Initialize the Ridge Model
ridge_reg = Ridge(alpha=1.0)
ridge_reg.fit(X_train, y_train)
ridge_pred = ridge_reg.predict(X_test)

# Initialize the Lasso Model
lasso_reg = Lasso(alpha=1.0)
lasso_reg.fit(X_train, y_train)
lasso_pred = lasso_reg.predict(X_test)

# Error Evaluation
print("Linear Regression R2 Score:", r2_score(y_test, prediction))
print("Ridge Regression R2 Score:", r2_score(y_test, ridge_pred))
print("Lasso Regression R2 Score:", r2_score(y_test, lasso_pred))

# Plotting the predictions
plt.figure(figsize=(12, 10))
plt.scatter(y_test, prediction, label='Linear Regression', alpha=0.5)
plt.scatter(y_test, ridge_pred, label='Ridge Regression', alpha=0.5)
plt.scatter(y_test, lasso_pred, label='Lasso Regression', alpha=0.5)
plt.xlabel('True Fare Amount')
plt.ylabel('Predicted Fare Amount')
plt.legend()
plt.title('True vs Predicted Fare Amount')
plt.show()
