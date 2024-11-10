import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.metrics import r2_score, mean_squared_error
from geopy.distance import geodesic
from sklearn.preprocessing import RobustScaler

df = pd.read_csv("/content/ML2.csv")

df = df.drop(["Unnamed: 0",	"key"],axis=1)

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
df = df.drop("pickup_datetime", axis=1)
x_data = df.drop("fare_amount",axis=1)
y = df['fare_amount']

df.corr()

scaler = RobustScaler()
X_scaled = scaler.fit_transform(x_data)

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
