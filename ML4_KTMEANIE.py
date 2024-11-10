import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score

# Load the dataset
df = pd.read_csv('Iris.csv')

# Drop the 'Id' column
df.drop('Id', axis=1, inplace=True)

# Scatter plot to show distribution of petal length and width based on species
sns.scatterplot(data=df, x='PetalLengthCm', y='PetalWidthCm', hue='Species', palette='Set1')
plt.title('Iris Species Distribution')
plt.legend(title='Species')
plt.show()

# Select input features
X = df.iloc[:, 0:4]

# Elbow method to find the optimal number of clusters
sse = []
for k in range(1, 10):
    km = KMeans(n_clusters=k, random_state=0)
    km.fit(X)
    sse.append(km.inertia_)

# Plot the Elbow Curve
plt.plot(range(1, 10), sse, marker='o')
plt.title('The Elbow Method')
plt.xlabel('Number of clusters')
plt.ylabel('SSE')
plt.show()

# Train K-means with 3 clusters
km = KMeans(n_clusters=3, max_iter=300, random_state=0)
y_means = km.fit_predict(X)

# Calculate and print the silhouette score
silhouette_avg = silhouette_score(X, y_means)
print(f"Silhouette Score: {silhouette_avg:.2f}")
