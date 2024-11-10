"""
To use PCA Algorithm for dimensionality reduction.
You have a dataset that includes measurements for different variables on wine
(alcohol, ash, magnesium, and so on). Apply PCA algorithm & transform this data
so that most variations in the measurements of the variables are captured by a small
number of principal components so that it is easier to distinguish between red and
white wine by inspecting these principal components.
Dataset Link: https://media.geeksforgeeks.org/wp-content/uploads/Wine.csv
"""

import pandas as pd 
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt 
import seaborn as sns
from sklearn.decomposition import PCA 
import plotly.express as px

df = pd.read_csv("/content/ML1.csv")

X = df.drop("Customer_Segment",axis=1)
y = df["Customer_Segment"]

X_stand = StandardScaler().fit_transform(X)

pca = PCA(n_components=2)
X_pca = pca.fit_transform(X_stand)

pca_df= pd.DataFrame(X_pca, columns=['PC1','PC2'])
pca_df['Customer_Segment'] = y

plt.figure(figsize=(8,6))
sns.scatterplot(x='PC1',y='PC2',hue='Customer_Segment',data=pca_df,palette="Set1")
plt.title("PCA of Wine Dataset")
plt.xlabel("PC1")
plt.ylabel("PC2")
plt.show()

