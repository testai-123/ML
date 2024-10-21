import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
import seaborn as sns

# Load the dataset
df = pd.read_csv('/content/Wine.csv')

# Separate features and target variable
X = df.drop('Customer_Segment', axis=1)
y = df['Customer_Segment']

# Standardize the features
X_standardized = StandardScaler().fit_transform(X)

# Apply PCA to reduce dimensions to 2 components
pca = PCA(n_components=2)
X_pca = pca.fit_transform(X_standardized)

# Create a DataFrame with the PCA results and the target variable
pca_df = pd.DataFrame(X_pca, columns=['PC1', 'PC2'])
pca_df['Customer_Segment'] = y

# Plot the PCA results
plt.figure(figsize=(8, 6))
sns.scatterplot(x='PC1', y='PC2', hue='Customer_Segment', data=pca_df, palette='Set1')
plt.title('PCA of Wine Dataset')
plt.xlabel('Principal Component 1')
plt.ylabel('Principal Component 2')
plt.show()

# Print the explained variance of each principal component
print('Explained Variance:', pca.explained_variance_ratio_)
