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

# Data before PCA 
sns.scatterplot(X_stand)


pca = PCA(n_components=2)
X_pca = pca.fit_transform(X_stand)

pca_df= pd.DataFrame(X_pca, columns=['PC1','PC2'])
pca_df['Customer_Segment'] = y

# After PCA plot
plt.figure(figsize=(8,6))
sns.scatterplot(x='PC1',y='PC2',hue='Customer_Segment',data=pca_df,palette="Set1")
plt.title("PCA of Wine Dataset")
plt.xlabel("PC1")
plt.ylabel("PC2")
plt.show()

