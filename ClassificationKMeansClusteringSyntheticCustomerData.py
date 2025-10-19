import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler

# Sample synthetic customer data (e.g., [Annual Income, Spending Score])
data = {'Income': [15, 16, 17, 18, 60, 62, 63, 65, 99, 101],
        'Spending': [39, 81, 6, 77, 40, 12, 73, 80, 82, 90]}

df = pd.DataFrame(data)

# Data scaling for fair clustering
scaler = StandardScaler()
scaled_data = scaler.fit_transform(df)

# Run K-means clustering to segment customers
kmeans = KMeans(n_clusters=3, random_state=42)
clusters = kmeans.fit_predict(scaled_data)

# Add cluster labels to DataFrame
df['Cluster'] = clusters

# Visualize the clusters
plt.figure(figsize=(8,6))
for cluster in np.unique(clusters):
    plt.scatter(df[df.Cluster == cluster].Income,
                df[df.Cluster == cluster].Spending,
                label=f'Cluster {cluster}')
plt.xlabel('Annual Income')
plt.ylabel('Spending Score')
plt.title('Customer Segmentation using K-Means')
plt.legend()
plt.show()

# Print results
print(df)
