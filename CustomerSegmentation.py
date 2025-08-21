# Import necessary libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans, DBSCAN
from sklearn.metrics import silhouette_score
import warnings
warnings.filterwarnings('ignore')

# Set style for plots
plt.style.use('fivethirtyeight')
plt.rcParams['figure.figsize'] = (12, 6)

# Load the dataset
# Note: You'll need to download the dataset from Kaggle first:
# https://www.kaggle.com/vjchoudhary7/customer-segmentation-tutorial-in-python
df = pd.read_csv('Mall_Customers.csv')

# Display basic information about the dataset
print("Dataset Overview:")
print(df.head())
print("\nDataset Info:")
print(df.info())
print("\nDescriptive Statistics:")
print(df.describe())

# Check for missing values
print("\nMissing Values:")
print(df.isnull().sum())

# Rename columns for easier handling
df.rename(columns={'Annual Income (k$)': 'Income', 
                   'Spending Score (1-100)': 'SpendingScore'}, inplace=True)

# Exploratory Data Analysis
print("\nGender Distribution:")
print(df['Gender'].value_counts())

# Plot gender distribution
plt.figure(figsize=(12, 5))
plt.subplot(1, 2, 1)
sns.countplot(x='Gender', data=df)
plt.title('Gender Distribution')

plt.subplot(1, 2, 2)
sns.histplot(df['Age'], bins=30, kde=True)
plt.title('Age Distribution')
plt.tight_layout()
plt.show()

# Plot income and spending score distributions
plt.figure(figsize=(12, 5))
plt.subplot(1, 2, 1)
sns.histplot(df['Income'], bins=30, kde=True)
plt.title('Annual Income Distribution')

plt.subplot(1, 2, 2)
sns.histplot(df['SpendingScore'], bins=30, kde=True)
plt.title('Spending Score Distribution')
plt.tight_layout()
plt.show()

# Scatter plot of Income vs Spending Score
plt.figure(figsize=(10, 6))
sns.scatterplot(x='Income', y='SpendingScore', hue='Gender', data=df, alpha=0.7)
plt.title('Income vs Spending Score by Gender')
plt.show()

# Prepare data for clustering
X = df[['Income', 'SpendingScore']].values

# Standardize the data
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Determine the optimal number of clusters using the Elbow Method
wcss = []  # Within-Cluster Sum of Square
silhouette_scores = []

# Try different values of k from 2 to 10
for i in range(2, 11):
    kmeans = KMeans(n_clusters=i, init='k-means++', random_state=42, n_init=10)
    kmeans.fit(X_scaled)
    wcss.append(kmeans.inertia_)
    
    # Calculate silhouette score
    silhouette_avg = silhouette_score(X_scaled, kmeans.labels_)
    silhouette_scores.append(silhouette_avg)

# Plot the Elbow Method graph
plt.figure(figsize=(12, 5))
plt.subplot(1, 2, 1)
plt.plot(range(2, 11), wcss, marker='o', linestyle='--')
plt.xlabel('Number of Clusters')
plt.ylabel('WCSS')
plt.title('Elbow Method')

plt.subplot(1, 2, 2)
plt.plot(range(2, 11), silhouette_scores, marker='o')
plt.xlabel('Number of Clusters')
plt.ylabel('Silhouette Score')
plt.title('Silhouette Score Method')
plt.tight_layout()
plt.show()

# Based on the elbow method and silhouette score, choose optimal k
optimal_k = 5  # This is typically the optimal value for this dataset

# Apply K-Means with the optimal number of clusters
kmeans = KMeans(n_clusters=optimal_k, init='k-means++', random_state=42, n_init=10)
y_kmeans = kmeans.fit_predict(X_scaled)

# Add cluster labels to the original dataframe
df['Cluster'] = y_kmeans

# Visualize the clusters
plt.figure(figsize=(10, 6))
colors = ['red', 'blue', 'green', 'cyan', 'magenta']
for i in range(optimal_k):
    plt.scatter(X[y_kmeans == i, 0], X[y_kmeans == i, 1], 
                s=50, c=colors[i], label=f'Cluster {i+1}')

plt.scatter(kmeans.cluster_centers_[:, 0], kmeans.cluster_centers_[:, 1], 
            s=200, c='yellow', label='Centroids', marker='X')
plt.xlabel('Annual Income (k$)')
plt.ylabel('Spending Score (1-100)')
plt.title('Customer Segments')
plt.legend()
plt.show()

# Analyze cluster characteristics
cluster_summary = df.groupby('Cluster').agg({
    'Income': ['mean', 'median', 'min', 'max'],
    'SpendingScore': ['mean', 'median', 'min', 'max'],
    'Age': 'mean',
    'CustomerID': 'count'
}).round(2)

cluster_summary.columns = ['Income_Mean', 'Income_Median', 'Income_Min', 'Income_Max',
                          'Spending_Mean', 'Spending_Median', 'Spending_Min', 'Spending_Max',
                          'Age_Mean', 'Count']

print("Cluster Summary:")
print(cluster_summary)

# Plot cluster characteristics
fig, axes = plt.subplots(2, 2, figsize=(14, 10))

# Income by cluster
sns.barplot(x=df['Cluster'], y=df['Income'], ax=axes[0, 0], ci=None)
axes[0, 0].set_title('Average Income by Cluster')

# Spending score by cluster
sns.barplot(x=df['Cluster'], y=df['SpendingScore'], ax=axes[0, 1], ci=None)
axes[0, 1].set_title('Average Spending Score by Cluster')

# Age by cluster
sns.barplot(x=df['Cluster'], y=df['Age'], ax=axes[1, 0], ci=None)
axes[1, 0].set_title('Average Age by Cluster')

# Count of customers by cluster
sns.countplot(x=df['Cluster'], ax=axes[1, 1])
axes[1, 1].set_title('Customer Count by Cluster')

plt.tight_layout()
plt.show()

# BONUS: Try DBSCAN clustering
dbscan = DBSCAN(eps=0.5, min_samples=5)
y_dbscan = dbscan.fit_predict(X_scaled)

# Add DBSCAN labels to dataframe
df['DBSCAN_Cluster'] = y_dbscan

# Count the number of clusters in DBSCAN (excluding noise points labeled as -1)
n_clusters_dbscan = len(set(y_dbscan)) - (1 if -1 in y_dbscan else 0)
n_noise = list(y_dbscan).count(-1)

print(f"DBSCAN found {n_clusters_dbscan} clusters with {n_noise} noise points")

# Visualize DBSCAN clusters
plt.figure(figsize=(10, 6))
unique_labels = set(y_dbscan)
colors = [plt.cm.Spectral(each) for each in np.linspace(0, 1, len(unique_labels))]

for k, col in zip(unique_labels, colors):
    if k == -1:
        # Black used for noise
        col = [0, 0, 0, 1]
        label = 'Noise'
    else:
        label = f'Cluster {k}'
    
    class_member_mask = (y_dbscan == k)
    xy = X[class_member_mask]
    plt.scatter(xy[:, 0], xy[:, 1], s=50, c=[col], label=label, alpha=0.7)

plt.xlabel('Annual Income (k$)')
plt.ylabel('Spending Score (1-100)')
plt.title('DBSCAN Clustering')
plt.legend()
plt.show()

# Compare K-Means and DBSCAN results
comparison = pd.DataFrame({
    'KMeans_Cluster': y_kmeans,
    'DBSCAN_Cluster': y_dbscan
})

print("Cluster Comparison (first 10 rows):")
print(comparison.head(10))

# Calculate average spending per cluster for K-Means
kmeans_spending = df.groupby('Cluster')['SpendingScore'].mean().reset_index()
print("\nAverage Spending per Cluster (K-Means):")
print(kmeans_spending)

# Calculate average spending per cluster for DBSCAN (excluding noise)
dbscan_spending = df[df['DBSCAN_Cluster'] != -1].groupby('DBSCAN_Cluster')['SpendingScore'].mean().reset_index()
print("\nAverage Spending per Cluster (DBSCAN):")
print(dbscan_spending)