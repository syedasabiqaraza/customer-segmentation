# customer-segmentation
Customer Segmentation using K-Means and DBSCAN clustering


Project Overview

This project applies unsupervised machine learning (clustering) on a dataset of mall customers to identify different customer segments based on Annual Income and Spending Score.
It uses K-Means and DBSCAN clustering algorithms, visualizes the clusters, and summarizes key characteristics of each segment.

Dataset

Source: Kaggle - Mall Customer Segmentation Dataset

Size: 200 rows, 5 columns

Features used:

CustomerID – Unique identifier

Gender – Male/Female

Age – Customer age

Annual Income (k$) – Income in thousand dollars

Spending Score (1-100) – How much the customer spends

Technologies Used

Python 🐍

Pandas & NumPy

Matplotlib & Seaborn (visualizations)

Scikit-learn (KMeans, DBSCAN)

Steps Performed

Data Exploration: Check dataset info, descriptive statistics, missing values.

Data Preprocessing: Rename columns, scale features using StandardScaler.

Exploratory Data Analysis (EDA):

Gender distribution

Age, Income, Spending Score distributions

Scatter plots of Income vs Spending Score

Clustering:

K-Means: Use Elbow Method & Silhouette Score to find optimal clusters

DBSCAN: Identify density-based clusters and noise points

Analysis:

Compare K-Means and DBSCAN clusters

Calculate average spending, income, and age per cluster

Visualization: Cluster plots, bar charts for characteristics

Results

Optimal K for K-Means: 5 clusters

DBSCAN clusters found: 2 clusters + 8 noise points

Insights:

High-income, high-spending customers

High-income, low-spending customers

Low-income, high-spending customers

Average customers

Outliers/noise
