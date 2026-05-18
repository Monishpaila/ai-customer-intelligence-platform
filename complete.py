# 🚀 CUSTOMER INTELLIGENCE SYSTEM (SINGLE FILE PROJECT)

# ======================================================

# FEATURES:

# 1. Data Cleaning

# 2. Feature Engineering

# 3. RFM Analysis

# 4. Customer Segmentation (K-Means)

# 5. Elbow Method

# 6. Churn Prediction

# 7. Spending Prediction

# ======================================================

# =========================

# IMPORT LIBRARIES

# =========================

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, mean_absolute_error, mean_squared_error, r2_score

# =========================

# LOAD DATASET

# =========================

print("\nLOADING DATASET...")

# Replace with your dataset path if needed

file_path = r"C:/Users/monis/OneDrive/Desktop/AI Customer Intelligence Platform/online_retail_II.csv"

df = pd.read_csv(file_path)

print("\nFIRST 5 ROWS:")
print(df.head())

print("\nDATASET INFO:")
print(df.info())

print("\nDATASET SHAPE:")
print(df.shape)

# =========================

# MISSING VALUES

# =========================

print("\nMISSING VALUES:")
print(df.isnull().sum())

# =========================

# REMOVE NULL VALUES

# =========================

print("\nREMOVING NULL VALUES...")

df.dropna(inplace=True)

print("NEW SHAPE:", df.shape)

# =========================

# REMOVE DUPLICATES

# =========================

print("\nREMOVING DUPLICATES...")

df.drop_duplicates(inplace=True)

print("NEW SHAPE:", df.shape)

# =========================

# REMOVE CANCELLED ORDERS

# =========================

print("\nREMOVING CANCELLED ORDERS...")

# Invoice starting with C means cancelled

# Example: C536391

df = df[~df['Invoice'].astype(str).str.startswith('C')]

print("NEW SHAPE:", df.shape)

# =========================

# FEATURE ENGINEERING

# =========================

print("\nCREATING TOTAL PRICE FEATURE...")

df['TotalPrice'] = df['Quantity'] * df['Price']

print(df[['Quantity', 'Price', 'TotalPrice']].head())

# =========================

# CONVERT DATE COLUMN

# =========================

print("\nCONVERTING DATE COLUMN...")

df['InvoiceDate'] = pd.to_datetime(df['InvoiceDate'])

# =========================

# RFM ANALYSIS

# =========================

print("\nPERFORMING RFM ANALYSIS...")

# Reference date
reference_date = df['InvoiceDate'].max() + pd.Timedelta(days=1)

# Create RFM table
rfm = df.groupby('Customer ID').agg({
    'InvoiceDate': lambda x: (reference_date - x.max()).days,
    'Invoice': 'nunique',
    'TotalPrice': 'sum'
})

# Rename columns
rfm.columns = ['Recency', 'Frequency', 'Monetary']

# Reset index
rfm = rfm.reset_index()

# =========================
# ADVANCED FEATURE ENGINEERING
# =========================

print("\nCREATING ADVANCED FEATURES...")

# ---------------------------------
# Average Order Value
# ---------------------------------
rfm['AvgOrderValue'] = (
    rfm['Monetary'] / rfm['Frequency']
)

# ---------------------------------
# Total Quantity Purchased
# ---------------------------------
total_quantity = df.groupby('Customer ID')['Quantity'].sum().reset_index()

total_quantity.columns = ['Customer ID', 'TotalQuantity']

rfm = rfm.merge(total_quantity, on='Customer ID')

# ---------------------------------
# Product Diversity
# Number of unique products bought
# ---------------------------------
product_diversity = df.groupby('Customer ID')['StockCode'].nunique().reset_index()

product_diversity.columns = ['Customer ID', 'ProductDiversity']

rfm = rfm.merge(product_diversity, on='Customer ID')

# ---------------------------------
# Customer Lifetime
# ---------------------------------
customer_lifetime = df.groupby('Customer ID')['InvoiceDate'].agg(
    lambda x: (x.max() - x.min()).days
).reset_index()

customer_lifetime.columns = ['Customer ID', 'CustomerLifetimeDays']

rfm = rfm.merge(customer_lifetime, on='Customer ID')

# ---------------------------------
# Purchase Frequency Rate
# ---------------------------------
rfm['PurchaseFrequencyRate'] = (
    rfm['Frequency'] /
    (rfm['CustomerLifetimeDays'] + 1)
)

# ---------------------------------
# Avg Quantity Per Order
# ---------------------------------
rfm['AvgQuantityPerOrder'] = (
    rfm['TotalQuantity'] /
    rfm['Frequency']
)

print("\nADVANCED FEATURES CREATED!")

print(rfm.head())

# =========================
# OUTLIER DETECTION
# =========================

print("\nDETECTING OUTLIERS...")

# Boxplot for Monetary
plt.figure(figsize=(8,5))

plt.boxplot(rfm['Monetary'])

plt.title("Boxplot - Monetary")

plt.show()

# IQR Method
Q1 = rfm['Monetary'].quantile(0.25)
Q3 = rfm['Monetary'].quantile(0.75)

IQR = Q3 - Q1

lower_bound = Q1 - 1.5 * IQR
upper_bound = Q3 + 1.5 * IQR

print("\nLOWER BOUND:", lower_bound)
print("UPPER BOUND:", upper_bound)

# Remove outliers
rfm = rfm[
    (rfm['Monetary'] >= lower_bound) &
    (rfm['Monetary'] <= upper_bound)
]

print("\nAFTER REMOVING OUTLIERS:")
print(rfm.shape)

print("\nRFM DATA:")
print(rfm.head())

print("\nRFM SHAPE:")
print(rfm.shape)

# =========================

# SCALING RFM DATA

# =========================

print("\nSCALING RFM FEATURES...")

scaler = StandardScaler()

rfm_scaled = scaler.fit_transform(rfm[['Recency', 'Frequency', 'Monetary']])

# =========================

# ELBOW METHOD

# =========================

print("\nAPPLYING ELBOW METHOD...")

wcss = []

for k in range(1, 11):

    kmeans = KMeans(n_clusters=k, random_state=42)

    kmeans.fit(rfm_scaled)

    wcss.append(kmeans.inertia_)


# Plot elbow graph

plt.figure(figsize=(8,5))
plt.plot(range(1, 11), wcss, marker='o')
plt.xlabel("Number of Clusters (K)")
plt.ylabel("WCSS")
plt.title("Elbow Method")
plt.show()

# =========================

# K-MEANS CLUSTERING

# =========================

print("\nAPPLYING K-MEANS CLUSTERING...")

# Choose K = 3

kmeans = KMeans(n_clusters=3, random_state=42)

clusters = kmeans.fit_predict(rfm_scaled)

# Add cluster column

rfm['Cluster'] = clusters

print("\nCUSTOMER CLUSTERS:")
print(rfm.head())

# =========================

# VISUALIZATION

# =========================

print("\nVISUALIZING CLUSTERS...")

plt.figure(figsize=(8,5))

plt.scatter(
rfm['Recency'],
rfm['Monetary'],
c=rfm['Cluster']
)

plt.xlabel("Recency")
plt.ylabel("Monetary")
plt.title("Customer Segmentation")
plt.show()

# =========================

# CHURN PREDICTION

# LOGISTIC REGRESSION

# =========================

# =========================
# REALISTIC CHURN CREATION
# =========================

print("\nCREATING REALISTIC CHURN LABEL...")

# Customers with:
# low frequency + low monetary + high recency

rfm['Churn'] = np.where(
    (rfm['Recency'] > 180) &
    (rfm['Frequency'] <= 2) &
    (rfm['Monetary'] < 500),
    1,
    0
)

print(rfm[['Recency', 'Frequency', 'Monetary', 'Churn']].head())


# =========================
# RANDOM FOREST CLASSIFIER
# =========================

from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix

print("\nTRAINING RANDOM FOREST CLASSIFIER...")

X = rfm[[
    'Recency',
    'Frequency',
    'Monetary',
    'ProductDiversity',
    'PurchaseFrequencyRate',
    'TotalQuantity'
]]

y = rfm['Churn']

X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.2,
    random_state=42
)

# Model
rf_model = RandomForestClassifier(
    n_estimators=100,
    random_state=42
)

# Train
rf_model.fit(X_train, y_train)

# Predict
y_pred = rf_model.predict(X_test)

# Accuracy
accuracy = accuracy_score(y_test, y_pred)

print("\nRANDOM FOREST ACCURACY:")
print(accuracy)

# Classification report
print("\nCLASSIFICATION REPORT:")
print(classification_report(y_test, y_pred))

# Confusion matrix
print("\nCONFUSION MATRIX:")
print(confusion_matrix(y_test, y_pred))

# =========================

# SPENDING PREDICTION

# LINEAR REGRESSION

# =========================

# =========================
# RANDOM FOREST REGRESSOR
# =========================

from sklearn.ensemble import RandomForestRegressor

print("\nTRAINING RANDOM FOREST REGRESSOR...")

X = rfm[[
    'Recency',
    'Frequency',
    'ProductDiversity',
    'PurchaseFrequencyRate',
    'TotalQuantity'
]]

y = rfm['Monetary']

X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.2,
    random_state=42
)

# Model
rf_regressor = RandomForestRegressor(
    n_estimators=100,
    random_state=42
)

# Train
rf_regressor.fit(X_train, y_train)

# Predict
y_pred = rf_regressor.predict(X_test)

# Metrics
mae = mean_absolute_error(y_test, y_pred)
mse = mean_squared_error(y_test, y_pred)
rmse = np.sqrt(mse)
r2 = r2_score(y_test, y_pred)

print("\nRANDOM FOREST REGRESSION RESULTS:")
print("MAE:", mae)
print("MSE:", mse)
print("RMSE:", rmse)
print("R2 SCORE:", r2)

# =========================

# SAVE FINAL DATA

# =========================

print("\nSAVING FINAL DATASET...")

rfm.to_csv("final_customer_data.csv", index=False)

print("\nPROJECT COMPLETED SUCCESSFULLY!")
print("FINAL DATA SAVED AS: final_customer_data.csv")

# ======================================================

# PROJECT SUMMARY

# ======================================================

print("""

================ PROJECT SUMMARY ================

1. Cleaned retail transaction data
2. Created TotalPrice feature
3. Performed RFM analysis
4. Segmented customers using K-Means
5. Predicted customer churn
6. Predicted customer spending
7. Generated business insights

=================================================

""")

# =========================
# SAVE MODELS
# =========================

import joblib

print("\nSAVING MODELS...")

# Save classification model
joblib.dump(
    rf_model,
    "churn_model.pkl"
)

# Save regression model
joblib.dump(
    rf_regressor,
    "spending_model.pkl"
)

print("\nMODELS SAVED SUCCESSFULLY!")