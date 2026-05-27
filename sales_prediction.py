# =========================================
# IMPORT LIBRARIES
# =========================================

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.linear_model import LinearRegression
from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    r2_score
)

# =========================================
# LOAD DATASET
# =========================================

df = pd.read_csv("dataset/Advertising.csv")

# =========================================
# DATASET INFORMATION
# =========================================

print("FIRST 5 ROWS:")
print(df.head())

print("\nDATASET INFO:")
print(df.info())

print("\nSTATISTICAL SUMMARY:")
print(df.describe())

# =========================================
# DATASET SHAPE
# =========================================

print("\nDATASET SHAPE:")
print(df.shape)

# =========================================
# COLUMN NAMES
# =========================================

print("\nCOLUMN NAMES:")
print(df.columns)

# =========================================
# MISSING VALUES
# =========================================

print("\nMISSING VALUES:")
print(df.isnull().sum())

# =========================================
# DUPLICATE RECORDS
# =========================================

print("\nDUPLICATE ROWS:")
print(df.duplicated().sum())

# =========================================
# CORRELATION MATRIX
# =========================================

print("\nCORRELATION MATRIX:")
print(df.corr(numeric_only=True))

# =========================================
# SALES DISTRIBUTION
# =========================================

plt.figure(figsize=(8,5))

sns.histplot(df['Sales'], bins=20, kde=True)

plt.title("Distribution of Sales")
plt.xlabel("Sales")
plt.ylabel("Frequency")

plt.show()

# =========================================
# TV VS SALES
# =========================================

plt.figure(figsize=(8,5))

sns.scatterplot(x='TV', y='Sales', data=df)

plt.title("TV Advertising vs Sales")
plt.xlabel("TV Advertising Budget")
plt.ylabel("Sales")

plt.show()

# =========================================
# RADIO VS SALES
# =========================================

plt.figure(figsize=(8,5))

sns.scatterplot(x='Radio', y='Sales', data=df)

plt.title("Radio Advertising vs Sales")
plt.xlabel("Radio Advertising Budget")
plt.ylabel("Sales")

plt.show()

# =========================================
# NEWSPAPER VS SALES
# =========================================

plt.figure(figsize=(8,5))

sns.scatterplot(x='Newspaper', y='Sales', data=df)

plt.title("Newspaper Advertising vs Sales")
plt.xlabel("Newspaper Advertising Budget")
plt.ylabel("Sales")

plt.show()

# =========================================
# CORRELATION HEATMAP
# =========================================

plt.figure(figsize=(8,6))

sns.heatmap(
    df.corr(numeric_only=True),
    annot=True,
    cmap='coolwarm'
)

plt.title("Correlation Heatmap")

plt.show()

# Remove unnecessary column if present
if 'Unnamed: 0' in df.columns:
    df = df.drop('Unnamed: 0', axis=1)

    # =========================================
# FEATURES (X)
# =========================================

X = df[['TV', 'Radio', 'Newspaper']]

# =========================================
# TARGET (y)
# =========================================

y = df['Sales']

print("\nFEATURES (X):")
print(X.head())

print("\nTARGET (y):")
print(y.head())

print("\nFEATURE SHAPE:")
print(X.shape)

print("\nTARGET SHAPE:")
print(y.shape)

from sklearn.model_selection import train_test_split

# =========================================
# TRAIN TEST SPLIT
# =========================================

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)

print("\nTRAINING FEATURES SHAPE:")
print(X_train.shape)

print("\nTESTING FEATURES SHAPE:")
print(X_test.shape)

print("\nTRAINING TARGET SHAPE:")
print(y_train.shape)

print("\nTESTING TARGET SHAPE:")
print(y_test.shape)

# =========================================
# LINEAR REGRESSION MODEL
# =========================================

model = LinearRegression()

model.fit(X_train, y_train)

print("\nMODEL TRAINED SUCCESSFULLY")

print("\nMODEL COEFFICIENTS:")
print(model.coef_)

print("\nMODEL INTERCEPT:")
print(model.intercept_)

# =========================================
# SALES PREDICTION
# =========================================

y_pred = model.predict(X_test)

print("\nPREDICTED SALES:")
print(y_pred)

# =========================================
# ACTUAL VS PREDICTED
# =========================================

comparison = pd.DataFrame({
    'Actual Sales': y_test.values,
    'Predicted Sales': y_pred
})

print("\nACTUAL VS PREDICTED SALES:")
print(comparison.head(10))

# =========================================
# MODEL EVALUATION
# =========================================

mae = mean_absolute_error(y_test, y_pred)
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print("\nMEAN ABSOLUTE ERROR:")
print(mae)

print("\nMEAN SQUARED ERROR:")
print(mse)

print("\nR2 SCORE:")
print(r2)

# =========================================
# ACTUAL VS PREDICTED GRAPH
# =========================================

plt.figure(figsize=(8,5))

plt.scatter(y_test, y_pred)

plt.xlabel("Actual Sales")
plt.ylabel("Predicted Sales")

plt.title("Actual Sales vs Predicted Sales")

plt.savefig("actual_vs_predicted.png")

plt.show()

# =========================================
# BUSINESS INSIGHTS
# =========================================

print("\nBUSINESS INSIGHTS")

print("1. TV advertising has the strongest impact on sales.")

print("2. Radio advertising also contributes positively to sales.")

print("3. Newspaper advertising has comparatively less influence on sales.")

print("4. Increasing advertising budget generally increases sales.")