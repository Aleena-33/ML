import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.datasets import fetch_openml
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures
from sklearn.metrics import mean_squared_error, r2_score

# Load Auto MPG Dataset
df = fetch_openml(name="autoMpg", version=1, as_frame=True).frame

# Check whether target column is 'mpg' or 'class'
if 'mpg' in df.columns:
    target = 'mpg'
elif 'class' in df.columns:
    target = 'class'
else:
    print("Available columns:", df.columns)
    raise ValueError("MPG column not found!")

# Select required columns and preprocess
df = df[['displacement', target]]
df['displacement'] = pd.to_numeric(df['displacement'], errors='coerce')
df[target] = pd.to_numeric(df[target], errors='coerce')
df.dropna(inplace=True)

# Feature and Target
X = df[['displacement']]
y = df[target]

# Train-Test Split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Linear Regression
lr = LinearRegression()
lr.fit(X_train, y_train)
y_pred_lr = lr.predict(X_test)

# Polynomial Regression
poly = PolynomialFeatures(degree=2)

X_train_poly = poly.fit_transform(X_train)
X_test_poly = poly.transform(X_test)

pr = LinearRegression()
pr.fit(X_train_poly, y_train)
y_pred_pr = pr.predict(X_test_poly)

# Performance
print("Linear Regression")
print("MSE :", mean_squared_error(y_test, y_pred_lr))
print("R2  :", r2_score(y_test, y_pred_lr))

print("\nPolynomial Regression")
print("MSE :", mean_squared_error(y_test, y_pred_pr))
print("R2  :", r2_score(y_test, y_pred_pr))

# Visualization Data
x = np.linspace(X.min().values[0], X.max().values[0], 300).reshape(-1, 1)

# ===============================
# Graph 1: Linear Regression
# ===============================
plt.figure(figsize=(8,5))

plt.scatter(X, y, color='gray', s=20, label='Actual Data')
plt.plot(x, lr.predict(x), color='blue', linewidth=2, label='Linear Regression')

plt.xlabel("Engine Displacement")
plt.ylabel("Miles Per Gallon (MPG)")
plt.title("Linear Regression")
plt.legend()
plt.grid(True)

plt.show()


# ===============================
# Graph 2: Polynomial Regression
# ===============================
plt.figure(figsize=(8,5))

plt.scatter(X, y, color='gray', s=20, label='Actual Data')
plt.plot(x, pr.predict(poly.transform(x)), color='red', linewidth=2, label='Polynomial Regression')

plt.xlabel("Engine Displacement")
plt.ylabel("Miles Per Gallon (MPG)")
plt.title("Polynomial Regression")
plt.legend()
plt.grid(True)

plt.show()
