Python 3.14.0 (tags/v3.14.0:ebf955d, Oct  7 2025, 10:15:03) [MSC v.1944 64 bit (AMD64)] on win32
Enter "help" below or click "Help" above for more information.
data = pd.read_csv("train.csv")
print("Dataset Loaded Successfully ✅")
print(data.head())
# 🏠 House Price Prediction
# Project Type: Regression (Supervised Machine Learning)
# -------------------------------------------------------

# 📦 Step 1: Import Required Libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

# -------------------------------------------------------
# 📥 Step 2: Load the Dataset
# Download dataset from:
# https://www.kaggle.com/c/house-prices-advanced-regression-techniques
# (Use train.csv file)
# -------------------------------------------------------

data = pd.read_csv("train.csv")   # Ensure 'train.csv' is in your working directory
print("Dataset Loaded Successfully ✅")
print("Shape of the dataset:", data.shape)

# -------------------------------------------------------
# 👀 Step 3: Explore the Dataset
print("\nFirst 5 rows of the dataset:")
print(data.head())

print("\nDataset Information:")
print(data.info())

print("\nMissing Values in Each Column:")
print(data.isnull().sum().sort_values(ascending=False).head(10))

# -------------------------------------------------------
# 🧹 Step 4: Data Preprocessing
# Select relevant features for prediction
features = ['OverallQual', 'GrLivArea', 'GarageCars', 'GarageArea', 
            'TotalBsmtSF', 'FullBath', 'YearBuilt']
target = 'SalePrice'

df = data[features + [target]]

# Fill missing values with mean
df = df.fillna(df.mean())

# -------------------------------------------------------
# 📊 Step 5: Exploratory Data Analysis (Optional Visuals)
plt.figure(figsize=(8, 5))
sns.heatmap(df.corr(), annot=True, cmap='coolwarm')
plt.title("Correlation Between Features")
plt.show()

# -------------------------------------------------------
# ✂️ Step 6: Split Data into Training and Testing Sets
X = df[features]
y = df[target]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

print("Training Set Shape:", X_train.shape)
print("Testing Set Shape:", X_test.shape)

# -------------------------------------------------------
# 🧠 Step 7: Train Models

# Linear Regression
lin_reg = LinearRegression()
lin_reg.fit(X_train, y_train)

# Decision Tree Regressor
tree_reg = DecisionTreeRegressor(random_state=42)
tree_reg.fit(X_train, y_train)

print("\n✅ Models Trained Successfully")

# -------------------------------------------------------
# 📈 Step 8: Model Evaluation

# Linear Regression Predictions
y_pred_lin = lin_reg.predict(X_test)

# Decision Tree Predictions
y_pred_tree = tree_reg.predict(X_test)

# Evaluation Function
... def evaluate_model(y_true, y_pred, model_name):
...     mae = mean_absolute_error(y_true, y_pred)
...     mse = mean_squared_error(y_true, y_pred)
...     rmse = np.sqrt(mse)
...     r2 = r2_score(y_true, y_pred)
...     print(f"\n📊 {model_name} Evaluation:")
...     print(f"Mean Absolute Error: {mae:.2f}")
...     print(f"Root Mean Squared Error: {rmse:.2f}")
...     print(f"R² Score: {r2:.4f}")
... 
... # Evaluate both models
... evaluate_model(y_test, y_pred_lin, "Linear Regression")
... evaluate_model(y_test, y_pred_tree, "Decision Tree Regressor")
... 
... # -------------------------------------------------------
... # 🔮 Step 9: Predict New House Prices
... # Example Input: [OverallQual, GrLivArea, GarageCars, GarageArea, TotalBsmtSF, FullBath, YearBuilt]
... 
... sample_input = np.array([[7, 1800, 2, 500, 1000, 2, 2005]])
... predicted_price = lin_reg.predict(sample_input)
... print("\n💰 Predicted House Price (Linear Regression): ₹", round(predicted_price[0], 2))
... 
... sample_input_tree = np.array([[8, 2500, 2, 600, 1200, 2, 2010]])
... predicted_price_tree = tree_reg.predict(sample_input_tree)
... print("💰 Predicted House Price (Decision Tree): ₹", round(predicted_price_tree[0], 2))
... 
... # -------------------------------------------------------
... # 📉 Step 10: Visualization of Predictions
... plt.figure(figsize=(8,5))
... plt.scatter(y_test, y_pred_lin, color='blue', label='Linear Regression')
... plt.scatter(y_test, y_pred_tree, color='green', label='Decision Tree')
... plt.xlabel("Actual Prices")
... plt.ylabel("Predicted Prices")
... plt.title("Actual vs Predicted House Prices")
... plt.legend()
... plt.show()
