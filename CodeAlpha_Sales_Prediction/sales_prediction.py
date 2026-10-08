import pandas as pd
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

# ==========================================
# 1. Load Dataset
# ==========================================

df = pd.read_csv("data/Advertising.csv")

print("Original Dataset Shape:")
print(df.shape)

print("\nColumn Names:")
print(df.columns.tolist())

print("\nFirst 5 Rows:")
print(df.head())

# ==========================================
# 2. Data Cleaning
# ==========================================

# Remove unnecessary index column
if "Unnamed: 0" in df.columns:
    df = df.drop("Unnamed: 0", axis=1)

print("\nMissing Values:")
print(df.isnull().sum())

# Remove duplicate rows
df = df.drop_duplicates()

print("\nDataset Shape After Cleaning:")
print(df.shape)

# ==========================================
# 3. Dataset Information
# ==========================================

print("\nDataset Information:")
print(df.info())

print("\nStatistical Summary:")
print(df.describe())

# ==========================================
# 4. Feature Selection
# ==========================================

X = df[["TV", "Radio", "Newspaper"]]
y = df["Sales"]

print("\nInput Features:")
print(X.head())

print("\nTarget Variable:")
print(y.head())

# ==========================================
# 5. Train-Test Split
# ==========================================

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)

print("\nTraining Data Shape:")
print(X_train.shape)

print("\nTesting Data Shape:")
print(X_test.shape)

# ==========================================
# 6. Train Linear Regression Model
# ==========================================

model = LinearRegression()

model.fit(X_train, y_train)

# ==========================================
# 7. Make Predictions
# ==========================================

y_pred = model.predict(X_test)

print("\nActual Sales:")
print(y_test.values)

print("\nPredicted Sales:")
print(y_pred)

# ==========================================
# 8. Model Evaluation
# ==========================================

mae = mean_absolute_error(y_test, y_pred)
mse = mean_squared_error(y_test, y_pred)
rmse = mse ** 0.5
r2 = r2_score(y_test, y_pred)

print("\n========== MODEL EVALUATION ==========")

print("Mean Absolute Error:", mae)
print("Mean Squared Error:", mse)
print("Root Mean Squared Error:", rmse)
print("R2 Score:", r2)

# ==========================================
# 9. Model Coefficients
# ==========================================

print("\n========== FEATURE IMPACT ==========")

print("TV Coefficient:", model.coef_[0])
print("Radio Coefficient:", model.coef_[1])
print("Newspaper Coefficient:", model.coef_[2])
print("Intercept:", model.intercept_)

# ==========================================
# 10. Actual vs Predicted Sales
# ==========================================

comparison = pd.DataFrame({
    "Actual Sales": y_test.values,
    "Predicted Sales": y_pred
})

comparison.to_csv(
    "output/actual_vs_predicted.csv",
    index=False
)

print("\nActual vs Predicted values saved.")

# ==========================================
# 11. Prediction Graph
# ==========================================

plt.figure(figsize=(8, 6))

plt.scatter(y_test, y_pred)

plt.xlabel("Actual Sales")
plt.ylabel("Predicted Sales")
plt.title("Actual vs Predicted Sales")

plt.savefig(
    "output/actual_vs_predicted.png",
    dpi=300,
    bbox_inches="tight"
)

plt.show()

# ==========================================
# 12. Advertising Impact Graph
# ==========================================

plt.figure(figsize=(8, 6))

plt.bar(
    ["TV", "Radio", "Newspaper"],
    model.coef_
)

plt.xlabel("Advertising Platform")
plt.ylabel("Regression Coefficient")
plt.title("Impact of Advertising on Sales")

plt.savefig(
    "output/advertising_impact.png",
    dpi=300,
    bbox_inches="tight"
)

plt.show()

# ==========================================
# 13. Sales Distribution
# ==========================================

plt.figure(figsize=(8, 6))

plt.hist(df["Sales"], bins=15)

plt.xlabel("Sales")
plt.ylabel("Frequency")
plt.title("Sales Distribution")

plt.savefig(
    "output/sales_distribution.png",
    dpi=300,
    bbox_inches="tight"
)

plt.show()

print("\n====================================")
print("TASK 4 SALES PREDICTION COMPLETED")
print("====================================")