import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

import joblib


# ============================================================
# 1. LOAD DATASET
# ============================================================

file_path = "data/car data.csv"

df = pd.read_csv(file_path)

print("=" * 60)
print("CAR PRICE PREDICTION USING MACHINE LEARNING")
print("=" * 60)

print("\nOriginal Dataset Shape:")
print(df.shape)

print("\nColumn Names:")
print(df.columns.tolist())

print("\nFirst 5 Rows:")
print(df.head())


# ============================================================
# 2. DATA INFORMATION
# ============================================================

print("\nDataset Information:")
print(df.info())

print("\nMissing Values:")
print(df.isnull().sum())

print("\nDuplicate Rows:")
print(df.duplicated().sum())


# ============================================================
# 3. REMOVE DUPLICATES
# ============================================================

df = df.drop_duplicates()

print("\nDataset Shape After Removing Duplicates:")
print(df.shape)


# ============================================================
# 4. DESCRIPTIVE STATISTICS
# ============================================================

print("\nDescriptive Statistics:")
print(df.describe())


# ============================================================
# 5. EXPLORATORY DATA ANALYSIS
# ============================================================

plt.figure(figsize=(10, 6))
sns.histplot(df["Selling_Price"], kde=True)

plt.title("Distribution of Car Selling Prices")
plt.xlabel("Selling Price")
plt.ylabel("Frequency")

plt.tight_layout()
plt.savefig("selling_price_distribution.png")
plt.show()


# ============================================================
# 6. PRICE VS PRESENT PRICE
# ============================================================

plt.figure(figsize=(10, 6))

sns.scatterplot(
    data=df,
    x="Present_Price",
    y="Selling_Price"
)

plt.title("Present Price vs Selling Price")
plt.xlabel("Present Price")
plt.ylabel("Selling Price")

plt.tight_layout()
plt.savefig("present_vs_selling_price.png")
plt.show()


# ============================================================
# 7. PRICE VS KILOMETERS DRIVEN
# ============================================================

plt.figure(figsize=(10, 6))

sns.scatterplot(
    data=df,
    x="Driven_kms",
    y="Selling_Price"
)

plt.title("Kilometers Driven vs Selling Price")
plt.xlabel("Driven Kilometers")
plt.ylabel("Selling Price")

plt.tight_layout()
plt.savefig("kms_vs_selling_price.png")
plt.show()


# ============================================================
# 8. PREPARE FEATURES AND TARGET
# ============================================================

X = df.drop("Selling_Price", axis=1)

y = df["Selling_Price"]


# ============================================================
# 9. IDENTIFY CATEGORICAL AND NUMERICAL FEATURES
# ============================================================

categorical_features = [
    "Fuel_Type",
    "Selling_type",
    "Transmission"
]

numerical_features = [
    "Year",
    "Present_Price",
    "Driven_kms",
    "Owner"
]

print("\nCategorical Features:")
print(categorical_features)

print("\nNumerical Features:")
print(numerical_features)


# ============================================================
# 10. PREPROCESSING
# ============================================================

preprocessor = ColumnTransformer(
    transformers=[
        (
            "categorical",
            OneHotEncoder(handle_unknown="ignore"),
            categorical_features
        ),
        (
            "numerical",
            "passthrough",
            numerical_features
        )
    ]
)


# ============================================================
# 11. TRAIN TEST SPLIT
# ============================================================

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42
)

print("\nTraining Data:")
print(X_train.shape)

print("\nTesting Data:")
print(X_test.shape)


# ============================================================
# 12. LINEAR REGRESSION MODEL
# ============================================================

linear_model = Pipeline(
    steps=[
        ("preprocessor", preprocessor),
        ("model", LinearRegression())
    ]
)

linear_model.fit(X_train, y_train)

linear_predictions = linear_model.predict(X_test)


# ============================================================
# 13. LINEAR REGRESSION EVALUATION
# ============================================================

linear_mae = mean_absolute_error(
    y_test,
    linear_predictions
)

linear_mse = mean_squared_error(
    y_test,
    linear_predictions
)

linear_rmse = np.sqrt(linear_mse)

linear_r2 = r2_score(
    y_test,
    linear_predictions
)

print("\n" + "=" * 60)
print("LINEAR REGRESSION RESULTS")
print("=" * 60)

print("MAE :", linear_mae)
print("MSE :", linear_mse)
print("RMSE:", linear_rmse)
print("R2  :", linear_r2)


# ============================================================
# 14. RANDOM FOREST REGRESSION
# ============================================================

random_forest_model = Pipeline(
    steps=[
        ("preprocessor", preprocessor),
        (
            "model",
            RandomForestRegressor(
                n_estimators=200,
                random_state=42
            )
        )
    ]
)

random_forest_model.fit(X_train, y_train)

rf_predictions = random_forest_model.predict(X_test)


# ============================================================
# 15. RANDOM FOREST EVALUATION
# ============================================================

rf_mae = mean_absolute_error(
    y_test,
    rf_predictions
)

rf_mse = mean_squared_error(
    y_test,
    rf_predictions
)

rf_rmse = np.sqrt(rf_mse)

rf_r2 = r2_score(
    y_test,
    rf_predictions
)

print("\n" + "=" * 60)
print("RANDOM FOREST RESULTS")
print("=" * 60)

print("MAE :", rf_mae)
print("MSE :", rf_mse)
print("RMSE:", rf_rmse)
print("R2  :", rf_r2)


# ============================================================
# 16. ACTUAL VS PREDICTED
# ============================================================

results = pd.DataFrame({
    "Actual Price": y_test.values,
    "Linear Regression": linear_predictions,
    "Random Forest": rf_predictions
})

print("\nActual vs Predicted Prices:")
print(results.head(10))


# ============================================================
# 17. ACTUAL VS PREDICTED GRAPH
# ============================================================

plt.figure(figsize=(10, 6))

plt.scatter(
    y_test,
    rf_predictions,
    alpha=0.7
)

plt.xlabel("Actual Selling Price")
plt.ylabel("Predicted Selling Price")

plt.title("Actual vs Predicted Car Prices - Random Forest")

plt.tight_layout()
plt.savefig("actual_vs_predicted.png")
plt.show()


# ============================================================
# 18. SAVE RESULTS
# ============================================================

results.to_csv(
    "prediction_results.csv",
    index=False
)

print("\nPrediction results saved as:")
print("prediction_results.csv")


# ============================================================
# 19. SAVE MODEL
# ============================================================

joblib.dump(
    random_forest_model,
    "car_price_model.pkl"
)

print("\nModel saved as:")
print("car_price_model.pkl")


# ============================================================
# 20. FINAL MESSAGE
# ============================================================

print("\n" + "=" * 60)
print("TASK 3 COMPLETED SUCCESSFULLY")
print("=" * 60)