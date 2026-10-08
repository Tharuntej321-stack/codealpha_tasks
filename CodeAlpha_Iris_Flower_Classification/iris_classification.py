# ============================================================
# CODEALPHA DATA SCIENCE INTERNSHIP
# TASK 1: IRIS FLOWER CLASSIFICATION
# ============================================================

from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix
)

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


print("=" * 60)
print("IRIS FLOWER CLASSIFICATION")
print("=" * 60)


# ------------------------------------------------------------
# 1. LOAD IRIS DATASET
# ------------------------------------------------------------

iris = load_iris()

print("\nDataset loaded successfully!")


# ------------------------------------------------------------
# 2. CREATE DATAFRAME
# ------------------------------------------------------------

df = pd.DataFrame(
    iris.data,
    columns=iris.feature_names
)

df["species"] = iris.target


print("\nFirst 5 records:")
print(df.head())


# ------------------------------------------------------------
# 3. DATASET INFORMATION
# ------------------------------------------------------------

print("\n" + "=" * 60)
print("DATASET INFORMATION")
print("=" * 60)

print("\nDataset shape:")
print(df.shape)

print("\nColumn names:")
print(df.columns.tolist())

print("\nMissing values:")
print(df.isnull().sum())

print("\nStatistical summary:")
print(df.describe())


# ------------------------------------------------------------
# 4. IRIS SPECIES
# ------------------------------------------------------------

print("\n" + "=" * 60)
print("IRIS SPECIES")
print("=" * 60)

for i, species in enumerate(iris.target_names):
    print(i, "=", species)


# ------------------------------------------------------------
# 5. FEATURES AND TARGET
# ------------------------------------------------------------

X = iris.data
y = iris.target

print("\nFeatures:")
print(iris.feature_names)

print("\nTarget:")
print("Iris flower species")


# ------------------------------------------------------------
# 6. TRAIN TEST SPLIT
# ------------------------------------------------------------

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42,
    stratify=y
)


print("\n" + "=" * 60)
print("TRAIN TEST SPLIT")
print("=" * 60)

print("Training samples:", len(X_train))
print("Testing samples:", len(X_test))


# ------------------------------------------------------------
# 7. CREATE MACHINE LEARNING MODEL
# ------------------------------------------------------------

model = LogisticRegression(max_iter=200)

print("\nMachine learning model created:")
print("Logistic Regression")


# ------------------------------------------------------------
# 8. TRAIN MODEL
# ------------------------------------------------------------

model.fit(X_train, y_train)

print("\nModel training completed successfully!")


# ------------------------------------------------------------
# 9. MAKE PREDICTIONS
# ------------------------------------------------------------

y_pred = model.predict(X_test)

print("\nPredictions generated successfully!")


# ------------------------------------------------------------
# 10. MODEL ACCURACY
# ------------------------------------------------------------

accuracy = accuracy_score(y_test, y_pred)

print("\n" + "=" * 60)
print("MODEL ACCURACY")
print("=" * 60)

print(f"Accuracy: {accuracy * 100:.2f}%")


# ------------------------------------------------------------
# 11. CLASSIFICATION REPORT
# ------------------------------------------------------------

print("\n" + "=" * 60)
print("CLASSIFICATION REPORT")
print("=" * 60)

report = classification_report(
    y_test,
    y_pred,
    target_names=iris.target_names
)

print(report)


# ------------------------------------------------------------
# 12. CONFUSION MATRIX
# ------------------------------------------------------------

cm = confusion_matrix(y_test, y_pred)

print("\n" + "=" * 60)
print("CONFUSION MATRIX")
print("=" * 60)

print(cm)


# ------------------------------------------------------------
# 13. CONFUSION MATRIX VISUALIZATION
# ------------------------------------------------------------

plt.figure(figsize=(8, 6))

sns.heatmap(
    cm,
    annot=True,
    fmt="d",
    cmap="Blues",
    xticklabels=iris.target_names,
    yticklabels=iris.target_names
)

plt.xlabel("Predicted Species")
plt.ylabel("Actual Species")
plt.title("Iris Flower Classification - Confusion Matrix")

plt.tight_layout()

plt.savefig("confusion_matrix.png")

print("\nConfusion matrix saved as:")
print("confusion_matrix.png")

plt.show()


# ------------------------------------------------------------
# 14. IRIS DATA VISUALIZATION
# ------------------------------------------------------------

plt.figure(figsize=(8, 6))

sns.scatterplot(
    x=iris.data[:, 2],
    y=iris.data[:, 3],
    hue=iris.target_names[iris.target],
    s=80
)

plt.xlabel("Petal Length")
plt.ylabel("Petal Width")
plt.title("Iris Flower Classification")

plt.tight_layout()

plt.savefig("iris_visualization.png")

print("\nIris visualization saved as:")
print("iris_visualization.png")

plt.show()


# ------------------------------------------------------------
# 15. NEW FLOWER PREDICTION
# ------------------------------------------------------------

new_flower = [[5.1, 3.5, 1.4, 0.2]]

prediction = model.predict(new_flower)

predicted_species = iris.target_names[prediction[0]]


print("\n" + "=" * 60)
print("NEW FLOWER PREDICTION")
print("=" * 60)

print("Input measurements:")
print("Sepal Length: 5.1")
print("Sepal Width : 3.5")
print("Petal Length: 1.4")
print("Petal Width : 0.2")

print("\nPredicted Species:", predicted_species)


# ------------------------------------------------------------
# 16. COMPLETION MESSAGE
# ------------------------------------------------------------

print("\n" + "=" * 60)
print("TASK 1 COMPLETED SUCCESSFULLY")
print("=" * 60)

print("\nGenerated files:")
print("1. confusion_matrix.png")
print("2. iris_visualization.png")

print("\nThank you!")