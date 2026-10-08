import pandas as pd
import matplotlib.pyplot as plt
import os

# ==========================================
# CODEALPHA - UNEMPLOYMENT ANALYSIS
# ==========================================

# Create output folder if it does not exist
os.makedirs("output", exist_ok=True)

# ------------------------------------------
# 1. LOAD DATASET
# ------------------------------------------

df = pd.read_csv(r"data\Unemployment in India.csv")

print("\nOriginal Dataset Shape:")
print(df.shape)

# ------------------------------------------
# 2. CLEAN COLUMN NAMES
# ------------------------------------------

df.columns = df.columns.str.strip()

print("\nColumn Names:")
print(df.columns)

# ------------------------------------------
# 3. REMOVE COMPLETELY EMPTY ROWS
# ------------------------------------------

df = df.dropna(how="all")

print("\nDataset Shape After Removing Empty Rows:")
print(df.shape)

# ------------------------------------------
# 4. CLEAN TEXT COLUMNS
# ------------------------------------------

df["Frequency"] = df["Frequency"].str.strip()
df["Region"] = df["Region"].str.strip()
df["Area"] = df["Area"].str.strip()

# ------------------------------------------
# 5. CONVERT DATE COLUMN
# ------------------------------------------

df["Date"] = pd.to_datetime(df["Date"], dayfirst=True)

# ------------------------------------------
# 6. CHECK MISSING VALUES
# ------------------------------------------

print("\nMissing Values:")
print(df.isnull().sum())

# ------------------------------------------
# 7. BASIC STATISTICS
# ------------------------------------------

print("\nBasic Statistics:")
print(df.describe())

# ------------------------------------------
# 8. SAVE CLEANED DATASET
# ------------------------------------------

df.to_csv(
    r"data\Unemployment in India_cleaned.csv",
    index=False
)

print("\nCleaned dataset saved successfully.")

# ==========================================
# 9. OVERALL MONTHLY UNEMPLOYMENT
# ==========================================

monthly_unemployment = (
    df.groupby("Date")["Estimated Unemployment Rate (%)"]
    .mean()
)

print("\nOverall Monthly Unemployment:")
print(monthly_unemployment.round(2))

# ------------------------------------------
# Overall Trend Graph
# ------------------------------------------

plt.figure(figsize=(12, 6))

plt.plot(
    monthly_unemployment.index,
    monthly_unemployment.values,
    marker="o"
)

plt.title("Average Unemployment Rate in India (2019-2020)")
plt.xlabel("Date")
plt.ylabel("Unemployment Rate (%)")
plt.grid(True)
plt.xticks(rotation=45)
plt.tight_layout()

plt.savefig(
    r"output\overall_unemployment_trend.png",
    dpi=300,
    bbox_inches="tight"
)

plt.close()

# ==========================================
# 10. COVID-19 ANALYSIS
# ==========================================

pre_covid = df[df["Date"] < "2020-03-01"]

covid_period = df[df["Date"] >= "2020-03-01"]

pre_covid_avg = (
    pre_covid["Estimated Unemployment Rate (%)"].mean()
)

covid_avg = (
    covid_period["Estimated Unemployment Rate (%)"].mean()
)

increase = covid_avg - pre_covid_avg

print("\nCOVID-19 Analysis")
print("-------------------------")
print("Pre-COVID Average:",
      round(pre_covid_avg, 2), "%")

print("COVID Period Average:",
      round(covid_avg, 2), "%")

print("Increase:",
      round(increase, 2),
      "percentage points")

# ------------------------------------------
# COVID Comparison Graph
# ------------------------------------------

periods = ["Pre-COVID", "COVID Period"]

rates = [
    pre_covid_avg,
    covid_avg
]

plt.figure(figsize=(8, 6))

plt.bar(periods, rates)

plt.title(
    "Average Unemployment Rate: "
    "Pre-COVID vs COVID Period"
)

plt.ylabel("Unemployment Rate (%)")
plt.grid(axis="y", alpha=0.3)
plt.tight_layout()

plt.savefig(
    r"output\covid_unemployment_comparison.png",
    dpi=300,
    bbox_inches="tight"
)

plt.close()

# ==========================================
# 11. REGIONAL ANALYSIS
# ==========================================

regional_unemployment = (
    df.groupby("Region")[
        "Estimated Unemployment Rate (%)"
    ]
    .mean()
    .sort_values(ascending=False)
)

print("\nAverage Unemployment Rate by Region:")
print(regional_unemployment.round(2))

# ------------------------------------------
# Regional Graph
# ------------------------------------------

plt.figure(figsize=(12, 8))

plt.barh(
    regional_unemployment.index,
    regional_unemployment.values
)

plt.title(
    "Average Unemployment Rate by Region (2019-2020)"
)

plt.xlabel("Average Unemployment Rate (%)")
plt.grid(axis="x", alpha=0.3)

plt.gca().invert_yaxis()

plt.tight_layout()

plt.savefig(
    r"output\regional_unemployment.png",
    dpi=300,
    bbox_inches="tight"
)

plt.close()

# ==========================================
# 12. RURAL VS URBAN ANALYSIS
# ==========================================

area_unemployment = (
    df.groupby("Area")[
        "Estimated Unemployment Rate (%)"
    ].mean()
)

print("\nRural vs Urban Unemployment:")
print(area_unemployment.round(2))

# ------------------------------------------
# Rural vs Urban Graph
# ------------------------------------------

plt.figure(figsize=(8, 6))

plt.bar(
    area_unemployment.index,
    area_unemployment.values
)

plt.title(
    "Average Unemployment Rate: Rural vs Urban"
)

plt.xlabel("Area")
plt.ylabel("Unemployment Rate (%)")
plt.grid(axis="y", alpha=0.3)

plt.tight_layout()

plt.savefig(
    r"output\rural_vs_urban_unemployment.png",
    dpi=300,
    bbox_inches="tight"
)

plt.close()

# ==========================================
# 13. MONTHLY PATTERN / SEASONAL ANALYSIS
# ==========================================

df["Month"] = df["Date"].dt.month

monthly_pattern = (
    df.groupby("Month")[
        "Estimated Unemployment Rate (%)"
    ].mean()
)

print("\nMonthly Unemployment Pattern:")
print(monthly_pattern.round(2))

# ------------------------------------------
# Monthly Pattern Graph
# ------------------------------------------

plt.figure(figsize=(10, 6))

plt.plot(
    monthly_pattern.index,
    monthly_pattern.values,
    marker="o"
)

plt.title("Monthly Unemployment Pattern")
plt.xlabel("Month")
plt.ylabel("Average Unemployment Rate (%)")

plt.xticks(range(1, 13))

plt.grid(True)
plt.tight_layout()

plt.savefig(
    r"output\monthly_unemployment_pattern.png",
    dpi=300,
    bbox_inches="tight"
)

plt.close()

# ==========================================
# 14. FINAL INFORMATION
# ==========================================

print("\n====================================")
print("ANALYSIS COMPLETED SUCCESSFULLY")
print("====================================")

print("\nDataset:")
print("Rows:", len(df))
print("Columns:", len(df.columns))

print("\nDate Range:")
print(df["Date"].min(), "to", df["Date"].max())

print("\nNumber of Regions:")
print(df["Region"].nunique())

print("\nOutput files created:")
print("1. overall_unemployment_trend.png")
print("2. covid_unemployment_comparison.png")
print("3. regional_unemployment.png")
print("4. rural_vs_urban_unemployment.png")
print("5. monthly_unemployment_pattern.png")
print("6. Unemployment in India_cleaned.csv")

print("\nProject completed.")