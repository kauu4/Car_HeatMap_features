import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

# 1. Load dataset
df = pd.read_csv("data/indian_cars.csv")


# 2. Extract numeric values from text columns (e.g., "1197 cc" -> 1197.0)
def extract_number(series):
  return (
      series.astype(str)
      .str.replace(",", "", regex=False)
      .str.extract(r"(\d+\.?\d*)")[0]
      .astype(float)
  )


clean_df = pd.DataFrame({
    "Price_INR": pd.to_numeric(df["Price"], errors="coerce"),
    "Year": pd.to_numeric(df["Year"], errors="coerce"),
    "Kilometers_Driven": pd.to_numeric(df["Kilometer"], errors="coerce"),
    "Engine_CC": extract_number(df["Engine"]),
    "Max_Power_bhp": extract_number(df["Max Power"]),
    "Max_Torque_Nm": extract_number(df["Max Torque"]),
    "Length_mm": pd.to_numeric(df["Length"], errors="coerce"),
    "Width_mm": pd.to_numeric(df["Width"], errors="coerce"),
    "Height_mm": pd.to_numeric(df["Height"], errors="coerce"),
    "Seating": pd.to_numeric(df["Seating Capacity"], errors="coerce"),
    "Fuel_Tank_L": pd.to_numeric(df["Fuel Tank Capacity"], errors="coerce"),
}).dropna()

# 3. Compute Pearson correlation matrix
corr_matrix = clean_df.corr(method="pearson")

# 4. Plot and save the Heatmap
plt.figure(figsize=(11, 9))
mask = np.triu(np.ones_like(corr_matrix, dtype=bool))

sns.heatmap(
    corr_matrix,
    mask=mask,
    annot=True,
    fmt=".2f",
    cmap="coolwarm",
    vmin=-1,
    vmax=1,
    center=0,
    linewidths=0.5,
)
plt.title("Correlation Matrix: Indian Car Specifications & Pricing")
plt.tight_layout()
plt.savefig("output/correlation_heatmap.png", dpi=300)
print("Saved heatmap to output/correlation_heatmap.png\n")

# 5. Print strongest positive and negative correlations
pairs = corr_matrix.where(np.tril(np.ones(corr_matrix.shape), k=-1).astype(bool))
sorted_pairs = pairs.unstack().dropna().sort_values(ascending=False)

print("=== TOP 5 STRONGEST POSITIVE CORRELATIONS ===")
print(sorted_pairs.head(5))

print("\n=== TOP 5 STRONGEST NEGATIVE CORRELATIONS ===")
print(sorted_pairs.tail(5))