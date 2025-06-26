import pandas as pd
import numpy as np
from scipy.stats import zscore
import matplotlib.pyplot as plt
import seaborn as sns

# Load dataset
df = pd.read_csv("9837467f-17ce-489f-8485-9192020fc508.csv")

# Detect outliers using IQR
Q1 = df['TotalSale'].quantile(0.25)
Q3 = df['TotalSale'].quantile(0.75)
IQR = Q3 - Q1
lower_bound = Q1 - 1.5 * IQR
upper_bound = Q3 + 1.5 * IQR
df['IQR_Outlier'] = ((df['TotalSale'] < lower_bound) | (df['TotalSale'] > upper_bound))

# Detect outliers using Z-score
df['Z_Score'] = zscore(df['TotalSale'])
df['Z_Outlier'] = df['Z_Score'].abs() > 3

# Outlier treatment
# 1. Capping at 1st and 99th percentiles
p1 = df['TotalSale'].quantile(0.01)
p99 = df['TotalSale'].quantile(0.99)
df['TotalSale_Capped'] = np.clip(df['TotalSale'], p1, p99)

# 2. Imputation using median
median_value = df.loc[~df['IQR_Outlier'], 'TotalSale'].median()
df['TotalSale_Imputed'] = df['TotalSale'].where(~df['IQR_Outlier'], median_value)

# Save cleaned dataset
df.to_csv("SalesDataset_Cleaned.csv", index=False)

# Visualize (optional)
sns.set(style="whitegrid")
plt.figure(figsize=(6, 4))
sns.boxplot(y=df['TotalSale'])
plt.title("Boxplot of TotalSale (Before Treatment)")
plt.savefig("boxplot_before.png")

plt.figure(figsize=(6, 4))
sns.boxplot(y=df['TotalSale_Capped'])
plt.title("Boxplot of TotalSale (Capped)")
plt.savefig("boxplot_capped.png")

plt.figure(figsize=(6, 4))
sns.boxplot(y=df['TotalSale_Imputed'])
plt.title("Boxplot of TotalSale (Imputed)")
plt.savefig("boxplot_imputed.png")
