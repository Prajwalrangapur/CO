#1
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.datasets import fetch_california_housing

# Load dataset
data = fetch_california_housing(as_frame=True)
df = data.frame

# Histograms
df.hist(figsize=(10, 8))
plt.suptitle("Histograms of Numerical Features")
plt.show()

# Box Plots
df.plot(kind='box', subplots=True, layout=(3, 3), figsize=(10, 8))
plt.suptitle("Box Plots of Numerical Features")
plt.show()

# Outlier Detection using IQR
print("Outliers in each feature:\n")
for col in df.columns:
    Q1 = df[col].quantile(0.25)
    Q3 = df[col].quantile(0.75)
    IQR = Q3 - Q1

    lower = Q1 - 1.5 * IQR
    upper = Q3 + 1.5 * IQR

    outliers = df[(df[col] < lower) | (df[col] > upper)]
    print(f"{col}: {len(outliers)} outliers")
