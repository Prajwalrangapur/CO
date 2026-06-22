#7
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures
from sklearn.metrics import mean_squared_error, r2_score

# Linear Regression
from sklearn.datasets import fetch_openml
df = fetch_openml("boston", version=1, as_frame=True).frame
X, y = df[["RM"]], df["MEDV"]

Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.2, random_state=42)
m1 = LinearRegression().fit(Xtr, ytr)
p1 = m1.predict(Xte)

print("Linear:", mean_squared_error(yte, p1), r2_score(yte, p1))

plt.scatter(Xte, yte); plt.plot(Xte, p1, "r")
plt.show()

# Polynomial Regression
url = "https://archive.ics.uci.edu/ml/machine-learning-databases/auto-mpg/auto-mpg.data"
cols = ["mpg","cyl","dis","hp","wt","acc","yr","org"]

df = pd.read_csv(url, sep=r"\s+", names=cols, na_values="?").dropna()
X, y = df[["hp"]], df["mpg"]

Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.2, random_state=42)

poly = PolynomialFeatures(2)
Xtr, Xte = poly.fit_transform(Xtr), poly.transform(Xte)

m2 = LinearRegression().fit(Xtr, ytr)
p2 = m2.predict(Xte)

print("Poly:", mean_squared_error(yte, p2), r2_score(yte, p2))

plt.scatter(X, y)
plt.plot(X, m2.predict(poly.transform(X)), "r")
plt.show()
