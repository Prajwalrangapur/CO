#6
import numpy as np
import matplotlib.pyplot as plt

# Data
X = np.linspace(0, 10, 20)
y = np.sin(X)

# Prediction points
x_test = np.linspace(0, 10, 100)
y_pred = []

tau = 1.0

#target
for x in x_test:
    w = np.exp(-(X - x)**2 / (2 * tau**2))
    y_hat = np.sum(w * y) / np.sum(w)
    y_pred.append(y_hat)

plt.scatter(X, y, label="Data")
plt.plot(x_test, y_pred, label="LWR")
plt.title("Locally Weighted Regression")
plt.legend()
plt.show()
