#5
import numpy as np
from collections import Counter
import matplotlib.pyplot as plt

# Generate 100 random values
np.random.seed(42)
x = np.random.rand(100)

# Training and Testing data
train = x[:50]
test = x[50:]

# Labels for training data
labels = ["Class1" if i <= 0.5 else "Class2" for i in train]

# KNN Function
def knn(point, k):
    d = sorted([(abs(point - train[i]), labels[i]) for i in range(50)])
    return Counter([label for _, label in d[:k]]).most_common(1)[0][0]

# Test for different k values
for k in [1, 2, 3, 4, 5, 20, 30]:

    predictions = [knn(p, k) for p in test]

    print(f"\nResults for k = {k}")
    for p, pred in zip(test[:5], predictions[:5]):
        print(f"{p:.3f} -> {pred}")

    # Graph
    y = [1 if p == "Class1" else 2 for p in predictions]

    plt.figure(figsize=(6, 4))
    plt.scatter(test, y)
    plt.yticks([1, 2], ["Class1", "Class2"])
    plt.xlabel("Test Values")
    plt.ylabel("Predicted Class")
    plt.title(f"KNN Classification (k={k})")
    plt.grid(True)
    plt.show()
