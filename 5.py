#5
import numpy as np
from collections import Counter

# Generate 100 random values
np.random.seed(42)
x = np.random.rand(100)

# Training and Testing data
train = x[:50]
test = x[50:]

# Labels
labels = ["Class1" if i <= 0.5 else "Class2" for i in train]

# KNN Function
def knn(point, k):
    d = sorted([(abs(point - train[i]), labels[i]) for i in range(50)])
    return Counter([label for _, label in d[:k]]).most_common(1)[0][0]

# Test for different k values
for k in [1, 2, 3, 4, 5, 20, 30]:
    print(f"\nk = {k}")
    for p in test[:5]:      # showing first 5 predictions only
        print(f"{p:.3f} -> {knn(p, k)}")
