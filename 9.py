#9
from sklearn.datasets import fetch_olivetti_faces
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt

# Load dataset
faces = fetch_olivetti_faces()
X, y = faces.data, faces.target

# Split data
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Train Naive Bayes classifier
model = GaussianNB()
model.fit(X_train, y_train)

# Predict and find accuracy
y_pred = model.predict(X_test)
print("Accuracy:", accuracy_score(y_test, y_pred) * 100)

# Show first 5 test images
for i in range(5):
    plt.figure(figsize=(2,2))
    plt.imshow(X_test[i].reshape(64,64), cmap="gray")
    plt.title(f"Actual: {y_test[i]}  Pred: {y_pred[i]}")
    plt.axis("off")
    plt.show()
