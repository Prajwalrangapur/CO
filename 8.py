#8
from sklearn.datasets import load_breast_cancer
from sklearn.tree import DecisionTreeClassifier, plot_tree
# from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt

# Load dataset
data = load_breast_cancer()
X, y = data.data, data.target

# Train model
model = DecisionTreeClassifier()
model.fit(X, y)

# # Model accuracy
# y_pred = model.predict(X)
# accuracy = accuracy_score(y, y)
# print("Model Accuracy:", round(accuracy * 100, 2), "%")

# Predict first sample
result = model.predict([X[20]])
print("Prediction:", data.target_names[result[0]])

# Draw tree
plot_tree(model, filled=True)
plt.show()
