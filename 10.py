from sklearn.datasets import load_breast_cancer
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.metrics import confusion_matrix, classification_report
import matplotlib.pyplot as plt

X, y = load_breast_cancer(return_X_y=True)

kmeans = KMeans(n_clusters=2)
clusters = kmeans.fit_predict(X)

print(confusion_matrix(y, clusters))
print(classification_report(y, clusters))

X_pca = PCA(n_components=2).fit_transform(X)

plt.scatter(X_pca[:,0], X_pca[:,1], c=clusters)
plt.title("K-Means Clustering")
plt.xlabel("Principal Component 1")
plt.ylabel("Principal Component 2")
plt.show()
