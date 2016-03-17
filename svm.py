# ----------------
# IMPORT PACKAGES
# ----------------

from sklearn import datasets, svm
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
import numpy as np

# ----------------
# OBTAIN DATA
# ----------------

iris = datasets.load_iris()

# ----------------
# VISUALIZE DATA
# ----------------

plt.scatter(iris.data[:, 1], iris.data[:, 2], c=iris.target)
plt.xlabel(iris.feature_names[1])
plt.ylabel(iris.feature_names[2])
plt.title("Iris Dataset - Setosa, Versicolor and Virginica")
plt.show()

plt.scatter(iris.data[0:100, 1], iris.data[0:100, 2], c=iris.target[0:100])
plt.xlabel(iris.feature_names[1])
plt.ylabel(iris.feature_names[2])
plt.title("Iris Dataset - Setosa and Versicolor")
plt.show()

# ----------------
# MODEL DATA
# ----------------

svc = svm.SVC(kernel="linear")
X = iris.data[0:100, [0,2]]
y = iris.target[0:100]
svc.fit(X, y)

cmap_light = ListedColormap(["#FFAAAA", "#AAFFAA", "#AAAAFF"])
cmap_bold = ListedColormap(["#FF0000", "#00FF00", "#0000FF"])

def plot_estimator(estimator, X, y):
	estimator.fit(X, y)
	x_min, x_max = X[:, 0].min() - 0.1, X[:, 0].max() + 0.1
	y_min, y_max = X[:, 1].min() - 0.1, X[:, 1].max() + 0.1
	xx, yy = np.meshgrid(np.linspace(x_min, x_max, 100), np.linspace(y_min, y_max, 100))
	Z = estimator.predict(np.c_[xx.ravel(), yy.ravel()])

	Z = Z.reshape(xx.shape)
	plt.figure()
	plt.pcolormesh(xx, yy, Z, cmap=cmap_light)

	plt.scatter(X[:, 0], X[:, 1], c=iris.target[0:100], cmap=cmap_bold)
	plt.axis("tight")
	plt.xlabel(iris.feature_names[1])
	plt.ylabel(iris.feature_names[2])
	plt.tight_layout()
	plt.show()

# ----------------
# TEST DATA
# ----------------

plot_estimator(svc, X, y)