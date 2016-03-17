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