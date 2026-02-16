from scipy.io import loadmat
import matplotlib.pyplot as plt
import numpy as np
from functions import MDS, distmat
import os
from scipy.spatial.distance import cdist


# Loading cat and dog dataset
cwd = os.getcwd()
mat_data = loadmat(os.path.join(cwd, 'Homework1', 'data', 'CDdata.mat'))

# Extracting just the data from the dictionary
data = mat_data['Y']
X = data.astype(np.float64) / 255.0

D = distmat(X)

X, eigval, _ = MDS(D)
ac_labels = np.repeat([1, -1], 99)

print(np.min(eigval))
print(np.max(eigval))

plt.figure(figsize=(5, 5))
plt.scatter(X[:, 1], -X[:, 0], c = ac_labels, cmap ='bwr')
plt.xlabel("Second Embedding Dimension")
plt.ylabel("First Embedding Dimension")
plt.title("Cats and Dogs MDS")
plt.show()


plt.figure(figsize=(8,6))
plt.scatter(range(len(eigval)), eigval)
plt.hlines(0, xmin = 0, xmax = len(eigval), colors = 'red')
plt.show()
