from scipy.io import loadmat
import matplotlib.pyplot as plt
import numpy as np
from scipy.linalg import eigh
from functions import MDS
import os

D = np.array([[0,1,1,1],
              [1,0,1,1],
              [1,1,0,1],
              [1,1,1,0]])

X, eigval, _ = MDS(D)

plt.figure(figsize=(5, 5))
plt.scatter(X[:, 0], X[:, 1], c = 'red', cmap ='bwr')
plt.xlabel("Second Embedding Dimension")
plt.ylabel("First Embedding Dimension")
plt.title("Two Dimensional embedding of four equidistant points MDS")
plt.show()


plt.figure(figsize=(8,6))
plt.scatter(range(len(eigval)), eigval)
plt.hlines(0, xmin = 0, xmax = len(eigval), colors = 'red')
plt.show()