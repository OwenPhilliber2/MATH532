import numpy as np
from sklearn.neighbors import kneighbors_graph
import matplotlib.pyplot as plt
from scipy.linalg import eigh
from functions import laplacian, MDS

k = 2
T = 10000
labels = ['a', 'b', 'c', 'd', 'e']
offset = 0.01

D = np.array([[0, 177, 177, 166, 188],
[177, 0, 96, 79, 166],
[177, 96, 0, 144, 177],
[166, 79, 144, 0, 177],
[188, 166, 177, 177, 0]])

eigval, eigvec = laplacian(X = D, weighted=False,k = k, T = T)
eigvalw, eigvecw = laplacian(X = D, weighted=True,k = k, T = T)

embedding = eigvec[:, 1:3]
embeddingw = eigvecw[:, 1:3]

# Plotting 
plt.figure(figsize=(8,6))
plt.scatter(embedding[:,0], embedding[:,1])
for i, letter in enumerate(labels):
    plt.text(embedding[i,0]+offset, embedding[i,1]+offset, letter)
plt.scatter(embeddingw[:,0], embeddingw[:,1])
for i, letter in enumerate(labels):
    plt.text(embeddingw[i,0]+offset, embeddingw[i,1]+offset, letter)

plt.xlabel('Eigenvector 2')
plt.ylabel('Eigenvector 3')
plt.title(f'2D Laplacian Eigenmap ({k}-NN)')
plt.legend()
plt.grid(True)
plt.show()
