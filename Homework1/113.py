import numpy as np
from sklearn.neighbors import kneighbors_graph
import matplotlib.pyplot as plt
from scipy.linalg import eigh
from functions import laplacian, MDS, laplacian_w_A_D

k = 2
T = 10000
labels = ['1', '2', '3', '4', '5', '6']
offset = 0.01

A = np.array([[0, 1, 0, 0, 0, 0],
[1, 0, 1, 1, 0, 0],
[0, 1, 0, 0, 0, 0],
[0, 1, 0, 0, 1, 1],
[0, 0, 0, 1, 0, 1],
[0, 0, 0, 1, 1, 0]])

D = np.array([[1, 0, 0, 0, 0, 0],
[0, 3, 0, 0, 0, 0],
[0, 0, 1, 0, 0, 0],
[0, 0, 0, 3, 0, 0],
[0, 0, 0, 0, 2, 0],
[0, 0, 0, 0, 0, 2]])


# Typo in 113 with the adjency matrix (has 6 and 3s)

eigval, eigvec = laplacian_w_A_D(A = A, D = D, weighted=False,k = k, T = T)

embedding = eigvec[:, 1:3]

# Plotting 
plt.figure(figsize=(8,6))
plt.scatter(embedding[:,0], embedding[:,1])
for i, letter in enumerate(labels):
    plt.text(embedding[i,0]+offset, embedding[i,1]+offset, letter)
    
plt.xlabel('Eigenvector 2')
plt.ylabel('Eigenvector 3')
plt.title(f'2D Laplacian Eigenmap ({k}-NN)')
plt.legend()
plt.grid(True)
plt.show()
