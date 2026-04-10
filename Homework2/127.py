from CCA import CCA
import os
import numpy as np
from scipy.io import loadmat
import matplotlib.pyplot as plt

cd = os.getcwd()

# Loading cat and dog dataset
c2 = loadmat(os.path.join(cd, 'Homework2', 'data', 'IPcornnotilC2.mat'))
c14 = loadmat(os.path.join(cd, 'Homework2', 'data', 'IPwoodsC14.mat'))

X = c2['X'].T
Y = c14['Y'].T

list = np.random.random_integers(0, X.shape[0] - 1, Y.shape[0])

# X = X[list, :] # random rows for the Y matrix
X = X[:Y.shape[0], :] # First 1265 for the Y matrix
print(X.shape)
print(Y.shape)

dict = CCA(X, Y)

z = dict["z"]
angle_z = np.arccos(z)

print(f'Correlation coefficient: {z}')
print(f"Principal angle between c2 and c14: {angle_z}")

plt.scatter(dict["alpha"], dict["beta"], s = 5)
plt.axline((0, 0), slope=1, color = "black")
plt.xlabel("Alpha")
plt.ylabel("Beta")
plt.title("Alpha Beta Scatterplot")
plt.show()

a = dict["a"]
b = dict["b"]

print(a.shape)
print(b.shape)

plt.plot(range(a.shape[0]), a)
plt.title("a Vector Plot")
plt.show()

plt.plot(range(b.shape[0]), b)
plt.title("b Vector Plot")
plt.show()