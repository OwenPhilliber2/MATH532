from CCA import CCA
import os
import numpy as np
from scipy.io import loadmat
import matplotlib.pyplot as plt

cd = os.getcwd()

# Loading cat and dog dataset
c2 = loadmat(os.path.join(cd, 'Homework2', 'data', 'IPcornnotilC2.mat'))
c14 = loadmat(os.path.join(cd, 'Homework2', 'data', 'IPwoodsC14.mat'))

X = c2['X']
Y = c14['Y']

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
plt.show()

a = dict["a"]
b = dict["b"]

print(a.shape)
print(b.shape)

plt.plot(range(a.shape[0]), a)
plt.show()

plt.plot(range(b.shape[0]), b)
plt.show()