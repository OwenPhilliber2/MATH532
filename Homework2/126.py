from CCA import CCA
import os
import numpy as np
from scipy.io import loadmat
import matplotlib.pyplot as plt

cd = os.getcwd()

# Loading cat and dog dataset
mat_data = loadmat(os.path.join(cd, 'Homework1', 'data', 'CDdata.mat'))

# Extracting just the data from the dictionary
data = mat_data['Y']
X = data[:, :99].astype(np.float64) / 255.0
Y = data[:, 99:].astype(np.float64) / 255.0

print(X.shape)
print(Y.shape)

dict = CCA(X, Y)

z = dict["z"]
angle_z = np.arccos(z)

print(f'Correlation coefficient: {z}')
print(f"Principal angle between cats and dogs: {angle_z}")

print(dict['alpha'].shape)

plt.scatter(dict["alpha"], dict["beta"], s = 5)
plt.axline((0, 0), slope=1, color = "black")
plt.xlabel("Alpha")
plt.ylabel("Beta")
plt.title("Scatterplot of Alpha and Beta")
plt.show()

cor_alpha = dict['alpha'].reshape(64,64).T
cor_beta = dict['beta'].reshape(64,64).T

plt.imshow(cor_alpha, cmap = 'grey')
plt.title("Canonical Correlation Matrix Alpha Reshaped")
plt.show()

plt.imshow(cor_beta, cmap = 'grey')
plt.title("Canonical Correlation Matrix Beta Reshaped")
plt.show()

gamma = cor_alpha * cor_beta
plt.imshow(gamma, cmap = 'grey')
plt.title("Gamma Canonical Correlation Matrix (Alpha * Beta)")
plt.show()

a = dict["a"]
b = dict["b"]

plt.plot(range(a.shape[0]), a)
plt.title("Canonical Correlation Vector Alpha")
plt.show()

plt.plot(range(b.shape[0]), b)
plt.title("Canonical Correlation Vector Beta")
plt.show()
