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

plt.scatter(dict["alpha"], dict["beta"], s = 5)
plt.axline((0, 0), slope=1, color = "black")
plt.xlabel("Alpha")
plt.ylabel("Beta")
plt.show()

cor_alpha = dict['alpha'].reshape(64,64)
cor_beta = dict['beta'].reshape(64,64)
rotated_cor_alpha = np.rot90(cor_alpha, k=-1)
rotated_cor_beta = np.rot90(cor_beta, k=-1)

plt.imshow(rotated_cor_alpha, cmap = 'grey')
plt.title("Cannonical Correlation Matrix Alpha Reshaped")
plt.show()

plt.imshow(rotated_cor_beta, cmap = 'grey')
plt.title("Cannonical Correlation Matrix Beta Reshaped")
plt.show()

gamma = rotated_cor_alpha * rotated_cor_beta
plt.imshow(gamma, cmap = 'grey')
plt.show()

a = dict["a"]
b = dict["b"]

plt.plot(range(a.shape[0]), a)
plt.show()

plt.plot(range(b.shape[0]), b)
plt.show()
