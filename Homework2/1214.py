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

d_ind = 100

c_ind = np.random.choice(99, 21)
c_ind_1 = np.append(c_ind[:6], d_ind)
c_ind_2 = np.append(c_ind[6:13], d_ind)
c_ind_3 = np.append(c_ind[13:], d_ind)

A1 = data[:, c_ind_1]
A2 = data[:, c_ind_2]
A3 = data[:, c_ind_3]

X1, _ = np.linalg.qr(A1)
X2, _ = np.linalg.qr(A2)
X3, _ = np.linalg.qr(A3)

X = np.concatenate((X1, X2, X3), axis = 1)

U, s, Vh = np.linalg.svd(X, full_matrices=False)

fig, axs = plt.subplots(1,3)
fig.suptitle('SVD Approximations')
axs[0].imshow(np.rot90(U[:, 0].reshape(64, 64), k = 3), cmap = 'gist_yarg')
axs[0].set_title('1st Dim')
axs[0].set_xticks([])
axs[0].set_yticks([])
axs[1].imshow(np.rot90(U[:, 1].reshape(64, 64), k = 3), cmap = 'gist_yarg')
axs[1].set_title('2nd Dim')
axs[1].set_xticks([])
axs[1].set_yticks([])
axs[2].imshow(np.rot90(U[:, 2].reshape(64, 64), k = 3), cmap = 'gist_yarg')
axs[2].set_title('3rd Dim')
axs[2].set_xticks([])
axs[2].set_yticks([])
plt.show()