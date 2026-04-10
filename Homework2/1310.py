from scipy.io import loadmat
import matplotlib.pyplot as plt
import numpy as np
import os
from GSVD_improved import GSVD

cd = os.getcwd()

data = loadmat(os.path.join(cd, 'Homework2', 'data', 'Indian_pines_corrected.mat'))
data = data['indian_pines_corrected']

X = data[:, :, 24]

X_shift = np.roll(X, 1, axis = 0)

N = (X - X_shift) / np.sqrt(2)

U, V, C, S, G = GSVD(X, N)

s_vals = np.diagonal(S)
c_vals = np.diagonal(C)

plt.plot(s_vals)
plt.plot(c_vals)
plt.show()

print(np.argmin(np.abs(s_vals - c_vals)))
