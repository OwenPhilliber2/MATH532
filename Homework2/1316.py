import numpy as np
from scipy.io import loadmat
from scipy.linalg import qr
import os
from GSVD import GSVD
import matplotlib.pyplot as plt

cd = os.getcwd()

# Load dataset
data = loadmat(os.path.join(cd, 'Homework2', 'data','trig_dataset_526x4.mat'))
X = data['X']

MIX = np.random.rand(4, 4)

# S1: reverse columns of X, then scale each column
S1 = X[:, ::-1] @ np.diag([4, 3, 2, 1])

# Economy QR decomposition
T1, R1 = qr(S1, mode='economic')
print("T1'*T1 =")
print(T1.T @ T1)

# Shifted version of X (circular shift rows down by 1)
XS = np.zeros_like(X)
XS[0, :]  = X[-1, :]
XS[1:, :] = X[:-1, :]

# T2: finite difference / discrete derivative
T2 = (X - XS) / np.sqrt(2)
Dtrue = T2.T @ T2
print("\nDtrue =")
print(Dtrue)

# A and B (requires MIX to be defined)
A = T1 @ MIX
B = T2 @ MIX

U, V, C, S, G = GSVD(A, B)

U_thin = U[:, :4]

U_thin[-1,:] = U_thin[-1,:] * -1
U_thin[-2,:] = U_thin[-2,:] * -1
U_thin = -1 * U_thin

print(T1.shape)
print(U.shape)


fig, axs = plt.subplots(1,2)
axs[0].imshow(T1, aspect='auto')
axs[0].set_title('T1')
axs[0].set_xticks([])
axs[0].set_yticks([])
axs[1].imshow(U_thin, aspect='auto')
axs[1].set_title('thin U')
axs[1].set_xticks([])
axs[1].set_yticks([])
plt.show()