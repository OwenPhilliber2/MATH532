import numpy as np
import matplotlib.pyplot as plt


# --- First shape: hollow square ---
A1 = np.zeros((8, 8))

A1[2, 2:6] = 1   # top edge (MATLAB 3 -> Python 2)
A1[5, 2:6] = 1   # bottom edge
A1[2:6, 2] = 1   # left edge
A1[2:6, 5] = 1   # right edge

# --- Second shape: filled square ---
A2 = np.zeros((8, 8))
A2[2:6, 2:6] = 1

# --- Plot ---
fig, axs = plt.subplots(1, 2)

axs[0].imshow(A1, cmap='gray')
axs[0].set_title('Hollow Square')
axs[0].axis('off')

axs[1].imshow(A2, cmap='gray')
axs[1].set_title('Filled Square')
axs[1].axis('off')

plt.show()

# --- Transform ---
diagonal = [1, -1] * int(A1.shape[0] / 2)
off_diagonal = [1, -0] * int(A1.shape[0] / 2)

U = (1/np.sqrt(2)) * (np.diag(diagonal) + np.diag(off_diagonal[:-1], k = 1) + np.diag(off_diagonal[:-1], k = -1))

U_H1 = U[:, ::2]
U_L1 = U[:, 1::2]

top_left = U_L1.T @ A1 @ U_L1
top_right = U_L1.T @ A1 @ U_H1
bottom_left = U_H1.T @ A1 @ U_L1
bottom_right = U_H1.T @ A1 @ U_H1

fig, axs = plt.subplots(2, 2)

axs[0, 0].imshow(top_left, cmap='gray')
axs[0, 0].set_title('v1, v1')
axs[0, 0].axis('off')

axs[0, 1].imshow(top_right, cmap='gray')
axs[0, 1].set_title('v1, w1')
axs[0, 1].axis('off')

axs[1, 0].imshow(bottom_left, cmap='gray')
axs[1, 0].set_title('w1, v1')
axs[1, 0].axis('off')

axs[1, 1].imshow(bottom_right, cmap='gray')
axs[1, 1].set_title('w1, w1')
axs[1, 1].axis('off')

plt.show()