from scipy.io import loadmat
import os
import numpy as np
import matplotlib.pyplot as plt

cd = os.getcwd()
# Loading cat and dog dataset
mat_data = loadmat(os.path.join(cd, 'Homework2', 'data', 'Kingrynormalized.mat'))

data = mat_data['Kingrynorm']
print(data.shape)

def SVD_energy(data, title):

    U, S, Vh = np.linalg.svd(data, full_matrices=False)

    S_squared = S ** 2

    total_energy = np.sum(S_squared)

    E = np.cumsum(S_squared) / total_energy

    

    plt.plot(E, label = title)
    plt.hlines(.95, xmin = 0, xmax = len(E), colors = 'black')
    plt.title(f"Energy of {title}")

    return U, S, Vh, E, plt
U, S, Vh, E, plt = SVD_energy(data, title = "All Data")
plt.show()

print(E[2])
print(E[72])

A = (U[:, :3] * S[:3]) @ Vh[:3, :]
plt.figure(figsize=(6, 6))  # very wide, very short
plt.imshow(A, aspect='auto',cmap = 'inferno')
plt.show()

a1 = U[:, 0] @ data
a2 = U[:, 1] @ data
a3 = U[:, 2] @ data

cords = np.stack((a1, a2, a3), axis = 1)
print(cords.shape)


fig = plt.figure()
ax = fig.add_subplot(projection='3d')

# For each set of style and range settings, plot n random points in the box
# defined by x in [23, 32], y in [0, 100], z in [zlow, zhigh].
for i in range(cords.shape[0]):
    ax.scatter(cords[i, 0], cords[i, 1], cords[i, 2], marker='o', c = 'blue')

ax.set_xlabel('First LSV')
ax.set_ylabel('Second LSV')
ax.set_zlabel('Third LSV')

plt.show()


Schu4_lung = data[:, 6:30]
LVS_lung = data[:, 30:54]
Schu4_spl = data[:, 60:84]
LVS_spl = data[:, 84:108]

plt1 = SVD_energy(Schu4_lung, title = "Schu4 Lung Data")
plt2 = SVD_energy(LVS_lung, title = "LVS Lung Data")
plt3 = SVD_energy(Schu4_spl, title = "Schu4 Spleen Data")
plt4 = SVD_energy(LVS_spl, title = "LVS Spleen Data")
plt.legend()
plt.show()