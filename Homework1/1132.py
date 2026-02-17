from scipy.io import loadmat
import matplotlib.pyplot as plt
import numpy as np
from scipy.linalg import eigh
from functions import laplacian, distmat
import os

show_sample_plot = False # Shows a sample image from the data
k = 100 # number of nearest neighbors

cd = os.getcwd()

# Loading cat and dog dataset
mat_data = loadmat(os.path.join(cd, 'Homework1', 'data', 'CDdata.mat'))

# Extracting just the data from the dictionary
data = mat_data['Y']
X = data.astype(np.float64) / 255.0

# Sample plot
if show_sample_plot:
    r_int = np.random.randint(0, 198)
    plt.imshow(data[:, r_int].reshape(64,64), cmap = 'grey')
    plt.title(f"Sample plot of Cat and Dog Dataset (Index: {r_int})")
    plt.show()

D = distmat(X)
T = np.median(D[D > 0])**2

eigval, eigvec = laplacian(X = D, weighted=False, k = k, T = T)

# 2D embeddings
embedding = eigvec[:, 1:3]


pred_labels = np.sign(embedding[:, 0])
ac_labels = np.repeat([1, -1], 99)

acc1 = np.mean(pred_labels == ac_labels)
acc2 = np.mean(-pred_labels == ac_labels)

accuracy = max(acc1, acc2)

print(accuracy)

plt.scatter(embedding[:,0], embedding[:,1], c = pred_labels, cmap ='bwr', s=60)
# plt.scatter(range(198), embedding[:,0], c = pred_labels, cmap ='bwr', label='Unweighted', s=60)

plt.xlabel('Elements of the Fiedler vector')
plt.ylabel('Second nonzero eigenvector')
plt.title(f'Fiedler Vector Embedding (Accuracy: {accuracy*100:.2f} %)')
plt.legend()
plt.grid(True)
plt.show()

