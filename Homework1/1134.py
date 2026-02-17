import torchvision
import torchvision.transforms as transforms
import torch
from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np
from torch.utils.data import DataLoader
from torch.utils.data import Subset
from functions import laplacian, MDS, distmat_3D

k = 3 # number of nearest neighbors

# Transform to tensor
transform = transforms.ToTensor()

# Load MNIST
dataset = torchvision.datasets.MNIST(
    root='./MNISTdata',
    train=True,
    download=True,
    transform=transform
)

# Pick 10 random indices
indices = torch.arange(0, 15)

images = torch.stack([dataset[i][0] for i in indices]).squeeze(1)
labels = [dataset[i][1] for i in indices]


# Plot images
fig, axes = plt.subplots(3, 5, figsize=(10, 6))

for ax, idx in zip(axes.flatten(), indices):
    img, label = dataset[idx]
    ax.imshow(img.squeeze(), cmap='gray')
    ax.set_title(f"Index: {idx}, Label: {label}")
    ax.axis('off')

plt.tight_layout()
plt.show()

D = distmat_3D(images)

# Laplacian embedding unweighted
eigval_laplacian, eigvec_laplacian = laplacian(X = D, weighted=False)

embedding_laplacian = eigvec_laplacian[:, 1:3]

plt.scatter(embedding_laplacian[:,1], embedding_laplacian[:,0], color='red', label='Unweighted', s=60)
for i, index in enumerate(labels):
    plt.text(embedding_laplacian[i,1]+.01, embedding_laplacian[i,0]+.01, f"{i},{index}")
plt.xlabel('Eigenvector 2')
plt.ylabel('Eigenvector 3')
plt.title(f'2D Laplacian Eigenmap ({k}-NN)')
plt.legend()
plt.grid(True)
plt.show()

X_MDS, eigval_MDS, _ = MDS(D)

plt.figure(figsize=(5, 5))
plt.scatter(X_MDS[:, 0], X_MDS[:, 1], c = 'red', cmap ='bwr')
for i, index in enumerate(labels):
    plt.text(X_MDS[i,0]+.01, X_MDS[i,1]+.01, f"{i},{index}")
plt.xlabel("Second Embedding Dimension")
plt.ylabel("First Embedding Dimension")
plt.title("MNIST MDS")
plt.show()
