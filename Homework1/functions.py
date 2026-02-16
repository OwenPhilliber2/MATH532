import numpy as np
from numpy import linalg as lg
from scipy.linalg import eigh

def laplacian(X, weighted = True, k = 3, T = 5000):
    '''
    X - Distance matrix
    weighted - Boolian whether laplacian with weighted weights or not
    '''
    n = X.shape[0]
    # Unweighted
    A = np.zeros(X.shape)

    for i in range(n):
        smallest_k = np.argsort(X[i, :])[1:k+1]
        A[i, smallest_k] = 1
    A = np.maximum(A, A.T)
    if weighted:
        # Weighted
        A = np.exp(-X ** 2 / T) * A
        
    # Create the degree matrix
    D = np.diag(np.sum(A, axis = 0))
    # Create the Laplacian matrix
    L = D - A
    # Find the eigenvalues and eigenvectors
    eigval, eigvec = eigh(L, D)
    # Sort the eigenvectors
    idx = np.argsort(eigval)

    return eigval[idx], eigvec[:, idx]

def MDS(D, tol = 1e-9):
    # Number of datapoints
    n = D.shape[0]

    # Creating H matrix
    H = np.identity(n) - np.ones((n, n)) / n

    # Calculating B
    B = (-1 / 2) * (H @ (D ** 2) @ H)

    # Finding eigenvalues and eigenvectors of B
    eigval, eigvec = lg.eigh(B)

    # Sorting eigenvalues and eigenvectors
    idx = np.argsort(eigval)[::-1]
    eigval, eigvec = eigval[idx], eigvec[:, idx]

    # keep only positive eigenvalues
    pos = eigval > tol
    eigval = eigval[pos]
    eigvec = eigvec[:, pos]

    # Calculating normalized eigenvalue matrix
    X = eigvec @ np.diag(np.sqrt(eigval))

    # Question do we cut V before or after normalizing?
    return X, eigval, eigvec

def distmat(X, method='2norm'):
    n = X.shape[1]
    D = np.zeros((n, n))

    if method == '2norm':
        for i in range(n - 1):
            for j in range(i + 1, n):
                D[i, j] = np.linalg.norm(X[:, i] - X[:, j])
    elif method == 'angle':
        for i in range(n - 1):
            for j in range(i + 1, n):
                x = X[:, i]
                y = X[:, j]
                D[i, j] = np.arccos(np.dot(x.T, y) / (np.linalg.norm(x) * np.linalg.norm(y)))
                # Alternatively, use D(i,j) = min(j-i, n+i-j) * 2 * np.pi / 7
    else:
        print('Unknown distance method')
    
    D += D.T
    return D