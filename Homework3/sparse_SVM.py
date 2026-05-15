import numpy as np
from scipy.optimize import linprog

def sparse_SVM(X, y, C = 1):
    m = X.shape[0]
    n = X.shape[1]

    # Creating the b constraint vector
    b = np.concatenate((np.zeros(2 * n), -1 * np.ones(m)), axis = 0)

    # Creating the A constraint matrix row by row
    row1 = np.concatenate((np.identity(n), -1 * np.identity(n), np.zeros((n, m + 1))), axis = 1)
    row2 = np.concatenate((-1 * np.identity(n), -1 * np.identity(n), np.zeros((n, m + 1))), axis = 1)
    row3 = -1 * np.concatenate((np.diag(y.flatten()) @ X, np.zeros((m,n)), np.identity(m), y), axis = 1)

    A = np.concatenate((row1, row2, row3))

    # Creating the cost matrix
    c = np.concatenate((np.zeros(n), np.ones(n), C * np.ones(m), np.zeros(1)))

    # Creating the bounds
    unbounded = (None, None)
    bounded = (0, None)

    bounds = [unbounded] * n + [bounded] * (n + m) + [unbounded]

    # Solving the linear program
    res = linprog(c = c, A_ub = A, b_ub = b, bounds = bounds)
    
    weights = res['x'][:n]
    bias = res['x'][-1]
    
    return weights, bias
