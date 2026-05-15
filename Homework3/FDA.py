import numpy as np

def FDA(X, Y):
    x_bar = np.mean(X, axis = 1)
    y_bar = np.mean(Y, axis = 1)

    X_tilde = X - np.mean(X, axis = 1, keepdims = True)
    Y_tilde = Y - np.mean(Y, axis = 1, keepdims = True)

    S = (1 / X.shape[1]) * (X_tilde) @ (X_tilde).T + (1 / Y.shape[1]) * (Y_tilde) @ (Y_tilde).T

    w = np.linalg.solve(S, (x_bar - y_bar))
    w = w / np.linalg.norm(w)

    x_red = w.T @ X
    y_red = w.T @ Y

    return x_red, y_red, w