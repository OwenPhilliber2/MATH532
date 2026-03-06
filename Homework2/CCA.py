import numpy as np

def CCA(X, Y):
    return_dict = dict()
    m = X.shape[0]
    nx = X.shape[1]
    ny = Y.shape[1]

    XX = (np.eye(m) - np.ones(m) / m) @ X
    YY = (np.eye(m) - np.ones(m) / m) @ Y

    Qx, Rx = np.linalg.qr(XX, mode='reduced')
    Qy, Ry = np.linalg.qr(YY, mode='reduced')

    R, D, S, = np.linalg.svd(Qx.T @ Qy)

    return_dict["R"] = R
    return_dict["D"] = np.diag(D)
    return_dict["S"] = S

    z = D[0]
    return_dict["z"] = min(z, 1)

    angle_z = np.arccos(z)

    a = np.linalg.pinv(Rx) @ R[:, 0] # X weight
    b = np.linalg.pinv(Ry) @ S[:, 0] # X weight
    return_dict["a"] = a
    return_dict["b"] = b

    alpha = XX @ a
    beta = YY @ b
    return_dict["alpha"] = alpha
    return_dict["beta"] = beta

    return return_dict

