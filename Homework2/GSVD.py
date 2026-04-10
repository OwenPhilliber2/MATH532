import numpy as np

def GSVD(A, B):
    M = np.concatenate((A, B), axis = 0)
    k = np.linalg.matrix_rank(M)
    m = A.shape[0]
    n = A.shape[1]

    print(f"m: {m}, n: {n}")

    Qh, SigmaH, W = np.linalg.svd(M, full_matrices=False)
    W = W.T

    SigmaH = np.diag(SigmaH)
    W1 = W[:, :k]
    W2 = W[:, k:]

    Sigmak = SigmaH[:k, :k]
    Q11 = Qh[:m, :k]
    Q21 = Qh[m  :, :k]

    U, V, C, S, X = CSdecomp(Q11, Q21)
    
    G = W @ SigmaH @ X

    return U, V, C, S, G

def CSdecomp(Q11, Q21):
    k = Q11.shape[1]
    p = Q21.shape[0]

    U, s, Vh = np.linalg.svd(Q11)
    C = np.diag(s)
    X = Vh.T

    s_vals = np.sqrt(1 - s**2)

    S = np.diag(s_vals)
    r = np.sum(s < 1e-8)
    s_number = np.sum((1e-8 < s) & (s < 1 - 1e-8))

    V_plus = np.array((1 / s_vals[r]) * Q21 @ X @ np.identity(X.shape[1])[: , r]).reshape(-1, 1)

    for i in range(r + 1, k):
        V_col = np.array((1 / s_vals[i]) * Q21 @ X @ np.identity(X.shape[1])[:, i]).reshape(-1, 1)
        V_plus = np.concatenate((V_plus, V_col), axis = 1)


    prod = V_plus @ V_plus.T
    F = np.random.rand(p, p - k + r)
    print(f"r: {r}, p: {p}, k: {k}, s: {s_number}")
    V_temp = (np.identity((prod.shape[0])) - prod)
    V_temp = V_temp @ F

    V_purp, _ = np.linalg.qr(V_temp)

    V = np.concatenate((V_purp, V_plus), axis = 1)

    # S_mat = np.concatenate((np.zeros((k,k)), S), axis = 0)

    return U, V, C, S, X
        
if __name__ == "__main__":
    A = np.array([[1,2,3,4], [4,3,6,1],[4,5,1,5],[8,4,3,6]])
    B = np.array([[5,4,2,7], [3,1,5,3],[2,1,5,1],[4,1,6,3]])

    U, V, C, S, G = GSVD(A, B)

    print("U orthogonality:", np.linalg.norm(U.T @ U - np.eye(U.shape[1])))
    print("V orthogonality:", np.linalg.norm(V.T @ V - np.eye(V.shape[1])))

    G_inv = np.linalg.pinv(G)

    print("A reconstruction error:", np.linalg.norm(A - U @ C @ G.T))
    print("B reconstruction error:", np.linalg.norm(B - V @ S @ G.T))