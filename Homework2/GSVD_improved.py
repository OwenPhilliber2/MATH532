import numpy as np
from scipy.linalg import null_space

def thin_GSVD(A, B):
    M = np.concatenate((A, B), axis=0)
    k = np.linalg.matrix_rank(M)
    m = A.shape[0]
    n = A.shape[1]
    p = B.shape[0]

    print(f"m: {m}, n: {n}, p: {p}, k: {k}")

    # Thin SVD of M — only keep k columns
    Qh, sigma_h, Wt = np.linalg.svd(M, full_matrices=False)
    W = Wt.T  # n x k (only rank-k part)

    W1     = W[:, :k]           # n x k
    Sigmak = np.diag(sigma_h[:k])  # k x k

    Q11 = Qh[:m, :k]   # m x k
    Q21 = Qh[m:,  :k]  # p x k

    U, V, C, S, X = CSdecomp(Q11, Q21)

    # Thin generalized singular value matrix: n x k
    G = W1 @ Sigmak @ X.T

    return U, V, C, S, G


def CSdecomp(Q11, Q21):
    """
    Thin CS decomposition of [Q11; Q21] where Q11 is m x k, Q21 is p x k,
    and [Q11; Q21] has orthonormal columns.

    Returns U (m x k), V (p x k), C (k x k), S (k x k), X (k x k) such that:
        Q11 = U @ C @ X.T
        Q21 = V @ S @ X.T
        C^2 + S^2 = I
        U, V, X orthogonal (square), C, S diagonal with non-negative entries
    """
    m, k = Q11.shape
    p    = Q21.shape[0]

    # Step 1: SVD of Q11 gives C and X directly
    U, c, Xt = np.linalg.svd(Q11, full_matrices=True)  # U: m x m, Xt: k x k
    c  = np.clip(c, 0, 1)                               # numerical safety
    C  = np.diag(c)                                     # k x k
    X  = Xt.T                                           # k x k  (X.T = Xt)

    # Step 2: recover S = sqrt(I - C^2)
    s_vals = np.sqrt(np.clip(1 - c**2, 0, 1))
    S = np.diag(s_vals)

    # Partition indices
    r        = np.sum(c > 1 - 1e-8)   # number of c_i ≈ 1  (s_i ≈ 0)
    mid_end  = np.sum(c > 1e-8)        # r + number of strictly interior values

    # Step 3: build V column by column
    # For the middle block (r <= i < mid_end), V[:,i] = (1/s_i) * Q21 @ X[:,i]
    # For the top block (i < r) and bottom block (i >= mid_end), use QR on nullspace

    V = np.zeros((p, p))

    # Middle columns — well-determined
    mid_cols = slice(r, mid_end)
    if mid_end > r:
        V_mid = (Q21 @ X[:, mid_cols]) / s_vals[mid_end - 1: r - 1: -1][
            ::-1
        ]  # safer:
        for i in range(r, mid_end):
            V[:, i] = Q21 @ X[:, i] / s_vals[i]

    # Top block (s_i ≈ 0, c_i ≈ 1): V[:,i] arbitrary — fill via QR of complement
    # Bottom block (c_i ≈ 0, s_i ≈ 1): similarly unconstrained
    # Build both at once: orthogonalise a random matrix against the middle columns
    if mid_end > r:
        V_mid_block = V[:, r:mid_end]
        complement  = np.eye(p) - V_mid_block @ V_mid_block.T
    else:
        complement  = np.eye(p)

    n_free = p - (mid_end - r)
    if n_free > 0:
        rand_cols = np.random.randn(p, n_free)
        proj      = complement @ rand_cols
        V_free, _ = np.linalg.qr(proj)   # p x n_free orthonormal

        # Place free columns: first r go to top, rest go to bottom
        free_indices = list(range(r)) + list(range(mid_end, p))
        for j, idx in enumerate(free_indices):
            V[:, idx] = V_free[:, j]

    # Trim U and V to thin form: m x k and p x k
    U_thin = U[:, :k]
    V_thin = V[:, :k]

    return U_thin, V_thin, C, S, X
