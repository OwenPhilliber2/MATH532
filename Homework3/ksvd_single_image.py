"""
K-SVD Dictionary Learning — single image version
Accepts any PNG / JPG / BMP file; converts to grayscale automatically.

Usage:
    python ksvd_single_image.py --image path/to/image.png

Dependencies:
    pip install numpy scipy matplotlib Pillow
"""

import argparse
import time
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image


# ─────────────────────────────────────────────
# OMP  (Orthogonal Matching Pursuit)
# ─────────────────────────────────────────────
def OMP(y, D, K):
    """
    Parameters
    ----------
    y : (M,) signal vector
    D : (M, N) dictionary
    K : int, sparsity (max non-zero coefficients)

    Returns
    -------
    xapp   : (N,) sparse coefficient vector
    Ind    : list of selected column indices
    omperr : list of residual norms at each iteration
    """
    M, N = D.shape
    r    = y.copy().astype(float)
    Ind  = []
    omperr = []

    for _ in range(K):
        correlations      = np.abs(D.T @ r)
        correlations[Ind] = -np.inf          # mask already-chosen atoms
        best = int(np.argmax(correlations))
        Ind.append(best)

        Ds = D[:, Ind]
        coefs, _, _, _ = np.linalg.lstsq(Ds, y, rcond=None)
        r = y - Ds @ coefs
        omperr.append(float(np.linalg.norm(r)))

        if omperr[-1] < 1e-12:
            break

    xapp = np.zeros(N)
    xapp[Ind] = coefs
    return xapp, Ind, omperr


# ─────────────────────────────────────────────
# MyKSVD
# ─────────────────────────────────────────────
def MyKSVD(Y, D, K, MaxIter):
    """
    Parameters
    ----------
    Y       : (M, N_patches) training patches
    D       : (M, DsizeK) initial dictionary
    K       : int, sparsity parameter
    MaxIter : int, number of iterations

    Returns
    -------
    XK, D, dprog, Yerr, Yomperror
    """
    M, N     = Y.shape
    n_atoms  = D.shape[1]
    dprog     = np.zeros(MaxIter)
    Yerr      = np.zeros(MaxIter)
    Yomperror = np.zeros(MaxIter)

    for n in range(MaxIter):
        print(f"\n=== Iteration {n+1}/{MaxIter} ===")
        Dold = D.copy()

        # ── Phase I: sparse coding ──────────────────────────
        print(f"  Phase I  (sparse coding {N} patches)")
        XK              = np.zeros((n_atoms, N))
        OMP_final_error = np.zeros(N)

        for i in range(N):
            xapp, _, omperr  = OMP(Y[:, i], D, K)
            XK[:, i]         = xapp
            OMP_final_error[i] = omperr[-1]

        Yomperror[n] = np.linalg.norm(OMP_final_error)

        # ── Phase II: dictionary update ─────────────────────
        print(f"  Phase II (dictionary update)")
        for k in range(n_atoms):
            ix = np.where(XK[k, :] != 0)[0]

            if len(ix) == 0:
                # Repurpose unused atom with a random unit vector
                v = np.random.randn(M)
                D[:, k] = v / np.linalg.norm(v)
                continue

            Dind = [j for j in range(n_atoms) if j != k]
            Dp   = D[:, Dind]
            Xp   = XK[Dind, :]

            YRk = Y[:, ix]
            XRk = Xp[:, ix]
            ER  = YRk - Dp @ XRk

            if ER.shape[1] > 0:
                U, s, Vt   = np.linalg.svd(ER, full_matrices=False)
                D[:, k]    = U[:, 0]
                XK[k, ix]  = s[0] * Vt[0, :]

        # ── Progress ─────────────────────────────────────────
        dprog[n] = np.linalg.norm(D - Dold) / (D.shape[0] * D.shape[1])
        Yerr[n]  = np.linalg.norm(Y - D @ XK) ** 2
        print(f"  dict change={dprog[n]:.6f}   Yerr={Yerr[n]:.2f}")

    return XK, D, dprog, Yerr, Yomperror


# ─────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────
def load_image_as_gray(path):
    """Load any PNG/JPG/BMP and return a float64 grayscale array."""
    img = Image.open(path).convert('L')   # 'L' = 8-bit grayscale
    return np.array(img, dtype=np.float64)


def extract_patches(img, patch_size=8):
    """
    Extract non-overlapping patch_size×patch_size patches.
    Returns Y (patch_dim, n_patches), mu (n_patches,), and padded image shape.
    """
    h, w = img.shape
    ps   = patch_size

    # Pad image so dimensions are multiples of patch_size
    pad_h = (ps - h % ps) % ps
    pad_w = (ps - w % ps) % ps
    if pad_h or pad_w:
        img = np.pad(img, ((0, pad_h), (0, pad_w)), mode='reflect')

    ph, pw      = img.shape
    n_patches   = (ph // ps) * (pw // ps)
    patch_dim   = ps * ps

    Y  = np.zeros((patch_dim, n_patches))
    mu = np.zeros(n_patches)
    k  = 0

    for i in range(0, ph, ps):
        for j in range(0, pw, ps):
            patch    = img[i:i+ps, j:j+ps].copy()
            mpatch   = patch.mean()
            patch   -= mpatch
            Y[:, k]  = patch.ravel()
            mu[k]    = mpatch
            k       += 1

    return Y, mu, (ph, pw)


def reconstruct_image(Yapp, padded_shape, patch_size=8):
    """Fold patches back into an image."""
    ph, pw = padded_shape
    ps     = patch_size
    img    = np.zeros((ph, pw))
    k      = 0

    for i in range(0, ph, ps):
        for j in range(0, pw, ps):
            img[i:i+ps, j:j+ps] = Yapp[:, k].reshape(ps, ps)
            k += 1

    return img


def init_dictionary(Y, DsizeK):
    """Initialise dictionary with random non-zero training patches."""
    patch_dim   = Y.shape[0]
    numpatches  = Y.shape[1]
    D0          = np.zeros((patch_dim, DsizeK))
    idx         = np.random.permutation(numpatches)
    count       = 0

    for i in range(DsizeK):
        while count < numpatches and np.linalg.norm(Y[:, idx[count]]) < 1e-8:
            count += 1
        if count >= numpatches:
            raise RuntimeError('Not enough nonzero patches to initialise dictionary')
        D0[:, i] = Y[:, idx[count]]
        count    += 1

    norms          = np.linalg.norm(D0, axis=0, keepdims=True)
    norms[norms == 0] = 1
    return D0 / norms


# ─────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser(description='K-SVD on a single image')
    parser.add_argument('--image',   required=True,       help='Path to PNG/JPG/BMP image')
    parser.add_argument('--iters',   type=int, default=5, help='K-SVD iterations (default 5)')
    parser.add_argument('--T',       type=int, default=10,help='Sparsity (default 10)')
    parser.add_argument('--dict',    type=int, default=512,help='Dictionary size (default 512)')
    parser.add_argument('--patch',   type=int, default=8, help='Patch size (default 8)')
    args = parser.parse_args()

    MaxIter    = args.iters
    T          = args.T
    DsizeK     = args.dict
    patch_size = args.patch
    method     = 'OMP'
    title_base = f"{method}, T={T}, Size={DsizeK}, Iters={MaxIter}"

    # ── Load & convert ───────────────────────────────────────
    print(f"Loading image: {args.image}")
    original_gray = load_image_as_gray(args.image)
    orig_h, orig_w = original_gray.shape
    print(f"  Image size: {orig_h}×{orig_w}")

    tic = time.time()

    # ── Extract patches ──────────────────────────────────────
    Y, mu, padded_shape = extract_patches(original_gray, patch_size)
    print(f"  Patches: {Y.shape[1]}  Patch dim: {Y.shape[0]}")

    # ── Initialise dictionary ────────────────────────────────
    D0 = init_dictionary(Y, DsizeK)

    # ── K-SVD ────────────────────────────────────────────────
    XK, D, dprog, Yerr, Yomperror = MyKSVD(Y, D0, T, MaxIter)

    # ── Reconstruct ──────────────────────────────────────────
    Yapp   = D @ XK + mu                             # restore mean
    appimg = reconstruct_image(Yapp, padded_shape, patch_size)
    appimg = appimg[:orig_h, :orig_w]                # crop padding

    etime = time.time() - tic
    err1  = np.linalg.norm(appimg - original_gray) / np.sqrt(orig_h * orig_w)
    print(f"\nExecution time : {etime:.2f}s")
    print(f"Mean pixel error: {err1:.4f}")

    # ── Plot 1: original vs approximation ────────────────────
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    axes[0].imshow(original_gray, cmap='gray', vmin=0, vmax=255)
    axes[0].set_title('Original'); axes[0].axis('off')
    im = axes[1].imshow(appimg, cmap='gray', vmin=0, vmax=255)
    axes[1].set_title('K-SVD Approximation'); axes[1].axis('off')
    plt.colorbar(im, ax=axes[1])
    fig.suptitle(f"Approximation ({title_base})\n"
                 f"Time: {etime:.2f}s   Mean pixel error: {err1:.4f}", fontsize=12)
    plt.tight_layout()

    # ── Plot 2: error image ───────────────────────────────────
    diff = appimg - original_gray
    fig2, ax2 = plt.subplots(figsize=(6, 5))
    im2 = ax2.imshow(diff, cmap='gray')
    plt.colorbar(im2, ax=ax2)
    ax2.axis('off')
    fig2.suptitle(f"Error ({title_base})", fontsize=12)
    plt.tight_layout()

    # ── Plot 3: Yerr curve ────────────────────────────────────
    fig3, ax3 = plt.subplots()
    ax3.plot(np.arange(1, MaxIter+1), Yerr, marker='o')
    ax3.set_ylabel('Y approximation error')
    ax3.set_xlabel('Iteration')
    fig3.suptitle(f"Y Error ({title_base})")
    plt.tight_layout()

    # ── Plot 4: error histogram ───────────────────────────────
    fig4, ax4 = plt.subplots()
    ax4.hist(diff.ravel(), bins=50)
    ax4.set_xlabel('Pixel error')
    ax4.set_ylabel('Count')
    fig4.suptitle(f"Error Distribution ({title_base})", fontsize=12)
    plt.tight_layout()

    # ── Plot 5: top-16 atoms by variance ─────────────────────
    v       = np.var(D, axis=0, ddof=0)
    ord_idx = np.argsort(v)[::-1]
    mx      = np.abs(D).max()

    fig5, axes5 = plt.subplots(4, 4, figsize=(10, 10))
    for pos, atom_i in enumerate(ord_idx[:16]):
        ax = axes5[pos // 4][pos % 4]
        ax.imshow(D[:, atom_i].reshape(patch_size, patch_size),
                  cmap='gray', vmin=-mx, vmax=mx)
        ax.axis('off')
    fig5.suptitle(f"Atoms by decreasing variance ({title_base})", fontsize=12)
    plt.tight_layout()

    plt.show()

    # ── Save ──────────────────────────────────────────────────
    fname = f"KSVD_{method}_T{T}_Size{DsizeK}_Iters{MaxIter}.npz"
    np.savez(fname, appimg=appimg, original=original_gray,
             err1=err1, Yerr=Yerr, D=D)
    print(f"Results saved to {fname}")


if __name__ == '__main__':
    main()
    # Run code:
    # python Homework3/ksvd_single_image.py --image Homework3/data/cat.jpg
    # Additional arguments:
    # --image (required) Path to your PNG/JPG/BMP
    # --T Default: 10 Sparsity level
    # --dict Default: 512 Dictionary size
    # --iters Default: 5 K-SVD iterations
    # --patch Default: 8 Patch size (8×8)