"""
Matrix-Free Lanczos Solver
==========================

Computes lowest eigenvalues/eigenvectors WITHOUT building or storing
the full Hamiltonian matrix. Instead, H|ψ⟩ is computed on-the-fly
by looping over all basis states and accumulating hopping/diagonal
contributions directly.

Memory savings:
    Stored H:    O(D × nnz_per_row) ≈ O(D × L)   [can be ~80 GB for D=300M]
    Matrix-free: O(D)                               [just 3 vectors ≈ 7 GB]

Performance:
    Pure Python fallback works for small systems. For large systems,
    install numba (pip install numba) for JIT compilation + automatic
    multicore parallelism via @prange.

The matrix-free H|ψ⟩ is wrapped as a scipy LinearOperator and passed
to ARPACK (eigsh) which handles the Lanczos iteration, implicit
restart, and eigenvector reconstruction.

This solver is model-independent: any model that provides a basis
and neighbor_pairs can use it.
"""

import numpy as np
from scipy.sparse.linalg import LinearOperator, eigsh

# Attempt to import numba for JIT acceleration
try:
    from numba import njit, prange
    HAS_NUMBA = True
except ImportError:
    HAS_NUMBA = False


# =====================================================================
# Pure Python fallback (slow, but always available)
# =====================================================================

def _apply_H_python(psi_in, psi_out, basis, hopping, interaction,
                    chemical_potential, neighbor_pairs, max_occupation):
    """Compute psi_out = H @ psi_in without building H.

    Row-based approach: for each state k, accumulates
        psi_out[k] = diagonal[k] * psi_in[k]
                    + sum_{targets of k} H[k, target] * psi_in[target]

    Each k only writes to psi_out[k], so this is naturally parallelizable
    (no write conflicts). The matrix element H[k, target] is obtained
    by hopping FROM state k (giving H[target, k]) and using Hermiticity.

    Parameters
    ----------
    psi_in : numpy.ndarray, shape (dim,)
    psi_out : numpy.ndarray, shape (dim,)
        Output array (overwritten).
    basis : UnaryBasis or FockBasis
    hopping, interaction, chemical_potential : float
    neighbor_pairs : list of (int, int)
    max_occupation : int or None
    """
    dim = basis.dim
    L = basis.num_sites
    psi_out[:] = 0.0

    for k in range(dim):
        occupation = basis.get_state(k)

        # --- Diagonal: (U/2) n_i(n_i-1) - μ n_i ---
        diag = 0.0
        for s in range(L):
            n_i = occupation[s]
            diag += (interaction / 2.0) * n_i * (n_i - 1)
            diag -= chemical_potential * n_i

        val = diag * psi_in[k]

        # --- Off-diagonal: hopping ---
        for site_i, site_j in neighbor_pairs:

            # b†_i b_j : hop from site_j to site_i
            n_src = occupation[site_j]
            n_tgt = occupation[site_i]
            if n_src > 0 and (max_occupation is None
                              or n_tgt < max_occupation):
                new_occ = list(occupation)
                new_occ[site_j] -= 1
                new_occ[site_i] += 1
                target = basis.get_index(tuple(new_occ))
                if target >= 0:
                    mel = -hopping * np.sqrt(n_src * new_occ[site_i])
                    val += mel * psi_in[target]

            # b†_j b_i : hop from site_i to site_j
            n_src = occupation[site_i]
            n_tgt = occupation[site_j]
            if n_src > 0 and (max_occupation is None
                              or n_tgt < max_occupation):
                new_occ = list(occupation)
                new_occ[site_i] -= 1
                new_occ[site_j] += 1
                target = basis.get_index(tuple(new_occ))
                if target >= 0:
                    mel = -hopping * np.sqrt(n_src * new_occ[site_j])
                    val += mel * psi_in[target]

        psi_out[k] = val


# =====================================================================
# Numba-accelerated version (optional, ~50-100× faster)
# =====================================================================

if HAS_NUMBA:

    @njit(cache=True)
    def _unary_to_occ(v, L, occ):
        """Decode unary integer to occupation array (Numba-compatible)."""
        for s in range(L):
            n_i = 0
            while (v & 1) == 0 and v != 0:
                n_i += 1
                v >>= 1
            occ[s] = n_i
            v >>= 1  # skip wall bit

    @njit(cache=True)
    def _occ_to_unary(occ, L):
        """Encode occupation array as unary integer (Numba-compatible)."""
        v = np.int64(0)
        bp = np.int64(0)
        for s in range(L):
            bp += np.int64(occ[s])
            v |= (np.int64(1) << bp)
            bp += np.int64(1)
        return v

    @njit(cache=True)
    def _binary_search(arr, val):
        """O(log D) binary search on sorted int64 array."""
        lo = np.int64(0)
        hi = np.int64(len(arr))
        while lo < hi:
            mid = (lo + hi) // np.int64(2)
            if arr[mid] < val:
                lo = mid + np.int64(1)
            else:
                hi = mid
        if lo < np.int64(len(arr)) and arr[lo] == val:
            return lo
        return np.int64(-1)

    @njit(parallel=True, cache=True)
    def _apply_H_numba(psi_in, psi_out, integers, L, n_max,
                       hopping, interaction, chem_pot, bonds):
        """Numba-parallelized H|ψ⟩ computation.

        Each thread processes a chunk of basis states (via prange).
        The row-based approach means each thread writes only to its
        own psi_out[k], avoiding race conditions.
        """
        dim = len(psi_in)
        num_bonds = bonds.shape[0]

        for k in prange(dim):
            # Thread-local occupation buffers
            occ = np.empty(L, dtype=np.int64)
            new_occ = np.empty(L, dtype=np.int64)

            # Decode state k from unary integer
            _unary_to_occ(integers[k], L, occ)

            # --- Diagonal ---
            diag = 0.0
            for s in range(L):
                n_i = occ[s]
                diag += (interaction / 2.0) * n_i * (n_i - 1)
                diag -= chem_pot * n_i

            val = diag * psi_in[k]

            # --- Off-diagonal: hopping over all bonds ---
            for b in range(num_bonds):
                si = bonds[b, 0]
                sj = bonds[b, 1]

                # b†_i b_j : hop from sj to si
                if occ[sj] > 0 and (n_max < 0 or occ[si] < n_max):
                    for s in range(L):
                        new_occ[s] = occ[s]
                    new_occ[sj] -= 1
                    new_occ[si] += 1
                    tv = _occ_to_unary(new_occ, L)
                    ti = _binary_search(integers, tv)
                    if ti >= 0:
                        val += (-hopping
                                * np.sqrt(float(occ[sj] * new_occ[si]))
                                * psi_in[ti])

                # b†_j b_i : hop from si to sj
                if occ[si] > 0 and (n_max < 0 or occ[sj] < n_max):
                    for s in range(L):
                        new_occ[s] = occ[s]
                    new_occ[si] -= 1
                    new_occ[sj] += 1
                    tv = _occ_to_unary(new_occ, L)
                    ti = _binary_search(integers, tv)
                    if ti >= 0:
                        val += (-hopping
                                * np.sqrt(float(occ[si] * new_occ[sj]))
                                * psi_in[ti])

            psi_out[k] = val


# =====================================================================
# Solver interface
# =====================================================================

def solve_matrix_free(model, num_eigenvalues=1, use_numba=None):
    """Solve for lowest eigenvalues/eigenvectors without building H.

    Uses scipy's ARPACK (eigsh) with a LinearOperator that computes
    H|ψ⟩ on the fly. If numba is installed and use_numba is not False,
    the matrix-vector product is JIT-compiled and parallelized.

    Parameters
    ----------
    model : BoseHubbard1D (or any model with .basis, ._get_neighbor_pairs(),
            .hopping, .interaction, .chemical_potential, .max_occupation)
        The model to solve. Uses the FULL basis, regardless of
        use_symmetry setting (symmetry is a separate optimization).
    num_eigenvalues : int
        Number of lowest eigenvalues to compute (default 1).
    use_numba : bool or None
        Force Numba on/off. None = auto-detect.

    Returns
    -------
    eigenvalues : numpy.ndarray, shape (num_eigenvalues,)
    eigenvectors : numpy.ndarray, shape (dim, num_eigenvalues)
    """
    if use_numba is None:
        use_numba = HAS_NUMBA

    basis = model.basis
    dim = basis.dim
    bonds = np.array(model._get_neighbor_pairs(), dtype=np.int64)

    if use_numba and HAS_NUMBA:
        # Use Numba-accelerated H|ψ⟩
        integers = basis._integers.astype(np.int64)
        L = np.int64(basis.num_sites)
        # n_max = -1 signals "unrestricted" to the Numba kernel
        n_max = np.int64(
            basis.max_occupation
            if basis.max_occupation != basis.total_particles
            else -1
        )
        t = float(model.hopping)
        U = float(model.interaction)
        mu = float(model.chemical_potential)

        def matvec(v):
            out = np.zeros(dim, dtype=np.float64)
            _apply_H_numba(v, out, integers, L, n_max, t, U, mu, bonds)
            return out

    else:
        # Pure Python fallback
        neighbor_pairs = model._get_neighbor_pairs()
        max_occ = model.max_occupation

        def matvec(v):
            out = np.zeros(dim, dtype=np.float64)
            _apply_H_python(v, out, basis, model.hopping, model.interaction,
                            model.chemical_potential, neighbor_pairs, max_occ)
            return out

    H_op = LinearOperator((dim, dim), matvec=matvec, dtype=np.float64)

    # ARPACK (Fortran) uses 32-bit integers for maxiter.
    # dim × 10 overflows int32 when dim > ~214 million
    # (e.g. L=N=16 has dim ≈ 601M, giving 6 billion → negative).
    # Cap at 1M; ground state typically converges in O(100–1000).
    safe_maxiter = min(dim * 10, 1_000_000)

    eigenvalues, eigenvectors = eigsh(
        H_op, k=num_eigenvalues, which='SA',
        maxiter=safe_maxiter,
    )

    # Sort by eigenvalue (eigsh doesn't guarantee order)
    sort_idx = np.argsort(eigenvalues)
    return eigenvalues[sort_idx], eigenvectors[:, sort_idx]
