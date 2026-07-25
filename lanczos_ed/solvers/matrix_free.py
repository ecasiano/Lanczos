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
    Numba JIT compilation + automatic multicore parallelism via @prange
    gives ~50-100× speedup over pure Python. Numba is required.

The matrix-free H|ψ⟩ is wrapped as a scipy LinearOperator and passed
to ARPACK (eigsh) which handles the Lanczos iteration, implicit
restart, and eigenvector reconstruction.

Supported models: 1D, 2D, 3D Bose-Hubbard (any model with a UnaryBasis
and standard hopping + on-site interaction structure).
"""

import numpy as np
from scipy.sparse.linalg import LinearOperator, eigsh
from numba import njit, prange


# =====================================================================
# Numba-accelerated H|ψ⟩ kernels
# =====================================================================

@njit(cache=True)
def _unary_to_occ(v, L, occ):
    """Decode unary integer to occupation array."""
    for s in range(L):
        n_i = 0
        while (v & 1) == 0 and v != 0:
            n_i += 1
            v >>= 1
        occ[s] = n_i
        v >>= 1  # skip wall bit


@njit(cache=True)
def _occ_to_unary(occ, L):
    """Encode occupation array as unary integer."""
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

    H = -t Σ_{<i,j>} (b†_i b_j + h.c.) + (U/2) Σ_i n_i(n_i-1) - μ Σ_i n_i
    """
    dim = len(psi_in)
    num_bonds = bonds.shape[0]

    for k in prange(dim):
        # Thread-local occupation buffers
        occ = np.empty(L, dtype=np.int64)
        new_occ = np.empty(L, dtype=np.int64)

        # Decode state k from unary integer
        _unary_to_occ(integers[k], L, occ)

        # --- Diagonal: (U/2) n_i(n_i-1) - μ n_i ---
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


@njit(parallel=True, cache=True)
def _apply_H_numba_nn(psi_in, psi_out, integers, L, n_max,
                      hopping, interaction, chem_pot, nn_int, bonds):
    """H|ψ⟩ with nearest-neighbor (extended) interaction V Σ_{<ij>} n_i n_j."""
    dim = len(psi_in)
    num_bonds = bonds.shape[0]

    for k in prange(dim):
        occ = np.empty(L, dtype=np.int64)
        new_occ = np.empty(L, dtype=np.int64)
        _unary_to_occ(integers[k], L, occ)

        # --- Diagonal: U + V + μ ---
        diag = 0.0
        for s in range(L):
            n_i = occ[s]
            diag += (interaction / 2.0) * n_i * (n_i - 1)
            diag -= chem_pot * n_i

        for b in range(num_bonds):
            diag += nn_int * occ[bonds[b, 0]] * occ[bonds[b, 1]]

        val = diag * psi_in[k]

        # --- Off-diagonal: hopping ---
        for b in range(num_bonds):
            si = bonds[b, 0]
            sj = bonds[b, 1]

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

def solve_matrix_free(model, num_eigenvalues=1):
    """Solve for lowest eigenvalues/eigenvectors without building H.

    Uses scipy's ARPACK (eigsh) with a LinearOperator that computes
    H|ψ⟩ on the fly via Numba-parallelized kernels.

    Parameters
    ----------
    model : BoseHubbard1D / 2D / 3D
        The model to solve. Must have .basis (UnaryBasis with ._integers),
        ._get_neighbor_pairs(), .hopping, .interaction, .chemical_potential,
        .max_occupation, and optionally .nn_interaction.
    num_eigenvalues : int
        Number of lowest eigenvalues to compute (default 1).

    Returns
    -------
    eigenvalues : numpy.ndarray, shape (num_eigenvalues,)
    eigenvectors : numpy.ndarray, shape (dim, num_eigenvalues)
    """
    basis = model.basis
    dim = basis.dim
    bonds = np.array(model._get_neighbor_pairs(), dtype=np.int64)
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
    nn_V = float(getattr(model, 'nn_interaction', 0.0))

    if abs(nn_V) > 1e-15:
        def matvec(v):
            out = np.zeros(dim, dtype=np.float64)
            _apply_H_numba_nn(v, out, integers, L, n_max,
                              t, U, mu, nn_V, bonds)
            return out
    else:
        def matvec(v):
            out = np.zeros(dim, dtype=np.float64)
            _apply_H_numba(v, out, integers, L, n_max, t, U, mu, bonds)
            return out

    H_op = LinearOperator((dim, dim), matvec=matvec, dtype=np.float64)

    # ARPACK (Fortran) uses 32-bit integers for maxiter.
    # dim × 10 overflows int32 when dim > ~214 million.
    # Cap at 1M; ground state typically converges in O(100–1000).
    safe_maxiter = min(dim * 10, 1_000_000)

    eigenvalues, eigenvectors = eigsh(
        H_op, k=num_eigenvalues, which='SA',
        maxiter=safe_maxiter,
    )

    # Sort by eigenvalue (eigsh doesn't guarantee order)
    sort_idx = np.argsort(eigenvalues)
    return eigenvalues[sort_idx], eigenvectors[:, sort_idx]
