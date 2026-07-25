"""
Particle-Partitioned Entanglement Entropy — Large-system implementation
=======================================================================

Computes the n-body reduced density matrix rho_n and its Renyi entropy
S_2(n) using chunked BLAS accumulation.  Designed for systems where
the full V matrix (D_n x D_{N-n}) doesn't fit in RAM but the reduced
density matrix rho_n (D_n x D_n) does.

Algorithm (Herdman et al., PRB 94, 064524, 2016):

    1. Expand the ground state to a flat array c[rank] indexed by the
       combinatorial rank of each occupation vector.  O(1) lookup.

    2. Precompute a sqrt(C(a,b)) table — no comb() calls in the hot loop.

    3. Stream through (N-n)-particle configs in chunks of size K:
         V[f, g] = sqrt(prod_i C(f_i+g_i, f_i) / C(N,n)) * c[rank(f+g)]
         rho += V_chunk @ V_chunk^T                  (BLAS dgemm)

    4. Compute S_2(n) = -log Tr(rho^2) = -log ||rho||_F^2

Memory: O(dim_full) for the expanded wavefunction + O(D_n^2) for rho
        + O(D_n x chunk_size) for each V chunk.

Feasibility at L=N=16:
    n=1..2 : trivial (seconds)
    n=3    : ~2 min
    n=4    : ~10-20 min
    n=5    : ~1-3 h
    n>=6   : rho doesn't fit in RAM (>23 GB)

Reference:
    H. Barghathi, E. Casiano-Diaz, A. Del Maestro,
    PRB 105, L121116 (2022)
"""

import time
import numpy as np
from math import comb

from numba import njit, prange


# =====================================================================
# Precomputed tables
# =====================================================================

def _build_binom_table(max_n):
    """Pascal's triangle: btable[n, k] = C(n, k) for 0 <= k <= n <= max_n.

    Stored as int64.  Overflows at C(66, 33) ~ 7.2e18, which is well
    beyond any system we'd diagonalize.
    """
    bt = np.zeros((max_n + 1, max_n + 1), dtype=np.int64)
    for n in range(max_n + 1):
        bt[n, 0] = 1
        for k in range(1, n + 1):
            bt[n, k] = bt[n - 1, k - 1] + bt[n - 1, k]
    return bt


def _build_sqrt_binom_table(max_n):
    """sqrt(C(n, k)) lookup.  Same shape as _build_binom_table."""
    return np.sqrt(_build_binom_table(max_n).astype(np.float64))


# =====================================================================
# Combinatorial rank / unrank  (Numba-compiled)
#
# Stars-and-bars encoding: occupation (n_0, ..., n_{L-1}) with sum = N
# maps to a binary string of N zeros (balls) and L-1 ones (walls):
#     0^{n_0} 1  0^{n_1} 1  ...  1  0^{n_{L-1}}
# of length N + L - 1.
#
# Wall s sits at position  p_s = sum(occ[0:s+1]) + s.
# Co-lex rank = sum_{s=0}^{L-2} C(p_s, s+1).
# =====================================================================

@njit(cache=True)
def _occ_rank(occ, L, btable):
    """Rank of an occupation vector (co-lex on wall positions)."""
    rank = np.int64(0)
    cumsum = np.int64(0)
    for s in range(L - 1):
        cumsum += np.int64(occ[s])
        p = cumsum + np.int64(s)
        rank += btable[p, s + 1]
    return rank


@njit(cache=True)
def _unrank_occ(rank, L, N, btable):
    """Occupation vector from its co-lex combinatorial rank.

    Decodes wall positions via binary search on the precomputed
    binomial table, then converts to site occupations.
    """
    occ = np.zeros(L, dtype=np.int64)
    remaining = rank
    prev_p = np.int64(N + L)          # sentinel > any valid position

    walls = np.empty(L - 1, dtype=np.int64)
    for s in range(L - 2, -1, -1):
        k = np.int64(s + 1)
        lo = k - np.int64(1)
        hi = min(prev_p - np.int64(1), np.int64(N + L - 2))
        # Binary search: largest p in [lo, hi] with C(p, k) <= remaining
        while lo < hi:
            mid = (lo + hi + np.int64(1)) // np.int64(2)
            if btable[mid, k] <= remaining:
                lo = mid
            else:
                hi = mid - np.int64(1)
        walls[s] = lo
        remaining -= btable[lo, k]
        prev_p = lo

    occ[0] = walls[0]
    for s in range(1, L - 1):
        occ[s] = walls[s] - walls[s - 1] - np.int64(1)
    occ[L - 1] = np.int64(N + L - 2) - walls[L - 2]
    return occ


# =====================================================================
# Expand wavefunction to rank-indexed array
# =====================================================================

@njit(cache=True)
def _unary_to_occ_local(v, L, occ):
    """Decode a unary (balls-and-walls) integer into an occupation array."""
    for s in range(L):
        n_i = np.int64(0)
        while (v & np.int64(1)) == np.int64(0) and v != np.int64(0):
            n_i += np.int64(1)
            v >>= np.int64(1)
        occ[s] = n_i
        v >>= np.int64(1)


@njit(parallel=True, cache=True)
def _expand_to_rank_array(integers, amplitudes, L, btable, dim_full):
    """Build dense array  c[rank(occ_k)] = amplitude_k  for every basis state.

    After this call, looking up the amplitude for any occupation vector
    is a single rank computation + one array read — no hashing.
    """
    c = np.zeros(dim_full, dtype=np.float64)
    dim = len(integers)
    for k in prange(dim):
        occ = np.empty(L, dtype=np.int64)
        _unary_to_occ_local(integers[k], L, occ)
        r = _occ_rank(occ, L, btable)
        c[r] = amplitudes[k]
    return c


# =====================================================================
# Enumerate a chunk of occupation vectors via unranking
# =====================================================================

@njit(parallel=True, cache=True)
def _enumerate_chunk(L, N, btable, start, count):
    """Compute occupation vectors for ranks [start, start+count).

    Each unranking is O(L log(N+L)), parallelised over the chunk.
    """
    out = np.empty((count, L), dtype=np.int64)
    for i in prange(count):
        occ = _unrank_occ(np.int64(start + i), L, N, btable)
        for j in range(L):
            out[i, j] = occ[j]
    return out


# =====================================================================
# Fill one V chunk  (core inner loop — Numba parallel)
# =====================================================================

@njit(parallel=True, cache=True)
def _fill_V_chunk(states_A, chunk_B, c_full, L, btable, sqrt_bt,
                  inv_sqrt_C):
    """Fill V[f_idx, g_idx] for one chunk of B-partition configs.

    V[f, g] = sqrt(prod_i C(f_i+g_i, f_i) / C(N, n_A)) * c[rank(f+g)]

    The product of sqrt-binomials is evaluated via the precomputed
    sqrt_bt table (no comb() calls).  The amplitude lookup is O(L)
    via _occ_rank into the flat c_full array.

    Parallelised over g (columns), since each column is independent.
    """
    D_A = states_A.shape[0]
    D_chunk = chunk_B.shape[0]
    V = np.zeros((D_A, D_chunk), dtype=np.float64)

    for g_idx in prange(D_chunk):
        F = np.empty(L, dtype=np.int64)
        for f_idx in range(D_A):
            weight = 1.0
            for i in range(L):
                f_i = states_A[f_idx, i]
                g_i = chunk_B[g_idx, i]
                F[i] = f_i + g_i
                # sqrt(C(f_i + g_i, f_i))  from the precomputed table
                weight *= sqrt_bt[F[i], f_i]

            r = _occ_rank(F, L, btable)
            V[f_idx, g_idx] = c_full[r] * weight * inv_sqrt_C

    return V


# =====================================================================
# Public API
# =====================================================================

def compute_ppee(wavefunction, basis, n_A, renyi_index=2.0,
                 chunk_size=30000, verbose=False):
    """Compute particle-partitioned Renyi entropy using chunked BLAS.

    Parameters
    ----------
    wavefunction : ndarray, shape (dim,)
        Ground state in the FULL (non-symmetry-reduced) Fock basis.
    basis : UnaryBasis
        Full canonical basis with ``._integers`` attribute.
    n_A : int
        Number of particles in partition A  (1 <= n_A <= N-1).
    renyi_index : float
        Renyi index alpha (default 2.0).
    chunk_size : int
        B-partition configs per BLAS chunk.  Memory per chunk is
        D_nA x chunk_size x 8 bytes.  Tune down if RAM is tight.
    verbose : bool
        Print progress, timing, and built-in sanity checks.

    Returns
    -------
    result : dict
        S_alpha     : float  — Renyi entropy S_alpha(n_A)
        trace       : float  — Tr(rho_n)  (should be 1.0)
        trace_rho2  : float  — Tr(rho_n^2)
        D_nA        : int    — dim of n_A-particle space
        D_nB        : int    — dim of (N-n_A)-particle space
        time_s      : float  — wall time in seconds
    """
    L = basis.num_sites
    N = getattr(basis, 'total_particles', None)
    if N is None:
        raise ValueError("Basis must have fixed particle number (canonical)")
    n_B = N - n_A
    if n_A < 1 or n_A >= N:
        raise ValueError(f"n_A must be in 1..{N-1}, got {n_A}")

    # ---- precompute tables ----
    max_val = N + L
    btable = _build_binom_table(max_val)
    sqrt_bt = _build_sqrt_binom_table(max_val)

    D_nA = int(btable[n_A + L - 1, n_A])
    D_nB = int(btable[n_B + L - 1, n_B])
    dim_full = int(btable[N + L - 1, N])
    inv_sqrt_C = 1.0 / np.sqrt(float(comb(N, n_A)))

    if verbose:
        rho_mb = D_nA ** 2 * 8 / 1e6
        chunk_mb = D_nA * chunk_size * 8 / 1e6
        cfull_mb = dim_full * 8 / 1e6
        print(f"PPEE  L={L}  N={N}  n_A={n_A}")
        print(f"  D_nA={D_nA:,}  D_nB={D_nB:,}  dim_full={dim_full:,}")
        print(f"  rho: {rho_mb:.0f} MB   chunk buf: {chunk_mb:.0f} MB   "
              f"c_full: {cfull_mb:.0f} MB")

    t0 = time.time()

    # ---- step 1: expand wavefunction to rank-indexed array ----
    if verbose:
        print("  [1] Expanding wavefunction to rank array ...", flush=True)
    integers = basis._integers.astype(np.int64)
    c_full = _expand_to_rank_array(
        integers, wavefunction.astype(np.float64),
        np.int64(L), btable, dim_full,
    )
    if verbose:
        norm = np.dot(c_full, c_full)
        print(f"      done  |c|^2 = {norm:.12f}  ({time.time()-t0:.1f} s)")

    # ---- step 2: enumerate n_A-particle states ----
    if verbose:
        print(f"  [2] Enumerating {D_nA:,} n_A-particle states ...",
              flush=True)
    states_A = _enumerate_chunk(np.int64(L), np.int64(n_A),
                                btable, np.int64(0), np.int64(D_nA))
    if verbose:
        print(f"      done  ({time.time()-t0:.1f} s)")

    # ---- step 3: chunked V V^T accumulation ----
    rho = np.zeros((D_nA, D_nA), dtype=np.float64)
    n_chunks = (D_nB + chunk_size - 1) // chunk_size

    if verbose:
        print(f"  [3] Processing {n_chunks} chunks of {chunk_size} ...")

    for ci in range(n_chunks):
        start = ci * chunk_size
        count = min(chunk_size, D_nB - start)

        chunk_B = _enumerate_chunk(np.int64(L), np.int64(n_B),
                                   btable, np.int64(start), np.int64(count))

        V = _fill_V_chunk(states_A, chunk_B, c_full, np.int64(L),
                          btable, sqrt_bt, inv_sqrt_C)

        # rho += V @ V^T  (calls BLAS dgemm under the hood)
        rho += V @ V.T

        if verbose:
            done = ci + 1
            if done % max(1, n_chunks // 10) == 0 or done == n_chunks:
                elapsed = time.time() - t0
                eta = elapsed / done * (n_chunks - done)
                print(f"      chunk {done}/{n_chunks}  "
                      f"({100*done/n_chunks:.0f}%  ETA {eta:.0f} s)")

    del c_full  # free 2.4 GB

    # ---- step 4: entropy and validation ----
    trace = np.trace(rho)
    # Tr(rho^2) = ||rho||_F^2 for real symmetric rho
    trace_rho2 = np.sum(rho * rho)

    alpha = renyi_index
    if abs(alpha - 2.0) < 1e-10:
        S = -np.log(trace_rho2) if trace_rho2 > 1e-30 else 0.0
    elif abs(alpha - 1.0) < 1e-10:
        evals = np.linalg.eigvalsh(rho)
        evals = evals[evals > 1e-30]
        S = -np.sum(evals * np.log(evals))
    else:
        evals = np.linalg.eigvalsh(rho)
        evals = evals[evals > 1e-30]
        S = np.log(np.sum(evals ** alpha)) / (1.0 - alpha)

    elapsed = time.time() - t0

    if verbose:
        print(f"  Tr(rho) = {trace:.12f}  (should be 1)")
        print(f"  S_{alpha}(n={n_A}) = {S:.10f}")
        print(f"  Total: {elapsed:.1f} s")

    return {
        'S_alpha': S,
        'renyi_index': alpha,
        'n_A': n_A,
        'trace': trace,
        'trace_rho2': trace_rho2,
        'D_nA': D_nA,
        'D_nB': D_nB,
        'time_s': elapsed,
    }


# =====================================================================
# Validation utilities  (§4 of the handoff document)
# =====================================================================

def validate_n1_diagonal(rho_1, wavefunction, basis):
    """Cross-check: diagonal of rho_1 should equal <n_i>/N.

    For n_A = 1 the PPEE formula reduces to the one-body density
    matrix divided by N (Herdman et al. 2016).  Comparing the
    diagonal against the cheaply computed density profile provides
    a strong independent check.

    Parameters
    ----------
    rho_1 : ndarray, shape (L, L)
        The 1-particle RDM from compute_ppee (extract via the rho
        return or re-run with n_A=1 storing the rho).
    wavefunction : ndarray
    basis : UnaryBasis

    Returns
    -------
    max_error : float
        max_i |rho_1[i,i] - <n_i>/N|.  Should be < 1e-10.
    """
    from .basic import density_profile
    N = basis.total_particles
    density = density_profile(wavefunction, basis)
    expected_diag = density / N
    # rho_1 rows correspond to unit-occupation states in co-lex rank order
    # For n_A=1 on L sites, rank of e_i = i (since wall at position i)
    # Verify: e_0 = (1,0,...,0), walls at [0, ...]. rank = C(0,1) = 0. ✓ for i=0
    # Actually, e_s = occupation with 1 particle on site s.
    # Wall position: p_0 = occ[0] + 0.
    #   For e_0: p_0 = 1. rank = C(1,1) = 1.   Not 0!
    # Hmm, let me recompute.  For L=3, N=1:
    #   e_0 = (1,0,0): walls at 1,2. rank = C(1,1)+C(2,2) = 1+1 = 2
    #   e_1 = (0,1,0): walls at 0,2. rank = C(0,1)+C(2,2) = 0+1 = 1
    #   e_2 = (0,0,1): walls at 0,1. rank = C(0,1)+C(1,2) = 0+0 = 0
    # So the order is reversed!  e_s maps to rank = D-1-s in general? No,
    # it depends on L.  We should just compute the rank for each e_s.

    L = basis.num_sites
    btable = _build_binom_table(1 + L)
    errors = np.empty(L)
    for s in range(L):
        occ = np.zeros(L, dtype=np.int64)
        occ[s] = 1
        r = _occ_rank(occ, np.int64(L), btable)
        errors[s] = abs(rho_1[r, r] - expected_diag[s])

    return float(np.max(errors))


def compute_ppee_with_rho(wavefunction, basis, n_A, renyi_index=2.0,
                          chunk_size=30000, verbose=False):
    """Like compute_ppee but also returns the full rho matrix.

    Useful for the n=1 cross-check and debugging.
    Memory: stores D_nA x D_nA matrix (may be large for n >= 5).
    """
    L = basis.num_sites
    N = basis.total_particles
    n_B = N - n_A

    max_val = N + L
    btable = _build_binom_table(max_val)
    sqrt_bt = _build_sqrt_binom_table(max_val)

    D_nA = int(btable[n_A + L - 1, n_A])
    D_nB = int(btable[n_B + L - 1, n_B])
    dim_full = int(btable[N + L - 1, N])
    inv_sqrt_C = 1.0 / np.sqrt(float(comb(N, n_A)))

    t0 = time.time()
    integers = basis._integers.astype(np.int64)
    c_full = _expand_to_rank_array(
        integers, wavefunction.astype(np.float64),
        np.int64(L), btable, dim_full,
    )

    states_A = _enumerate_chunk(np.int64(L), np.int64(n_A),
                                btable, np.int64(0), np.int64(D_nA))

    rho = np.zeros((D_nA, D_nA), dtype=np.float64)
    n_chunks = (D_nB + chunk_size - 1) // chunk_size

    for ci in range(n_chunks):
        start = ci * chunk_size
        count = min(chunk_size, D_nB - start)
        chunk_B = _enumerate_chunk(np.int64(L), np.int64(n_B),
                                   btable, np.int64(start), np.int64(count))
        V = _fill_V_chunk(states_A, chunk_B, c_full, np.int64(L),
                          btable, sqrt_bt, inv_sqrt_C)
        rho += V @ V.T

    del c_full

    trace = np.trace(rho)
    trace_rho2 = np.sum(rho * rho)
    alpha = renyi_index

    if abs(alpha - 2.0) < 1e-10:
        S = -np.log(trace_rho2) if trace_rho2 > 1e-30 else 0.0
    elif abs(alpha - 1.0) < 1e-10:
        evals = np.linalg.eigvalsh(rho)
        evals = evals[evals > 1e-30]
        S = -np.sum(evals * np.log(evals))
    else:
        evals = np.linalg.eigvalsh(rho)
        evals = evals[evals > 1e-30]
        S = np.log(np.sum(evals ** alpha)) / (1.0 - alpha)

    return {
        'S_alpha': S,
        'renyi_index': alpha,
        'n_A': n_A,
        'trace': trace,
        'trace_rho2': trace_rho2,
        'rho': rho,
        'D_nA': D_nA,
        'D_nB': D_nB,
        'time_s': time.time() - t0,
    }
