"""
Basic Observables
=================

Compute physical observables from a many-body wavefunction |psi>
expressed in the Fock basis { |n_0, n_1, ..., n_{L-1}> }.

Available observables:

    1. Density profile:  <n_i>  =  sum_alpha |c_alpha|^2 * n_i^(alpha)

    2. Density-density correlations:  <n_i n_j>

    3. Bipartite particle number fluctuations:
           F_A = <N_A^2> - <N_A>^2
       where N_A = sum_{i in A} n_i is the particle number in subsystem A.

    4. Rényi entanglement entropy of a spatial bipartition A|B:
           S_alpha = 1/(1 - alpha) * log( Tr(rho_A^alpha) )
       For alpha -> 1, this gives the von Neumann entropy:
           S_1 = -Tr(rho_A log rho_A)

Memory-efficient computation
-----------------------------
For large Hilbert spaces, materializing the full dim × L occupation
array (all_states_as_array) can exceed available RAM. For example,
L = N = 16 has dim ~ 601M; the array would be 77 GB.

All observables support a loop-based path that decodes each state
on the fly via basis.get_state(k), using O(L) memory per state.
A size threshold selects the fast vectorized path for small systems.

Sector-by-sector SVD for entanglement entropy
----------------------------------------------
The old approach reshapes |psi> into a single dim_A × dim_B matrix.
For L=16 equal bipartition, that matrix is ~735K × 735K = 4.3 TB.

The sector-by-sector approach exploits the fact that any additive
conserved charge Q (particle number, S_z, etc.) makes rho_A block-
diagonal in the subsystem charge Q_A = sum_{i in A} q_i.  We build
one small matrix per charge sector, SVD each independently, and
combine the Schmidt spectra.

For L=N=16, the largest sector matrix is 6435 × 6435 ~ 330 MB.

This decomposition is model-independent: it works for any basis
where get_state(k) returns local quantum numbers and the total
charge sum(q_i) is conserved:
    - Bose-Hubbard, extended BH, disordered BH  (particle number N)
    - tV model of itinerant fermions             (fermion number N)
    - Heisenberg / XXZ spin chains               (total S_z)
    - Rydberg atom arrays                        (excitation number)

Numba acceleration (optional)
-----------------------------
When numba is installed and the basis is a UnaryBasis (has a
._integers array of encoded states), the observable loops are
JIT-compiled for ~100x speedup.  The compiled kernels are cached
to disk (cache=True), so only the first-ever call pays the
compilation cost (~2-5 seconds).

Reference:
    Laflorencie, Phys. Rep. 646, 1 (2016), Appendix A
"""

import time
import numpy as np
from ..basis import FockBasis


# =====================================================================
# Optional Numba acceleration
# =====================================================================

try:
    from numba import njit, prange
    HAS_NUMBA = True
except ImportError:
    HAS_NUMBA = False


# =====================================================================
# Memory threshold for path selection
# =====================================================================
#
# If all_states_as_array() would exceed this (bytes), use the
# loop-based path.  1 GB is conservative for most machines.
_MEMORY_THRESHOLD_BYTES = 1 * 1024**3  # 1 GB


def _use_vectorized(basis) -> bool:
    """Return True if materializing all_states_as_array() is safe."""
    return basis.dim * basis.num_sites * 8 <= _MEMORY_THRESHOLD_BYTES


def _can_use_numba(basis) -> bool:
    """Return True if Numba-accelerated path is available."""
    return HAS_NUMBA and hasattr(basis, '_integers')


# =====================================================================
# Numba kernels (compiled on first call, cached to disk)
#
# These are self-contained: they duplicate the unary encoding/
# decoding logic from unary_basis.py so that the observables module
# has no import dependency on the solvers module.
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

    # -----------------------------------------------------------------
    # Density profile kernel
    # -----------------------------------------------------------------
    @njit(cache=True)
    def _density_kernel(probabilities, integers, L):
        """Accumulate <n_i> = sum_k |c_k|^2 * n_i^(k) over all states.

        Serial loop (no write conflicts on density array).
        ~100x faster than Python even without parallelism.
        """
        dim = len(probabilities)
        density = np.zeros(L, dtype=np.float64)
        occ = np.empty(L, dtype=np.int64)

        for k in range(dim):
            p = probabilities[k]
            if p < 1e-30:
                continue
            _unary_to_occ(integers[k], L, occ)
            for s in range(L):
                density[s] += p * float(occ[s])

        return density

    # -----------------------------------------------------------------
    # Bipartite fluctuations kernel
    # -----------------------------------------------------------------
    @njit(cache=True)
    def _fluctuations_kernel(probabilities, integers, L, sub_sites):
        """Accumulate <N_A> and <N_A^2> over all states.

        Returns F_A = <N_A^2> - <N_A>^2.
        """
        dim = len(probabilities)
        n_sub = len(sub_sites)
        occ = np.empty(L, dtype=np.int64)

        mean_NA = 0.0
        mean_NA2 = 0.0

        for k in range(dim):
            p = probabilities[k]
            if p < 1e-30:
                continue
            _unary_to_occ(integers[k], L, occ)
            NA = np.int64(0)
            for i in range(n_sub):
                NA += occ[sub_sites[i]]
            fNA = float(NA)
            mean_NA += p * fNA
            mean_NA2 += p * fNA * fNA

        return mean_NA2 - mean_NA * mean_NA

    # -----------------------------------------------------------------
    # Entropy: compute subsystem charge Q_A for all states (parallel)
    # -----------------------------------------------------------------
    @njit(parallel=True, cache=True)
    def _compute_charges(integers, L, sub_sites):
        """For each state, compute Q_A = sum of occupations at A-sites.

        Uses prange for automatic multicore parallelism.
        Memory: dim × 4 bytes (int32).
        """
        dim = len(integers)
        n_sub = len(sub_sites)
        charges = np.empty(dim, dtype=np.int32)

        for k in prange(dim):
            occ = np.empty(L, dtype=np.int64)
            _unary_to_occ(integers[k], L, occ)
            q = np.int32(0)
            for i in range(n_sub):
                q += np.int32(occ[sub_sites[i]])
            charges[k] = q

        return charges

    # -----------------------------------------------------------------
    # l-sweep: compute per-site occupations for all states (parallel)
    # Then charges for any l are just cumulative sums over site columns.
    # -----------------------------------------------------------------
    @njit(parallel=True, cache=True)
    def _compute_site_occupations(integers, L, site_order):
        """Decode all states and store occupation at ordered sites.

        Parameters
        ----------
        integers : int64 array, shape (dim,)
        L : int64, total number of sites
        site_order : int64 array, shape (L_sub,)
            Ordered site indices for the subsystem sweep.
            For 1D: [0, 1, 2, ..., L//2-1].
            For 2D/3D: the full subregion site list at l_max,
            arranged so sites at l are a prefix.

        Returns
        -------
        occ_at_sites : int32 array, shape (dim, len(site_order))
            occ_at_sites[k, j] = occupation of site_order[j] in state k.
        """
        dim = len(integers)
        n_sites = len(site_order)
        occ_at_sites = np.empty((dim, n_sites), dtype=np.int32)

        for k in prange(dim):
            occ = np.empty(L, dtype=np.int64)
            _unary_to_occ(integers[k], L, occ)
            for j in range(n_sites):
                occ_at_sites[k, j] = np.int32(occ[site_order[j]])

        return occ_at_sites

    @njit(parallel=True, cache=True)
    def _cumulative_charges(occ_at_sites, l):
        """Compute Q_A = sum of first l columns of occ_at_sites.

        Parameters
        ----------
        occ_at_sites : int32 array, shape (dim, n_sites)
        l : int, number of site columns to sum

        Returns
        -------
        charges : int32 array, shape (dim,)
        """
        dim = occ_at_sites.shape[0]
        charges = np.empty(dim, dtype=np.int32)
        for k in prange(dim):
            q = np.int32(0)
            for j in range(l):
                q += occ_at_sites[k, j]
            charges[k] = q
        return charges

    @njit(parallel=True, cache=True)
    def _fluctuations_from_occ(probabilities, occ_at_sites, n_cols):
        """Compute bipartite fluctuations from precomputed occupations.

        F_A = <N_A²> - <N_A>² where N_A = sum of first n_cols columns.

        Uses parallel reduction to compute <N_A> and <N_A²>.
        """
        dim = len(probabilities)
        # Thread-local accumulators via prange reduction
        mean_NA = 0.0
        mean_NA2 = 0.0
        for k in prange(dim):
            p = probabilities[k]
            if p < 1e-30:
                continue
            NA = np.int64(0)
            for j in range(n_cols):
                NA += np.int64(occ_at_sites[k, j])
            fNA = float(NA)
            mean_NA += p * fNA
            mean_NA2 += p * fNA * fNA
        return mean_NA2 - mean_NA * mean_NA

    # -----------------------------------------------------------------
    # Entropy: encode A/B configs for one sector (parallel)
    # -----------------------------------------------------------------
    @njit(parallel=True, cache=True)
    def _sector_codes_kernel(integers, indices, L, sub_sites, comp_sites):
        """For each state in a sector, encode A-config and B-config
        as small unary integers for fast index lookup.

        Parameters
        ----------
        integers : int64 array, full basis integers
        indices : int64 array, state indices in this sector
        L : int, total number of sites
        sub_sites : int64 array, subsystem A site indices
        comp_sites : int64 array, subsystem B site indices

        Returns
        -------
        codes_A, codes_B : int64 arrays, unary codes for A/B configs
        """
        n = len(indices)
        L_A = len(sub_sites)
        L_B = len(comp_sites)
        codes_A = np.empty(n, dtype=np.int64)
        codes_B = np.empty(n, dtype=np.int64)

        for i in prange(n):
            k = indices[i]
            occ = np.empty(L, dtype=np.int64)
            _unary_to_occ(integers[k], L, occ)

            # Encode A-config
            occ_A = np.empty(L_A, dtype=np.int64)
            for j in range(L_A):
                occ_A[j] = occ[sub_sites[j]]
            codes_A[i] = _occ_to_unary(occ_A, L_A)

            # Encode B-config
            occ_B = np.empty(L_B, dtype=np.int64)
            for j in range(L_B):
                occ_B[j] = occ[comp_sites[j]]
            codes_B[i] = _occ_to_unary(occ_B, L_B)

        return codes_A, codes_B


# =====================================================================
# 1. Density profile
# =====================================================================

def density_profile(wavefunction: np.ndarray, basis) -> np.ndarray:
    """Compute the local density profile <n_i> for each site.

    The expectation value in the Fock basis is:
        <n_i> = sum_alpha |c_alpha|^2 * n_i^(alpha)

    where c_alpha = <alpha|psi> and n_i^(alpha) is the occupation
    of site i in basis state |alpha>.

    Parameters
    ----------
    wavefunction : ndarray of shape (dim,)
        State vector |psi> in the Fock basis.
    basis : FockBasis or UnaryBasis
        The basis.  Only requires dim, num_sites, get_state(k).

    Returns
    -------
    density : ndarray of shape (num_sites,)
        Local density <n_i> at each site.
    """
    probabilities = np.abs(wavefunction) ** 2

    # --- Numba path: always preferred when available ---
    # Faster than vectorized because it skips the Python-loop-based
    # all_states_as_array() and decodes unary integers directly.
    if _can_use_numba(basis):
        integers = basis._integers.astype(np.int64)
        return _density_kernel(probabilities, integers,
                               np.int64(basis.num_sites))

    # --- Vectorized path: for non-UnaryBasis small systems ---
    if _use_vectorized(basis):
        all_states = basis.all_states_as_array()
        return probabilities @ all_states

    # --- Pure Python fallback: decode states one at a time ---
    num_sites = basis.num_sites
    density = np.zeros(num_sites, dtype=np.float64)

    for k in range(basis.dim):
        p_k = probabilities[k]
        if p_k < 1e-30:
            continue
        occupation = basis.get_state(k)
        for site in range(num_sites):
            density[site] += p_k * occupation[site]

    return density


# =====================================================================
# 2. Density-density correlations
# =====================================================================

def density_density_correlation(wavefunction: np.ndarray,
                                 basis) -> np.ndarray:
    """Compute the density-density correlation matrix <n_i n_j>.

    Parameters
    ----------
    wavefunction : ndarray of shape (dim,)
    basis : FockBasis or UnaryBasis

    Returns
    -------
    correlation : ndarray of shape (num_sites, num_sites)
        correlation[i, j] = <n_i n_j>.
    """
    num_sites = basis.num_sites
    probabilities = np.abs(wavefunction) ** 2
    correlation = np.zeros((num_sites, num_sites))

    if _use_vectorized(basis):
        all_states = basis.all_states_as_array()
        for state_idx in range(basis.dim):
            prob = probabilities[state_idx]
            if prob < 1e-30:
                continue
            occ = all_states[state_idx]
            correlation += prob * np.outer(occ, occ)
    else:
        for state_idx in range(basis.dim):
            prob = probabilities[state_idx]
            if prob < 1e-30:
                continue
            occ = np.array(basis.get_state(state_idx), dtype=np.float64)
            correlation += prob * np.outer(occ, occ)

    return correlation


# =====================================================================
# 3. Bipartite particle number fluctuations
# =====================================================================

def bipartite_fluctuations(wavefunction: np.ndarray, basis,
                            subsystem_sites: list = None) -> float:
    """Compute bipartite particle number fluctuations.

        F_A = <N_A^2> - <N_A>^2

    where N_A = sum_{i in A} n_i.

    Parameters
    ----------
    wavefunction : ndarray of shape (dim,)
    basis : FockBasis or UnaryBasis
    subsystem_sites : list of int or None
        Sites in subsystem A.  Default: first L//2 sites.

    Returns
    -------
    fluctuation : float
        F_A = Var(N_A) in state |psi>.
    """
    num_sites = basis.num_sites
    if subsystem_sites is None:
        subsystem_sites = list(range(num_sites // 2))

    probabilities = np.abs(wavefunction) ** 2

    # --- Numba path: always preferred when available ---
    if _can_use_numba(basis):
        integers = basis._integers.astype(np.int64)
        sub_arr = np.array(subsystem_sites, dtype=np.int64)
        return _fluctuations_kernel(probabilities, integers,
                                     np.int64(basis.num_sites), sub_arr)

    # --- Vectorized path: for non-UnaryBasis small systems ---
    if _use_vectorized(basis):
        all_states = basis.all_states_as_array()
        particles_in_A = (
            all_states[:, subsystem_sites].sum(axis=1).astype(np.float64)
        )
        mean_N_A = np.dot(probabilities, particles_in_A)
        mean_N_A_sq = np.dot(probabilities, particles_in_A ** 2)
        return mean_N_A_sq - mean_N_A ** 2

    # --- Pure Python fallback ---
    mean_N_A = 0.0
    mean_N_A_sq = 0.0

    for k in range(basis.dim):
        p_k = probabilities[k]
        if p_k < 1e-30:
            continue
        occupation = basis.get_state(k)
        N_A = sum(occupation[s] for s in subsystem_sites)
        mean_N_A += p_k * N_A
        mean_N_A_sq += p_k * N_A * N_A

    return mean_N_A_sq - mean_N_A ** 2


# =====================================================================
# 4. Entanglement entropy (sector-by-sector SVD)
# =====================================================================

def entanglement_entropy(wavefunction: np.ndarray, basis,
                          subsystem_sites: list = None,
                          renyi_index: float = 1.0) -> float:
    """Compute the Rényi entanglement entropy of a spatial bipartition.

    Given a bipartition of sites into A and B = complement(A),
    the reduced density matrix is:

        rho_A = Tr_B( |psi><psi| )

    The Rényi-alpha entropy is:

        S_alpha = 1/(1 - alpha) * log( Tr(rho_A^alpha) )

    For alpha = 1, this gives the von Neumann entropy:

        S_1 = -Tr(rho_A * log(rho_A)) = -sum_k lambda_k * log(lambda_k)

    Method: sector-by-sector SVD exploiting Q_A conservation.
    See module docstring for details.

    Parameters
    ----------
    wavefunction : ndarray of shape (dim,)
    basis : FockBasis or UnaryBasis
        Only requires get_state(k) -> tuple of local quantum numbers.
    subsystem_sites : list of int or None
        Sites in subsystem A. Default: first L//2 sites.
    renyi_index : float
        The Rényi index alpha. alpha=1 gives von Neumann entropy.

    Returns
    -------
    entropy : float
    """
    num_sites = basis.num_sites
    if subsystem_sites is None:
        subsystem_sites = list(range(num_sites // 2))

    complement_sites = [i for i in range(num_sites)
                        if i not in subsystem_sites]

    if len(subsystem_sites) == 0 or len(complement_sites) == 0:
        return 0.0

    # Compute the Schmidt spectrum via sector-by-sector SVD
    schmidt_weights = _sector_schmidt_spectrum(
        wavefunction, basis, subsystem_sites, complement_sites,
    )

    if len(schmidt_weights) == 0:
        return 0.0

    # Compute entropy from the Schmidt weights lambda_k = sigma_k^2
    if abs(renyi_index - 1.0) < 1e-10:
        # Von Neumann: S_1 = -sum_k lambda_k * log(lambda_k)
        entropy = -np.sum(schmidt_weights * np.log(schmidt_weights))
    else:
        # Rényi: S_alpha = log(sum_k lambda_k^alpha) / (1 - alpha)
        entropy = (np.log(np.sum(schmidt_weights ** renyi_index))
                   / (1.0 - renyi_index))

    return entropy


def accessible_entanglement_entropy(wavefunction: np.ndarray, basis,
                                     subsystem_sites: list = None,
                                     renyi_index: float = 2.0) -> float:
    """Compute the generalized accessible entanglement entropy.

    Uses the formula of Barghathi, Herdman & Del Maestro
    (PRB 105, L121116, 2022):

        S_α_acc = α/(1-α) · log( Σₙ (Σₖ wₙₖ^α)^(1/α) )

    where wₙₖ = σ²ₙₖ are the squared Schmidt values in sector n.

    For α=2 this simplifies to  S₂_acc = -2 log( Σₙ √(Σₖ wₙₖ²) ).
    For α→1 (von Neumann) it reduces to  S₁_acc = Σₙ p(n) · S₁(n),
    i.e. the probability-weighted sum of per-sector von Neumann
    entropies, which equals S₁ - S₁_num.

    Reference:
        Barghathi et al., PRB 105, L121116 (2022)

    Parameters
    ----------
    wavefunction : ndarray of shape (dim,)
    basis : FockBasis or UnaryBasis
    subsystem_sites : list of int or None
        Sites in subsystem A. Default: first L//2 sites.
    renyi_index : float
        The Rényi index alpha (default 2.0).

    Returns
    -------
    s_acc : float
        The accessible entanglement entropy.
    """
    num_sites = basis.num_sites
    if subsystem_sites is None:
        subsystem_sites = list(range(num_sites // 2))

    complement_sites = [i for i in range(num_sites)
                        if i not in subsystem_sites]

    if len(subsystem_sites) == 0 or len(complement_sites) == 0:
        return 0.0

    # Get per-sector Schmidt spectra
    sector_spectra = _sector_schmidt_spectrum(
        wavefunction, basis, subsystem_sites, complement_sites,
        return_sectors=True,
    )

    if not sector_spectra:
        return 0.0

    # Generalized accessible entanglement (Barghathi et al. PRB 2022)
    # sector_spectra is a dict: charge -> array of wₙₖ = σ²ₙₖ values

    alpha = renyi_index

    if abs(alpha - 1.0) < 1e-10:
        # Von Neumann limit: S₁_acc = Σₙ p(n) · S₁(n)
        # where S₁(n) = -Σₖ (wₙₖ/p(n)) log(wₙₖ/p(n))
        s_acc = 0.0
        for charge, weights in sector_spectra.items():
            p_n = weights.sum()
            if p_n < 1e-30:
                continue
            probs = weights / p_n
            mask = probs > 1e-30
            s_acc += p_n * (-np.sum(probs[mask] * np.log(probs[mask])))
        return s_acc
    else:
        # General Rényi-α (Barghathi et al. PRB 2022):
        # S_α_acc = α/(1-α) · log( Σₙ (Σₖ wₙₖ^α)^(1/α) )
        inner = 0.0
        for charge, weights in sector_spectra.items():
            if len(weights) > 0 and weights.sum() > 1e-30:
                inner += np.sum(weights ** alpha) ** (1.0 / alpha)
        if inner < 1e-30:
            return 0.0
        return (alpha / (1.0 - alpha)) * np.log(inner)


# =====================================================================
# Schmidt spectrum: dispatch to Numba or Python
# =====================================================================

def _sector_schmidt_spectrum(wavefunction, basis, sub_sites, comp_sites,
                              return_sectors=False):
    """Compute Schmidt weights via sector-by-sector SVD.

    Dispatches to Numba-accelerated or pure Python implementation
    depending on availability and basis type.

    Parameters
    ----------
    return_sectors : bool
        If True, return a dict {charge: array of sigma^2 values}
        instead of a flat array. Used by accessible_entanglement_entropy.
    """
    if _can_use_numba(basis):
        return _sector_schmidt_numba(wavefunction, basis,
                                      sub_sites, comp_sites,
                                      return_sectors=return_sectors)
    return _sector_schmidt_python(wavefunction, basis,
                                   sub_sites, comp_sites,
                                   return_sectors=return_sectors)


# =====================================================================
# Numba-accelerated sector SVD
# =====================================================================

def _sector_schmidt_numba(wavefunction, basis, sub_sites, comp_sites,
                           return_sectors=False):
    """Schmidt spectrum using Numba-accelerated pre-computation.

    Algorithm:
        1. Parallel Numba pass: compute charge Q_A for every state.
           Memory: dim × 4 bytes (int32).

        2. For each sector Q_A:
           a. np.where to find state indices in this sector
           b. Parallel Numba pass: encode A/B configs as unary integers
           c. np.searchsorted for config -> matrix index mapping
           d. Vectorized fill: psi_sector[a_idx, b_idx] = wfn[indices]
           e. SVD, collect singular values, free sector matrix

    Peak memory (beyond wavefunction + basis):
        charges array + largest sector's data.
        For L=N=16: ~4 GB charges + ~1.3 GB largest sector = ~5.3 GB.

    If return_sectors=True, returns a dict {charge: sigma^2 array}
    for accessible entropy computation.
    """
    integers = basis._integers.astype(np.int64)
    L = np.int64(basis.num_sites)
    sub_arr = np.array(sub_sites, dtype=np.int64)
    comp_arr = np.array(comp_sites, dtype=np.int64)
    wfn_dtype = np.result_type(wavefunction.dtype, np.float64)

    # --- Step 1: Compute charges for all states (parallel Numba) ---
    charges = _compute_charges(integers, L, sub_arr)

    # --- Step 2: Process each charge sector ---
    all_schmidt_sq = []
    sector_data = {} if return_sectors else None
    q_min, q_max = int(charges.min()), int(charges.max())

    for q in range(q_min, q_max + 1):
        # Find states in this sector
        sector_indices = np.where(charges == q)[0].astype(np.int64)
        if len(sector_indices) == 0:
            continue

        # Encode A/B configs as unary integers (parallel Numba)
        codes_A, codes_B = _sector_codes_kernel(
            integers, sector_indices, L, sub_arr, comp_arr,
        )

        # Build index maps via sorted unique arrays
        unique_A = np.unique(codes_A)
        unique_B = np.unique(codes_B)
        dim_A, dim_B = len(unique_A), len(unique_B)

        # Map each state to its (row, col) in the sector matrix
        a_idx = np.searchsorted(unique_A, codes_A)
        b_idx = np.searchsorted(unique_B, codes_B)

        # Fill sector matrix (vectorized — each (a,b) is unique)
        psi_sector = np.zeros((dim_A, dim_B), dtype=wfn_dtype)
        sector_wfn = wavefunction[sector_indices]
        psi_sector[a_idx, b_idx] = sector_wfn

        # SVD this sector
        singular_values = np.linalg.svd(psi_sector, compute_uv=False)
        sq = singular_values ** 2
        valid = sq[sq > 1e-30]

        if return_sectors and len(valid) > 0:
            sector_data[q] = valid
        all_schmidt_sq.extend(valid)

        # Free this sector's memory
        del psi_sector, codes_A, codes_B, sector_wfn, sector_indices

    del charges

    if return_sectors:
        return sector_data

    if not all_schmidt_sq:
        return np.array([])

    schmidt_weights = np.array(all_schmidt_sq, dtype=np.float64)
    return schmidt_weights[schmidt_weights > 1e-15]


# =====================================================================
# Pure Python sector SVD (fallback when Numba is not available)
# =====================================================================

def _sector_schmidt_python(wavefunction, basis, sub_sites, comp_sites,
                            return_sectors=False):
    """Schmidt spectrum using pure Python state-by-state iteration.

    Two passes over the basis:
        Pass 1: discover unique A/B configs per charge sector.
        Pass 2: fill sector matrices.

    Then SVD each sector independently.

    If return_sectors=True, returns a dict {charge: sigma^2 array}
    for accessible entropy computation.
    """
    dim = basis.dim
    wfn_dtype = np.result_type(wavefunction.dtype, np.float64)

    # =================================================================
    # Pass 1: Discover unique configurations per charge sector
    # =================================================================
    sector_A_configs = {}   # Q_A -> set of occ_A tuples
    sector_B_configs = {}   # Q_A -> set of occ_B tuples

    for k in range(dim):
        occupation = basis.get_state(k)
        occ_A = tuple(occupation[s] for s in sub_sites)
        occ_B = tuple(occupation[s] for s in comp_sites)
        charge_A = sum(occ_A)

        if charge_A not in sector_A_configs:
            sector_A_configs[charge_A] = set()
            sector_B_configs[charge_A] = set()
        sector_A_configs[charge_A].add(occ_A)
        sector_B_configs[charge_A].add(occ_B)

    # Build lookup tables and allocate sector matrices
    sector_A_map = {}
    sector_B_map = {}
    sector_matrices = {}

    for charge_A in sorted(sector_A_configs):
        A_sorted = sorted(sector_A_configs[charge_A])
        B_sorted = sorted(sector_B_configs[charge_A])
        sector_A_map[charge_A] = {c: i for i, c in enumerate(A_sorted)}
        sector_B_map[charge_A] = {c: i for i, c in enumerate(B_sorted)}
        sector_matrices[charge_A] = np.zeros(
            (len(A_sorted), len(B_sorted)), dtype=wfn_dtype,
        )

    del sector_A_configs, sector_B_configs

    # =================================================================
    # Pass 2: Fill sector matrices
    # =================================================================
    for k in range(dim):
        occupation = basis.get_state(k)
        occ_A = tuple(occupation[s] for s in sub_sites)
        occ_B = tuple(occupation[s] for s in comp_sites)
        charge_A = sum(occ_A)

        a = sector_A_map[charge_A][occ_A]
        b = sector_B_map[charge_A][occ_B]
        sector_matrices[charge_A][a, b] = wavefunction[k]

    del sector_A_map, sector_B_map

    # =================================================================
    # SVD each sector, collect Schmidt spectrum
    # =================================================================
    all_schmidt_sq = []
    sector_data = {} if return_sectors else None

    for charge_A in sorted(sector_matrices):
        mat = sector_matrices[charge_A]
        if mat.size == 0:
            continue

        singular_values = np.linalg.svd(mat, compute_uv=False)
        sq = singular_values ** 2
        valid = sq[sq > 1e-30]

        if return_sectors and len(valid) > 0:
            sector_data[charge_A] = valid
        all_schmidt_sq.extend(valid)
        del mat

    sector_matrices.clear()

    if return_sectors:
        return sector_data

    if not all_schmidt_sq:
        return np.array([])

    schmidt_weights = np.array(all_schmidt_sq, dtype=np.float64)
    return schmidt_weights[schmidt_weights > 1e-15]


# =====================================================================
# 5. Combined l-sweep: all observables from one decode pass
# =====================================================================

def _entropies_from_sector_spectra(sector_spectra):
    """Compute entropies and per-sector data from Schmidt weights.

    Returns total S₁, S₂, generalized S₂_acc (Barghathi et al.
    PRB 2022), plus the symmetry-resolved per-sector entropies
    and particle number distribution.

    Parameters
    ----------
    sector_spectra : dict
        {charge: array of sigma_k² values} from sector SVD.

    Returns
    -------
    result : dict with keys
        'S_1'          : float — von Neumann entropy
        'S_2'          : float — Rényi-2 entropy
        'S_2_acc'      : float — generalized accessible Rényi-2
        'sector_S_2'   : dict {n_A: S₂(n_A)} — per-sector Rényi-2
        'sector_probs' : dict {n_A: p(n_A)}  — particle number
                         distribution (probability of n_A particles
                         in subsystem A)
    """
    all_weights = []
    sector_probs = {}    # p(n) = Σₖ wₙₖ
    sector_S_2 = {}      # S₂(n) = -log(Σₖ (wₙₖ/p(n))²)

    for charge, weights in sector_spectra.items():
        if len(weights) == 0:
            continue
        p_n = weights.sum()
        if p_n > 1e-30:
            all_weights.extend(weights)
            sector_probs[charge] = float(p_n)

            # Per-sector Rényi-2: normalize weights to probabilities
            # within this sector, then compute S₂(n)
            probs_n = weights / p_n       # conditional probs
            sector_S_2[charge] = float(-np.log(np.sum(probs_n ** 2)))

    if not all_weights:
        return {
            'S_1': 0.0, 'S_2': 0.0, 'S_2_acc': 0.0,
            'sector_S_2': {}, 'sector_probs': {},
        }

    w = np.array(all_weights, dtype=np.float64)

    # S₁ (von Neumann): -Σₖ λₖ log(λₖ)
    mask = w > 1e-30
    S_1 = -np.sum(w[mask] * np.log(w[mask]))

    # S₂ (Rényi-2): -log(Σₖ λₖ²)
    S_2 = -np.log(np.sum(w ** 2))

    # S₂_acc (generalized, Barghathi et al. PRB 2022)
    # S₂_acc = -2 log( Σₙ √(Σₖ wₙₖ²) )
    acc_total = 0.0
    for charge, weights in sector_spectra.items():
        if len(weights) > 0 and weights.sum() > 1e-30:
            acc_total += np.sqrt(np.sum(weights ** 2))
    S_2_acc = -2.0 * np.log(acc_total) if acc_total > 1e-30 else 0.0

    return {
        'S_1': S_1, 'S_2': S_2, 'S_2_acc': S_2_acc,
        'sector_S_2': sector_S_2, 'sector_probs': sector_probs,
    }


def _sector_schmidt_with_charges(wavefunction, basis, sub_sites,
                                  comp_sites, charges):
    """Sector SVD using precomputed charges (skips charge computation).

    Like _sector_schmidt_numba but takes charges as input instead
    of recomputing them.

    Parameters
    ----------
    charges : int32 array, shape (dim,)
        Precomputed Q_A for each basis state.

    Returns
    -------
    sector_spectra : dict {charge: array of sigma² values}
    """
    integers = basis._integers.astype(np.int64)
    L = np.int64(basis.num_sites)
    sub_arr = np.array(sub_sites, dtype=np.int64)
    comp_arr = np.array(comp_sites, dtype=np.int64)
    wfn_dtype = np.result_type(wavefunction.dtype, np.float64)

    sector_data = {}
    q_min, q_max = int(charges.min()), int(charges.max())

    for q in range(q_min, q_max + 1):
        sector_indices = np.where(charges == q)[0].astype(np.int64)
        if len(sector_indices) == 0:
            continue

        codes_A, codes_B = _sector_codes_kernel(
            integers, sector_indices, L, sub_arr, comp_arr,
        )

        unique_A = np.unique(codes_A)
        unique_B = np.unique(codes_B)
        dim_A, dim_B = len(unique_A), len(unique_B)

        a_idx = np.searchsorted(unique_A, codes_A)
        b_idx = np.searchsorted(unique_B, codes_B)

        psi_sector = np.zeros((dim_A, dim_B), dtype=wfn_dtype)
        psi_sector[a_idx, b_idx] = wavefunction[sector_indices]

        singular_values = np.linalg.svd(psi_sector, compute_uv=False)
        sq = singular_values ** 2
        valid = sq[sq > 1e-30]

        if len(valid) > 0:
            sector_data[q] = valid

        del psi_sector, codes_A, codes_B, sector_indices

    return sector_data


def _sector_schmidt_with_charges_python(wavefunction, basis, sub_sites,
                                         comp_sites, charges):
    """Pure Python sector SVD using precomputed charges.

    Fallback for non-UnaryBasis or when Numba is not available.
    """
    dim = basis.dim
    wfn_dtype = np.result_type(wavefunction.dtype, np.float64)

    # Pass 1: discover unique configs per sector
    sector_A_configs = {}
    sector_B_configs = {}

    for k in range(dim):
        q = int(charges[k])
        occupation = basis.get_state(k)
        occ_A = tuple(occupation[s] for s in sub_sites)
        occ_B = tuple(occupation[s] for s in comp_sites)

        if q not in sector_A_configs:
            sector_A_configs[q] = set()
            sector_B_configs[q] = set()
        sector_A_configs[q].add(occ_A)
        sector_B_configs[q].add(occ_B)

    # Build index maps and allocate matrices
    sector_A_map = {}
    sector_B_map = {}
    sector_matrices = {}

    for q in sorted(sector_A_configs):
        A_sorted = sorted(sector_A_configs[q])
        B_sorted = sorted(sector_B_configs[q])
        sector_A_map[q] = {c: i for i, c in enumerate(A_sorted)}
        sector_B_map[q] = {c: i for i, c in enumerate(B_sorted)}
        sector_matrices[q] = np.zeros(
            (len(A_sorted), len(B_sorted)), dtype=wfn_dtype,
        )

    del sector_A_configs, sector_B_configs

    # Pass 2: fill matrices
    for k in range(dim):
        q = int(charges[k])
        occupation = basis.get_state(k)
        occ_A = tuple(occupation[s] for s in sub_sites)
        occ_B = tuple(occupation[s] for s in comp_sites)
        a = sector_A_map[q][occ_A]
        b = sector_B_map[q][occ_B]
        sector_matrices[q][a, b] = wavefunction[k]

    del sector_A_map, sector_B_map

    # SVD each sector
    sector_data = {}
    for q in sorted(sector_matrices):
        mat = sector_matrices[q]
        if mat.size == 0:
            continue
        singular_values = np.linalg.svd(mat, compute_uv=False)
        sq = singular_values ** 2
        valid = sq[sq > 1e-30]
        if len(valid) > 0:
            sector_data[q] = valid
        del mat

    sector_matrices.clear()
    return sector_data


def sweep_observables(wavefunction, basis, make_subsystem, l_max):
    """Compute F_A, S₁, S₂, S₂_acc for l=1..l_max in one decode pass.

    This is the optimized replacement for calling bipartite_fluctuations
    + entanglement_entropy(alpha=1) + entanglement_entropy(alpha=2)
    + accessible_entanglement_entropy separately at each l value.

    Optimizations over the naive approach:
        1. Decode each basis state ONCE (not 7× per l value).
        2. Compute sector SVD ONCE per l (not 3× for S₁, S₂, S₂_acc).

    Parameters
    ----------
    wavefunction : ndarray, shape (dim,)
        Ground state in the full basis.
    basis : FockBasis or UnaryBasis
    make_subsystem : callable
        make_subsystem(l) -> list of site indices for subsystem A at cut l.
    l_max : int
        Sweep l = 1, 2, ..., l_max.

    Returns
    -------
    sweep_data : list of dict
        One dict per l value with keys:
        'l', 'num_sites_A', 'F_A', 'S_1', 'S_2', 'S_2_acc',
        'sector_S_2' (dict {n_A: S₂(n_A)}),
        'sector_probs' (dict {n_A: p(n_A)}).
    """
    num_sites = basis.num_sites
    probabilities = np.abs(wavefunction) ** 2
    use_numba_path = _can_use_numba(basis)

    # =================================================================
    # Step 1: Precompute per-site occupations (one decode pass)
    #
    # For the Numba path, decode all states once and store the
    # occupation at each subsystem site. Charges for any l are
    # then just partial sums of these columns.
    # =================================================================

    # Build the subsystem site lists for each l upfront
    sub_sites_by_l = []
    for l in range(1, l_max + 1):
        sub_sites_by_l.append(make_subsystem(l))

    # Get the full site list at l_max
    all_sub_sites = sub_sites_by_l[-1]

    if use_numba_path:
        integers = basis._integers.astype(np.int64)
        L = np.int64(num_sites)

        # Check if subsystems are nested (each l is a prefix of l+1).
        # True for 1D (sites 0..l-1) and 2D/3D subregions where l
        # increases the subregion monotonically.
        is_nested = True
        for l_idx in range(len(sub_sites_by_l) - 1):
            if not set(sub_sites_by_l[l_idx]) <= set(sub_sites_by_l[l_idx + 1]):
                is_nested = False
                break

        if is_nested and len(all_sub_sites) > 0:
            # Build ordered site array: sites added as l increases
            site_order_list = list(sub_sites_by_l[0])  # sites at l=1
            for l_idx in range(1, len(sub_sites_by_l)):
                prev = set(sub_sites_by_l[l_idx - 1])
                for s in sub_sites_by_l[l_idx]:
                    if s not in prev:
                        site_order_list.append(s)

            site_order = np.array(site_order_list, dtype=np.int64)

            # One parallel pass: decode all states, store per-site occs
            occ_at_sites = _compute_site_occupations(integers, L, site_order)

            # Number of site_order columns to sum for each l
            num_cols_at_l = [len(sub_sites_by_l[l_idx])
                             for l_idx in range(len(sub_sites_by_l))]
        else:
            occ_at_sites = None
    else:
        occ_at_sites = None
        integers = None
        L = None

    # =================================================================
    # Step 2: Sweep over l values
    # =================================================================
    sweep_data = []

    for l_idx, l in enumerate(range(1, l_max + 1)):
        sub_sites = sub_sites_by_l[l_idx]
        comp_sites = [i for i in range(num_sites) if i not in sub_sites]
        entry = {'l': l, 'num_sites_A': len(sub_sites)}

        if len(sub_sites) == 0 or len(comp_sites) == 0:
            entry.update({'F_A': 0.0, 'S_1': 0.0,
                          'S_2': 0.0, 'S_2_acc': 0.0,
                          'sector_S_2': {}, 'sector_probs': {}})
            sweep_data.append(entry)
            continue

        # --- Bipartite fluctuations ---
        if use_numba_path and occ_at_sites is not None:
            # Use precomputed occupations (parallel, no re-decode)
            n_cols = num_cols_at_l[l_idx]
            entry['F_A'] = _fluctuations_from_occ(
                probabilities, occ_at_sites, np.int64(n_cols),
            )
        elif use_numba_path:
            sub_arr = np.array(sub_sites, dtype=np.int64)
            entry['F_A'] = _fluctuations_kernel(
                probabilities, integers, L, sub_arr,
            )
        else:
            entry['F_A'] = bipartite_fluctuations(
                wavefunction, basis, subsystem_sites=sub_sites,
            )

        # --- Charges for this l ---
        if use_numba_path and occ_at_sites is not None:
            # n_cols set in the fluctuations block above
            charges = _cumulative_charges(
                occ_at_sites, np.int64(n_cols),
            )
        elif use_numba_path:
            # Not nested: sub_arr set in fluctuations block above
            charges = _compute_charges(integers, L, sub_arr)
        else:
            # Pure Python fallback
            charges = np.empty(basis.dim, dtype=np.int32)
            for k in range(basis.dim):
                occ = basis.get_state(k)
                charges[k] = sum(occ[s] for s in sub_sites)

        # --- Sector SVD (ONCE per l) ---
        if use_numba_path:
            sector_spectra = _sector_schmidt_with_charges(
                wavefunction, basis, sub_sites, comp_sites, charges,
            )
        else:
            sector_spectra = _sector_schmidt_with_charges_python(
                wavefunction, basis, sub_sites, comp_sites, charges,
            )

        # --- Derive all entropies from single SVD result ---
        ent = _entropies_from_sector_spectra(sector_spectra)
        entry['S_1'] = ent['S_1']
        entry['S_2'] = ent['S_2']
        entry['S_2_acc'] = ent['S_2_acc']
        entry['sector_S_2'] = ent['sector_S_2']
        entry['sector_probs'] = ent['sector_probs']

        del charges, sector_spectra
        sweep_data.append(entry)

    # Clean up large precomputed array
    if occ_at_sites is not None:
        del occ_at_sites

    return sweep_data
