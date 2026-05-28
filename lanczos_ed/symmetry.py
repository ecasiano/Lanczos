"""
Symmetry Reduction for 1D Lattice Models (PBC)
===============================================

Provides translational and reflection symmetry operations as a
standalone, model-independent module. Any Hamiltonian that commutes
with the lattice symmetry group can use these tools to reduce the
Hilbert space dimension.

**Not all models have these symmetries.** Disordered models (e.g.,
Bose glass with random on-site potentials) break translational
invariance. This module is opt-in: the model decides whether to
use it based on its own symmetry properties.

Translational symmetry (cyclic shift):
    T |n_0, n_1, ..., n_{L-1}> = |n_1, n_2, ..., n_{L-1}, n_0>

Reflection symmetry (spatial inversion):
    R |n_0, n_1, ..., n_{L-1}> = |n_{L-1}, ..., n_1, n_0>

In the unary (balls-and-walls) encoding, both operations are
efficient bitwise manipulations:
    - Translation: strip LSB site, append at MSB
    - Reflection: reverse the inner bits, restore MSB wall

The q=0 (zero momentum), R=+1 (even parity) sector is constructed
by grouping basis states into symmetry orbits ("cycles"), choosing
one representative per orbit, and building the Hamiltonian with
normalization factors that account for orbit multiplicities.

Reference:
    Sandvik, AIP Conf. Proc. 1297, 135 (2010)
    Barghathi et al., Phys. Rev. B 105, L121116 (2022)
"""

import numpy as np
from scipy import sparse
from .unary_basis import (
    UnaryBasis, occupation_to_unary, unary_to_occupation,
    _count_trailing_zeros,
)

# Optional Numba acceleration
try:
    from numba import njit, prange
    HAS_NUMBA = True
except ImportError:
    HAS_NUMBA = False


# =====================================================================
# Numba-accelerated kernels (compiled on first call)
# =====================================================================

if HAS_NUMBA:

    @njit(cache=True)
    def _shift_v_numba(v, total_bits):
        """Numba-compiled cyclic left shift (translation)."""
        # Count trailing zeros = occupation of site 0
        if v == 0:
            n_0 = np.int64(64)
        else:
            n_0 = np.int64(0)
            tmp = v
            while (tmp & np.int64(1)) == np.int64(0) and tmp != np.int64(0):
                n_0 += np.int64(1)
                tmp >>= np.int64(1)
        v >>= (n_0 + np.int64(1))
        v |= (np.int64(1) << (total_bits - np.int64(1)))
        return v

    @njit(cache=True)
    def _reverse_ket_numba(v, total_bits):
        """Numba-compiled spatial reflection."""
        k = total_bits
        # Strip MSB wall bit
        v_inner = v ^ (np.int64(1) << (k - np.int64(1)))
        # Reverse the remaining (k-1) bits
        reversed_v = np.int64(0)
        for i in range(k - np.int64(1)):
            if v_inner & (np.int64(1) << i):
                reversed_v |= (np.int64(1) << (k - np.int64(2) - i))
        # Restore MSB wall
        return reversed_v | (np.int64(1) << (k - np.int64(1)))

    @njit(cache=True)
    def _binary_search_numba(arr, val):
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

    @njit(cache=True)
    def _unary_to_occ_sym(v, L, occ):
        """Decode unary integer into pre-allocated occupation array."""
        for s in range(L):
            n_i = np.int64(0)
            while (v & np.int64(1)) == np.int64(0) and v != np.int64(0):
                n_i += np.int64(1)
                v >>= np.int64(1)
            occ[s] = n_i
            v >>= np.int64(1)

    @njit(cache=True)
    def _occ_to_unary_sym(occ, L):
        """Encode occupation array as unary integer."""
        v = np.int64(0)
        bp = np.int64(0)
        for s in range(L):
            bp += np.int64(occ[s])
            v |= (np.int64(1) << bp)
            bp += np.int64(1)
        return v

    @njit(cache=True)
    def _find_cycles_numba(integers, total_bits, use_reflection):
        """Numba-compiled cycle enumeration.

        Returns flat arrays: cycle_leaders, cycle_sizes, state_to_cycle.
        """
        D = len(integers)
        state_to_cycle = np.full(D, np.int64(-1), dtype=np.int64)
        # Upper bound: every state could be its own cycle
        cycle_leaders = np.empty(D, dtype=np.int64)
        cycle_sizes = np.empty(D, dtype=np.int64)
        num_cycles = np.int64(0)

        for i in range(D):
            if state_to_cycle[i] >= np.int64(0):
                continue

            cycle_id = num_cycles
            cycle_leaders[num_cycles] = i
            member_count = np.int64(0)

            # Phase 1: translational orbit
            v = integers[i]
            while True:
                idx = _binary_search_numba(integers, v)
                if idx < np.int64(0) or state_to_cycle[idx] >= np.int64(0):
                    break
                state_to_cycle[idx] = cycle_id
                member_count += np.int64(1)
                v = _shift_v_numba(v, total_bits)

            # Phase 2: reflection orbit
            if use_reflection:
                v_ref = _reverse_ket_numba(integers[i], total_bits)
                idx_ref = _binary_search_numba(integers, v_ref)

                if idx_ref >= np.int64(0) and state_to_cycle[idx_ref] < np.int64(0):
                    v = v_ref
                    while True:
                        idx = _binary_search_numba(integers, v)
                        if idx < np.int64(0) or state_to_cycle[idx] >= np.int64(0):
                            break
                        state_to_cycle[idx] = cycle_id
                        member_count += np.int64(1)
                        v = _shift_v_numba(v, total_bits)

            cycle_sizes[num_cycles] = member_count
            num_cycles += np.int64(1)

        return cycle_leaders[:num_cycles], cycle_sizes[:num_cycles], num_cycles, state_to_cycle

    @njit(cache=True)
    def _build_reduced_H_numba(integers, L, n_max, cycle_leaders,
                                cycle_sizes, num_cycles, state_to_cycle,
                                hopping, interaction, chem_pot, bonds):
        """Numba-compiled symmetry-reduced Hamiltonian construction.

        Returns COO triple arrays (rows, cols, vals) for the upper triangle.
        """
        num_bonds = bonds.shape[0]

        # Upper bound on nnz: num_cycles (diagonal) + num_cycles * 2 * num_bonds
        max_nnz = num_cycles * (1 + 2 * num_bonds)
        rows = np.empty(max_nnz, dtype=np.int64)
        cols = np.empty(max_nnz, dtype=np.int64)
        vals = np.empty(max_nnz, dtype=np.float64)
        nnz = np.int64(0)

        occ = np.empty(L, dtype=np.int64)
        new_occ = np.empty(L, dtype=np.int64)

        for c in range(num_cycles):
            leader_idx = cycle_leaders[c]
            _unary_to_occ_sym(integers[leader_idx], L, occ)
            bra_size = np.float64(cycle_sizes[c])

            # --- Diagonal ---
            diag = 0.0
            for s in range(L):
                n_i = occ[s]
                diag += (interaction / 2.0) * n_i * (n_i - 1)
                diag -= chem_pot * n_i

            if diag != 0.0:
                rows[nnz] = c
                cols[nnz] = c
                vals[nnz] = diag
                nnz += np.int64(1)

            # --- Off-diagonal: hopping ---
            for b in range(num_bonds):
                si = bonds[b, 0]
                sj = bonds[b, 1]

                # b†_i b_j : hop from sj to si
                if occ[sj] > 0 and (n_max < 0 or occ[si] < n_max):
                    for s2 in range(L):
                        new_occ[s2] = occ[s2]
                    new_occ[sj] -= np.int64(1)
                    new_occ[si] += np.int64(1)
                    ket_v = _occ_to_unary_sym(new_occ, L)
                    ket_idx = _binary_search_numba(integers, ket_v)
                    if ket_idx >= np.int64(0):
                        target_cycle = state_to_cycle[ket_idx]
                        target_size = np.float64(cycle_sizes[target_cycle])
                        if c >= target_cycle:
                            factor = (-hopping
                                      * np.sqrt(bra_size / target_size)
                                      * np.sqrt(np.float64(occ[sj] * new_occ[si])))
                            rows[nnz] = target_cycle
                            cols[nnz] = c
                            vals[nnz] = factor
                            nnz += np.int64(1)

                # b†_j b_i : hop from si to sj
                if occ[si] > 0 and (n_max < 0 or occ[sj] < n_max):
                    for s2 in range(L):
                        new_occ[s2] = occ[s2]
                    new_occ[si] -= np.int64(1)
                    new_occ[sj] += np.int64(1)
                    ket_v = _occ_to_unary_sym(new_occ, L)
                    ket_idx = _binary_search_numba(integers, ket_v)
                    if ket_idx >= np.int64(0):
                        target_cycle = state_to_cycle[ket_idx]
                        target_size = np.float64(cycle_sizes[target_cycle])
                        if c >= target_cycle:
                            factor = (-hopping
                                      * np.sqrt(bra_size / target_size)
                                      * np.sqrt(np.float64(occ[si] * new_occ[sj])))
                            rows[nnz] = target_cycle
                            cols[nnz] = c
                            vals[nnz] = factor
                            nnz += np.int64(1)

        return rows[:nnz], cols[:nnz], vals[:nnz]


# =====================================================================
# Bitwise symmetry operations on unary-encoded integers
# =====================================================================

def shift_v(v: int, total_bits: int) -> int:
    """Cyclic left shift of the occupation pattern (translation).

    Maps  |n_0, n_1, ..., n_{L-1}>  to  |n_1, n_2, ..., n_{L-1}, n_0>.

    In the unary encoding, each site occupies (n_i zeros + 1 wall bit).
    Site 0 sits at the LSB end. Removing it and appending at the MSB
    end implements a cyclic rotation:

        1. n_0 = trailing_zeros(v)      (balls at site 0)
        2. v >>= (n_0 + 1)             (strip site 0)
        3. v |= (1 << (total_bits-1))  (new wall at MSB = old site 0)

    The gap between the old MSB wall and the new MSB wall is exactly
    n_0 zeros, encoding the moved site's occupation.

    Parameters
    ----------
    v : int
        Unary-encoded basis state.
    total_bits : int
        Bit-string length (L + N).

    Returns
    -------
    v_shifted : int
        State after one cyclic left shift.

    Example
    -------
    >>> # (2, 0, 1) on L=3, N=3 -> (0, 1, 2)
    >>> shift_v(44, 6)   # 0b101100 -> 0b110001 = 49
    49
    """
    n_0 = _count_trailing_zeros(v)
    v >>= (n_0 + 1)
    v |= (1 << (total_bits - 1))
    return v


def reverse_ket(v: int, total_bits: int) -> int:
    """Reflect the occupation pattern (spatial inversion).

    Maps  |n_0, n_1, ..., n_{L-1}>  to  |n_{L-1}, ..., n_1, n_0>.

    Algorithm:
        The MSB of a valid unary integer is always 1 (the last wall).
        We strip the MSB, reverse the remaining (total_bits - 1) bits,
        then restore the MSB. This reverses the site ordering.

    Parameters
    ----------
    v : int
        Unary-encoded basis state.
    total_bits : int
        Bit-string length (L + N).

    Returns
    -------
    v_reflected : int
        State after spatial reflection.

    Example
    -------
    >>> # (2, 0, 1) on L=3, N=3 -> (1, 0, 2)
    >>> reverse_ket(44, 6)   # 0b101100 -> 0b100110 = 38
    38
    """
    k = total_bits

    # Strip the MSB wall bit (always set for valid states)
    v_inner = v ^ (1 << (k - 1))

    # Reverse the remaining (k-1) bits
    reversed_v = 0
    for i in range(k - 1):
        if v_inner & (1 << i):
            reversed_v |= (1 << (k - 2 - i))

    # Restore MSB wall
    return reversed_v | (1 << (k - 1))


# =====================================================================
# Symmetry orbit (cycle) enumeration
# =====================================================================

def find_cycles(basis: UnaryBasis, use_reflection: bool = True):
    """Find symmetry orbits under translation (and optionally reflection).

    Each orbit is a set of basis states related by repeated application
    of the cyclic shift T (and optionally the reflection R). Within an
    orbit, all states contribute equally to the q=0, R=+1 sector, so
    only one representative is needed.

    The algorithm visits every basis state exactly once:
        1. Pick an unvisited state i -> new orbit with i as leader
        2. Apply T repeatedly until returning to a visited state
           (this traces out the translational orbit, period p)
        3. If using reflection: apply R to i, then T repeatedly
           (if R|i> is outside the translational orbit, it and its
            T-orbit join the same cycle, doubling it to size 2p)

    Parameters
    ----------
    basis : UnaryBasis
        The full (unsymmetrized) canonical basis.
    use_reflection : bool
        Whether to include reflection symmetry (default True).

    Returns
    -------
    cycle_leaders : numpy.ndarray of int64, shape (num_cycles,)
        Basis index of the representative state for each cycle.
    cycle_sizes : numpy.ndarray of int64, shape (num_cycles,)
        Number of states in each cycle (translational period,
        or 2× period if reflection doubles it).
    num_cycles : int
        Total number of cycles = dimension of the reduced basis.
    state_to_cycle : numpy.ndarray of int64, shape (dim,)
        Maps each full-basis index to its cycle index.
    """
    D = basis.dim
    total_bits = basis._bit_length  # L + N

    # -----------------------------------------------------------------
    # Numba path: JIT-compiled cycle enumeration
    # -----------------------------------------------------------------
    if HAS_NUMBA and hasattr(basis, '_integers'):
        integers = basis._integers.astype(np.int64)
        cycle_leaders, cycle_sizes, num_cycles, state_to_cycle = (
            _find_cycles_numba(integers, np.int64(total_bits), use_reflection)
        )
        num_cycles = int(num_cycles)

        # Sanity checks
        assert np.all(state_to_cycle >= 0), "Some states were not assigned to cycles"
        assert cycle_sizes.sum() == D, (
            f"Cycle sizes sum to {cycle_sizes.sum()}, expected {D}"
        )
        return cycle_leaders, cycle_sizes, num_cycles, state_to_cycle

    # -----------------------------------------------------------------
    # Python fallback
    # -----------------------------------------------------------------
    cycle_leaders_list = []
    cycle_sizes_list = []
    state_to_cycle = np.full(D, -1, dtype=np.int64)

    num_cycles = 0

    for i in range(D):
        if state_to_cycle[i] >= 0:
            continue  # already assigned to an orbit

        cycle_id = num_cycles
        cycle_leaders_list.append(i)
        member_count = 0

        # --- Phase 1: translational orbit of state i ---
        v = int(basis._integers[i])
        while True:
            idx = basis.get_index_from_integer(v)
            if idx < 0 or state_to_cycle[idx] >= 0:
                break
            state_to_cycle[idx] = cycle_id
            member_count += 1
            v = shift_v(v, total_bits)

        # --- Phase 2: reflection orbit (optional) ---
        if use_reflection:
            v_ref = reverse_ket(int(basis._integers[i]), total_bits)
            idx_ref = basis.get_index_from_integer(v_ref)

            if idx_ref >= 0 and state_to_cycle[idx_ref] < 0:
                # Reflected state is new — extend the cycle
                v = v_ref
                while True:
                    idx = basis.get_index_from_integer(v)
                    if idx < 0 or state_to_cycle[idx] >= 0:
                        break
                    state_to_cycle[idx] = cycle_id
                    member_count += 1
                    v = shift_v(v, total_bits)

        cycle_sizes_list.append(member_count)
        num_cycles += 1

    cycle_leaders = np.array(cycle_leaders_list, dtype=np.int64)
    cycle_sizes = np.array(cycle_sizes_list, dtype=np.int64)

    # Sanity check: every state should be assigned
    assert np.all(state_to_cycle >= 0), "Some states were not assigned to cycles"
    assert cycle_sizes.sum() == D, (
        f"Cycle sizes sum to {cycle_sizes.sum()}, expected {D}"
    )

    return cycle_leaders, cycle_sizes, num_cycles, state_to_cycle


# =====================================================================
# Symmetry-reduced Hamiltonian construction
# =====================================================================

def build_reduced_hamiltonian(
    basis: UnaryBasis,
    cycle_leaders: np.ndarray,
    cycle_sizes: np.ndarray,
    num_cycles: int,
    state_to_cycle: np.ndarray,
    hopping: float,
    interaction: float,
    chemical_potential: float,
    neighbor_pairs: list,
) -> sparse.csr_matrix:
    """Build the Hamiltonian in the q=0, R=+1 symmetry-reduced basis.

    For each cycle α with representative r_α and size |C_α|:

        Diagonal:
            H_αα = (U/2) sum_i n_i(n_i - 1) - μ sum_i n_i

        Off-diagonal (hopping):
            H_βα = sqrt(|C_α| / |C_β|) × (-t) × sqrt(n_source × n_new_target)

    where β is the cycle containing the state reached by hopping
    from r_α. The sqrt(|C_α|/|C_β|) normalization factor arises from
    the symmetrized basis vectors:

        |α> = (1/sqrt(|C_α|))  sum_{g in orbit α}  g|r_α>

    Only the upper triangle (α ≥ β) is stored; the matrix is then
    symmetrized. Multiple hops landing in the same target cycle
    accumulate via the sparse constructor.

    Parameters
    ----------
    basis : UnaryBasis
        Full (unsymmetrized) basis.
    cycle_leaders, cycle_sizes, num_cycles, state_to_cycle :
        Output from find_cycles().
    hopping : float
        Hopping amplitude t.
    interaction : float
        On-site interaction U.
    chemical_potential : float
        Chemical potential μ.
    neighbor_pairs : list of (int, int)
        Nearest-neighbor bond pairs.

    Returns
    -------
    H_reduced : scipy.sparse.csr_matrix of shape (num_cycles, num_cycles)
    """
    # -----------------------------------------------------------------
    # Numba path: JIT-compiled H construction returning COO triples
    # -----------------------------------------------------------------
    if HAS_NUMBA and hasattr(basis, '_integers'):
        integers = basis._integers.astype(np.int64)
        L = np.int64(basis.num_sites)
        # n_max = -1 signals "unrestricted" to the Numba kernel
        n_max_numba = np.int64(
            basis.max_occupation
            if basis.max_occupation != basis.total_particles
            else -1
        )
        bonds = np.array(neighbor_pairs, dtype=np.int64)

        rows, cols, vals = _build_reduced_H_numba(
            integers, L, n_max_numba, cycle_leaders, cycle_sizes,
            num_cycles, state_to_cycle,
            float(hopping), float(interaction), float(chemical_potential),
            bonds,
        )

        H_upper = sparse.csr_matrix(
            (vals, (rows, cols)),
            shape=(num_cycles, num_cycles),
            dtype=np.float64,
        )
        H_reduced = H_upper + H_upper.T - sparse.diags(H_upper.diagonal())
        H_reduced.eliminate_zeros()
        return H_reduced

    # -----------------------------------------------------------------
    # Python fallback
    # -----------------------------------------------------------------
    rows = []
    cols = []
    elements = []

    for cycle_id in range(num_cycles):
        leader_idx = cycle_leaders[cycle_id]
        occupation = basis.get_state(leader_idx)
        bra_cycle_size = float(cycle_sizes[cycle_id])

        # =============================================================
        # Diagonal: on-site interaction + chemical potential
        #
        #   H_diag = (U/2) sum_i n_i(n_i - 1) - mu sum_i n_i
        #
        # No normalization needed for diagonal elements.
        # =============================================================
        diagonal_energy = 0.0
        for site in range(basis.num_sites):
            n_i = occupation[site]
            diagonal_energy += (interaction / 2.0) * n_i * (n_i - 1)
            diagonal_energy -= chemical_potential * n_i

        if diagonal_energy != 0.0:
            rows.append(cycle_id)
            cols.append(cycle_id)
            elements.append(diagonal_energy)

        # =============================================================
        # Off-diagonal: hopping with symmetry normalization
        #
        # For each bond (site_i, site_j), apply both hop directions:
        #   b†_i b_j : hop from j to i
        #   b†_j b_i : hop from i to j
        #
        # Matrix element includes:
        #   - Bosonic factor:  -t × sqrt(n_source × n_new_target)
        #   - Symmetry factor: sqrt(|C_bra| / |C_target|)
        #
        # Only store upper triangle (cycle_id >= target_cycle).
        # =============================================================
        for site_i, site_j in neighbor_pairs:

            # --- b†_i b_j : hop a boson from site_j to site_i ---
            n_source = occupation[site_j]
            n_target = occupation[site_i]

            if n_source > 0 and (basis.max_occupation is None
                                  or n_target < basis.max_occupation):
                new_occ = list(occupation)
                new_occ[site_j] -= 1
                new_occ[site_i] += 1

                ket_code = occupation_to_unary(tuple(new_occ))
                ket_idx = basis.get_index_from_integer(ket_code)

                if ket_idx >= 0:
                    target_cycle = state_to_cycle[ket_idx]
                    target_cycle_size = float(cycle_sizes[target_cycle])

                    # Upper triangle: cycle_id >= target_cycle
                    if cycle_id >= target_cycle:
                        factor = (
                            -hopping
                            * np.sqrt(bra_cycle_size / target_cycle_size)
                            * np.sqrt(n_source * new_occ[site_i])
                        )
                        rows.append(target_cycle)
                        cols.append(cycle_id)
                        elements.append(factor)

            # --- b†_j b_i : hop a boson from site_i to site_j ---
            n_source = occupation[site_i]
            n_target = occupation[site_j]

            if n_source > 0 and (basis.max_occupation is None
                                  or n_target < basis.max_occupation):
                new_occ = list(occupation)
                new_occ[site_i] -= 1
                new_occ[site_j] += 1

                ket_code = occupation_to_unary(tuple(new_occ))
                ket_idx = basis.get_index_from_integer(ket_code)

                if ket_idx >= 0:
                    target_cycle = state_to_cycle[ket_idx]
                    target_cycle_size = float(cycle_sizes[target_cycle])

                    if cycle_id >= target_cycle:
                        factor = (
                            -hopping
                            * np.sqrt(bra_cycle_size / target_cycle_size)
                            * np.sqrt(n_source * new_occ[site_j])
                        )
                        rows.append(target_cycle)
                        cols.append(cycle_id)
                        elements.append(factor)

    # Assemble upper triangle (sparse sums duplicate entries)
    H_upper = sparse.csr_matrix(
        (elements, (rows, cols)),
        shape=(num_cycles, num_cycles),
        dtype=np.float64,
    )

    # Symmetrize:  H = H_upper + H_upper^T - diag(H_upper)
    # The transpose copies the upper triangle to the lower triangle;
    # we subtract the diagonal to avoid double-counting it.
    H_reduced = H_upper + H_upper.T - sparse.diags(H_upper.diagonal())
    H_reduced.eliminate_zeros()

    return H_reduced


# =====================================================================
# Wavefunction reconstruction
# =====================================================================

def reconstruct_wavefunction(
    d: np.ndarray,
    basis: UnaryBasis,
    cycle_sizes: np.ndarray,
    state_to_cycle: np.ndarray,
) -> np.ndarray:
    """Reconstruct the full wavefunction from a symmetry-reduced eigenvector.

    The symmetrized basis vector for cycle α is:

        |α> = (1/sqrt(|C_α|))  sum_{k in orbit α}  |k>

    So the full wavefunction ψ = sum_α d_α |α> has amplitudes:

        ψ[k] = d[α(k)] / sqrt(|C_{α(k)}|)

    where α(k) is the cycle containing state k.

    Parameters
    ----------
    d : numpy.ndarray, shape (num_cycles,)
        Eigenvector in the symmetry-reduced basis.
    basis : UnaryBasis
        Full (unsymmetrized) basis.
    cycle_sizes : numpy.ndarray
        Size of each cycle.
    state_to_cycle : numpy.ndarray
        Maps each basis index to its cycle index.

    Returns
    -------
    psi : numpy.ndarray, shape (basis.dim,)
        Full wavefunction in the original basis.
    """
    psi = np.empty(basis.dim, dtype=np.float64)

    for k in range(basis.dim):
        cycle = state_to_cycle[k]
        psi[k] = d[cycle] / np.sqrt(cycle_sizes[cycle])

    return psi
