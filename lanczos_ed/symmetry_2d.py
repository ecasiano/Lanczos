"""
2D Translational Symmetry Reduction for Square-Lattice Models (PBC)
===================================================================

Provides translational symmetry operations on an L × L square lattice
with periodic boundary conditions (torus topology). Any Hamiltonian
that commutes with the 2D translation group can use these tools to
reduce the Hilbert space dimension by a factor of up to L².

Translational symmetry generators:

    Tx |..., n_{x,y}, ...> = |..., n_{x-1 mod L, y}, ...>
    Ty |..., n_{x,y}, ...> = |..., n_{x, y-1 mod L}, ...>

These generate a group of L² translations: {Tx^a · Ty^b : 0 ≤ a,b < L}.

In the (kx, ky) = (0, 0) momentum sector (zero total momentum), the
symmetrized basis states are:

    |α> = (1/sqrt(|O_α|)) sum_{g in orbit α} g|r_α>

where |O_α| is the orbit size and r_α is the orbit representative.
All phase factors are +1 in this sector, so the reduced Hamiltonian
is real symmetric — no complex arithmetic needed.

For general momentum sectors (kx, ky) ≠ (0, 0), the phase factors
exp(-i(kx·a + ky·b)) are complex, and the Hamiltonian becomes
complex Hermitian. This module supports both cases.

Site indexing convention (matching BoseHubbard2D and pigsfli2):
    site = x + y * L
    x = site % L      (column, fast index)
    y = site // L      (row, slow index)

Translation operations:
    Tx: (x, y) → ((x+1) mod L, y)  — cyclic shift within each row
    Ty: (x, y) → (x, (y+1) mod L)  — cyclic shift of entire rows

Unlike the 1D case where translation is a fast bitwise shift on the
unary integer, 2D translations require:
    1. Decode unary integer → occupation array
    2. Permute the occupation array (row-wise for Tx, row-swap for Ty)
    3. Re-encode occupation array → unary integer

Reference:
    Sandvik, AIP Conf. Proc. 1297, 135 (2010) — general ED techniques
    Läuchli, "Exact Diagonalization" in Springer LNCS (2011)
"""

import numpy as np
from scipy import sparse
from .unary_basis import (
    UnaryBasis, occupation_to_unary, unary_to_occupation,
)

# Optional Numba acceleration
try:
    from numba import njit, prange
    HAS_NUMBA = True
except ImportError:
    HAS_NUMBA = False


# =====================================================================
# Translation operations on occupation arrays
# =====================================================================

def translate_x_occ(occupation: tuple, L: int) -> tuple:
    """Apply Tx: cyclic shift by +1 in x-direction.

    Maps  n_{x, y}  →  n_{x-1 mod L, y}  for all sites.
    Equivalently, the boson at site (x, y) moves to ((x+1) mod L, y).

    In terms of occupation arrays indexed by site = x + y*L:
    for each row y, cyclically shift the L entries by 1 to the right.

        row_y = [n(0,y), n(1,y), ..., n(L-1,y)]
        →       [n(L-1,y), n(0,y), ..., n(L-2,y)]

    Parameters
    ----------
    occupation : tuple of int, length L*L
        Occupation numbers indexed by site = x + y*L.
    L : int
        Linear lattice size.

    Returns
    -------
    new_occupation : tuple of int
        Occupation after applying Tx.
    """
    occ = list(occupation)
    new_occ = [0] * len(occ)
    for y in range(L):
        for x in range(L):
            # Site (x, y) gets the value from site ((x-1) mod L, y)
            src_x = (x - 1) % L
            new_occ[x + y * L] = occ[src_x + y * L]
    return tuple(new_occ)


def translate_y_occ(occupation: tuple, L: int) -> tuple:
    """Apply Ty: cyclic shift by +1 in y-direction.

    Maps  n_{x, y}  →  n_{x, y-1 mod L}  for all sites.
    Equivalently, the boson at site (x, y) moves to (x, (y+1) mod L).

    In terms of occupation arrays: permute entire rows cyclically.

        [row_0, row_1, ..., row_{L-1}]
        → [row_{L-1}, row_0, ..., row_{L-2}]

    Parameters
    ----------
    occupation : tuple of int, length L*L
        Occupation numbers indexed by site = x + y*L.
    L : int
        Linear lattice size.

    Returns
    -------
    new_occupation : tuple of int
        Occupation after applying Ty.
    """
    occ = list(occupation)
    new_occ = [0] * len(occ)
    for y in range(L):
        for x in range(L):
            src_y = (y - 1) % L
            new_occ[x + y * L] = occ[x + src_y * L]
    return tuple(new_occ)


def translate_x_integer(v: int, L: int, num_sites: int) -> int:
    """Apply Tx to a unary-encoded integer.

    Decode → permute → re-encode.
    """
    occ = unary_to_occupation(v, num_sites)
    new_occ = translate_x_occ(occ, L)
    return occupation_to_unary(new_occ)


def translate_y_integer(v: int, L: int, num_sites: int) -> int:
    """Apply Ty to a unary-encoded integer.

    Decode → permute → re-encode.
    """
    occ = unary_to_occupation(v, num_sites)
    new_occ = translate_y_occ(occ, L)
    return occupation_to_unary(new_occ)


# =====================================================================
# Numba-accelerated kernels
# =====================================================================

if HAS_NUMBA:

    @njit(cache=True)
    def _unary_to_occ_2d(v, num_sites, occ):
        """Decode unary integer into pre-allocated occupation array."""
        for s in range(num_sites):
            n_i = np.int64(0)
            while (v & np.int64(1)) == np.int64(0) and v != np.int64(0):
                n_i += np.int64(1)
                v >>= np.int64(1)
            occ[s] = n_i
            v >>= np.int64(1)

    @njit(cache=True)
    def _occ_to_unary_2d(occ, num_sites):
        """Encode occupation array as unary integer."""
        v = np.int64(0)
        bp = np.int64(0)
        for s in range(num_sites):
            bp += np.int64(occ[s])
            v |= (np.int64(1) << bp)
            bp += np.int64(1)
        return v

    @njit(cache=True)
    def _translate_x_numba(occ, new_occ, L, num_sites):
        """Apply Tx to occupation array (in-place into new_occ)."""
        for y in range(L):
            for x in range(L):
                src_x = (x - np.int64(1)) % L
                new_occ[x + y * L] = occ[src_x + y * L]

    @njit(cache=True)
    def _translate_y_numba(occ, new_occ, L, num_sites):
        """Apply Ty to occupation array (in-place into new_occ)."""
        for y in range(L):
            for x in range(L):
                src_y = (y - np.int64(1)) % L
                new_occ[x + y * L] = occ[x + src_y * L]

    @njit(cache=True)
    def _binary_search_2d(arr, val):
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
    def _find_orbits_2d_numba(integers, L, num_sites):
        """Numba-compiled 2D orbit enumeration.

        Groups all basis states into orbits under {Tx^a · Ty^b}.
        For each state, records:
          - orbit_id: which orbit it belongs to
          - (tx, ty): translation from representative to this state

        Returns
        -------
        orbit_leaders : array of orbit representative indices
        orbit_sizes : array of orbit sizes
        num_orbits : number of orbits
        state_to_orbit : array mapping state index → orbit index
        state_to_tx : array mapping state index → tx shift from leader
        state_to_ty : array mapping state index → ty shift from leader
        """
        D = len(integers)
        state_to_orbit = np.full(D, np.int64(-1), dtype=np.int64)
        state_to_tx = np.zeros(D, dtype=np.int64)
        state_to_ty = np.zeros(D, dtype=np.int64)

        orbit_leaders = np.empty(D, dtype=np.int64)
        orbit_sizes = np.empty(D, dtype=np.int64)
        num_orbits = np.int64(0)

        occ = np.empty(num_sites, dtype=np.int64)
        shifted_occ = np.empty(num_sites, dtype=np.int64)
        tmp_occ = np.empty(num_sites, dtype=np.int64)

        for i in range(D):
            if state_to_orbit[i] >= np.int64(0):
                continue

            orbit_id = num_orbits
            orbit_leaders[num_orbits] = i
            member_count = np.int64(0)

            # Decode the representative state
            _unary_to_occ_2d(integers[i], num_sites, occ)

            # Apply all L^2 translations: Tx^a · Ty^b
            # Start with occ (ty=0), then apply Ty repeatedly
            for s2 in range(num_sites):
                shifted_occ[s2] = occ[s2]

            for ty in range(L):
                # Now apply Tx^a for a = 0..L-1
                for s2 in range(num_sites):
                    tmp_occ[s2] = shifted_occ[s2]

                for tx in range(L):
                    # tmp_occ is now Tx^tx · Ty^ty |r>
                    v_translated = _occ_to_unary_2d(tmp_occ, num_sites)
                    idx = _binary_search_2d(integers, v_translated)

                    if idx >= np.int64(0) and state_to_orbit[idx] < np.int64(0):
                        state_to_orbit[idx] = orbit_id
                        state_to_tx[idx] = tx
                        state_to_ty[idx] = ty
                        member_count += np.int64(1)

                    # Apply one more Tx (for next iteration)
                    # tmp_occ -> Tx(tmp_occ)
                    _translate_x_numba(tmp_occ, occ, L, num_sites)
                    # Copy result back to tmp_occ (reuse occ as temp)
                    for s2 in range(num_sites):
                        tmp_occ[s2] = occ[s2]

                # Apply one more Ty to shifted_occ for next ty
                _translate_y_numba(shifted_occ, occ, L, num_sites)
                for s2 in range(num_sites):
                    shifted_occ[s2] = occ[s2]

            orbit_sizes[num_orbits] = member_count
            num_orbits += np.int64(1)

        return (orbit_leaders[:num_orbits], orbit_sizes[:num_orbits],
                num_orbits, state_to_orbit, state_to_tx, state_to_ty)

    @njit(cache=True)
    def _build_reduced_H_2d_numba(integers, L, num_sites, n_max,
                                   orbit_leaders, orbit_sizes, num_orbits,
                                   state_to_orbit, state_to_tx, state_to_ty,
                                   hopping, interaction, chem_pot, bonds,
                                   kx, ky):
        """Numba-compiled 2D symmetry-reduced Hamiltonian construction.

        For (kx, ky) = (0, 0), all phases are 1.0 and the result is
        real symmetric (imaginary parts are zero).

        For general (kx, ky), the Hamiltonian is complex Hermitian.
        Phase factor for hopping from orbit α to orbit β:
            exp(-i * (kx * tx + ky * ty))
        where (tx, ty) is the translation mapping r_β → hopped state.

        Returns COO triple arrays (rows, cols, vals_re, vals_im).
        """
        num_bonds = bonds.shape[0]
        max_nnz = num_orbits * (1 + 2 * num_bonds)
        rows = np.empty(max_nnz, dtype=np.int64)
        cols = np.empty(max_nnz, dtype=np.int64)
        vals_re = np.empty(max_nnz, dtype=np.float64)
        vals_im = np.empty(max_nnz, dtype=np.float64)
        nnz = np.int64(0)

        occ = np.empty(num_sites, dtype=np.int64)
        new_occ = np.empty(num_sites, dtype=np.int64)

        is_zero_momentum = (kx == 0.0 and ky == 0.0)

        for c in range(num_orbits):
            leader_idx = orbit_leaders[c]
            _unary_to_occ_2d(integers[leader_idx], num_sites, occ)
            bra_size = np.float64(orbit_sizes[c])

            # --- Diagonal: (U/2) sum n_i(n_i-1) - mu sum n_i ---
            diag = 0.0
            for s in range(num_sites):
                n_i = occ[s]
                diag += (interaction / 2.0) * n_i * (n_i - 1)
                diag -= chem_pot * n_i

            if diag != 0.0:
                rows[nnz] = c
                cols[nnz] = c
                vals_re[nnz] = diag
                vals_im[nnz] = 0.0
                nnz += np.int64(1)

            # --- Off-diagonal: hopping ---
            for b in range(num_bonds):
                si = bonds[b, 0]
                sj = bonds[b, 1]

                # b†_i b_j : hop from sj to si
                if occ[sj] > 0 and (n_max < 0 or occ[si] < n_max):
                    for s2 in range(num_sites):
                        new_occ[s2] = occ[s2]
                    new_occ[sj] -= np.int64(1)
                    new_occ[si] += np.int64(1)
                    ket_v = _occ_to_unary_2d(new_occ, num_sites)
                    ket_idx = _binary_search_2d(integers, ket_v)
                    if ket_idx >= np.int64(0):
                        target_orbit = state_to_orbit[ket_idx]
                        target_size = np.float64(orbit_sizes[target_orbit])
                        amplitude = (-hopping
                                     * np.sqrt(bra_size / target_size)
                                     * np.sqrt(np.float64(occ[sj])
                                               * np.float64(new_occ[si])))

                        if is_zero_momentum:
                            # Real symmetric: store upper triangle
                            if c >= target_orbit:
                                rows[nnz] = target_orbit
                                cols[nnz] = c
                                vals_re[nnz] = amplitude
                                vals_im[nnz] = 0.0
                                nnz += np.int64(1)
                        else:
                            # Complex Hermitian: include phase factor
                            tx = state_to_tx[ket_idx]
                            ty = state_to_ty[ket_idx]
                            phase_angle = -(kx * tx + ky * ty)
                            phase_re = np.cos(phase_angle)
                            phase_im = np.sin(phase_angle)
                            rows[nnz] = target_orbit
                            cols[nnz] = c
                            vals_re[nnz] = amplitude * phase_re
                            vals_im[nnz] = amplitude * phase_im
                            nnz += np.int64(1)

                # b†_j b_i : hop from si to sj
                if occ[si] > 0 and (n_max < 0 or occ[sj] < n_max):
                    for s2 in range(num_sites):
                        new_occ[s2] = occ[s2]
                    new_occ[si] -= np.int64(1)
                    new_occ[sj] += np.int64(1)
                    ket_v = _occ_to_unary_2d(new_occ, num_sites)
                    ket_idx = _binary_search_2d(integers, ket_v)
                    if ket_idx >= np.int64(0):
                        target_orbit = state_to_orbit[ket_idx]
                        target_size = np.float64(orbit_sizes[target_orbit])
                        amplitude = (-hopping
                                     * np.sqrt(bra_size / target_size)
                                     * np.sqrt(np.float64(occ[si])
                                               * np.float64(new_occ[sj])))

                        if is_zero_momentum:
                            if c >= target_orbit:
                                rows[nnz] = target_orbit
                                cols[nnz] = c
                                vals_re[nnz] = amplitude
                                vals_im[nnz] = 0.0
                                nnz += np.int64(1)
                        else:
                            tx = state_to_tx[ket_idx]
                            ty = state_to_ty[ket_idx]
                            phase_angle = -(kx * tx + ky * ty)
                            phase_re = np.cos(phase_angle)
                            phase_im = np.sin(phase_angle)
                            rows[nnz] = target_orbit
                            cols[nnz] = c
                            vals_re[nnz] = amplitude * phase_re
                            vals_im[nnz] = amplitude * phase_im
                            nnz += np.int64(1)

        return rows[:nnz], cols[:nnz], vals_re[:nnz], vals_im[:nnz]


# =====================================================================
# Orbit (cycle) enumeration for 2D lattice
# =====================================================================

def find_orbits_2d(basis: UnaryBasis, L: int):
    """Find symmetry orbits under the 2D translation group.

    Groups all basis states into orbits under the L² translations
    {Tx^a · Ty^b : 0 ≤ a, b < L}. Each orbit gets one representative
    (the first encountered in basis order = the lexicographically
    smallest unary integer in the orbit).

    For each state, tracks the translation (tx, ty) that maps the
    orbit representative to that state:
        |state> = Tx^tx · Ty^ty |representative>

    This translation information is needed for phase factors in
    general momentum sectors.

    Parameters
    ----------
    basis : UnaryBasis
        The full (unsymmetrized) canonical basis.
    L : int
        Linear lattice size (L × L lattice).

    Returns
    -------
    orbit_leaders : numpy.ndarray of int64, shape (num_orbits,)
        Basis index of the representative for each orbit.
    orbit_sizes : numpy.ndarray of int64, shape (num_orbits,)
        Number of states in each orbit (divides L²).
    num_orbits : int
        Total number of orbits = dimension of the reduced basis.
    state_to_orbit : numpy.ndarray of int64, shape (dim,)
        Maps each full-basis index to its orbit index.
    state_to_tx : numpy.ndarray of int64, shape (dim,)
        x-translation from representative to this state.
    state_to_ty : numpy.ndarray of int64, shape (dim,)
        y-translation from representative to this state.
    """
    num_sites = L * L

    # -----------------------------------------------------------------
    # Numba path
    # -----------------------------------------------------------------
    if HAS_NUMBA and hasattr(basis, '_integers'):
        integers = basis._integers.astype(np.int64)
        result = _find_orbits_2d_numba(
            integers, np.int64(L), np.int64(num_sites)
        )
        orbit_leaders, orbit_sizes, num_orbits = result[0], result[1], int(result[2])
        state_to_orbit, state_to_tx, state_to_ty = result[3], result[4], result[5]

        # Sanity checks
        assert np.all(state_to_orbit >= 0), (
            "Some states were not assigned to orbits"
        )
        assert orbit_sizes.sum() == basis.dim, (
            f"Orbit sizes sum to {orbit_sizes.sum()}, expected {basis.dim}"
        )
        return (orbit_leaders, orbit_sizes, num_orbits,
                state_to_orbit, state_to_tx, state_to_ty)

    # -----------------------------------------------------------------
    # Python fallback
    # -----------------------------------------------------------------
    D = basis.dim
    state_to_orbit = np.full(D, -1, dtype=np.int64)
    state_to_tx = np.zeros(D, dtype=np.int64)
    state_to_ty = np.zeros(D, dtype=np.int64)

    orbit_leaders_list = []
    orbit_sizes_list = []
    num_orbits = 0

    for i in range(D):
        if state_to_orbit[i] >= 0:
            continue

        orbit_id = num_orbits
        orbit_leaders_list.append(i)
        member_count = 0

        # Get the representative occupation
        rep_occ = basis.get_state(i)

        # Apply all L² translations: Tx^tx · Ty^ty
        shifted_occ = rep_occ
        for ty in range(L):
            current_occ = shifted_occ
            for tx in range(L):
                # current_occ = Tx^tx · Ty^ty |rep>
                v = occupation_to_unary(current_occ)
                idx = basis.get_index_from_integer(v)

                if idx >= 0 and state_to_orbit[idx] < 0:
                    state_to_orbit[idx] = orbit_id
                    state_to_tx[idx] = tx
                    state_to_ty[idx] = ty
                    member_count += 1

                # Apply one more Tx
                current_occ = translate_x_occ(current_occ, L)

            # Apply one more Ty to the base (ty row)
            shifted_occ = translate_y_occ(shifted_occ, L)

        orbit_sizes_list.append(member_count)
        num_orbits += 1

    orbit_leaders = np.array(orbit_leaders_list, dtype=np.int64)
    orbit_sizes = np.array(orbit_sizes_list, dtype=np.int64)

    assert np.all(state_to_orbit >= 0), (
        "Some states were not assigned to orbits"
    )
    assert orbit_sizes.sum() == D, (
        f"Orbit sizes sum to {orbit_sizes.sum()}, expected {D}"
    )

    return (orbit_leaders, orbit_sizes, num_orbits,
            state_to_orbit, state_to_tx, state_to_ty)


# =====================================================================
# Symmetry-reduced Hamiltonian construction
# =====================================================================

def build_reduced_hamiltonian_2d(
    basis: UnaryBasis,
    L: int,
    orbit_leaders: np.ndarray,
    orbit_sizes: np.ndarray,
    num_orbits: int,
    state_to_orbit: np.ndarray,
    state_to_tx: np.ndarray,
    state_to_ty: np.ndarray,
    hopping: float,
    interaction: float,
    chemical_potential: float,
    neighbor_pairs: list,
    kx: float = 0.0,
    ky: float = 0.0,
) -> sparse.csr_matrix:
    """Build the Hamiltonian in a 2D momentum sector.

    For (kx, ky) = (0, 0):
        H is real symmetric, identical structure to the 1D q=0 case.
        The normalization factor is sqrt(|O_α| / |O_β|).

    For general (kx, ky):
        H is complex Hermitian. Off-diagonal elements include phase:
            exp(-i(kx·tx + ky·ty))
        where (tx, ty) maps r_β to the hopped state.

    Parameters
    ----------
    basis : UnaryBasis
        Full (unsymmetrized) basis.
    L : int
        Linear lattice size.
    orbit_leaders, orbit_sizes, num_orbits, state_to_orbit,
    state_to_tx, state_to_ty :
        Output from find_orbits_2d().
    hopping, interaction, chemical_potential : float
        Hamiltonian parameters.
    neighbor_pairs : list of (int, int)
        Nearest-neighbor bond pairs.
    kx, ky : float
        Momentum sector. Default (0, 0) for the ground state sector.

    Returns
    -------
    H_reduced : scipy.sparse.csr_matrix
        Shape (num_orbits, num_orbits). Real for (0,0), complex otherwise.
    """
    is_zero_momentum = (kx == 0.0 and ky == 0.0)
    num_sites = L * L

    # -----------------------------------------------------------------
    # Numba path
    # -----------------------------------------------------------------
    if HAS_NUMBA and hasattr(basis, '_integers'):
        integers = basis._integers.astype(np.int64)
        n_max_numba = np.int64(
            basis.max_occupation
            if basis.max_occupation != basis.total_particles
            else -1
        )
        bonds = np.array(neighbor_pairs, dtype=np.int64)

        rows, cols, vals_re, vals_im = _build_reduced_H_2d_numba(
            integers, np.int64(L), np.int64(num_sites), n_max_numba,
            orbit_leaders, orbit_sizes, np.int64(num_orbits),
            state_to_orbit, state_to_tx, state_to_ty,
            float(hopping), float(interaction), float(chemical_potential),
            bonds, float(kx), float(ky),
        )

        if is_zero_momentum:
            H_upper = sparse.csr_matrix(
                (vals_re, (rows, cols)),
                shape=(num_orbits, num_orbits),
                dtype=np.float64,
            )
            H_reduced = H_upper + H_upper.T - sparse.diags(H_upper.diagonal())
        else:
            vals_complex = vals_re + 1j * vals_im
            H_raw = sparse.csr_matrix(
                (vals_complex, (rows, cols)),
                shape=(num_orbits, num_orbits),
                dtype=np.complex128,
            )
            # Hermitize: H = (H + H†) / 2
            H_reduced = (H_raw + H_raw.conj().T) / 2.0

        H_reduced.eliminate_zeros()
        return H_reduced

    # -----------------------------------------------------------------
    # Python fallback
    # -----------------------------------------------------------------
    rows = []
    cols = []
    elements = []

    for orbit_id in range(num_orbits):
        leader_idx = orbit_leaders[orbit_id]
        occupation = basis.get_state(leader_idx)
        bra_orbit_size = float(orbit_sizes[orbit_id])

        # --- Diagonal ---
        diagonal_energy = 0.0
        for site in range(num_sites):
            n_i = occupation[site]
            diagonal_energy += (interaction / 2.0) * n_i * (n_i - 1)
            diagonal_energy -= chemical_potential * n_i

        if diagonal_energy != 0.0:
            rows.append(orbit_id)
            cols.append(orbit_id)
            elements.append(diagonal_energy)

        # --- Off-diagonal: hopping ---
        for site_i, site_j in neighbor_pairs:

            # b†_i b_j : hop from site_j to site_i
            n_source = occupation[site_j]
            n_target = occupation[site_i]

            if (n_source > 0
                    and (basis.max_occupation is None
                         or n_target < basis.max_occupation)):
                new_occ = list(occupation)
                new_occ[site_j] -= 1
                new_occ[site_i] += 1

                ket_code = occupation_to_unary(tuple(new_occ))
                ket_idx = basis.get_index_from_integer(ket_code)

                if ket_idx >= 0:
                    target_orbit = state_to_orbit[ket_idx]
                    target_size = float(orbit_sizes[target_orbit])
                    amplitude = (
                        -hopping
                        * np.sqrt(bra_orbit_size / target_size)
                        * np.sqrt(n_source * new_occ[site_i])
                    )

                    if is_zero_momentum:
                        if orbit_id >= target_orbit:
                            rows.append(target_orbit)
                            cols.append(orbit_id)
                            elements.append(amplitude)
                    else:
                        tx = state_to_tx[ket_idx]
                        ty = state_to_ty[ket_idx]
                        phase = np.exp(-1j * (kx * tx + ky * ty))
                        rows.append(target_orbit)
                        cols.append(orbit_id)
                        elements.append(amplitude * phase)

            # b†_j b_i : hop from site_i to site_j
            n_source = occupation[site_i]
            n_target = occupation[site_j]

            if (n_source > 0
                    and (basis.max_occupation is None
                         or n_target < basis.max_occupation)):
                new_occ = list(occupation)
                new_occ[site_i] -= 1
                new_occ[site_j] += 1

                ket_code = occupation_to_unary(tuple(new_occ))
                ket_idx = basis.get_index_from_integer(ket_code)

                if ket_idx >= 0:
                    target_orbit = state_to_orbit[ket_idx]
                    target_size = float(orbit_sizes[target_orbit])
                    amplitude = (
                        -hopping
                        * np.sqrt(bra_orbit_size / target_size)
                        * np.sqrt(n_source * new_occ[site_j])
                    )

                    if is_zero_momentum:
                        if orbit_id >= target_orbit:
                            rows.append(target_orbit)
                            cols.append(orbit_id)
                            elements.append(amplitude)
                    else:
                        tx = state_to_tx[ket_idx]
                        ty = state_to_ty[ket_idx]
                        phase = np.exp(-1j * (kx * tx + ky * ty))
                        rows.append(target_orbit)
                        cols.append(orbit_id)
                        elements.append(amplitude * phase)

    if is_zero_momentum:
        H_upper = sparse.csr_matrix(
            (elements, (rows, cols)),
            shape=(num_orbits, num_orbits),
            dtype=np.float64,
        )
        H_reduced = H_upper + H_upper.T - sparse.diags(H_upper.diagonal())
    else:
        H_raw = sparse.csr_matrix(
            (elements, (rows, cols)),
            shape=(num_orbits, num_orbits),
            dtype=np.complex128,
        )
        H_reduced = (H_raw + H_raw.conj().T) / 2.0

    H_reduced.eliminate_zeros()
    return H_reduced


# =====================================================================
# Wavefunction reconstruction
# =====================================================================

def reconstruct_wavefunction_2d(
    d: np.ndarray,
    basis: UnaryBasis,
    orbit_sizes: np.ndarray,
    state_to_orbit: np.ndarray,
    state_to_tx: np.ndarray = None,
    state_to_ty: np.ndarray = None,
    kx: float = 0.0,
    ky: float = 0.0,
) -> np.ndarray:
    """Reconstruct the full wavefunction from a symmetry-reduced eigenvector.

    The symmetrized basis vector for orbit α in momentum sector (kx, ky):

        |α, k> = (1/sqrt(N_α)) sum_{(tx,ty)} exp(i(kx·tx + ky·ty)) T(tx,ty)|r_α>

    So the full wavefunction ψ = sum_α d_α |α, k> has amplitudes:

        ψ[s] = d[α(s)] * exp(i(kx·tx_s + ky·ty_s)) / sqrt(|O_{α(s)}|)

    For (kx, ky) = (0, 0):
        ψ[s] = d[α(s)] / sqrt(|O_{α(s)}|)
        (same as 1D q=0 case — all phases are 1)

    Parameters
    ----------
    d : numpy.ndarray, shape (num_orbits,)
        Eigenvector in the symmetry-reduced basis.
    basis : UnaryBasis
        Full (unsymmetrized) basis.
    orbit_sizes, state_to_orbit, state_to_tx, state_to_ty :
        Output from find_orbits_2d().
    kx, ky : float
        Momentum sector. Default (0, 0).

    Returns
    -------
    psi : numpy.ndarray, shape (basis.dim,)
        Full wavefunction in the original basis.
    """
    is_zero_momentum = (kx == 0.0 and ky == 0.0)

    if is_zero_momentum:
        # Real wavefunction — same formula as 1D
        psi = np.empty(basis.dim, dtype=np.float64)
        for k in range(basis.dim):
            orbit = state_to_orbit[k]
            psi[k] = d[orbit] / np.sqrt(orbit_sizes[orbit])
        return psi
    else:
        # Complex wavefunction
        psi = np.empty(basis.dim, dtype=np.complex128)
        for k in range(basis.dim):
            orbit = state_to_orbit[k]
            tx = state_to_tx[k]
            ty = state_to_ty[k]
            phase = np.exp(1j * (kx * tx + ky * ty))
            psi[k] = d[orbit] * phase / np.sqrt(orbit_sizes[orbit])
        return psi
