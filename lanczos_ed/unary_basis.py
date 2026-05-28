"""
Unary (Balls-and-Walls) Basis for Bosonic Lattice Models
=========================================================

Implements the compact integer encoding from:
    Barghathi, Usadi, Beck, Del Maestro
    Phys. Rev. B 105, L121116 (2022)

The idea exploits the bijection between N indistinguishable bosons
on L distinguishable sites and binary strings of length N + L with
exactly L ones:

    |n_0, n_1, ..., n_{L-1}>  <-->  integer with (N+L) significant bits

In the bit string (read from least-significant to most-significant bit):
    - 0-bits represent bosons ("balls")
    - 1-bits represent site boundaries ("walls")

Each site's occupation is encoded as n_i zeros followed by a 1.

Example (L=4, N=6):
    |2, 0, 1, 3>  -->  bits: 1 000 1 0 1 1 00
                              ^     ^ ^ ^
                              wall4 w3 w2 w1  (reading MSB to LSB)

    Reading LSB to MSB: [00][1][1][0][1][000][1]
                         ^^  ^  ^  ^  ^  ^^^  ^
                         2 balls wall 0 balls wall 1 ball wall 3 balls wall

    Integer = 0b1000101100 = 556

Advantages over tuple-based storage:
    - Memory: 1 integer per state instead of L integers
    - Lookup: O(log D) binary search on sorted integer list
    - Translation symmetry: cyclic shift is a bitwise operation
    - Occupation extraction: via trailing_zeros (hardware intrinsic)

Note on indexing:
    The Julia reference implementation uses 1-based indexing for both
    arrays and bit positions. This Python implementation uses 0-based
    indexing throughout: bit 0 is the LSB, array index 0 is the first
    element.
"""

import numpy as np
from math import comb
from typing import Optional

# Optional Numba acceleration for basis enumeration
try:
    from numba import njit
    HAS_NUMBA = True
except ImportError:
    HAS_NUMBA = False


# =====================================================================
# Numba-accelerated basis enumeration (compiled on first call)
# =====================================================================

if HAS_NUMBA:

    @njit(cache=True)
    def _recurse_numba(site, remaining, code, bit_pos,
                       L, n_max, integers, counter):
        """Numba-compiled recursive state enumeration.

        Identical logic to the Python version but JIT-compiled.
        Uses a 1-element counter array as a mutable reference
        (Numba doesn't support nonlocal).

        Recursion depth = L (at most ~30), well within limits.
        """
        if site == L - np.int64(1):
            if remaining <= n_max:
                integers[counter[0]] = code | (
                    np.int64(1) << (bit_pos + remaining)
                )
                counter[0] += np.int64(1)
            return

        sites_left = L - site - np.int64(1)
        max_here = min(n_max, remaining)
        min_here = max(np.int64(0), remaining - n_max * sites_left)

        for n_i in range(min_here, max_here + np.int64(1)):
            new_code = code | (np.int64(1) << (bit_pos + n_i))
            _recurse_numba(
                site + np.int64(1), remaining - n_i,
                new_code, bit_pos + n_i + np.int64(1),
                L, n_max, integers, counter,
            )


# =====================================================================
# Core encoding / decoding functions
# =====================================================================

def occupation_to_unary(occupation: tuple) -> int:
    """Encode an occupation-number tuple into a unary integer.

    The encoding places, for each site i, n_i zero-bits ("balls")
    followed by a one-bit ("wall"), reading from the LSB upward.

    Parameters
    ----------
    occupation : tuple of int
        (n_0, n_1, ..., n_{L-1}) occupation numbers.

    Returns
    -------
    v : int
        The unary-encoded integer.

    Example
    -------
    >>> occupation_to_unary((2, 0, 1))
    44  # = 0b101100
    """
    v = 0
    bit_position = 0
    for n_i in occupation:
        # Skip n_i zero-bits (balls for this site)
        bit_position += n_i
        # Place a one-bit (wall marking end of this site)
        v |= (1 << bit_position)
        bit_position += 1
    return v


def unary_to_occupation(v: int, num_sites: int) -> tuple:
    """Decode a unary integer into an occupation-number tuple.

    Reads the bit string from LSB to MSB. For each site, the
    number of trailing zeros gives the occupation (number of balls
    before the next wall).

    Parameters
    ----------
    v : int
        The unary-encoded integer.
    num_sites : int
        Number of lattice sites (L). Needed because trailing
        high-order zeros are not stored in the integer.

    Returns
    -------
    occupation : tuple of int
        (n_0, n_1, ..., n_{L-1}) occupation numbers.

    Example
    -------
    >>> unary_to_occupation(44, 3)
    (2, 0, 1)
    """
    occupation = []
    for _ in range(num_sites):
        # Count trailing zeros = number of bosons at this site
        n_i = _count_trailing_zeros(v)
        occupation.append(n_i)
        # Shift past the n_i zeros and the wall bit
        v >>= (n_i + 1)
    return tuple(occupation)


def _count_trailing_zeros(v: int) -> int:
    """Count the number of trailing zero bits in integer v.

    Uses the trick: v & (-v) isolates the lowest set bit,
    then bit_length() - 1 gives its position (= number of
    trailing zeros).

    Returns 0 for v with LSB = 1, and a large number for v = 0.
    """
    if v == 0:
        return 64  # effectively infinite for our purposes
    return (v & -v).bit_length() - 1


# =====================================================================
# Wall position array (U-array from the Julia implementation)
# =====================================================================

def generate_wall_positions(v: int, num_sites: int) -> list:
    """Find the bit positions of all wall bits (1-bits) in v.

    This corresponds to generateU! in the Julia code, but uses
    0-based bit positions (Julia uses 1-based).

    Parameters
    ----------
    v : int
        Unary-encoded basis state.
    num_sites : int
        Number of sites (L).

    Returns
    -------
    wall_positions : list of int
        0-based bit position of each site's wall bit.
        wall_positions[i] is the bit index of the wall for site i.

    Example
    -------
    >>> generate_wall_positions(44, 3)  # 0b101100
    [2, 3, 5]
    """
    wall_positions = []
    cumulative_position = 0
    temp_v = v
    for _ in range(num_sites):
        trailing = _count_trailing_zeros(temp_v)
        cumulative_position += trailing  # skip the zeros
        wall_positions.append(cumulative_position)
        cumulative_position += 1  # skip the wall bit itself
        temp_v >>= (trailing + 1)
    return wall_positions


def wall_positions_to_occupation(wall_positions: list) -> tuple:
    """Convert wall positions to occupation numbers.

    The occupation of site i is the gap between consecutive walls,
    minus 1 (for the wall bit itself):
        n_0 = wall_positions[0]
        n_i = wall_positions[i] - wall_positions[i-1] - 1  (for i > 0)

    This corresponds to generateN in the Julia code, adjusted for
    0-based indexing.

    Example
    -------
    >>> wall_positions_to_occupation([2, 3, 5])
    (2, 0, 1)
    """
    occupation = [wall_positions[0]]
    for i in range(1, len(wall_positions)):
        occupation.append(wall_positions[i] - wall_positions[i - 1] - 1)
    return tuple(occupation)


# =====================================================================
# Basis class
# =====================================================================

class UnaryBasis:
    """Fock space basis using unary (balls-and-walls) encoding.

    Each basis state is stored as a single integer, providing a
    factor-of-L memory reduction compared to storing occupation
    tuples. State lookup uses O(log D) binary search on the
    sorted integer list.

    Currently supports the canonical ensemble (fixed particle number N).
    For grand canonical, use FockBasis instead (the unary encoding
    requires fixed N to define the bit-string length).

    Parameters
    ----------
    num_sites : int
        Number of lattice sites (L).
    total_particles : int
        Total number of bosons (N).
    max_occupation : int or None
        Maximum occupation per site (n_max).
        None means unrestricted (effectively n_max = N).

    Attributes
    ----------
    dim : int
        Hilbert space dimension.
    """

    def __init__(self, num_sites: int, total_particles: int,
                 max_occupation: Optional[int] = None):
        self.num_sites = num_sites
        self.total_particles = total_particles
        self.is_canonical = True

        # Effective max occupation
        if max_occupation is not None:
            self.max_occupation = max_occupation
        else:
            self.max_occupation = total_particles

        # Total bit-string length for each state
        self._bit_length = num_sites + total_particles

        # Pre-compute the Hilbert space dimension
        self._dim = self._count_states(num_sites, total_particles,
                                       self.max_occupation)

        # Build the sorted array of unary-encoded integers
        # Stored as a numpy int64 array for ~3x memory savings
        # over a Python list of int objects (8 bytes vs ~28 bytes each)
        self._integers = np.empty(self._dim, dtype=np.int64)
        self._build()

    # --- Convenience aliases (match FockBasis interface) ---

    @property
    def L(self) -> int:
        return self.num_sites

    @property
    def N(self) -> Optional[int]:
        return self.total_particles

    @property
    def n_max(self) -> int:
        return self.max_occupation

    @property
    def dim(self) -> int:
        return self._dim

    # -----------------------------------------------------------------
    # Hilbert space dimension (combinatorial pre-computation)
    # -----------------------------------------------------------------

    @staticmethod
    def _count_states(L: int, N: int, n_max: int) -> int:
        """Compute the Hilbert space dimension combinatorially.

        For unrestricted occupation (n_max >= N), this is the
        "stars and bars" formula:

            D = C(N + L - 1, N)

        For restricted occupation (n_max < N), we use inclusion-exclusion:

            D = sum_{k=0}^{floor(N/(n_max+1))} (-1)^k C(L, k)
                * C(N - k*(n_max+1) + L - 1, L - 1)

        This avoids enumerating states just to count them.

        Parameters
        ----------
        L : int
            Number of sites.
        N : int
            Total particle number.
        n_max : int
            Maximum occupation per site.

        Returns
        -------
        dim : int
            Number of valid Fock states.
        """
        if n_max >= N:
            # Unrestricted: simple stars-and-bars
            return comb(N + L - 1, N)

        # Restricted: inclusion-exclusion over the constraint
        # n_i <= n_max for all i
        total = 0
        for k in range(N // (n_max + 1) + 1):
            # k = number of sites violating the constraint
            remainder = N - k * (n_max + 1)
            if remainder < 0:
                break
            sign = (-1) ** k
            total += sign * comb(L, k) * comb(remainder + L - 1, L - 1)
        return total

    # -----------------------------------------------------------------
    # Basis construction (constrained recursive enumeration)
    # -----------------------------------------------------------------

    def _build(self):
        """Enumerate all valid Fock states, encode as unary integers, and sort.

        Uses constrained recursive generation: at each site, only the
        range of occupations that can lead to a valid total is explored.
        This visits exactly D = dim valid states, compared to the old
        cartesian-product approach which iterated through (n_max+1)^L
        candidates and rejected most of them.

        For L=N=16 (unrestricted), the old method tried ~5×10^19
        candidates; this one visits exactly C(31,16) ≈ 300 million.
        """
        L = self.num_sites
        N = self.total_particles
        n_max = self.max_occupation
        integers = self._integers   # pre-allocated numpy array
        count = 0                    # running insertion index

        # ----------------------------------------------------------
        # Recursive generator: place particles site by site.
        #
        # At each site, we constrain the occupation to the range
        # [min_here, max_here] where:
        #   max_here = min(n_max, remaining)
        #       can't exceed n_max or the remaining particle budget
        #   min_here = max(0, remaining - n_max * sites_left)
        #       must place enough here so that later sites (each
        #       holding at most n_max) can absorb the rest
        #
        # The unary integer is built incrementally via bit operations:
        #   - n_i zero-bits (balls) are implicit (just advance bit_pos)
        #   - a one-bit (wall) marks the end of each site
        # ----------------------------------------------------------

        def recurse(site, remaining, code, bit_pos):
            """Place particles on sites[site:], building the unary code.

            Parameters
            ----------
            site : int
                Current site index (0 to L-1).
            remaining : int
                Particles still to be placed.
            code : int
                Unary integer built so far (sites 0..site-1).
            bit_pos : int
                Next available bit position in the code.
            """
            nonlocal count

            if site == L - 1:
                # Last site: must absorb all remaining particles
                if remaining <= n_max:
                    # Place 'remaining' balls + one wall
                    integers[count] = code | (1 << (bit_pos + remaining))
                    count += 1
                return

            sites_left = L - site - 1   # sites after this one
            max_here = min(n_max, remaining)
            min_here = max(0, remaining - n_max * sites_left)

            for n_i in range(min_here, max_here + 1):
                # Encode this site: n_i zeros (skip) + one wall bit
                new_code = code | (1 << (bit_pos + n_i))
                recurse(site + 1, remaining - n_i,
                        new_code, bit_pos + n_i + 1)

        # Dispatch: Numba-compiled recursion when available
        if HAS_NUMBA:
            counter = np.zeros(1, dtype=np.int64)
            _recurse_numba(
                np.int64(0), np.int64(N), np.int64(0), np.int64(0),
                np.int64(L), np.int64(n_max), integers, counter,
            )
            count = int(counter[0])
        else:
            recurse(0, N, 0, 0)

        # Sanity check: we should have found exactly _dim states
        assert count == self._dim, (
            f"Enumeration mismatch: found {count} states, expected {self._dim}"
        )

        # Sort for O(log D) binary search lookups
        self._integers.sort()

    def get_state(self, basis_index: int) -> tuple:
        """Return the occupation-number tuple for a given basis index.

        Parameters
        ----------
        basis_index : int
            Index into the basis (0 <= basis_index < dim).

        Returns
        -------
        occupation : tuple of int
        """
        # Convert numpy int64 to Python int for bit operations
        return unary_to_occupation(int(self._integers[basis_index]),
                                   self.num_sites)

    def get_index(self, occupation: tuple) -> int:
        """Return the basis index for an occupation-number tuple.

        Uses O(log D) binary search on the sorted integer list.

        Parameters
        ----------
        occupation : tuple of int
            The occupation numbers to look up.

        Returns
        -------
        basis_index : int
            Position in the basis, or -1 if not found.
        """
        code = occupation_to_unary(occupation)
        return self.get_index_from_integer(code)

    def get_index_from_integer(self, unary_integer: int) -> int:
        """Return the basis index for a unary-encoded integer.

        Uses O(log D) binary search via numpy's searchsorted.

        Returns -1 if the integer is not in the basis.
        """
        pos = np.searchsorted(self._integers, unary_integer)
        if pos < self._dim and self._integers[pos] == unary_integer:
            return int(pos)
        return -1

    def get_integer(self, basis_index: int) -> int:
        """Return the unary-encoded integer for a given basis index."""
        return int(self._integers[basis_index])

    def all_states_as_array(self) -> np.ndarray:
        """Return all states as a 2D numpy array of shape (dim, num_sites).

        This decodes every unary integer back to an occupation tuple.
        Useful for computing observables, but note this reconstructs
        the full occupation representation that the unary encoding
        was designed to avoid storing.
        """
        states = np.zeros((self.dim, self.num_sites), dtype=np.int32)
        for idx in range(self.dim):
            # Convert numpy int64 to Python int for bit operations
            states[idx, :] = unary_to_occupation(
                int(self._integers[idx]), self.num_sites
            )
        return states

    def __len__(self):
        return self.dim

    def __repr__(self):
        return (
            f"UnaryBasis(num_sites={self.num_sites}, "
            f"total_particles={self.total_particles}, "
            f"max_occupation={self.max_occupation}, "
            f"dim={self.dim})"
        )
