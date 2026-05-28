"""
Fock basis construction for bosonic lattice models.

The many-body Hilbert space for bosons on a lattice is spanned by
occupation-number states (Fock states):

    |n_0, n_1, ..., n_{L-1}>

where n_i is the number of bosons at site i.

Two ensemble types are supported:

    Canonical (fixed N):
        Only states with sum_i n_i = N are included.
        The Hilbert space dimension is C(N + L - 1, L - 1).

    Grand canonical (no fixed N):
        All states with 0 <= n_i <= n_max are included.
        The Hilbert space dimension is (n_max + 1)^L.
        Requires specifying n_max (otherwise the space is infinite).

Each state is stored as a tuple of occupation numbers and also
encoded as a single integer (mixed-radix with base n_max + 1)
for fast dictionary-based lookup.
"""

import numpy as np
from itertools import product as cartesian_product
from typing import Optional


def encode_occupation_to_integer(occupation: tuple, max_occupation: int) -> int:
    """Encode an occupation-number tuple into a single integer.

    Uses a mixed-radix representation where each site contributes
    a digit in base (max_occupation + 1).

    Example with max_occupation=2, occupation=(1, 0, 2):
        code = 1 * 3^2 + 0 * 3^1 + 2 * 3^0 = 9 + 0 + 2 = 11
    """
    base = max_occupation + 1
    code = 0
    for site_occupation in occupation:
        code = code * base + site_occupation
    return code


def decode_integer_to_occupation(code: int, num_sites: int,
                                  max_occupation: int) -> tuple:
    """Decode an integer back into an occupation-number tuple."""
    base = max_occupation + 1
    occupation = []
    for _ in range(num_sites):
        occupation.append(code % base)
        code //= base
    return tuple(reversed(occupation))


class FockBasis:
    """Fock space basis for bosons on a lattice.

    Parameters
    ----------
    num_sites : int
        Number of lattice sites (L).
    max_occupation : int or None
        Maximum number of bosons allowed per site (n_max).
        - For canonical ensemble: defaults to N (total particles) if not given.
        - For grand canonical: must be specified (Hilbert space is infinite otherwise).
    total_particles : int or None
        Fixed total particle number (N).
        - If given: canonical ensemble (only states with sum n_i = N).
        - If None: grand canonical ensemble (all particle sectors).

    Attributes
    ----------
    dim : int
        Hilbert space dimension (number of basis states).
    """

    def __init__(self, num_sites: int, max_occupation: Optional[int] = None,
                 total_particles: Optional[int] = None):
        self.num_sites = num_sites
        self.total_particles = total_particles
        self.is_canonical = (total_particles is not None)

        # Determine the effective maximum occupation per site
        if max_occupation is not None:
            self.max_occupation = max_occupation
        elif self.is_canonical:
            # In canonical ensemble, no site can hold more than N bosons
            self.max_occupation = total_particles
        else:
            raise ValueError(
                "max_occupation must be specified for grand canonical ensemble "
                "(the Hilbert space is infinite without an occupation cutoff)."
            )

        # These will hold the enumerated basis states
        self._states = []       # list of occupation tuples
        self._state_index = {}  # encoded integer -> position in _states

        self._enumerate_basis()

    # --- Convenience aliases for short-form access (used in Hamiltonian code) ---

    @property
    def L(self) -> int:
        """Number of lattice sites (shorthand)."""
        return self.num_sites

    @property
    def N(self) -> Optional[int]:
        """Total particle number (shorthand). None if grand canonical."""
        return self.total_particles

    @property
    def n_max(self) -> int:
        """Maximum occupation per site (shorthand)."""
        return self.max_occupation

    @property
    def dim(self) -> int:
        """Hilbert space dimension."""
        return len(self._states)

    def _enumerate_basis(self):
        """Build the list of all valid Fock states.

        Iterates over all possible occupation configurations and keeps
        those satisfying the particle number constraint (if canonical).
        """
        occupation_range_per_site = range(self.max_occupation + 1)
        all_site_ranges = [occupation_range_per_site] * self.num_sites

        for occupation in cartesian_product(*all_site_ranges):
            # In canonical ensemble, skip states with wrong particle number
            if self.total_particles is not None:
                if sum(occupation) != self.total_particles:
                    continue

            code = encode_occupation_to_integer(occupation, self.max_occupation)
            self._state_index[code] = len(self._states)
            self._states.append(occupation)

    def get_state(self, basis_index: int) -> tuple:
        """Return the occupation-number tuple for a given basis index.

        Parameters
        ----------
        basis_index : int
            Index into the basis (0 <= basis_index < dim).

        Returns
        -------
        occupation : tuple of int
            The occupation numbers (n_0, n_1, ..., n_{L-1}).
        """
        return self._states[basis_index]

    def get_index(self, occupation: tuple) -> int:
        """Return the basis index for a given occupation-number tuple.

        Parameters
        ----------
        occupation : tuple of int
            The occupation numbers to look up.

        Returns
        -------
        basis_index : int
            Position in the basis, or -1 if the state is not in the basis.
        """
        code = encode_occupation_to_integer(occupation, self.max_occupation)
        return self._state_index.get(code, -1)

    def all_states_as_array(self) -> np.ndarray:
        """Return all basis states as a 2D numpy array.

        Returns
        -------
        states : ndarray of shape (dim, num_sites), dtype int32
            Each row is the occupation numbers of one basis state.
        """
        return np.array(self._states, dtype=np.int32)

    def __len__(self):
        return self.dim

    def __repr__(self):
        if self.is_canonical:
            ensemble_info = f"N={self.total_particles}"
        else:
            ensemble_info = "grand canonical"
        return (f"FockBasis(num_sites={self.num_sites}, "
                f"max_occupation={self.max_occupation}, "
                f"{ensemble_info}, dim={self.dim})")
