"""
2D Bose-Hubbard Model on a Square Lattice
==========================================

The Bose-Hubbard Hamiltonian on a 2D square lattice:

    H = -t  sum_{<i,j>} (b†_i b_j + h.c.)
        + (U/2) sum_i  n_i (n_i - 1)
        - mu    sum_i  n_i

where <i,j> runs over nearest-neighbor bonds on an L × L square
lattice with periodic (torus) or open boundary conditions.

Site indexing convention (matching pigsfli2):
    site = x + y * L
    x = site % L      (column, fast index)
    y = site // L      (row, slow index)

This gives the standard row-major layout:
    (0,0)  (1,0)  (2,0)  ...  (L-1,0)
    (0,1)  (1,1)  (2,1)  ...  (L-1,1)
      .      .      .            .
    (0,L-1)(1,L-1)(2,L-1) ... (L-1,L-1)

Subregion definitions (for entanglement and fluctuations):

    Strip:  all sites with x < l  →  l × L sites (vertical slab)
    Square: all sites with x < l AND y < l  →  l × l corner block

Both use the same site = x + y*L convention for consistency with
pigsfli2 QMC cross-validation.
"""

import numpy as np
from scipy import sparse
from ..basis import FockBasis
from ..unary_basis import UnaryBasis
from typing import Optional


# =====================================================================
# Subregion site-list generators
# =====================================================================

def strip_subregion(L: int, l: int) -> list:
    """Return site indices for a strip subregion of width l.

    The strip consists of all sites with x < l, giving an l × L
    vertical slab of sites (l columns, all L rows).

    Parameters
    ----------
    L : int
        Linear system size (L × L lattice).
    l : int
        Strip width (number of columns in subsystem A).

    Returns
    -------
    sites : list of int
        Sorted site indices in the strip subregion.

    Example
    -------
    >>> strip_subregion(4, 2)
    [0, 1, 4, 5, 8, 9, 12, 13]
    """
    sites = []
    for y in range(L):
        for x in range(l):
            sites.append(x + y * L)
    return sorted(sites)


def square_subregion(L: int, l: int) -> list:
    """Return site indices for a square subregion of size l × l.

    The square consists of all sites with x < l AND y < l, giving
    an l × l corner block anchored at the origin (0,0).

    Parameters
    ----------
    L : int
        Linear system size (L × L lattice).
    l : int
        Side length of the square subregion.

    Returns
    -------
    sites : list of int
        Sorted site indices in the square subregion.

    Example
    -------
    >>> square_subregion(4, 2)
    [0, 1, 4, 5]
    """
    sites = []
    for y in range(l):
        for x in range(l):
            sites.append(x + y * L)
    return sorted(sites)


# =====================================================================
# 2D Bose-Hubbard Model
# =====================================================================

class BoseHubbard2D:
    """2D Bose-Hubbard model on an L × L square lattice.

    Parameters
    ----------
    linear_size : int
        Linear system size L (total sites = L²).
    hopping : float
        Hopping amplitude t (default 1.0).
    interaction : float
        On-site interaction strength U (default 1.0).
    chemical_potential : float
        Chemical potential mu (default 0.0).
    max_occupation : int or None
        Maximum bosons per site. None defaults to N for canonical.
    total_particles : int or None
        Fixed particle number N. None for grand canonical.
    boundary : str
        'pbc' (periodic, torus) or 'obc' (fully open). Default 'pbc'.
    use_symmetry : bool
        If True and boundary='pbc', exploit 2D translational symmetry
        (Tx, Ty generators) to reduce the Hilbert space. Works in the
        (kx, ky) = (0, 0) momentum sector where the ground state lives.
        Provides up to L² reduction in Hilbert space dimension.

        **Not all models have these symmetries.** This should only be
        enabled for translationally invariant Hamiltonians (uniform
        couplings, no disorder).

    Attributes
    ----------
    num_sites : int
        Total number of sites (L²).
    linear_size : int
        Linear dimension L.

    Examples
    --------
    >>> model = BoseHubbard2D(linear_size=3, hopping=1.0, interaction=4.0,
    ...                       total_particles=9)
    >>> H = model.hamiltonian()
    >>> subsystem = strip_subregion(3, 1)  # first column
    """

    def __init__(self, linear_size: int, hopping: float = 1.0,
                 interaction: float = 1.0, chemical_potential: float = 0.0,
                 max_occupation: Optional[int] = None,
                 total_particles: Optional[int] = None,
                 boundary: str = 'pbc',
                 use_symmetry: bool = False):

        self.linear_size = linear_size
        self.num_sites = linear_size * linear_size
        self.hopping = hopping
        self.interaction = interaction
        self.chemical_potential = chemical_potential
        self.max_occupation = max_occupation
        self.total_particles = total_particles
        self.boundary = boundary.lower()

        if self.boundary not in ('pbc', 'obc'):
            raise ValueError(
                f"boundary must be 'pbc' or 'obc', got '{boundary}'"
            )

        # Build the basis
        if total_particles is not None:
            self.basis = UnaryBasis(
                num_sites=self.num_sites,
                total_particles=total_particles,
                max_occupation=max_occupation,
            )
        else:
            self.basis = FockBasis(
                num_sites=self.num_sites,
                max_occupation=max_occupation,
                total_particles=total_particles,
            )

        # --- 2D Translational symmetry reduction (opt-in, PBC only) ---
        # Only valid for translationally invariant Hamiltonians.
        # Requires canonical ensemble (UnaryBasis) and PBC.
        self.use_symmetry = (
            use_symmetry
            and self.boundary == 'pbc'
            and isinstance(self.basis, UnaryBasis)
        )
        self.symmetry_info = None

        if self.use_symmetry:
            from ..symmetry_2d import find_orbits_2d
            (orbit_leaders, orbit_sizes, num_orbits,
             state_to_orbit, state_to_tx, state_to_ty) = find_orbits_2d(
                self.basis, self.linear_size
            )
            self.symmetry_info = {
                'orbit_leaders': orbit_leaders,
                'orbit_sizes': orbit_sizes,
                'num_orbits': num_orbits,
                'state_to_orbit': state_to_orbit,
                'state_to_tx': state_to_tx,
                'state_to_ty': state_to_ty,
            }

        # Cache for the Hamiltonian matrix
        self._hamiltonian_matrix = None

    @property
    def dim(self) -> int:
        """Hilbert space dimension (reduced if symmetry is active)."""
        if self.use_symmetry:
            return self.symmetry_info['num_orbits']
        return self.basis.dim

    @property
    def full_dim(self) -> int:
        """Full Hilbert space dimension (always the unsymmetrized value)."""
        return self.basis.dim

    def _get_neighbor_pairs(self):
        """Generate nearest-neighbor bond pairs on the 2D square lattice.

        Site indexing: site = x + y * L

        Bonds:
            Horizontal: (x, y) -- (x+1, y)
            Vertical:   (x, y) -- (x, y+1)

        For PBC, horizontal bonds wrap x=L-1 to x=0, and vertical
        bonds wrap y=L-1 to y=0.
        For OBC, no wrapping.
        """
        L = self.linear_size
        bonds = []

        for y in range(L):
            for x in range(L):
                site = x + y * L

                # --- Horizontal bond: (x, y) -- (x+1, y) ---
                if x < L - 1:
                    # Interior bond (always present)
                    bonds.append((site, site + 1))
                elif self.boundary == 'pbc' and L > 2:
                    # PBC wrap: (L-1, y) -- (0, y)
                    bonds.append((site, y * L))

                # --- Vertical bond: (x, y) -- (x, y+1) ---
                if y < L - 1:
                    # Interior bond (always present)
                    bonds.append((site, site + L))
                elif self.boundary == 'pbc' and L > 2:
                    # PBC wrap: (x, L-1) -- (x, 0)
                    bonds.append((site, x))

        return bonds

    def hamiltonian(self) -> sparse.csr_matrix:
        """Build and return the sparse Hamiltonian matrix.

        If use_symmetry=True, returns the symmetry-reduced Hamiltonian
        in the (kx, ky) = (0, 0) sector (smaller dimension). Otherwise
        returns the full Hamiltonian.

        Returns
        -------
        H : scipy.sparse.csr_matrix
            Shape (dim, dim) where dim is the (possibly reduced) dimension.
        """
        if self._hamiltonian_matrix is not None:
            return self._hamiltonian_matrix

        if self.use_symmetry:
            return self._build_symmetry_hamiltonian()

        basis = self.basis
        hilbert_dim = basis.dim
        neighbor_pairs = self._get_neighbor_pairs()

        row_indices = []
        col_indices = []
        matrix_elements = []

        for state_index in range(hilbert_dim):
            occupation = basis.get_state(state_index)

            # =============================================================
            # Diagonal: (U/2) sum_i n_i(n_i - 1) - mu sum_i n_i
            # =============================================================
            diagonal_energy = 0.0
            for site in range(self.num_sites):
                n_i = occupation[site]
                diagonal_energy += (self.interaction / 2.0) * n_i * (n_i - 1)
                diagonal_energy -= self.chemical_potential * n_i

            if diagonal_energy != 0.0:
                row_indices.append(state_index)
                col_indices.append(state_index)
                matrix_elements.append(diagonal_energy)

            # =============================================================
            # Off-diagonal: -t sum_{<i,j>} (b†_i b_j + b†_j b_i)
            # =============================================================
            for site_i, site_j in neighbor_pairs:

                # --- b†_i b_j : hop from site_j to site_i ---
                n_at_source = occupation[site_j]
                n_at_target = occupation[site_i]

                if (n_at_source > 0
                        and (self.max_occupation is None
                             or n_at_target < self.max_occupation)):
                    new_occupation = list(occupation)
                    new_occupation[site_j] -= 1
                    new_occupation[site_i] += 1

                    target_state_index = basis.get_index(
                        tuple(new_occupation)
                    )
                    if target_state_index >= 0:
                        hopping_element = -self.hopping * np.sqrt(
                            n_at_source * new_occupation[site_i]
                        )
                        row_indices.append(state_index)
                        col_indices.append(target_state_index)
                        matrix_elements.append(hopping_element)

                # --- b†_j b_i : hop from site_i to site_j ---
                n_at_source = occupation[site_i]
                n_at_target = occupation[site_j]

                if (n_at_source > 0
                        and (self.max_occupation is None
                             or n_at_target < self.max_occupation)):
                    new_occupation = list(occupation)
                    new_occupation[site_i] -= 1
                    new_occupation[site_j] += 1

                    target_state_index = basis.get_index(
                        tuple(new_occupation)
                    )
                    if target_state_index >= 0:
                        hopping_element = -self.hopping * np.sqrt(
                            n_at_source * new_occupation[site_j]
                        )
                        row_indices.append(state_index)
                        col_indices.append(target_state_index)
                        matrix_elements.append(hopping_element)

        # Assemble sparse matrix
        self._hamiltonian_matrix = sparse.csr_matrix(
            (matrix_elements, (row_indices, col_indices)),
            shape=(hilbert_dim, hilbert_dim),
            dtype=np.float64,
        )

        # Enforce exact Hermiticity: H = (H + H†) / 2
        self._hamiltonian_matrix = (
            self._hamiltonian_matrix + self._hamiltonian_matrix.T
        ) / 2.0
        self._hamiltonian_matrix.eliminate_zeros()

        return self._hamiltonian_matrix

    def _build_symmetry_hamiltonian(self) -> sparse.csr_matrix:
        """Build the 2D symmetry-reduced Hamiltonian via symmetry_2d module.

        Uses the (kx, ky) = (0, 0) momentum sector where the ground
        state of the translationally invariant Bose-Hubbard model lives.
        The reduced Hamiltonian is real symmetric.
        """
        from ..symmetry_2d import build_reduced_hamiltonian_2d

        info = self.symmetry_info
        self._hamiltonian_matrix = build_reduced_hamiltonian_2d(
            basis=self.basis,
            L=self.linear_size,
            orbit_leaders=info['orbit_leaders'],
            orbit_sizes=info['orbit_sizes'],
            num_orbits=info['num_orbits'],
            state_to_orbit=info['state_to_orbit'],
            state_to_tx=info['state_to_tx'],
            state_to_ty=info['state_to_ty'],
            hopping=self.hopping,
            interaction=self.interaction,
            chemical_potential=self.chemical_potential,
            neighbor_pairs=self._get_neighbor_pairs(),
            kx=0.0,
            ky=0.0,
        )
        return self._hamiltonian_matrix

    def reconstruct_wavefunction(self, d: np.ndarray) -> np.ndarray:
        """Reconstruct the full wavefunction from a symmetry-reduced eigenvector.

        Only available when use_symmetry=True. Maps the reduced
        eigenvector (length num_orbits) to the full basis (length full_dim).

        Parameters
        ----------
        d : numpy.ndarray, shape (num_orbits,)
            Eigenvector in the reduced (orbit) basis.

        Returns
        -------
        psi : numpy.ndarray, shape (full_dim,)
            Full wavefunction for computing observables.
        """
        if not self.use_symmetry or self.symmetry_info is None:
            raise RuntimeError(
                "reconstruct_wavefunction() requires use_symmetry=True"
            )

        from ..symmetry_2d import reconstruct_wavefunction_2d
        info = self.symmetry_info
        return reconstruct_wavefunction_2d(
            d, self.basis,
            info['orbit_sizes'],
            info['state_to_orbit'],
            info['state_to_tx'],
            info['state_to_ty'],
            kx=0.0, ky=0.0,
        )

    def get_subregion(self, subregion_type: str, l: int) -> list:
        """Return site indices for a subregion of the lattice.

        Parameters
        ----------
        subregion_type : str
            'strip' for vertical slab (x < l), 'square' for corner
            block (x < l AND y < l).
        l : int
            Subregion parameter (width for strip, side length for square).

        Returns
        -------
        sites : list of int
            Sorted site indices in the subregion.
        """
        L = self.linear_size
        if subregion_type == 'strip':
            return strip_subregion(L, l)
        elif subregion_type == 'square':
            return square_subregion(L, l)
        else:
            raise ValueError(
                f"subregion_type must be 'strip' or 'square', "
                f"got '{subregion_type}'"
            )

    def __repr__(self):
        L = self.linear_size
        if self.total_particles is not None:
            ensemble_info = f"N={self.total_particles}"
        else:
            ensemble_info = "grand canonical"

        sym_info = ""
        if self.use_symmetry:
            sym_info = (
                f", symmetry=k00, "
                f"reduced_dim={self.dim}, full_dim={self.full_dim}"
            )

        return (
            f"BoseHubbard2D({L}x{L}, "
            f"t={self.hopping}, U={self.interaction}, "
            f"mu={self.chemical_potential}, "
            f"max_occupation={self.max_occupation}, "
            f"{ensemble_info}, boundary='{self.boundary}', "
            f"basis={type(self.basis).__name__}"
            f"{sym_info if self.use_symmetry else f', dim={self.dim}'}"
            f")"
        )
