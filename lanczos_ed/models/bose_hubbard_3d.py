"""
3D Bose-Hubbard Model on a Cubic Lattice
=========================================

The Bose-Hubbard Hamiltonian on a 3D cubic lattice:

    H = -t  sum_{<i,j>} (b†_i b_j + h.c.)
        + (U/2) sum_i  n_i (n_i - 1)
        - mu    sum_i  n_i

where <i,j> runs over nearest-neighbor bonds on an L × L × L cubic
lattice with periodic (torus) or open boundary conditions.

Site indexing convention (matching pigsfli2):
    site = x + y * L + z * L²
    x = site % L           (column, fast index)
    y = (site // L) % L    (row, middle index)
    z = site // L²         (layer, slow index)

Subregion definitions (for entanglement and fluctuations):

    Slab:    all sites with x < l                      →  l × L × L sites
    Column:  all sites with x < l AND y < l            →  l × l × L sites
    Cube:    all sites with x < l AND y < l AND z < l  →  l × l × l sites

Each subregion type restricts one more spatial dimension, forming a
natural hierarchy for studying boundary-shape dependence of entanglement.

Practical system sizes for ED:
    L=2:  8 sites   (feasible at any filling)
    L=3: 27 sites   (feasible at low filling, e.g. N=3: dim ≈ 3654)
    L=4: 64 sites   (impractical for most fillings)
"""

import numpy as np
from scipy import sparse
from ..basis import FockBasis
from ..unary_basis import UnaryBasis
from typing import Optional


# =====================================================================
# Subregion site-list generators
# =====================================================================

def slab_subregion(L: int, l: int) -> list:
    """Return site indices for a slab subregion of width l.

    The slab consists of all sites with x < l, giving an l × L × L
    block (l columns, all rows and layers).

    Parameters
    ----------
    L : int
        Linear system size (L × L × L lattice).
    l : int
        Slab width (number of x-columns in subsystem A).

    Returns
    -------
    sites : list of int
        Sorted site indices in the slab subregion.

    Example
    -------
    >>> slab_subregion(3, 1)  # x=0 column, all y, all z
    [0, 3, 6, 9, 12, 15, 18, 21, 24]
    """
    sites = []
    for z in range(L):
        for y in range(L):
            for x in range(l):
                sites.append(x + y * L + z * L * L)
    return sorted(sites)


def column_subregion(L: int, l: int) -> list:
    """Return site indices for a column subregion of size l × l × L.

    The column consists of all sites with x < l AND y < l, giving
    an l × l × L bar running through the z-direction.

    Parameters
    ----------
    L : int
        Linear system size (L × L × L lattice).
    l : int
        Cross-section side length.

    Returns
    -------
    sites : list of int
        Sorted site indices in the column subregion.

    Example
    -------
    >>> column_subregion(3, 1)  # x=0, y=0 column through all z
    [0, 9, 18]
    """
    sites = []
    for z in range(L):
        for y in range(l):
            for x in range(l):
                sites.append(x + y * L + z * L * L)
    return sorted(sites)


def cube_subregion(L: int, l: int) -> list:
    """Return site indices for a cube subregion of size l × l × l.

    The cube consists of all sites with x < l AND y < l AND z < l,
    giving an l³ corner block anchored at the origin (0,0,0).

    Parameters
    ----------
    L : int
        Linear system size (L × L × L lattice).
    l : int
        Side length of the cube subregion.

    Returns
    -------
    sites : list of int
        Sorted site indices in the cube subregion.

    Example
    -------
    >>> cube_subregion(3, 2)
    [0, 1, 3, 4, 9, 10, 12, 13]
    """
    sites = []
    for z in range(l):
        for y in range(l):
            for x in range(l):
                sites.append(x + y * L + z * L * L)
    return sorted(sites)


# =====================================================================
# 3D Bose-Hubbard Model
# =====================================================================

class BoseHubbard3D:
    """3D Bose-Hubbard model on an L × L × L cubic lattice.

    Parameters
    ----------
    linear_size : int
        Linear system size L (total sites = L³).
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
        'pbc' (periodic) or 'obc' (fully open). Default 'pbc'.
    use_symmetry : bool
        Accepted for interface consistency but raises
        NotImplementedError (3D symmetry reduction not yet implemented).

    Attributes
    ----------
    num_sites : int
        Total number of sites (L³).
    linear_size : int
        Linear dimension L.
    """

    def __init__(self, linear_size: int, hopping: float = 1.0,
                 interaction: float = 1.0, chemical_potential: float = 0.0,
                 max_occupation: Optional[int] = None,
                 total_particles: Optional[int] = None,
                 boundary: str = 'pbc',
                 use_symmetry: bool = False):

        self.linear_size = linear_size
        self.num_sites = linear_size ** 3
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

        # Symmetry: not yet implemented for 3D
        self.use_symmetry = False
        self.symmetry_info = None
        if use_symmetry:
            raise NotImplementedError(
                "3D translational symmetry reduction is not yet implemented. "
                "Set use_symmetry=False for 3D models."
            )

        self._hamiltonian_matrix = None

    @property
    def dim(self) -> int:
        """Hilbert space dimension."""
        return self.basis.dim

    @property
    def full_dim(self) -> int:
        """Full Hilbert space dimension."""
        return self.basis.dim

    def _get_neighbor_pairs(self):
        """Generate nearest-neighbor bond pairs on the 3D cubic lattice.

        Site indexing: site = x + y * L + z * L²

        Bonds along three directions:
            x-direction: (x, y, z) -- (x+1, y, z)
            y-direction: (x, y, z) -- (x, y+1, z)
            z-direction: (x, y, z) -- (x, y, z+1)

        For PBC, bonds wrap in all three directions.
        For OBC, no wrapping.
        """
        L = self.linear_size
        L2 = L * L
        bonds = []

        for z in range(L):
            for y in range(L):
                for x in range(L):
                    site = x + y * L + z * L2

                    # --- x-direction bond ---
                    if x < L - 1:
                        bonds.append((site, site + 1))
                    elif self.boundary == 'pbc' and L > 2:
                        bonds.append((site, y * L + z * L2))

                    # --- y-direction bond ---
                    if y < L - 1:
                        bonds.append((site, site + L))
                    elif self.boundary == 'pbc' and L > 2:
                        bonds.append((site, x + z * L2))

                    # --- z-direction bond ---
                    if z < L - 1:
                        bonds.append((site, site + L2))
                    elif self.boundary == 'pbc' and L > 2:
                        bonds.append((site, x + y * L))

        return bonds

    def hamiltonian(self) -> sparse.csr_matrix:
        """Build and return the sparse Hamiltonian matrix.

        Returns
        -------
        H : scipy.sparse.csr_matrix
            Shape (dim, dim).
        """
        if self._hamiltonian_matrix is not None:
            return self._hamiltonian_matrix

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

        # Enforce exact Hermiticity
        self._hamiltonian_matrix = (
            self._hamiltonian_matrix + self._hamiltonian_matrix.T
        ) / 2.0
        self._hamiltonian_matrix.eliminate_zeros()

        return self._hamiltonian_matrix

    def get_subregion(self, subregion_type: str, l: int) -> list:
        """Return site indices for a subregion of the lattice.

        Parameters
        ----------
        subregion_type : str
            'slab'   for x < l                      (l × L × L sites)
            'column' for x < l AND y < l             (l × l × L sites)
            'cube'   for x < l AND y < l AND z < l   (l³ sites)
        l : int
            Subregion parameter.

        Returns
        -------
        sites : list of int
        """
        L = self.linear_size
        if subregion_type == 'slab':
            return slab_subregion(L, l)
        elif subregion_type == 'column':
            return column_subregion(L, l)
        elif subregion_type == 'cube':
            return cube_subregion(L, l)
        else:
            raise ValueError(
                f"subregion_type must be 'slab', 'column', or 'cube', "
                f"got '{subregion_type}'"
            )

    def __repr__(self):
        L = self.linear_size
        if self.total_particles is not None:
            ensemble_info = f"N={self.total_particles}"
        else:
            ensemble_info = "grand canonical"

        return (
            f"BoseHubbard3D({L}x{L}x{L}, "
            f"t={self.hopping}, U={self.interaction}, "
            f"mu={self.chemical_potential}, "
            f"max_occupation={self.max_occupation}, "
            f"{ensemble_info}, boundary='{self.boundary}', "
            f"basis={type(self.basis).__name__}, "
            f"dim={self.dim})"
        )
