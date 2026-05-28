"""
1D Bose-Hubbard Model
=====================

The Bose-Hubbard Hamiltonian describes interacting bosons on a lattice:

    H = -t  sum_{<i,j>} (b†_i b_j + h.c.)
        + (U/2) sum_i  n_i (n_i - 1)
        - mu    sum_i  n_i

where:
    b†_i, b_i  = bosonic creation/annihilation operators at site i
    n_i = b†_i b_i  = number operator at site i
    t   = hopping amplitude (kinetic energy scale)
    U   = on-site interaction strength (repulsive for U > 0)
    mu  = chemical potential (controls average particle number)
    <i,j> denotes nearest-neighbor pairs

The bosonic operators satisfy:
    b_i |n_i> = sqrt(n_i)     |n_i - 1>
    b†_i|n_i> = sqrt(n_i + 1) |n_i + 1>

This module supports:
    - Periodic (PBC) and open (OBC) boundary conditions
    - Canonical (fixed N) and grand canonical ensembles
    - Adjustable maximum site occupation (n_max)
"""

import numpy as np
from scipy import sparse
from ..basis import FockBasis
from ..unary_basis import UnaryBasis
from typing import Optional


class BoseHubbard1D:
    """1D Bose-Hubbard model Hamiltonian builder.

    Parameters
    ----------
    num_sites : int
        Number of lattice sites (L).
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
        'pbc' (periodic) or 'obc' (open). Default 'pbc'.
    basis_type : str
        'unary' (default) uses the compact balls-and-walls encoding
        from Barghathi et al. PRB 105, L121116 (2022).
        'fock' uses the original tuple-based FockBasis.
        Note: 'unary' only supports canonical ensemble (fixed N).
    use_symmetry : bool
        If True and boundary='pbc', exploit translational + reflection
        symmetry to reduce the Hilbert space (q=0, R=+1 sector).
        The reduced Hamiltonian is returned by hamiltonian().
        Use reconstruct_wavefunction() to recover the full eigenvector
        for computing observables.

        **Not all models have these symmetries.** This should only be
        enabled for translationally invariant Hamiltonians (uniform
        couplings, no disorder). Models like the Bose glass with
        random on-site potentials must set use_symmetry=False.

    Examples
    --------
    >>> # Canonical: 3 bosons on 6 sites, U/t = 4
    >>> model = BoseHubbard1D(num_sites=6, hopping=1.0, interaction=4.0,
    ...                       total_particles=3)
    >>> H = model.hamiltonian()

    >>> # Grand canonical: 4 sites, n_max=2, mu=0.5
    >>> model = BoseHubbard1D(num_sites=4, max_occupation=2,
    ...                       chemical_potential=0.5)

    >>> # With symmetry reduction (PBC only)
    >>> model = BoseHubbard1D(num_sites=6, hopping=1.0, interaction=4.0,
    ...                       total_particles=3, use_symmetry=True)
    >>> H_red = model.hamiltonian()    # reduced dimension
    >>> psi_full = model.reconstruct_wavefunction(evec_reduced)
    """

    def __init__(self, num_sites: int, hopping: float = 1.0,
                 interaction: float = 1.0, chemical_potential: float = 0.0,
                 max_occupation: Optional[int] = None,
                 total_particles: Optional[int] = None,
                 boundary: str = 'pbc',
                 basis_type: str = 'unary',
                 use_symmetry: bool = False):

        self.num_sites = num_sites
        self.hopping = hopping
        self.interaction = interaction
        self.chemical_potential = chemical_potential
        self.max_occupation = max_occupation
        self.total_particles = total_particles
        self.boundary = boundary.lower()
        self.basis_type = basis_type.lower()

        if self.boundary not in ('pbc', 'obc'):
            raise ValueError(
                f"boundary must be 'pbc' or 'obc', got '{boundary}'"
            )

        # Build the basis for this model
        if self.basis_type == 'unary' and total_particles is not None:
            # Unary (balls-and-walls) encoding: canonical ensemble only
            self.basis = UnaryBasis(
                num_sites=num_sites,
                total_particles=total_particles,
                max_occupation=max_occupation,
            )
        else:
            # Fall back to tuple-based FockBasis
            # (required for grand canonical, or if explicitly requested)
            self.basis = FockBasis(
                num_sites=num_sites,
                max_occupation=max_occupation,
                total_particles=total_particles,
            )

        # --- Symmetry reduction (opt-in, PBC only) ---
        # Only valid for translationally invariant Hamiltonians.
        # Models with disorder or non-uniform couplings must NOT use this.
        self.use_symmetry = (
            use_symmetry
            and self.boundary == 'pbc'
            and isinstance(self.basis, UnaryBasis)
        )
        self.symmetry_info = None

        if self.use_symmetry:
            from ..symmetry import find_cycles
            leaders, sizes, n_cycles, s2c = find_cycles(self.basis)
            self.symmetry_info = {
                'cycle_leaders': leaders,
                'cycle_sizes': sizes,
                'num_cycles': n_cycles,
                'state_to_cycle': s2c,
            }

        # Cache for the Hamiltonian matrix (built on first request)
        self._hamiltonian_matrix = None

    @property
    def dim(self) -> int:
        """Hilbert space dimension (reduced if symmetry is active)."""
        if self.use_symmetry:
            return self.symmetry_info['num_cycles']
        return self.basis.dim

    @property
    def full_dim(self) -> int:
        """Full Hilbert space dimension (always the unsymmetrized value)."""
        return self.basis.dim

    def _get_neighbor_pairs(self):
        """Generate nearest-neighbor bond pairs (site_i, site_j).

        For a 1D chain of L sites:
            OBC: bonds (0,1), (1,2), ..., (L-2, L-1)
            PBC: same as OBC plus the wrap-around bond (L-1, 0)
                 (excluded for L=2 to avoid double-counting the single bond)
        """
        bonds = [(site, site + 1) for site in range(self.num_sites - 1)]

        if self.boundary == 'pbc' and self.num_sites > 2:
            bonds.append((self.num_sites - 1, 0))

        return bonds

    def hamiltonian(self) -> sparse.csr_matrix:
        """Build and return the sparse Hamiltonian matrix.

        If use_symmetry=True, returns the symmetry-reduced Hamiltonian
        in the q=0, R=+1 sector (smaller dimension). Otherwise returns
        the full Hamiltonian.

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

        # Sparse matrix entries: lists of (row, col, value)
        row_indices = []
        col_indices = []
        matrix_elements = []

        for state_index in range(hilbert_dim):
            occupation = basis.get_state(state_index)

            # =============================================================
            # Diagonal part: on-site interaction + chemical potential
            #
            #   H_diag = (U/2) sum_i n_i(n_i - 1) - mu sum_i n_i
            # =============================================================
            diagonal_energy = 0.0
            for site in range(self.num_sites):
                n_i = occupation[site]
                # On-site repulsion: U/2 * n_i * (n_i - 1)
                diagonal_energy += (self.interaction / 2.0) * n_i * (n_i - 1)
                # Chemical potential: -mu * n_i
                diagonal_energy -= self.chemical_potential * n_i

            if diagonal_energy != 0.0:
                row_indices.append(state_index)
                col_indices.append(state_index)
                matrix_elements.append(diagonal_energy)

            # =============================================================
            # Off-diagonal part: hopping between nearest neighbors
            #
            #   H_hop = -t sum_{<i,j>} (b†_i b_j + b†_j b_i)
            #
            # For each bond (site_i, site_j), we apply both:
            #   b†_i b_j : destroy a boson at j, create one at i
            #   b†_j b_i : destroy a boson at i, create one at j
            # =============================================================
            for site_i, site_j in self._get_neighbor_pairs():

                # --- b†_i b_j : hop a boson from site_j to site_i ---
                n_at_source = occupation[site_j]  # n_j before hopping
                n_at_target = occupation[site_i]   # n_i before hopping

                can_hop = (
                    n_at_source > 0  # must have a boson to destroy
                    and (self.max_occupation is None
                         or n_at_target < self.max_occupation)  # target not full
                )

                if can_hop:
                    new_occupation = list(occupation)
                    new_occupation[site_j] -= 1  # annihilate at source
                    new_occupation[site_i] += 1  # create at target

                    target_state_index = basis.get_index(tuple(new_occupation))

                    if target_state_index >= 0:
                        # Matrix element: -t * sqrt(n_j) * sqrt(n_i + 1)
                        # from the bosonic ladder operator algebra
                        hopping_element = -self.hopping * np.sqrt(
                            n_at_source * new_occupation[site_i]
                        )
                        row_indices.append(state_index)
                        col_indices.append(target_state_index)
                        matrix_elements.append(hopping_element)

                # --- b†_j b_i : hop a boson from site_i to site_j ---
                n_at_source = occupation[site_i]  # n_i before hopping
                n_at_target = occupation[site_j]   # n_j before hopping

                can_hop = (
                    n_at_source > 0
                    and (self.max_occupation is None
                         or n_at_target < self.max_occupation)
                )

                if can_hop:
                    new_occupation = list(occupation)
                    new_occupation[site_i] -= 1  # annihilate at source
                    new_occupation[site_j] += 1  # create at target

                    target_state_index = basis.get_index(tuple(new_occupation))

                    if target_state_index >= 0:
                        hopping_element = -self.hopping * np.sqrt(
                            n_at_source * new_occupation[site_j]
                        )
                        row_indices.append(state_index)
                        col_indices.append(target_state_index)
                        matrix_elements.append(hopping_element)

        # Assemble the sparse matrix
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
        """Build the symmetry-reduced Hamiltonian via the symmetry module."""
        from ..symmetry import build_reduced_hamiltonian

        info = self.symmetry_info
        self._hamiltonian_matrix = build_reduced_hamiltonian(
            basis=self.basis,
            cycle_leaders=info['cycle_leaders'],
            cycle_sizes=info['cycle_sizes'],
            num_cycles=info['num_cycles'],
            state_to_cycle=info['state_to_cycle'],
            hopping=self.hopping,
            interaction=self.interaction,
            chemical_potential=self.chemical_potential,
            neighbor_pairs=self._get_neighbor_pairs(),
        )
        return self._hamiltonian_matrix

    def reconstruct_wavefunction(self, d: np.ndarray) -> np.ndarray:
        """Reconstruct the full wavefunction from a symmetry-reduced eigenvector.

        Only available when use_symmetry=True. Maps the reduced
        eigenvector (length num_cycles) to the full basis (length dim_full).

        Parameters
        ----------
        d : numpy.ndarray, shape (num_cycles,)
            Eigenvector in the reduced basis.

        Returns
        -------
        psi : numpy.ndarray, shape (full_dim,)
            Full wavefunction for computing observables.
        """
        if not self.use_symmetry or self.symmetry_info is None:
            raise RuntimeError(
                "reconstruct_wavefunction() requires use_symmetry=True"
            )

        from ..symmetry import reconstruct_wavefunction
        info = self.symmetry_info
        return reconstruct_wavefunction(
            d, self.basis, info['cycle_sizes'], info['state_to_cycle']
        )

    def __repr__(self):
        if self.total_particles is not None:
            ensemble_info = f"N={self.total_particles}"
        else:
            ensemble_info = "grand canonical"

        sym_info = ""
        if self.use_symmetry:
            sym_info = (
                f", symmetry=q0R1, "
                f"reduced_dim={self.dim}, full_dim={self.full_dim}"
            )

        return (
            f"BoseHubbard1D(num_sites={self.num_sites}, "
            f"t={self.hopping}, U={self.interaction}, "
            f"mu={self.chemical_potential}, "
            f"max_occupation={self.max_occupation}, "
            f"{ensemble_info}, boundary='{self.boundary}', "
            f"basis={type(self.basis).__name__}"
            f"{sym_info if self.use_symmetry else f', dim={self.dim}'}"
            f")"
        )
