"""
Extended Bose-Hubbard Model on the Kagome Lattice
=================================================

Hardcore / soft-core bosons on the kagome lattice, with the extended
interactions needed to reach a Z2 topological (spin-liquid) phase and study
its topological entanglement entropy (TEE).

Hamiltonian
-----------

    H = -t Σ_<ij> (b†_i b_j + h.c.)                         (nearest-neighbor hopping)
        + (U/2) Σ_i n_i(n_i - 1)                            (on-site; zero for hardcore)
        - μ Σ_i n_i                                         (chemical potential)
        + V1 Σ_{<ij>}   n_i n_j                             (pairwise 1st-neighbor)
        + V2 Σ_{<<ij>>} n_i n_j                             (pairwise 2nd-neighbor)
        + V3 Σ_{<<<ij>>>} n_i n_j                           (pairwise 3rd-neighbor)
        + W  Σ_hex (n_hex)²                                 (hexagon cluster charging)

Two literature routes to a Z2 topological liquid are both supported:

  Route A (Isakov-Hastings-Melko, Nat. Phys. 7, 772 (2011)):
      cluster-charging term W Σ_hex (n_hex)², half filling, W/t ≈ 8.
  Route B (Roychowdhury-Bhattacharjee-Pollmann, PRB 92, 075141 (2015)):
      pairwise V1, V2, V3, 1/3 filling.

For hardcore bosons n_i² = n_i, so the two are related:
      W (n_hex)² = W·n_hex + 2W Σ_{pairs in hexagon} n_i n_j
i.e. the cluster term is an extended pairwise interaction coupling every pair
sharing a hexagon (plus a filling-fixed constant). The class implements the
cluster term faithfully as W Σ_hex (n_hex)² (it is diagonal).

Geometry
--------
Kagome = triangular Bravais lattice (a1, a2) with a 3-site basis. An L×L cell
grid has 3L² sites. Neighbor shells (1st/2nd/3rd) are assigned by real-space
distance under the periodic minimum image, so the shell structure is derived,
not hand-wired. Hexagonal plaquettes are located geometrically.

    a1 = (1, 0),  a2 = (1/2, √3/2)
    sublattice offsets: A=(0,0), B=a1/2, C=a2/2
    site index: 3*(i + j*L) + s   for cell (i,j), sublattice s∈{0,1,2}

Notes
-----
- Symmetry reduction is NOT yet implemented for kagome (the square-lattice
  bitwise translations in symmetry_2d.py do not apply). use_symmetry=True
  raises NotImplementedError. Full ED only for now.
- On the kagome lattice the 3rd-neighbor shell is subtle (there are inequivalent
  "across-hexagon" vs "collinear" third neighbors at nearby distances); the
  shell assignment here is purely by distance. The validation script prints the
  shell distances and multiplicities so the mapping is explicit.
"""

import numpy as np
from scipy import sparse
from ..basis import FockBasis
from ..unary_basis import UnaryBasis
from typing import Optional


# =====================================================================
# Geometry helpers (module-level, independently testable)
# =====================================================================

def kagome_lattice_vectors():
    """Return the two Bravais vectors and the 3 sublattice offsets."""
    a1 = np.array([1.0, 0.0])
    a2 = np.array([0.5, np.sqrt(3.0) / 2.0])
    sub = np.array([[0.0, 0.0], a1 / 2.0, a2 / 2.0])
    return a1, a2, sub


def kagome_positions(L: int):
    """Real-space positions of all 3L² sites.

    Returns
    -------
    pos : ndarray (3L², 2)
        pos[site] = (x, y).  site = 3*(i + j*L) + s.
    """
    a1, a2, sub = kagome_lattice_vectors()
    pos = np.zeros((3 * L * L, 2))
    for j in range(L):
        for i in range(L):
            cell_origin = i * a1 + j * a2
            for s in range(3):
                site = 3 * (i + j * L) + s
                pos[site] = cell_origin + sub[s]
    return pos


def _min_image_distance(p, q, L):
    """Minimum-image distance between points p, q on the oblique L×L torus.

    Periods are L*a1 and L*a2. Images in {-1,0,1}² suffice for L≥2 and the
    short (≤ 3rd-neighbor) distances used here.
    """
    a1, a2, _ = kagome_lattice_vectors()
    A1 = L * a1
    A2 = L * a2
    delta0 = p - q
    best = np.inf
    for m in (-1, 0, 1):
        for n in (-1, 0, 1):
            d = delta0 - m * A1 - n * A2
            r = np.hypot(d[0], d[1])
            if r < best:
                best = r
    return best


def kagome_neighbor_shells(L: int, n_shells: int = 3, tol: float = 1e-6):
    """Bond lists for the first ``n_shells`` neighbor distances.

    Returns
    -------
    shells : list of list of (i, j)
        shells[k] holds the (i<j) site-pairs at the (k+1)-th smallest distance.
    dists : list of float
        The distance of each shell.
    """
    pos = kagome_positions(L)
    num = pos.shape[0]

    # collect all unique pair distances
    pair_d = {}
    for a in range(num):
        for b in range(a + 1, num):
            pair_d[(a, b)] = _min_image_distance(pos[a], pos[b], L)

    unique = sorted(set(round(d, 6) for d in pair_d.values()))
    shell_dists = unique[:n_shells]

    shells = [[] for _ in range(len(shell_dists))]
    for (a, b), d in pair_d.items():
        for k, sd in enumerate(shell_dists):
            if abs(d - sd) < tol:
                shells[k].append((a, b))
                break
    for s in shells:
        s.sort()
    return shells, shell_dists


def kagome_hexagons(L: int, tol: float = 1e-3):
    """Locate the hexagonal plaquettes.

    A kagome hexagon center is equidistant (at the NN distance = circumradius
    = 1/2) from exactly its 6 ring sites. We grid-search center candidates over
    the torus, keep points with exactly 6 sites at distance ≈ 1/2, cluster
    them, and read off the 6 sites per hexagon.

    Returns
    -------
    hexagons : list of tuple(6 ints)
        Sorted site indices for each hexagon (there are L² of them).
    """
    pos = kagome_positions(L)
    a1, a2, _ = kagome_lattice_vectors()
    nn = 0.5  # kagome NN distance = hexagon circumradius

    # grid search over fractional coordinates (u along a1, v along a2)
    steps = 12 * L
    candidates = []
    for iu in range(steps):
        u = iu / steps * L
        for iv in range(steps):
            v = iv / steps * L
            c = u * a1 + v * a2
            ring = [b for b in range(pos.shape[0])
                    if abs(_min_image_distance(c, pos[b], L) - nn) < 0.03]
            if len(ring) == 6:
                candidates.append((c, tuple(sorted(ring))))

    # dedup by the 6-site set
    seen = {}
    for c, ring in candidates:
        seen[ring] = c
    hexagons = sorted(seen.keys())
    return hexagons


# =====================================================================
# Model
# =====================================================================

class BoseHubbardKagome:
    """Extended Bose-Hubbard model on an L×L kagome lattice (3L² sites).

    Parameters
    ----------
    linear_size : int
        Number of unit cells per direction, L. Total sites = 3 L².
    hopping : float
        NN hopping amplitude t.
    interaction : float
        On-site U (zero-effect for hardcore since n(n-1)=0).
    chemical_potential : float
        μ.
    max_occupation, total_particles : int or None
        Occupation cap / fixed particle number (canonical if given).
    boundary : str
        'pbc' only (kagome torus). 'obc' not supported (raises).
    hardcore : bool
        If True, force max_occupation=1.
    nn_interaction : float
        V1, pairwise 1st-neighbor density-density (Route B).
    v2_interaction, v3_interaction : float
        V2, V3, pairwise 2nd / 3rd neighbor (Route B).
    cluster_charging : float
        W, hexagon cluster-charging W Σ_hex (n_hex)² (Route A).
    use_symmetry : bool
        Not implemented for kagome; True raises NotImplementedError.
    """

    def __init__(self, linear_size: int, hopping: float = 1.0,
                 interaction: float = 0.0, chemical_potential: float = 0.0,
                 max_occupation: Optional[int] = None,
                 total_particles: Optional[int] = None,
                 boundary: str = 'pbc',
                 hardcore: bool = False,
                 nn_interaction: float = 0.0,
                 v2_interaction: float = 0.0,
                 v3_interaction: float = 0.0,
                 cluster_charging: float = 0.0,
                 use_symmetry: bool = False):

        self.hardcore = hardcore
        self.nn_interaction = nn_interaction
        self.v2_interaction = v2_interaction
        self.v3_interaction = v3_interaction
        self.cluster_charging = cluster_charging

        if hardcore:
            if max_occupation is not None and max_occupation != 1:
                raise ValueError(
                    "hardcore=True requires max_occupation=1 "
                    f"(got max_occupation={max_occupation})"
                )
            max_occupation = 1

        self.linear_size = linear_size
        self.num_sites = 3 * linear_size * linear_size
        self.hopping = hopping
        self.interaction = interaction
        self.chemical_potential = chemical_potential
        self.max_occupation = max_occupation
        self.total_particles = total_particles
        self.boundary = boundary.lower()

        if self.boundary != 'pbc':
            raise ValueError(
                "BoseHubbardKagome supports boundary='pbc' only "
                f"(got '{boundary}')"
            )

        if use_symmetry:
            raise NotImplementedError(
                "Symmetry reduction is not yet implemented for the kagome "
                "lattice (square-lattice bitwise translations do not apply). "
                "Use use_symmetry=False (full ED)."
            )
        self.use_symmetry = False

        # --- geometry ---
        need_shells = 1
        if self.v2_interaction != 0.0:
            need_shells = max(need_shells, 2)
        if self.v3_interaction != 0.0:
            need_shells = max(need_shells, 3)
        self._shells, self._shell_dists = kagome_neighbor_shells(
            linear_size, n_shells=max(need_shells, 3)
        )
        self.nn_bonds = self._shells[0]
        self._hexagons = None  # lazily computed (only if cluster_charging != 0)

        # --- basis ---
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

        self._hamiltonian_matrix = None

    @property
    def dim(self) -> int:
        return self.basis.dim

    @property
    def full_dim(self) -> int:
        return self.basis.dim

    def positions(self) -> np.ndarray:
        """Real-space site positions (3L², 2)."""
        return kagome_positions(self.linear_size)

    def hexagons(self):
        """List of hexagon plaquettes (6 site indices each), computed lazily."""
        if self._hexagons is None:
            self._hexagons = kagome_hexagons(self.linear_size)
        return self._hexagons

    def _get_neighbor_pairs(self):
        """Nearest-neighbor bonds (for hopping)."""
        return self.nn_bonds

    def _v_terms(self):
        """Pairwise-V (coupling, bond-list) pairs actually in use."""
        v_terms = []
        if self.nn_interaction != 0.0:
            v_terms.append((self.nn_interaction, self._shells[0]))
        if self.v2_interaction != 0.0:
            v_terms.append((self.v2_interaction, self._shells[1]))
        if self.v3_interaction != 0.0:
            v_terms.append((self.v3_interaction, self._shells[2]))
        return v_terms

    def diagonal_vector(self, occ: np.ndarray) -> np.ndarray:
        """Vectorized interaction diagonal for an occupation array (dim, nsites).

        Diagonal energy of each basis state:
            (U/2) Σ n(n-1) − μ Σ n + Σ_shells V·n_i n_j + W Σ_hex (n_hex)²
        """
        occ = occ.astype(np.float64)
        dim = occ.shape[0]
        diag = np.zeros(dim)
        if self.interaction != 0.0:
            diag += (self.interaction / 2.0) * (occ * (occ - 1.0)).sum(axis=1)
        if self.chemical_potential != 0.0:
            diag -= self.chemical_potential * occ.sum(axis=1)
        for coupling, bonds in self._v_terms():
            if bonds:
                b = np.asarray(bonds)
                diag += coupling * (occ[:, b[:, 0]] * occ[:, b[:, 1]]).sum(axis=1)
        if self.cluster_charging != 0.0:
            for hexa in self.hexagons():
                nh = occ[:, list(hexa)].sum(axis=1)
                diag += self.cluster_charging * nh * nh
        return diag

    def _can_use_fast(self) -> bool:
        """Fast vectorized path: hardcore, canonical (UnaryBasis), nsites<=62."""
        return (self.hardcore
                and isinstance(self.basis, UnaryBasis)
                and self.num_sites <= 62)

    def _hamiltonian_fast(self) -> sparse.csr_matrix:
        """Vectorized (numpy, no-numba) hardcore Hamiltonian build.

        Each hardcore Fock state maps to a unique bitmask Σ_i n_i 2^i. Hopping
        targets are computed by bit arithmetic and located with a single
        vectorized np.searchsorted over the sorted bitmasks — replacing the
        per-state Python loop. Produces the SAME matrix (same basis ordering)
        as the reference builder.
        """
        basis = self.basis
        dim = basis.dim
        nsite = self.num_sites

        occ = basis.all_states_as_array().astype(np.int64)  # (dim, nsite)
        powers = (np.int64(1) << np.arange(nsite, dtype=np.int64))
        bitmask = occ @ powers                              # unique per state
        order = np.argsort(bitmask, kind='stable')
        sorted_bm = bitmask[order]

        rows, cols, vals = [], [], []
        for (i, j) in self.nn_bonds:
            # single direction: move a particle j -> i (j occupied, i empty);
            # the reverse hop is added by the transpose below.
            src = np.nonzero((occ[:, j] == 1) & (occ[:, i] == 0))[0]
            if src.size == 0:
                continue
            tbm = bitmask[src] - (np.int64(1) << np.int64(j)) \
                               + (np.int64(1) << np.int64(i))
            p = np.searchsorted(sorted_bm, tbm)
            tgt = order[p]
            # every target lives in the same particle-number sector, so it
            # must be present; assert to catch any lookup error.
            assert np.all(sorted_bm[p] == tbm), "hop target missing from basis"
            rows.append(tgt); cols.append(src)
            vals.append(np.full(src.size, -self.hopping))

        if rows:
            rows = np.concatenate(rows); cols = np.concatenate(cols)
            vals = np.concatenate(vals)
        else:
            rows = np.empty(0, int); cols = np.empty(0, int); vals = np.empty(0)

        H_off = sparse.csr_matrix((vals, (rows, cols)), shape=(dim, dim),
                                  dtype=np.float64)
        H_off = H_off + H_off.T                     # add the reverse hops
        H = (H_off + sparse.diags(self.diagonal_vector(occ))).tocsr()
        H.eliminate_zeros()
        self._hamiltonian_matrix = H
        return H

    def hamiltonian(self, force_reference: bool = False) -> sparse.csr_matrix:
        """Build and return the full sparse Hamiltonian.

        Uses the vectorized fast path for hardcore/canonical systems; set
        ``force_reference=True`` to force the plain-Python reference builder
        (used to validate the fast path).
        """
        if self._hamiltonian_matrix is not None:
            return self._hamiltonian_matrix

        if self._can_use_fast() and not force_reference:
            return self._hamiltonian_fast()

        basis = self.basis
        hilbert_dim = basis.dim

        # precompute the pairwise-V bond lists actually in use
        v_terms = []
        if self.nn_interaction != 0.0:
            v_terms.append((self.nn_interaction, self._shells[0]))
        if self.v2_interaction != 0.0:
            v_terms.append((self.v2_interaction, self._shells[1]))
        if self.v3_interaction != 0.0:
            v_terms.append((self.v3_interaction, self._shells[2]))

        use_cluster = (self.cluster_charging != 0.0)
        hexagons = self.hexagons() if use_cluster else []

        row_indices = []
        col_indices = []
        matrix_elements = []

        for state_index in range(hilbert_dim):
            occupation = basis.get_state(state_index)

            # ---- diagonal ----
            diagonal_energy = 0.0
            for site in range(self.num_sites):
                n_i = occupation[site]
                diagonal_energy += (self.interaction / 2.0) * n_i * (n_i - 1)
                diagonal_energy -= self.chemical_potential * n_i

            for coupling, bonds in v_terms:
                for site_i, site_j in bonds:
                    diagonal_energy += coupling * occupation[site_i] * occupation[site_j]

            if use_cluster:
                for hexa in hexagons:
                    n_hex = 0
                    for site in hexa:
                        n_hex += occupation[site]
                    diagonal_energy += self.cluster_charging * n_hex * n_hex

            if diagonal_energy != 0.0:
                row_indices.append(state_index)
                col_indices.append(state_index)
                matrix_elements.append(diagonal_energy)

            # ---- off-diagonal: NN hopping ----
            for site_i, site_j in self.nn_bonds:
                # b†_i b_j : hop from j to i
                n_source = occupation[site_j]
                n_target = occupation[site_i]
                if (n_source > 0
                        and (self.max_occupation is None
                             or n_target < self.max_occupation)):
                    new_occ = list(occupation)
                    new_occ[site_j] -= 1
                    new_occ[site_i] += 1
                    t_idx = basis.get_index(tuple(new_occ))
                    if t_idx >= 0:
                        elem = -self.hopping * np.sqrt(n_source * new_occ[site_i])
                        row_indices.append(state_index)
                        col_indices.append(t_idx)
                        matrix_elements.append(elem)

                # b†_j b_i : hop from i to j
                n_source = occupation[site_i]
                n_target = occupation[site_j]
                if (n_source > 0
                        and (self.max_occupation is None
                             or n_target < self.max_occupation)):
                    new_occ = list(occupation)
                    new_occ[site_i] -= 1
                    new_occ[site_j] += 1
                    t_idx = basis.get_index(tuple(new_occ))
                    if t_idx >= 0:
                        elem = -self.hopping * np.sqrt(n_source * new_occ[site_j])
                        row_indices.append(state_index)
                        col_indices.append(t_idx)
                        matrix_elements.append(elem)

        H = sparse.csr_matrix(
            (matrix_elements, (row_indices, col_indices)),
            shape=(hilbert_dim, hilbert_dim),
            dtype=np.float64,
        )
        H = (H + H.T) / 2.0
        H.eliminate_zeros()
        self._hamiltonian_matrix = H
        return H

    def single_particle_hopping_matrix(self) -> np.ndarray:
        """Dense 3L²×3L² tight-binding matrix M_ij = -t on NN bonds.

        Useful for the kagome flat-band check (independent of the many-body
        basis).
        """
        n = self.num_sites
        M = np.zeros((n, n))
        for i, j in self.nn_bonds:
            M[i, j] += -self.hopping
            M[j, i] += -self.hopping
        return M

    def __repr__(self):
        L = self.linear_size
        ens = (f"N={self.total_particles}" if self.total_particles is not None
               else "grand canonical")
        hc = ", hardcore=True" if self.hardcore else ""
        vs = []
        if self.nn_interaction != 0.0:
            vs.append(f"V1={self.nn_interaction}")
        if self.v2_interaction != 0.0:
            vs.append(f"V2={self.v2_interaction}")
        if self.v3_interaction != 0.0:
            vs.append(f"V3={self.v3_interaction}")
        if self.cluster_charging != 0.0:
            vs.append(f"W={self.cluster_charging}")
        vstr = (", " + ", ".join(vs)) if vs else ""
        return (f"BoseHubbardKagome(L={L}, sites={self.num_sites}, "
                f"t={self.hopping}, U={self.interaction}{vstr}{hc}, "
                f"{ens}, boundary='{self.boundary}', dim={self.dim})")
