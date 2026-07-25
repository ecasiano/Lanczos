"""
Fractional Chern Insulator on the Kagome Lattice
=================================================

Spinless fermions on the kagome tight-binding model with complex
nearest-neighbor hopping t·e^{±iφ}.  At φ = 5π/4 the lowest of the
three bands has Chern number C = 1 and relatively flat dispersion.

Projecting a nearest-neighbor density-density interaction V into
the lowest Chern band and filling at ν = 1/3 produces a lattice
analog of the ν = 1/3 Laughlin state — a Fractional Chern Insulator.

Properties of the FCI ground state:
    - 3-fold topological ground-state degeneracy on the torus
    - Topological entanglement entropy  γ = ½ ln 3
    - Fractional excitations (anyons)

Band projection reduces the Hilbert space from C(3N, N_p) to
C(N, N_p), where N = N₁ × N₂ is the number of unit cells and
N_p = N/3 is the particle count.  Typical sizes:

    N₁×N₂ = 3×4  →  N=12, N_p=4, dim = C(12,4) = 495
    N₁×N₂ = 3×5  →  N=15, N_p=5, dim = C(15,5) = 3003
    N₁×N₂ = 4×6  →  N=24, N_p=8, dim = C(24,8) = 735471

Kagome lattice geometry:
    Bravais vectors  a₁ = (1, 0),  a₂ = (1/2, √3/2)
    Sublattices      A = (0, 0),  B = (1/2, 0),  C = (1/4, √3/4)

Bloch Hamiltonian (Eq. 6 of the tutorial):

    h(k) = t  ⎡  0                e^{-iφ}(1+e^{-ik₁})    e^{iφ}(1+e^{-ik₂})      ⎤
               ⎢  e^{iφ}(1+e^{ik₁})     0                e^{-iφ}(1+e^{i(k₁-k₂)})  ⎥
               ⎣  e^{-iφ}(1+e^{ik₂})   e^{iφ}(1+e^{i(k₂-k₁)})   0                ⎦

    with k = (k̃₁/N₁)b₁ + (k̃₂/N₂)b₂  and  k₁ = 2π k̃₁/N₁,  k₂ = 2π k̃₂/N₂.

Band-projected interaction (Eq. 22-23 of the tutorial):

    H_int = (1/2N) Σ_{k₁k₂k₃k₄} δ_{k₄=k₁+k₂-k₃} U_{k₁k₂k₃k₄}
            d†_{k₁} d†_{k₂} d_{k₄} d_{k₃}

    U_{k₁k₂k₃k₄} = Σ_{αβ} u*_α(k₁) u*_β(k₂) u_α(k₃) u_β(k₄) V^{αβ}(k₃-k₁)

References:
    Regnault & Bernevig, Phys. Rev. X 1, 021014 (2011)
    Kwan & Regnault, FCI ED Tutorial (DIPC 2024)
    Tang, Mei & Wen, PRL 106, 236802 (2011)
"""

import numpy as np
from scipy import sparse
from scipy.sparse.linalg import eigsh
from math import comb
from itertools import combinations


# =====================================================================
# Kagome geometry helpers
# =====================================================================

# Sublattice positions within one unit cell (in units of lattice constant a=1)
SUBLATTICE_POS = np.array([
    [0.0,         0.0],           # A
    [0.5,         0.0],           # B
    [0.25, np.sqrt(3)/4],         # C
])

# Bravais lattice vectors
A1 = np.array([1.0, 0.0])
A2 = np.array([0.5, np.sqrt(3)/2])


def kagome_site_positions(N1, N2):
    """Real-space positions of all 3·N₁·N₂ sites on the kagome lattice.

    Returns
    -------
    positions : ndarray, shape (3*N1*N2, 2)
        Cartesian coordinates.  Site index convention:
        site = alpha + 3*(R2*N1 + R1)
        where alpha ∈ {0,1,2} is the sublattice and
        R = R1*a₁ + R2*a₂ with R1 ∈ [0,N1), R2 ∈ [0,N2).
    """
    positions = []
    for R2 in range(N2):
        for R1 in range(N1):
            R = R1 * A1 + R2 * A2
            for alpha in range(3):
                positions.append(R + SUBLATTICE_POS[alpha])
    return np.array(positions)


# =====================================================================
# Single-particle band structure
# =====================================================================

def bloch_hamiltonian(k1, k2, phi, t=1.0):
    """3×3 Bloch Hamiltonian h(k) for the kagome lattice.

    Parameters
    ----------
    k1, k2 : float
        Crystal momenta k₁ = 2π k̃₁/N₁,  k₂ = 2π k̃₂/N₂.
    phi : float
        Hopping phase φ in t·e^{±iφ}.
    t : float
        Hopping amplitude (default 1).

    Returns
    -------
    h : ndarray, shape (3, 3), complex
    """
    h = np.zeros((3, 3), dtype=complex)

    # A-B bonds: within cell (R=0) and across a₁  →  factor (1 + e^{-ik₁})
    h[0, 1] = t * np.exp(-1j * phi) * (1.0 + np.exp(-1j * k1))
    h[1, 0] = np.conj(h[0, 1])

    # A-C bonds: within cell and across a₂  →  factor (1 + e^{-ik₂})
    h[0, 2] = t * np.exp(1j * phi) * (1.0 + np.exp(-1j * k2))
    h[2, 0] = np.conj(h[0, 2])

    # B-C bonds: within cell and across a₁-a₂  →  factor (1 + e^{i(k₁-k₂)})
    h[1, 2] = t * np.exp(-1j * phi) * (1.0 + np.exp(1j * (k1 - k2)))
    h[2, 1] = np.conj(h[1, 2])

    return h


def compute_bands(N1, N2, phi, t=1.0):
    """Diagonalize h(k) at every k-point on the N₁×N₂ mesh.

    Returns
    -------
    energies : ndarray, shape (N, 3)
        Band energies sorted by increasing energy.  Index K labels
        the linearized momentum K = k̃₁·N₂ + k̃₂.
    eigvecs : ndarray, shape (N, 3, 3), complex
        eigvecs[K, :, n] = Bloch eigenvector u_{n,α}(K).
        Columns are ordered by increasing energy (n = 0 is lowest band).
    """
    N = N1 * N2
    energies = np.zeros((N, 3))
    eigvecs = np.zeros((N, 3, 3), dtype=complex)

    for K in range(N):
        kt1 = K // N2          # k̃₁
        kt2 = K % N2           # k̃₂
        k1 = 2.0 * np.pi * kt1 / N1
        k2 = 2.0 * np.pi * kt2 / N2

        h = bloch_hamiltonian(k1, k2, phi, t)
        evals, evecs = np.linalg.eigh(h)  # sorted ascending
        energies[K] = evals
        eigvecs[K] = evecs

    return energies, eigvecs


def chern_number(N1, N2, eigvecs, band=0):
    """Chern number via discretized Berry curvature (Fukui et al. 2005).

    Uses link variables on the momentum-space lattice to compute the
    gauge-invariant lattice field strength F̃(k), whose sum gives C.

    Parameters
    ----------
    N1, N2 : int
        Torus dimensions.
    eigvecs : ndarray, shape (N, 3, 3)
        Bloch eigenvectors from compute_bands.
    band : int
        Band index (0 = lowest).

    Returns
    -------
    C : float
        Chern number (integer to numerical precision).
    """
    def K_idx(kt1, kt2):
        return (kt1 % N1) * N2 + (kt2 % N2)

    def u(kt1, kt2):
        return eigvecs[K_idx(kt1, kt2), :, band]

    def link(kt1a, kt2a, kt1b, kt2b):
        """Normalized overlap  U_μ = ⟨u(a)|u(b)⟩ / |⟨u(a)|u(b)⟩|."""
        ov = np.vdot(u(kt1a, kt2a), u(kt1b, kt2b))
        return ov / abs(ov)

    total_F = 0.0
    for kt1 in range(N1):
        for kt2 in range(N2):
            # Plaquette product: U₁(k)·U₂(k+d₁)·U₁(k+d₂)⁻¹·U₂(k)⁻¹
            # U⁻¹ = U* since |U| = 1
            U1 = link(kt1, kt2,   kt1+1, kt2)      # along d₁
            U2 = link(kt1+1, kt2, kt1+1, kt2+1)    # along d₂ at k+d₁
            U1i = np.conj(link(kt1, kt2+1, kt1+1, kt2+1))  # U₁(k+d₂)⁻¹
            U2i = np.conj(link(kt1, kt2,   kt1, kt2+1))    # U₂(k)⁻¹

            plaquette = U1 * U2 * U1i * U2i
            total_F += np.angle(plaquette)     # F̃(k) ∈ (-π, π]

    return total_F / (2.0 * np.pi)


# =====================================================================
# Projected interaction matrix elements
# =====================================================================

def nn_interaction_fourier(q1, q2):
    """Fourier-transformed NN interaction V^{αβ}(q) for the kagome lattice.

    For nearest-neighbor density-density interaction V=1, the nonzero
    sublattice pairs and their Fourier transforms are:

        V^{AB}(q) = 1 + e^{iq₁}           (A-B bonds span a₁ direction)
        V^{AC}(q) = 1 + e^{iq₂}           (A-C bonds span a₂ direction)
        V^{BC}(q) = 1 + e^{-i(q₁-q₂)}    (B-C bonds span a₁-a₂ direction)

    Same-sublattice terms are zero (no NN pair connects same sublattice
    on kagome).

    Parameters
    ----------
    q1, q2 : float
        Momentum transfer  q = k₃ - k₁  in Bloch momentum units.

    Returns
    -------
    Vq : ndarray, shape (3, 3), complex
        V^{αβ}(q).
    """
    Vq = np.zeros((3, 3), dtype=complex)

    Vq[0, 1] = 1.0 + np.exp(1j * q1)             # AB
    Vq[1, 0] = 1.0 + np.exp(-1j * q1)            # BA = AB(-q)*

    Vq[0, 2] = 1.0 + np.exp(1j * q2)             # AC
    Vq[2, 0] = 1.0 + np.exp(-1j * q2)            # CA

    Vq[1, 2] = 1.0 + np.exp(-1j * (q1 - q2))    # BC
    Vq[2, 1] = 1.0 + np.exp(1j * (q1 - q2))     # CB

    return Vq


def compute_interaction_matrix_elements(N1, N2, eigvecs, band=0, V=1.0):
    """Compute the band-projected interaction U_{k₁k₂k₃k₄}.

    From the tutorial (Eq. 23):
        U_{k₁k₂k₃k₄} = Σ_{αβ} u*_α(k₁) u*_β(k₂) u_α(k₃) u_β(k₄)
                         × V^{αβ}(k₃-k₁)

    Only on-shell elements (k₄ = k₁+k₂-k₃ mod G) are computed.

    Parameters
    ----------
    N1, N2 : int
        Torus dimensions.
    eigvecs : ndarray, shape (N, 3, 3)
        Bloch eigenvectors from compute_bands.
    band : int
        Which band to project into (0 = lowest).
    V : float
        Interaction strength.

    Returns
    -------
    U : dict
        Maps (K1, K2, K3) → U_{k₁k₂k₃k₄} (complex).
        K4 is determined by momentum conservation.
    """
    N = N1 * N2

    # Extract band eigenvectors: u[K, alpha] = eigvecs[K, alpha, band]
    u = eigvecs[:, :, band]  # shape (N, 3)

    # Momentum mesh:  K = k̃₁·N₂ + k̃₂
    def k_components(K):
        return K // N2, K % N2   # (k̃₁, k̃₂)

    def k_angles(K):
        kt1, kt2 = k_components(K)
        return 2.0 * np.pi * kt1 / N1, 2.0 * np.pi * kt2 / N2

    def k4_from_conservation(K1, K2, K3):
        """k₄ = k₁ + k₂ - k₃  mod reciprocal lattice."""
        kt1_1, kt2_1 = k_components(K1)
        kt1_2, kt2_2 = k_components(K2)
        kt1_3, kt2_3 = k_components(K3)
        kt1_4 = (kt1_1 + kt1_2 - kt1_3) % N1
        kt2_4 = (kt2_1 + kt2_2 - kt2_3) % N2
        return kt1_4 * N2 + kt2_4

    Udict = {}

    for K1 in range(N):
        k1_a, k1_b = k_angles(K1)
        u1 = u[K1]   # shape (3,)

        for K2 in range(N):
            u2 = u[K2]

            for K3 in range(N):
                K4 = k4_from_conservation(K1, K2, K3)
                k3_a, k3_b = k_angles(K3)
                u3 = u[K3]
                u4 = u[K4]

                # Momentum transfer q = k₃ - k₁
                q1 = k3_a - k1_a
                q2 = k3_b - k1_b

                # Fourier-transformed NN interaction
                Vq = nn_interaction_fourier(q1, q2)  # shape (3,3)

                # U = Σ_{αβ} u*_α(k1) u*_β(k2) u_α(k3) u_β(k4) V^{αβ}(q)
                #   = Σ_{αβ} conj(u1[α]) conj(u2[β]) u3[α] u4[β] Vq[α,β]
                #   = (u1*.u3) ⊗ (u2*.u4) contracted with Vq
                # More efficiently: U = Σ_α u1*[α] u3[α] Σ_β u2*[β] u4[β] Vq[α,β]
                #                     = Σ_α (u1*u3)[α] · (Vq @ (u2*u4))[α]

                form_13 = np.conj(u1) * u3       # shape (3,)
                form_24 = np.conj(u2) * u4       # shape (3,)
                val = V * np.dot(form_13, Vq @ form_24)

                if abs(val) > 1e-15:
                    Udict[(K1, K2, K3)] = val

    return Udict


# =====================================================================
# Many-body Fock basis (fermionic, one band)
# =====================================================================

class FermionicBandBasis:
    """Fock basis for N_p spinless fermions in N_orb single-particle
    orbitals (one per k-point in the projected band).

    Each basis state is a sorted tuple of occupied orbital indices K.
    States are enumerated in lexicographic order of combinations.

    Attributes
    ----------
    N_orb : int
        Number of orbitals (= number of unit cells N).
    N_particles : int
        Number of particles.
    dimension : int
        Hilbert space dimension = C(N_orb, N_particles).
    """

    def __init__(self, N_orb, N_particles):
        self.N_orb = N_orb
        self.N_particles = N_particles
        self.num_sites = N_orb   # for interface compatibility
        self.dimension = comb(N_orb, N_particles)

        # Enumerate all states as sorted tuples of occupied K values
        self._states = list(combinations(range(N_orb), N_particles))
        self._index_map = {s: i for i, s in enumerate(self._states)}

        assert len(self._states) == self.dimension

    def get_state(self, idx):
        """Return occupation-number tuple (n_0, n_1, ..., n_{N_orb-1}).

        Each n_K ∈ {0, 1}.
        """
        occ = [0] * self.N_orb
        for K in self._states[idx]:
            occ[K] = 1
        return tuple(occ)

    def get_occupied(self, idx):
        """Return sorted tuple of occupied orbital indices."""
        return self._states[idx]

    def get_index(self, state_tuple):
        """Return basis index for a sorted tuple of occupied orbitals."""
        return self._index_map.get(state_tuple, -1)


# =====================================================================
# Many-body Hamiltonian builder
# =====================================================================

def _raw_U(K1, K2, K3, K4, u, N1, N2, V):
    """Single projected interaction matrix element U_{K1,K2,K3,K4}.

    U = Σ_{αβ} u*_α(K1) u*_β(K2) u_α(K3) u_β(K4) V^{αβ}(q)
    where q₁ = 2π(k̃3₁ - k̃1₁)/N₁,  q₂ = 2π(k̃3₂ - k̃1₂)/N₂.
    """
    q1 = 2.0 * np.pi * (K3 // N2 - K1 // N2) / N1
    q2 = 2.0 * np.pi * (K3 % N2 - K1 % N2) / N2
    Vq = nn_interaction_fourier(q1, q2)
    form_13 = np.conj(u[K1]) * u[K3]
    form_24 = np.conj(u[K2]) * u[K4]
    return V * np.dot(form_13, Vq @ form_24)


def _k4_from(K1, K2, K3, N1, N2):
    """Momentum conservation: K4 = K1 + K2 - K3 (mod reciprocal lattice)."""
    kt1 = (K1 // N2 + K2 // N2 - K3 // N2) % N1
    kt2 = (K1 % N2 + K2 % N2 - K3 % N2) % N2
    return kt1 * N2 + kt2


def build_hamiltonian(N1, N2, eigvecs, band_energies, band=0,
                      V=1.0, kappa=0.0):
    """Build the band-projected many-body Hamiltonian as a sparse matrix.

    H = κ·H₀ + H_int

    H₀ = Σ_k ε(k) d†_k d_k                     (single-particle dispersion)

    H_int = (1/N) Σ_{k1<k2, k3<k4}              (Eq. 32 of Kwan-Regnault)
            δ_{k4=k1+k2-k3}  Ũ_{k1k2k3k4}
            d†_{k1} d†_{k2} d_{k4} d_{k3}

    Ũ = U_{k1k2k3k4} - U_{k1k2k4k3}            (antisymmetrized, Eq. 33)

    Parameters
    ----------
    N1, N2 : int
        Torus dimensions.
    eigvecs : ndarray, shape (N, 3, 3)
        Bloch eigenvectors.
    band_energies : ndarray, shape (N,)
        Single-particle energies of the target band.
    band : int
        Target band index (0 = lowest).
    V : float
        NN interaction strength.
    kappa : float
        Band dispersion weight (0 = flat band limit).

    Returns
    -------
    H : sparse.csr_matrix
        Many-body Hamiltonian in the projected Fock space.
    basis : FermionicBandBasis
        The Fock basis used.
    """
    import bisect

    N = N1 * N2
    N_p = N // 3   # ν = 1/3 filling

    basis = FermionicBandBasis(N, N_p)
    dim = basis.dimension
    u = eigvecs[:, :, band]   # shape (N, 3)

    print(f"  FCI Hamiltonian: N={N}, N_p={N_p}, dim={dim}")

    # Build Hamiltonian as dense matrix (dim ≤ ~3000 for target systems)
    H = np.zeros((dim, dim), dtype=complex)

    for idx in range(dim):
        state = basis.get_occupied(idx)

        # --- Diagonal: single-particle dispersion (if κ > 0) ---
        if abs(kappa) > 1e-15:
            H[idx, idx] += kappa * sum(band_energies[K] for K in state)

        # --- Interaction: loop over ordered annihilation pairs ---
        # Annihilate (Ka1, Ka2) with Ka1 < Ka2 from |state⟩
        for ia in range(N_p):
            Ka1 = state[ia]
            for ib in range(ia + 1, N_p):
                Ka2 = state[ib]

                # Intermediate state after removing Ka1, Ka2
                intermediate = [K for K in state
                                if K != Ka1 and K != Ka2]
                inter_set = set(intermediate)

                # Fermionic sign for  d_{Ka2} d_{Ka1} |state⟩
                # Ka1 is at position ia, Ka2 at position ib in the sorted state.
                # Removing Ka1 first:  sign_1 = (-1)^ia
                # Then Ka2 in the shortened list: sign_2 = (-1)^(ib-1)
                sign_ann = (-1) ** ia * (-1) ** (ib - 1)

                # Loop over creation pairs Kc1 < Kc2 with
                # Kc1 + Kc2 ≡ Ka1 + Ka2 (mod G)
                for Kc1 in range(N):
                    if Kc1 in inter_set:
                        continue
                    Kc2 = _k4_from(Ka1, Ka2, Kc1, N1, N2)
                    if Kc2 <= Kc1 or Kc2 in inter_set:
                        continue

                    # Antisymmetrized matrix element (Eq. 33):
                    # Ũ = U_{Kc1,Kc2,Ka1,Ka2} - U_{Kc1,Kc2,Ka2,Ka1}
                    U_dir = _raw_U(Kc1, Kc2, Ka1, Ka2, u, N1, N2, V)
                    U_exc = _raw_U(Kc1, Kc2, Ka2, Ka1, u, N1, N2, V)
                    U_a = U_dir - U_exc
                    if abs(U_a) < 1e-15:
                        continue

                    # Fermionic sign for  d†_{Kc1} d†_{Kc2} |intermediate⟩
                    # Insert Kc2 first (in sorted position), then Kc1
                    pos_c2 = bisect.bisect_left(intermediate, Kc2)
                    sign_c2 = (-1) ** pos_c2
                    new_list = list(intermediate)
                    new_list.insert(pos_c2, Kc2)

                    pos_c1 = bisect.bisect_left(new_list, Kc1)
                    sign_c1 = (-1) ** pos_c1
                    new_list.insert(pos_c1, Kc1)

                    total_sign = sign_ann * sign_c1 * sign_c2

                    # Final state index
                    final = tuple(sorted(new_list))
                    jdx = basis.get_index(final)
                    if jdx < 0:
                        continue

                    H[jdx, idx] += total_sign * U_a / N

    # Hermitianize (should already be Hermitian up to numerics)
    H = 0.5 * (H + H.conj().T)

    return sparse.csr_matrix(H), basis


# =====================================================================
# Real-space transformation
# =====================================================================

def single_particle_transform(N1, N2, eigvecs, band=0):
    """Single-particle transformation matrix  T_{i,K} = φ_i(K).

    Maps band orbital K to real-space site i:
        d†_K = Σ_i T_{iK} c†_i

    where  T_{iK} = u_{band,α}(K) · e^{iK·R} / √N
    for site i = (R, α).

    Parameters
    ----------
    N1, N2 : int
        Torus dimensions.
    eigvecs : ndarray, shape (N, 3, 3)
        Bloch eigenvectors.
    band : int
        Band index.

    Returns
    -------
    T : ndarray, shape (N_sites, N_orb), complex
        Transformation matrix.
    """
    N = N1 * N2
    N_sites = 3 * N
    u = eigvecs[:, :, band]   # shape (N, 3)

    T = np.zeros((N_sites, N), dtype=complex)

    for K in range(N):
        kt1 = K // N2
        kt2 = K % N2
        # k = kt1·(2π/N1)·(1,0) + kt2·(2π/N2)·(...)  in reciprocal space
        # k·R = kt1·(2π/N1)·R1 + kt2·(2π/N2)·R2  (since aᵢ·bⱼ = 2π δᵢⱼ)
        # Wait, actually k = (kt1/N1)b₁ + (kt2/N2)b₂, so
        # k·R = (kt1/N1)(b₁·R) + (kt2/N2)(b₂·R)
        # With R = R1·a₁ + R2·a₂ and aᵢ·bⱼ = 2π δᵢⱼ:
        # k·R = 2π(kt1·R1/N1 + kt2·R2/N2)

        for R2 in range(N2):
            for R1 in range(N1):
                phase = np.exp(2j * np.pi * (kt1 * R1 / N1 + kt2 * R2 / N2))
                cell_idx = R2 * N1 + R1   # unit cell index

                for alpha in range(3):
                    site = alpha + 3 * cell_idx
                    T[site, K] = u[K, alpha] * phase / np.sqrt(N)

    return T


def transform_to_real_space(psi_k, k_basis, T, N_sites):
    """Transform a many-body state from band-projected k-space to real space.

    Given |ψ⟩ = Σ_α c_α d†_{K₁}...d†_{K_{Np}} |vac⟩  in the band basis,
    compute the wavefunction in the real-space Fock basis
    {c†_{i₁}...c†_{i_{Np}} |vac⟩} with i₁ < ... < i_{Np}.

    The transformation uses Slater determinants:
        ⟨i₁...i_{Np} | K₁...K_{Np}⟩ = det[T_{iₐ,Kᵦ}]

    Parameters
    ----------
    psi_k : ndarray, shape (dim_k,)
        Ground state in the band-projected Fock basis.
    k_basis : FermionicBandBasis
        k-space Fock basis.
    T : ndarray, shape (N_sites, N_orb), complex
        Single-particle transformation matrix.
    N_sites : int
        Number of real-space sites.

    Returns
    -------
    psi_real : ndarray
        Wavefunction in the real-space Fock basis (dense).
    real_states : list of tuple
        Sorted list of real-space Fock states (tuples of occupied sites).
    """
    N_p = k_basis.N_particles

    # Enumerate real-space Fock states: all C(N_sites, N_p) configurations
    real_states = list(combinations(range(N_sites), N_p))
    dim_real = len(real_states)
    real_index = {s: i for i, s in enumerate(real_states)}

    print(f"  Real-space transformation: {k_basis.dimension} k-states "
          f"→ {dim_real} real-space states")

    psi_real = np.zeros(dim_real, dtype=complex)

    # For each k-space Fock state with nonzero amplitude
    for idx_k in range(k_basis.dimension):
        c_alpha = psi_k[idx_k]
        if abs(c_alpha) < 1e-15:
            continue

        K_occ = k_basis.get_occupied(idx_k)   # sorted tuple of occupied K's

        # Build the submatrix T[sites, K_occ] for all possible real-space states
        # T_sub[a, b] = T[i_a, K_b]  →  Slater determinant for each real-space state
        T_cols = T[:, K_occ]  # shape (N_sites, N_p)

        # For each real-space configuration (i₁, ..., i_{Np}):
        # amplitude += c_alpha * det(T_cols[sites, :])
        for idx_r, sites in enumerate(real_states):
            T_sub = T_cols[sites, :]     # shape (N_p, N_p)
            det_val = np.linalg.det(T_sub)
            psi_real[idx_r] += c_alpha * det_val

    # Normalize (should already be normalized, but numerical safety)
    norm = np.sqrt(np.sum(np.abs(psi_real) ** 2))
    if norm > 1e-10:
        psi_real /= norm

    return psi_real, real_states


# =====================================================================
# Adapter: real-space basis for observable compatibility
# =====================================================================

class RealSpaceFermionBasis:
    """Minimal basis adapter for real-space fermionic Fock states.

    Provides the interface expected by the observable functions
    (entanglement_entropy, accessible_entanglement_entropy, etc.):
        - num_sites : int
        - dimension : int
        - get_state(k) → tuple of occupations (n₀, n₁, ...)
    """

    def __init__(self, N_sites, states_list):
        """
        Parameters
        ----------
        N_sites : int
            Total number of real-space sites.
        states_list : list of tuple
            Each element is a sorted tuple of occupied site indices.
        """
        self.num_sites = N_sites
        self._states = states_list
        self.dimension = len(states_list)
        self._index_map = {s: i for i, s in enumerate(states_list)}

    def get_state(self, idx):
        """Return occupation-number tuple (n₀, n₁, ..., n_{N-1})."""
        occ = [0] * self.num_sites
        for site in self._states[idx]:
            occ[site] = 1
        return tuple(occ)

    def get_index(self, occupation_tuple):
        """Return index for an occupation tuple."""
        sites = tuple(i for i, n in enumerate(occupation_tuple) if n > 0)
        return self._index_map.get(sites, -1)


# =====================================================================
# Main model class
# =====================================================================

class FractionalChernInsulator:
    """Fractional Chern Insulator model on the kagome lattice.

    Spinless fermions with complex NN hopping  t·e^{±iφ}  on the
    kagome lattice (3-band model).  The lowest band at φ = 5π/4 has
    Chern number C = 1.  A NN density-density interaction V is
    projected into this band.  At filling ν = 1/3, the ground state
    is a lattice Laughlin state (FCI) with γ = ½ ln 3.

    Parameters
    ----------
    N1, N2 : int
        Torus dimensions (N₁ × N₂ unit cells, 3·N₁·N₂ real-space sites).
    hopping_phase : float
        Phase φ in the complex hopping (default 5π/4).
    interaction : float
        NN interaction strength V (default 1.0).
    band_dispersion_weight : float
        κ, weight of the single-particle dispersion in the projected
        Hamiltonian.  κ = 0 is the flat-band limit (default).
    target_band : int
        Which band to project into (0 = lowest, default).
    hopping : float
        Hopping amplitude t (default 1.0).
    """

    def __init__(self, N1, N2, hopping_phase=5*np.pi/4,
                 interaction=1.0, band_dispersion_weight=0.0,
                 target_band=0, hopping=1.0):

        self.N1 = N1
        self.N2 = N2
        self.N = N1 * N2
        self.N_sites = 3 * self.N
        self.N_particles = self.N // 3
        self.phi = hopping_phase
        self.V = interaction
        self.kappa = band_dispersion_weight
        self.target_band = target_band
        self.hopping = hopping

        if self.N % 3 != 0:
            raise ValueError(
                f"N = N1*N2 = {self.N} must be divisible by 3 for ν=1/3 filling"
            )

        # Compute single-particle band structure
        print(f"FCI model: {N1}×{N2} kagome torus, "
              f"{self.N_sites} sites, {self.N_particles} particles")
        self.band_energies, self.eigvecs = compute_bands(
            N1, N2, hopping_phase, hopping
        )

        # Verify Chern number
        C = chern_number(N1, N2, self.eigvecs, band=target_band)
        print(f"  Chern number of band {target_band}: C = {C:.4f}")
        if abs(round(C) - C) > 0.1:
            print(f"  WARNING: Chern number not well-quantized")

        # k-space Fock basis (for the solver)
        self.k_basis = FermionicBandBasis(self.N, self.N_particles)

        # Real-space basis (for observables) — built on demand
        self._real_basis = None
        self._T = None

        # Expose a basis attribute for compatibility with the solver
        self.basis = self.k_basis

        # Band gap and bandwidth
        gap = (np.min(self.band_energies[:, target_band + 1])
               - np.max(self.band_energies[:, target_band]))
        bw = (np.max(self.band_energies[:, target_band])
              - np.min(self.band_energies[:, target_band]))
        flatness = gap / bw if bw > 1e-10 else float('inf')
        print(f"  Band gap = {gap:.4f},  bandwidth = {bw:.4f},  "
              f"flatness ratio = {flatness:.2f}")

    def hamiltonian(self):
        """Build the band-projected many-body Hamiltonian.

        Returns
        -------
        H : sparse.csr_matrix
            The projected Hamiltonian in the k-space Fock basis.
        """
        H, _ = build_hamiltonian(
            self.N1, self.N2, self.eigvecs,
            self.band_energies[:, self.target_band],
            band=self.target_band,
            V=self.V,
            kappa=self.kappa,
        )
        return H

    def _ensure_transform(self):
        """Lazily compute the single-particle transformation matrix."""
        if self._T is None:
            self._T = single_particle_transform(
                self.N1, self.N2, self.eigvecs, band=self.target_band
            )

    def transform_to_real_space(self, psi_k):
        """Transform a k-space ground state to the real-space Fock basis.

        Parameters
        ----------
        psi_k : ndarray, shape (dim_k,)
            Ground state in the band-projected Fock basis.

        Returns
        -------
        psi_real : ndarray
            Wavefunction in the real-space Fock basis.
        real_basis : RealSpaceFermionBasis
            Basis adapter compatible with the observable functions.
        """
        self._ensure_transform()

        psi_real, real_states = transform_to_real_space(
            psi_k, self.k_basis, self._T, self.N_sites
        )

        real_basis = RealSpaceFermionBasis(self.N_sites, real_states)
        return psi_real, real_basis

    def solve(self, num_eigenvalues=6):
        """Find the lowest eigenvalues and eigenstates.

        Parameters
        ----------
        num_eigenvalues : int
            Number of lowest eigenvalues to compute.

        Returns
        -------
        eigenvalues : ndarray
        eigenvectors : ndarray, shape (dim, num_eigenvalues)
        """
        H = self.hamiltonian()
        dim = H.shape[0]

        if dim <= 2 * num_eigenvalues + 1:
            # Full diagonalization for tiny matrices
            H_dense = H.toarray()
            evals, evecs = np.linalg.eigh(H_dense)
            return evals[:num_eigenvalues], evecs[:, :num_eigenvalues]
        else:
            evals, evecs = eigsh(H, k=num_eigenvalues, which='SA')
            order = np.argsort(evals)
            return evals[order], evecs[:, order]

    def momentum_resolved_spectrum(self, num_eigenvalues=3):
        """Compute eigenvalues in each total-momentum sector.

        Returns a dictionary  (kt1, kt2) → eigenvalues  useful for
        comparing with Fig. 3 of the Kwan-Regnault tutorial.

        Returns
        -------
        spectrum : dict
            Maps (kt1, kt2) → sorted eigenvalues array.
        """
        N = self.N
        N1, N2 = self.N1, self.N2

        # Classify basis states by total momentum
        sectors = {}
        for idx in range(self.k_basis.dimension):
            state = self.k_basis.get_occupied(idx)
            kt1_tot = sum(K // N2 for K in state) % N1
            kt2_tot = sum(K % N2 for K in state) % N2
            key = (kt1_tot, kt2_tot)
            if key not in sectors:
                sectors[key] = []
            sectors[key].append(idx)

        # Build and diagonalize H in each sector
        H_full = self.hamiltonian()
        spectrum = {}

        for key, indices in sorted(sectors.items()):
            dim_sec = len(indices)
            # Extract submatrix for this sector
            H_sec = H_full[np.ix_(indices, indices)].toarray()
            evals = np.linalg.eigvalsh(H_sec)
            spectrum[key] = evals

        return spectrum

    def site_positions(self):
        """Return real-space positions of all sites."""
        return kagome_site_positions(self.N1, self.N2)

    def positions(self):
        """Alias for site_positions (interface for kitaev_preskill_regions)."""
        return self.site_positions()
