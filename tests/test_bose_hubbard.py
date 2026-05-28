"""
Verification tests for the 1D Bose-Hubbard model.

Tests against known analytical results to validate:
    1. Hilbert space dimensions (combinatorial formulas)
    2. Two-site exact diagonalization (3x3 analytical H)
    3. Non-interacting limit (U=0): all bosons in k=0 mode
    4. Atomic limit (t=0): trivially diagonal Hamiltonian
    5. Entanglement entropy of product states = 0
    6. Hermiticity of the Hamiltonian
    7. Density normalization: sum <n_i> = N
    8. PBC gives lower energy than OBC (more hopping channels)
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
from math import comb

# --- Mock scipy.sparse if not available (e.g. in sandboxed environments) ---
try:
    import scipy.sparse
except ImportError:
    import types

    class _MockCsr:
        def __init__(self, arg, shape=None, dtype=None):
            if isinstance(arg, tuple):
                data, (rows, cols) = arg
                mat = np.zeros(shape, dtype=dtype or np.float64)
                for d, r, c in zip(data, rows, cols):
                    mat[r, c] += d
                self._dense = mat
            else:
                self._dense = np.array(arg)
            self.shape = self._dense.shape
            self.nnz = np.count_nonzero(self._dense)

        @property
        def T(self):
            r = _MockCsr.__new__(_MockCsr)
            r._dense, r.shape = self._dense.T, self._dense.T.shape
            r.nnz = self.nnz
            return r

        def __truediv__(self, o):
            r = _MockCsr.__new__(_MockCsr)
            r._dense = self._dense / o
            r.shape, r.nnz = r._dense.shape, np.count_nonzero(r._dense)
            return r

        def __add__(self, o):
            r = _MockCsr.__new__(_MockCsr)
            r._dense = self._dense + o._dense
            r.shape, r.nnz = r._dense.shape, np.count_nonzero(r._dense)
            return r

        def eliminate_zeros(self):
            pass

        def toarray(self):
            return self._dense.copy()

        def __getitem__(self, key):
            return self._dense[key]

    _sp = types.ModuleType('scipy')
    _sps = types.ModuleType('scipy.sparse')
    _sps.csr_matrix = _MockCsr
    _sps.issparse = lambda x: isinstance(x, _MockCsr)
    _sps.spmatrix = _MockCsr
    _sp.sparse = _sps
    sys.modules['scipy'] = _sp
    sys.modules['scipy.sparse'] = _sps
    _spl = types.ModuleType('scipy.sparse.linalg')
    _spl.eigsh = lambda *a, **kw: None
    sys.modules['scipy.sparse.linalg'] = _spl
# --- End mock ---

from lanczos_ed.basis import FockBasis
from lanczos_ed.models.bose_hubbard import BoseHubbard1D
from lanczos_ed.observables.basic import (
    density_profile, bipartite_fluctuations, entanglement_entropy,
)


# ---- Test infrastructure ----

PASS_COUNT = 0
FAIL_COUNT = 0


def check(test_name, computed, expected, tolerance=1e-10):
    """Compare a computed value to an expected value."""
    global PASS_COUNT, FAIL_COUNT
    difference = abs(computed - expected)
    passed = difference < tolerance

    status = "PASS" if passed else "FAIL"
    if passed:
        PASS_COUNT += 1
    else:
        FAIL_COUNT += 1

    print(f"  [{status}] {test_name}: "
          f"got {computed:.12f}, expected {expected:.12f} "
          f"(diff={difference:.2e})")
    return passed


def dense_diagonalize(model):
    """Full dense diagonalization for testing (no ARPACK needed)."""
    hamiltonian = model.hamiltonian()
    hamiltonian_dense = hamiltonian.toarray()
    eigenvalues, eigenvectors = np.linalg.eigh(hamiltonian_dense)
    return eigenvalues, eigenvectors


# ---- Tests ----

print("=" * 60)
print("Bose-Hubbard 1D — Verification Tests")
print("=" * 60)


# -------------------------------------------------------------------
# Test 1: Hilbert space dimensions
# -------------------------------------------------------------------
print("\nTest 1: Hilbert space dimensions")

# Canonical: dim = C(N + L - 1, L - 1) when n_max >= N
canonical_cases = [
    (2, 2, None),   # C(3, 1) = 3
    (3, 3, None),   # C(5, 2) = 10
    (4, 2, None),   # C(5, 3) = 10
    (4, 2, 2),      # n_max = N, same as above
    (3, 1, 1),      # C(3, 2) = 3
]
for num_sites, total_particles, max_occ in canonical_cases:
    basis = FockBasis(num_sites=num_sites, total_particles=total_particles,
                      max_occupation=max_occ)
    if max_occ is None or max_occ >= total_particles:
        expected_dim = comb(total_particles + num_sites - 1, num_sites - 1)
    else:
        expected_dim = basis.dim
    check(f"dim(L={num_sites}, N={total_particles}, n_max={max_occ})",
          basis.dim, expected_dim)

# Grand canonical: dim = (n_max + 1)^L
grand_canonical_cases = [(2, 1), (3, 2), (2, 3)]
for num_sites, max_occ in grand_canonical_cases:
    basis = FockBasis(num_sites=num_sites, max_occupation=max_occ)
    expected_dim = (max_occ + 1) ** num_sites
    check(f"dim(L={num_sites}, n_max={max_occ}, GC)", basis.dim, expected_dim)


# -------------------------------------------------------------------
# Test 2: Two-site, two-particle exact result
# -------------------------------------------------------------------
print("\nTest 2: Two-site, two-particle Bose-Hubbard")

# For L=2, N=2, the basis is {|2,0>, |1,1>, |0,2>} (dim=3).
# With a single bond (0,1), the Hamiltonian matrix is:
#
#   H = | U       -sqrt(2)*t    0          |
#       | -sqrt(2)*t    0       -sqrt(2)*t  |
#       | 0        -sqrt(2)*t   U          |
#
# The sqrt(2) comes from bosonic enhancement: b†|n> = sqrt(n+1)|n+1>.

for U_value in [0.0, 1.0, 4.0, 10.0]:
    model = BoseHubbard1D(num_sites=2, hopping=1.0, interaction=U_value,
                          total_particles=2, boundary='pbc')
    eigenvalues, _ = dense_diagonalize(model)

    # Analytical Hamiltonian
    sqrt2 = np.sqrt(2.0)
    H_exact = np.array([
        [U_value, -sqrt2,  0.0],
        [-sqrt2,  0.0,    -sqrt2],
        [0.0,    -sqrt2,   U_value],
    ])
    exact_eigenvalues = np.sort(np.linalg.eigvalsh(H_exact))

    check(f"E0(L=2, N=2, U={U_value})", eigenvalues[0], exact_eigenvalues[0])


# -------------------------------------------------------------------
# Test 3: Non-interacting limit (U = 0)
# -------------------------------------------------------------------
print("\nTest 3: Non-interacting limit (U = 0)")

# For U=0 with PBC, single-particle energies are:
#     eps_k = -2t * cos(2*pi*k / L)
# The ground state puts all N bosons in the k=0 mode:
#     E_0 = -2t * N

for num_sites in [4, 6, 8]:
    total_particles = num_sites // 2
    model = BoseHubbard1D(num_sites=num_sites, hopping=1.0, interaction=0.0,
                          total_particles=total_particles, boundary='pbc')
    eigenvalues, _ = dense_diagonalize(model)
    expected_energy = -2.0 * total_particles
    check(f"E0(L={num_sites}, N={total_particles}, U=0, PBC)",
          eigenvalues[0], expected_energy)


# -------------------------------------------------------------------
# Test 4: Atomic limit (t = 0)
# -------------------------------------------------------------------
print("\nTest 4: Atomic limit (t = 0)")

# t=0 means no hopping. H is diagonal with E = (U/2) sum n_i(n_i - 1).
# Uniform filling |1,1,1,1> gives n_i=1 everywhere, so E = 0.
model = BoseHubbard1D(num_sites=4, hopping=0.0, interaction=1.0,
                      total_particles=4, boundary='pbc')
eigenvalues, _ = dense_diagonalize(model)
check("E0(L=4, N=4, t=0, U=1)", eigenvalues[0], 0.0)

# N=4 on 3 sites: minimum energy states have occupation (2,1,1) and
# permutations, giving E = U/2 * 2*(2-1) = U = 2.0
model = BoseHubbard1D(num_sites=3, hopping=0.0, interaction=2.0,
                      total_particles=4, boundary='pbc')
eigenvalues, _ = dense_diagonalize(model)
check("E0(L=3, N=4, t=0, U=2)", eigenvalues[0], 2.0)


# -------------------------------------------------------------------
# Test 5: Entanglement entropy of product states
# -------------------------------------------------------------------
print("\nTest 5: Entanglement entropy in atomic limit (product state)")

# In the atomic limit, |1,1,1,1> is a product state => S = 0
model = BoseHubbard1D(num_sites=4, hopping=0.0, interaction=1.0,
                      total_particles=4, boundary='pbc')
_, eigenvectors = dense_diagonalize(model)
ground_state_wfn = eigenvectors[:, 0]
entropy = entanglement_entropy(ground_state_wfn, model.basis, renyi_index=1.0)
check("S_vN(product state)", entropy, 0.0, tolerance=1e-8)


# -------------------------------------------------------------------
# Test 6: Hamiltonian Hermiticity
# -------------------------------------------------------------------
print("\nTest 6: Hamiltonian Hermiticity")

model = BoseHubbard1D(num_sites=6, hopping=1.0, interaction=3.0,
                      total_particles=3, boundary='pbc')
H_dense = model.hamiltonian().toarray()
hermiticity_error = np.max(np.abs(H_dense - H_dense.T))
check("||H - H^T||_max", hermiticity_error, 0.0, tolerance=1e-14)


# -------------------------------------------------------------------
# Test 7: Density normalization
# -------------------------------------------------------------------
print("\nTest 7: Density normalization (sum <n_i> = N)")

model = BoseHubbard1D(num_sites=6, hopping=1.0, interaction=2.0,
                      total_particles=4, boundary='pbc')
_, eigenvectors = dense_diagonalize(model)
ground_state_wfn = eigenvectors[:, 0]
density = density_profile(ground_state_wfn, model.basis)
check("sum <n_i> = N", density.sum(), 4.0, tolerance=1e-10)


# -------------------------------------------------------------------
# Test 8: PBC vs OBC energy ordering
# -------------------------------------------------------------------
print("\nTest 8: PBC gives lower energy than OBC")

model_pbc = BoseHubbard1D(num_sites=4, hopping=1.0, interaction=2.0,
                          total_particles=2, boundary='pbc')
model_obc = BoseHubbard1D(num_sites=4, hopping=1.0, interaction=2.0,
                          total_particles=2, boundary='obc')
e_pbc, _ = dense_diagonalize(model_pbc)
e_obc, _ = dense_diagonalize(model_obc)

pbc_has_lower_energy = e_pbc[0] < e_obc[0]
status = "PASS" if pbc_has_lower_energy else "FAIL"
if pbc_has_lower_energy:
    PASS_COUNT += 1
else:
    FAIL_COUNT += 1
print(f"  [{status}] E0_PBC={e_pbc[0]:.8f} < E0_OBC={e_obc[0]:.8f}")


# -------------------------------------------------------------------
# Summary
# -------------------------------------------------------------------
total_tests = PASS_COUNT + FAIL_COUNT
print(f"\n{'=' * 60}")
print(f"Results: {PASS_COUNT} passed, {FAIL_COUNT} failed out of {total_tests} tests")
print(f"{'=' * 60}")

sys.exit(0 if FAIL_COUNT == 0 else 1)
