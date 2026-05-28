"""Verify unary basis integration matches the GUI screenshot values."""
import sys
import os
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

# --- Mock scipy.sparse if not available ---
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
from lanczos_ed.unary_basis import UnaryBasis
from lanczos_ed.models.bose_hubbard import BoseHubbard1D
from lanczos_ed.observables.basic import (
    bipartite_fluctuations, entanglement_entropy, density_profile,
)


def dense_diag(model):
    H_dense = model.hamiltonian().toarray()
    eigenvalues, eigenvectors = np.linalg.eigh(H_dense)
    return eigenvalues, eigenvectors


# =================================================================
# Test 1: UnaryBasis dimensions match FockBasis
# =================================================================
print("=" * 60)
print("Test 1: UnaryBasis vs FockBasis dimensions")
print("=" * 60)

cases = [(4, 4, None), (6, 3, None), (4, 2, 2), (3, 4, None)]
for L, N, nmax in cases:
    ub = UnaryBasis(num_sites=L, total_particles=N, max_occupation=nmax)
    fb = FockBasis(num_sites=L, total_particles=N, max_occupation=nmax)
    match = ub.dim == fb.dim
    print(f"  L={L}, N={N}, nmax={nmax}: unary={ub.dim}, fock={fb.dim} {'PASS' if match else 'FAIL'}")

# =================================================================
# Test 2: Full spectra match between unary and fock basis
# =================================================================
print("\n" + "=" * 60)
print("Test 2: Spectra match (UnaryBasis vs FockBasis)")
print("=" * 60)

for L, N, U in [(4, 4, 3.275), (6, 3, 2.0), (4, 2, 1.0)]:
    m_u = BoseHubbard1D(num_sites=L, hopping=1.0, interaction=U,
                        total_particles=N, boundary='pbc', basis_type='unary')
    m_f = BoseHubbard1D(num_sites=L, hopping=1.0, interaction=U,
                        total_particles=N, boundary='pbc', basis_type='fock')
    eigs_u = np.linalg.eigvalsh(m_u.hamiltonian().toarray())
    eigs_f = np.linalg.eigvalsh(m_f.hamiltonian().toarray())
    ok = np.allclose(eigs_u, eigs_f, atol=1e-10)
    diff = np.max(np.abs(eigs_u - eigs_f))
    print(f"  L={L}, N={N}, U={U}: {'PASS' if ok else 'FAIL'} (max diff={diff:.2e})")

# =================================================================
# Test 3: Match screenshot values (L=4, N=4, U=3.275)
# =================================================================
print("\n" + "=" * 60)
print("Test 3: Match screenshot values (L=4, N=4, U=3.275)")
print("=" * 60)

model = BoseHubbard1D(num_sites=4, hopping=1.0, interaction=3.275,
                      total_particles=4, boundary='pbc', basis_type='unary')
print(f"  Model: {model}")

eigenvalues, eigenvectors = dense_diag(model)
E0 = eigenvalues[0]
psi0 = eigenvectors[:, 0]

subsystem = list(range(model.num_sites // 2))  # sites [0, 1]
F_A = bipartite_fluctuations(psi0, model.basis, subsystem)
S1 = entanglement_entropy(psi0, model.basis, subsystem_sites=subsystem, renyi_index=1.0)
S2 = entanglement_entropy(psi0, model.basis, subsystem_sites=subsystem, renyi_index=2.0)

print(f"  E0  = {E0:.10f}  (expected: -4.5025314716)")
print(f"  F_A = {F_A:.8f}  (expected:  0.46889609)")
print(f"  S_1 = {S1:.8f}  (expected:  1.06972516)")
print(f"  S_2 = {S2:.8f}  (expected:  0.86493042)")

tol = 1e-5
results = {
    'E0': (E0, -4.5025314716),
    'F_A': (F_A, 0.46889609),
    'S_1': (S1, 1.06972516),
    'S_2': (S2, 0.86493042),
}

all_pass = True
for name, (got, expected) in results.items():
    ok = abs(got - expected) < tol
    status = "PASS" if ok else "FAIL"
    if not ok:
        all_pass = False
    print(f"  {name}: {status} (diff={abs(got-expected):.2e})")

# =================================================================
# Test 4: Existing test suite still passes with unary default
# =================================================================
print("\n" + "=" * 60)
print("Test 4: Existing analytical tests with unary basis")
print("=" * 60)

# Two-site exact result
for U_value in [0.0, 1.0, 4.0, 10.0]:
    model = BoseHubbard1D(num_sites=2, hopping=1.0, interaction=U_value,
                          total_particles=2, boundary='pbc', basis_type='unary')
    evals, _ = dense_diag(model)
    s2 = np.sqrt(2.0)
    H_exact = np.array([
        [U_value, -s2,  0.0],
        [-s2,     0.0, -s2],
        [0.0,    -s2,   U_value],
    ])
    exact_evals = np.sort(np.linalg.eigvalsh(H_exact))
    ok = np.allclose(evals, exact_evals, atol=1e-10)
    status = "PASS" if ok else "FAIL"
    if not ok:
        all_pass = False
    print(f"  2-site U={U_value}: {status}")

# U=0 limit: E0 = -2tN
for L in [4, 6]:
    N = L // 2
    model = BoseHubbard1D(num_sites=L, hopping=1.0, interaction=0.0,
                          total_particles=N, boundary='pbc', basis_type='unary')
    evals, _ = dense_diag(model)
    expected = -2.0 * N
    ok = abs(evals[0] - expected) < 1e-10
    if not ok:
        all_pass = False
    print(f"  U=0, L={L}, N={N}: E0={evals[0]:.10f} expected={expected:.1f} {'PASS' if ok else 'FAIL'}")

# t=0 limit
model = BoseHubbard1D(num_sites=4, hopping=0.0, interaction=1.0,
                      total_particles=4, boundary='pbc', basis_type='unary')
evals, _ = dense_diag(model)
ok = abs(evals[0]) < 1e-10
if not ok:
    all_pass = False
print(f"  t=0, L=4, N=4: E0={evals[0]:.10f} expected=0.0 {'PASS' if ok else 'FAIL'}")

# Hermiticity
model = BoseHubbard1D(num_sites=6, hopping=1.0, interaction=3.0,
                      total_particles=3, boundary='pbc', basis_type='unary')
H_dense = model.hamiltonian().toarray()
herm_err = np.max(np.abs(H_dense - H_dense.T))
ok = herm_err < 1e-14
if not ok:
    all_pass = False
print(f"  Hermiticity: max|H-H^T|={herm_err:.2e} {'PASS' if ok else 'FAIL'}")

# Density normalization
model = BoseHubbard1D(num_sites=6, hopping=1.0, interaction=2.0,
                      total_particles=4, boundary='pbc', basis_type='unary')
_, evecs = dense_diag(model)
density = density_profile(evecs[:, 0], model.basis)
ok = abs(density.sum() - 4.0) < 1e-10
if not ok:
    all_pass = False
print(f"  Density sum={density.sum():.10f} expected=4.0 {'PASS' if ok else 'FAIL'}")

# =================================================================
# Summary
# =================================================================
print(f"\n{'=' * 60}")
if all_pass:
    print("ALL TESTS PASSED")
else:
    print("SOME TESTS FAILED")
print(f"{'=' * 60}")

sys.exit(0 if all_pass else 1)
