"""Verify symmetry reduction (Part A) and matrix-free solver (Part B).

Compare all three approaches against the screenshot reference values:
    L=4, N=4, U=3.275, PBC
    E0   = -4.5025314716
    F_A  =  0.46889609
    S_1  =  1.06972516
    S_2  =  0.86493042
"""
import sys, os, time
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

# --- Mock scipy.sparse if not available ---
try:
    import scipy.sparse
    import scipy.sparse.linalg
    HAS_SCIPY = True
except ImportError:
    HAS_SCIPY = False
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
            r._dense, r.shape = self._dense.T.copy(), self._dense.T.shape
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

        def __sub__(self, o):
            r = _MockCsr.__new__(_MockCsr)
            r._dense = self._dense - o._dense
            r.shape, r.nnz = r._dense.shape, np.count_nonzero(r._dense)
            return r

        def diagonal(self):
            return np.diag(self._dense)

        def eliminate_zeros(self):
            pass

        def toarray(self):
            return self._dense.copy()

        def __getitem__(self, key):
            return self._dense[key]

    class _MockDiags:
        @staticmethod
        def __call__(d, *args, **kwargs):
            r = _MockCsr.__new__(_MockCsr)
            r._dense = np.diag(d)
            r.shape = r._dense.shape
            r.nnz = np.count_nonzero(r._dense)
            return r

    _sp = types.ModuleType('scipy')
    _sps = types.ModuleType('scipy.sparse')
    _sps.csr_matrix = _MockCsr
    _sps.coo_matrix = _MockCsr
    _sps.issparse = lambda x: isinstance(x, _MockCsr)
    _sps.spmatrix = _MockCsr
    _sps.diags = _MockDiags()
    _sp.sparse = _sps
    sys.modules['scipy'] = _sp
    sys.modules['scipy.sparse'] = _sps
    _spl = types.ModuleType('scipy.sparse.linalg')
    _spl.eigsh = lambda *a, **kw: None
    _spl.LinearOperator = None
    sys.modules['scipy.sparse.linalg'] = _spl
# --- End mock ---

from lanczos_ed.models.bose_hubbard import BoseHubbard1D
from lanczos_ed.observables.basic import (
    bipartite_fluctuations, entanglement_entropy, density_profile,
)
from lanczos_ed.symmetry import shift_v, reverse_ket, find_cycles

# Reference values from GUI screenshot
REF = {
    'E0': -4.5025314716,
    'F_A': 0.46889609,
    'S_1': 1.06972516,
    'S_2': 0.86493042,
}
TOL = 1e-5


def dense_diag(H):
    """Dense diagonalization, return sorted eigenvalues and eigenvectors."""
    H_dense = H.toarray() if hasattr(H, 'toarray') else np.array(H)
    eigenvalues, eigenvectors = np.linalg.eigh(H_dense)
    return eigenvalues, eigenvectors


def compute_observables(psi, basis, L):
    """Compute all observables for a ground state wavefunction."""
    subsystem = list(range(L // 2))
    F_A = bipartite_fluctuations(psi, basis, subsystem)
    S1 = entanglement_entropy(psi, basis, subsystem_sites=subsystem, renyi_index=1.0)
    S2 = entanglement_entropy(psi, basis, subsystem_sites=subsystem, renyi_index=2.0)
    return F_A, S1, S2


def check_results(label, E0, F_A, S1, S2):
    """Compare against reference values and print results."""
    results = {'E0': E0, 'F_A': F_A, 'S_1': S1, 'S_2': S2}
    all_pass = True
    for name, got in results.items():
        expected = REF[name]
        ok = abs(got - expected) < TOL
        if not ok:
            all_pass = False
        print(f"    {name}: {got:14.10f}  expected: {expected:14.10f}  "
              f"diff={abs(got-expected):.2e}  {'PASS' if ok else 'FAIL'}")
    return all_pass


# =================================================================
# Test 0: Verify bitwise symmetry operations
# =================================================================
print("=" * 70)
print("Test 0: Bitwise symmetry operations")
print("=" * 70)

from lanczos_ed.unary_basis import UnaryBasis, occupation_to_unary, unary_to_occupation

# shift_v: (2,0,1) -> (0,1,2)
v = occupation_to_unary((2, 0, 1))  # L=3, N=3, total_bits=6
v_shifted = shift_v(v, 6)
occ_shifted = unary_to_occupation(v_shifted, 3)
assert occ_shifted == (0, 1, 2), f"shift_v failed: got {occ_shifted}"
print(f"  shift_v((2,0,1)) = {occ_shifted}  PASS")

# reverse_ket: (2,0,1) -> (1,0,2)
v_ref = reverse_ket(v, 6)
occ_ref = unary_to_occupation(v_ref, 3)
assert occ_ref == (1, 0, 2), f"reverse_ket failed: got {occ_ref}"
print(f"  reverse_ket((2,0,1)) = {occ_ref}  PASS")

# Full cycle test: applying shift_v L times returns to original
basis_test = UnaryBasis(num_sites=4, total_particles=4)
num_checked = 0
for idx in range(basis_test.dim):
    v0 = int(basis_test._integers[idx])
    v = v0
    for _ in range(4):  # L=4 shifts
        v = shift_v(v, 8)
    assert v == v0, f"4 shifts didn't return to original for idx={idx}"
    num_checked += 1
print(f"  T^L = identity: checked {num_checked} states  PASS")

# Reflection is an involution: R^2 = identity
for idx in range(basis_test.dim):
    v0 = int(basis_test._integers[idx])
    v_rr = reverse_ket(reverse_ket(v0, 8), 8)
    assert v_rr == v0, f"R^2 != I for idx={idx}"
print(f"  R^2 = identity: checked {num_checked} states  PASS")


# =================================================================
# Test 1: Cycle structure sanity checks
# =================================================================
print("\n" + "=" * 70)
print("Test 1: Cycle structure sanity checks")
print("=" * 70)

all_pass = True

for L, N in [(4, 4), (6, 3), (6, 6), (8, 4)]:
    basis = UnaryBasis(num_sites=L, total_particles=N)
    leaders, sizes, n_cycles, s2c = find_cycles(basis)

    # Every state assigned
    ok1 = np.all(s2c >= 0)
    # Sizes sum to dim
    ok2 = sizes.sum() == basis.dim
    # All cycle sizes divide 2L (translation period divides L,
    # reflection can double it)
    ok3 = all(2 * L % s == 0 for s in sizes)

    ok = ok1 and ok2 and ok3
    if not ok:
        all_pass = False
    reduction = basis.dim / n_cycles
    print(f"  L={L}, N={N}: dim={basis.dim:>8}, cycles={n_cycles:>6}, "
          f"reduction={reduction:.1f}×  {'PASS' if ok else 'FAIL'}")

if not all_pass:
    print("  CYCLE STRUCTURE TESTS FAILED")


# =================================================================
# Test 2: Full basis — reference results (L=4, N=4, U=3.275)
# =================================================================
print("\n" + "=" * 70)
print("Test 2: Full basis (no symmetry)")
print("=" * 70)

t0 = time.time()
model_full = BoseHubbard1D(
    num_sites=4, hopping=1.0, interaction=3.275,
    total_particles=4, boundary='pbc', use_symmetry=False
)
H_full = model_full.hamiltonian()
evals_full, evecs_full = dense_diag(H_full)
E0_full = evals_full[0]
psi_full = evecs_full[:, 0]
F_A_f, S1_f, S2_f = compute_observables(psi_full, model_full.basis, 4)
t_full = time.time() - t0

print(f"  Model: {model_full}")
print(f"  Time: {t_full:.4f}s")
pass_full = check_results("Full basis", E0_full, F_A_f, S1_f, S2_f)


# =================================================================
# Test 3: Symmetry-reduced basis (L=4, N=4, U=3.275)
# =================================================================
print("\n" + "=" * 70)
print("Test 3: Symmetry-reduced basis (q=0, R=+1)")
print("=" * 70)

t0 = time.time()
model_sym = BoseHubbard1D(
    num_sites=4, hopping=1.0, interaction=3.275,
    total_particles=4, boundary='pbc', use_symmetry=True
)
H_sym = model_sym.hamiltonian()
evals_sym, evecs_sym = dense_diag(H_sym)
E0_sym = evals_sym[0]

# Reconstruct full wavefunction for observables
psi_reconstructed = model_sym.reconstruct_wavefunction(evecs_sym[:, 0])
F_A_s, S1_s, S2_s = compute_observables(psi_reconstructed, model_sym.basis, 4)
t_sym = time.time() - t0

print(f"  Model: {model_sym}")
print(f"  Reduced dim: {model_sym.dim} (from {model_sym.full_dim})")
print(f"  Time: {t_sym:.4f}s")
pass_sym = check_results("Symmetry", E0_sym, F_A_s, S1_s, S2_s)


# =================================================================
# Test 4: Spectra comparison — full vs symmetry (multiple cases)
# =================================================================
print("\n" + "=" * 70)
print("Test 4: Ground state energy match (full vs symmetry)")
print("=" * 70)

spectra_pass = True
for L, N, U in [(4, 4, 3.275), (6, 3, 2.0), (6, 6, 1.0), (4, 2, 1.0)]:
    m_f = BoseHubbard1D(num_sites=L, hopping=1.0, interaction=U,
                        total_particles=N, boundary='pbc', use_symmetry=False)
    m_s = BoseHubbard1D(num_sites=L, hopping=1.0, interaction=U,
                        total_particles=N, boundary='pbc', use_symmetry=True)
    e_f = np.linalg.eigvalsh(m_f.hamiltonian().toarray())[0]
    e_s = np.linalg.eigvalsh(m_s.hamiltonian().toarray())[0]
    diff = abs(e_f - e_s)
    ok = diff < 1e-10
    if not ok:
        spectra_pass = False
    print(f"  L={L}, N={N}, U={U}: E0_full={e_f:.10f}, "
          f"E0_sym={e_s:.10f}, diff={diff:.2e}  {'PASS' if ok else 'FAIL'}")


# =================================================================
# Test 5: Matrix-free solver (pure Python, no Numba in sandbox)
# =================================================================
print("\n" + "=" * 70)
print("Test 5: Matrix-free solver (pure Python)")
print("=" * 70)

if HAS_SCIPY:
    from lanczos_ed.solvers.matrix_free import solve_matrix_free

    t0 = time.time()
    model_mf = BoseHubbard1D(
        num_sites=4, hopping=1.0, interaction=3.275,
        total_particles=4, boundary='pbc', use_symmetry=False
    )
    evals_mf, evecs_mf = solve_matrix_free(model_mf, num_eigenvalues=1,
                                            use_numba=False)
    E0_mf = evals_mf[0]
    psi_mf = evecs_mf[:, 0]
    F_A_m, S1_m, S2_m = compute_observables(psi_mf, model_mf.basis, 4)
    t_mf = time.time() - t0

    print(f"  Time: {t_mf:.4f}s")
    pass_mf = check_results("Matrix-free", E0_mf, F_A_m, S1_m, S2_m)
else:
    print("  SKIPPED (scipy.sparse.linalg not available in sandbox)")
    pass_mf = True  # don't fail the suite


# =================================================================
# Test 6: Scaling benchmark
# =================================================================
print("\n" + "=" * 70)
print("Test 6: Scaling benchmark (symmetry reduction factor)")
print("=" * 70)

print(f"  {'L':>4} {'N':>4} {'full_dim':>12} {'reduced_dim':>12} "
      f"{'reduction':>10} {'build (s)':>10}")
print("  " + "-" * 60)

for L, N in [(4, 4), (6, 3), (6, 6), (8, 4), (8, 8), (10, 10)]:
    t0 = time.time()
    m = BoseHubbard1D(num_sites=L, hopping=1.0, interaction=1.0,
                      total_particles=N, boundary='pbc', use_symmetry=True)
    dt = time.time() - t0
    print(f"  {L:4d} {N:4d} {m.full_dim:12,} {m.dim:12,} "
          f"{m.full_dim/m.dim:10.1f}× {dt:10.3f}")


# =================================================================
# Summary
# =================================================================
print(f"\n{'=' * 70}")
overall = pass_full and pass_sym and spectra_pass and pass_mf
if overall:
    print("ALL TESTS PASSED")
else:
    print("SOME TESTS FAILED")
print(f"{'=' * 70}")

sys.exit(0 if overall else 1)
