"""
Verification tests for hardcore-boson mode and the extended
(nearest-neighbor V n_i n_j) interaction, added for the TEE-decomposition
project (hardcore Extended Bose-Hubbard Model).

Tests:
    1. hardcore=True forces max_occupation=1 and rejects conflicting n_max
    2. Symmetry-reduced Hamiltonian matches full Hamiltonian ground energy
       (1D, hardcore + V != 0)
    3. Symmetry-reduced Hamiltonian matches full Hamiltonian ground energy
       (2D, hardcore + V != 0)
    4. Atomic limit (t=0): diagonal V n_i n_j energy matches direct count
       of occupied-neighbor bonds in the ground state configuration
    5. Hermiticity holds with nn_interaction != 0
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np

from lanczos_ed.models.bose_hubbard import BoseHubbard1D
from lanczos_ed.models.bose_hubbard_2d import BoseHubbard2D

PASS_COUNT = 0
FAIL_COUNT = 0


def check(name, got, expected, tol=1e-8):
    global PASS_COUNT, FAIL_COUNT
    diff = abs(got - expected)
    ok = diff < tol
    status = "PASS" if ok else "FAIL"
    if ok:
        PASS_COUNT += 1
    else:
        FAIL_COUNT += 1
    print(f"  [{status}] {name}: got {got:.10f}, expected {expected:.10f} (diff={diff:.2e})")


def check_true(name, cond):
    global PASS_COUNT, FAIL_COUNT
    status = "PASS" if cond else "FAIL"
    if cond:
        PASS_COUNT += 1
    else:
        FAIL_COUNT += 1
    print(f"  [{status}] {name}")


print("Test 1: hardcore=True forces max_occupation=1")
m = BoseHubbard1D(num_sites=6, total_particles=3, hardcore=True)
check_true("max_occupation == 1", m.max_occupation == 1)
try:
    BoseHubbard1D(num_sites=6, total_particles=3, hardcore=True, max_occupation=2)
    check_true("conflicting max_occupation raises ValueError", False)
except ValueError:
    check_true("conflicting max_occupation raises ValueError", True)

print("\nTest 2: 1D full vs symmetry-reduced (hardcore + V)")
for L, N, V, t in [(6, 3, 1.5, 1.0), (8, 4, 2.0, 1.0)]:
    m_full = BoseHubbard1D(num_sites=L, hopping=t, interaction=0.0,
                            total_particles=N, boundary='pbc',
                            hardcore=True, nn_interaction=V, use_symmetry=False)
    m_sym = BoseHubbard1D(num_sites=L, hopping=t, interaction=0.0,
                           total_particles=N, boundary='pbc',
                           hardcore=True, nn_interaction=V, use_symmetry=True)
    e_full = np.linalg.eigvalsh(m_full.hamiltonian().toarray())[0]
    e_sym = np.linalg.eigvalsh(m_sym.hamiltonian().toarray())[0]
    check(f"E0 full==sym (L={L},N={N},V={V})", e_sym, e_full, tol=1e-7)

print("\nTest 3: 2D full vs symmetry-reduced (hardcore + V)")
m_full = BoseHubbard2D(linear_size=3, hopping=1.0, interaction=0.0,
                        total_particles=4, boundary='pbc',
                        hardcore=True, nn_interaction=2.0, use_symmetry=False)
m_sym = BoseHubbard2D(linear_size=3, hopping=1.0, interaction=0.0,
                       total_particles=4, boundary='pbc',
                       hardcore=True, nn_interaction=2.0, use_symmetry=True)
e_full = np.linalg.eigvalsh(m_full.hamiltonian().toarray())[0]
e_sym = np.linalg.eigvalsh(m_sym.hamiltonian().toarray())[0].real
check("E0 full==sym (2D, 3x3, N=4, V=2.0)", e_sym, e_full, tol=1e-6)

print("\nTest 4: Atomic limit (t=0) diagonal V n_i n_j energy")
# t=0, hardcore, V != 0: Hamiltonian is exactly diagonal, so the ground
# energy is min over basis states of V * (# occupied NN bonds).
m = BoseHubbard1D(num_sites=6, hopping=0.0, interaction=0.0,
                   total_particles=3, boundary='pbc',
                   hardcore=True, nn_interaction=1.0, use_symmetry=False)
H = m.hamiltonian().toarray()
check_true("H is diagonal at t=0", np.allclose(H, np.diag(np.diag(H))))
# Minimum energy config for 3 hardcore bosons on a 6-site PBC ring with
# repulsive V=1 NN interaction: maximally spread out (every other site
# occupied) -> zero occupied NN bonds -> E0 = 0.
check("E0 (t=0, maximally spread, no NN bonds occupied)",
      np.diag(H).min(), 0.0)

print("\nTest 5: Hermiticity with nn_interaction != 0")
m = BoseHubbard2D(linear_size=3, hopping=1.0, interaction=0.5,
                   total_particles=5, boundary='pbc',
                   nn_interaction=1.3, use_symmetry=False)
H = m.hamiltonian().toarray()
check("||H - H^T||_max", np.abs(H - H.T).max(), 0.0)

total = PASS_COUNT + FAIL_COUNT
print(f"\n{'=' * 60}")
print(f"Results: {PASS_COUNT} passed, {FAIL_COUNT} failed out of {total} tests")
print(f"{'=' * 60}")
sys.exit(0 if FAIL_COUNT == 0 else 1)
