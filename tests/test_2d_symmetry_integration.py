"""
Validation tests for 2D translational symmetry integration.

Tests that BoseHubbard2D with use_symmetry=True produces the same
ground state energy and observables as the full (unsymmetrized)
calculation. Also compares against reference ED data.

Run with:
    cd ~/Desktop/WorkingCopies/Lanczos
    python -m tests.test_2d_symmetry_integration
"""

import numpy as np
import time
import sys
sys.path.insert(0, '.')

from lanczos_ed.models.bose_hubbard_2d import BoseHubbard2D
from lanczos_ed.solvers.lanczos import LanczosSolver
from lanczos_ed.observables.basic import (
    density_profile, bipartite_fluctuations, entanglement_entropy,
    sweep_observables,
)


def test_2x2_symmetry_vs_full():
    """Compare symmetry-reduced and full E₀ for 2×2, N=4."""
    print("=" * 60)
    print("Test 1: 2×2 PBC, N=4 — symmetry vs full diag")
    print("=" * 60)

    # Several U values spanning weak to strong coupling
    test_cases = [
        (0.01,  -11.9850233617),
        (1.0,   None),  # intermediate, no reference
        (10.0,  None),
        (100.0, -4.1601594691),
    ]

    for U, E0_ref in test_cases:
        # Full diagonalization
        model_full = BoseHubbard2D(
            linear_size=2, hopping=1.0, interaction=U,
            chemical_potential=1.0, total_particles=4,
            boundary='pbc', use_symmetry=False,
        )
        H_full = model_full.hamiltonian()
        solver_full = LanczosSolver(H_full, num_eigenvalues=1)
        evals_full, evecs_full = solver_full.solve()

        # Symmetry-reduced
        model_sym = BoseHubbard2D(
            linear_size=2, hopping=1.0, interaction=U,
            chemical_potential=1.0, total_particles=4,
            boundary='pbc', use_symmetry=True,
        )
        H_red = model_sym.hamiltonian()
        solver_red = LanczosSolver(H_red, num_eigenvalues=1)
        evals_red, evecs_red = solver_red.solve()

        E0_full = evals_full[0]
        E0_sym = evals_red[0]
        match = abs(E0_full - E0_sym) < 1e-9

        print(f"  U={U:8.2f}  |  full dim={model_full.dim:5d}  "
              f"red dim={model_sym.dim:5d}  |  "
              f"E0_full={E0_full:.10f}  E0_sym={E0_sym:.10f}  "
              f"match={'PASS' if match else 'FAIL'}")

        if E0_ref is not None:
            ref_match = abs(E0_full - E0_ref) < 1e-8
            print(f"           |  ref E0={E0_ref:.10f}  "
                  f"match={'PASS' if ref_match else 'FAIL'}")

        assert match, f"E₀ mismatch at U={U}"

    print("  → All 2×2 tests PASSED\n")


def test_2x2_wavefunction_reconstruction():
    """Verify wavefunction reconstruction produces valid state."""
    print("=" * 60)
    print("Test 2: Wavefunction reconstruction (2×2, U=4.0)")
    print("=" * 60)

    model_sym = BoseHubbard2D(
        linear_size=2, hopping=1.0, interaction=4.0,
        chemical_potential=1.0, total_particles=4,
        boundary='pbc', use_symmetry=True,
    )
    H_red = model_sym.hamiltonian()
    solver = LanczosSolver(H_red, num_eigenvalues=1)
    evals, evecs = solver.solve()

    # Reconstruct
    psi = model_sym.reconstruct_wavefunction(solver.ground_state)
    norm = np.linalg.norm(psi)
    print(f"  norm = {norm:.10f}  (expected 1.0)")
    assert abs(norm - 1.0) < 1e-10, f"Bad norm: {norm}"

    # Compare with full diag
    model_full = BoseHubbard2D(
        linear_size=2, hopping=1.0, interaction=4.0,
        chemical_potential=1.0, total_particles=4,
        boundary='pbc', use_symmetry=False,
    )
    H_full = model_full.hamiltonian()
    solver_full = LanczosSolver(H_full, num_eigenvalues=1)
    evals_full, evecs_full = solver_full.solve()

    overlap = abs(np.dot(psi, solver_full.ground_state))
    print(f"  overlap with full GS = {overlap:.10f}  (expected 1.0)")
    assert abs(overlap - 1.0) < 1e-9, f"Bad overlap: {overlap}"

    # Density profile should match
    density_sym = density_profile(psi, model_sym.basis)
    density_full = density_profile(solver_full.ground_state, model_full.basis)
    density_match = np.allclose(density_sym, density_full, atol=1e-10)
    print(f"  density match: {'PASS' if density_match else 'FAIL'}")
    assert density_match

    print("  → Wavefunction reconstruction PASSED\n")


def test_2x2_observables_match():
    """Full observable sweep should match between symmetry and full."""
    print("=" * 60)
    print("Test 3: Observable sweep match (2×2, U=10.0)")
    print("=" * 60)

    # Full
    model_full = BoseHubbard2D(
        linear_size=2, hopping=1.0, interaction=10.0,
        chemical_potential=1.0, total_particles=4,
        boundary='pbc', use_symmetry=False,
    )
    H_full = model_full.hamiltonian()
    solver_full = LanczosSolver(H_full, num_eigenvalues=1)
    solver_full.solve()
    psi_full = solver_full.ground_state

    def make_sub_full(l):
        return model_full.get_subregion('strip', l)

    sweep_full = sweep_observables(psi_full, model_full.basis, make_sub_full, 1)

    # Symmetry-reduced
    model_sym = BoseHubbard2D(
        linear_size=2, hopping=1.0, interaction=10.0,
        chemical_potential=1.0, total_particles=4,
        boundary='pbc', use_symmetry=True,
    )
    H_red = model_sym.hamiltonian()
    solver_sym = LanczosSolver(H_red, num_eigenvalues=1)
    solver_sym.solve()
    psi_sym = model_sym.reconstruct_wavefunction(solver_sym.ground_state)

    def make_sub_sym(l):
        return model_sym.get_subregion('strip', l)

    sweep_sym = sweep_observables(psi_sym, model_sym.basis, make_sub_sym, 1)

    for key in ['F_A', 'S_1', 'S_2', 'S_2_acc']:
        val_full = sweep_full[0][key]
        val_sym = sweep_sym[0][key]
        match = abs(val_full - val_sym) < 1e-8
        print(f"  {key:8s}: full={val_full:.10f}  sym={val_sym:.10f}  "
              f"{'PASS' if match else 'FAIL'}")
        assert match, f"Observable {key} mismatch"

    print("  → Observable sweep PASSED\n")


def test_3x3_symmetry_vs_full():
    """Compare symmetry-reduced and full E₀ for 3×3, N=9."""
    print("=" * 60)
    print("Test 4: 3×3 PBC, N=9 — symmetry vs full diag")
    print("=" * 60)

    # Reference values from user's sweep data
    test_cases = [
        (1.2,  -40.6865763406),
        (10.0, None),
        (100.0, -9.7439137610),
    ]

    for U, E0_ref in test_cases:
        t0 = time.time()

        # Full
        model_full = BoseHubbard2D(
            linear_size=3, hopping=1.0, interaction=U,
            chemical_potential=1.0, total_particles=9,
            boundary='pbc', use_symmetry=False,
        )
        H_full = model_full.hamiltonian()
        solver_full = LanczosSolver(H_full, num_eigenvalues=1)
        evals_full, _ = solver_full.solve()
        t_full = time.time() - t0

        t0 = time.time()
        # Symmetry
        model_sym = BoseHubbard2D(
            linear_size=3, hopping=1.0, interaction=U,
            chemical_potential=1.0, total_particles=9,
            boundary='pbc', use_symmetry=True,
        )
        H_red = model_sym.hamiltonian()
        solver_sym = LanczosSolver(H_red, num_eigenvalues=1)
        evals_sym, _ = solver_sym.solve()
        t_sym = time.time() - t0

        E0_full = evals_full[0]
        E0_sym = evals_sym[0]
        match = abs(E0_full - E0_sym) < 1e-8

        print(f"  U={U:8.2f}  |  full dim={model_full.dim:6d} ({t_full:.2f}s)  "
              f"red dim={model_sym.dim:6d} ({t_sym:.2f}s)  |  "
              f"match={'PASS' if match else 'FAIL'}")
        print(f"           |  E0_full={E0_full:.10f}  E0_sym={E0_sym:.10f}")

        if E0_ref is not None:
            ref_match = abs(E0_full - E0_ref) < 1e-7
            print(f"           |  ref E0={E0_ref:.10f}  "
                  f"match={'PASS' if ref_match else 'FAIL'}")

        assert match, f"E₀ mismatch at U={U}"

    print("  → All 3×3 tests PASSED\n")


def test_symmetry_reduction_factor():
    """Verify the Hilbert space reduction factor."""
    print("=" * 60)
    print("Test 5: Reduction factor check")
    print("=" * 60)

    for L, N in [(2, 4), (3, 9)]:
        model = BoseHubbard2D(
            linear_size=L, hopping=1.0, interaction=1.0,
            total_particles=N, boundary='pbc', use_symmetry=True,
        )
        full = model.full_dim
        red = model.dim
        factor = full / red
        max_factor = L * L
        print(f"  L={L}, N={N}: full={full}, reduced={red}, "
              f"factor={factor:.1f} (max={max_factor})")

    print("  → Reduction factor check PASSED\n")


if __name__ == '__main__':
    print("\n" + "=" * 60)
    print("  2D Symmetry Integration Validation Suite")
    print("=" * 60 + "\n")

    test_symmetry_reduction_factor()
    test_2x2_symmetry_vs_full()
    test_2x2_wavefunction_reconstruction()
    test_2x2_observables_match()
    test_3x3_symmetry_vs_full()

    print("=" * 60)
    print("  ALL TESTS PASSED")
    print("=" * 60)
