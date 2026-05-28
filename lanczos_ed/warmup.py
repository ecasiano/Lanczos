"""
Pre-compile all Numba JIT kernels with tiny dummy inputs.

First-time Numba compilation can take several seconds per function.
Calling warmup() once at startup ensures all kernels are compiled
before the user runs a real calculation. Subsequent calls are fast
(cache hit).

Usage:
    from lanczos_ed.warmup import warmup
    warmup()            # blocking
    warmup_async()      # returns immediately, compiles in background
"""

import threading
import time
import numpy as np

_warmup_done = False
_warmup_lock = threading.Lock()


def warmup(on_progress=None):
    """Pre-compile all Numba kernels with tiny dummy problems.

    Parameters
    ----------
    on_progress : callable or None
        Optional callback: on_progress(step_name, step_num, total_steps).
        Useful for GUI status updates during warmup.

    Returns
    -------
    float
        Time taken in seconds.
    """
    global _warmup_done
    with _warmup_lock:
        if _warmup_done:
            return 0.0

    t0 = time.time()
    total = 6
    step = 0

    def _report(name):
        nonlocal step
        step += 1
        if on_progress:
            on_progress(name, step, total)

    # ------------------------------------------------------------------
    # 1. Basis enumeration (unary_basis._recurse_numba)
    # ------------------------------------------------------------------
    from .unary_basis import UnaryBasis
    _report("Basis enumeration")
    basis_1d = UnaryBasis(num_sites=4, total_particles=4)

    # ------------------------------------------------------------------
    # 2. 1D symmetry (shift, reflect, find_cycles, build_reduced_H)
    # ------------------------------------------------------------------
    _report("1D symmetry")
    try:
        from .symmetry import find_cycles, build_reduced_hamiltonian
        integers = basis_1d._integers.astype(np.int64)
        total_bits = np.int64(4 + 4)  # N + L
        find_cycles(basis_1d, use_reflection=True)
        # Trigger build_reduced_hamiltonian via a tiny 1D model
        from .models.bose_hubbard_1d import BoseHubbard1D
        m1d = BoseHubbard1D(
            num_sites=4, hopping=1.0, interaction=1.0,
            total_particles=4, boundary='pbc', use_symmetry=True,
        )
        m1d.hamiltonian()
    except Exception:
        pass  # symmetry module may not import without scipy

    # ------------------------------------------------------------------
    # 3. 2D symmetry (bitwise translations, orbit finding, reduced H)
    # ------------------------------------------------------------------
    _report("2D symmetry")
    try:
        from .symmetry_2d import find_orbits_2d, build_reduced_hamiltonian_2d
        basis_2d = UnaryBasis(num_sites=4, total_particles=4)
        find_orbits_2d(basis_2d, L=2)
        from .models.bose_hubbard_2d import BoseHubbard2D
        m2d = BoseHubbard2D(
            linear_size=2, hopping=1.0, interaction=1.0,
            total_particles=4, boundary='pbc', use_symmetry=True,
        )
        m2d.hamiltonian()
    except Exception:
        pass

    # ------------------------------------------------------------------
    # 4. Lanczos solver
    # ------------------------------------------------------------------
    _report("Lanczos solver")
    try:
        from .solvers.lanczos import LanczosSolver
        H = m1d.hamiltonian() if 'm1d' in dir() else None
        if H is not None:
            solver = LanczosSolver(H, num_eigenvalues=1)
            solver.solve()
    except Exception:
        pass

    # ------------------------------------------------------------------
    # 5. Observable kernels (density, fluctuations, charges, SVD)
    # ------------------------------------------------------------------
    _report("Observables")
    try:
        from .observables.basic import (
            density_profile, bipartite_fluctuations,
            entanglement_entropy, sweep_observables,
        )
        # Use the 1D model ground state
        if 'solver' in dir() and hasattr(solver, 'ground_state'):
            psi = solver.ground_state
            density_profile(psi, basis_1d)
            bipartite_fluctuations(psi, basis_1d, subsystem_sites=[0, 1])
            sweep_observables(psi, basis_1d, lambda l: list(range(l)), 2)
    except Exception:
        pass

    # ------------------------------------------------------------------
    # 6. Matrix-free matvec (if available)
    # ------------------------------------------------------------------
    _report("Matrix-free solver")
    try:
        from .solvers.matrix_free import MatrixFreeHamiltonian
        if 'm1d' in dir():
            mfh = MatrixFreeHamiltonian(m1d)
            x = np.random.randn(m1d.dim)
            mfh.matvec(x)
    except Exception:
        pass

    dt = time.time() - t0
    with _warmup_lock:
        _warmup_done = True

    return dt


def warmup_async(on_progress=None, on_done=None):
    """Run warmup in a background thread.

    Parameters
    ----------
    on_progress : callable or None
        on_progress(step_name, step_num, total_steps)
    on_done : callable or None
        on_done(elapsed_seconds) called when warmup finishes.

    Returns
    -------
    threading.Thread
        The background thread (already started).
    """
    def _worker():
        dt = warmup(on_progress=on_progress)
        if on_done:
            on_done(dt)

    t = threading.Thread(target=_worker, daemon=True, name="numba-warmup")
    t.start()
    return t
