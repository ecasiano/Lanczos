"""
Command-line interface for Lanczos ED.

Usage examples
--------------
Canonical ensemble (fixed particle number):
    python -m lanczos_ed --L 6 --N 3 --U 4.0

Grand canonical ensemble (fluctuating particle number):
    python -m lanczos_ed --L 4 --n_max 2 --grand_canonical --mu 0.5

Open boundary conditions:
    python -m lanczos_ed --L 8 --N 4 --boundary obc

Launch the GUI:
    python -m lanczos_ed --gui
"""

import argparse
import numpy as np
import sys
import time

from .models.bose_hubbard import BoseHubbard1D
from .solvers.lanczos import LanczosSolver
from .observables.basic import (
    density_profile, bipartite_fluctuations, entanglement_entropy,
)


def parse_args(argv=None):
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description="Lanczos Exact Diagonalization for quantum lattice models",
    )

    # Model selection
    parser.add_argument(
        "--model", default="bose_hubbard",
        choices=["bose_hubbard"],
        help="Model to diagonalize (default: bose_hubbard)",
    )

    # Lattice parameters
    parser.add_argument(
        "--L", type=int, default=None,
        help="Number of lattice sites (required for CLI mode)",
    )
    parser.add_argument(
        "--boundary", default="pbc",
        choices=["pbc", "obc"],
        help="Boundary conditions: pbc (periodic) or obc (open). Default: pbc",
    )

    # Bose-Hubbard parameters
    parser.add_argument(
        "--t", type=float, default=1.0,
        help="Hopping amplitude (default: 1.0)",
    )
    parser.add_argument(
        "--U", type=float, default=1.0,
        help="On-site interaction strength (default: 1.0)",
    )
    parser.add_argument(
        "--mu", type=float, default=0.0,
        help="Chemical potential (default: 0.0)",
    )
    parser.add_argument(
        "--n_max", type=int, default=None,
        help="Max occupation per site. Default: N for canonical, "
             "required for grand canonical",
    )

    # Ensemble
    parser.add_argument(
        "--N", type=int, default=None,
        help="Total particle number (canonical ensemble)",
    )
    parser.add_argument(
        "--grand_canonical", action="store_true",
        help="Use grand canonical ensemble (no fixed particle number)",
    )

    # Solver options
    parser.add_argument(
        "--num_states", type=int, default=1,
        help="Number of lowest eigenvalues to compute (default: 1)",
    )

    # Observable options
    parser.add_argument(
        "--renyi_alpha", type=float, nargs='*', default=[1.0, 2.0],
        help="Rényi indices for entanglement entropy (default: 1.0 2.0)",
    )

    # GUI mode
    parser.add_argument(
        "--gui", action="store_true",
        help="Launch the graphical interface instead of CLI",
    )

    args = parser.parse_args(argv)

    # In GUI mode, no other arguments are required
    if args.gui:
        return args

    # In CLI mode, validate required parameters
    if args.L is None:
        parser.error("--L is required for CLI mode")
    if not args.grand_canonical and args.N is None:
        parser.error(
            "Must specify --N for canonical ensemble, or use --grand_canonical"
        )
    if args.grand_canonical and args.n_max is None:
        parser.error(
            "Must specify --n_max for grand canonical ensemble"
        )

    return args


def main(argv=None):
    """Main entry point for CLI execution."""
    args = parse_args(argv)

    # If --gui flag is set, launch the graphical interface
    if args.gui:
        from .gui.main_window import run_gui
        run_gui()
        return

    total_particles = None if args.grand_canonical else args.N

    # Print header
    print(f"{'=' * 60}")
    print(f"Lanczos ED — 1D Bose-Hubbard Model")
    print(f"{'=' * 60}")
    print(f"  Sites (L):            {args.L}")
    print(f"  Hopping (t):          {args.t}")
    print(f"  Interaction (U):      {args.U}")
    print(f"  Chemical pot. (mu):   {args.mu}")
    nmax_display = args.n_max if args.n_max else f"unrestricted (= N = {args.N})"
    print(f"  Max occupation:       {nmax_display}")
    if args.grand_canonical:
        print(f"  Ensemble:             Grand Canonical")
    else:
        print(f"  Ensemble:             Canonical (N = {args.N})")
    print(f"  Boundary:             {args.boundary.upper()}")
    print()

    # Build the model
    time_start = time.time()

    model = BoseHubbard1D(
        num_sites=args.L,
        hopping=args.t,
        interaction=args.U,
        chemical_potential=args.mu,
        max_occupation=args.n_max,
        total_particles=total_particles,
        boundary=args.boundary,
    )
    print(f"Hilbert space dimension: {model.dim}")

    # Build the Hamiltonian
    hamiltonian = model.hamiltonian()
    time_hamiltonian = time.time()
    print(f"Hamiltonian built in {time_hamiltonian - time_start:.3f}s "
          f"(non-zero elements: {hamiltonian.nnz})")

    # Lanczos diagonalization
    solver = LanczosSolver(hamiltonian, num_eigenvalues=args.num_states)
    eigenvalues, eigenvectors = solver.solve()
    time_diag = time.time()
    print(f"Diagonalization done in {time_diag - time_hamiltonian:.3f}s")
    print()

    # Print eigenvalues
    print("Eigenvalues:")
    for i, energy in enumerate(eigenvalues):
        print(f"  E_{i} = {energy:.12f}")
    print(f"\nGround state energy:     {eigenvalues[0]:.12f}")
    print(f"Energy per site:         {eigenvalues[0] / args.L:.12f}")
    print()

    # Ground state observables
    ground_state_wfn = solver.ground_state
    basis = model.basis

    density = density_profile(ground_state_wfn, basis)
    print("Density profile <n_i>:")
    for site, n_i in enumerate(density):
        print(f"  site {site}: {n_i:.8f}")
    print(f"  Total:  {density.sum():.8f}")
    print()

    fluctuation = bipartite_fluctuations(ground_state_wfn, basis)
    print(f"Bipartite fluctuations F_A (L/2 cut): {fluctuation:.10f}")

    for alpha in args.renyi_alpha:
        entropy = entanglement_entropy(
            ground_state_wfn, basis, renyi_index=alpha
        )
        if abs(alpha - 1.0) < 1e-10:
            print(f"Von Neumann entropy S_1 (L/2 cut):    {entropy:.10f}")
        else:
            print(f"Rényi-{alpha} entropy S_{alpha} (L/2 cut):      {entropy:.10f}")

    total_time = time.time() - time_start
    print(f"\nTotal time: {total_time:.3f}s")
