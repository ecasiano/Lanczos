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
    particle_partition_entropy,
)


def parse_args(argv=None):
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description="Lanczos Exact Diagonalization for quantum lattice models",
    )

    # Model selection
    parser.add_argument(
        "--model", default="bose_hubbard",
        choices=["bose_hubbard", "fci"],
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
    parser.add_argument(
        "--hardcore", action="store_true",
        help="Hardcore bosons (n_max=1 forced; U term is then trivially "
             "zero, so interaction physics lives in --V)",
    )
    parser.add_argument(
        "--V", type=float, default=0.0,
        help="Nearest-neighbor (extended) interaction strength V "
             "(Extended Bose-Hubbard Model): V * sum_<i,j> n_i n_j. "
             "Default: 0.0 (standard Bose-Hubbard)",
    )

    # FCI parameters
    parser.add_argument(
        "--N1", type=int, default=3,
        help="FCI torus dimension N₁ (default: 3)",
    )
    parser.add_argument(
        "--N2", type=int, default=4,
        help="FCI torus dimension N₂ (default: 4)",
    )
    parser.add_argument(
        "--phi", type=float, default=5*np.pi/4,
        help="FCI hopping phase φ (default: 5π/4)",
    )
    parser.add_argument(
        "--kappa", type=float, default=0.0,
        help="FCI band dispersion weight κ (0 = flat band, default: 0)",
    )
    parser.add_argument(
        "--kp_radius", type=float, default=1.0,
        help="Kitaev-Preskill disk radius for TEE (default: 1.0)",
    )
    parser.add_argument(
        "--tee", action="store_true",
        help="Compute Kitaev-Preskill TEE decomposition (FCI only)",
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
    if args.model == "fci":
        # FCI uses --N1, --N2 instead of --L, --N
        pass
    else:
        if args.L is None:
            parser.error("--L is required for CLI mode")
        if not args.grand_canonical and args.N is None:
            parser.error(
                "Must specify --N for canonical ensemble, "
                "or use --grand_canonical"
            )
        if args.grand_canonical and args.n_max is None:
            parser.error(
                "Must specify --n_max for grand canonical ensemble"
            )

    return args


def _run_fci(args):
    """Run the Fractional Chern Insulator workflow.

    Builds the band-projected Hamiltonian, solves for the ground state,
    transforms to real space, and computes the entanglement decomposition
    S = S_acc + H (and optionally KP TEE).
    """
    from .models.fractional_chern import FractionalChernInsulator
    from .observables.basic import (
        entanglement_entropy, accessible_entanglement_entropy,
    )

    time_start = time.time()

    # Build model
    model = FractionalChernInsulator(
        N1=args.N1, N2=args.N2,
        hopping_phase=args.phi,
        interaction=1.0,
        band_dispersion_weight=args.kappa,
    )

    # Solve for ground state(s)
    num_eig = max(args.num_states, 6)
    eigenvalues, eigenvectors = model.solve(num_eigenvalues=num_eig)
    time_solve = time.time()

    print(f"\nEigenvalues (lowest {min(8, len(eigenvalues))}):")
    for i in range(min(8, len(eigenvalues))):
        print(f"  E_{i} = {eigenvalues[i]:.10f}")

    spread = eigenvalues[2] - eigenvalues[0]
    gap_3 = eigenvalues[3] - eigenvalues[2]
    print(f"\n  3-fold GSD spread  δ₃ = {spread:.6e}")
    print(f"  Gap above triplet  Δ₃ = {gap_3:.6f}")
    print(f"  δ₃/Δ₃             = {spread/gap_3:.6e}")
    print(f"  Solve time: {time_solve - time_start:.1f}s")

    # Transform ground state to real space
    psi_k = eigenvectors[:, 0]
    psi_real, real_basis = model.transform_to_real_space(psi_k)
    time_transform = time.time()
    print(f"  Transform time: {time_transform - time_solve:.1f}s")

    # Bipartite entanglement (half-system cut)
    N_sites = model.N_sites
    subsys_half = list(range(N_sites // 2))

    S = entanglement_entropy(psi_real, real_basis, subsys_half, renyi_index=1)
    Sacc = accessible_entanglement_entropy(
        psi_real, real_basis, subsys_half, renyi_index=1
    )
    H = S - Sacc
    print(f"\nBipartite entanglement ({len(subsys_half)} of {N_sites} sites):")
    print(f"  S     = {S:.6f}")
    print(f"  S_acc = {Sacc:.6f}")
    print(f"  H     = {H:.6f}")

    # Kitaev-Preskill TEE decomposition
    if args.tee:
        from .observables.tee import (
            kitaev_preskill_regions, topological_entanglement_entropy,
        )
        # kitaev_preskill_regions expects an object with .positions()
        # We use the model itself (it has .positions())
        regA, regB, regC = kitaev_preskill_regions(
            model, radius=args.kp_radius
        )
        disk = regA + regB + regC
        env = [s for s in range(N_sites) if s not in set(disk)]

        print(f"\nKP TEE (radius={args.kp_radius}):")
        print(f"  Disk: {len(disk)} sites "
              f"({len(regA)}+{len(regB)}+{len(regC)})")
        print(f"  Environment: {len(env)} sites")

        if len(regA) == 0 or len(regB) == 0 or len(regC) == 0:
            print("  ERROR: empty KP region — increase radius")
        elif len(env) < 3:
            print("  WARNING: very small environment — TEE may not converge")
        else:
            result = topological_entanglement_entropy(
                psi_real, real_basis, regA, regB, regC,
            )
            gamma = result['gamma']
            gamma_acc = result['gamma_acc']
            gamma_H = result['gamma_H']
            target = 0.5 * np.log(3)

            print(f"\n  γ       = {gamma:.6f}  (target: {target:.6f})")
            print(f"  γ_acc   = {gamma_acc:.6f}")
            print(f"  γ_H     = {gamma_H:.6f}")

    total_time = time.time() - time_start
    print(f"\nTotal time: {total_time:.1f}s")


def main(argv=None):
    """Main entry point for CLI execution."""
    args = parse_args(argv)

    # If --gui flag is set, launch the graphical interface
    if args.gui:
        from .gui.main_window import run_gui
        run_gui()
        return

    # FCI model has its own workflow
    if args.model == "fci":
        _run_fci(args)
        return

    # Pre-compile Numba kernels (blocking for CLI so first run is fast)
    from .warmup import warmup
    print("Pre-compiling Numba kernels…", end=" ", flush=True)
    dt_warmup = warmup()
    print(f"done ({dt_warmup:.1f}s)")

    total_particles = None if args.grand_canonical else args.N

    # Print header
    print(f"{'=' * 60}")
    print(f"Lanczos ED — 1D Bose-Hubbard Model")
    print(f"{'=' * 60}")
    print(f"  Sites (L):            {args.L}")
    print(f"  Hopping (t):          {args.t}")
    print(f"  Interaction (U):      {args.U}")
    if args.V != 0.0:
        print(f"  NN interaction (V):   {args.V}")
    print(f"  Chemical pot. (mu):   {args.mu}")
    if args.hardcore:
        print(f"  Hardcore bosons:      True (n_max=1)")
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
        hardcore=args.hardcore,
        nn_interaction=args.V,
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

    # Particle-partitioned entanglement entropy
    N_particles = getattr(basis, 'total_particles', None)
    if N_particles is not None and N_particles > 1:
        print("\nParticle-partitioned entanglement entropy S₂(n_A):")
        for n_A in range(1, N_particles):
            s2_ppee = particle_partition_entropy(
                ground_state_wfn, basis, n_A,
            )
            print(f"  n_A = {n_A}:  S_2 = {s2_ppee:.10f}")

    total_time = time.time() - time_start
    print(f"\nTotal time: {total_time:.3f}s")
