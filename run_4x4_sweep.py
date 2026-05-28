"""
4×4 Bose-Hubbard 2D sweep: U/t = 0.5 to 50 (log-spaced), PBC, N=16.

Uses 2D translational symmetry to reduce the Hilbert space.
Computes observables for strip subregions l=1 and l=2.

Usage:
    cd ~/Desktop/WorkingCopies/Lanczos
    python run_4x4_sweep.py
"""

import numpy as np
import time
import sys

from lanczos_ed.models.bose_hubbard_2d import BoseHubbard2D, strip_subregion
from lanczos_ed.solvers.lanczos import LanczosSolver
from lanczos_ed.observables.basic import (
    density_profile, bipartite_fluctuations, entanglement_entropy,
    accessible_entanglement_entropy, sweep_observables,
)

# =====================================================================
# Parameters
# =====================================================================
L = 4
N = L * L           # unit filling
boundary = 'pbc'
use_symmetry = True
l_max = 2            # strip subregion l = 1, 2

U_values = np.logspace(np.log10(0.5), np.log10(50), 30)
t_hop = 1.0
mu = 0.0

output_file = "sweep_results_4x4.dat"

# =====================================================================
# Header
# =====================================================================
print(f"4×4 Bose-Hubbard 2D sweep")
print(f"  L={L}, N={N}, t={t_hop}, mu={mu}, BC={boundary}")
print(f"  U/t range: {U_values[0]:.4f} to {U_values[-1]:.4f} "
      f"({len(U_values)} points, log-spaced)")
print(f"  Symmetry: {use_symmetry} (2D translations, k=(0,0) sector)")
print(f"  Subregion: strip, l = 1..{l_max}")
print(f"  Output: {output_file}")
print()

# =====================================================================
# Run sweep
# =====================================================================
all_rows = []
t_total_start = time.time()

for i, U in enumerate(U_values):
    t0 = time.time()

    # Build model with symmetry
    model = BoseHubbard2D(
        linear_size=L, hopping=t_hop, interaction=U,
        chemical_potential=mu, total_particles=N,
        boundary=boundary, use_symmetry=use_symmetry,
    )

    if i == 0:
        print(f"  Full Hilbert space dim: {model.full_dim:,}")
        print(f"  Reduced dim (symmetry): {model.dim:,}")
        print(f"  Reduction factor: {model.full_dim / model.dim:.1f}×")
        print()

    # Build and solve
    H = model.hamiltonian()
    solver = LanczosSolver(H, num_eigenvalues=1)
    evals, evecs = solver.solve()
    E0 = evals[0]

    # Reconstruct full wavefunction for observables
    psi = model.reconstruct_wavefunction(solver.ground_state)

    # Sweep observables over l = 1..l_max
    def make_subsystem(l):
        return model.get_subregion('strip', l)

    sweep_data = sweep_observables(psi, model.basis, make_subsystem, l_max)

    dt = time.time() - t0

    # Store results
    for entry in sweep_data:
        all_rows.append({
            'U': U,
            'E0': E0,
            'dim': model.dim,
            'full_dim': model.full_dim,
            'l': entry['l'],
            'num_sites_A': entry['num_sites_A'],
            'F_A': entry['F_A'],
            'S_1': entry['S_1'],
            'S_2': entry['S_2'],
            'S_2_acc': entry['S_2_acc'],
            'sector_probs': entry.get('sector_probs', {}),
            'sector_S_2': entry.get('sector_S_2', {}),
            'time': dt,
        })

    print(f"  [{i+1:2d}/{len(U_values)}]  U/t={U:10.4f}  "
          f"E0={E0:16.10f}  dim={model.dim}  ({dt:.1f}s)")

    # Clear cached Hamiltonian to free memory
    del model, H, solver, psi
    import gc; gc.collect()

t_total = time.time() - t_total_start
print(f"\nTotal time: {t_total:.1f}s")

# =====================================================================
# Write output
# =====================================================================
print(f"\nWriting {output_file} ...")

with open(output_file, 'w') as f:
    f.write("# Lanczos ED — 4×4 Bose-Hubbard 2D Sweep\n")
    f.write(f"# Date: {time.strftime('%Y-%m-%d %H:%M:%S')}\n")
    f.write(f"# L={L}, N={N}, t={t_hop}, mu={mu}, BC={boundary}\n")
    f.write(f"# Symmetry: 2D translations, (kx,ky)=(0,0) sector\n")
    f.write(f"# U/t range: {U_values[0]:.6f} to {U_values[-1]:.6f} "
            f"({len(U_values)} points, log-spaced)\n")
    f.write(f"# Total time: {t_total:.1f}s\n")
    f.write("#\n")

    hdr = (
        f"# {'model':>5s}  {'L':>4s}  {'t':>10s}  {'U/t':>12s}  "
        f"{'mu/t':>12s}  {'nmax':>5s}  {'N':>5s}  {'BC':>4s}  "
        f"{'solver':>10s}  {'dim':>10s}  {'E_0':>16s}  "
        f"{'l':>4s}  {'|A|':>5s}  {'F_A':>14s}  {'S_1':>14s}  "
        f"{'S_2':>14s}  {'S_2_acc':>14s}  {'time_s':>8s}"
    )
    f.write(hdr + "\n")
    f.write("#" + "-" * (len(hdr) - 1) + "\n")

    for r in all_rows:
        f.write(
            f"     2D  {L:4d}  {t_hop:10.6f}  {r['U']:12.6f}  "
            f"{mu:12.6f}  {N:5d}  {N:5d}  {boundary:>4s}  "
            f"{'standard':>10s}  {r['dim']:10d}  {r['E0']:16.10f}  "
            f"{r['l']:4d}  {r['num_sites_A']:5d}  "
            f"{r['F_A']:14.10f}  {r['S_1']:14.10f}  "
            f"{r['S_2']:14.10f}  {r['S_2_acc']:14.10f}  "
            f"{r['time']:8.3f}\n"
        )

# =====================================================================
# Write sector-resolved data (p(n_A), S₂(n_A)) to separate file
# =====================================================================
sector_file = output_file.replace('.dat', '_sectors.dat')
print(f"Writing {sector_file} ...")

with open(sector_file, 'w') as f:
    f.write("# Lanczos ED — Sector-Resolved Data\n")
    f.write(f"# Date: {time.strftime('%Y-%m-%d %H:%M:%S')}\n")
    f.write(f"# L={L}, N={N}, t={t_hop}, mu={mu}, BC={boundary}\n")
    f.write("# Particle number distribution p(n_A) and "
            "symmetry-resolved S_2(n_A)\n")
    f.write("#\n")

    shdr = (
        f"# {'U/t':>12s}  {'l':>4s}  {'|A|':>5s}  {'n_A':>5s}  "
        f"{'p(n_A)':>14s}  {'S_2(n_A)':>14s}"
    )
    f.write(shdr + "\n")
    f.write("#" + "-" * (len(shdr) - 1) + "\n")

    for r in all_rows:
        s_probs = r.get('sector_probs', {})
        s_S2 = r.get('sector_S_2', {})
        for n in sorted(s_probs.keys()):
            f.write(
                f"  {r['U']:12.6f}  {r['l']:4d}  "
                f"{r['num_sites_A']:5d}  {n:5d}  "
                f"{s_probs[n]:14.10f}  "
                f"{s_S2.get(n, 0.0):14.10f}\n"
            )

print(f"Done. Results saved to {output_file} and {sector_file}")
