# Lanczos ED

A fast, simple desktop application for exact diagonalization of quantum lattice models. Double-click the app, pick your parameters, and go — no terminal required.

<!-- screenshot placeholder: replace with actual screenshot -->
<!-- ![Lanczos ED screenshot](docs/screenshot.png) -->

## Download

Pre-built apps are available on the [Releases](https://github.com/ecasiano/Lanczos/releases) page:

| Platform | Download |
|----------|----------|
| macOS    | `Lanczos.ED.dmg` |
| Windows  | `Lanczos.ED.Setup.exe` (coming soon) |
| Linux    | `Lanczos.ED.AppImage` (coming soon) |

On macOS, if you see "app can't be opened because it is from an unidentified developer," right-click the app and choose **Open**.

## What it does

Lanczos ED solves quantum lattice Hamiltonians by exact diagonalization using the Lanczos algorithm. It targets small-to-moderate system sizes where the full many-body Hilbert space is tractable, and computes ground-state energies, entanglement entropies, density profiles, and topological invariants.

### Supported models

- **Bose-Hubbard** — 1D chains, 2D square lattices, 3D cubic lattices, and the kagome lattice. Periodic or open boundary conditions, canonical or grand-canonical ensemble, tunable occupation cutoff.
- **Fractional Chern insulator** — Kagome lattice with complex nearest-neighbor hopping (Chern band C = 1) and band-projected interactions at fractional filling ν = 1/3. Includes momentum-resolved spectrum and real-space transform for entanglement.

### Observables

Ground-state energy, density profile ⟨nᵢ⟩, bipartite particle-number fluctuations, von Neumann and Rényi entanglement entropies (S₁, S₂) via sector-by-sector SVD, accessible entanglement entropy S_acc, particle-partitioned entanglement, symmetry-resolved entanglement S₂(nₐ) per charge sector, particle-number distributions p(nₐ), and topological entanglement entropy (Kitaev-Preskill).

### Performance

All performance-critical kernels (basis enumeration, Hamiltonian application, symmetry operations, observable computation) are JIT-compiled with [Numba](https://numba.pydata.org). The matrix-free Lanczos solver computes H|ψ⟩ on-the-fly with parallel threads, avoiding the memory cost of storing the full sparse matrix. Translational symmetry (with optional reflection) reduces the Hilbert space by a factor of L in 1D and L² in 2D.

## Running from source

If you prefer to run from source rather than the desktop app:

```bash
git clone https://github.com/ecasiano/Lanczos.git
cd Lanczos
pip install -r requirements.txt
```

Launch the GUI:

```bash
python -m lanczos_ed --gui
```

### Command line

```bash
python -m lanczos_ed --L 6 --N 3 --U 4.0
python -m lanczos_ed --L 4 --n_max 2 --grand_canonical --mu 0.5
python -m lanczos_ed --L 8 --N 4 --boundary obc --solver matrix_free
```

### As a library

```python
from lanczos_ed.models.bose_hubbard import BoseHubbard1D
from lanczos_ed.solvers.lanczos import LanczosSolver
from lanczos_ed.observables.basic import sweep_observables

model = BoseHubbard1D(
    num_sites=8, hopping=1.0, interaction=4.0,
    total_particles=8, boundary='pbc', use_symmetry=True,
)

H = model.hamiltonian()
solver = LanczosSolver(H, num_eigenvalues=1)
solver.solve()

psi = model.reconstruct_wavefunction(solver.ground_state)
results = sweep_observables(psi, model.basis, lambda l: list(range(l)), L_max=4)

for r in results:
    print(f"l={r['l']}  S₂={r['S_2']:.6f}  S₂_acc={r['S_2_acc']:.6f}")
```

## Building the desktop app

To build the `.app` bundle yourself (macOS):

```bash
cd Lanczos
pip install pyinstaller Pillow
./build_mac.sh
open "dist/Lanczos ED.app"
```

## Project structure

```
lanczos_ed/
├── basis.py                 # mixed-radix Fock basis (grand canonical)
├── unary_basis.py           # unary (balls-and-walls) basis encoding
├── symmetry.py              # 1D translational + reflection symmetry
├── symmetry_2d.py           # 2D translational symmetry (bitwise orbits)
├── warmup.py                # Numba JIT pre-compilation at startup
├── models/
│   ├── bose_hubbard.py      # 1D Bose-Hubbard
│   ├── bose_hubbard_2d.py   # 2D square lattice
│   ├── bose_hubbard_3d.py   # 3D cubic lattice
│   ├── bose_hubbard_kagome.py  # kagome lattice
│   └── fractional_chern.py  # FCI on kagome (band-projected)
├── solvers/
│   ├── lanczos.py           # ARPACK sparse eigensolver
│   └── matrix_free.py       # matrix-free Lanczos (Numba parallel)
├── observables/
│   ├── basic.py             # density, fluctuations, entropies, sweeps
│   ├── ppee.py              # particle-partitioned entropy (chunked BLAS)
│   └── tee.py               # topological entanglement entropy
└── gui/
    └── main_window.py       # PySide6 desktop interface
```

## References

- H. Barghathi, E. Casiano-Diaz, A. Del Maestro, *Operationally accessible entanglement of one-dimensional spinless fermions*, [PRB 105, L121116 (2022)](https://doi.org/10.1103/PhysRevB.105.L121116)

## License

MIT
