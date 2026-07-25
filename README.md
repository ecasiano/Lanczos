# Lanczos ED

A fast, simple desktop application for exact diagonalization of quantum lattice models. **Download, double-click, and go** — no terminal, no Python install, no dependencies.

<!-- screenshot placeholder: replace with actual screenshot -->
<!-- ![Lanczos ED screenshot](docs/screenshot.png) -->

## Option 1: Download the app

Grab the latest pre-built app from [**Releases**](https://github.com/ecasiano/Lanczos/releases):

| Platform | File | Status |
|----------|------|--------|
| **macOS** | `Lanczos.ED.dmg` | Available |
| **Windows** | `Lanczos.ED.Setup.exe` | Coming soon |
| **Linux** | `Lanczos.ED.AppImage` | Coming soon |

> **macOS Gatekeeper note:** If macOS says "app can't be opened because it is from an unidentified developer," right-click the app → **Open** → click **Open** again in the dialog.

That's it. Open the app, pick your lattice, set your parameters, and run.

## Option 2: Build the app yourself

If you'd rather build the `.app` / `.dmg` from source (requires Python 3.9+):

```bash
git clone https://github.com/ecasiano/Lanczos.git
cd Lanczos
pip install -r requirements.txt
pip install pyinstaller Pillow
./build_mac.sh
open "dist/Lanczos ED.app"
```

This produces the same desktop app as the pre-built release, just compiled on your machine.

## Option 3: Run from the terminal

For scripting, batch sweeps, or integration into your own workflow, you can run directly from source without building the app:

```bash
git clone https://github.com/ecasiano/Lanczos.git
cd Lanczos
pip install -r requirements.txt
```

Launch the GUI from the terminal:

```bash
python -m lanczos_ed --gui
```

Or run headless from the command line:

```bash
python -m lanczos_ed --L 8 --N 4 --U 4.0
python -m lanczos_ed --L 4 --n_max 2 --grand_canonical --mu 0.5
python -m lanczos_ed --L 8 --N 4 --boundary obc --solver matrix_free
```

<details>
<summary>Using as a Python library</summary>

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

</details>

## What it computes

Lanczos ED solves quantum lattice Hamiltonians via the Lanczos algorithm at small-to-moderate system sizes where the full many-body Hilbert space fits in memory.

**Models:** Bose-Hubbard on 1D chains, 2D square lattices, 3D cubic lattices, and the kagome lattice (periodic or open boundaries, canonical or grand-canonical, tunable occupation cutoff). Fractional Chern insulator on kagome with complex hopping (C = 1 band) and band-projected interactions at ν = 1/3.

**Observables:** Ground-state energy, density profile ⟨nᵢ⟩, bipartite particle-number fluctuations, von Neumann and Rényi entanglement entropies (S₁, S₂) via sector SVD, accessible entanglement entropy S_acc, particle-partitioned entanglement entropy S₂(nₐ) with chunked BLAS acceleration, symmetry-resolved entanglement per charge sector, particle-number distributions p(nₐ), and topological entanglement entropy (Kitaev-Preskill).

**Performance:** All critical kernels are JIT-compiled with [Numba](https://numba.pydata.org). The matrix-free Lanczos solver computes H|ψ⟩ on-the-fly with parallel threads, avoiding the cost of storing the full sparse matrix. Translational symmetry (with optional reflection) reduces the Hilbert space by a factor of L in 1D and L² in 2D.

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
