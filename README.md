# Lanczos ED

Exact diagonalization of the Bose-Hubbard model on 1D chains, 2D square lattices, and 3D cubic lattices using the Lanczos algorithm. Includes a PySide6 desktop GUI.

## Features

**Models** — Bose-Hubbard Hamiltonian in 1D, 2D, and 3D with periodic or open boundary conditions. Supports canonical (fixed particle number) and grand-canonical (fluctuating number) ensembles, with optional occupation truncation (`n_max`).

**Basis encoding** — States are stored as unary (balls-and-walls) integers following [Barghathi et al., PRB 105, L121116 (2022)](https://doi.org/10.1103/PhysRevB.105.L121116). Each Fock state maps to a single 64-bit integer, making enumeration, lookup, and symmetry operations efficient.

**Symmetry reduction** — Translational symmetry (with optional reflection) in 1D, and full 2D translational symmetry via bitwise orbit finding. Symmetry-reduced Hamiltonians are built in the momentum sector of interest, typically cutting the Hilbert space dimension by a factor of L (1D) or L² (2D).

**Solvers** — Standard sparse diagonalization via ARPACK (`scipy.sparse.linalg.eigsh`), and a matrix-free Lanczos solver that applies the Hamiltonian on-the-fly without storing the full sparse matrix.

**Observables** — Ground state energy, density profile ⟨n_i⟩, bipartite particle number fluctuations F_A, von Neumann and Rényi entanglement entropies (S₁, S₂) via sector-by-sector SVD, generalized accessible entanglement entropy S_α^acc ([Barghathi et al., PRB 2022](https://doi.org/10.1103/PhysRevB.105.L121116)), symmetry-resolved entanglement entropy S₂(n_A) per charge sector, and particle number distributions p(n_A).

**Sweep mode** — Scan over ranges of U/t (or other parameters) in a single run, computing all observables for multiple subregion sizes at each point. Results are saved to `.dat` files with sector-resolved companion files.

**Numba acceleration** — Performance-critical kernels (basis enumeration, symmetry operations, observable computations) are JIT-compiled with Numba. A warmup module pre-compiles all kernels at startup so the first real calculation runs at full speed.

**GUI** — PySide6 desktop interface with parameter input, solver selection, symmetry toggle, sweep mode, formatted results display, and density profile plotting.

## Installation

```bash
git clone https://github.com/ecasiano/Lanczos.git
cd Lanczos
pip install -r requirements.txt
```

Dependencies: `numpy`, `scipy`, `PySide6`, `matplotlib`, and optionally `numba`.

## Usage

### GUI

```bash
python -m lanczos_ed --gui
# or
python -m lanczos_ed.gui
```

### Command line

```bash
# Canonical ensemble: L=6 sites, N=3 particles, U/t=4
python -m lanczos_ed --L 6 --N 3 --U 4.0

# Grand canonical with occupation cap
python -m lanczos_ed --L 4 --n_max 2 --grand_canonical --mu 0.5

# Open boundary conditions
python -m lanczos_ed --L 8 --N 4 --boundary obc
```

### Scripting

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
evals, evecs = solver.solve()

psi = model.reconstruct_wavefunction(solver.ground_state)
results = sweep_observables(psi, model.basis, lambda l: list(range(l)), L_max=4)

for r in results:
    print(f"l={r['l']}  S₂={r['S_2']:.6f}  S₂_acc={r['S_2_acc']:.6f}  F_A={r['F_A']:.6f}")
```

For 2D systems:

```python
from lanczos_ed.models.bose_hubbard_2d import BoseHubbard2D

model = BoseHubbard2D(
    linear_size=3, hopping=1.0, interaction=10.0,
    total_particles=9, boundary='pbc', use_symmetry=True,
)

H = model.hamiltonian()
solver = LanczosSolver(H, num_eigenvalues=1)
solver.solve()
psi = model.reconstruct_wavefunction(solver.ground_state)
```

### 4×4 sweep script

A ready-made sweep over U/t for the 4×4 Bose-Hubbard model at unit filling:

```bash
python run_4x4_sweep.py
```

This produces `sweep_results_4x4.dat` and `sweep_results_4x4_sectors.dat`.

## Project structure

```
lanczos_ed/
├── __init__.py
├── __main__.py              # python -m lanczos_ed entry point
├── cli.py                   # command-line interface
├── basis.py                 # mixed-radix Fock basis (grand canonical)
├── unary_basis.py           # unary (balls-and-walls) basis encoding
├── symmetry.py              # 1D translational + reflection symmetry
├── symmetry_2d.py           # 2D translational symmetry (bitwise orbits)
├── warmup.py                # Numba JIT pre-compilation
├── models/
│   ├── bose_hubbard.py      # 1D Bose-Hubbard
│   ├── bose_hubbard_2d.py   # 2D Bose-Hubbard (square lattice)
│   └── bose_hubbard_3d.py   # 3D Bose-Hubbard (cubic lattice)
├── solvers/
│   ├── lanczos.py           # ARPACK sparse eigensolver wrapper
│   └── matrix_free.py       # matrix-free Lanczos (on-the-fly matvec)
├── observables/
│   └── basic.py             # density, fluctuations, entropies, sweeps
└── gui/
    ├── __main__.py           # python -m lanczos_ed.gui entry point
    └── main_window.py        # PySide6 main window
```

## References

- H. Barghathi, E. Casiano-Diaz, A. Del Maestro, *Operationally accessible entanglement of one-dimensional spinless fermions*, [PRB 105, L121116 (2022)](https://doi.org/10.1103/PhysRevB.105.L121116)

## License

MIT
