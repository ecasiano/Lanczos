"""
PySide6 GUI for Lanczos Exact Diagonalization.

Provides:
    - Parameter input panel (model parameters, ensemble, boundary conditions)
    - Solver selection (standard sparse diagonalization or matrix-free Lanczos)
    - Symmetry reduction toggle (translational + reflection, PBC only)
    - Optional n_max truncation (unchecked = unrestricted occupation)
    - Run button with background thread execution
    - Results tab with formatted text output
    - Density profile plot tab (requires matplotlib)
"""

import sys
import time
import numpy as np

from itertools import product as iterproduct
from pathlib import Path

from PySide6.QtWidgets import (
    QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout,
    QGridLayout, QLabel, QComboBox, QPushButton, QCheckBox,
    QGroupBox, QTextEdit, QTabWidget, QSpinBox, QDoubleSpinBox,
    QMessageBox, QFileDialog, QProgressBar, QScrollArea, QFrame,
)
from PySide6.QtCore import Qt, QThread, Signal

from ..models.bose_hubbard import BoseHubbard1D
from ..models.bose_hubbard_2d import BoseHubbard2D, strip_subregion, square_subregion
from ..models.bose_hubbard_3d import BoseHubbard3D, slab_subregion, column_subregion, cube_subregion
from ..solvers.lanczos import LanczosSolver
from ..observables.basic import (
    density_profile, bipartite_fluctuations, entanglement_entropy,
    accessible_entanglement_entropy, sweep_observables,
)

# Optional: matplotlib for plotting density profiles
try:
    from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg
    from matplotlib.figure import Figure
    HAS_MATPLOTLIB = True
except ImportError:
    HAS_MATPLOTLIB = False


class SolverWorker(QThread):
    """Background thread for running the diagonalization.

    Runs the full pipeline (build basis -> Hamiltonian -> solve -> observables)
    without blocking the GUI. Supports three solver modes:
        - Standard:     build sparse H, use ARPACK eigsh
        - Matrix-free:  compute H|ψ⟩ on-the-fly, no H stored
    And optional symmetry reduction (q=0, R=+1 sector for PBC).
    """
    finished = Signal(dict)
    error = Signal(str)

    def __init__(self, parameters: dict):
        super().__init__()
        self.parameters = parameters

    def run(self):
        try:
            p = self.parameters
            t_start = time.time()
            model_type = p.get('model_type', '1D')

            # Build the model (1D, 2D, or 3D)
            if model_type == '3D':
                model = BoseHubbard3D(
                    linear_size=p['linear_size'],
                    hopping=p['hopping'],
                    interaction=p['interaction'],
                    chemical_potential=p['chemical_potential'],
                    max_occupation=p['max_occupation'],
                    total_particles=p['total_particles'],
                    boundary=p['boundary'],
                    use_symmetry=False,
                )
            elif model_type == '2D':
                model = BoseHubbard2D(
                    linear_size=p['linear_size'],
                    hopping=p['hopping'],
                    interaction=p['interaction'],
                    chemical_potential=p['chemical_potential'],
                    max_occupation=p['max_occupation'],
                    total_particles=p['total_particles'],
                    boundary=p['boundary'],
                    use_symmetry=p.get('use_symmetry', False),
                )
            else:
                model = BoseHubbard1D(
                    num_sites=p['num_sites'],
                    hopping=p['hopping'],
                    interaction=p['interaction'],
                    chemical_potential=p['chemical_potential'],
                    max_occupation=p['max_occupation'],
                    total_particles=p['total_particles'],
                    boundary=p['boundary'],
                    use_symmetry=p.get('use_symmetry', False),
                )

            t_basis = time.time()

            # Solve for eigenvalues/eigenvectors
            num_evals = p.get('num_eigenvalues', 1)
            solver_type = p.get('solver', 'standard')

            if solver_type == 'matrix_free':
                # Matrix-free: H|ψ⟩ computed on-the-fly, no H stored.
                from ..solvers.matrix_free import solve_matrix_free
                if model_type in ('2D', '3D'):
                    model_for_mf = model  # matrix-free needs unsymmetrized
                else:
                    model_for_mf = BoseHubbard1D(
                        num_sites=p['num_sites'],
                        hopping=p['hopping'],
                        interaction=p['interaction'],
                        chemical_potential=p['chemical_potential'],
                        max_occupation=p['max_occupation'],
                        total_particles=p['total_particles'],
                        boundary=p['boundary'],
                        use_symmetry=False,
                    )
                eigenvalues, eigenvectors = solve_matrix_free(
                    model_for_mf, num_eigenvalues=num_evals,
                )
                ground_state_wfn = eigenvectors[:, 0]
                basis_for_obs = model_for_mf.basis

            else:
                # Standard: build sparse H (full or symmetry-reduced)
                hamiltonian = model.hamiltonian()

                solver = LanczosSolver(
                    hamiltonian,
                    num_eigenvalues=num_evals,
                )
                eigenvalues, eigenvectors = solver.solve()

                if model.use_symmetry:
                    ground_state_wfn = model.reconstruct_wavefunction(
                        solver.ground_state
                    )
                else:
                    ground_state_wfn = solver.ground_state

                basis_for_obs = model.basis

            t_solve = time.time()

            # =============================================================
            # Observables: single diagonalization, sweep over l values
            #
            # For 2D: sweep l = 1..l_max for the chosen subregion type
            # For 1D: sweep l = 1..L//2 (sites 0..l-1)
            #
            # Each l value re-uses the same ground state |ψ₀⟩.
            # =============================================================

            # Density profile (independent of subregion)
            t_obs_start = time.time()
            obs_density = density_profile(ground_state_wfn, basis_for_obs)
            t_density = time.time() - t_obs_start

            # Determine l-sweep range and subregion generator
            if model_type in ('2D', '3D'):
                L_lin = p['linear_size']
                sub_type = p.get('subregion_type',
                                 'slab' if model_type == '3D' else 'strip')
                l_max = p.get('subregion_l', L_lin // 2)

                def make_subsystem(l):
                    return model.get_subregion(sub_type, l)
            else:
                L_lin = p['num_sites']
                sub_type = 'strip'
                l_max = L_lin // 2

                def make_subsystem(l):
                    return list(range(l))

            # Optimized l-sweep: one decode pass, one SVD per l
            t_sweep_start = time.time()
            sweep_data = sweep_observables(
                ground_state_wfn, basis_for_obs,
                make_subsystem, l_max,
            )
            t_sweep = time.time() - t_sweep_start
            t_end = time.time()

            # For backward compatibility, use l_max as the "primary" cut
            primary = sweep_data[-1] if sweep_data else {}

            result = {
                'model_description': repr(model),
                'model_type': model_type,
                'hilbert_dim': model.dim,
                'full_dim': model.full_dim if hasattr(model, 'full_dim') else model.dim,
                'eigenvalues': eigenvalues,
                'ground_state_energy': eigenvalues[0],
                'density': obs_density,
                'bipartite_fluctuations': primary.get('F_A', 0.0),
                'von_neumann_entropy': primary.get('S_1', 0.0),
                'renyi_2_entropy': primary.get('S_2', 0.0),
                'accessible_entropy': primary.get('S_2_acc', 0.0),
                'use_symmetry': model.use_symmetry,
                'solver': solver_type,
                'sweep_data': sweep_data,
                'subregion_type': sub_type,
                'l_max': l_max,
                'time_total': t_end - t_start,
                'time_basis': t_basis - t_start,
                'time_solve': t_solve - t_basis,
                'time_density': t_density,
                'time_sweep': t_sweep,
            }

            if model_type in ('2D', '3D'):
                result['linear_size'] = p['linear_size']

            self.finished.emit(result)

        except Exception as e:
            import traceback
            self.error.emit(f"{e}\n\n{traceback.format_exc()}")


class SweepWorker(QThread):
    """Background thread for running parameter sweeps.

    Takes a list of parameter dicts (one per grid point), runs each
    diagonalization sequentially, and emits progress after each point.
    Grid points are pre-sorted so that smallest system sizes run first.
    """
    point_finished = Signal(int, int, dict)   # (index, total, result_dict)
    all_finished = Signal(list)               # list of all result dicts
    error = Signal(str)

    def __init__(self, param_grid: list, save_path: str):
        super().__init__()
        self.param_grid = param_grid
        self.save_path = save_path

    # -----------------------------------------------------------------
    # Basis-reuse helpers
    # -----------------------------------------------------------------

    @staticmethod
    def _basis_key(p: dict) -> tuple:
        """Return a hashable key for the basis-defining parameters.

        The basis (and symmetry orbits) depend only on L, N, n_max,
        boundary, model_type, and use_symmetry — NOT on t, U, or mu.
        When this key is constant across sweep points we can reuse
        the model and just rebuild the Hamiltonian.
        """
        model_type = p.get('model_type', '1D')
        if model_type in ('2D', '3D'):
            size_key = p.get('linear_size', 0)
        else:
            size_key = p.get('num_sites', 0)
        return (
            model_type,
            size_key,
            p.get('total_particles'),
            p.get('max_occupation'),
            p.get('boundary', 'pbc'),
            p.get('use_symmetry', False),
        )

    @staticmethod
    def _build_model(p: dict):
        """Build a fresh model from parameters."""
        model_type = p.get('model_type', '1D')
        if model_type == '3D':
            return BoseHubbard3D(
                linear_size=p['linear_size'],
                hopping=p['hopping'],
                interaction=p['interaction'],
                chemical_potential=p['chemical_potential'],
                max_occupation=p['max_occupation'],
                total_particles=p['total_particles'],
                boundary=p['boundary'],
                use_symmetry=False,
            )
        elif model_type == '2D':
            return BoseHubbard2D(
                linear_size=p['linear_size'],
                hopping=p['hopping'],
                interaction=p['interaction'],
                chemical_potential=p['chemical_potential'],
                max_occupation=p['max_occupation'],
                total_particles=p['total_particles'],
                boundary=p['boundary'],
                use_symmetry=p.get('use_symmetry', False),
            )
        else:
            return BoseHubbard1D(
                num_sites=p['num_sites'],
                hopping=p['hopping'],
                interaction=p['interaction'],
                chemical_potential=p['chemical_potential'],
                max_occupation=p['max_occupation'],
                total_particles=p['total_particles'],
                boundary=p['boundary'],
                use_symmetry=p.get('use_symmetry', False),
            )

    @staticmethod
    def _update_model_params(model, p: dict):
        """Update only the Hamiltonian parameters (t, U, mu) on an
        existing model, then clear the cached Hamiltonian so it gets
        rebuilt on the next hamiltonian() call.

        This avoids re-enumerating the basis and re-finding symmetry
        orbits — the expensive parts for large systems.
        """
        model.hopping = p['hopping']
        model.interaction = p['interaction']
        model.chemical_potential = p['chemical_potential']
        model._hamiltonian_matrix = None   # force rebuild

    # -----------------------------------------------------------------
    # Main sweep loop
    # -----------------------------------------------------------------

    def run(self):
        import gc

        try:
            all_results = []
            total = len(self.param_grid)

            # Check if we can reuse basis across all sweep points.
            # This saves enormous time for large systems (e.g., 4×4 2D)
            # where building the basis + orbits dominates the cost.
            first_key = self._basis_key(self.param_grid[0])
            can_reuse = all(
                self._basis_key(p) == first_key for p in self.param_grid
            )

            cached_model = None

            for idx, p in enumerate(self.param_grid):
                t_start = time.time()
                model_type = p.get('model_type', '1D')

                # Build or reuse model
                if can_reuse and cached_model is not None:
                    # Reuse existing basis + orbits, just update t/U/mu
                    model = cached_model
                    self._update_model_params(model, p)
                else:
                    model = self._build_model(p)
                    if can_reuse:
                        cached_model = model

                # Solve
                num_evals = p.get('num_eigenvalues', 1)
                solver_type = p.get('solver', 'standard')

                if solver_type == 'matrix_free':
                    from ..solvers.matrix_free import solve_matrix_free
                    if model_type in ('2D', '3D'):
                        model_for_mf = model
                    else:
                        model_for_mf = BoseHubbard1D(
                            num_sites=p['num_sites'],
                            hopping=p['hopping'],
                            interaction=p['interaction'],
                            chemical_potential=p['chemical_potential'],
                            max_occupation=p['max_occupation'],
                            total_particles=p['total_particles'],
                            boundary=p['boundary'],
                            use_symmetry=False,
                        )
                    eigenvalues, eigenvectors = solve_matrix_free(
                        model_for_mf, num_eigenvalues=num_evals,
                    )
                    ground_state_wfn = eigenvectors[:, 0]
                    basis_for_obs = model_for_mf.basis
                else:
                    hamiltonian = model.hamiltonian()
                    solver = LanczosSolver(
                        hamiltonian, num_eigenvalues=num_evals,
                    )
                    eigenvalues, eigenvectors = solver.solve()
                    if model.use_symmetry:
                        ground_state_wfn = model.reconstruct_wavefunction(
                            solver.ground_state
                        )
                    else:
                        ground_state_wfn = solver.ground_state
                    basis_for_obs = model.basis

                # Observables
                obs_density = density_profile(ground_state_wfn, basis_for_obs)

                if model_type in ('2D', '3D'):
                    L_lin = p['linear_size']
                    sub_type = p.get('subregion_type',
                                     'slab' if model_type == '3D' else 'strip')
                    l_max = p.get('subregion_l', L_lin // 2)

                    def make_subsystem(l, _m=model, _st=sub_type):
                        return _m.get_subregion(_st, l)
                else:
                    L_lin = p['num_sites']
                    l_max = L_lin // 2

                    def make_subsystem(l):
                        return list(range(l))

                sweep_data = sweep_observables(
                    ground_state_wfn, basis_for_obs,
                    make_subsystem, l_max,
                )

                t_total = time.time() - t_start

                # Collect result for this grid point
                point_result = {
                    'parameters': dict(p),  # copy
                    'hilbert_dim': model.dim,
                    'eigenvalues': eigenvalues,
                    'ground_state_energy': eigenvalues[0],
                    'density': obs_density,
                    'sweep_data': sweep_data,
                    'time_total': t_total,
                }
                all_results.append(point_result)
                self.point_finished.emit(idx + 1, total, point_result)

                # Free per-point temporaries to reduce memory pressure
                gc.collect()

            # Write results to .dat file
            self._save_results(all_results)
            self.all_finished.emit(all_results)

        except Exception as e:
            import traceback
            self.error.emit(f"{e}\n\n{traceback.format_exc()}")

    def _save_results(self, all_results):
        """Write sweep results to .dat files.

        Produces:
          1. One master file (user-chosen path) with all data.
          2. One file per system size L in the same directory,
             named  <basename>_L<value>.dat  for finite-size scaling.

        Columns are fixed-width and aligned for readability.
        Each row = one (grid_point, l) combination.
        """
        if not all_results:
            return

        # Identify which parameters were swept
        first_p = all_results[0]['parameters']
        swept_keys = []
        for key in ['interaction', 'chemical_potential', 'num_sites',
                     'linear_size', 'max_occupation', 'hopping']:
            values = set()
            for r in all_results:
                v = r['parameters'].get(key)
                if v is not None:
                    values.add(v)
            if len(values) > 1:
                swept_keys.append(key)

        def _build_file_lines(results, extra_header=""):
            """Build aligned .dat lines for a set of results."""
            lines = []
            lines.append("# Lanczos ED — Parameter Sweep Results")
            lines.append(f"# Date: {time.strftime('%Y-%m-%d %H:%M:%S')}")
            lines.append(f"# Grid points: {len(results)}")
            lines.append(f"# Swept parameters: {', '.join(swept_keys)}")
            lines.append(
                f"# Fixed parameters: "
                f"{_format_fixed_params(first_p, swept_keys)}"
            )
            if extra_header:
                lines.append(f"# {extra_header}")
            lines.append("#")

            # Fixed-width column header
            #   model  L   t       U/t         mu/t
            #   nmax   N   BC  solver  dim
            #   E0     l   |A|  F_A  S1  S2  S2acc  time
            hdr = (
                f"# {'model':>5s}  {'L':>4s}  {'t':>10s}  {'U/t':>12s}  "
                f"{'mu/t':>12s}  {'nmax':>5s}  {'N':>5s}  {'BC':>4s}  "
                f"{'solver':>10s}  {'dim':>10s}  {'E_0':>16s}  "
                f"{'l':>4s}  {'|A|':>5s}  {'F_A':>14s}  {'S_1':>14s}  "
                f"{'S_2':>14s}  {'S_2_acc':>14s}  {'time_s':>8s}"
            )
            lines.append(hdr)
            lines.append("#" + "-" * (len(hdr) - 1))

            for r in results:
                p = r['parameters']
                model_type = p.get('model_type', '1D')
                # L = linear size for display
                if model_type in ('2D', '3D'):
                    L_disp = p.get('linear_size', 0)
                else:
                    L_disp = p.get('num_sites', 0)
                t_hop = p.get('hopping', 1.0)
                U = p.get('interaction', 0.0)
                mu = p.get('chemical_potential', 0.0)
                nmax = p.get('max_occupation', -1)
                if nmax is None:
                    nmax = -1
                N = p.get('total_particles', -1)
                if N is None:
                    N = -1
                bc = p.get('boundary', 'pbc')
                slv = p.get('solver', 'standard')
                dim = r['hilbert_dim']
                E0 = r['ground_state_energy']
                t_s = r['time_total']

                sweep = r.get('sweep_data', [])
                if sweep:
                    for entry in sweep:
                        lines.append(
                            f"  {model_type:>5s}  {L_disp:4d}  "
                            f"{t_hop:10.6f}  {U:12.6f}  {mu:12.6f}  "
                            f"{nmax:5d}  {N:5d}  {bc:>4s}  {slv:>10s}  "
                            f"{dim:10d}  {E0:16.10f}  "
                            f"{entry['l']:4d}  {entry['num_sites_A']:5d}  "
                            f"{entry['F_A']:14.10f}  "
                            f"{entry['S_1']:14.10f}  "
                            f"{entry['S_2']:14.10f}  "
                            f"{entry['S_2_acc']:14.10f}  "
                            f"{t_s:8.3f}"
                        )
                else:
                    lines.append(
                        f"  {model_type:>5s}  {L_disp:4d}  "
                        f"{t_hop:10.6f}  {U:12.6f}  {mu:12.6f}  "
                        f"{nmax:5d}  {N:5d}  {bc:>4s}  {slv:>10s}  "
                        f"{dim:10d}  {E0:16.10f}  "
                        f"{'0':>4s}  {'0':>5s}  "
                        f"{'0.0':>14s}  {'0.0':>14s}  "
                        f"{'0.0':>14s}  {'0.0':>14s}  "
                        f"{t_s:8.3f}"
                    )
            return lines

        # --- Master file: all results ---
        master_lines = _build_file_lines(all_results)
        with open(self.save_path, 'w') as f:
            f.write("\n".join(master_lines) + "\n")

        # --- Sector-resolved data file ---
        # Write p(n_A) and S₂(n_A) for each (parameter point, l) pair
        sector_base = Path(self.save_path)
        sector_path = (sector_base.parent
                       / f"{sector_base.stem}_sectors{sector_base.suffix}")
        sector_lines = []
        sector_lines.append("# Lanczos ED — Sector-Resolved Data")
        sector_lines.append(
            f"# Date: {time.strftime('%Y-%m-%d %H:%M:%S')}")
        sector_lines.append(
            f"# Particle number distribution p(n_A) and "
            f"symmetry-resolved Rényi-2 entropy S_2(n_A)")
        sector_lines.append("#")
        sec_hdr = (
            f"# {'U/t':>12s}  {'l':>4s}  {'|A|':>5s}  {'n_A':>5s}  "
            f"{'p(n_A)':>14s}  {'S_2(n_A)':>14s}"
        )
        sector_lines.append(sec_hdr)
        sector_lines.append("#" + "-" * (len(sec_hdr) - 1))

        for r in all_results:
            p = r['parameters']
            U = p.get('interaction', 0.0)
            sweep = r.get('sweep_data', [])
            for entry in sweep:
                s_probs = entry.get('sector_probs', {})
                s_S2 = entry.get('sector_S_2', {})
                for n in sorted(s_probs.keys()):
                    sector_lines.append(
                        f"  {U:12.6f}  {entry['l']:4d}  "
                        f"{entry['num_sites_A']:5d}  {n:5d}  "
                        f"{s_probs[n]:14.10f}  "
                        f"{s_S2.get(n, 0.0):14.10f}"
                    )

        with open(sector_path, 'w') as f:
            f.write("\n".join(sector_lines) + "\n")

        # --- Per-L files for finite-size scaling ---
        from collections import defaultdict
        by_L = defaultdict(list)
        for r in all_results:
            p = r['parameters']
            model_type = p.get('model_type', '1D')
            if model_type in ('2D', '3D'):
                L_val = p.get('linear_size', 0)
            else:
                L_val = p.get('num_sites', 0)
            by_L[L_val].append(r)

        if len(by_L) > 1:
            # Only split if there are multiple system sizes
            base = Path(self.save_path)
            stem = base.stem       # e.g. "sweep_results"
            suffix = base.suffix   # e.g. ".dat"
            parent = base.parent

            for L_val in sorted(by_L.keys()):
                L_results = by_L[L_val]
                L_path = parent / f"{stem}_L{L_val}{suffix}"
                L_lines = _build_file_lines(
                    L_results, extra_header=f"System size L = {L_val}"
                )
                with open(L_path, 'w') as f:
                    f.write("\n".join(L_lines) + "\n")


def _format_fixed_params(params, swept_keys):
    """Format non-swept parameters for the .dat header."""
    parts = []
    for key, val in params.items():
        if key in swept_keys:
            continue
        if key in ('model_type', 'boundary', 'solver', 'use_symmetry',
                   'total_particles', 'subregion_type', 'subregion_l',
                   'num_eigenvalues'):
            parts.append(f"{key}={val}")
    return ', '.join(parts) if parts else 'none'


class MainWindow(QMainWindow):
    """Main application window for Lanczos ED."""

    def __init__(self):
        super().__init__()
        self.setWindowTitle("Lanczos ED — Bose-Hubbard")
        self.setMinimumSize(900, 700)
        self.worker = None
        self.sweep_worker = None

        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        main_layout = QHBoxLayout(central_widget)

        # Left panel: model parameters (scrollable)
        left_scroll = QScrollArea()
        left_scroll.setWidgetResizable(True)
        left_scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        left_container = QWidget()
        left_layout = QVBoxLayout(left_container)
        left_layout.setContentsMargins(0, 0, 0, 0)

        parameter_panel = self._build_parameter_panel()
        left_layout.addWidget(parameter_panel)

        sweep_panel = self._build_sweep_panel()
        left_layout.addWidget(sweep_panel)

        left_layout.addStretch()
        left_scroll.setWidget(left_container)
        left_scroll.setMaximumWidth(340)
        main_layout.addWidget(left_scroll)

        # Right panel: results display
        results_panel = self._build_results_panel()
        main_layout.addWidget(results_panel, stretch=1)

        self.statusBar().showMessage("Compiling Numba kernels…")
        self._start_warmup()

    # ------------------------------------------------------------------
    # Numba warmup
    # ------------------------------------------------------------------
    def _start_warmup(self):
        """Pre-compile Numba JIT kernels in a background thread."""
        from PySide6.QtCore import QTimer
        from ..warmup import warmup_async

        def _on_done(elapsed):
            # Marshal the status-bar update to the main (GUI) thread
            QTimer.singleShot(0, lambda: self.statusBar().showMessage(
                f"Ready  (Numba compiled in {elapsed:.1f}s)", 8000
            ))

        warmup_async(on_done=_on_done)

    def _build_parameter_panel(self) -> QGroupBox:
        """Create the parameter input panel."""
        group = QGroupBox("Model Parameters")
        grid = QGridLayout()
        row = 0

        # --- Run mode (Single / Sweep) ---
        grid.addWidget(QLabel("Mode:"), row, 0)
        self.input_mode = QComboBox()
        self.input_mode.addItems(["Single Run", "Sweep"])
        self.input_mode.currentIndexChanged.connect(self._on_mode_changed)
        grid.addWidget(self.input_mode, row, 1)
        row += 1

        # --- Model type (1D chain or 2D square) ---
        grid.addWidget(QLabel("Model:"), row, 0)
        self.input_model_type = QComboBox()
        self.input_model_type.addItems([
            "Bose-Hubbard 1D", "Bose-Hubbard 2D", "Bose-Hubbard 3D"
        ])
        self.input_model_type.currentIndexChanged.connect(
            self._on_model_type_changed
        )
        grid.addWidget(self.input_model_type, row, 1)
        row += 1

        # --- Number of lattice sites (1D) ---
        self.label_num_sites = QLabel("Sites (L):")
        grid.addWidget(self.label_num_sites, row, 0)
        self.input_num_sites = QSpinBox()
        self.input_num_sites.setRange(2, 30)
        self.input_num_sites.setValue(6)
        grid.addWidget(self.input_num_sites, row, 1)
        row += 1

        # --- Linear size (2D), initially hidden ---
        self.label_linear_size = QLabel("Linear size (L):")
        self.label_linear_size.setToolTip("L × L square lattice")
        grid.addWidget(self.label_linear_size, row, 0)
        self.input_linear_size = QSpinBox()
        self.input_linear_size.setRange(2, 10)
        self.input_linear_size.setValue(3)
        self.input_linear_size.setToolTip(
            "L × L square lattice.\n"
            "L=3: 9 sites, L=4: 16 sites."
        )
        grid.addWidget(self.input_linear_size, row, 1)
        self.label_linear_size.hide()
        self.input_linear_size.hide()
        row += 1

        # --- Hopping amplitude t ---
        grid.addWidget(QLabel("Hopping (t):"), row, 0)
        self.input_hopping = QDoubleSpinBox()
        self.input_hopping.setRange(0.0, 100.0)
        self.input_hopping.setValue(1.0)
        self.input_hopping.setDecimals(4)
        self.input_hopping.setSingleStep(0.1)
        grid.addWidget(self.input_hopping, row, 1)
        row += 1

        # --- On-site interaction U ---
        grid.addWidget(QLabel("Interaction (U):"), row, 0)
        self.input_interaction = QDoubleSpinBox()
        self.input_interaction.setRange(0.0, 100.0)
        self.input_interaction.setValue(1.0)
        self.input_interaction.setDecimals(4)
        self.input_interaction.setSingleStep(0.1)
        grid.addWidget(self.input_interaction, row, 1)
        row += 1

        # --- Chemical potential mu ---
        grid.addWidget(QLabel("Chem. potential (μ):"), row, 0)
        self.input_chemical_potential = QDoubleSpinBox()
        self.input_chemical_potential.setRange(-100.0, 100.0)
        self.input_chemical_potential.setValue(0.0)
        self.input_chemical_potential.setDecimals(4)
        self.input_chemical_potential.setSingleStep(0.1)
        grid.addWidget(self.input_chemical_potential, row, 1)
        row += 1

        # --- Maximum occupation per site (optional) ---
        self.checkbox_limit_nmax = QCheckBox("Limit n_max:")
        self.checkbox_limit_nmax.setChecked(False)
        self.checkbox_limit_nmax.toggled.connect(self._on_nmax_toggled)
        grid.addWidget(self.checkbox_limit_nmax, row, 0)
        self.input_max_occupation = QSpinBox()
        self.input_max_occupation.setRange(1, 100)
        self.input_max_occupation.setValue(3)
        self.input_max_occupation.setEnabled(False)
        grid.addWidget(self.input_max_occupation, row, 1)
        row += 1

        # --- Ensemble selection ---
        grid.addWidget(QLabel("Ensemble:"), row, 0)
        self.input_ensemble = QComboBox()
        self.input_ensemble.addItems(["Canonical", "Grand Canonical"])
        self.input_ensemble.currentIndexChanged.connect(
            self._on_ensemble_changed
        )
        grid.addWidget(self.input_ensemble, row, 1)
        row += 1

        # --- Total particle number (canonical only) ---
        self.label_particles = QLabel("Particles (N):")
        grid.addWidget(self.label_particles, row, 0)
        self.input_total_particles = QSpinBox()
        self.input_total_particles.setRange(1, 100)
        self.input_total_particles.setValue(3)
        grid.addWidget(self.input_total_particles, row, 1)
        row += 1

        # --- Boundary conditions ---
        grid.addWidget(QLabel("Boundary:"), row, 0)
        self.input_boundary = QComboBox()
        self.input_boundary.addItems(["PBC", "OBC"])
        self.input_boundary.currentIndexChanged.connect(
            self._on_boundary_changed
        )
        grid.addWidget(self.input_boundary, row, 1)
        row += 1

        # --- Subregion type (2D only), initially hidden ---
        self.label_subregion_type = QLabel("Subregion:")
        grid.addWidget(self.label_subregion_type, row, 0)
        self.input_subregion_type = QComboBox()
        self.input_subregion_type.addItems(["Strip", "Square"])
        self.input_subregion_type.setToolTip(
            "Strip: all sites with x < l (vertical slab, l×L sites).\n"
            "Square: all sites with x < l AND y < l (corner block, l×l sites)."
        )
        grid.addWidget(self.input_subregion_type, row, 1)
        self.label_subregion_type.hide()
        self.input_subregion_type.hide()
        row += 1

        # --- Subregion size l (2D only), initially hidden ---
        self.label_subregion_l = QLabel("Subregion size (l):")
        grid.addWidget(self.label_subregion_l, row, 0)
        self.input_subregion_l = QSpinBox()
        self.input_subregion_l.setRange(1, 10)
        self.input_subregion_l.setValue(1)
        self.input_subregion_l.setToolTip(
            "Width of strip (columns) or side of square block."
        )
        grid.addWidget(self.input_subregion_l, row, 1)
        self.label_subregion_l.hide()
        self.input_subregion_l.hide()
        row += 1

        # --- Solver selection ---
        grid.addWidget(QLabel("Solver:"), row, 0)
        self.input_solver = QComboBox()
        self.input_solver.addItems(["Standard", "Matrix-free"])
        self.input_solver.setToolTip(
            "Standard: builds sparse H, uses ARPACK eigsh.\n"
            "Matrix-free: computes H|ψ⟩ on the fly (less memory).\n"
            "Install numba for ~50-100× speedup on matrix-free."
        )
        grid.addWidget(self.input_solver, row, 1)
        row += 1

        # --- Symmetry reduction ---
        self.checkbox_symmetry = QCheckBox("Use symmetry (T+R)")
        self.checkbox_symmetry.setChecked(False)
        self.checkbox_symmetry.setToolTip(
            "Exploit translational symmetry to reduce Hilbert space.\n"
            "1D: T+R (translation + reflection), up to 2L× reduction.\n"
            "2D: Tx+Ty (2D translations), up to L²× reduction.\n"
            "Only available for PBC + canonical ensemble.\n"
            "Not valid for disordered models."
        )
        grid.addWidget(self.checkbox_symmetry, row, 0, 1, 2)
        row += 1

        # --- Number of eigenvalues to compute ---
        grid.addWidget(QLabel("Eigenvalues:"), row, 0)
        self.input_num_eigenvalues = QSpinBox()
        self.input_num_eigenvalues.setRange(1, 20)
        self.input_num_eigenvalues.setValue(1)
        grid.addWidget(self.input_num_eigenvalues, row, 1)
        row += 1

        # --- Run button ---
        self.button_run = QPushButton("Run Diagonalization")
        self.button_run.clicked.connect(self._run_diagonalization)
        grid.addWidget(self.button_run, row, 0, 1, 2)
        row += 1

        # --- Progress bar (hidden until sweep runs) ---
        self.progress_bar = QProgressBar()
        self.progress_bar.setVisible(False)
        grid.addWidget(self.progress_bar, row, 0, 1, 2)

        group.setLayout(grid)
        group.setMaximumWidth(320)
        return group

    def _build_sweep_panel(self) -> QGroupBox:
        """Create the sweep configuration panel.

        Each sweepable parameter gets:
          - A checkbox to include it in the sweep
          - Start, End, Num points spin boxes
          - A linear/log spacing selector
        Hidden until Sweep mode is selected.
        """
        self.sweep_group = QGroupBox("Sweep Parameters")
        layout = QVBoxLayout()

        # Define all sweepable parameters with (key, label, type, range)
        # type: 'double' or 'int'
        self._sweep_param_defs = [
            ('interaction', 'U (Interaction)', 'double',
             (0.0, 100.0, 4, 0.0, 10.0)),
            ('chemical_potential', 'μ (Chem. potential)', 'double',
             (-100.0, 100.0, 4, 0.0, 0.0)),
            ('hopping', 't (Hopping)', 'double',
             (0.0, 100.0, 4, 1.0, 1.0)),
            ('num_sites', 'L (System size)', 'int',
             (2, 30, None, 4, 12)),
            ('max_occupation', 'n_max', 'int',
             (1, 20, None, 1, 5)),
        ]

        self._sweep_widgets = {}  # key -> {checkbox, start, end, npts, spacing}

        for key, label, ptype, (lo, hi, decimals, default_start, default_end) in self._sweep_param_defs:
            # Container frame for this parameter
            frame = QFrame()
            frame.setFrameShape(QFrame.StyledPanel)
            fgrid = QGridLayout(frame)
            fgrid.setContentsMargins(4, 2, 4, 2)

            # Checkbox
            cb = QCheckBox(label)
            cb.setChecked(False)
            fgrid.addWidget(cb, 0, 0, 1, 4)

            # Start
            fgrid.addWidget(QLabel("Start:"), 1, 0)
            if ptype == 'double':
                start_sb = QDoubleSpinBox()
                start_sb.setRange(lo, hi)
                start_sb.setDecimals(decimals)
                start_sb.setValue(default_start)
                start_sb.setSingleStep(0.1)
            else:
                start_sb = QSpinBox()
                start_sb.setRange(lo, hi)
                start_sb.setValue(int(default_start))
            fgrid.addWidget(start_sb, 1, 1)

            # End
            fgrid.addWidget(QLabel("End:"), 1, 2)
            if ptype == 'double':
                end_sb = QDoubleSpinBox()
                end_sb.setRange(lo, hi)
                end_sb.setDecimals(decimals)
                end_sb.setValue(default_end)
                end_sb.setSingleStep(0.1)
            else:
                end_sb = QSpinBox()
                end_sb.setRange(lo, hi)
                end_sb.setValue(int(default_end))
            fgrid.addWidget(end_sb, 1, 3)

            # Num points
            fgrid.addWidget(QLabel("Points:"), 2, 0)
            npts_sb = QSpinBox()
            npts_sb.setRange(2, 500)
            npts_sb.setValue(10)
            fgrid.addWidget(npts_sb, 2, 1)

            # Spacing
            fgrid.addWidget(QLabel("Spacing:"), 2, 2)
            spacing_cb = QComboBox()
            spacing_cb.addItems(["Linear", "Log"])
            fgrid.addWidget(spacing_cb, 2, 3)

            # Initially disable inputs until checkbox is checked
            for w in (start_sb, end_sb, npts_sb, spacing_cb):
                w.setEnabled(False)
            cb.toggled.connect(
                lambda checked, widgets=(start_sb, end_sb, npts_sb, spacing_cb):
                    [w.setEnabled(checked) for w in widgets]
            )

            layout.addWidget(frame)
            self._sweep_widgets[key] = {
                'checkbox': cb,
                'start': start_sb,
                'end': end_sb,
                'npts': npts_sb,
                'spacing': spacing_cb,
                'type': ptype,
            }

        self.sweep_group.setLayout(layout)
        self.sweep_group.setVisible(False)
        return self.sweep_group

    def _build_results_panel(self) -> QTabWidget:
        """Create the results display panel with tabs."""
        self.tabs = QTabWidget()

        # Tab 1: Text results
        self.text_results = QTextEdit()
        self.text_results.setReadOnly(True)
        self.text_results.setFontFamily("Courier")
        self.tabs.addTab(self.text_results, "Results")

        # Tab 2: Density profile plot
        if HAS_MATPLOTLIB:
            self.figure = Figure(figsize=(6, 4))
            self.canvas = FigureCanvasQTAgg(self.figure)
            self.tabs.addTab(self.canvas, "Density Plot")
        else:
            placeholder = QLabel(
                "Install matplotlib for plotting:\n"
                "  pip install matplotlib"
            )
            placeholder.setAlignment(Qt.AlignCenter)
            self.tabs.addTab(placeholder, "Density Plot")

        return self.tabs

    def _on_model_type_changed(self, index):
        """Show/hide 1D vs 2D/3D controls based on model selection."""
        is_1d = (index == 0)
        is_2d = (index == 1)
        is_3d = (index == 2)
        is_higher_d = is_2d or is_3d

        # 1D controls
        self.label_num_sites.setVisible(is_1d)
        self.input_num_sites.setVisible(is_1d)

        # 2D/3D controls (shared: linear size and subregion)
        self.label_linear_size.setVisible(is_higher_d)
        self.input_linear_size.setVisible(is_higher_d)
        self.label_subregion_type.setVisible(is_higher_d)
        self.input_subregion_type.setVisible(is_higher_d)
        self.label_subregion_l.setVisible(is_higher_d)
        self.input_subregion_l.setVisible(is_higher_d)

        # Update linear size range and tooltip
        if is_3d:
            self.input_linear_size.setRange(2, 6)
            self.input_linear_size.setValue(2)
            self.label_linear_size.setToolTip("L × L × L cubic lattice")
            self.input_linear_size.setToolTip(
                "L × L × L cubic lattice.\n"
                "L=2: 8 sites, L=3: 27 sites."
            )
        elif is_2d:
            self.input_linear_size.setRange(2, 10)
            self.input_linear_size.setValue(3)
            self.label_linear_size.setToolTip("L × L square lattice")
            self.input_linear_size.setToolTip(
                "L × L square lattice.\n"
                "L=3: 9 sites, L=4: 16 sites."
            )

        # Update subregion type options
        self.input_subregion_type.clear()
        if is_3d:
            self.input_subregion_type.addItems(["Slab", "Column", "Cube"])
            self.input_subregion_type.setToolTip(
                "Slab:   all sites with x < l (l×L×L sites).\n"
                "Column: all sites with x < l AND y < l (l×l×L sites).\n"
                "Cube:   all sites with x < l AND y < l AND z < l (l³ sites)."
            )
        elif is_2d:
            self.input_subregion_type.addItems(["Strip", "Square"])
            self.input_subregion_type.setToolTip(
                "Strip: all sites with x < l (vertical slab, l×L sites).\n"
                "Square: all sites with x < l AND y < l (corner block, l×l sites)."
            )

        self._update_symmetry_availability()

    def _on_ensemble_changed(self, index):
        """Enable/disable particle number input based on ensemble choice."""
        is_canonical = (index == 0)
        self.input_total_particles.setEnabled(is_canonical)
        self.label_particles.setEnabled(is_canonical)
        self._update_symmetry_availability()

    def _on_boundary_changed(self, index):
        """Update symmetry availability when boundary conditions change."""
        self._update_symmetry_availability()

    def _on_nmax_toggled(self, checked):
        """Enable/disable the n_max spin box."""
        self.input_max_occupation.setEnabled(checked)

    def _update_symmetry_availability(self):
        """Enable symmetry checkbox for (1D or 2D) + PBC + canonical."""
        model_idx = self.input_model_type.currentIndex()
        is_1d_or_2d = (model_idx == 0 or model_idx == 1)
        is_pbc = (self.input_boundary.currentText() == "PBC")
        is_canonical = (self.input_ensemble.currentIndex() == 0)
        available = is_1d_or_2d and is_pbc and is_canonical
        self.checkbox_symmetry.setEnabled(available)
        if not available:
            self.checkbox_symmetry.setChecked(False)

    def _on_mode_changed(self, index):
        """Show/hide sweep panel based on mode selection."""
        is_sweep = (index == 1)
        self.sweep_group.setVisible(is_sweep)
        if is_sweep:
            self.button_run.setText("Run Sweep")
        else:
            self.button_run.setText("Run Diagonalization")

    def _collect_parameters(self) -> dict:
        """Read all parameter values from the GUI inputs."""
        is_canonical = (self.input_ensemble.currentIndex() == 0)
        limit_nmax = self.checkbox_limit_nmax.isChecked()
        model_index = self.input_model_type.currentIndex()
        is_2d = (model_index == 1)
        is_3d = (model_index == 2)

        # Map model index to type string
        if is_3d:
            model_type = '3D'
        elif is_2d:
            model_type = '2D'
        else:
            model_type = '1D'

        params = {
            'model_type': model_type,
            'hopping': self.input_hopping.value(),
            'interaction': self.input_interaction.value(),
            'chemical_potential': self.input_chemical_potential.value(),
            'max_occupation': (
                self.input_max_occupation.value() if limit_nmax else None
            ),
            'total_particles': (
                self.input_total_particles.value() if is_canonical else None
            ),
            'boundary': self.input_boundary.currentText().lower(),
            'num_eigenvalues': self.input_num_eigenvalues.value(),
            'use_symmetry': self.checkbox_symmetry.isChecked(),
            'solver': (
                'matrix_free'
                if self.input_solver.currentIndex() == 1
                else 'standard'
            ),
        }

        if is_3d:
            L = self.input_linear_size.value()
            params['linear_size'] = L
            params['num_sites'] = L ** 3
            params['subregion_type'] = (
                self.input_subregion_type.currentText().lower()
            )
            params['subregion_l'] = self.input_subregion_l.value()
        elif is_2d:
            L = self.input_linear_size.value()
            params['linear_size'] = L
            params['num_sites'] = L * L
            params['subregion_type'] = (
                self.input_subregion_type.currentText().lower()
            )
            params['subregion_l'] = self.input_subregion_l.value()
        else:
            params['num_sites'] = self.input_num_sites.value()

        return params

    def _run_diagonalization(self):
        """Start the diagonalization in a background thread."""
        if self.worker is not None and self.worker.isRunning():
            return
        if self.sweep_worker is not None and self.sweep_worker.isRunning():
            return

        # Dispatch to sweep mode if selected
        if self.input_mode.currentIndex() == 1:
            self._run_sweep()
            return

        parameters = self._collect_parameters()

        # Estimate Hilbert space size and warn if large
        if parameters['total_particles'] is None:
            n_max = parameters['max_occupation'] or 10
            estimated_dim = (n_max + 1) ** parameters['num_sites']
        else:
            from math import comb
            n = parameters['total_particles']
            L = parameters['num_sites']
            if parameters['max_occupation'] is not None:
                # Rough estimate with n_max restriction
                estimated_dim = comb(n + L - 1, L - 1)
            else:
                estimated_dim = comb(n + L - 1, L - 1)

        # Adjust estimate for symmetry
        effective_dim = estimated_dim
        if parameters['use_symmetry']:
            model_type = parameters.get('model_type', '1D')
            if model_type == '2D':
                # 2D translations: up to L² reduction
                L = parameters.get('linear_size', 1)
                effective_dim = estimated_dim // (L * L)
            else:
                # 1D T+R: up to 2L reduction
                effective_dim = estimated_dim // (2 * parameters['num_sites'])

        if effective_dim > 500_000:
            reply = QMessageBox.question(
                self, "Large Hilbert Space",
                f"Estimated dimension ~{estimated_dim:,}"
                + (f" (reduced ~{effective_dim:,} with symmetry)"
                   if parameters['use_symmetry'] else "")
                + ". This may be slow. Continue?",
                QMessageBox.Yes | QMessageBox.No,
            )
            if reply == QMessageBox.No:
                return

        self.button_run.setEnabled(False)
        self.statusBar().showMessage("Running diagonalization...")
        self.text_results.clear()

        self.worker = SolverWorker(parameters)
        self.worker.finished.connect(self._display_results)
        self.worker.error.connect(self._display_error)
        self.worker.start()

    # -----------------------------------------------------------------
    #  Sweep mode
    # -----------------------------------------------------------------

    def _build_sweep_grid(self) -> list:
        """Generate the full parameter grid for a sweep.

        Returns a list of parameter dicts, one per grid point.
        Grid is sorted so the smallest system sizes run first.
        """
        base_params = self._collect_parameters()

        # Collect sweep axes: list of (key, values_array)
        sweep_axes = []
        for key, widgets in self._sweep_widgets.items():
            if not widgets['checkbox'].isChecked():
                continue

            start = widgets['start'].value()
            end = widgets['end'].value()
            npts = widgets['npts'].value()
            spacing = widgets['spacing'].currentText()

            if widgets['type'] == 'int':
                if spacing == 'Log':
                    vals = np.logspace(
                        np.log10(max(start, 1)),
                        np.log10(max(end, 1)),
                        npts,
                    )
                    vals = sorted(set(int(round(v)) for v in vals))
                else:
                    vals = np.linspace(start, end, npts)
                    vals = sorted(set(int(round(v)) for v in vals))
                vals = list(vals)
            else:
                if spacing == 'Log':
                    if start <= 0:
                        start = 1e-6
                    vals = list(np.logspace(
                        np.log10(start), np.log10(end), npts,
                    ))
                else:
                    vals = list(np.linspace(start, end, npts))

            sweep_axes.append((key, vals))

        if not sweep_axes:
            QMessageBox.warning(
                self, "No Sweep Parameters",
                "Check at least one parameter to sweep over.",
            )
            return []

        # Build Cartesian product of all sweep axes
        axis_keys = [k for k, _ in sweep_axes]
        axis_vals = [v for _, v in sweep_axes]

        grid = []
        for combo in iterproduct(*axis_vals):
            p = dict(base_params)
            for k, v in zip(axis_keys, combo):
                p[k] = v
                # If sweeping system size, update num_sites accordingly
                if k == 'num_sites':
                    model_type = p.get('model_type', '1D')
                    L_val = int(v)
                    if model_type == '1D':
                        p['num_sites'] = L_val
                    elif model_type == '2D':
                        p['linear_size'] = L_val
                        p['num_sites'] = L_val ** 2
                        p['subregion_l'] = L_val // 2
                    elif model_type == '3D':
                        p['linear_size'] = L_val
                        p['num_sites'] = L_val ** 3
                        p['subregion_l'] = L_val // 2
            grid.append(p)

        # Sort by system size (smallest first) for efficiency
        def _sort_key(p):
            return p.get('num_sites', 0)
        grid.sort(key=_sort_key)

        return grid

    def _run_sweep(self):
        """Launch a parameter sweep after asking for save location."""
        # Build the grid first to validate
        grid = self._build_sweep_grid()
        if not grid:
            return

        # Ask where to save before starting
        save_path, _ = QFileDialog.getSaveFileName(
            self, "Save Sweep Results",
            str(Path.home() / "sweep_results.dat"),
            "Data files (*.dat);;All files (*)",
        )
        if not save_path:
            return  # user cancelled

        total = len(grid)

        # Confirm
        reply = QMessageBox.question(
            self, "Start Sweep",
            f"Sweep will run {total} grid points.\n"
            f"Results will be saved to:\n{save_path}\n\n"
            f"Continue?",
            QMessageBox.Yes | QMessageBox.No,
        )
        if reply == QMessageBox.No:
            return

        self.button_run.setEnabled(False)
        self.progress_bar.setVisible(True)
        self.progress_bar.setRange(0, total)
        self.progress_bar.setValue(0)
        self.statusBar().showMessage(f"Sweep: 0/{total}...")
        self.text_results.clear()
        self.text_results.append(
            f"Starting sweep: {total} grid points\n"
            f"Saving to: {save_path}\n"
        )

        self.sweep_worker = SweepWorker(grid, save_path)
        self.sweep_worker.point_finished.connect(self._on_sweep_point)
        self.sweep_worker.all_finished.connect(self._on_sweep_done)
        self.sweep_worker.error.connect(self._display_error)
        self.sweep_worker.start()

    def _on_sweep_point(self, idx, total, point_result):
        """Called after each sweep grid point completes."""
        self.progress_bar.setValue(idx)
        self.statusBar().showMessage(f"Sweep: {idx}/{total}...")

        p = point_result['parameters']
        E0 = point_result['ground_state_energy']
        t = point_result['time_total']
        model_type = p.get('model_type', '1D')

        if model_type == '1D':
            size_str = f"L={p['num_sites']}"
        else:
            size_str = f"L={p.get('linear_size', '?')}"

        self.text_results.append(
            f"  [{idx}/{total}] {size_str}, "
            f"U={p.get('interaction', 0):.4f}, "
            f"μ={p.get('chemical_potential', 0):.4f} "
            f"→ E₀={E0:.8f}  ({t:.2f}s)"
        )

    def _on_sweep_done(self, all_results):
        """Called when the full sweep finishes."""
        self.button_run.setEnabled(True)
        self.progress_bar.setVisible(False)
        total = len(all_results)
        total_time = sum(r['time_total'] for r in all_results)
        self.statusBar().showMessage(f"Sweep complete: {total} points", 10000)

        # Check if per-L files were created
        from collections import defaultdict
        by_L = defaultdict(list)
        for r in all_results:
            p = r['parameters']
            mt = p.get('model_type', '1D')
            L_val = p.get('linear_size', p.get('num_sites', 0)) \
                if mt in ('2D', '3D') else p.get('num_sites', 0)
            by_L[L_val].append(r)

        save_msg = f"Results saved to: {self.sweep_worker.save_path}"
        if len(by_L) > 1:
            base = Path(self.sweep_worker.save_path)
            L_files = [
                f"  {base.stem}_L{L}{base.suffix}"
                for L in sorted(by_L.keys())
            ]
            save_msg += "\nPer-L files:\n" + "\n".join(L_files)

        self.text_results.append(
            f"\nSweep complete: {total} points in {total_time:.1f}s\n"
            f"{save_msg}"
        )

    def _display_results(self, result: dict):
        """Format and show results in the text tab and plot tab."""
        self.button_run.setEnabled(True)
        self.statusBar().showMessage("Done", 5000)

        params = self._collect_parameters()
        num_sites = params['num_sites']
        model_type = result.get('model_type', '1D')
        is_2d = (model_type == '2D')
        is_3d = (model_type == '3D')

        lines = []
        lines.append(f"Model: {result['model_description']}")

        if result.get('use_symmetry'):
            lines.append(
                f"Hilbert space dimension: {result['full_dim']} "
                f"(reduced to {result['hilbert_dim']} with symmetry)"
            )
        else:
            lines.append(
                f"Hilbert space dimension: {result['hilbert_dim']}"
            )

        solver_label = result.get('solver', 'standard')
        lines.append(f"Solver: {solver_label}")
        lines.append(f"Total time: {result['time_total']:.3f}s")

        # Per-step timing breakdown
        timing_parts = []
        if 'time_basis' in result:
            timing_parts.append(
                f"basis {result['time_basis']:.3f}s"
            )
        if 'time_solve' in result:
            timing_parts.append(
                f"H+solve {result['time_solve']:.3f}s"
            )
        if 'time_density' in result:
            timing_parts.append(
                f"density {result['time_density']:.3f}s"
            )
        if 'time_sweep' in result:
            timing_parts.append(
                f"observables {result['time_sweep']:.3f}s"
            )
        if timing_parts:
            lines.append(f"  Breakdown: {' | '.join(timing_parts)}")

        lines.append("")

        lines.append(
            f"Ground state energy:  {result['ground_state_energy']:.10f}"
        )
        lines.append(
            f"Energy per site:      "
            f"{result['ground_state_energy'] / num_sites:.10f}"
        )
        lines.append("")

        lines.append("Eigenvalues:")
        for i, energy in enumerate(result['eigenvalues']):
            lines.append(f"  E_{i} = {energy:.10f}")
        lines.append("")

        # Density profile
        density = result['density']
        if is_3d:
            L = result.get('linear_size', int(round(len(density) ** (1/3))))
            lines.append(
                f"Density profile <n_i> ({L}×{L}×{L} lattice):"
            )
            lines.append(
                f"  site = x + y*L + z*L²,  "
                f"x = site % L,  y = (site//L) % L,  z = site // L²"
            )
            lines.append("")
            # Print as z-layer slices
            for z in range(L):
                lines.append(f"  --- Layer z={z} ---")
                header = "  y\\x " + "".join(
                    f"{x:>10d}" for x in range(L)
                )
                lines.append(header)
                lines.append("  " + "-" * (5 + 10 * L))
                for y in range(L):
                    row_vals = [
                        density[x + y * L + z * L * L] for x in range(L)
                    ]
                    row_str = f"  {y:>3d} |" + "".join(
                        f"{v:>10.6f}" for v in row_vals
                    )
                    lines.append(row_str)
                lines.append("")
            lines.append(f"  Total: {density.sum():.6f}")
        elif is_2d:
            L = result.get('linear_size', int(np.sqrt(len(density))))
            lines.append(f"Density profile <n_i> ({L}×{L} lattice):")
            lines.append(f"  site = x + y*L,  x = site % L,  y = site // L")
            lines.append("")
            # Print as a 2D grid for readability
            header = "  y\\x " + "".join(f"{x:>10d}" for x in range(L))
            lines.append(header)
            lines.append("  " + "-" * (5 + 10 * L))
            for y in range(L):
                row_vals = [density[x + y * L] for x in range(L)]
                row_str = f"  {y:>3d} |" + "".join(
                    f"{v:>10.6f}" for v in row_vals
                )
                lines.append(row_str)
            lines.append(f"  Total: {density.sum():.6f}")
        else:
            lines.append("Density profile <n_i>:")
            for site, n_i in enumerate(density):
                lines.append(f"  site {site}: {n_i:.6f}")
            lines.append(f"  Total:  {density.sum():.6f}")
        lines.append("")

        # Subregion sweep table
        sweep_data = result.get('sweep_data', [])
        sub_type = result.get('subregion_type', 'strip')

        if is_3d:
            L_lin = result.get('linear_size', 2)
            sub_desc = {
                'slab': 'x < l  (l×L×L sites)',
                'column': 'x < l, y < l  (l×l×L sites)',
                'cube': 'x < l, y < l, z < l  (l³ sites)',
            }
            lines.append(
                f"Subregion type: {sub_type} "
                f"({sub_desc.get(sub_type, sub_type)})"
            )
        elif is_2d:
            L_lin = result.get('linear_size', 3)
            lines.append(
                f"Subregion type: {sub_type} "
                f"({'x < l' if sub_type == 'strip' else 'x < l, y < l'})"
            )
        else:
            L_lin = num_sites
            lines.append("Subregion: sites 0..l-1 (1D bipartition)")

        if sweep_data:
            lines.append("")
            lines.append(
                f"{'l':>4s}  {'|A|':>5s}  {'F_A':>12s}  "
                f"{'S_1':>12s}  {'S_2':>12s}  {'S_2_acc':>12s}"
            )
            lines.append("  " + "-" * 65)
            for entry in sweep_data:
                lines.append(
                    f"{entry['l']:4d}  {entry['num_sites_A']:5d}  "
                    f"{entry['F_A']:12.8f}  {entry['S_1']:12.8f}  "
                    f"{entry['S_2']:12.8f}  {entry['S_2_acc']:12.8f}"
                )

            # Particle number distribution & symmetry-resolved S₂(n)
            lines.append("")
            lines.append("Particle number distribution p(n_A) "
                         "and symmetry-resolved S₂(n_A):")
            for entry in sweep_data:
                s_probs = entry.get('sector_probs', {})
                s_S2 = entry.get('sector_S_2', {})
                if not s_probs:
                    continue
                lines.append("")
                lines.append(f"  l = {entry['l']}  "
                             f"(|A| = {entry['num_sites_A']})")
                lines.append(f"    {'n_A':>5s}  {'p(n_A)':>14s}  "
                             f"{'S_2(n_A)':>14s}")
                lines.append("    " + "-" * 37)
                for n in sorted(s_probs.keys()):
                    p_n = s_probs[n]
                    s2_n = s_S2.get(n, 0.0)
                    lines.append(
                        f"    {n:5d}  {p_n:14.10f}  {s2_n:14.10f}"
                    )

        self.text_results.setText("\n".join(lines))

        # Update the density plot
        if HAS_MATPLOTLIB:
            self.figure.clear()
            if is_3d:
                # 3D: show each z-layer as a subplot heatmap
                L = result.get(
                    'linear_size',
                    int(round(len(density) ** (1/3))),
                )
                # Determine subplot grid: up to 3 columns
                ncols = min(L, 3)
                nrows = (L + ncols - 1) // ncols
                vmin = density.min()
                vmax = density.max()
                axes = []
                for z in range(L):
                    ax = self.figure.add_subplot(nrows, ncols, z + 1)
                    axes.append(ax)
                    grid_2d = np.zeros((L, L))
                    for y in range(L):
                        for x in range(L):
                            grid_2d[y, x] = density[
                                x + y * L + z * L * L
                            ]
                    im = ax.imshow(
                        grid_2d, origin='lower', cmap='viridis',
                        aspect='equal', vmin=vmin, vmax=vmax,
                    )
                    ax.set_title(f"$z={z}$", fontsize=10)
                    ax.set_xticks(range(L))
                    ax.set_yticks(range(L))
                    if z % ncols == 0:
                        ax.set_ylabel("$y$")
                    if z >= L - ncols:
                        ax.set_xlabel("$x$")
                self.figure.suptitle(
                    r"Ground State Density $\langle n_{x,y,z} \rangle$"
                )
                self.figure.colorbar(im, ax=axes, shrink=0.6)
            elif is_2d:
                # 2D heatmap of density
                L = result.get('linear_size', int(np.sqrt(len(density))))
                density_grid = density.reshape(L, L, order='C')
                # Reshape with site = x + y*L means column x, row y
                # density_grid[flat_idx] where flat_idx = x + y*L
                # To display as a proper 2D grid with y on vertical axis:
                grid_2d = np.zeros((L, L))
                for y in range(L):
                    for x in range(L):
                        grid_2d[y, x] = density[x + y * L]
                ax = self.figure.add_subplot(111)
                im = ax.imshow(grid_2d, origin='lower', cmap='viridis',
                               aspect='equal')
                ax.set_xlabel("$x$")
                ax.set_ylabel("$y$")
                ax.set_title("Ground State Density $\\langle n_{x,y} \\rangle$")
                ax.set_xticks(range(L))
                ax.set_yticks(range(L))
                self.figure.colorbar(im, ax=ax, shrink=0.8)
            else:
                ax = self.figure.add_subplot(111)
                sites = np.arange(len(density))
                ax.bar(sites, density, color='steelblue', alpha=0.8)
                ax.set_xlabel("Site $i$")
                ax.set_ylabel(r"$\langle n_i \rangle$")
                ax.set_title("Ground State Density Profile")
                ax.set_xticks(sites)
            self.figure.tight_layout()
            self.canvas.draw()

    def _display_error(self, message: str):
        """Show an error dialog if the solver fails."""
        self.button_run.setEnabled(True)
        self.statusBar().showMessage("Error", 5000)
        QMessageBox.critical(self, "Error", f"Solver error:\n{message}")


def run_gui():
    """Entry point for launching the GUI application."""
    app = QApplication(sys.argv)
    window = MainWindow()
    window.show()
    sys.exit(app.exec())
