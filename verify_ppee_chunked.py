"""
Verify the chunked BLAS PPEE implementation against the original.

Runs at L=8, N=8, U/t=3.275 (the Herdman benchmark point):
  1. Computes ground state via Lanczos
  2. Computes S_2(n) for n=1..4 using BOTH implementations
  3. Checks Tr(rho) = 1 for the chunked version
  4. Checks the n=1 diagonal against density_profile/N
  5. Checks particle-hole symmetry: S_2(n) = S_2(N-n)

Run:
    cd ~/Desktop/WorkingCopies/Lanczos
    python verify_ppee_chunked.py
"""
import numpy as np
import time

from lanczos_ed.models.bose_hubbard import BoseHubbard1D
from lanczos_ed.solvers.lanczos import LanczosSolver
from lanczos_ed.observables.basic import particle_partition_entropy, density_profile
from lanczos_ed.observables.ppee import (
    compute_ppee, compute_ppee_with_rho, validate_n1_diagonal,
)

# ---- build ground state ----
L, N, Ut = 8, 8, 3.275
print(f"=== PPEE chunked BLAS validation: L={L}, N={N}, U/t={Ut} ===\n")

model = BoseHubbard1D(
    num_sites=L, hopping=1.0, interaction=Ut,
    total_particles=N, boundary='pbc', use_symmetry=False,
)
H = model.hamiltonian()
solver = LanczosSolver(H, num_eigenvalues=1)
solver.solve()
psi = solver.ground_state
basis = model.basis
print(f"Ground state: E0 = {solver.eigenvalues[0]:.8f}")
print(f"Hilbert space dim = {basis.dim}\n")

# ---- compute S_2(n) both ways ----
print(f"{'n':>3}  {'S2_original':>14}  {'S2_chunked':>14}  {'diff':>12}  "
      f"{'Tr(rho)':>10}  {'status':>8}")
print("-" * 75)

all_pass = True
s2_values = {}

for n_A in range(1, N // 2 + 1):
    # Original implementation (Python loops + get_index)
    t0 = time.time()
    s2_orig = particle_partition_entropy(psi, basis, n_A, renyi_index=2.0)
    t_orig = time.time() - t0

    # Chunked BLAS implementation
    t0 = time.time()
    result = compute_ppee(psi, basis, n_A, renyi_index=2.0, chunk_size=5000)
    t_chunk = time.time() - t0

    s2_chunk = result['S_alpha']
    trace = result['trace']
    diff = abs(s2_orig - s2_chunk)
    s2_values[n_A] = s2_chunk

    ok = diff < 1e-8 and abs(trace - 1.0) < 1e-10
    status = "PASS" if ok else "FAIL"
    if not ok:
        all_pass = False

    print(f"{n_A:3d}  {s2_orig:14.10f}  {s2_chunk:14.10f}  {diff:12.2e}  "
          f"{trace:10.8f}  {status:>8}")

# ---- n=1 cross-check: diagonal vs density_profile/N ----
print("\n--- n=1 diagonal cross-check ---")
result_n1 = compute_ppee_with_rho(psi, basis, n_A=1, renyi_index=2.0)
rho_1 = result_n1['rho']
max_err = validate_n1_diagonal(rho_1, psi, basis)
status = "PASS" if max_err < 1e-10 else "FAIL"
if max_err >= 1e-10:
    all_pass = False
print(f"max |rho_1[i,i] - <n_i>/N| = {max_err:.2e}   {status}")

# ---- particle-hole symmetry: S_2(n) = S_2(N-n) ----
print("\n--- particle-hole symmetry S_2(n) = S_2(N-n) ---")
for n_A in range(1, N // 2 + 1):
    n_comp = N - n_A
    if n_comp <= N // 2:
        continue  # already computed
    result_comp = compute_ppee(psi, basis, n_comp, renyi_index=2.0)
    diff = abs(s2_values[n_A] - result_comp['S_alpha'])
    status = "PASS" if diff < 1e-8 else "FAIL"
    if diff >= 1e-8:
        all_pass = False
    print(f"S_2({n_A}) = {s2_values[n_A]:.10f}   "
          f"S_2({n_comp}) = {result_comp['S_alpha']:.10f}   "
          f"diff = {diff:.2e}   {status}")

print(f"\n{'=' * 75}")
print(f"Overall: {'ALL CHECKS PASSED' if all_pass else 'SOME CHECKS FAILED'}")
