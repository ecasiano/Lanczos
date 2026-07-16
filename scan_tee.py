#!/usr/bin/env python3
"""
Scan the extended Bose-Hubbard model on the kagome lattice to pinpoint a
parameter region suitable for the TEE-decomposition test: where the topological
entanglement entropy is nonzero (gamma > 0) AND particle-number fluctuations
across the cut are alive (number entropy H > 0).

For each coupling value it reports the ground-state:
    S      : full (von Neumann) bipartite entanglement entropy (half cut)
    S_acc  : operationally accessible entanglement entropy
    H      : number/fluctuation entropy = S - S_acc   <-- criterion (b): H > 0
    varNA  : bipartite particle-number variance (independent fluctuation probe)
and, with --tee, the Kitaev-Preskill decomposition
    gamma, gamma_acc, gamma_H = gamma - gamma_acc     <-- criterion (a): gamma>0

Two routes (both hardcore bosons on kagome):
    A : hexagon cluster-charging  W * sum_hex (n_hex)^2   (half filling; Isakov-
        Hastings-Melko). Half filling is integer only for even L.
    B : pairwise V1(NN)+V2(2NN)+V3(3NN)  (1/3 filling; Roychowdhury et al).
        --v2-ratio, --v3-ratio tie V2,V3 to the scanned V1 (default 1,1).

Performance: the hopping matrix and per-site occupations are built ONCE; each
scan point only rebuilds the (cheap, vectorized) diagonal. gamma (--tee) is
only physically meaningful for L >= 3 (a proper disk needs an environment); at
L = 2 use the scan for the H(coupling) map and to check the pipeline.

Examples
--------
# Fast scouting (L=2, instant): number-entropy map vs NN repulsion, Route B
python scan_tee.py --route B --L 2 --coupling-min 0 --coupling-max 12 --steps 25

# Route A cluster-charging scouting at L=2 (half filling = 6 particles)
python scan_tee.py --route A --L 2 --coupling-min 0 --coupling-max 12 --steps 25

# Full TEE decomposition (heavier; L=3 Route B = 1/3 filling, 4.7M states)
python scan_tee.py --route B --L 3 --coupling-min 4 --coupling-max 10 --steps 7 --tee
"""

import argparse
import time
import numpy as np
from scipy import sparse
from scipy.sparse.linalg import eigsh

from lanczos_ed.models.bose_hubbard_kagome import (
    BoseHubbardKagome, kagome_neighbor_shells, kagome_hexagons)
from lanczos_ed.observables.basic import bipartite_fluctuations
from lanczos_ed.observables.tee import (
    bipartite_number_entropy, region_entropies,
    topological_entanglement_entropy, kitaev_preskill_regions)
from lanczos_ed.observables.basic import (
    entanglement_entropy, accessible_entanglement_entropy)


def build_once(L, N):
    """Basis, hopping (off-diagonal) matrix, occupation array, geometry.

    Built once and reused across the whole scan (only the diagonal changes).
    """
    # a model with all interactions off => hamiltonian() is pure hopping
    m = BoseHubbardKagome(linear_size=L, hopping=1.0, total_particles=N,
                          hardcore=True)
    H_hop = m.hamiltonian().copy()          # pure NN hopping (t=1)
    basis = m.basis
    dim = basis.dim
    nsite = m.num_sites
    occ = np.empty((dim, nsite), dtype=np.int16)
    for k in range(dim):
        occ[k] = basis.get_state(k)
    shells, dists = kagome_neighbor_shells(L, n_shells=3)
    hexes = kagome_hexagons(L)
    return m, basis, H_hop, occ, shells, hexes, dists


def diagonal(occ, route, coupling, shells, hexes, v2r, v3r):
    """Vectorized interaction diagonal for the given coupling (t=1 units)."""
    dim = occ.shape[0]
    diag = np.zeros(dim)
    if route == 'A':
        W = coupling
        for h in hexes:
            n_hex = occ[:, list(h)].sum(axis=1).astype(np.float64)
            diag += W * n_hex * n_hex
    else:  # route B: pairwise V1,V2,V3
        for coup, bonds in [(coupling, shells[0]),
                            (coupling * v2r, shells[1]),
                            (coupling * v3r, shells[2])]:
            if coup != 0.0 and bonds:
                b = np.array(bonds)
                diag += coup * (occ[:, b[:, 0]] * occ[:, b[:, 1]]).sum(axis=1)
    return diag


def ground_state(H, dim):
    if dim <= 400:
        w, v = np.linalg.eigh(H.toarray())
        return v[:, 0], w[0]
    w, v = eigsh(H, k=1, which='SA')
    return v[:, 0], w[0]


def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--route', choices=['A', 'B'], default='B')
    p.add_argument('--L', type=int, default=2)
    p.add_argument('--N', type=int, default=None,
                   help="particle number; default: half filling (A) or 1/3 (B)")
    p.add_argument('--coupling-min', type=float, default=0.0)
    p.add_argument('--coupling-max', type=float, default=12.0)
    p.add_argument('--steps', type=int, default=25)
    p.add_argument('--v2-ratio', type=float, default=1.0)
    p.add_argument('--v3-ratio', type=float, default=1.0)
    p.add_argument('--tee', action='store_true',
                   help="also compute Kitaev-Preskill gamma/gamma_acc/gamma_H "
                        "(meaningful only for L>=3)")
    p.add_argument('--kp-radius', type=float, default=1.05)
    p.add_argument('--out', default=None)
    args = p.parse_args()

    nsite = 3 * args.L * args.L
    if args.N is not None:
        N = args.N
    elif args.route == 'A':
        if nsite % 2 != 0:
            p.error(f"half filling not integer for L={args.L} "
                    f"({nsite} sites); pass --N")
        N = nsite // 2
    else:
        if nsite % 3 != 0:
            p.error(f"1/3 filling not integer for L={args.L}; pass --N")
        N = nsite // 3

    out = args.out or f"scan_tee_route{args.route}_L{args.L}_N{N}.dat"
    print(f"# kagome {args.L}x{args.L} = {nsite} sites, N={N}, route {args.route}")
    t0 = time.time()
    m, basis, H_hop, occ, shells, hexes, dists = build_once(args.L, N)
    print(f"# dim = {basis.dim};  build-once {time.time()-t0:.1f}s;  "
          f"shell distances {[round(d,3) for d in dists]}")

    half = list(range(nsite // 2))
    if args.tee:
        A, B, C = kitaev_preskill_regions(m, radius=args.kp_radius)
        print(f"# KP regions: |A|={len(A)} |B|={len(B)} |C|={len(C)} "
              f"disk={len(set(A)|set(B)|set(C))}/{nsite}"
              + ("   (WARNING: L<3, gamma not physically meaningful)"
                 if args.L < 3 else ""))

    # warn if the interaction is a pure constant shift at this size/ratio
    # (e.g. equal V1=V2=V3 on the tiny L=2 kagome torus)
    probe = diagonal(occ, args.route, 1.0, shells, hexes,
                     args.v2_ratio, args.v3_ratio)
    if probe.std() < 1e-9:
        print("# WARNING: interaction diagonal is CONSTANT across Fock states "
              "at this size/ratio -> coupling is a pure energy shift and the "
              "ground state will not change. Use L>=3, or (route B) unequal "
              "--v2-ratio/--v3-ratio or V1-only (--v2-ratio 0 --v3-ratio 0).")

    couplings = np.linspace(args.coupling_min, args.coupling_max, args.steps)
    rows = []
    header = "coupling   S        S_acc    H        varNA"
    if args.tee:
        header += "     gamma    gamma_acc gamma_H"
    print("#", header)
    for c in couplings:
        diag = diagonal(occ, args.route, c, shells, hexes,
                        args.v2_ratio, args.v3_ratio)
        H = (H_hop + sparse.diags(diag)).tocsr()
        psi, e0 = ground_state(H, basis.dim)

        S = entanglement_entropy(psi, basis, half, 1.0)
        Sacc = accessible_entanglement_entropy(psi, basis, half, 1.0)
        Hnum = S - Sacc
        varNA = bipartite_fluctuations(psi, basis, half)
        line = [c, S, Sacc, Hnum, varNA]
        if args.tee:
            r = topological_entanglement_entropy(psi, basis, A, B, C, 1.0)
            line += [r['gamma'], r['gamma_acc'], r['gamma_H']]
        rows.append(line)
        fmt = "  ".join(f"{x:8.4f}" for x in line)
        print(f"  {fmt}")

    rows = np.array(rows)
    np.savetxt(out, rows, header=header, fmt="%.6f")
    print(f"# wrote {out}  ({time.time()-t0:.1f}s total)")
    # crude pointer to the target regime
    if args.tee and rows.shape[1] >= 8:
        good = (rows[:, 5] > 0.02) & (rows[:, 3] > 0.05)  # gamma>0 and H>0
        if good.any():
            cg = rows[good, 0]
            print(f"# candidate region (gamma>0.02 AND H>0.05): "
                  f"coupling in [{cg.min():.3f}, {cg.max():.3f}]")
        else:
            print("# no scan point had both gamma>0.02 and H>0.05 "
                  "(size too small, or scan the other coupling range)")


if __name__ == '__main__':
    main()
