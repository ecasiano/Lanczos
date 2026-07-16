"""
Validation tests for the kagome-lattice extended Bose-Hubbard model
(lanczos_ed/models/bose_hubbard_kagome.py), added for the TEE project.

Covers:
    1. Geometry: 3L^2 sites, coordination number 4, NN bond count 6L^2,
       neighbor-shell distances (0.5, sqrt(3)/2, 1.0).
    2. Kagome flat band: single-particle spectrum has an L^2(+1)-fold
       degenerate flat band at E = +2t (the defining kagome signature).
    3. Hexagons: exactly L^2 hexagons, 6 sites each, every site in 2 hexagons.
    4. Full Hamiltonian spectrum vs an independent from-scratch builder
       (hardcore), for Route B (V1,V2,V3) and Route A (cluster charging).
    5. Hermiticity.
    6. Cluster-charging = pairwise-in-hexagon identity for hardcore bosons:
       W (n_hex)^2 = W n_hex + 2W sum_{pairs in hex} n_i n_j.

Run directly:  python3 tests/test_kagome.py
"""

import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
from itertools import combinations

from lanczos_ed.models.bose_hubbard_kagome import (
    BoseHubbardKagome, kagome_positions, kagome_neighbor_shells,
    kagome_hexagons,
)

PASS = 0
FAIL = 0

def check(name, ok, extra=""):
    global PASS, FAIL
    print(f"  [{'PASS' if ok else 'FAIL'}] {name}  {extra}")
    if ok: PASS += 1
    else:  FAIL += 1


print("Test 1: geometry")
for L in (2, 3, 4):
    pos = kagome_positions(L)
    shells, dists = kagome_neighbor_shells(L, n_shells=3)
    n = 3 * L * L
    deg = np.zeros(n, int)
    for a, b in shells[0]:
        deg[a] += 1; deg[b] += 1
    check(f"L={L}: 3L^2={n} sites", pos.shape[0] == n)
    check(f"L={L}: coordination number 4", np.all(deg == 4))
    check(f"L={L}: NN bonds = 6L^2 = {6*L*L}", len(shells[0]) == 6*L*L)
    check(f"L={L}: shell distances = (0.5, sqrt3/2, 1.0)",
          np.allclose(dists, [0.5, np.sqrt(3)/2, 1.0], atol=1e-3),
          f"got {[round(d,4) for d in dists]}")

print("\nTest 2: kagome flat band (single particle)")
for L in (2, 3, 4):
    m = BoseHubbardKagome(linear_size=L, hopping=1.0, total_particles=1, hardcore=True)
    ev = np.linalg.eigvalsh(m.single_particle_hopping_matrix())
    vals, counts = np.unique(np.round(ev, 6), return_counts=True)
    top = counts.max(); topval = vals[counts.argmax()]
    check(f"L={L}: flat band deg >= L^2 = {L*L} at E=+2t",
          top >= L*L - 1 and abs(topval - 2.0) < 1e-6,
          f"deg={top} at E={topval:.4f}")

print("\nTest 3: hexagons")
for L in (2, 3, 4):
    hexes = kagome_hexagons(L)
    n = 3 * L * L
    cnt = np.zeros(n, int)
    for h in hexes:
        for s in h: cnt[s] += 1
    check(f"L={L}: L^2 = {L*L} hexagons", len(hexes) == L*L)
    check(f"L={L}: 6 sites each", all(len(set(h)) == 6 for h in hexes))
    check(f"L={L}: each site in 2 hexagons", np.all(cnt == 2))


def indep_kagome_H(L, N, t, V1, V2, V3, W):
    """Independent hardcore Hamiltonian; states = occupied-site frozensets."""
    shells, _ = kagome_neighbor_shells(L, n_shells=3)
    hexes = kagome_hexagons(L) if W != 0 else []
    nsite = 3 * L * L
    states = [frozenset(c) for c in combinations(range(nsite), N)]
    idx = {s: i for i, s in enumerate(states)}
    D = len(states)
    H = np.zeros((D, D))
    couplings = [(V1, shells[0]), (V2, shells[1]), (V3, shells[2])]
    for i, s in enumerate(states):
        d = 0.0
        for c, bonds in couplings:
            if c != 0:
                for a, b in bonds:
                    if a in s and b in s: d += c
        if W != 0:
            for h in hexes:
                nh = sum(1 for q in h if q in s); d += W * nh * nh
        H[i, i] += d
        for a, b in shells[0]:
            if a in s and b not in s:
                H[idx[(s - {a}) | {b}], i] += -t
            if b in s and a not in s:
                H[idx[(s - {b}) | {a}], i] += -t
    return H


print("\nTest 4/5: full Hamiltonian spectrum vs independent builder + Hermiticity")
cases = [
    (2, 4, 1.0, 1.5, 0.0, 0.0, 0.0),   # Route B: NN only
    (2, 4, 1.0, 1.0, 0.7, 0.4, 0.0),   # Route B: V1,V2,V3
    (2, 4, 1.0, 0.0, 0.0, 0.0, 2.0),   # Route A: cluster charging
    (2, 6, 1.0, 0.8, 0.0, 0.0, 1.3),   # mixed, half filling
    (3, 3, 1.0, 1.0, 0.5, 0.0, 0.0),   # L=3 Route B
]
for (L, N, t, V1, V2, V3, W) in cases:
    m = BoseHubbardKagome(linear_size=L, hopping=t, total_particles=N, hardcore=True,
                          nn_interaction=V1, v2_interaction=V2,
                          v3_interaction=V3, cluster_charging=W)
    Ha = m.hamiltonian().toarray()
    Hi = indep_kagome_H(L, N, t, V1, V2, V3, W)
    herm = np.allclose(Ha, Ha.T)
    same = Ha.shape == Hi.shape and np.allclose(
        np.linalg.eigvalsh(Ha), np.linalg.eigvalsh(Hi), atol=1e-9)
    check(f"L={L} N={N} V=({V1},{V2},{V3}) W={W}: Hermitian + spectrum==indep",
          herm and same)

print("\nTest 6: cluster = pairwise-in-hexagon identity (hardcore)")
for (L, N) in [(2, 4), (2, 6), (3, 3), (3, 6)]:
    hexes = kagome_hexagons(L)
    m = BoseHubbardKagome(linear_size=L, hopping=0.0, total_particles=N,
                          hardcore=True, cluster_charging=1.0)
    d_clu = m.hamiltonian().diagonal()  # sparse diagonal (no dense blowup)
    d_pair = np.zeros(m.dim)
    for k in range(m.dim):
        s = m.basis.get_state(k)
        tot = 0
        for h in hexes:
            for a, b in combinations(h, 2):
                if s[a] and s[b]: tot += 1
        d_pair[k] = 2 * tot
    diff = d_clu - d_pair
    check(f"L={L} N={N}: cluster - 2*pairwise_in_hex = 2N = {2*N}",
          np.allclose(diff, diff[0]) and abs(diff[0] - 2*N) < 1e-9)

total = PASS + FAIL
print(f"\n{'='*60}")
print(f"Results: {PASS} passed, {FAIL} failed out of {total} tests")
print(f"{'='*60}")
sys.exit(0 if FAIL == 0 else 1)
