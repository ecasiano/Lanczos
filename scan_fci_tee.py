#!/usr/bin/env python3
"""
Fractional Chern Insulator: TEE accessible decomposition scan
=============================================================

Computes the ground state of the kagome FCI at ν = 1/3 and measures
the Kitaev-Preskill TEE decomposition  γ = γ_Acc + γ_H.

The model: kagome tight-binding with complex hopping t·e^{iφ} (φ = 5π/4),
nearest-neighbor interaction V projected into the lowest Chern band (C=1).
Flat-band limit (κ=0) by default.

References:
    Regnault & Bernevig, Phys. Rev. X 1, 021014 (2011)
    Kwan & Regnault, FCI ED Tutorial (DIPC 2024)
"""

import argparse, time, bisect
import numpy as np
from itertools import combinations
from collections import defaultdict


# =====================================================================
# Single-particle: kagome Chern insulator
# =====================================================================

def bloch_hamiltonian(k1, k2, phi):
    """3×3 Bloch Hamiltonian for kagome with complex hopping.

    h(k) = t  [ 0                e^{-iφ}(1+e^{-ik₁})    e^{iφ}(1+e^{-ik₂})      ]
               [ e^{iφ}(1+e^{ik₁})     0                e^{-iφ}(1+e^{i(k₁-k₂)})  ]
               [ e^{-iφ}(1+e^{ik₂})   e^{iφ}(1+e^{i(k₂-k₁)})   0                ]
    """
    h = np.zeros((3, 3), dtype=complex)
    h[0, 1] = np.exp(-1j*phi) * (1.0 + np.exp(-1j*k1))
    h[1, 0] = h[0, 1].conj()
    h[0, 2] = np.exp(1j*phi) * (1.0 + np.exp(-1j*k2))
    h[2, 0] = h[0, 2].conj()
    h[1, 2] = np.exp(-1j*phi) * (1.0 + np.exp(1j*(k1-k2)))
    h[2, 1] = h[1, 2].conj()
    return h


def compute_bands(N1, N2, phi):
    """Diagonalize h(k) at every k-point on the N₁×N₂ mesh."""
    N = N1 * N2
    eigvecs = np.zeros((N, 3, 3), dtype=complex)
    energies = np.zeros((N, 3))
    for K in range(N):
        e, v = np.linalg.eigh(bloch_hamiltonian(2*np.pi*(K//N2)/N1, 2*np.pi*(K%N2)/N2, phi))
        energies[K] = e
        eigvecs[K] = v
    return energies, eigvecs


def chern_number(N1, N2, eigvecs, band=0):
    """Chern number via discretized Berry curvature (Fukui et al. 2005)."""
    def u(kt1, kt2):
        return eigvecs[((kt1 % N1)*N2 + (kt2 % N2)), :, band]
    def link(a1, a2, b1, b2):
        ov = np.vdot(u(a1, a2), u(b1, b2))
        return ov / abs(ov)
    total = 0.0
    for kt1 in range(N1):
        for kt2 in range(N2):
            P = (link(kt1, kt2, kt1+1, kt2)
                 * link(kt1+1, kt2, kt1+1, kt2+1)
                 * np.conj(link(kt1, kt2+1, kt1+1, kt2+1))
                 * np.conj(link(kt1, kt2, kt1, kt2+1)))
            total += np.angle(P)
    return total / (2*np.pi)


# =====================================================================
# Many-body: projected Hamiltonian
# =====================================================================

def k4_from(K1, K2, K3, N1, N2):
    """Momentum conservation: k₄ = k₁ + k₂ - k₃ (mod reciprocal lattice)."""
    return ((K1//N2+K2//N2-K3//N2)%N1)*N2 + (K1%N2+K2%N2-K3%N2)%N2


def nn_vq(q1, q2):
    """Fourier-transformed NN interaction V^{αβ}(q) on kagome.

    V^{AB}(q) = 1 + e^{iq₁}
    V^{AC}(q) = 1 + e^{iq₂}
    V^{BC}(q) = 1 + e^{-i(q₁-q₂)}
    """
    V = np.zeros((3, 3), dtype=complex)
    V[0,1] = 1+np.exp(1j*q1);  V[1,0] = 1+np.exp(-1j*q1)
    V[0,2] = 1+np.exp(1j*q2);  V[2,0] = 1+np.exp(-1j*q2)
    V[1,2] = 1+np.exp(-1j*(q1-q2)); V[2,1] = 1+np.exp(1j*(q1-q2))
    return V


def raw_U(K1, K2, K3, K4, u, N1, N2):
    """Band-projected interaction matrix element U_{K1K2K3K4}.

    U = Σ_{αβ} u*_α(K1) u*_β(K2) u_α(K3) u_β(K4) V^{αβ}(q)
    where q = k₃ - k₁.
    """
    q1 = 2*np.pi*(K3//N2-K1//N2)/N1
    q2 = 2*np.pi*(K3%N2-K1%N2)/N2
    Vq = nn_vq(q1, q2)
    return np.dot(np.conj(u[K1])*u[K3], Vq @ (np.conj(u[K2])*u[K4]))


def build_and_solve(N1, N2, eigvecs, V=1.0, kappa=0.0, num_eig=6):
    """Build band-projected Hamiltonian and diagonalize.

    H_int = (1/N) Σ_{k1<k2, k3<k4} δ_{k4=k1+k2-k3}
            Ũ_{k1k2k3k4} d†_{k1} d†_{k2} d_{k4} d_{k3}

    where Ũ = U_{k1k2k3k4} - U_{k1k2k4k3} (antisymmetrized).
    """
    N = N1 * N2
    N_p = N // 3
    u = eigvecs[:, :, 0]
    states = list(combinations(range(N), N_p))
    dim = len(states)
    idx_map = {s: i for i, s in enumerate(states)}
    print(f"  Building H: dim = {dim}")
    t0 = time.time()
    H = np.zeros((dim, dim), dtype=complex)

    for idx, st in enumerate(states):
        for ia in range(N_p):
            Ka1 = st[ia]
            for ib in range(ia+1, N_p):
                Ka2 = st[ib]
                inter = [K for K in st if K != Ka1 and K != Ka2]
                iset = set(inter)
                # Fermionic sign for d_{Ka2} d_{Ka1} |state⟩
                sa = (-1)**ia * (-1)**(ib-1)

                for Kc1 in range(N):
                    if Kc1 in iset:
                        continue
                    Kc2 = k4_from(Ka1, Ka2, Kc1, N1, N2)
                    if Kc2 <= Kc1 or Kc2 in iset:
                        continue

                    # Antisymmetrized matrix element (Eq. 33 of tutorial)
                    Ua = (raw_U(Kc1, Kc2, Ka1, Ka2, u, N1, N2)
                          - raw_U(Kc1, Kc2, Ka2, Ka1, u, N1, N2))
                    if abs(Ua) < 1e-15:
                        continue

                    # Fermionic sign for d†_{Kc1} d†_{Kc2} |intermediate⟩
                    p2 = bisect.bisect_left(inter, Kc2)
                    nl = list(inter)
                    nl.insert(p2, Kc2)
                    p1 = bisect.bisect_left(nl, Kc1)
                    nl.insert(p1, Kc1)

                    jdx = idx_map.get(tuple(sorted(nl)), -1)
                    if jdx < 0:
                        continue
                    H[jdx, idx] += sa * (-1)**p1 * (-1)**p2 * Ua / N

    H = 0.5 * (H + H.conj().T)
    t_build = time.time() - t0
    print(f"  Diagonalizing...")
    t0 = time.time()
    evals, evecs = np.linalg.eigh(H)
    t_diag = time.time() - t0
    print(f"  Build: {t_build:.1f}s,  diag: {t_diag:.1f}s")
    return evals, evecs, states, idx_map


# =====================================================================
# Real-space transformation via Slater determinants
# =====================================================================

def build_transform(N1, N2, eigvecs, band=0):
    """Single-particle transformation T_{i,K} = u_{α}(K) e^{iK·R} / √N.

    Maps band orbital K to real-space site i = (R, α):
        d†_K = Σ_i T_{iK} c†_i
    """
    N = N1 * N2
    N_sites = 3 * N
    u = eigvecs[:, :, band]
    T = np.zeros((N_sites, N), dtype=complex)
    for K in range(N):
        kt1, kt2 = K // N2, K % N2
        for R2 in range(N2):
            for R1 in range(N1):
                # k·R = 2π(kt1·R1/N1 + kt2·R2/N2) from aᵢ·bⱼ = 2π δᵢⱼ
                phase = np.exp(2j*np.pi*(kt1*R1/N1 + kt2*R2/N2))
                cell = R2*N1 + R1
                for alpha in range(3):
                    T[alpha + 3*cell, K] = u[K, alpha] * phase / np.sqrt(N)
    return T


def to_real_space(psi_k, k_states, T, N_sites, N_p):
    """Transform k-space ground state to real-space Fock basis.

    ⟨i₁...i_{Np} | K₁...K_{Np}⟩ = det[T_{iₐ,Kᵦ}]  (Slater determinant)
    """
    real_states = list(combinations(range(N_sites), N_p))
    dim_real = len(real_states)
    print(f"  Transforming to real space: {len(k_states)} -> {dim_real} states")
    t0 = time.time()
    psi_real = np.zeros(dim_real, dtype=complex)
    for idx_k in range(len(k_states)):
        c = psi_k[idx_k]
        if abs(c) < 1e-15:
            continue
        T_cols = T[:, k_states[idx_k]]
        for idx_r, sites in enumerate(real_states):
            psi_real[idx_r] += c * np.linalg.det(T_cols[sites, :])
    norm = np.sqrt(np.sum(np.abs(psi_real)**2))
    psi_real /= norm
    print(f"  Transform: {time.time()-t0:.1f}s, norm = {norm:.6f}")
    return psi_real, real_states


# =====================================================================
# Entanglement decomposition S = S_acc + H
# =====================================================================

def compute_S_Sacc(psi, real_states, subsys):
    """Von Neumann entropy S and accessible entropy S_acc for a subregion.

    Decomposes ρ_A = ⊕_n p(n) ρ_A^{(n)} by particle number in A.
    S_acc = Σ_n p(n) S(ρ_A^{(n)})  (within-sector entanglement).
    H = S - S_acc = -Σ_n p(n) ln p(n)  (number fluctuation entropy).
    """
    sub_set = set(subsys)
    sectors = defaultdict(list)
    for idx_r, sites in enumerate(real_states):
        amp = psi[idx_r]
        if abs(amp) < 1e-15:
            continue
        sA = tuple(s for s in sites if s in sub_set)
        sB = tuple(s for s in sites if s not in sub_set)
        sectors[len(sA)].append((sA, sB, amp))
    S = 0.0
    Sacc = 0.0
    for nA, entries in sectors.items():
        A_configs = sorted(set(e[0] for e in entries))
        B_configs = sorted(set(e[1] for e in entries))
        Ai = {c: i for i, c in enumerate(A_configs)}
        Bi = {c: i for i, c in enumerate(B_configs)}
        M = np.zeros((len(A_configs), len(B_configs)), dtype=complex)
        for sA, sB, amp in entries:
            M[Ai[sA], Bi[sB]] = amp
        _, sigma, _ = np.linalg.svd(M, full_matrices=False)
        w = sigma**2
        pn = np.sum(w)
        for wi in w:
            if wi > 1e-15:
                S -= wi * np.log(wi)
        if pn > 1e-15:
            for wi in w:
                if wi > 1e-15:
                    Sacc -= wi * np.log(wi / pn)
    return S, Sacc


# =====================================================================
# Kitaev-Preskill TEE construction
# =====================================================================

def kitaev_preskill_regions(positions, N_sites, center=None, radius=1.05):
    """Three 120° pie-slice regions of a disk for the KP combination.

    S_topo = S_A + S_B + S_C - S_AB - S_BC - S_AC + S_ABC = -γ
    """
    if center is None:
        center = positions.mean(axis=0)
    center = np.asarray(center)
    A, B, C = [], [], []
    for s in range(N_sites):
        d = positions[s] - center
        r = np.hypot(d[0], d[1])
        if r > radius:
            continue
        theta = np.arctan2(d[1], d[0]) % (2*np.pi)
        sector = int(theta // (2*np.pi/3))
        if sector == 0:
            A.append(s)
        elif sector == 1:
            B.append(s)
        else:
            C.append(s)
    return A, B, C


def compute_tee(psi, real_states, regA, regB, regC):
    """Full KP TEE decomposition: γ = γ_acc + γ_H."""
    regions = {
        'A': regA, 'B': regB, 'C': regC,
        'AB': regA+regB, 'BC': regB+regC, 'AC': regA+regC,
        'ABC': regA+regB+regC,
    }
    Sd, Sad, Hd = {}, {}, {}
    for name, sites in regions.items():
        s, sa = compute_S_Sacc(psi, real_states, sites)
        Sd[name] = s
        Sad[name] = sa
        Hd[name] = s - sa
    def kp(x):
        return x['A']+x['B']+x['C']-x['AB']-x['BC']-x['AC']+x['ABC']
    return {
        'gamma': -kp(Sd), 'gamma_acc': -kp(Sad), 'gamma_H': -kp(Hd),
        'S': Sd, 'S_acc': Sad, 'H': Hd,
    }


# =====================================================================
# Kagome lattice geometry
# =====================================================================

def kagome_positions(N1, N2):
    """Real-space positions of all 3·N₁·N₂ sites.

    Bravais vectors: a₁ = (1,0), a₂ = (1/2, √3/2)
    Sublattices: A=(0,0), B=(1/2,0), C=(1/4, √3/4)
    """
    A1 = np.array([1.0, 0.0])
    A2 = np.array([0.5, np.sqrt(3)/2])
    sub = np.array([[0, 0], [0.5, 0], [0.25, np.sqrt(3)/4]])
    pos = []
    for R2 in range(N2):
        for R1 in range(N1):
            R = R1*A1 + R2*A2
            for alpha in range(3):
                pos.append(R + sub[alpha])
    return np.array(pos)


# =====================================================================
# Main
# =====================================================================

def main():
    parser = argparse.ArgumentParser(description="FCI TEE scan")
    parser.add_argument("--N1", type=int, default=3)
    parser.add_argument("--N2", type=int, default=4)
    parser.add_argument("--phi", type=float, default=5*np.pi/4)
    parser.add_argument("--V", type=float, default=1.0)
    parser.add_argument("--kappa", type=float, default=0.0)
    parser.add_argument("--kp-radius", type=float, default=1.0)
    parser.add_argument("--spectrum", action="store_true")
    parser.add_argument("--tee", action="store_true")
    parser.add_argument("--bipartite", action="store_true")
    args = parser.parse_args()

    N1, N2 = args.N1, args.N2
    N = N1 * N2
    N_p = N // 3
    N_sites = 3 * N

    print(f"Kagome FCI: {N1}x{N2} torus, {N_sites} sites, {N_p} particles, v=1/3")
    print(f"  phi = {args.phi:.4f}, V = {args.V}, kappa = {args.kappa}")

    energies, eigvecs = compute_bands(N1, N2, args.phi)
    C = chern_number(N1, N2, eigvecs)
    gap = np.min(energies[:, 1]) - np.max(energies[:, 0])
    bw = np.max(energies[:, 0]) - np.min(energies[:, 0])
    print(f"  Chern number = {C:.4f}")
    print(f"  Band gap = {gap:.4f}, bandwidth = {bw:.4f}, flatness = {gap/bw:.2f}")

    evals, evecs, states, idx_map = build_and_solve(N1, N2, eigvecs, V=args.V, kappa=args.kappa)

    print(f"\nLowest 8 eigenvalues:")
    for i in range(min(8, len(evals))):
        print(f"  E_{i} = {evals[i]:.8f}")
    spread = evals[2] - evals[0]
    gap_3 = evals[3] - evals[2]
    print(f"  3-fold spread = {spread:.6e}")
    print(f"  Gap = {gap_3:.6f}")
    print(f"  spread/gap = {spread/gap_3:.6e}")

    if args.spectrum:
        sectors = {}
        for idx, st in enumerate(states):
            kt1 = sum(K // N2 for K in st) % N1
            kt2 = sum(K % N2 for K in st) % N2
            sectors.setdefault((kt1, kt2), []).append(idx)
        print(f"\nMomentum-resolved spectrum:")
        all_es = []
        H_full = np.zeros((len(states), len(states)), dtype=complex)
        for n in range(len(evals)):
            H_full += evals[n] * np.outer(evecs[:, n], evecs[:, n].conj())
        for key in sorted(sectors.keys()):
            idxs = sectors[key]
            H_sec = H_full[np.ix_(idxs, idxs)]
            esec = np.linalg.eigvalsh(H_sec)
            for e in esec[:5]:
                all_es.append((e, key))
        all_es.sort()
        for e, k in all_es[:15]:
            print(f"  E = {e:.8f}  K = ({k[0]},{k[1]})")
        return

    if args.bipartite or args.tee:
        T = build_transform(N1, N2, eigvecs)
        psi_k = evecs[:, 0]
        psi_real, real_states = to_real_space(psi_k, states, T, N_sites, N_p)

    if args.bipartite:
        subsys = list(range(N_sites // 2))
        S, Sacc = compute_S_Sacc(psi_real, real_states, subsys)
        H = S - Sacc
        print(f"\nBipartite entanglement ({len(subsys)} of {N_sites} sites):")
        print(f"  S     = {S:.6f}")
        print(f"  S_acc = {Sacc:.6f}")
        print(f"  H     = {H:.6f}")

    if args.tee:
        positions = kagome_positions(N1, N2)
        regA, regB, regC = kitaev_preskill_regions(positions, N_sites, radius=args.kp_radius)
        disk = regA + regB + regC
        env = [s for s in range(N_sites) if s not in set(disk)]
        print(f"\nKP regions (radius={args.kp_radius}):")
        print(f"  Disk: {len(disk)} sites ({len(regA)}+{len(regB)}+{len(regC)})")
        print(f"  Environment: {len(env)} sites")
        if len(regA) == 0 or len(regB) == 0 or len(regC) == 0:
            print("  ERROR: empty KP region")
            return
        if len(env) < 3:
            print("  WARNING: very small environment")
        print("  Computing 7 region entropies...")
        t0 = time.time()
        result = compute_tee(psi_real, real_states, regA, regB, regC)
        print(f"  TEE computation: {time.time()-t0:.1f}s")
        print(f"\n  gamma     = {result['gamma']:.6f}  (target: {0.5*np.log(3):.6f} = 1/2 ln 3)")
        print(f"  gamma_acc = {result['gamma_acc']:.6f}")
        print(f"  gamma_H   = {result['gamma_H']:.6f}")
        print(f"\n  Region entropies:")
        for name in ['A', 'B', 'C', 'AB', 'BC', 'AC', 'ABC']:
            print(f"    {name:>3s}: S={result['S'][name]:.6f}  S_acc={result['S_acc'][name]:.6f}  H={result['H'][name]:.6f}")

    if not args.bipartite and not args.tee and not args.spectrum:
        print("\nUse --tee, --bipartite, or --spectrum for detailed output.")


if __name__ == "__main__":
    main()
