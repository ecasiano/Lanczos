"""
Topological entanglement entropy (TEE) and its accessible / number-fluctuation
decomposition.
============================================================================

The topological entanglement entropy gamma is extracted with the
**Kitaev-Preskill** 3-region construction (Phys. Rev. Lett. 96, 110404 (2006)),
which is the exact-diagonalization-viable one (the Levin-Wen annulus needs far
larger systems). For three regions A, B, C arranged so that their union is a
simply-connected disk meeting at a central tripoint,

    S_topo = S_A + S_B + S_C - S_AB - S_BC - S_AC + S_ABC = -gamma ,

with the boundary-law and corner terms cancelling, leaving the universal -gamma.

The point of THIS project is to split gamma. For every region we compute both

    S      = full entanglement entropy            (von Neumann or Renyi)
    S_acc  = operationally accessible entropy      (Barghathi et al.,
             PRB 105, L121116 (2022)) -- entanglement usable under a
             particle-number superselection rule
    H      = S - S_acc                             (number / fluctuation entropy)

Because the Kitaev-Preskill combination is *linear* in the region entropies,

    gamma      = -KP(S)
    gamma_Acc  = -KP(S_acc)
    gamma_H    = -KP(H) = gamma - gamma_Acc

gamma_H is the part of the topological entanglement entropy that comes from
particle-number fluctuations across the cut; gamma_Acc is the part that is
operationally accessible. The central question is whether gamma_H = 0 (all of
gamma is accessible) or gamma_H != 0.

Design note
-----------
gamma is only meaningful when (a) the state is topological (gamma > 0) AND
(b) number fluctuations across the cut are alive (H > 0). In the trivial
Mott/frozen limit H -> 0 and the split is vacuous. Use `number_entropy` /
`bipartite_number_entropy` as the cheap indicator of (b) when scanning.
"""

import numpy as np
from .basic import (
    entanglement_entropy,
    accessible_entanglement_entropy,
    bipartite_fluctuations,
)


def region_entropies(wavefunction, basis, sites, renyi_index=1.0):
    """Return (S, S_acc, H) for one subregion given by ``sites``.

    H = S - S_acc is the number (fluctuation) entropy.
    """
    sites = sorted(set(int(s) for s in sites))
    S = entanglement_entropy(wavefunction, basis, sites, renyi_index)
    S_acc = accessible_entanglement_entropy(wavefunction, basis, sites, renyi_index)
    return S, S_acc, (S - S_acc)


def number_entropy(wavefunction, basis, sites, renyi_index=1.0):
    """Number (fluctuation) entropy H = S - S_acc of one subregion."""
    return region_entropies(wavefunction, basis, sites, renyi_index)[2]


def bipartite_number_entropy(wavefunction, basis, subsystem_sites=None,
                             renyi_index=1.0):
    """H for a bipartition (default: first half of sites). Cheap fluctuation probe."""
    num_sites = basis.num_sites
    if subsystem_sites is None:
        subsystem_sites = list(range(num_sites // 2))
    return number_entropy(wavefunction, basis, subsystem_sites, renyi_index)


def topological_entanglement_entropy(wavefunction, basis, A, B, C,
                                     renyi_index=1.0):
    """Kitaev-Preskill gamma and its accessible / number-fluctuation split.

    Parameters
    ----------
    wavefunction : ndarray (dim,)
        Ground-state vector in the full basis.
    basis : FockBasis or UnaryBasis
    A, B, C : list[int]
        Disjoint site lists forming the three Kitaev-Preskill regions.
        Their union should be a compact, simply-connected disk with the three
        regions meeting at a central tripoint, embedded in a larger system.
    renyi_index : float
        alpha for the entropies (1.0 = von Neumann, the standard TEE).

    Returns
    -------
    dict with keys:
        'gamma', 'gamma_acc', 'gamma_H'  : the three topological quantities
        'S', 'S_acc', 'H'                : dict {region_name: value} for the
                                           7 regions A,B,C,AB,BC,AC,ABC
        'renyi_index'
    """
    A = set(int(s) for s in A)
    B = set(int(s) for s in B)
    C = set(int(s) for s in C)
    if A & B or B & C or A & C:
        raise ValueError("Kitaev-Preskill regions A, B, C must be disjoint")

    regions = {
        'A': A, 'B': B, 'C': C,
        'AB': A | B, 'BC': B | C, 'AC': A | C,
        'ABC': A | B | C,
    }

    S, S_acc, H = {}, {}, {}
    for name, sites in regions.items():
        s, sacc, h = region_entropies(wavefunction, basis, sites, renyi_index)
        S[name], S_acc[name], H[name] = s, sacc, h

    def kp(x):
        return (x['A'] + x['B'] + x['C']
                - x['AB'] - x['BC'] - x['AC']
                + x['ABC'])

    gamma = -kp(S)
    gamma_acc = -kp(S_acc)
    gamma_H = -kp(H)  # == gamma - gamma_acc

    return {
        'gamma': gamma,
        'gamma_acc': gamma_acc,
        'gamma_H': gamma_H,
        'S': S, 'S_acc': S_acc, 'H': H,
        'renyi_index': renyi_index,
    }


# =====================================================================
# Kitaev-Preskill region construction on a lattice
# =====================================================================

def kitaev_preskill_regions(model, center=None, radius=None,
                            angle_offset=0.0):
    """Build three Kitaev-Preskill regions (pie slices of a disk).

    Sites within ``radius`` of ``center`` form the disk ABC; each disk site is
    assigned to A, B, or C by its polar angle (three 120-degree sectors). The
    complement is the environment.

    Uses the model's real-space positions WITHOUT minimum image, so choose a
    ``center`` in the interior of the (unfolded) cluster and a ``radius`` small
    enough that the disk does not wrap the torus. Returns the three site lists.

    Parameters
    ----------
    model : object with .positions() -> (num_sites, 2) and .num_sites
    center : (x, y) or None
        Disk center. Default: geometric centroid of all sites.
    radius : float or None
        Disk radius. Default: 1.05 (about two NN shells on the unit-scale
        kagome lattice -> a compact disk of order ~9-12 sites).
    angle_offset : float
        Rotate the 3 sector boundaries (radians).

    Returns
    -------
    A, B, C : list[int]
    """
    pos = model.positions()
    n = pos.shape[0]
    if center is None:
        center = pos.mean(axis=0)
    center = np.asarray(center, dtype=float)
    if radius is None:
        radius = 1.05

    A, B, C = [], [], []
    two_pi_third = 2.0 * np.pi / 3.0
    for s in range(n):
        d = pos[s] - center
        r = np.hypot(d[0], d[1])
        if r > radius:
            continue
        theta = (np.arctan2(d[1], d[0]) - angle_offset) % (2.0 * np.pi)
        sector = int(theta // two_pi_third)
        if sector == 0:
            A.append(s)
        elif sector == 1:
            B.append(s)
        else:
            C.append(s)
    return A, B, C
