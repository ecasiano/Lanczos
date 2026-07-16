import sys; import os; sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import numpy as np
from lanczos_ed.models.bose_hubbard_kagome import BoseHubbardKagome
from lanczos_ed.observables.tee import (
    topological_entanglement_entropy, kitaev_preskill_regions,
    bipartite_number_entropy, region_entropies)

fails=0
def check(name, ok, extra=""):
    global fails
    print(f"  [{'PASS' if ok else 'FAIL'}] {name}  {extra}")
    if not ok: fails+=1

# ---- helper: ground state of a kagome model via dense eigh (small) ----
def ground_state(m):
    H = m.hamiltonian().toarray()
    w,v = np.linalg.eigh(H)
    return v[:,0], w[0]

print("Test 1: KP region construction (L=2 kagome, 12 sites)")
m = BoseHubbardKagome(linear_size=2, hopping=1.0, total_particles=4, hardcore=True)
A,B,C = kitaev_preskill_regions(m, radius=1.05)
disk = set(A)|set(B)|set(C)
check("regions disjoint", not(set(A)&set(B)) and not(set(B)&set(C)) and not(set(A)&set(C)))
check("disk non-empty and leaves an environment", 0 < len(disk) < m.num_sites,
      f"|A|={len(A)} |B|={len(B)} |C|={len(C)} disk={len(disk)}/{m.num_sites}")

print("\nTest 2: product (single Fock) state -> gamma=0, H=0")
# build a wavefunction that is a single basis state (no entanglement)
psi = np.zeros(m.dim); psi[0] = 1.0
res = topological_entanglement_entropy(psi, m.basis, A, B, C, renyi_index=1.0)
check("gamma == 0", abs(res['gamma'])<1e-12, f"gamma={res['gamma']:.2e}")
check("gamma_acc == 0", abs(res['gamma_acc'])<1e-12)
check("gamma_H == 0", abs(res['gamma_H'])<1e-12)
check("H(ABC)=0 for product state", abs(res['H']['ABC'])<1e-12)

print("\nTest 3: gamma_H = gamma - gamma_acc identity (random state)")
rng = np.random.default_rng(0)
psi = rng.standard_normal(m.dim); psi/=np.linalg.norm(psi)
res = topological_entanglement_entropy(psi, m.basis, A, B, C, renyi_index=1.0)
check("gamma_H == gamma - gamma_acc", abs(res['gamma_H']-(res['gamma']-res['gamma_acc']))<1e-10,
      f"gamma={res['gamma']:.4f} acc={res['gamma_acc']:.4f} H={res['gamma_H']:.4f}")

print("\nTest 4: number entropy H tracks fluctuations (hopping vs frozen)")
# hopping-dominated (superfluid-like): number fluctuations alive -> H>0
m_sf = BoseHubbardKagome(linear_size=2, hopping=1.0, total_particles=4, hardcore=True,
                         nn_interaction=0.0)
psi_sf,_ = ground_state(m_sf)
H_sf = bipartite_number_entropy(psi_sf, m_sf.basis, renyi_index=1.0)
# strong NN repulsion: charges freeze -> H smaller
m_fr = BoseHubbardKagome(linear_size=2, hopping=1.0, total_particles=4, hardcore=True,
                         nn_interaction=40.0)
psi_fr,_ = ground_state(m_fr)
H_fr = bipartite_number_entropy(psi_fr, m_fr.basis, renyi_index=1.0)
check("H(superfluid) > 0", H_sf>1e-3, f"H_sf={H_sf:.4f}")
check("H(frozen) < H(superfluid)", H_fr < H_sf, f"H_frozen={H_fr:.4f} < H_sf={H_sf:.4f}")

print("\nTest 5: full TEE decomposition runs on a ground state (L=2)")
m2 = BoseHubbardKagome(linear_size=2, hopping=1.0, total_particles=4, hardcore=True,
                       nn_interaction=1.0, v2_interaction=1.0, v3_interaction=1.0)
psi2,e0 = ground_state(m2)
res = topological_entanglement_entropy(psi2, m2.basis, A, B, C, renyi_index=1.0)
finite = all(np.isfinite(v) for v in [res['gamma'],res['gamma_acc'],res['gamma_H']])
check("gamma, gamma_acc, gamma_H all finite", finite,
      f"gamma={res['gamma']:.4f} gamma_acc={res['gamma_acc']:.4f} gamma_H={res['gamma_H']:.4f}")
check("identity holds on GS", abs(res['gamma_H']-(res['gamma']-res['gamma_acc']))<1e-10)

print("\n"+"="*60)
print(f"RESULT: {'ALL PASS' if fails==0 else str(fails)+' FAIL'}")
print("="*60)
sys.exit(1 if fails else 0)
