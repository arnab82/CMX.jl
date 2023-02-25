import pyscf
import pyscf.tools
molecule = """
 Fe 1.67785607 0.00052233 0.06475932
 Fe -1.67785607 -0.00052233 0.06475932
 O 0.00000000 0.00000000 -0.47099074
 Cl 1.87002704 -1.09796437 1.99091682
 Cl 2.93244917 -0.98210488 -1.47467288
 Cl 2.37160936 2.07954091 -0.50446591
 Cl -1.87002704 1.09796437 1.99091682
 Cl -2.93244917 0.98210488 -1.47467288
 Cl -2.37160936 -2.07954091 -0.50446591
"""
basis = "def2-svp"
pymol = pyscf.gto.Mole(
        atom    =   molecule,
        symmetry=   True,
        spin    =   10, # number of unpaired electrons
        charge  =   -2,
        basis   =   basis)


pymol.build()
print("symmetry: ",pymol.topgroup)
# mf = pyscf.scf.UHF(pymol).x2c()
mf = pyscf.scf.ROHF(pymol)
mf.verbose = 4
mf.conv_tol = 1e-8
mf.conv_tol_grad = 1e-5
mf.chkfile = "scf.fchk"
mf.init_guess = "sad"
mf.run(max_cycle=200)

print(" Hartree-Fock Energy: %12.8f" % mf.e_tot)
# mf.analyze()
import numpy as np
import scipy
import copy as cp
import math
F = mf.get_fock()
def sym_ortho(frags, S, thresh=1e-8):
    Nbas = S.shape[1]
    
    inds = []
    Cnonorth = np.hstack(frags)
    shift = 0
    for f in frags:
        inds.append(list(range(shift, shift+f.shape[1])))
        shift += f.shape[1]
        
    
    Smo = Cnonorth.T @ S @ Cnonorth
    X = np.linalg.inv(scipy.linalg.sqrtm(Smo))
    # print(Cnonorth.shape, X.shape)
    Corth = Cnonorth @ X
    
    frags2 = []
    for f in inds:
        frags2.append(Corth[:,f])
    return frags2

def get_spade_orbitals(orb_list, C, S, thresh=1e-6):
    """
    Find the columns in C that span the rows specified by orb_list
    """
    # First orthogonalize C
    X = scipy.linalg.sqrtm(S)
    Corth = X @ C
    
    U,s,V = np.linalg.svd(Corth[orb_list,:])
    nkeep = 0
    for idx,si in enumerate(s):
        if si > thresh:
            nkeep += 1
        print(" Singular value: ", si)
    print(" # of orbitals to keep: ", nkeep)
    
    Xinv = scipy.linalg.inv(X)
    
    Csys = Xinv @ Corth @ V[0:nkeep,:].T
    Cenv = Xinv @ Corth @ V[nkeep::,:].T
    return Csys, Cenv

def semi_canonicalize(C,F):
    e,V = np.linalg.eigh(C.T @ F @ C)
    for ei in e:
        print(" Orbital Energy: ", ei)
    return C @ V
import scipy

# collect 2s,2p from O in Occ add to singly occupied orbitals
docc_list = []
socc_list = []
virt_list = []
oxygen_list = []
for idx,i in enumerate(mf.mo_occ):
    if i == 0:
        virt_list.append(idx)
    elif i == 1:
        socc_list.append(idx)
    elif i == 2:
        docc_list.append(idx)
        
for ao_idx,ao in enumerate(mf.mol.ao_labels(fmt=False)):
    if ao[0] == 2:
        if ao[2] in ("2s", "2p"):
            oxygen_list.append(ao_idx)

C = mf.mo_coeff
S = mf.get_ovlp()
P = mf.make_rdm1()
Pa = P[0,:,:]
Pb = P[1,:,:]

Cdocc = C[:,docc_list]
Csocc = C[:,socc_list]
Cvirt = C[:,virt_list]


# Cact = np.hstack((Cfrag, Csocc))
Cact = cp.deepcopy(Csocc)
Cenv = cp.deepcopy(Cdocc)
na_act = np.trace(Cact.T @ S @ Pa @ S @ Cact)
na_env = np.trace(Cenv.T @ S @ Pa @ S @ Cenv)
na_vir = np.trace(Cvirt.T @ S @ Pa @ S @ Cvirt)
nb_act = np.trace(Cact.T @ S @ Pb @ S @ Cact)
nb_env = np.trace(Cenv.T @ S @ Pb @ S @ Cenv)
nb_vir = np.trace(Cvirt.T @ S @ Pb @ S @ Cvirt)
print(" # electrons: %12s %12s" %("α", "β"))
print("         Env: %12.8f %12.8f" %(na_env, nb_env))
print("         Act: %12.8f %12.8f" %(na_act, nb_act))
print("         Vir: %12.8f %12.8f" %(na_vir, nb_vir))

frag1 = []
frag2 = []

for ao_idx,ao in enumerate(mf.mol.ao_labels(fmt=False)):
    if ao[0] == 0:
        frag1.append(ao_idx)
    elif ao[0] == 1:
        frag2.append(ao_idx)    


C1, C1env = get_spade_orbitals(frag1, Cact, S, thresh=.5)
C2, C2env = get_spade_orbitals(frag2, Cact, S, thresh=.5)
C1, C2 = sym_ortho([C1, C2], S)
C1 = semi_canonicalize(C1, F)
C2 = semi_canonicalize(C2, F)

Cact = np.hstack((C1, C2))
pyscf.tools.molden.from_mo(mf.mol, "Cact.molden", Cact);
d1_embed = 2 * Cenv @ Cenv.T

h0 = pyscf.gto.mole.energy_nuc(mf.mol)
h  = pyscf.scf.hf.get_hcore(mf.mol)
j, k = pyscf.scf.hf.get_jk(mf.mol, d1_embed, hermi=1)
h0 += np.trace(d1_embed @ ( h + .5*j - .25*k))

h = Cact.T @ h @ Cact;
j = Cact.T @ j @ Cact;
k = Cact.T @ k @ Cact;
nact = h.shape[0]

h2 = pyscf.ao2mo.kernel(pymol, Cact, aosym="s4", compact=False)
h2.shape = (nact, nact, nact, nact)
# The use of d1_embed only really makes sense if it has zero electrons in the
# active space. Let's warn the user if that's not true

S = pymol.intor("int1e_ovlp_sph")
n_act = np.trace(S @ d1_embed @ S @ Cact @ Cact.T)
if abs(n_act) > 1e-8 == False:
    print(n_act)
    error(" I found embedded electrons in the active space?!")

h1 = h + j - .5*k;
np.save("ints_h0", h0)
np.save("ints_h1", h1)
np.save("ints_h2", h2)
np.save("mo_coeffs", Cact)
np.save("overlap_mat", S)
Cold = mf.mo_coeff
U = Cold.T @ S @ Cact
Pa = mf.make_rdm1()[0]
Pb = mf.make_rdm1()[1]

Pa = Cact.T @ S @ Pa @ S @ Cact
Pb = Cact.T @ S @ Pb @ S @ Cact
print(np.trace(Pa), np.trace(Pb))
np.save("Pa", Pa)
np.save("Pb", Pb)
np.trace(Cact @ Cact.T @ S @ F) 
Ccmf = np.load("Ccmf.npy")
pyscf.tools.molden.from_mo(mf.mol, "Ccmf.molden", Ccmf);
