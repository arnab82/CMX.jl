import pyscf
import numpy as np
from pyscf import gto, scf, ao2mo, tools, lo
molden=tools.molden
pyscf.lib.num_threads(1)  #with degenerate states and multiple processors there can be issues
#PYSCF inputs
print(" ---------------------------------------------------------")
print("                      Using Pyscf:")
print(" ---------------------------------------------------------")
print("                                                          ")
charge=0
spin=0
basis_set='6-31G'
loc_nstop=17
loc_nstart=0
n_a=5
molecule = '''
    C   0.63  0.63  0.63 
    H   1.26  1.26   0
    H    0   1.26  1.26
    H   1.26   0   1.26
    H   -0.2  -0.2  -0.2
    '''
mol = gto.Mole()
mol.atom = molecule

mol.max_memory = 1000 # MB
mol.symmetry = True
mol.charge = charge
mol.spin = spin
mol.basis = basis_set
mol.build()
loc_vstop =  loc_nstop - n_a
print(loc_vstop)
mf = scf.RHF(mol).run(conv_tol=1e-8)
mo_occ = mf.mo_coeff[:,mf.mo_occ>0]
mo_vir = mf.mo_coeff[:,mf.mo_occ==0]
print(np.shape(mo_occ))
print(np.shape(mo_vir))
c_core = mo_occ[:,:loc_nstart]
print(np.shape(c_core))
iao_occ = lo.iao.iao(mol, mo_occ[:,loc_nstart:])
print(np.shape(iao_occ))
iao_vir = lo.iao.iao(mol, mo_vir[:,:loc_vstop])
print(np.shape(iao_vir))
c_out  = mo_vir[:,loc_vstop:]
print(np.shape(c_out))

# Orthogonalize IAO
iao_occ = lo.vec_lowdin(iao_occ, mf.get_ovlp())
iao_vir = lo.vec_lowdin(iao_vir, mf.get_ovlp())

            #
            # Method 1, using Knizia's alogrithm to localize IAO orbitals
            #
'''
            Generate IBOS from orthogonal IAOs
'''
ibo_occ = lo.ibo.ibo(mol, mo_occ[:,loc_nstart:], iaos = iao_occ)
ibo_vir = lo.ibo.ibo(mol, mo_vir[:,:loc_vstop], iaos = iao_vir)

C = np.column_stack((c_core,ibo_occ,ibo_vir,c_out))
molden.from_mo(mol, 'ibmo_ch4.molden', C)
