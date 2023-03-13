import math
import numpy as np
from pyscf import gto,scf
mol = gto.Mole()
geometry=[
["Cr", (0.0 ,-0.0 ,0.00)],
["Cr", (1.0, -0.0 ,0.00)],
]
print(geometry)
mol.build(
    atom = geometry,
    basis = 'sto3g',
    charge = 0,
    spin = 0,
)

mf = scf.RHF(mol)
scf_energies=[]
mf.kernel()
dm1 = mf.make_rdm1()
mol.build()
scf_energies=[]
geom=[]

basis="sto-3g"
n_steps = 110
step_size = .02
energies_cmf=[]
scf_phi=[]
for R in range(n_steps):
    scale = 1+R*step_size
    tmp=[]
    tmp.append(["Cr",[0.0 ,-0.0 ,0.00]])
    tmp.append(["Cr",[1.0*scale ,-0.0 ,0.00]])
    mol.build(
    atom = tmp,
    basis = 'sto3g',
    charge = 0,
    spin = 0,
    )
    print(tmp)
    geom.append(tmp)
    mf = scf.RHF(mol)
    mf.kernel()
    dm1 = mf.make_rdm1()
    mol.build()
    scf_energies.append(mf.e_tot)
    print("\n",R,"\n")
    print(scf_energies)
    print(np.shape(scf_energies))


