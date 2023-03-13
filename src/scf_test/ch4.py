import math
import numpy as np
from pyscf import gto,scf
mol = gto.Mole()
scf_energy=[]
with open("traj_ch4.xyz","w") as f:
    for ri in range(0,88):
        f.write("5")
        f.write("\n")
    ###     PYSCF INPUT
        r0 = 0.6 + 0.03 * ri
        molecule = """
        C   {1}  {1}  {1} 
        H   {0}  {0}   0
        H    0   {0}  {0}
        H   {0}   0   {0}
        H    0    0    0
        """.format(r0,r0/2)
        print(molecule)
        f.write(molecule)
        f.write("\n")
        mol.build(
        atom = molecule,
        basis = 'sto3g',
        charge = 0,
        spin = 0,
        )

        mf = scf.RHF(mol)
        scf_energies=[]
        mf.kernel()
        dm1 = mf.make_rdm1()
        mol.build()
        scf_energy.append(mf.e_tot)
        print(scf_energy)
f.close()