import numpy as np
import sys
import h5py
from pyscf import gto, scf , mp , dft
from pyscf.geomopt.geometric_solver import optimize

inFile = sys.argv[1]
with open(inFile,'r') as i:
    content = i.readlines()
input_file =[]
for line in content:
    v_line=line.strip()
    if len(v_line)>0:
        input_file.append(v_line.split())
geom_file = input_file

geom = ''
for i in range(len(geom_file)):
    if i==len(geom_file)-1:
        geom += geom_file[i][0]+" "+geom_file[i][1]+" "+geom_file[i][2]+" "+geom_file[i][3]
    else:
        geom += geom_file[i][0]+" "+geom_file[i][1]+" "+geom_file[i][2]+" "+geom_file[i][3]+";"



basis_sets =  ["sto3g","321g","321g*","631g","631g*"]

with h5py.File(f"{inFile}_results.h5" , "w") as f:
    for basis in basis_sets:

        mol = gto.M(atom=geom, basis=basis)
        mf = scf.RHF(mol)
        mol_eq = optimize(mf, maxsteps=100)
        hf_optgeom = mol_eq.atom_coords()
        f.create_dataset(f"hf_{basis}" , data = 0.52917721067121*hf_optgeom)
        print(f"****************************** SCF opt done for {inFile} and {basis} ******************************")


        mf.kernel()
        mfmp2 =  mp.MP2(mf)
        mol_eq = optimize(mfmp2, maxsteps=100)
        mp2_optgeom = mol_eq.atom_coords()
        f.create_dataset(f"mp2_{basis}" , data = 0.52917721067121*mp2_optgeom)
        print(f"****************************** MP2 opt done for {inFile} and {basis} ******************************")


        mfdft= dft.RKS(mol)
        mfdft.xc = 'b3lyp'
        mol_eq = optimize(mfdft , maxsteps=100)
        dft_optgeom  = mol_eq.atom_coords()
        f.create_dataset(f"dft_{basis}" , data = 0.52917721067121*dft_optgeom)
        print(f"****************************** DFT opt done for {inFile} and {basis} ******************************")







