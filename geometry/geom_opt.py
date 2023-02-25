
from pyscf import gto, scf,dft
from pyscf.geomopt.geometric_solver import optimize
geometry=[
["O",(0.216047388826 , 0.123477802503 ,-1.011843124236)],
["H", (0.216047388826 ,0.876420508809,-0.420819333665)],
["H",(0.216047388826 , -0.629464905803,-0.420819336213)],
["O",(0.508028121174 ,1.215443597497,-3.635184444358)],
["H",(0.508028121174 , 1.968386305803,-3.044160656335)],
["H",(0.508028121174 ,0.462500891191,-3.044160653787)]
]

mol = gto.M()
mol.build(
    atom=geometry, 
    basis="aug-cc-pVDZ",unit="angstrom")

# geometric
mfdft= dft.RKS(mol)
mfdft.xc = 'b3lyp'
mol_eq = optimize(mfdft , maxsteps=100)
dft_optgeom  = mol_eq.atom_coords()
print(dft_optgeom)
