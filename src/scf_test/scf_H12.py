from pyscf import gto,scf
mol = gto.Mole()
geometry=[
["H", (0.0, 0.0, 0.0)],
["H", (0.0, 0.0, 0.5)],
["H", (0.0, 0.5, 1.0)],
["H", (0.0, 0.5, 1.5)],
["H", (0.0, 1.0, 2.0)],
["H", (0.0, 1.0, 2.5)],
["H", (0.0, 1.5, 3.0)],
["H", (0.0, 1.5, 3.5)],
["H", (0.0, 2.0, 4.0)],
["H", (0.0, 2.0, 4.5)],
["H", (0.0, 2.5, 5.0)],
["H", (0.0, 2.5, 5.5)]
]
print(geometry)
mol.build(
    atom = geometry,
    basis = 'aug-cc-pVDZ',
    charge = 0,
    spin = 0,
)

mf = scf.RHF(mol)
scf_energies=[]
mf.kernel()
dm1 = mf.make_rdm1()
mol.build()
scf_energies.append(mf.kernel(dm1))
n_steps = 50   
step_size = .0016
scf_energies=[]
geom=[]
with open("traj_H12_all_bond_stretch.xyz","w") as f:
    for R in range(n_steps):
        f.write("12")
        f.write("\n\n")
        scale = 1+R*step_size
        tmp = []
        count=0
        for a in mol.atom:
            count+=1
            tmp.append(["H",(a[1][0]*scale, a[1][1]*scale, a[1][2]*scale)])
            f.write("H \t %24.16f %24.16f %24.16f \n" %(a[1][0]*scale, a[1][1]*scale, a[1][2]*scale))
        #print(count)
        f.write("\n\n")
        mol.build(
        atom = tmp,
        basis = '6-31g**',
        charge = 0,
        spin = 0,
        )
        
        print(tmp)
        geom.append(tmp)
        mf = scf.RHF(mol)
        mf.kernel()
        dm1 = mf.make_rdm1()
        mol.build()
        scf_energies.append(mf.kernel(dm1))
        print("\n",R,"\n")
        print(scf_energies)

f.close()


    #the scf is converging  from 0.5 to 3.5 angstrom for all bond stretching
    #the scf is converging  from 1 to 10 angstrom for inter-molecular distance stretching with the H-H bond-distance fixed at 0.74 angstrom.