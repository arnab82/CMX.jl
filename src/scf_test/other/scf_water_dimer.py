from pyscf import gto,scf
mol = gto.Mole()
geometry=[
["O",(-0.008448  ,0.352933  ,-0.788731 )],
["H",(0.757920   ,0.319811  ,-0.202104 )],
["H",(-0.353687  ,-0.548825 ,-0.802717 )],
["O",(0.866473   ,1.087332  ,-3.469329 )],
["H",(0.407261   ,1.881339  ,-3.765798 )],
["H",(0.520065   ,0.913206  ,-2.576928 )]
]
print(geometry)
basis="aug-cc-pVDZ"
mol.build(
    atom = geometry,
    basis = basis,
    charge = 0,
    spin = 0,
)

mf = scf.RHF(mol)
mf.kernel()

n_steps = 60  
step_size = .001
scf_energies=[]
geom=[]
with open("traj_H20_dimer.xyz","w") as f:
    for R in range(n_steps):
        f.write("6")
        f.write("\n\n")
        scale = 1+R*step_size
        tmp = []
        count=0
        for a in mol.atom[0:3]:
            count+=1
            tmp.append([a[0][0],(a[1][0], a[1][1], a[1][2])])
            f.write("%6s %24.16f %24.16f %24.16f \n" %(a[0][0],a[1][0], a[1][1], a[1][2]))
        for a in mol.atom[3:6]:
            count+=1
            tmp.append([a[0][0],(a[1][0]*scale, a[1][1]*scale, a[1][2]*scale)])
            f.write("%6s %24.16f %24.16f %24.16f \n" %(a[0][0],a[1][0]*scale, a[1][1]*scale, a[1][2]*scale))
        print(count,"\n")
        f.write("\n")
        mol.build(
        atom = tmp,
        basis = basis,
        charge = 0,
        spin = 0,
        )
        print(tmp)
        
        geom.append(tmp)
        mf = scf.RHF(mol)
        scf_energies.append(mf.kernel())
        print("\n",R,"\n")
        print(scf_energies)
        del(tmp)
    print(geom)
f.close()
