import math
def get_circle_coordinates(center_x, center_y,center_z ,radius, num_points):
    coordinates = []
    for i in range(num_points):
        angle = 2 * math.pi * i / num_points
        x=center_x+0.0
        y =  center_y+ radius * math.cos(angle)
        z=  center_z+ radius * math.sin(angle)
        coordinates.append([x,y,z])
    return coordinates
 
from pyscf import gto,scf
mol = gto.Mole()
geometry=[
["H", (0.0 ,1.3 ,0.75)],
["H", (0.0, 0.75 ,1.3)],
["H", (0.0 ,-0.75 ,1.3)],
["H", (0.0 ,-1.3 ,0.75)],
["H", (0.0 ,-1.3 ,-0.75)],
["H", (0.0 ,-0.75, -1.3)],
["H", (0.0, 0.75 ,-1.3)],
["H", (0.0 ,1.3 ,-0.75)],
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
n_steps = 300  
step_size = .05
scf_energies=[]
geom=[]
with open("traj_ringH8.xyz","w") as f:
    for R in range(n_steps):
        f.write("8")
        f.write("\n\n")
        scale = 1+R*step_size
        tmp = []
        coordinates= get_circle_coordinates(0.0,0.0,0.0,0.8*scale,12)
        print(coordinates) 
        list=[0,1,3,4,6,7,9,10]
        for i in list:
            f.write("H \t %24.16f %24.16f %24.16f \n" %(coordinates[i][0], coordinates[i][1], coordinates[i][2]))
        count=0
        for i in coordinates:
            tmp.append(["H",(i[0], i[1], i[2])])
        del tmp[2]
        del tmp[4]
        del tmp[6]
        del tmp[8]
        
        #print(tmp)
        f.write("\n\n")
        mol.build(
        atom = tmp,
        basis = 'aug-cc-pVDZ',
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