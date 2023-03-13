import math
import numpy as np
from pyscf import gto,scf
mol = gto.Mole()
geometry=[
["H", (0.0 ,-2.2 ,3.81)],
["H", (0.0, -4.345 ,0.69)],
["H", (0.0 ,-2.2 ,-3.81)],
["H", (0.0 ,1.573 ,-4.11)],
["H", (0.0 ,4.4 ,0.0)],
["H", (0.0 ,2.77, 3.417)]
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
   
def get_circle_coordinates(center_x, center_y,center_z ,radius, num_points,R,scale1,scale):
    coordinates= []
    for i in range(1,num_points+1):
        angle = 2 * math.pi * i / num_points
        x=center_x+0.0
        y=  center_y+ radius * math.cos(angle)
        z=  center_z+ radius * math.sin(angle)
        coordinates.append([x,y,z])
    
    if R<2:
        for i in range(1,num_points+1):
            angle = 2 * math.pi * i / num_points+(scale)
            x=center_x+0.0
            y=  center_y+ radius * math.cos(angle)
            z=  center_z+ radius * math.sin(angle)
            coordinates.append([x,y,z])
    else:
        for i in range(1,num_points+1):
            angle = 2 * math.pi * i / num_points+(scale1)
            x=center_x+0.0
            y=  center_y+ radius * math.cos(angle)
            z=  center_z+ radius * math.sin(angle)
            coordinates.append([x,y,z])
    return coordinates


basis="sto-3g"
n_steps = 70
scf_energies=[]


angle_num=70
for r in range(1,angle_num+1):
    scale1=math.pi/24+(r*math.pi/250)
    c= get_circle_coordinates(0.0,0.0,0.0,2.2,3,r,scale1,math.pi/24)
        #print(c) 
    tmp=[]
    tmp.append(["H",[c[0][0], c[0][1], c[0][2]]])
    tmp.append(["H",[c[3][0], c[3][1], c[3][2]]])
    tmp.append(["H",[c[1][0], c[1][1], c[1][2]]])
    tmp.append(["H",[c[4][0], c[4][1], c[4][2]]])
    tmp.append(["H",[c[2][0], c[2][1], c[2][2]]])
    tmp.append(["H",[c[5][0], c[5][1], c[5][2]]])
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
    print("\n",r,"\n")
    print(scf_energies)
    print(np.shape(scf_energies))


