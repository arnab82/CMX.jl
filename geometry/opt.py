from pyscf import gto, scf
from pyscf.geomopt.berny_solver import optimize
​
# Define the molecular structure
mol = gto.Mole()
mol.atom = '''
O     0.114327724158      0.065341589321     -0.818731560566
H     0.114327724150      0.824678589321     -0.222688560565
H     0.114327724150     -0.693995410679     -0.222688560568
O     1.433026396884      0.643185102836     -2.206942886136
H     1.433026396876      1.402522102835     -1.610899886135
H     1.433026396876     -0.116151897165     -1.610899886138
'''
mol.basis = '6-31g'
mol.charge = 0
mol.spin = 0
mol.build()
​
# Run the SCF calculation
mf = scf.RHF(mol)
mf.kernel()
​
# Optimize the molecular geometry
opt = optimize(mf)
opt.kernel()
​
# Get the optimized molecular structure
print("Optimized geometry:")
print(mol.atom_coords())
