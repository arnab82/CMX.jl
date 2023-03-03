using FermiCG
using QCBase
using ClusterMeanField
using InCoreIntegrals
using RDM

using NPZ
using PyCall
using LinearAlgebra
using Printf
using JLD2

molecule = "
He       0.0000000000000000       0.0000000000000000       0.0000000000000000 
He       2.8065068164350007       0.0000000000000000       0.0000000000000000 
He       0.0000000000000000       2.8065068164350007       0.0000000000000000 
He       2.8065068164350007       2.8065068164350007       0.0000000000000000 
He       1.4032534049100001       1.4032534049100001       1.9845000000000008 
He       1.4032534049100001       1.4032534049100001      -1.9845000000000008 
He       1.4032534049100001       1.4032534049100001       0.0000000000000000 
"
atoms = []
for (li,line) in enumerate(split(rstrip(lstrip(molecule)), "\n"))
    l = split(line)
    push!(atoms, Atom(li, l[1], parse.(Float64,l[2:4])))
end

# pick a basis set
basis = "cc-pvdz" #5 orbs on each He

# Create FermiCG.Molecule type
mol = Molecule(0,1,atoms,basis)

# Run Hartree Fock
pyscf = pyimport("pyscf")
pymol = pyscf.gto.Mole(atom=molecule, spin=0, charge=0, basis=basis)
pymol.build()
mf = pyscf.scf.RHF(pymol).run()
s = mf.get_ovlp(pymol)

# This is lowdin localization to get a set of orthogonal Atomic Orbitals
lo = pyimport("pyscf.lo.orth")
lo_ao = lo.lowdin(s)
println("size of Lowdin ortho AO's:", size(lo_ao))

# This function writes the orbitals to a molden file
# You can now load this file into jmol or other viewing software
FermiCG.pyscf_write_molden(mol, lo_ao, filename="lowdin_ao_ccpvdz.molden")

# write fci dump file from the modified mo coefficients
# this is useful because the FermiCG can read in this dump file and not have to recompute the integrals
tools = pyimport("pyscf.tools")
tools.fcidump.from_mo(pymol, "fcidump.he07_oct", lo_ao)

# Can just read in pyscf dump file for integrals (once you have already run an scf calculation)
pyscf = pyimport("pyscf");
fcidump = pyimport("pyscf.tools.fcidump");
ctx = fcidump.read("fcidump.he07_oct");
h = ctx["H1"];
g = ctx["H2"];
ecore = ctx["ECORE"];
g = pyscf.ao2mo.restore("1", g, size(h,2))

# This creates our integral object that is in InCoreIntegrals julia repo 
# https://github.com/nmayhall-vt/InCoreIntegrals.jl
ints = InCoreInts(ecore,h,g);

# Define clusters and intial Fock space for inital CMF calc for 5 orbs each He
# clusters_list is where you would manually select which orbitals are in which cluster
# for these helium systems it is easy because we are just using atomic orbitals so we
# know that if there are 5 AOs for each helium atom then we know orbitals 1-5 will be in cluster 1
cluster_list = [(1:5),(6:10), (11:15), (16:20), (21:25), (26:30), (31:35)]

# This is the number of alpha and beta electrons per cluster so (1,1) means (1alpha, 1beta)
init_fspace = [(1,1),(1,1),(1,1),(1,1),(1,1),(1,1),(1,1)]
rdm1 = zeros(size(ints.h1))

# have to define total alpha and beta electrons
na=7
nb=7

# Define clusters now using FermiCG code
clusters = [MOCluster(i,collect(cluster_list[i])) for i = 1:length(cluster_list)]
display(clusters)

e_cmf, U_cmf, d1  = ClusterMeanField.cmf_oo_diis(ints, clusters, init_fspace, RDM1(rdm1, rdm1), verbose=0);

# rotate the integrals by the cmf calculation
ints = orbital_rotation(ints,U_cmf)

# rotate orbitlas by the cmf calculation
C_cmf = lo_ao*U_cmf

# can write these to a molden file to visualize cmf orbitals
FermiCG.pyscf_write_molden(mol, C_cmf, filename="cmf.molden")

# save the cmf data using JLD2
@save  "cmf_diis.jld2" ints d1 clusters init_fspace C_cmf
