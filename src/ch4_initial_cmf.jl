using QCBase
using RDM
using FermiCG
using Printf
using Test
using LinearAlgebra
using Profile
using Random
using PyCall
using Arpack
using JLD2
using Plots
using ClusterMeanField
using ActiveSpaceSolvers
using Plots
pyscf=pyimport("pyscf")
scf=pyimport("pyscf.scf")
function c_act(orb_no,cluster,coeff)
    C_ordered = zeros(orb_no,0)
    for (ci,c) in enumerate(cluster)
        C_ordered = hcat(C_ordered, coeff[:,c])
    end
    return C_ordered
end
basis = "sto-3g"
basis="6-31G"
atoms = []
r0 = 1.26 + 0.05 * 20
push!(atoms,Atom(1,"C", [r0/2,r0/2,r0/2]))
push!(atoms,Atom(2,"H", [r0,r0, 0]))
push!(atoms,Atom(3,"H", [0, r0,r0]))
push!(atoms,Atom(4,"H", [r0,0,r0]))
push!(atoms,Atom(5,"H", [0,0,0]))
println(atoms)
molecule="
    C  1.13 1.13 1.13
    H  2.26 2.26  0.0
    H   0.0 2.26 2.26
    H  2.26 0.0 2.26
    H  0   0   0
"
mol = pyscf.gto.Mole(atom=molecule,
    symmetry = true,spin =0,charge=0,
    basis = basis)
pymol = Molecule(0,1,atoms,basis)
display(pymol)
QCBase.write_xyz(pymol)
error("kkf")
na=5
nb=5
mf_ = scf.RHF(mol)
mf_.kernel()
# get integrals
mf = pyscf_do_scf(pymol)

nbas = size(mf.mo_coeff)[1]
ints = pyscf_build_ints(pymol,mf.mo_coeff, zeros(nbas,nbas));
nelec = na + nb
norb = size(ints.h1,1)
nuc_energy=mf.energy_nuc()
# localize orbitals
S = get_ovlp(mf)
C=mf.mo_coeff
Cl = localize(C,"boys",mf)
ClusterMeanField.pyscf_write_molden(pymol,Cl,filename="boys_ch4_631g_active.molden")

frozen= [1]
c_frozen=Cl[:,frozen]
d_frozen=2*c_frozen*c_frozen'

clusters_1=[[2,5,10,15],[3,8,11,14],[4,7,13,17],[6,9,12,16]]
#clusters_1=[[3,9],[4,7],[5,6],[8,15],[10,11,12,13],[2,14,16,17]]
#get the active ActiveSpace

C_act=c_act(norb,clusters_1,Cl)
mol_1=make_pyscf_mole(pymol)
h0 = pyscf.gto.mole.energy_nuc(mol_1)
nuc_energy= pyscf.gto.mole.energy_nuc(mol_1)
h  = pyscf.scf.hf.get_hcore(mol_1)
j, k = pyscf.scf.hf.get_jk(mol_1, d_frozen, hermi=1)
h0 += tr(d_frozen * ( h + .5*j - .25*k))
# now rotate to MO basis
h = C_act' * h * C_act
j = C_act' * j * C_act
k = C_act' * k * C_act
s= mol_1.intor("int1e_ovlp_sph")
n_act = size(C_act)[2]
h2 = pyscf.ao2mo.kernel(mol_1, C_act, aosym="s4",compact=false)
h2 = reshape(h2, (n_act, n_act, n_act, n_act))
nact=tr(s*d_frozen*s*C_act*C_act')
println(nact)
h1 = h + j - .5*k
ints = InCoreInts(h0, h1, h2)
#clusters_1=[(1,5),(3,6),(2,8),(4,7)]
clusters    = [(1:4),(5:8),(9:12),(13:16)]
init_fspace = [(1,1),(1,1),(1,1),(1,1)]
println("*************************************************************CMF ENERGY*******************************************************************","\n\n")

#define clusters
clusters = [MOCluster(i,collect(clusters[i])) for i = 1:length(clusters)]
display(clusters)
rdm1 = zeros(size(ints.h1))
e_cmf, U, d1  = ClusterMeanField.cmf_oo_diis(ints, clusters, init_fspace, RDM1(rdm1, rdm1),maxiter_oo=800, verbose=0, diis_start=3)
#ClusterMeanField.pyscf_write_molden(pymol,C_act*U,filename="cmf_ch4.molden")

ints_1 = FermiCG.orbital_rotation(ints,U)
@save "_testdata_cmf_ch4_631g.jld2" ints_1 d1 e_cmf clusters init_fspace C_act d_frozen