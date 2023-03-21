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
basis = "sto3g"
#basis = "6-31G"
atoms = []
r0 = 0.8 
push!(atoms,Atom(1,"C", [0.63, 0.63, 0.63]))
push!(atoms,Atom(2,"H", [1.26, 1.26, 0]))
push!(atoms,Atom(3,"H", [0, 1.26,1.26]))
push!(atoms,Atom(4,"H", [1.26,0,1.26]))
push!(atoms,Atom(5,"H", [-2.2,-2.2,-2.2]))
println(atoms)

pymol = Molecule(0,1,atoms,basis)
na=4
nb=4
# get integrals
mf = pyscf_do_scf(pymol)
nbas = size(mf.mo_coeff)[1]
ints = pyscf_build_ints(pymol,mf.mo_coeff, zeros(nbas,nbas));
C=mf.mo_coeff
nelec = na + nb
norb = size(ints.h1,1)
nuc_energy=mf.energy_nuc()
frozen= [1]  
c_frozen=C[:,frozen]
d_frozen=2*c_frozen*c_frozen'
clusters=[[2,3,4,6,7,8],[5,9]]
#get the active ActiveSpace 
function c_act()
    C_ordered = zeros(norb,0)
    for (ci,c) in enumerate(clusters)
        C_ordered = hcat(C_ordered, C[:,c])
    end
    return C_ordered
end
C_act=c_act()
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
clusters    = [(1:6),(7:8)]
init_fspace = [(3,3),(1,1)]
#clusters    = [(1:1),(2,3,4,5,6,7,8,9)]
#init_fspace = [(1,1),(4,4)]
#clusters    = [(1:5),(6:9),(10:11),(12,13),(14,15),(16:17)]
#init_fspace = [(1,1),(0,0),(1,1),(1,1),(1,1),(1,1)]
nroots =    1


println("*************************************************************CMF ENERGY*******************************************************************","\n\n")

#define clusters
clusters = [MOCluster(i,collect(clusters[i])) for i = 1:length(clusters)]
display(clusters)
rdm1 = zeros(size(ints.h1))
#d1 = RDM1(n_orb(ints))
e_cmf, U, d1  = ClusterMeanField.cmf_oo_diis(ints, clusters, init_fspace, RDM1(rdm1, rdm1),maxiter_oo=400, verbose=0, diis_start=3)
#e_cmf, U, d1  = FermiCG.cmf_oo(ints, clusters, init_fspace, d1,
                                    #max_iter_oo=40, verbose=0, gconv=1e-6, method="bfgs")
ClusterMeanField.pyscf_write_molden(pymol,C_act*U,filename="cmf_ch4.molden")
println(e_cmf)

ints = FermiCG.orbital_rotation(ints,U)
e_ref = e_cmf - ints.h0
max_roots = 100
cluster_bases = FermiCG.compute_cluster_eigenbasis(ints, clusters, verbose=0, max_roots=max_roots,
                                                            init_fspace=init_fspace, rdm1a=d1.a, rdm1b=d1.b, T=Float64)
clustered_ham = FermiCG.extract_ClusteredTerms(ints, clusters)
cluster_ops = FermiCG.compute_cluster_ops(cluster_bases, ints);
FermiCG.add_cmf_operators!(cluster_ops, cluster_bases, ints, d1.a, d1.b);
ref_fock = FermiCG.FockConfig(init_fspace)

#forming bst wavefunction
v = FermiCG.BSTstate(clusters, FockConfig(init_fspace), cluster_bases)

e_cepa, v_cepa = FermiCG.do_fois_cepa(v, cluster_ops, clustered_ham, thresh_foi=1e-3, max_iter=50, tol=1e-8)
display(e_cepa+ints.h0)