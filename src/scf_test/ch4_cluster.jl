
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
using Plots
using ClusterMeanField
using ActiveSpaceSolvers
using Plots
pyscf=pyimport("pyscf")
function c_act(orb_no,cluster,coeff)
    C_ordered = zeros(orb_no,0)
    for (ci,c) in enumerate(cluster)
        C_ordered = hcat(C_ordered, coeff[:,c])
    end
    return C_ordered
end

cepa=[]
cmf=[]
cmx=[]
pt2=[]
scf=[]


basis = "6-31G"
atoms = []
r0 = -0.02 - 0.06* 1
push!(atoms,Atom(1,"C", [0.63, 0.63, 0.63]))
push!(atoms,Atom(2,"H", [1.26, 1.26, 0]))
push!(atoms,Atom(3,"H", [0, 1.26,1.26]))
push!(atoms,Atom(4,"H", [1.26,0,1.26]))
push!(atoms,Atom(5,"Cl", [r0, r0,r0]))
println(atoms)
pymol = Molecule(0,1,atoms,basis)
na=12
nb=12

# get integrals
mf = pyscf_do_scf(pymol)
nbas = size(mf.mo_coeff)[1]
ints = pyscf_build_ints(pymol,mf.mo_coeff, zeros(nbas,nbas));
nelec = na + nb
norb = size(ints.h1,1)
nuc_energy=mf.energy_nuc()
norb = size(ints.h1,1)


# localize orbitals
C = mf.mo_coeff
Cl = localize(mf.mo_coeff,"boys",mf)
ClusterMeanField.pyscf_write_molden(pymol,Cl,filename="boys_ch3cl_test.molden")
S = get_ovlp(mf)




frozen= [1]
C=Cl
c_frozen=Cl[:,frozen]
d_frozen=2*c_frozen*c_frozen'
clusters_1=[[2,4],[3,13],[5,14],[15,16],[6,7,8,9],[10,11,12,17]]

#get the active ActiveSpace
C_act=c_act(norb,clusters_1,C)
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



clusters    = [(1,2),(3,4),(5,6),(7,8),(9:12),(13:16)]
init_fspace = [(2,2),(0,0),(1,0),(0,1),(1,1),(0,0)]
println("*************************************************************CMF ENERGY*******************************************************************","\n\n")

#define clusters
clusters = [MOCluster(i,collect(clusters[i])) for i = 1:length(clusters)]
display(clusters)
rdm1 = zeros(size(ints.h1))
println(rdm1)
e_cmf, U, d1  = ClusterMeanField.cmf_oo_diis(ints, clusters, init_fspace, RDM1(rdm1,rdm1),maxiter_oo=800, verbose=0, diis_start=3)
println(mf.e_tot)
#ClusterMeanField.pyscf_write_molden(pymol,C_act*U,filename="cmf_ch4_test.molden")
error("jkbj")
for ri in 0:3
    println(ri)
    println("\n")
    xyz = @sprintf("%5i\n\n", 5)
    basis = "6-31G"
    atoms = []
    r0 = 0.0 - 0.06* ri
    push!(atoms,Atom(1,"C", [0.63, 0.63, 0.63]))
    push!(atoms,Atom(2,"H", [1.26, 1.26, 0]))
    push!(atoms,Atom(3,"H", [0, 1.26,1.26]))
    push!(atoms,Atom(4,"H", [1.26,0,1.26]))
    push!(atoms,Atom(5,"H", [r0, r0,r0]))
    println(atoms)
    pymol = Molecule(0,1,atoms,basis)
    na=5
    nb=5

    # get integrals
    mf = pyscf_do_scf(pymol)
    nbas = size(mf.mo_coeff)[1]
    ints = pyscf_build_ints(pymol,mf.mo_coeff, zeros(nbas,nbas));
    nelec = na + nb
    norb = size(ints.h1,1)
    nuc_energy=mf.energy_nuc()
    norb = size(ints.h1,1)





    frozen= [1,2]
    C=mf.mo_coeff
    c_frozen=C[:,frozen]
    d_frozen=2*c_frozen*c_frozen'
    clusters_1=[[3,4,5,6],[7,8,9],[10,11],[12,13],[14,15],[16,17]]

    #get the active ActiveSpace
    C_act=c_act(norb,clusters_1,C)
    #ClusterMeanField.pyscf_write_molden(pymol,Cl,filename="lowdin_ch4_test.molden")
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



    clusters    = [(1,2,3,4),(5,6,7,8),(9,10),(11,12),(13,14),(15,16)]
    clusters    = [(1,2,3,4),(5,6,7),(8,9),(10,11),(12,13),(14,15)]
    init_fspace = [(3,3),(0,0),(0,0),(0,0),(0,0),(0,0)]
    println("*************************************************************CMF ENERGY*******************************************************************","\n\n")

    #define clusters
    clusters = [MOCluster(i,collect(clusters[i])) for i = 1:length(clusters)]
    display(clusters)
    rdm1 = zeros(size(ints.h1))
    e_cmf, U, d1  = ClusterMeanField.cmf_oo_diis(ints, clusters, init_fspace, RDM1(rdm1,rdm1),maxiter_oo=800, verbose=0, diis_start=3)
    println(mf.e_tot)
    ClusterMeanField.pyscf_write_molden(pymol,C_act*U,filename="cmf_ch4_test.molden")
end
