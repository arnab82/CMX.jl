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
basis = "6-31G"
#basis="sto3g"
atoms = []
r0 = -0.0
push!(atoms,Atom(1,"C", [0.63, 0.63, 0.63]))
push!(atoms,Atom(2,"H", [1.26, 1.26, 0]))
push!(atoms,Atom(3,"H", [0, 1.26,1.26]))
push!(atoms,Atom(4,"H", [1.26,0,1.26]))
push!(atoms,Atom(5,"H", [r0, r0,r0]))
println(atoms)
#=
pymol = Molecule(0,1,atoms,basis)
tot_na=5
tot_nb=5

mf = pyscf_do_scf(pymol)
tot_nelec = tot_na + tot_nb
C = mf.mo_coeff
norb=size(C)[1]
ClusterMeanField.pyscf_write_molden(pymol,C,filename="hf_ch4.molden")
S = get_ovlp(mf)
nuc_energy=mf.energy_nuc()


Cl= localize(C,"boys",mf)
h,j,k=ClusterMeanField.pyscf_get_jk(pymol,C[:,1:tot_na]*C[:,1:tot_na]')
#Cl=FermiCG.fiedler_sort(Cl,k)
#ClusterMeanField.pyscf_write_molden(pymol,Cl,filename="boys_ch4.molden")
u,s,v=svd(Cl)
println(s)

#Get Active Space
frozen= [1]  
c_frozen=Cl[:,frozen]
d_frozen=2*c_frozen*c_frozen'
#cluster=[[13,4],[15,3],[16,14],[2,5],[6,7,8,9],[10,11,12,17]]
cluster=[[2,5],[3,7],[4,8],[6,9]]
cluster=[[3,7,4,8,6,9],[2,5]]
cluster=[[13,4,15,3,16,14],[2,5],[6,7,8,9],[10,11,12,17]]
function c_act()
    C_ordered = zeros(norb,0)
    for (ci,c) in enumerate(cluster)
        C_ordered = hcat(C_ordered, Cl[:,c])
    end
    return C_ordered
end
C_act=c_act()
#ClusterMeanField.pyscf_write_molden(pymol,C_act,filename="ordered_ch4.molden")
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




#clusters    = [(1,2,3,4),(5,6,7),(8,9),(10,11),(12,13),(14,15)]
#init_fspace = [(1,0),(0,1),(1,0),(0,1),(0,1),(0,1)]# cmf converges but energy is less than scf energy 
#clusters    = [(1,2),(3,4),(5,6),(7,8)]
#init_fspace = [(1,1),(1,1),(1,1),(1,1)]
clusters=[(1:6),(7,8)]
init_fspace=[(3,4),(1,0)]
#clusters    = [(1,2),(3,4),(5,6),(7,8),(9:12),(13:16)]
#init_fspace = [(1,2),(1,1),(1,1),(1,0),(0,0),(0,0)]
clusters    = [(1,2,3,4,5,6),(7,8),(9:12),(13:16)]
init_fspace = [(3,4),(1,0),(0,0),(0,0)]
println("*************************************************************CMF ENERGY*******************************************************************","\n\n")

#define clusters
clusters = [MOCluster(i,collect(clusters[i])) for i = 1:length(clusters)]
display(clusters)
rdm1 = zeros(size(ints.h1))
e_cmf, U, d1  = ClusterMeanField.cmf_oo_diis(ints, clusters, init_fspace, RDM1(rdm1, rdm1),maxiter_oo=800, verbose=0, diis_start=3)
println(e_cmf)
println(mf.e_tot)
#ClusterMeanField.pyscf_write_molden(pymol,C_act*U,filename="cmf_ch4.molden")

=#

fci=[]
cmf=[]
cmx=[]
pt2=[]
scf=[]
for ri in 0:30
    println(ri)
    println("\n")
    xyz = @sprintf("%5i\n\n", 5)
    basis = "6-31G"
    atoms = []
    r0 = 0.3- 0.03 * ri
    push!(atoms,Atom(1,"C", [0.63, 0.63, 0.63]))
    push!(atoms,Atom(2,"H", [1.26, 1.26, 0]))
    push!(atoms,Atom(3,"H", [0, 1.26,1.26]))
    push!(atoms,Atom(4,"H", [1.26,0,1.26]))
    push!(atoms,Atom(5,"H", [r0, r0,r0]))
    println(atoms)
    pymol = Molecule(0,1,atoms,basis)
    tot_na=5
    tot_nb=5

    mf = pyscf_do_scf(pymol)
    tot_nelec = tot_na + tot_nb
    C = mf.mo_coeff
    norb=size(C)[1]
    #ClusterMeanField.pyscf_write_molden(pymol,C,filename="hf_ch4.molden")
    S = get_ovlp(mf)
    nuc_energy=mf.energy_nuc()




    #=Cl= localize(C,"boys",mf)
    h,j,k=ClusterMeanField.pyscf_get_jk(pymol,C[:,1:tot_na]*C[:,1:tot_na]')
    #Cl=FermiCG.fiedler_sort(Cl,k)
    ClusterMeanField.pyscf_write_molden(pymol,Cl,filename="boys_ch4_-0.0.molden")
    u,s,v=svd(Cl)=#
    #println(s)
    

    #Get Active Space
    frozen= [1]  
    c_frozen=Cl[:,frozen]
    d_frozen=2*c_frozen*c_frozen'
    cluster=[[2,4],[3,13],[5,14],[15,16],[6,7,8,9],[10,11,12,17]]
    function c_act()
        C_ordered = zeros(norb,0)
        for (ci,c) in enumerate(cluster)
            C_ordered = hcat(C_ordered, Cl[:,c])
        end
        return C_ordered
    end
    C_act=c_act()
    #ClusterMeanField.pyscf_write_molden(pymol,C_act,filename="ordered_ch4_2.molden")
    ints=ClusterMeanField.pyscf_build_ints(pymol,C_act,d_frozen)
    #=mol_1=make_pyscf_mole(pymol)
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
    ints = InCoreInts(h0, h1, h2)=#
    active=[2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17]
    Cl_act=Cl[:,active]
    S = ClusterMeanField.get_ovlp(mf)
    U =  C_act' * S * Cl_act
    println(" Rotate Integrals")
    flush(stdout)
    ints = FermiCG.orbital_rotation(ints,U)
    println(" done.")
    flush(stdout)



    clusters    = [(1,2),(3,4),(5,6),(7,8),(9:12),(13:16)]
    init_fspace = [(2,2),(2,2),(0,0),(0,0),(0,0),(0,0)]

    println("*************************************************************CMF ENERGY*******************************************************************","\n\n")

    #define clusters
    clusters = [MOCluster(i,collect(clusters[i])) for i = 1:length(clusters)]
    display(clusters)
    rdm1 = zeros(size(ints.h1))
    e_cmf, U, d1  = ClusterMeanField.cmf_oo_diis(ints, clusters, init_fspace, RDM1(rdm1, rdm1),maxiter_oo=1200, verbose=0, diis_start=3)
    println(e_cmf)
    println(mf.e_tot)
    #=
    ints_1 = FermiCG.orbital_rotation(ints,U)
    e_ref = e_cmf - ints.h0
    max_roots = 100
    cluster_bases = FermiCG.compute_cluster_eigenbasis(ints_1, clusters, verbose=0, max_roots=max_roots,
                                                            init_fspace=init_fspace, rdm1a=d1.a, rdm1b=d1.b, T=Float64)
    clustered_ham = FermiCG.extract_ClusteredTerms(ints_1, clusters)
    cluster_ops = FermiCG.compute_cluster_ops(cluster_bases, ints_1);
    FermiCG.add_cmf_operators!(cluster_ops, cluster_bases, ints_1, d1.a, d1.b);
    ref_fock = FermiCG.FockConfig(init_fspace)



    #BST CMX 
    println("*************************************************************BST-CMX ENERGY*******************************************************************","\n\n")  
    #forming bst wavefunction
    ψ = FermiCG.BSTstate(clusters, FockConfig(init_fspace), cluster_bases)
    #pt2_correction 
    ept2 = FermiCG.compute_pt2_energy(ψ, cluster_ops, clustered_ham, thresh_foi=1e-6,tol=1e-6,verbose=1)
    total_pt2=ept2[1]+ints_1.h0
    println("the value of pt2 correction energy value is",total_pt2)
    push!(pt2,total_pt2)
    display(ψ)
    
    #calculating hamiltonians
    σ = FermiCG.build_compressed_1st_order_state(ψ, cluster_ops, clustered_ham, nbody=4, thresh=1e-6)
    σ = FermiCG.compress(σ, thresh=1e-6)
    H1 = FermiCG.compute_expectation_value(ψ, cluster_ops, clustered_ham)
    H2 = FermiCG.orth_dot(σ,σ)
    H3 = FermiCG.compute_expectation_value(σ, cluster_ops, clustered_ham)
    sigma2 = FermiCG.build_compressed_1st_order_state(σ, cluster_ops, clustered_ham, nbody=4, thresh=1e-6)
    sigma2_compressed = FermiCG.compress(sigma2, thresh=1e-6)
    H4 = FermiCG.orth_dot(sigma2_compressed,sigma2_compressed)
    H5 = FermiCG.compute_expectation_value(sigma2_compressed, cluster_ops, clustered_ham)
    #calculating moments and cmx energy
    I_1=H1[1]
    I_2=H2[1]-I_1*H1[1]
    I_3=H3[1]-I_1*H2[1]-2*I_2*H1[1]
    I_4=H4[1]-I_1*H3[1]-3*I_2*H2[1]-3*I_3*H1[1]
    I_5=H5[1]-I_1*H4[1]-4*I_2*H3[1]-6*I_3*H2[1]-4*I_4*H1[1]
    E_K2=I_1-(I_2*I_2/I_3)*(1+(((I_4*I_2-I_3*I_3)^2)/(I_2*I_2*(I_5*I_3-I_4*I_4))))
    cmx_2=E_K2+ints_1.h0
    println(cmx_2)
    push!(cmx,cmx_2)
    println(cmx)
    println(pt2)
    println(e_cmf)=#
    push!(cmf,e_cmf)
    println(cmf)
    #push!(scf,mf.e_tot)
    #println(scf)
end
plot(cmf)
