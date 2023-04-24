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
function c_act(orb_no,cluster,coeff)
    C_ordered = zeros(orb_no,0)
    for (ci,c) in enumerate(cluster)
        C_ordered = hcat(C_ordered, coeff[:,c])
    end
    return C_ordered
end

io = open("traj_H20.xyz", "w");
cmf=[]
cmx=[]
pt2=[]
cepa=[]
for ri in 0:30
    println(ri)
    println("\n")
    xyz = @sprintf("%5i\n\n", 3)
    basis = "sto-3g"
    atoms = []
    r0 = 0.5 + 0.04 * ri
    push!(atoms,Atom(1,"O", [0,0,0]))
    push!(atoms,Atom(2,"H", [0, -r0,r0]))
    push!(atoms,Atom(3,"H", [0, r0,r0]))
    println(atoms)
    for a in atoms
        xyz = xyz * @sprintf("%6s %24.16f %24.16f %24.16f \n", a.symbol, a.xyz[1], a.xyz[2], a.xyz[3])
    end
    println(xyz)
    write(io, xyz);
    pymol = Molecule(0,1,atoms,basis)
    na=5
    nb=5
    # get integrals
    mf = ClusterMeanField.pyscf_do_scf(pymol)
    nbas = size(mf.mo_coeff)[1]
    ints = ClusterMeanField.pyscf_build_ints(pymol,mf.mo_coeff, zeros(nbas,nbas));
    nelec = na + nb
    norb = size(ints.h1,1)
    nuc_energy=mf.energy_nuc()
    
    frozen=[1]
    clusters_1=[[2,3,4,5],[6],[7]]
    # localize orbitals
    C = mf.mo_coeff
    Cl = localize(mf.mo_coeff,"lowdin",mf)
    #ClusterMeanField.pyscf_write_molden(pymol,Cl,filename="lowdin_h20_sto3g.molden")
    S = get_ovlp(mf)
    C=mf.mo_coeff
    c_frozen=Cl[:,frozen]
    d_frozen=2*c_frozen*c_frozen'
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
    clusters    = [(1:4),(5:5),(6:6)]
    init_fspace = [(2,2),(1,1),(1,1)]
    println("*************************************************************CMF ENERGY*******************************************************************","\n\n")

    #define clusters
    clusters = [MOCluster(i,collect(clusters[i])) for i = 1:length(clusters)]
    display(clusters)
    rdm1 = zeros(size(ints.h1))
    e_cmf, U, d1  = ClusterMeanField.cmf_oo_diis(ints, clusters, init_fspace, RDM1(rdm1, rdm1),maxiter_oo=800, verbose=0, diis_start=3)
    ClusterMeanField.pyscf_write_molden(pymol,C_act*U,filename="cmf_h20.molden")

    #=ints_1 = FermiCG.orbital_rotation(ints,U)
    e_ref = e_cmf - ints_1.h0
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
    e_cepa, v_cepa = FermiCG.do_fois_cepa(ψ, cluster_ops, clustered_ham, thresh_foi=1e-3, max_iter=50, tol=1e-8)
    display(e_cepa+ints.h0)
    push!(cepa,e_cepa+ints.h0)
    #pt2_correction
    ept2 = FermiCG.compute_pt2_energy(ψ, cluster_ops, clustered_ham, thresh_foi=1e-6,tol=1e-6,verbose=1)
    total_pt2=ept2[1]+ints_1.h0
    println("the value of pt2 correction energy value is",total_pt2)
    push!(pt2,total_pt2)
    display(ψ)
    #=
    #calculating hamiltonians
    σ = FermiCG.build_compressed_1st_order_state(ψ, cluster_ops, clustered_ham, nbody=4, thresh=1e-6)
    σ_compressed = FermiCG.compress(σ, thresh=1e-6)
    H1 = FermiCG.compute_expectation_value(ψ, cluster_ops, clustered_ham)
    H2 = FermiCG.orth_dot(σ_compressed,σ_compressed)
    H3 = FermiCG.compute_expectation_value(σ_compressed, cluster_ops, clustered_ham)
    sigma2 = FermiCG.build_compressed_1st_order_state(σ_compressed, cluster_ops, clustered_ham, nbody=4, thresh=1e-6)
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
    println("\n\n",ri,"\n\n")
    println(cmx)=#
    println(pt2)
    println(cepa)=#
    push!(cmf,e_cmf)
    println(cmf)
end
close(io)