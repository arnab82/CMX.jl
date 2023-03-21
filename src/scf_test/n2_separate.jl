using ClusterMeanField
using RDM
using QCBase
using PyCall
using Printf
using InCoreIntegrals
using LinearAlgebra
using FermiCG
pyscf=pyimport("pyscf")
np=pyimport("numpy")





bst_cmx=[]
cmf=[]
scf=[]
pt2=[]
for r in 1:35
    basis="sto3g"
    atoms = []
    scale=1.55+r*0.025
    push!(atoms,Atom(1,"N", [0.0, 0.0, 0.0]))
    push!(atoms,Atom(2,"N", [scale, 0.0, 0.0]))

    basis = "6-31G"
    basis="sto3g"

    molecule=@sprintf("%6s %24.16f %24.16f %24.16f \n","N", 0.0, 0.0, 0.0)
    molecule=molecule*@sprintf("%6s %24.16f %24.16f %24.16f \n", "N", scale, 0.0, 0.0)
    print(molecule)
    pymol = Molecule(0,1,atoms,basis)

    na=7
    nb=7
    nroots=1
    #specify the clusters and initial fock space electrons
    init_fspace = [(4,3),(3,4)]
    clusters=[(1:5),(6:10)]
    mf = pyscf_do_scf(pymol)
    push!(scf,mf.e_tot)
    nbas = size(mf.mo_coeff)[1]
    ints = pyscf_build_ints(pymol,mf.mo_coeff, zeros(nbas,nbas));
    elec = na + nb
    norb = size(ints.h1,1)
    nuc_energy=mf.energy_nuc()



    # localize orbitals
    C = mf.mo_coeff
    Cl = localize(mf.mo_coeff,"lowdin",mf)
    ClusterMeanField.pyscf_write_molden(pymol,Cl,filename="lowdin_n2.molden")
    S = get_ovlp(mf)
    U =  C' * S * Cl
    println(" Rotate Integrals")
    flush(stdout)
    ints = orbital_rotation(ints,U)
    println(" done.")
    flush(stdout)

    #cmf section

    clusters = [MOCluster(i,collect(clusters[i])) for i = 1:length(clusters)]
    display(clusters)
    rdm1 = zeros(size(ints.h1))
    println(size(rdm1))
    #d1 = RDM1(n_orb(ints))
    e_cmf, U, d1  = ClusterMeanField.cmf_oo_diis(ints, clusters, init_fspace, RDM1(rdm1, rdm1), verbose=0, maxiter_oo=800,maxiter_ci=400, diis_start=3)
    #e_cmf, U, d1  = ClusterMeanField.cmf_oo(ints, clusters, init_fspace, d1,
                                        #max_iter_oo=100, verbose=0, gconv=1e-6, method="bfgs")
    ClusterMeanField.pyscf_write_molden(pymol,C*U,filename="cmf_n2.molden")
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

        #BST CMX 
    println("*************************************************************BST-CMX ENERGY*******************************************************************","\n\n")

        
    ψ = FermiCG.BSTstate(clusters, FockConfig(init_fspace), cluster_bases)
    #ept2 = FermiCG.compute_pt2_energy(ψ, cluster_ops, clustered_ham, thresh_foi=1e-6,tol=1e-6,verbose=1)
    #total_pt2=ept2[1]+ints.h0
    #println("the value of pt2 correction energy value is",total_pt2)
    display(ψ)
    σ = FermiCG.build_compressed_1st_order_state(ψ, cluster_ops, clustered_ham, nbody=4, thresh=1e-6)
    σ = FermiCG.compress(σ, thresh=1e-6)

    #H = FermiCG.nonorth_dot(ψ,σ) 
    H1 = FermiCG.compute_expectation_value(ψ, cluster_ops, clustered_ham)
    H2 = FermiCG.orth_dot(σ,σ)
    H3 = FermiCG.compute_expectation_value(σ, cluster_ops, clustered_ham)
    sigma2 = FermiCG.build_compressed_1st_order_state(σ, cluster_ops, clustered_ham, nbody=4, thresh=1e-6)
    sigma2_compressed = FermiCG.compress(sigma2, thresh=1e-6)
    H4 = FermiCG.orth_dot(sigma2_compressed,sigma2_compressed)
    H5 = FermiCG.compute_expectation_value(sigma2_compressed, cluster_ops, clustered_ham)
        


    I_1=H1[1]
    I_2=H2[1]-I_1*H1[1]
    I_3=H3[1]-I_1*H2[1]-2*I_2*H1[1]
    I_4=H4[1]-I_1*H3[1]-3*I_2*H2[1]-3*I_3*H1[1]
    I_5=H5[1]-I_1*H4[1]-4*I_2*H3[1]-6*I_3*H2[1]-4*I_4*H1[1]
    E_K2=I_1-(I_2*I_2/I_3)*(1+(((I_4*I_2-I_3*I_3)^2)/(I_2*I_2*(I_5*I_3-I_4*I_4))))
    cmx_2=E_K2+ints.h0
    println(cmx_2)
    push!(bst_cmx,cmx_2)
    println("\n\n",r,"\n\n")
    println(scf)
    println(cmf)
    println(bst_cmx)
    println(pt2)
end
