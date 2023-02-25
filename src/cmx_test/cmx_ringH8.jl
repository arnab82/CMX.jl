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


function get_circle_coordinates(center_x, center_y,center_z ,radius, num_points)
    coordinates= []

    for i in 1:num_points
        angle = 2 * π * i / num_points
        x=center_x+0.0
        y=  center_y+ radius * cos(angle)
        z=  center_z+ radius * sin(angle)
        push!(coordinates,[x,y,z])
    end
    return coordinates
    
end
basis="sto-3g"
n_steps = 100
step_size = .02
energies_cmf=[]
energies_cmx=[]
energies_pt2=[]

for R in 1:n_steps
    
    scale = 1+R*step_size
    c= get_circle_coordinates(0.0,0.0,0.0,0.8*scale,12)
    #println(coordinates) 
    
    tmp=[]
    push!(tmp, Atom(1,"H",[c[1][1], c[1][2], c[1][3]]))
    push!(tmp, Atom(2,"H",[c[2][1], c[2][2], c[2][3]]))
    push!(tmp, Atom(3,"H",[c[4][1], c[4][2], c[4][3]]))
    push!(tmp, Atom(4,"H",[c[5][1], c[5][2], c[5][3]]))
    push!(tmp, Atom(5,"H",[c[7][1], c[7][2], c[7][3]]))
    push!(tmp, Atom(6,"H",[c[8][1], c[8][2], c[8][3]]))
    push!(tmp, Atom(7,"H",[c[10][1], c[10][2], c[10][3]]))
    push!(tmp, Atom(8,"H",[c[11][1], c[11][2], c[11][3]]))
    pymol=Molecule(0,1,tmp,basis)
    clusters    = [(1:2),(3:4),(5:6),(7:8)]
    init_fspace = [(1,1),(1,1),(1,1),(1,1)]
    na = 4
    nb = 4
    nroots = 1

    # get integrals
    mf = FermiCG.pyscf_do_scf(pymol)
    nbas = size(mf.mo_coeff)[1]
    ints = FermiCG.pyscf_build_ints(pymol,mf.mo_coeff, zeros(nbas,nbas));
    nelec = na + nb
    norb = size(ints.h1,1)
    # localize orbitals
    C = mf.mo_coeff
    Cl = FermiCG.localize(mf.mo_coeff,"lowdin",mf)
    FermiCG.pyscf_write_molden(pymol,Cl,filename="lowdin.molden")
    S = FermiCG.get_ovlp(mf)
    U =  C' * S * Cl
    println(" Rotate Integrals")
    flush(stdout)
    ints = FermiCG.orbital_rotation(ints,U)
    println(" done.")
    flush(stdout)

    #
    # define clusters
    clusters = [MOCluster(i,collect(clusters[i])) for i = 1:length(clusters)]
    display(clusters)

    rdm1 = zeros(size(ints.h1))
    #d1 = RDM1(n_orb(ints))
    e_cmf, U, d1  = ClusterMeanField.cmf_oo_diis(ints, clusters, init_fspace, RDM1(rdm1, rdm1), verbose=0, diis_start=3)
    #e_cmf, U, d1  = FermiCG.cmf_oo(ints, clusters, init_fspace, d1,
                                #max_iter_oo=40, verbose=0, gconv=1e-6, method="bfgs")
    FermiCG.pyscf_write_molden(pymol,Cl*U,filename="cmf.molden")
    #println(e_cmf)
    ints = FermiCG.orbital_rotation(ints,U)

    e_ref = e_cmf - ints.h0

    max_roots = 100

    cluster_bases = FermiCG.compute_cluster_eigenbasis(ints, clusters, verbose=0, max_roots=max_roots,
                                                    init_fspace=init_fspace, rdm1a=d1.a, rdm1b=d1.b, T=Float64)


    clustered_ham = FermiCG.extract_ClusteredTerms(ints, clusters)

    #
    cluster_ops = FermiCG.compute_cluster_ops(cluster_bases, ints);

    FermiCG.add_cmf_operators!(cluster_ops, cluster_bases, ints, d1.a, d1.b);

    ref_fock = FermiCG.FockConfig(init_fspace)


    ψ = FermiCG.BSTstate(clusters, FockConfig(init_fspace), cluster_bases)
    ept2 ,total_pt2= FermiCG.compute_pt2_energy(ψ, cluster_ops, clustered_ham, thresh_foi=1e-6)
    #println("the value of pt2 correction energy value is",total_pt2)
    push!(energies_pt2,total_pt2[1])
    display(ψ)
    σ = FermiCG.build_compressed_1st_order_state(ψ, cluster_ops, clustered_ham, nbody=4, thresh=1e-5)
    σ = FermiCG.compress(σ, thresh=1e-4)

    #H = FermiCG.nonorth_dot(ψ,σ)
    H1 = FermiCG.compute_expectation_value(ψ, cluster_ops, clustered_ham)

    H2 = FermiCG.orth_dot(σ,σ)
    H3 = FermiCG.compute_expectation_value(σ, cluster_ops, clustered_ham)
    #compute_expectation_value is not working; dimension mismatch

    sigma2 = FermiCG.build_compressed_1st_order_state(σ, cluster_ops, clustered_ham, nbody=4, thresh=1e-5)
    sigma2_compressed = FermiCG.compress(sigma2, thresh=1e-4)

    H4 = FermiCG.orth_dot(sigma2_compressed,sigma2_compressed)
    H5 = FermiCG.compute_expectation_value(sigma2_compressed, cluster_ops, clustered_ham)
    I_1=H1[1]
    I_2=H2[1]-I_1*H1[1]
    I_3=H3[1]-I_1*H2[1]-2*I_2*H1[1]
    I_4=H4[1]-I_1*H3[1]-3*I_2*H2[1]-3*I_3*H1[1]
    I_5=H5[1]-I_1*H4[1]-4*I_2*H3[1]-6*I_3*H2[1]-4*I_4*H1[1]
    E_K2=I_1-(I_2*I_2/I_3)*(1+(((I_4*I_2-I_3*I_3)^2)/(I_2*I_2*(I_5*I_3-I_4*I_4))))
    #=I_VEC=[I_2; I_3]
    I_MAT=[I_3 I_4;I_4 I_5]
    COR_ENERGY=inv(I_MAT)*I_VEC
    COR_ENERGY=I_VEC'*COR_ENERGY
    E_pk2=I_1-COR_ENERGY=#
    #println(E_pk2)
    push!(energies_cmx,E_K2)
    println(energies_cmx)
    push!(energies_cmf,e_cmf)
    println(energies_cmf)
    println(energies_pt2)
end
#plot(energies_cmf)
#savefig("cmf_ringH8.png")