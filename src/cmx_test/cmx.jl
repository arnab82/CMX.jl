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

@load "data_cmf.jld2"
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
ept2 = FermiCG.compute_pt2_energy(ψ, cluster_ops, clustered_ham, thresh_foi=1e-6,tol=1e-6,verbose=1)
total_pt2=ept2[1]+ints.h0
println("the value of pt2 correction energy value is",total_pt2)
display(ψ)
σ = FermiCG.build_compressed_1st_order_state(ψ, cluster_ops, clustered_ham, nbody=4, thresh=1e-9)
σ = FermiCG.compress(σ, thresh=1e-9)

#H = FermiCG.nonorth_dot(ψ,σ) 
H1 = FermiCG.compute_expectation_value(ψ, cluster_ops, clustered_ham)
H2 = FermiCG.orth_dot(σ,σ)
H3 = FermiCG.compute_expectation_value(σ, cluster_ops, clustered_ham)
sigma2 = FermiCG.build_compressed_1st_order_state(σ, cluster_ops, clustered_ham, nbody=4, thresh=1e-9)
sigma2_compressed = FermiCG.compress(sigma2, thresh=1e-9)
H4 = FermiCG.orth_dot(sigma2_compressed,sigma2_compressed)
H5 = FermiCG.compute_expectation_value(sigma2_compressed, cluster_ops, clustered_ham)
I_1=H1[1]
I_2=H2[1]-I_1*H1[1]
I_3=H3[1]-I_1*H2[1]-2*I_2*H1[1]
I_4=H4[1]-I_1*H3[1]-3*I_2*H2[1]-3*I_3*H1[1]
I_5=H5[1]-I_1*H4[1]-4*I_2*H3[1]-6*I_3*H2[1]-4*I_4*H1[1]
E_K2=I_1-(I_2*I_2/I_3)*(1+(((I_4*I_2-I_3*I_3)^2)/(I_2*I_2*(I_5*I_3-I_4*I_4))))
cmx_2=E_K2+ints.h0
println(cmx_2)=#