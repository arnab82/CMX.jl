using FermiCG
#using JLD2
using PyCall
#using Plots
using LinearAlgebra
using Printf
using QCBase
using RDM
using ClusterMeanField
pyscf = pyimport("pyscf");
fcidump = pyimport("pyscf.tools.fcidump");
ctx = fcidump.read("fcidump_4mer");
h = ctx["H1"];
g = ctx["H2"];
ecore = ctx["ECORE"];
g = pyscf.ao2mo.restore("1", g, size(h,2))
ints = InCoreInts(ecore,h,g)
rdm1 = zeros(size(ints.h1))
na = 12
nb = 12
clusters_in    = [(1:6),(7:12),(13:18),(19:24)]
#init_fspace = [(3,3),(3,3),(3,3),(3,3)]
n_clusters = 4
# define clusters
cluster_list = [collect(1:6), collect(7:12), collect(13:18), collect(19:24)]
clusters = [MOCluster(i,collect(cluster_list[i])) for i = 1:length(cluster_list)]
init_fspace = [ (3,3) for i in 1:n_clusters]
display(clusters)
#run cmf_oo
e_cmf, U_cmf, d1  = ClusterMeanField.cmf_oo_diis(ints, clusters, init_fspace, RDM1(rdm1, rdm1), verbose=0, diis_start=3);
ints = orbital_rotation(ints,U_cmf)
#@save  "/home/arnab22/cmx.jl/cmf_diis.jld2" ints d1 clusters init_fspace
M = 20
ref_fock = FockConfig(init_fspace)
# Build Cluster basis
cluster_bases = FermiCG.compute_cluster_eigenbasis_spin(ints, clusters, d1, [3,3,3,3], ref_fock, max_roots=M, verbose=1);
#
# Build ClusteredOperator
clustered_ham = FermiCG.extract_ClusteredTerms(ints, clusters);
#
# Build Cluster Operators
cluster_ops = FermiCG.compute_cluster_ops(cluster_bases, ints);
#
# Add cmf hamiltonians for doing MP-style PT2
FermiCG.add_cmf_operators!(cluster_ops, cluster_bases, ints, d1.a, d1.b, verbose=0);
nroots = 1
ci_vector = FermiCG.TPSCIstate(clusters, FermiCG.FockConfig(init_fspace), R=nroots);
ci_vector[FermiCG.FockConfig(init_fspace)][FermiCG.ClusterConfig([1,1,1,1])] = zeros(Float64,nroots)
e0, v0 = FermiCG.tpsci_ci(ci_vector, cluster_ops, clustered_ham,
                          thresh_asci =-1,     # Threshold of P-space configs to search from
                          thresh_foi  =1e-5,    # Threshold for keeping terms when defining FOIS
                          thresh_cipsi=0.001, # Threshold for adding to P-space
                          max_iter=10);
@time e2 = FermiCG.compute_pt2_energy(v0, cluster_ops, clustered_ham, thresh_foi=1e-8);
name = "learn_arc_thresh_0.001.jld2"
println()
println("  *======TPSCI results======*")
@printf("TCI Thresh: %8.6f  Dim:%8d\n",0.001,size(v0)[1])
println()
@printf("TCI %5s %12s %12s\n", "Root", "E(0)", "E(2)")
for r in 1:nroots
    @printf("TCI %5s %12.8f %12.8f\n",r, e0[r] + ecore, e0[r] + e2[r] + ecore)
end
clustered_S2 = FermiCG.extract_S2(ci_vector.clusters)
println()
println("  *======TPSCI S2 results======*")
@printf(" %-50s", "Compute FINAL S2 expectation values: ")
@time s2 = FermiCG.compute_expectation_value_parallel(v0, cluster_ops, clustered_S2)
@printf(" %5s %12s %12s\n", "Root", "Energy", "S2")
for r in 1:nroots
    @printf(" %5s %12.8f %12.8f\n",r, e0[r]+ecore, abs(s2[r]))
end
#@save "/home/arnab22/cmx.jl"*string(name) clusters d1 ints cluster_bases ci_vector e0 v0 e2 ecore s2
                        