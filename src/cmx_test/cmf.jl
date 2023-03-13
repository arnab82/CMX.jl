using QCBase
using ClusterMeanField 
using NPZ
using InCoreIntegrals
using RDM
using JLD2

h0 = npzread("ints_h0.npy")
h1 = npzread("ints_h1.npy")
h2 = npzread("ints_h2.npy")
ints = InCoreInts(h0, h1, h2)

clusters = [(1:6),(7:12)]
init_fspace = [(6, 6), (6, 6)]

clusters = [MOCluster(i,collect(clusters[i])) for i = 1:length(clusters)]
display(clusters)

rdm1 = zeros(size(ints.h1))

#e_cmf, U, d1 = cmf_oo(ints, clusters, init_fspace, d1,
#                           max_iter_oo=10,verbose=0, gconv=1e-5, method="bfgs",sequential=true)
#ints = orbital_rotation(ints, U)

e_cmf, U, d1 =ClusterMeanField.cmf_oo_diis(ints, clusters, init_fspace, RDM1(rdm1, rdm1),maxiter_oo= 300, verbose=0, diis_start=4)

ints = orbital_rotation(ints, U)

@save "data_cmf.jld2" clusters init_fspace ints d1 e_cmf U 