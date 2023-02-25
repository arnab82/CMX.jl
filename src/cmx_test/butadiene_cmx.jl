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
molecule= "
C         -2.30986        0.50909        0.01592
C         -0.98261        0.43259       -0.05975
H         -2.84237        1.33139       -0.45081
H         -2.89518       -0.26464        0.20336
H         -0.43345        1.20285       -0.59462
C         -0.26676       -0.68753        0.57690
C          1.06323       -0.78274        0.51273
H         -0.82592       -1.45252        1.10939
H          1.65570       -0.03776       -0.01031
H          1.57073       -1.61569        0.98847
"
atoms = []
for (li,line) in enumerate(split(rstrip(lstrip(molecule)), "\n"))
    l = split(line)
    push!(atoms, Atom(li, l[1], parse.(Float64,l[2:4])))
end
display(atoms)
cluster_list = [(1:13),(14:26)] 
init_fspace = [(1,1),(1,1)]
na= 15
nb=15
basis = "sto-3g"
nroots = 1

mol = Molecule(0,1,atoms,basis)

#get integrals
mf = FermiCG.pyscf_do_scf(mol)
nbas = size(mf.mo_coeff)[1]
println(nbas)
ints = FermiCG.pyscf_build_ints(mol,mf.mo_coeff, zeros(nbas,nbas));
#=@printf(" Do FCI\n")
pyscf = pyimport("pyscf")
pyscf.lib.num_threads(1)
fci = pyimport("pyscf.fci")
cisolver = pyscf.fci.direct_spin1.FCI()
cisolver.max_cycle = 200
cisolver.conv_tol = 1e-5
nelec = na + nb
norb = size(ints.h1,1)
e_fci, v_fci = cisolver.kernel(ints.h1, ints.h2, norb, nelec, ecore=0, nroots =nroots)
println("the value fci ground state energy is",e_fci)
=#

# localize orbitals
C = mf.mo_coeff
Cl = FermiCG.localize(mf.mo_coeff,"lowdin",mf)
FermiCG.pyscf_write_molden(mol,Cl,filename="lowdin.molden")
S = FermiCG.get_ovlp(mf)
U =  C' * S * Cl
println(" Rotate Integrals")
flush(stdout)
ints = FermiCG.orbital_rotation(ints,U)
println(" done.")
flush(stdout)

#
# define clusters
clusters = [MOCluster(i,collect(cluster_list[i])) for i = 1:length(cluster_list)]
display(clusters)

d1 = RDM1(n_orb(ints))
e_cmf, U, d1  = FermiCG.cmf_oo(ints, clusters, init_fspace, d1,
                               max_iter_oo=40, verbose=0, gconv=1e-6, method="bfgs")
FermiCG.pyscf_write_molden(mol,Cl*U,filename="cmf.molden")
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
display(ψ)
ept2 ,total_pt2= FermiCG.compute_pt2_energy(ψ, cluster_ops, clustered_ham, thresh_foi=1e-5)
println("the value of pt2 correction energy value is",total_pt2)
σ = FermiCG.build_compressed_1st_order_state(ψ, cluster_ops, clustered_ham, nbody=4, thresh=1e-5)
σ = FermiCG.compress(σ, thresh=1e-5)

#H = FermiCG.nonorth_dot(ψ,σ)
H1 = FermiCG.compute_expectation_value(ψ, cluster_ops, clustered_ham)

H2 = FermiCG.orth_dot(σ,σ)
H3 = FermiCG.compute_expectation_value(σ, cluster_ops, clustered_ham)


sigma2 = FermiCG.build_compressed_1st_order_state(σ, cluster_ops, clustered_ham, nbody=4, thresh=1e-5)
sigma2_compressed = FermiCG.compress(sigma2, thresh=1e-5)


H4 = FermiCG.orth_dot(sigma2_compressed,sigma2_compressed)
H5 = FermiCG.compute_expectation_value(sigma2_compressed, cluster_ops, clustered_ham)
#=sigma3 = FermiCG.build_compressed_1st_order_state(sigma2_compressed, cluster_ops, clustered_ham, nbody=4, thresh=1e-5)
sigma3_compressed= FermiCG.compress(sigma3, thresh=1e-5)
H6 = FermiCG.orth_dot(sigma3_compressed,sigma3_compressed)
H7 = FermiCG.compute_expectation_value(sigma3_compressed, cluster_ops, clustered_ham)=#
I_1=H1[1]
I_2=H2[1]-I_1*H1[1]
I_3=H3[1]-I_1*H2[1]-2*I_2*H1[1]
I_4=H4[1]-I_1*H3[1]-3*I_2*H2[1]-3*I_3*H1[1]
I_5=H5[1]-I_1*H4[1]-4*I_2*H3[1]-6*I_3*H2[1]-4*I_4*H1[1]
#I_6=H6[1]-I_1*H5[1]-5*I_2*H4[1]-10*I_3*H3[1]-10*I_4*H2[1]-5*I_5*H1[1]
#I_7=H7[1]-I_1*H6[1]-6*I_2*H5[1]-15*I_3*H4[1]-20*I_4*H3[1]-15*I_5*H2[1]
#KNOWLES approach of CMX
println("****************************************************")
println("KNOWLES appraoch to calculate connected moment expansion","\n\n\n")
E_K2=I_1-(I_2*I_2/I_3)*(1+(((I_4*I_2-I_3*I_3)^2)/(I_2*I_2*(I_5*I_3-I_4*I_4))))
display(E_K2)
#=I_V=[I_2;I_3;I_4]
I_M=[I_3 I_4 I_5;I_4 I_5 I_6;I_5 I_6 I_7]
CO_ENERGY=inv(I_M)*I_V
CO_ENERGY=I_V'*CO_ENERGY
E_PK3=I_1-CO_ENERGY
display(E_PK3)=#