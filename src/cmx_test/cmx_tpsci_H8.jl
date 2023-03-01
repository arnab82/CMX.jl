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
using ClusterMeanField
atoms = []

r = 1
a = 2
push!(atoms,Atom(1,"H", [0, 0*a, 0*r]))
push!(atoms,Atom(2,"H", [0, 0*a, 1*r]))
push!(atoms,Atom(3,"H", [0, 1*a, 2*r]))
push!(atoms,Atom(4,"H", [0, 1*a, 3*r]))
push!(atoms,Atom(5,"H", [0, 2*a, 4*r]))
push!(atoms,Atom(6,"H", [0, 2*a, 5*r]))
push!(atoms,Atom(7,"H", [0, 3*a, 6*r]))
push!(atoms,Atom(8,"H", [0, 3*a, 7*r]))
#push!(atoms,Atom(9,"H", [0, 4*a, 8*r]))
#push!(atoms,Atom(10,"H",[0, 4*a, 9*r]))
#push!(atoms,Atom(11,"H",[0, 5*a, 10*r]))
#push!(atoms,Atom(12,"H",[0, 5*a, 11*r]))


clusters    = [(1:2),(3:4),(5:6),(7:8)]
init_fspace = [(1,1),(1,1),(1,1),(1,1)]

na = 4
nb = 4


basis = "sto-3g"
mol     = Molecule(0,1,atoms,basis)

nroots = 1

# get integrals
mf = FermiCG.pyscf_do_scf(mol)
nbas = size(mf.mo_coeff)[1]
ints = FermiCG.pyscf_build_ints(mol,mf.mo_coeff, zeros(nbas,nbas));


# localize orbitals
C = mf.mo_coeff
Cl = FermiCG.localize(mf.mo_coeff,"lowdin",mf)
#FermiCG.pyscf_write_molden(mol,Cl,filename="lowdin.molden")
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
e_cmf, U, d1  = ClusterMeanField.cmf_oo_diis(ints, clusters, init_fspace, RDM1(rdm1, rdm1),maxiter_oo = 200, maxiter_ci   = 200, maxiter_d1= 200 ,verbose=0, diis_start=3)
#FermiCG.pyscf_write_molden(mol,Cl*U,filename="cmf.molden")
ints = FermiCG.orbital_rotation(ints,U)

e_ref = e_cmf - ints.h0

max_roots = 100

cluster_bases = FermiCG.compute_cluster_eigenbasis(ints, clusters, verbose=1, max_roots=max_roots,
                                                   init_fspace=init_fspace, rdm1a=d1.a, rdm1b=d1.b, T=Float64)


clustered_ham = FermiCG.extract_ClusteredTerms(ints, clusters)

#
cluster_ops = FermiCG.compute_cluster_ops(cluster_bases, ints);

FermiCG.add_cmf_operators!(cluster_ops, cluster_bases, ints, d1.a, d1.b);

ref_fock = FermiCG.FockConfig(init_fspace)
display(ref_fock)
display(clusters)
cmfstate = FermiCG.TPSCIstate(clusters, FockConfig(init_fspace),R=1,T=Float64)

cmfstate[FermiCG.FockConfig(init_fspace)][FermiCG.ClusterConfig([1,1,1,1])] = [1.0]
display(cmfstate)
sig = FermiCG.open_matvec_thread(cmfstate, cluster_ops, clustered_ham, nbody=4, thresh=1e-6, prescreen=true)
display(size(sig))

FermiCG.clip!(sig, thresh=1e-5)
display(size(sig))

# <H> = (<0|H)|0> = <sig|0>
H1 = dot(sig, cmfstate)
println(H1)
# <HH> = (<0|H)(H|0>) = <sig|sig>
H2 = dot(sig, sig)

# <HHH> = (<0|H)H(H|0>) = <sig|H|sig>
H3 = FermiCG.compute_expectation_value_parallel(sig, cluster_ops, clustered_ham)

# |sig> = H|sig> = HH|0>
sig = FermiCG.open_matvec_thread(cmfstate, cluster_ops, clustered_ham, nbody=4, thresh=1e-6, prescreen=true)
FermiCG.clip!(sig, thresh=1e-5)

# <HHHH> = (<0|HH)(HH|0>) = <sig|sig>
H4 = dot(sig, sig)

# <HHHH> = (<0|HH)H(HH|0>) = <sig|H|sig>
H5 = FermiCG.compute_expectation_value_parallel(sig, cluster_ops, clustered_ham)
println(H5)
println("********************************************************************************")
#println("Cioslowski approach of CMX","\n\n\n")
println("********************************************************************************")
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
println("KNOWLES appraoch of pade approximant to calculate connected moment expansion","\n\n\n")
#=I_VEC=[I_2; I_3]
I_MAT=[I_3 I_4;I_4 I_5]
#I_V=[I_2;I_3;I_4]
#I_M=[I_3 I_4 I_5;I_4 I_5 I_6;I_5 I_6 I_7]

CORR_ENERGY=inv(I_MAT)*I_VEC
CORR_ENERGY=I_VEC'*CORR_ENERGY
E_PK2=I_1-CORR_ENERGY
#CO_ENERGY=inv(I_M)*I_V
#CO_ENERGY=I_V'*CO_ENERGY
#E_PK3=I_1-CO_ENERGY
display(E_PK2)
#display(E_PK3)

println("****************************************************")
println("PDS appraoch to calculate connected moment expansion","\n\n\n")

#FOR N=3
M_11=H4
M_12=M_21=H3
M_22=H2
M_31=M_13=H2
M_32=M_23=H1
M_33=FermiCG.orth_dot(ψ,ψ)
B_1=H5
B_2=H4
B_3=H3

M= [M_11 M_12 M_13; M_21 M_22 M_23; M_31 M_32 M_33]
B=[B_1 ;B_2; B_3]
A=M\(-B)
display(A)
np=pyimport("numpy")
EN=[1;A]
x=np.roots(EN)
display(x)
#=FOR N=4
m_11=H6
m_12=m_21=H5
m_22=m_31=m_13=H4
m_32=m_23=m_14=m_41=H3
m_33=m_42=m_24=H2
m_34=m_43=H1
m_44=FermiCG.orth_dot(ψ,ψ)
b_1=H7
b_2=H6
b_3=H4
b_4=H3

m= [m_11 m_12 m_13 m_14; m_21 m_22 m_23 m_24; m_31 m_32 m_33 m_34;m_41 m_42 m_43 m_44]
b=[b_1 ;b_2; b_3;b_4]
a=m\(-b)=#