
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
display(atoms)
clusters    = [(1:2),(3:4),(5:6),(7:8)]
init_fspace = [(1,1),(1,1),(1,1),(1,1)]
#clusters    = [(1:4),(5:8),(9:12)]
#init_fspace = [(2,2),(2,2),(2,2)]
na = 4
nb = 4


basis = "sto-3g"
mol     = Molecule(0,1,atoms,basis)

nroots = 1

# get integrals
mf = FermiCG.pyscf_do_scf(mol)
nbas = size(mf.mo_coeff)[1]
ints = FermiCG.pyscf_build_ints(mol,mf.mo_coeff, zeros(nbas,nbas));

@printf(" Do FCI\n")
pyscf = pyimport("pyscf")
pyscf.lib.num_threads(1)
fci = pyimport("pyscf.fci")
cisolver = pyscf.fci.direct_spin1.FCI()
cisolver.max_cycle = 200
cisolver.conv_tol = 1e-8
nelec = na + nb
norb = size(ints.h1,1)
e_fci, v_fci = cisolver.kernel(ints.h1, ints.h2, norb, nelec, ecore=0, tol=1e-5,nroots =nroots)
println("the value of fci energy is",e_fci)
#e_fci = [-18.33022092,
#         -18.05457644]
#=e_fci  = [-18.33022092,
          -18.05457645,
          -18.02913047,
          -17.99661027
         ]

for i in 1:length(e_fci)
    @printf(" %4i %12.8f %12.8f\n", i, e_fci[i], e_fci[i]+ints.h0)
end=#


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
clusters = [MOCluster(i,collect(clusters[i])) for i = 1:length(clusters)]
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
display(ref_fock)

ψ = FermiCG.BSTstate(clusters, FockConfig(init_fspace), cluster_bases)
display(ψ)

ept2 ,total_pt2= FermiCG.compute_pt2_energy(ψ, cluster_ops, clustered_ham, thresh_foi=1e-6)
println("the value of pt2 correction energy value is",total_pt2)
σ = FermiCG.build_compressed_1st_order_state(ψ, cluster_ops, clustered_ham, nbody=4, thresh=1e-5)
σ = FermiCG.compress(σ, thresh=1e-4)

#H = FermiCG.nonorth_dot(ψ,σ)
H1 = FermiCG.compute_expectation_value(ψ, cluster_ops, clustered_ham)

H2 = FermiCG.orth_dot(σ,σ)
#sigma2 = FermiCG.build_compressed_1st_order_state(σ, cluster_ops, clustered_ham, nbody=4, thresh=1e-3)
#sigma2_compressed= FermiCG.compress(sigma2, thresh=1e-3)
#sigma3 = FermiCG.build_compressed_1st_order_state(sigma2_compressed, cluster_ops, clustered_ham, nbody=4, thresh=1e-3)
#sigma3_compressed= FermiCG.compress(sigma3, thresh=1e-3)
#sigma4 = FermiCG.build_compressed_1st_order_state(sigma3_compressed, cluster_ops, clustered_ham, nbody=4, thresh=1e-3)
#sigma4_compressed= FermiCG.compress(sigma4, thresh=1e-3)
#H3 = FermiCG.nonorth_dot(ψ,sigma3_compressed)
#display(H3)
#H4 = FermiCG.orth_dot(sigma2_compressed,sigma2_compressed)
#display(H4)
#H5=FermiCG.nonorth_dot(sigma3_compressed,sigma2_compressed)
#display(H5)
#H6 = FermiCG.orth_dot(sigma3_compressed,sigma3_compressed)
#display(H6)
#H7=FermiCG.nonorth_dot(sigma3_compressed,sigma4_compressed)
#display(H7)
H3 = FermiCG.compute_expectation_value(σ, cluster_ops, clustered_ham)

sigma2 = FermiCG.build_compressed_1st_order_state(σ, cluster_ops, clustered_ham, nbody=4, thresh=1e-5)
sigma2_compressed = FermiCG.compress(sigma2, thresh=1e-4)

H4 = FermiCG.orth_dot(sigma2_compressed,sigma2_compressed)
H5 = FermiCG.compute_expectation_value(sigma2_compressed, cluster_ops, clustered_ham)
#sigma3 = FermiCG.build_compressed_1st_order_state(sigma2_compressed, cluster_ops, clustered_ham, nbody=4, thresh=1e-5)
#sigma3_compressed= FermiCG.compress(sigma3, thresh=1e-5)
#H6 = FermiCG.orth_dot(sigma3_compressed,sigma3_compressed)
#H7 = FermiCG.compute_expectation_value(sigma3_compressed, cluster_ops, clustered_ham)
println("********************************************************************************")
println("Cioslowski approach of CMX","\n\n\n")
println("********************************************************************************")
I_1=H1[1]
I_2=H2[1]-I_1*H1[1]
I_3=H3[1]-I_1*H2[1]-2*I_2*H1[1]
I_4=H4[1]-I_1*H3[1]-3*I_2*H2[1]-3*I_3*H1[1]
I_5=H5[1]-I_1*H4[1]-4*I_2*H3[1]-6*I_3*H2[1]-4*I_4*H1[1]
#I_6=H6[1]-I_1*H5[1]-5*I_2*H4[1]-10*I_3*H3[1]-10*I_4*H2[1]-5*I_5*H1[1]
#I_7=H7[1]-I_1*H6[1]-6*I_2*H5[1]-15*I_3*H4[1]-20*I_4*H3[1]-15*I_5*H2[1]
E_2=I_1-(I_2*I*2/I_3)
display(E_2)
E_3=I_1-(I_2*I*2/I_3)-(1/I_3)*((I_2*I_4-I_3*I_3)*(I_2*I_4-I_3*I_3)/(I_5*I_3-I_4*I_4))
display(E_3)
println("the correlation energy is",E_2-I_1,E_3-I_1,"\n\n\n")
#
#KNOWLES approach of CMX
println("****************************************************")
println("KNOWLES appraoch to calculate connected moment expansion","\n\n\n")
E_K2=I_1-(I_2*I_2/I_3)*(1+(((I_4*I_2-I_3*I_3)^2)/(I_2*I_2*(I_5*I_3-I_4*I_4))))
display(E_K2)
println("KNOWLES appraoch of pade approximant to calculate connected moment expansion","\n\n\n")
I_VEC=[I_2; I_3]
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
a=m\(-b)
display(a)
np=pyimport("numpy")
en=[1;a]
y=np.roots(en)
display(y)=#
#r=1,a=2  =>cmx=-16.339068158015493,e_pt2=-16.339218324221992,e_fci=-16.339255259043547
#r=1,a=1.5  =>cmx=-17.280823154041652,e_pt2=-17.281137763308827,e_fci=-17.28137532355515
#r=1,a=1 =>cmx=-18.328994713532644,e_pt2=-18.328662816535758,e_fci=-18.33021971589803
#r=0.74 a=1 =>cmx=-21.520217490123503,pt2=-21.520136080673915,fci=-21.520924639670667,threshold value for bst state =1e-5,(thresh_foi=1e-6,tol=1e-5 for pt2)
#r=0.74 a=1 => cmx=-21.518387846175933,pt2=-21.5201256525544 threshold value for bst state =1e-4