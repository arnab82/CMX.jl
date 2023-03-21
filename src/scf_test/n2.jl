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
symm=pyimport("pyscf.symm")
scf_energy=[]
atoms = []
r0 = 0.8 
push!(atoms,Atom(1,"N", [0.0, 0.0, 0.0]))
push!(atoms,Atom(2,"N", [1.1, 0.0, 0.0]))

basis = "6-31G"
basis="sto3g"

molecule = """
    N   0.0 0.0 0.0
    N   1.1 0.0 0.0
    """
print(molecule)
    
mol = pyscf.gto.Mole(
    atom    =   molecule,
    symmetry=true,
    spin    =   0,
    charge  =   0,
    basis   =   basis)
mol.build()
mf = pyscf.scf.ROHF(mol)
mf.verbose = 4
mf.conv_tol = 1e-8
mf.conv_tol_grad = 1e-5
mf.chkfile = "scf.fchk"
mf.init_guess = "sad"
mf.run(max_cycle=200)
println(" Hartree-Fock Energy: ",mf.e_tot)
norb =size(mf.mo_coeff)[1]
#print(mf.mo_coeff)
println(norb)

pymol = Molecule(0,1,atoms,basis)
ClusterMeanField.pyscf_write_molden(pymol, mf.mo_coeff,filename="C_RHF_n2.molden")


#getting the symmetry based orbitals
function myocc(mf)    
    mol = mf.mol
    irrep_id = mol.irrep_id
    so = mol.symm_orb
    orbsym = symm.label_orb_symm(mol, irrep_id, so, mf.mo_coeff)
    println(orbsym)
    println(mol.irrep_id)
    println(mol.irrep_name)
end
myocc(mf)
sym_map = Dict()
for (ir,irname) in enumerate(mol.irrep_id)
    sym_map[irname] = ir
end
println(sym_map)


#getting the clusters for the active space
clusters = []
for i in 1:3
    push!(clusters,[])
end
println(norb)
C=mf.mo_coeff   
#specify frozen core
frozen= [1,2]  
c_frozen=C[:,frozen]
d_frozen=2*c_frozen*c_frozen'
for i in 1:norb
    name = mol.irrep_name[sym_map[mf.orbsym[i]]]
    if !(i in frozen)
        if name == "A1g"
            push!(clusters[1],i)
        elseif name == "A1u"
            push!(clusters[1],i)
        elseif name == "E1gx" 
            push!(clusters[2],i)
        elseif  name == "E1ux"
            push!(clusters[2],i)
        elseif name == "E1gy" 
            push!(clusters[3],i)
        elseif name == "E1uy"
            push!(clusters[3],i)
        end
    end
end
       
println(clusters) 



#get the active ActiveSpace 

function c_act()
    C_ordered = zeros(norb,0)
    for (ci,c) in enumerate(clusters)
        C_ordered = hcat(C_ordered, C[:,c])
    end
    return C_ordered
end
C_act=c_act()
pyscf.tools.molden.from_mo(mol, "C_ordered_n2.molden", C_act)

#specify the clusters and initial fock space electrons

init_fspace = [(3,3), (1,1), (1,1)]
#clusters=[(1,2,5,8),(4,6),(3,7)]
clusters=[(1, 2, 3, 4) ,(5, 6), (7, 8)]
#clusters=[(1, 2), (3, 4, 5, 10, 11, 14, 17, 18), (7, 8, 13, 15), (6, 9, 12, 16)]
#clusters=[(1:8), (9:12), (13:16)]
println(clusters)
println(init_fspace)
for ci in clusters
    print(length(ci))
end

pyscf.scf.hf_symm.analyze(mf)
mol_1=make_pyscf_mole(pymol)
#making the integrals
#ints=ClusterMeanField.pyscf_build_ints(pymol,C_act,d_frozen)
h0 = pyscf.gto.mole.energy_nuc(mol_1)
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
if isapprox(abs(nact),0,atol=1e-8) == false
    println(nact)
    display(d_frozen)
    error(" I found embedded electrons in the active space?!")
end


h1 = h + j - .5*k
ints = InCoreInts(h0, h1, h2);
na=5
nb=5
nelec = na + nb
nuc_energy=mol_1.energy_nuc()


#cmf section

clusters = [MOCluster(i,collect(clusters[i])) for i = 1:length(clusters)]
display(clusters)
rdm1 = zeros(size(h1))
println(size(rdm1))
#d1 = RDM1(n_orb(ints))
e_cmf, U, d1  = ClusterMeanField.cmf_oo_diis(ints, clusters, init_fspace, RDM1(rdm1, rdm1), verbose=0, maxiter_oo=800,maxiter_ci=400, diis_start=3)
#e_cmf, U, d1  = ClusterMeanField.cmf_oo(ints, clusters, init_fspace, d1,
                                    #max_iter_oo=100, verbose=0, gconv=1e-6, method="bfgs")
ClusterMeanField.pyscf_write_molden(pymol,C_act*U,filename="cmf_n2.molden")
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
#sigma3 = FermiCG.build_compressed_1st_order_state(sigma2_compressed, cluster_ops, clustered_ham, nbody=4, thresh=1e-9)
#sigma3_compressed= FermiCG.compress(sigma3, thresh=1e-9)
#H6 = FermiCG.orth_dot(sigma3_compressed,sigma3_compressed)
#H7 = FermiCG.compute_expectation_value(sigma3_compressed, cluster_ops, clustered_ham)
    


I_1=H1[1]
I_2=H2[1]-I_1*H1[1]
I_3=H3[1]-I_1*H2[1]-2*I_2*H1[1]
I_4=H4[1]-I_1*H3[1]-3*I_2*H2[1]-3*I_3*H1[1]
I_5=H5[1]-I_1*H4[1]-4*I_2*H3[1]-6*I_3*H2[1]-4*I_4*H1[1]
E_K2=I_1-(I_2*I_2/I_3)*(1+(((I_4*I_2-I_3*I_3)^2)/(I_2*I_2*(I_5*I_3-I_4*I_4))))
cmx_2=E_K2+ints.h0
println(cmx_2)
#=I_6=H6[1]-I_1*H5[1]-5*I_2*H4[1]-10*I_3*H3[1]-10*I_4*H2[1]-5*I_5*H1[1]
    I_7=H7[1]-I_1*H6[1]-6*I_2*H5[1]-15*I_3*H4[1]-20*I_4*H3[1]-15*I_5*H2[1]
    I_V=[I_2;I_3;I_4]
    I_M=[I_3 I_4 I_5;I_4 I_5 I_6;I_5 I_6 I_7]
    CO_ENERGY=inv(I_M)*I_V
    CO_ENERGY=I_V'*CO_ENERGY
    cmx_3=I_1-CO_ENERGY+ints.h0
    #pds approach
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


    #println("*************************************************************TPSCI-CMX ENERGY*******************************************************************","\n\n")

    #=cmfstate = FermiCG.TPSCIstate(clusters, FockConfig(init_fspace),R=1,T=Float64)
    cmfstate[FermiCG.FockConfig(init_fspace)][FermiCG.ClusterConfig([1,1,1])] = [1.0]
    display(cmfstate)
    sig = FermiCG.open_matvec_thread(cmfstate, cluster_ops, clustered_ham, nbody=4, thresh=1e-6, prescreen=true)
    #display(size(sig))
    FermiCG.clip!(sig, thresh=1e-6)
    #display(size(sig))
    # <H> = (<0|H)|0> = <sig|0>
    H_1 = dot(sig, cmfstate)
    # <HH> = (<0|H)(H|0>) = <sig|sig>
    H_2 = dot(sig, sig)
    # <HHH> = (<0|H)H(H|0>) = <sig|H|sig>
    H_3 = FermiCG.compute_expectation_value_parallel(sig, cluster_ops, clustered_ham)
    # |sig> = H|sig> = HH|0>
    sig = FermiCG.open_matvec_thread(cmfstate, cluster_ops, clustered_ham, nbody=4, thresh=1e-6, prescreen=true)
    FermiCG.clip!(sig, thresh=1e-6)
    # <HHHH> = (<0|HH)(HH|0>) = <sig|sig>
    H_4 = dot(sig, sig)
    # <HHHH> = (<0|HH)H(HH|0>) = <sig|H|sig>
    H_5 = FermiCG.compute_expectation_value_parallel(sig, cluster_ops, clustered_ham)



    I1=H_1[1]
    I2=H_2[1]-I1*H_1[1]
    I3=H_3[1]-I1*H_2[1]-2*I2*H_1[1]
    I4=H_4[1]-I1*H_3[1]-3*I2*H_2[1]-3*I3*H_1[1]
    I5=H_5[1]-I1*H_4[1]-4*I2*H_3[1]-6*I3*H_2[1]-4*I4*H_1[1]
    EK2=I1-(I2*I2/I3)*(1+(((I4*I2-I3*I3)^2)/(I2*I2*(I5*I3-I4*I4))))
    cmx2=EK2+nuc_energy=#


