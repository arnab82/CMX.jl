using ClusterMeanField
using RDM
using QCBase
using PyCall
using Printf
using InCoreIntegrals
pyscf=pyimport("pyscf")
np=pyimport("numpy")
symm=pyimport("pyscf.symm")
scf_energy=[]
atoms = []
r0 = 0.8 
push!(atoms,Atom(1,"N", [0.0, 0.0, 0.0]))
push!(atoms,Atom(2,"N", [2.0, 0.0, 0.0]))


#basis = "6-31G"
basis="sto3g"
pymol = Molecule(0,1,atoms,basis)
molecule = """
    N   0.0 0.0 0.0
    N   1.1 0.0 0.0
    """
    print(molecule)
    basis = basis
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
ClusterMeanField.pyscf_write_molden(pymol, mf.mo_coeff,filename="C_RHF_n2.molden")




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
clusters = []
for i in 1:4
    push!(clusters,[])
end
frozen= [1,2]  
println(norb)



for i in 1:norb
    name = mol.irrep_name[sym_map[mf.orbsym[i]]]
    if !(i in frozen)
        if name == "A1g"
            push!(clusters[2],i)
        elseif name == "A1u"
            push!(clusters[2],i)
        elseif name == "E1gx" 
            push!(clusters[3],i)
        elseif  name == "E1ux"
            push!(clusters[3],i)
        elseif name == "E1gy" 
            push!(clusters[4],i)
        elseif name == "E1uy"
            push!(clusters[4],i)
        end
    end
end
           
#specify frozen core
clusters[1] = [1,2]
println(clusters) 
C=mf.mo_coeff
function c_act()
    C_ordered = zeros(norb,0)
    for (ci,c) in enumerate(clusters)
        C_ordered = hcat(C_ordered, C[:,c])
    end
    return C_ordered
end
C_act=c_act()
pyscf.tools.molden.from_mo(mol, "C_ordered_n2.molden", C_act)
init_fspace = [(2,2), (3,3), (1,1), (1,1)]
clusters=[(1,2),(3,4,7,10),(6,8),(5,9)]
#clusters=[(1, 2), (3, 4, 5, 6), (7, 8), (9, 10)]
#clusters=[(1, 2), (3, 4, 5, 10, 11, 14, 17, 18), (7, 8, 13, 15), (6, 9, 12, 16)]
#clusters=[(1:2),(3:10),(11:14),(15:18)]
println(clusters)
println(init_fspace)
for ci in clusters
    print(length(ci))
end


pyscf.scf.hf_symm.analyze(mf)
c_frozen=C[:,frozen]
d_frozen=2*c_frozen*c_frozen'
h0 = pyscf.gto.mole.energy_nuc(mol)
h1 = pyscf.scf.hf.get_hcore(mol)
println(size(h1))
h1 = C_act' *h1 *C_act
norb= size(C_act)[2]
h2 = pyscf.ao2mo.kernel(mol, C_act, aosym="s4", compact=false)
h2 = reshape(h2, (norb, norb, norb, norb))
S = mol.intor("int1e_ovlp_sph")

# get integrals

ints=InCoreInts(h0, h1, h2)
pymol = Molecule(0,1,atoms,basis)
na=7
nb=7
mf = pyscf_do_scf(pymol)
nelec = na + nb
nuc_energy=mf.energy_nuc()


# localize orbitals
#Cact_lo= pyscf.lo.PM(mol).kernel(C_act, verbose=4);
#println(size(Cact_lo))
#Cact_lo= localize(C_act,"lowdin",mf)
#ClusterMeanField.pyscf_write_molden(pymol,Cact_lo,filename="lowdin_N2.molden")
#U =  transpose(C_act)* S * Cact_lo
#println(size(U))
#println(" Rotate Integrals")
#flush(stdout)
#ints = orbital_rotation(ints,U)
#println(" done.")
#flush(stdout)
#define clusters

#cmf section

clusters = [MOCluster(i,collect(clusters[i])) for i = 1:length(clusters)]
display(clusters)
rdm1 = zeros(size(h1))
#d1 = RDM1(n_orb(ints))
e_cmf, U, d1  = ClusterMeanField.cmf_oo_diis(ints, clusters, init_fspace, RDM1(rdm1, rdm1), verbose=0, maxiter_oo=800,maxiter_ci=400, diis_start=3)
#e_cmf, U, d1  = ClusterMeanField.cmf_oo(ints, clusters, init_fspace, d1,
                                    #max_iter_oo=100, verbose=0, gconv=1e-6, method="bfgs")
ClusterMeanField.pyscf_write_molden(pymol,C_act*U,filename="cmf_n2.molden")
println(e_cmf)
