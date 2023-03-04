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
atoms = []

r = 0.745
a = 2
push!(atoms,Atom(1,"H", [0, 0*a, 0*r]))
push!(atoms,Atom(2,"H", [0, 0*a, 1*r]))
push!(atoms,Atom(3,"H", [0, 1*a,2*r]))
push!(atoms,Atom(4,"H", [0, 1*a,3*r]))

display(atoms)
clusters    = [(1:10),(11:20)]
init_fspace = [(1,1),(1,1)]
na = 6
nb = 6

basis = "6-31g**"
mol     = Molecule(0,1,atoms,basis)

nroots = 1

# get integrals
mf = FermiCG.pyscf_do_scf(mol)
nbas = size(mf.mo_coeff)[1]
ints = FermiCG.pyscf_build_ints(mol,mf.mo_coeff, zeros(nbas,nbas));
n_steps = 80    
step_size = .03

energies_cmf=[]
scf_energies=[]

for R in 1:n_steps
    scale = 1+R*step_size
    xyz = @sprintf("%5i\n\n", length(mol.atoms))
    tmp = []
    count=0
    for a in mol.atoms
        push!(tmp, Atom(count+=1,"H",[a.xyz[1], a.xyz[2]*scale, a.xyz[3]]))
        xyz = xyz * @sprintf("%6s %24.16f %24.16f %24.16f \n", a.symbol, a.xyz[1], a.xyz[2]*scale, a.xyz[3])
    end
    display(tmp)
    pymol=Molecule(0,1,tmp,basis)
    println(xyz)
    clusters    = [(1:10),(11:20)]
    init_fspace = [(1,1),(1,1)]
    na = 6
    nb = 6

    nroots = 1

    # get integrals
    mf = FermiCG.pyscf_do_scf(pymol)
    push!(scf_energies,(mf.kernel()))
    nbas = size(mf.mo_coeff)[1]
    ints = FermiCG.pyscf_build_ints(mol,mf.mo_coeff, zeros(nbas,nbas));
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

    rdm1 = zeros(size(ints.h1))
    #d1 = RDM1(n_orb(ints))
    e_cmf, U, d1  = ClusterMeanField.cmf_oo_diis(ints, clusters, init_fspace, RDM1(rdm1, rdm1),maxiter_oo = 400, maxiter_ci   = 400, maxiter_d1= 400 ,verbose=1, diis_start=3)
    #e_cmf, U, d1  = FermiCG.cmf_oo(ints, clusters, init_fspace, d1,
                                #max_iter_oo=40, verbose=0, gconv=1e-6, method="bfgs")                            ints = orbital_rotation(ints, U)
    push!(energies_cmf,e_cmf)
    ints =FermiCG.orbital_rotation(ints, U)
    FermiCG.pyscf_write_molden(mol,Cl*U,filename="cmf.molden")
    println(energies_cmf)
    println(scf_energies)
end
empty!(atoms)
plot(energies_cmf)
savefig("cmf_h4_inter.png")

