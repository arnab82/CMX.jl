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

r = 0.74
a = 1
push!(atoms,Atom(1,"H", [0, 0*a, 0*r]))
push!(atoms,Atom(2,"H", [0, 0*a, 1*r]))
push!(atoms,Atom(3,"H", [0, 1*a,2*r]))
push!(atoms,Atom(4,"H", [0, 1*a,3*r]))
push!(atoms,Atom(5,"H", [0, 2*a, 4*r]))
push!(atoms,Atom(6,"H", [0, 2*a, 5*r]))
push!(atoms,Atom(7,"H", [0, 3*a, 6*r]))
push!(atoms,Atom(8,"H", [0, 3*a, 7*r]))
push!(atoms,Atom(9,"H", [0, 4*a, 8*r]))
push!(atoms,Atom(10,"H",[0, 4*a, 9*r]))
push!(atoms,Atom(11,"H",[0, 5*a, 10*r]))
push!(atoms,Atom(12,"H",[0, 5*a, 11*r]))
display(atoms)
#clusters    = [(1:2),(3:4),(5:6),(7:8),(9:10),(11:12)]
#init_fspace = [(1,1),(1,1),(1,1),(1,1),(1,1),(1,1)]
na = 6
nb = 6


basis = "sto-3g"
#basis = "6-31g**"
mol     = Molecule(0,1,atoms,basis)

nroots = 1

n_steps = 120    
step_size = .02

energies_cmf=[]
io = open("traj_H12_intra.xyz", "w");

for R in 1:n_steps
    scale = 1+R*step_size
    xyz = @sprintf("%5i\n\n", length(mol.atoms))
    tmp = []
    count=0
    for a in mol.atoms
        push!(tmp, Atom(count+=1,"H",[a.xyz[1]*scale, a.xyz[2]*scale, a.xyz[3]*scale]))
        xyz = xyz * @sprintf("%6s %24.16f %24.16f %24.16f \n", a.symbol, a.xyz[1]*scale, a.xyz[2]*scale, a.xyz[3]*scale)
    end
    display(tmp)
    pymol=Molecule(0,1,tmp,basis)
    println(xyz)
    write(io, xyz);
    #clusters    = [(1:10),(11:20),(21:30),(31:40),(41:50),(51:60)]
    clusters    = [(1:2),(3:4),(5:6),(7:8),(9:10),(11:12)]
    init_fspace = [(1,1),(1,1),(1,1),(1,1),(1,1),(1,1)]
    na = 6
    nb = 6

    nroots = 1

    # get integrals
    mf = pyscf_do_scf(pymol)
    nbas = size(mf.mo_coeff)[1]
    ints = pyscf_build_ints(mol,mf.mo_coeff, zeros(nbas,nbas));
    # localize orbitals
    C = mf.mo_coeff
    Cl = localize(mf.mo_coeff,"lowdin",mf)
    S = get_ovlp(mf)
    U =  C' * S * Cl
    println(" Rotate Integrals")
    flush(stdout)
    ints = orbital_rotation(ints,U)
    println(" done.")
    flush(stdout)

    #
    # define clusters
    clusters = [MOCluster(i,collect(clusters[i])) for i = 1:length(clusters)]
    display(clusters)

    rdm1 = zeros(size(ints.h1))
    #d1 = RDM1(n_orb(ints))
    e_cmf, U, d1  = ClusterMeanField.cmf_oo_diis(ints, clusters, init_fspace, RDM1(rdm1, rdm1),maxiter_oo = 200, maxiter_ci   = 200, maxiter_d1= 200 ,verbose=1, diis_start=3)
    #e_cmf, U, d1  = FermiCG.cmf_oo(ints, clusters, init_fspace, d1,
                                #max_iter_oo=40, verbose=0, gconv=1e-6, method="bfgs")                            ints = orbital_rotation(ints, U)
    push!(energies_cmf,e_cmf)
    ClusterMeanField.pyscf_write_molden(mol,Cl*U,filename="cmf.molden_H12")
    println(energies_cmf)
end
close(io)
empty!(atoms)
plot(energies_cmf)
savefig("cmf_h12_intra_.png")

