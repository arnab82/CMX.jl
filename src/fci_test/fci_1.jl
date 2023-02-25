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
atoms = []
#=r=0.5
a=2.0=#
r = 0.74
a = 1.5
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
println(atoms)
na = 6
nb = 6


basis = "sto-3g"
mol     = Molecule(0,1,atoms,basis)

nroots =    1

# get integrals
mf = FermiCG.pyscf_do_scf(mol)
nbas = size(mf.mo_coeff)[1]
ints = FermiCG.pyscf_build_ints(mol,mf.mo_coeff, zeros(nbas,nbas));

@printf(" Do FCI\n")
nelec = na + nb
norb = size(ints.h1,1)
ansatz = FCIAnsatz(norb, na, nb)
solver = SolverSettings(nroots=nroots, package="Arpack")
solution = solve(ints, ansatz, solver)
display(solution)
fci_energies=[]
push!(fci_energies,solution)
display(fci_energies)
#=pyscf = pyimport("pyscf")
pyscf.lib.num_threads(1)
fci = pyimport("pyscf.fci")
cisolver = pyscf.fci.direct_spin1.FCI()
cisolver.max_cycle = 200
cisolver.conv_tol = 1e-8
e_fci, v_fci = cisolver.kernel(ints.h1, ints.h2, norb, nelec, ecore=0, nroots =nroots)
println(e_fci)=#
n_steps=80
step_size = .05
fci_energies=[]
io = open("traj_0.74_1.5.xyz", "w");
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
    #clusters    = [(1:2),(3:4),(5:6),(7:8),(9:10),(11:12)]
    #init_fspace = [(1,1),(1,1),(1,1),(1,1),(1,1),(1,1)]
    #clusters    = [(1:4),(5:8),(9:12)]
    #init_fspace = [(2,2),(2,2),(2,2)]
    na = 6
    nb = 6

    nroots = 1

    # get integrals
    mf = FermiCG.pyscf_do_scf(pymol)
    nbas = size(mf.mo_coeff)[1]
    ints = FermiCG.pyscf_build_ints(mol,mf.mo_coeff, zeros(nbas,nbas));
    nelec = na + nb
    norb = size(ints.h1,1)
    ansatz = FCIAnsatz(norb, na, nb)
    solver = SolverSettings(nroots=1, package="Arpack")
    solution = solve(ints, ansatz, solver)
    display(solution)
    #=pyscf = pyimport("pyscf")
    pyscf.lib.num_threads(1)
    fci = pyimport("pyscf.fci")
    cisolver = fci.FCI(mf)
    cisolver.max_cycle = 200
    cisolver.conv_tol = 1e-8

    e_fci, v_fci = cisolver.kernel(ints.h1, ints.h2, norb, nelec, ecore=0, nroots =nroots,verbose=8)=#
    push!(fci_energies,solution)
    display(fci_energies)
end
display(fci_energies)
close(io)
empty!(atoms)