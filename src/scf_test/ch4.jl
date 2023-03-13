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
io = open("traj_ch4.xyz", "w");
fci_energies=[]
for ri in 0:62
    println(ri)
    println("\n")
    xyz = @sprintf("%5i\n\n", 5)
    basis = "sto-3g"
    atoms = []
    r0 = 0.43 + 0.03 * ri
    push!(atoms,Atom(1,"C", [1*r0, 1*r0, 1*r0]))
    push!(atoms,Atom(2,"H", [2*r0, 2*r0, 0]))
    push!(atoms,Atom(3,"H", [0, 2*r0,2*r0]))
    push!(atoms,Atom(4,"H", [2*r0,0,2*r0]))
    push!(atoms,Atom(5,"H", [0, 0, 0]))
    println(atoms)
    for a in atoms
        xyz = xyz * @sprintf("%6s %24.16f %24.16f %24.16f \n", a.symbol, a.xyz[1], a.xyz[2], a.xyz[3])
    end
    println(xyz)
    write(io, xyz);
    na = 5
    nb = 5
    mol     = Molecule(0,1,atoms,basis)
    nroots =    1
    #get integrals
    mf = ClusterMeanField.pyscf_do_scf(mol)
    nbas = size(mf.mo_coeff)[1]
    ints = ClusterMeanField.pyscf_build_ints(mol,mf.mo_coeff, zeros(nbas,nbas));

    @printf(" Do FCI\n")
    nelec = na + nb
    #=
    norb = size(ints.h1,1)
    ansatz = FCIAnsatz(norb, na, nb)
    solver = SolverSettings(nroots=nroots, package="Arpack")
    solution = solve(ints, ansatz, solver)
    display(solution.energies[1])

    push!(fci_energies,solution.energies[1])
    display(fci_energies)=#
    pyscf = pyimport("pyscf")
    pyscf.lib.num_threads(1)
    fci = pyimport("pyscf.fci")
    cisolver = fci.direct_spin1.FCI(mf)
    cisolver.max_cycle = 200
    cisolver.conv_tol = 1e-8
    e_fci, v_fci = cisolver.kernel(ints.h1, ints.h2, norb, nelec, ecore=0, nroots =nroots)
    push!(fci_energies,e_fci+ints.h0)
    println((e_fci+ints.h0))
end
close(io)
println(fci_energies)