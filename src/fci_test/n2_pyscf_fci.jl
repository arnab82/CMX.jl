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
using Plots
using ActiveSpaceSolvers
using Plots
using ClusterMeanField
pyscf=pyimport("pyscf")
fcienergies=[]

for r in 1:40
    println(r)

    basis = "6-31G"
    #basis="sto-3g"
    atoms = []
    scale=0.8+r*0.05
    push!(atoms,Atom(1,"N", [0.0, 0.0, 0.0]))
    push!(atoms,Atom(2,"N", [scale, 0.0, 0.0]))

    pymol = Molecule(0,1,atoms,basis)
    na=7
    nb=7
    # get integrals
    mf = pyscf_do_scf(pymol)
    nbas = size(mf.mo_coeff)[1]
    ints = pyscf_build_ints(pymol,mf.mo_coeff, zeros(nbas,nbas));
    nelec = na + nb
    norb = size(ints.h1,1)
    #=ansatz = FCIAnsatz(norb, na, nb)
    solver = SolverSettings(nroots=1, package="Arpack")
    solution = solve(ints, ansatz, solver)
    display(solution)=#
    pyscf.lib.num_threads(1)
    fci = pyimport("pyscf.fci")
    cisolver = pyscf.fci.direct_spin1.FCI()
    cisolver.max_cycle = 200
    cisolver.conv_tol = 1e-6
    e_fci, v_fci = cisolver.kernel(ints.h1, ints.h2, norb, nelec, ecore=0, nroots =1,verbose=8)
    println(e_fci)
    println(typeof(e_fci))
    e_fci=e_fci+ints.h0
    push!(fcienergies,e_fci)
    println(fcienergies)
end

