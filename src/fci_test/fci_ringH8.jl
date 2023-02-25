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

function get_circle_coordinates(center_x, center_y,center_z ,radius, num_points)
    coordinates= []

    for i in 1:num_points
        angle = 2 * π * i / num_points
        x=center_x+0.0
        y=  center_y+ radius * cos(angle)
        z=  center_z+ radius * sin(angle)
        push!(coordinates,[x,y,z])
    end
    return coordinates
    
end
basis="sto-3g"
n_steps = 150
step_size = .04
fci_energies=[]

for R in 1:n_steps
    scale = 1+R*step_size
    c= get_circle_coordinates(0.0,0.0,0.0,0.8*scale,12)
    #println(coordinates) 
    
    tmp=[]
    push!(tmp, Atom(1,"H",[c[1][1], c[1][2], c[1][3]]))
    push!(tmp, Atom(2,"H",[c[2][1], c[2][2], c[2][3]]))
    push!(tmp, Atom(3,"H",[c[4][1], c[4][2], c[4][3]]))
    push!(tmp, Atom(4,"H",[c[5][1], c[5][2], c[5][3]]))
    push!(tmp, Atom(5,"H",[c[7][1], c[7][2], c[7][3]]))
    push!(tmp, Atom(6,"H",[c[8][1], c[8][2], c[8][3]]))
    push!(tmp, Atom(7,"H",[c[10][1], c[10][2], c[10][3]]))
    push!(tmp, Atom(8,"H",[c[11][1], c[11][2], c[11][3]]))
    pymol=Molecule(0,1,tmp,basis)
    na = 4
    nb = 4

    nroots = 1

    # get integrals
    mf = FermiCG.pyscf_do_scf(pymol)
    nbas = size(mf.mo_coeff)[1]
    ints = FermiCG.pyscf_build_ints(pymol,mf.mo_coeff, zeros(nbas,nbas));
    nelec = na + nb
    norb = size(ints.h1,1)
    ansatz = FCIAnsatz(norb, na, nb)
    solver = SolverSettings(nroots=1, package="Arpack")
    solution = solve(ints, ansatz, solver)
    display(solution)
    println(typeof(FCIAnsatz))
    #=pyscf = pyimport("pyscf")
    pyscf.lib.num_threads(1)
    fci = pyimport("pyscf.fci")
    cisolver = fci.FCI(mf)
    cisolver.max_cycle = 400
    cisolver.conv_tol = 1e-8
    e_fci, v_fci = cisolver.kernel(ints.h1, ints.h2, norb, nelec, ecore=0, nroots =nroots,verbose=1)=#
    push!(fci_energies,solution.energies[1])
    println(fci_energies)

end
#plot(fci_energies)