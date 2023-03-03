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

function get_circle_coordinates(center_x, center_y,center_z ,radius, num_points,R,scale1,scale)
    coordinates= []
    for i in 1:num_points
        angle = 2 * π * i / num_points
        x=center_x+0.0
        y=  center_y+ radius * cos(angle)
        z=  center_z+ radius * sin(angle)
        push!(coordinates,[x,y,z])
    end
    
    if R<2
        for i in 1:num_points
            angle = 2 * π * i / num_points+(scale)
            x=center_x+0.0
            y=  center_y+ radius * cos(angle)
            z=  center_z+ radius * sin(angle)
            push!(coordinates,[x,y,z])
        end
    else
        for i in 1:num_points
            angle = 2 * π * i / num_points+(scale1)
            x=center_x+0.0
            y=  center_y+ radius * cos(angle)
            z=  center_z+ radius * sin(angle)
            push!(coordinates,[x,y,z])
        end
    end
    return coordinates

end

basis="sto-3g"
n_steps = 70
step_size = .025
fci_energies=[]

io = open("traj_H6", "w");
for R in 1:n_steps
    scale = 1+R*step_size
    angle_num=70
    println(R)
    for r in 1:angle_num
        xyz = @sprintf("%5i\n\n", 6)
        scale1=π/24+(r*π/250)
        c= get_circle_coordinates(0.0,0.0,0.0,1.6*scale,3,r,scale1,π/24)
        tmp=[]
        push!(tmp, Atom(1,"H",[c[1][1], c[1][2], c[1][3]]))
        push!(tmp, Atom(2,"H",[c[4][1], c[4][2], c[4][3]]))
        push!(tmp, Atom(3,"H",[c[2][1], c[2][2], c[2][3]]))
        push!(tmp, Atom(4,"H",[c[5][1], c[5][2], c[5][3]]))
        push!(tmp, Atom(5,"H",[c[3][1], c[3][2], c[3][3]]))
        push!(tmp, Atom(6,"H",[c[6][1], c[6][2], c[6][3]]))
        pymol=Molecule(0,1,tmp,basis)
        for a in tmp
            xyz = xyz * @sprintf("%6s %24.16f %24.16f %24.16f \n", a.symbol, a.xyz[1], a.xyz[2], a.xyz[3])
        end
        println(xyz)
        write(io, xyz);
        
        na = 3
        nb = 3

        nroots = 1

        # get integrals
        mf = pyscf_do_scf(pymol)
        nbas = size(mf.mo_coeff)[1]
        ints = pyscf_build_ints(pymol,mf.mo_coeff, zeros(nbas,nbas));
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
    end
    println(fci_energies)
end
close(io)
#plot(fci_energies)