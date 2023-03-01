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
energies_cmf=[]
io = open("traj_H6_RING_new.xyz", "w");
for R in 1:n_steps
    scale = 1+R*step_size
    angle_num=70
    #=if R<18
        angle_num=18
    elseif 17<R<24
        angle_num=21
    elseif  23<R<41
        angle_num=24
    else 
        angle_num=28
    end=#
    for r in 1:angle_num
        xyz = @sprintf("%5i\n\n", 6)
        if R<26
            scale1=π/13.72+(r*π/290)
            c= get_circle_coordinates(0.0,0.0,0.0,1.6*scale,3,r,scale1,π/13.72)
        elseif 25<R<50
            scale1=π/21+(r*π/290)
            c= get_circle_coordinates(0.0,0.0,0.0,1.6*scale,3,r,scale1,π/21)
        else
            scale1=π/28+(r*π/290)
            c= get_circle_coordinates(0.0,0.0,0.0,1.6*scale,3,r,scale1,π/28)
        end
        #println(c) 
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
        
    end
end
close(io)
