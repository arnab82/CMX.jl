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


function get_circle_coordinates(center_x, center_y,center_z ,radius, num_points,R)
    coordinates= []
    for i in 1:num_points
        angle = 2 * π * i / num_points
        x=center_x+0.0
        y=  center_y+ radius * cos(angle)
        z=  center_z+ radius * sin(angle)
        push!(coordinates,[x,y,z])
    end
    if R<2
        scale1=R*π/18
    elseif 1<R<4
        scale1=R*π/36
    elseif 3<R<6
        scale1=R*π/48
    elseif 5<R<17
        scale1=R*π/54
    else
        scale1=R*π/68
    end
    for i in 1:num_points
        angle = 2 * π * i / num_points+(scale1)
        x=center_x+0.0
        y=  center_y+ radius * cos(angle)
        z=  center_z+ radius * sin(angle)
        push!(coordinates,[x,y,z])
    end
    return coordinates

end

basis="sto-3g"
n_steps = 50
step_size = .03
energies_cmf=[]
io = open("traj_H8_RING.xyz", "w");
for R in 1:n_steps
    scale = 1+R*step_size
    if R<6
        angle_num=12
    elseif 5<R<16
        angle_num=16
    elseif 15<R<31
        angle_num=20
    elseif 30<R<51
        angle_num=24
    else
        angle_num=30
    end
    for r in 1:angle_num
        xyz = @sprintf("%5i\n\n", 8)
        c= get_circle_coordinates(0.0,0.0,0.0,1.6*scale,4,r)
        #println(c) 
        tmp=[]
        push!(tmp, Atom(1,"H",[c[1][1], c[1][2], c[1][3]]))
        push!(tmp, Atom(2,"H",[c[5][1], c[5][2], c[5][3]]))
        push!(tmp, Atom(3,"H",[c[2][1], c[2][2], c[2][3]]))
        push!(tmp, Atom(4,"H",[c[6][1], c[6][2], c[6][3]]))
        push!(tmp, Atom(5,"H",[c[3][1], c[3][2], c[3][3]]))
        push!(tmp, Atom(6,"H",[c[7][1], c[7][2], c[7][3]]))
        push!(tmp, Atom(7,"H",[c[4][1], c[4][2], c[4][3]]))
        push!(tmp, Atom(8,"H",[c[8][1], c[8][2], c[8][3]]))
        #println(tmp)
        pymol=Molecule(0,1,tmp,basis)
        for a in tmp
            xyz = xyz * @sprintf("%6s %24.16f %24.16f %24.16f \n", a.symbol, a.xyz[1], a.xyz[2], a.xyz[3])
        end
        println(xyz)
        write(io, xyz);
        #=clusters    = [(1:2),(3:4),(5:6),(7:8)]
        init_fspace = [(1,1),(1,1),(1,1),(1,1)]
        na = 4
        nb = 4
        nroots = 1

        # get integrals
        mf = pyscf_do_scf(pymol)
        nbas = size(mf.mo_coeff)[1]
        ints = pyscf_build_ints(pymol,mf.mo_coeff, zeros(nbas,nbas));
        nelec = na + nb
        norb = size(ints.h1,1)
        # localize orbitals
        C = mf.mo_coeff
        Cl = localize(mf.mo_coeff,"lowdin",mf)
        ClusterMeanField.pyscf_write_molden(pymol,Cl,filename="lowdin.molden")
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
        e_cmf, U, d1  = ClusterMeanField.cmf_oo_diis(ints, clusters, init_fspace, RDM1(rdm1, rdm1), verbose=0, diis_start=3)
        #e_cmf, U, d1  = FermiCG.cmf_oo(ints, clusters, init_fspace, d1,
                                    #max_iter_oo=40, verbose=0, gconv=1e-6, method="bfgs")
        ClusterMeanField.pyscf_write_molden(pymol,Cl*U,filename="cmf.molden")
        #println(e_cmf)
        push!(energies_cmf,e_cmf)
        println(energies_cmf)=#
    end
end
close(io)
plot(energies_cmf)
#savefig("cmf_ringH8.png")