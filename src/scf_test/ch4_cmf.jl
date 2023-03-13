sing QCBase
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
    xyz = @sprintf("%5i\n\n", 6)
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
    clusters    = [(1:2),(3:4),(5:6),(7,8),(9,10)]
    init_fspace = [(1,1),(1,1),(1,1),(1,1),(1,1)]
    na = 5
    nb = 5
    mol     = Molecule(0,1,atoms,basis)
    nroots =    1

    # get integrals
    mf = pyscf_do_scf(pymol)
    nbas = size(mf.mo_coeff)[1]
    ints = pyscf_build_ints(pymol,mf.mo_coeff, zeros(nbas,nbas));
    nelec = na + nb
    norb = size(ints.h1,1)
    nuc_energy=mf.energy_nuc()



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

    println("*************************************************************CMF ENERGY*******************************************************************","\n\n")

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