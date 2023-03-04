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

r = 0.5
a = 1
push!(atoms,Atom(1,"H", [0, 0*a, 0*r]))
push!(atoms,Atom(2,"H", [0, 0*a, 1*r]))
push!(atoms,Atom(3,"H", [0, 1*a,2*r]))
push!(atoms,Atom(4,"H", [0, 1*a,3*r]))
push!(atoms,Atom(5,"H", [0, 2*a, 4*r]))
push!(atoms,Atom(6,"H", [0, 2*a, 5*r]))
push!(atoms,Atom(7,"H", [0, 3*a, 6*r]))
push!(atoms,Atom(8,"H", [0, 3*a, 7*r]))
#push!(atoms,Atom(9,"H", [0, 4*a, 8*r]))
#push!(atoms,Atom(10,"H",[0, 4*a, 9*r]))
#push!(atoms,Atom(11,"H",[0, 5*a, 10*r]))
#push!(atoms,Atom(12,"H",[0, 5*a, 11*r]))
println(atoms)



basis = "sto-3g"
mol     = Molecule(0,1,atoms,basis)

nroots = 1

n_steps = 80    
step_size = .022
energies_cmx = []
energies_cmf=[]
energies_fci=[]
energies_pt2=[]
io = open("traj_H8_Intra.xyz", "w");

for R in 1:n_steps
    println("GEOMETRY NO",R,"\n")
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
    #clusterlist    = [(1:2),(3:4),(5:6),(7:8),(9:10),(11:12)]
    #init_fspace = [(1,1),(1,1),(1,1),(1,1),(1,1),(1,1)]
    clusterlist    = [(1:2),(3:4),(5:6),(7:8)]
    init_fspace = [(1,1),(1,1),(1,1),(1,1)]
    #clusters    = [(1:4),(5:8),(9:12)]
    #init_fspace = [(2,2),(2,2),(2,2)]
    na = 4
    nb = 4

    nroots = 1

    # get integrals
    mf = FermiCG.pyscf_do_scf(pymol)
    nbas = size(mf.mo_coeff)[1]
    ints = FermiCG.pyscf_build_ints(mol,mf.mo_coeff, zeros(nbas,nbas));
    #=pyscf = pyimport("pyscf")
    pyscf.lib.num_threads(1)
    fci = pyimport("pyscf.fci")
    cisolver = fci.FCI(mf)
    cisolver.max_cycle = 200
    cisolver.conv_tol = 1e-8
    nelec = na + nb
    norb = size(ints.h1,1)
    e_fci, v_fci = cisolver.kernel(ints.h1, ints.h2, norb, nelec, ecore=0, nroots =nroots,verbose=8)
    push!(energies_fci,e_fci)=#
    nelec = na + nb
    norb = size(ints.h1,1)
    ansatz = FCIAnsatz(norb, na, nb)
    solver = SolverSettings(nroots=1, package="Arpack")
    solution = solve(ints, ansatz, solver)
    push!(energies_fci,solution.energies[1])
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
    clusters = [MOCluster(i,collect(clusterlist[i])) for i = 1:length(clusterlist)]
    display(clusters)

    rdm1 = zeros(size(ints.h1))
    #d1 = RDM1(n_orb(ints))
    e_cmf, U, d1  = ClusterMeanField.cmf_oo_diis(ints, clusters, init_fspace, RDM1(rdm1, rdm1),maxiter_oo = 200, maxiter_ci   = 200, maxiter_d1= 200 ,verbose=0, diis_start=3)
    #e_cmf, U, d1  = FermiCG.cmf_oo(ints, clusters, init_fspace, d1,
                                #max_iter_oo=40, verbose=0, gconv=1e-6, method="bfgs")
    push!(energies_cmf,e_cmf)
    FermiCG.pyscf_write_molden(mol,Cl*U,filename="cmf.molden")
    ints = FermiCG.orbital_rotation(ints,U)

    e_ref = e_cmf - ints.h0
    max_roots = 100
    cluster_bases = FermiCG.compute_cluster_eigenbasis(ints, clusters, verbose=0, max_roots=max_roots,
                                                    init_fspace=init_fspace, rdm1a=d1.a, rdm1b=d1.b, T=Float64)


    clustered_ham = FermiCG.extract_ClusteredTerms(ints, clusters)

    #
    cluster_ops = FermiCG.compute_cluster_ops(cluster_bases, ints);

    FermiCG.add_cmf_operators!(cluster_ops, cluster_bases, ints, d1.a, d1.b);

    ref_fock = FermiCG.FockConfig(init_fspace)


    ψ = FermiCG.BSTstate(clusters, FockConfig(init_fspace), cluster_bases)
    display(ψ)
    ept2 ,total_pt2= FermiCG.compute_pt2_energy(ψ, cluster_ops, clustered_ham, thresh_foi=1e-6)
    #println("the value of pt2 correction energy value is",total_pt2)
    push!(energies_pt2,total_pt2[1])
    σ = FermiCG.build_compressed_1st_order_state(ψ, cluster_ops, clustered_ham, nbody=4, thresh=1e-4)
    σ = FermiCG.compress(σ, thresh=1e-4)

    #H = FermiCG.nonorth_dot(ψ,σ)
    H1 = FermiCG.compute_expectation_value(ψ, cluster_ops, clustered_ham)

    H2 = FermiCG.orth_dot(σ,σ)
    H3 = FermiCG.compute_expectation_value(σ, cluster_ops, clustered_ham)
    #compute_expectation_value is not working; dimension mismatch

    sigma2 = FermiCG.build_compressed_1st_order_state(σ, cluster_ops, clustered_ham, nbody=4, thresh=1e-4)
    sigma2_compressed = FermiCG.compress(sigma2, thresh=1e-4)

    H4 = FermiCG.orth_dot(sigma2_compressed,sigma2_compressed)
    H5 = FermiCG.compute_expectation_value(sigma2_compressed, cluster_ops, clustered_ham)
    I_1=H1[1]
    I_2=H2[1]-I_1*H1[1]
    I_3=H3[1]-I_1*H2[1]-2*I_2*H1[1]
    I_4=H4[1]-I_1*H3[1]-3*I_2*H2[1]-3*I_3*H1[1]
    I_5=H5[1]-I_1*H4[1]-4*I_2*H3[1]-6*I_3*H2[1]-4*I_4*H1[1]
    I_VEC=[I_2; I_3]
    I_MAT=[I_3 I_4;I_4 I_5]


    COR_ENERGY=inv(I_MAT)*I_VEC
    COR_ENERGY=I_VEC'*COR_ENERGY
    E_pk2=I_1-COR_ENERGY
    println(E_pk2)
    push!(energies_cmx,E_pk2)
    println(energies_cmx)
    println(energies_pt2)
    println(energies_cmf)
    println(energies_fci)
end
close(io)
#plot(energies_cmx)
#x = range(1, 50)
#plot(x, [energies_fci energies_pt2 energies_cmx],title="Energy of H12 ", label=[ "fci energies" "pt2_energies" "cmx energies"], linewidth=2)
#savefig("plot_cls.png")



#stepsize=0.05,a=1,r=1
#energies_cmx=[-18.32377764,-18.429516887590935, -18.53899939022807, -18.647132589154538, -18.750233054905717, -18.84576086664905, -18.932090500302262, -19.008301081649343, -19.074018415082932, -19.127048076058404, 
#                 -19.172625534542416, -19.20848524577694, -19.235413540443727, -19.25427001571972, -19.265942939914225,
#                        -19.271340822877754, -19.271361625490552, -19.266876001527407, -19.2587218667926, -19.2476714930382, -19.234478641389643]
#r=1,a=2
#fci_energies=[-16.435705907688632, -16.529350869256284, -16.6179852833884, -16.700174723578446, -16.77506000757202, -16.842203280899156, -16.90147149424638, -16.952950117530552, -16.996879934706374, -17.033610844823336, -17.06356791800125, -17.087226174187943, -17.105091532398827, -17.117686180617955, -17.125537108642014, -17.12916698671086, -17.129086845420385, -17.125790149179505, -17.11974803501789, -17.111405518672616, -17.10117857792783, -17.08945201394952, -17.076578052212046, -17.0628756038471, -17.048630376362553, -17.034096057125968, -17.02895191268254, -17.04089929634466, -17.051497090033372, -17.06088595787361, -17.069194955319734, -17.076541716229226, -17.083032891976465, -17.08876493254393, -17.093824623206153, -17.098289657710595, -17.102229702235324, -17.10571076296043, -17.108773581712956, -17.111482245696944]
#cmx_energies=[-16.435468066530618, -16.52912322848627, -16.61776227563748, -16.699952673655964, -16.774836816763923, -16.841977030685257, -16.90123697487706, -16.952703006762732, -16.996618905108566, -17.03333240551526, -17.06326792876692, -17.08689986991106, -17.104733311551424, -17.117290260858955, -17.125095495028194, -17.12867139519523, -17.128526954594513, -17.12515509471737, -17.119024381021838, -17.110578969094547, -17.100232482433828, -17.088367954102633, -17.075335046105202, -17.05764865460532, -17.062580478246925, -17.02383463411204, -17.00836188424628, -16.992917585494215, -16.977769203518903, -16.96290422788532, -16.94853398514708, -16.93457932602752, -16.92125313337945, -16.912439014378915, -16.900696812592265, -16.88967355873963, -16.87953595580127, -16.869928791610917, -16.861008269775503, -16.85084279186263]