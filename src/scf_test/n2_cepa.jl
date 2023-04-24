using ClusterMeanField
using RDM
using QCBase
using PyCall
using Printf
using InCoreIntegrals
using LinearAlgebra
using FermiCG
pyscf=pyimport("pyscf")
np=pyimport("numpy")
symm=pyimport("pyscf.symm")
cepa=[]
cmf=[]
for r in 0:50
    atoms = []
    scale=0.8+r*0.035
    push!(atoms,Atom(1,"N", [0.0, 0.0, 0.0]))
    push!(atoms,Atom(2,"N", [scale, 0.0, 0.0]))

    basis = "6-31G"
    basis="sto3g"

    molecule=@sprintf("%6s %24.16f %24.16f %24.16f \n","N", 0.0, 0.0, 0.0)
    molecule=molecule*@sprintf("%6s %24.16f %24.16f %24.16f \n", "N", scale, 0.0, 0.0)
    print(molecule)
        
    mol = pyscf.gto.Mole(
        atom    =   molecule,
        symmetry=true,
        spin    =   0,
        charge  =   0,
        basis   =   basis)
    mol.build()
    mf = pyscf.scf.ROHF(mol)
    mf.verbose = 4
    mf.conv_tol = 1e-8
    mf.conv_tol_grad = 1e-5
    mf.chkfile = "scf.fchk"
    mf.init_guess = "sad"
    mf.run(max_cycle=200)
    println(" Hartree-Fock Energy: ",mf.e_tot)
    norb =size(mf.mo_coeff)[1]
    #print(mf.mo_coeff)
    println(norb)

    pymol = Molecule(0,1,atoms,basis)
    ClusterMeanField.pyscf_write_molden(pymol, mf.mo_coeff,filename="C_RHF_n2.molden")


    #getting the symmetry based orbitals
    function myocc(mf)    
        mol = mf.mol
        irrep_id = mol.irrep_id
        so = mol.symm_orb
        orbsym = symm.label_orb_symm(mol, irrep_id, so, mf.mo_coeff)
        println(orbsym)
        println(mol.irrep_id)
        println(mol.irrep_name)
    end
    myocc(mf)
    sym_map = Dict()
    for (ir,irname) in enumerate(mol.irrep_id)
        sym_map[irname] = ir
    end
    println(sym_map)


    #getting the clusters for the active space
    clusters = []
    for i in 1:3
        push!(clusters,[])
    end
    println(norb)
    C=mf.mo_coeff   
    #specify frozen core
    frozen= [1,2]  
    c_frozen=C[:,frozen]
    d_frozen=2*c_frozen*c_frozen'
    for i in 1:norb
        name = mol.irrep_name[sym_map[mf.orbsym[i]]]
        if !(i in frozen)
            if name == "A1g"
                push!(clusters[1],i)
            elseif name == "A1u"
                push!(clusters[1],i)
            elseif name == "E1gx" 
                push!(clusters[2],i)
            elseif  name == "E1ux"
                push!(clusters[2],i)
            elseif name == "E1gy" 
                push!(clusters[3],i)
            elseif name == "E1uy"
                push!(clusters[3],i)
            end
        end
    end
        
    println(clusters) 



    #get the active ActiveSpace 

    function c_act()
        C_ordered = zeros(norb,0)
        for (ci,c) in enumerate(clusters)
            C_ordered = hcat(C_ordered, C[:,c])
        end
        return C_ordered
    end
    C_act=c_act()
    pyscf.tools.molden.from_mo(mol, "C_ordered_n2.molden", C_act)

    #specify the clusters and initial fock space electrons

    init_fspace = [(3,3), (1,1), (1,1)]
    #clusters=[(1,2,5,8),(4,6),(3,7)]
    clusters=[(1, 2, 3, 4) ,(5, 6), (7, 8)]
    #clusters=[(1, 2), (3, 4, 5, 10, 11, 14, 17, 18), (7, 8, 13, 15), (6, 9, 12, 16)]
    #clusters=[(1:8), (9:12), (13:16)]
    println(clusters)
    println(init_fspace)
    for ci in clusters
        print(length(ci))
    end

    #pyscf.scf.hf_symm.analyze(mf)
    mol_1=make_pyscf_mole(pymol)
    #making the integrals
    #ints=ClusterMeanField.pyscf_build_ints(pymol,C_act,d_frozen)
    h0 = pyscf.gto.mole.energy_nuc(mol_1)
    h  = pyscf.scf.hf.get_hcore(mol_1)
    j, k = pyscf.scf.hf.get_jk(mol_1, d_frozen, hermi=1)
    h0 += tr(d_frozen * ( h + .5*j - .25*k))

    # now rotate to MO basis
    h = C_act' * h * C_act
    j = C_act' * j * C_act
    k = C_act' * k * C_act
    s= mol_1.intor("int1e_ovlp_sph")
    n_act = size(C_act)[2]
    h2 = pyscf.ao2mo.kernel(mol_1, C_act, aosym="s4",compact=false)
    h2 = reshape(h2, (n_act, n_act, n_act, n_act))
    nact=tr(s*d_frozen*s*C_act*C_act')
    println("the value of nact is",nact)
    if isapprox(abs(nact),0,atol=1e-8) == false
        println(nact)
        display(d_frozen)
        error(" I found embedded electrons in the active space?!")
    end


    h1 = h + j - .5*k
    ints = InCoreInts(h0, h1, h2);
    na=5
    nb=5
    nelec = na + nb
    nuc_energy=mol_1.energy_nuc()


    #cmf section

    clusters = [MOCluster(i,collect(clusters[i])) for i = 1:length(clusters)]
    display(clusters)
    rdm1 = zeros(size(h1))
    println(size(rdm1))
    #d1 = RDM1(n_orb(ints))
    e_cmf, U, d1  = ClusterMeanField.cmf_oo_diis(ints, clusters, init_fspace, RDM1(rdm1, rdm1), verbose=0, maxiter_oo=800,maxiter_ci=400, diis_start=5)
    #e_cmf, U, d1  = ClusterMeanField.cmf_oo(ints, clusters, init_fspace, d1,
                                        #max_iter_oo=100, verbose=0, gconv=1e-6, method="bfgs")
    ClusterMeanField.pyscf_write_molden(pymol,C_act*U,filename="cmf_n2.molden")
    println(e_cmf)
    push!(cmf,e_cmf)

    ints = FermiCG.orbital_rotation(ints,U)
    e_ref = e_cmf - ints.h0
    max_roots = 100
    cluster_bases = FermiCG.compute_cluster_eigenbasis(ints, clusters, verbose=0, max_roots=max_roots,
                                                            init_fspace=init_fspace, rdm1a=d1.a, rdm1b=d1.b, T=Float64)
    clustered_ham = FermiCG.extract_ClusteredTerms(ints, clusters)
    cluster_ops = FermiCG.compute_cluster_ops(cluster_bases, ints);
    FermiCG.add_cmf_operators!(cluster_ops, cluster_bases, ints, d1.a, d1.b);
    ref_fock = FermiCG.FockConfig(init_fspace)

        #BST CMX 
    println("*************************************************************BST-cepa ENERGY*******************************************************************","\n\n")

        
    v = FermiCG.BSTstate(clusters, FockConfig(init_fspace), cluster_bases)
    e_cepa, v_cepa = FermiCG.do_fois_cepa(v, cluster_ops, clustered_ham, thresh_foi=1e-3, max_iter=50, tol=1e-8)
    display(e_cepa+ints.h0)
    push!(cepa,e_cepa+ints.h0)
    println(cmf)
    println(cepa)
end