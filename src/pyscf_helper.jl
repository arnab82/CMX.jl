using PyCall
using QCBase
using FermiCG
function lowdin(S::Matrix)
    println("Using Lowdin orthogonalized orbitals")
    #forming S^-1/2 to transform to A and B block.
    sal, svec = eigen(S)
    idx = sortperm(sal, rev=true)
    sal = sal[idx]
    svec = svec[:, idx]
    sal = sal.^(-0.5)
    sal = diagm(sal)
    X = svec * sal * svec'
    return X
end
function init_mol(molecule::Molecule,nelec::Int64,cas_nstop::Int64,cas_nstart::Int64,loc_nstart::Int64,loc_nstop::Int64,cas=false,orb_basis="scf",conv_tol=1e-10)
    pyscf=pyimport("pyscf")
    gto=pyimport("pyscf.gto")
    scf =pyimport("pyscf.scf")
    ao2mo=pyimport("pyscf.ao2mo") 
    lo=pyimport("pyscf.lo")
    tools=pyimport("pyscf.tools")
    symm=pyimport("pyscf.symm")
    molden=tools.molden
    pyscf.lib.num_threads(1)
    pymol=pyscf.gto.Mole()
    pymol.basis= molecule.basis
    geomstr = ""
    for i in molecule.atoms
        geomstr = geomstr * string(i.symbol,", ", join(map(string, i.xyz), ", "),"\n")
    end
    pymol.atom = geomstr
    pymol.charge = molecule.charge
    pymol.spin = molecule.multiplicity-1
    pymol.build()
    if pymol.symmetry==true
        println("symmetry")
        irrep_id = pymol.irrep_id
        so = pymol.symm_orb
        orbsym = symm.label_orb_symm(pymol, irrep_id, so, mf.mo_coeff)
        println(orbsym)
        println(pymol.irrep_id)
        println(pymol.irrep_name)
        println(pymol.topgroup)
    end

    #   SCF
    mf = pyscf.scf.RHF(pymol).run(conv_tol=conv_tol, verbose=4)
    enu = mf.energy_nuc()
    println("MO Energies")
    display(mf.mo_energy)

    #orbitals and electrons
    C=mf.mo_coeff
    norb=size(C[1])
    n_a=nelec/2
    n_b=n_a
    if cas== true
        cas_norb=cas_nstop-cas_nstart
        mcscf=pyimport("pyscf.mcscf")
    else
        cas_nstart=0
        cas_nstop=n_orb
        cas_nel=nelec
    end
    if orb_basis=="scf"
        println("\n using the cannonical Hartree Fock Orbitals..\n")
        println(" C shape")
        println(size(C))


    elseif orb_basis=="lowdin"    
        @assert cas==false
        S=pymol.intor("int1e_ovlp_sph")
        Cl=lowdin(S)
    elseif orb_basis=="boys"
        pyscf.lib.num_threads(1)
        cl_c = mf.mo_coeff[:, 1:cas_nstart]
        cl_a = lo.Boys(pymol, mf.mo_coeff[:, cas_nstart:cas_nstop]).kernel(verbose=4)
        cl_v = mf.mo_coeff[:, cas_nstop+1:end]
        Cl = hcat(cl_c, cl_a, cl_v)
    elseif orb_basis=="pm"
        pyscf.lib.num_threads(1)  
        cl_c = mf.mo_coeff[:, 1:cas_nstart]
        cl_a = lo.PM(pymol, mf.mo_coeff[:, cas_nstart:cas_nstop]).kernel(verbose=4)
        cl_v = mf.mo_coeff[:, cas_nstop+1:end]
        Cl = hcat(cl_c, cl_a, cl_v)
    elseif orb_basis=="ibmo"#showing error in this section
        loc_vstop = loc_nstop - n_a
        println(loc_vstop)
        println(loc_nstart)
        mo_occ = mf.mo_coeff[:, mf.mo_occ .> 0]
        println(size(mo_occ))
        mo_vir = mf.mo_coeff[:, mf.mo_occ .== 0]
        println(size(mo_vir))
        c_core = mo_occ[:, :loc_nstart]
        println(size(c_core))
        println(size(mo_occ[:, loc_nstart:end]))
        iao_occ = lo.iao.iao(mol, mo_occ[:, loc_nstart:end])
        iao_vir = lo.iao.iao(mol, mo_vir[:,:loc_vstop])
        c_out = mo_vir[:, loc_vstop:end]

        # Orthogonalize IAO
        iao_occ = lo.vec_lowdin(iao_occ, mf.get_ovlp())
        iao_vir = lo.vec_lowdin(iao_vir, mf.get_ovlp())

        #
        # Method 1, using Knizia's alogrithm to localize IAO orbitals
        #
        """
        Generate IBOS from orthogonal IAOs
        """
        ibo_occ = lo.ibo.ibo(mol, mo_occ[:, loc_nstart:end], iaos=iao_occ)
        ibo_vir = lo.ibo.ibo(mol, mo_vir[:, :loc_vstop], iaos=iao_vir)

        Cl = hcat(c_core, ibo_occ, ibo_vir, c_out)

    else
        println("No orbital basis is defined")
    end
end

atoms = []
r0 = 0.0
push!(atoms,Atom(1,"C", [0.63, 0.63, 0.63]))
push!(atoms,Atom(2,"H", [1.26, 1.26, 0]))
push!(atoms,Atom(3,"H", [0, 1.26,1.26]))
push!(atoms,Atom(4,"H", [1.26,0,1.26]))
push!(atoms,Atom(5,"H", [r0, r0,r0]))
println(atoms)

mol = Molecule(0,1,atoms,basis)
init_mol(mol,10,10,1,1,10,true,"boys",1e-6)




function localize_scf(mf,na,cas_nstart,cas_nstop)
    D = mf.make_rdm1(mo_coeff=C)
    S = mf.get_ovlp()
    sal, svec = np.linalg.eigh(S)
    idx = sortperm(sal, rev=true)
    sal = sal[idx]
    svec = svec[:, idx]
    sal = sal.^-0.5
    sal = np.diagflat(sal)
    X = svec *sal *svec'
    C_ao2mo = np.linalg.inv(X) * C
    Cocc = C_ao2mo[:,1:na]
    D = Cocc*Cocc'
    DMO = C_ao2mo'*D*C_ao2mo
    #only for cas space
    DMO = DMO[cas_nstart:cas_nstop,cas_nstart:cas_nstop]
    return DMO
end
function mulliken_ordering(mol, norb, C)
    S = mol.intor("int1e_ovlp_sph")
    mulliken = zeros((mol.natm, norb))
    for i in 1:norb
        Cocc = reshape(C[:, i], C.shape[1], 1)
        temp = Cocc * Cocc' * S
        for (m, lb) in enumerate(mol.ao_labels())
            #println(lb)
            v1, v2, v3 = split(lb)
            #println(v1)
            mulliken[parse(Int, v1), i] += temp[m, m]
        end
    end
    println(mulliken)
    return mulliken
end
function get_eff_for_casci(n_start, n_stop, h, g)
    const = 0.0
    for i in 1:n_start
        const += 2 * h[i, i]
        for j in 1:n_start
            const += 2 * g[i, i, j, j] - g[i, j, i, j]
        end
    end

    eff = zeros((n_stop - n_start, n_stop - n_start))
    for l in n_start:n_stop-1
        L = l - n_start + 1
        for m in n_start:n_stop-1
            M = m - n_start + 1
            for j in 1:n_start
                eff[L, M] += 2 * g[l+1, m+1, j, j] - g[l+1, j, j, m+1]
            end
        end
    end
    return const, eff
end
function ordering_diatomics(mol, C, basis_set)
    ##DZ basis diatomics reordering with frozen 1s

    if basis_set == "6-31g"
        orb_type = ["s", "pz", "px", "py"]
    elseif basis_set == "ccpvdz"
        orb_type = ["s", "pz", "dz", "px", "dxz", "py", "dyz", "dx2-y2", "dxy"]
    else
        println("clustering not general yet")
        exit()
    end

    ref = zeros(C.shape[2])

    ## Find dimension of each space
    dim_orb = []
    for orb in orb_type
        println("Orb type", orb)
        idx = 0
        for label in mol.ao_labels()
            if occursin(orb, label)
                idx += 1
            end
        end
        ##frozen 1s orbitals
        if orb == "s"
            idx -= 2
        end
        push!(dim_orb, idx)
        println(idx)
    end

    new_idx = []
    ## Find orbitals corresponding to each orb space
    for (i, orb) in enumerate(orb_type)
        println("Orbital type:", orb)
        s_pop = mo_mapping.mo_comps(orb, mol, C)
        #println(s_pop)
        ref += s_pop
        cas_list = sortperm(s_pop)[end-dim_orb[i]+1:end]
        cas_list = sort(cas_list)
        println("cas_list", cas_list)
        push!(new_idx, cas_list...)
        #println(orb,' population for active space orbitals', s_pop[cas_list])
    end

    ao_labels = mol.ao_labels()
    #idx = search_ao_label(mol, ["N.*s"])
    #for i in idx
    #    println(i, ao_labels[i])
    #end
    println(ref)
    println(new_idx)

    @assert length(new_idx) == length(Set(new_idx))
    return new_idx
end
function get_pi_space(mol, mf, cas_norb, cas_nel, local=true, p3=false)
    pyscf=pyimport("pyscf")
    np=pyimport("numpy")
    symm=pyimport("pyscf.symm")
    mcscf=pyimport("pyscf.mcscf")
    mo_mapping=pyimport("pyscf.mo_mapping")
    lo=pyimport("pyscf.lo")
    ao2mo=pyimport("pyscf.ao2mo")
    scf=pyimport("pyscf.scf")
    # find the 2pz orbitals using mo_mapping
    ao_labels = ["C 2pz"]

    # get the 3pz and 2pz orbitals
    if p3
        ao_labels = ["C 2pz", "C 3pz"]
        cas_norb = 2 * cas_norb
    end

    pop = mo_mapping.mo_comps(ao_labels, mol, mf.mo_coeff)
    cas_list = sort(pop.argsort()[(end-cas_norb+1):end])  #take the 2z orbitals and resort in MO order
    println("Population for pz orbitals", pop[cas_list])
    mo_occ = findall(mf.mo_occ .> 0)
    focc_list = setdiff(mo_occ, cas_list)
    focc = length(focc_list)

    # localize the active space
    if local
        cl_a = lo.Boys(mol, mf.mo_coeff[:, cas_list]).kernel(verbose=4)
        C = mf.mo_coeff
        C[:, cas_list] = cl_a
    else
        C = mf.mo_coeff
        mo_energy = mf.mo_energy[cas_list]
        J, K = mf.get_jk()
        K = K[cas_list, :][:, cas_list]
        println(K)

        if mol.symmetry
            mo = symm.symmetrize_orb(mol, C[:, cas_list])
            osym = symm.label_orb_symm(mol, mol.irrep_name, mol.symm_orb, mo)
            #symm.addons.symmetrize_space(mol, mo, s=None, check=True, tol=1e-07)
            for i in 1:length(osym)
                @printf("%4d %8s %16.8f\n", i, osym[i], mo_energy[i])
            end
        end
    end

    # reorder the orbitals to get docc, active, vir ordering
    mycas = mcscf.CASCI(mf, cas_norb, cas_nel)
    C = mycas.sort_mo(cas_list .+ 1, mo_coeff=C)
    np.save("C.npy", C)

    # Get the active space integrals and the frozen core energy
    h, ecore = mycas.get_h1eff(C)
    g = ao2mo.kernel(mol, C[:, focc:focc+cas_norb], aosym="s4", compact=false).reshape(4*cas_norb, 4*cas_norb)
    C = C[:, focc:focc+cas_norb]  #only carrying the active space orbs
    return h, ecore, g, C
end
function reorder_integrals(idx::Array{Int64,1}, h::Array{Float64,2}, g::Array{Float64,4})
    h = h[:, idx]
    h = h[idx, :]
    g = g[:, :, :, idx]
    g = g[:, :, idx, :]
    g = g[idx, :, :, :]
    g = g[:, idx, :, :]
    return h, g
end
