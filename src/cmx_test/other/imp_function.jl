"""
    function compute_expectation_value(ci_vector::BSTstate{T,N,R}, cluster_ops, clustered_op::FermiCG.ClusteredOperator; nbody) where {T,N,R}
"""
function compute_expectation_value(vector::BSTstate{T,N,R}, cluster_ops, clustered_op::FermiCG.ClusteredOperator; nbody=4) where {T,N,R}
    tmp = deepcopy(vector)
    zero!(tmp)
    build_sigma!(tmp, vector, cluster_ops, clustered_op, nbody=nbody)
    return orth_dot(tmp,vector)
end
"""
    build_sigma!(sigma_vector::BSstate, ci_vector::BSstate, cluster_ops, clustered_ham, nbody=4)
"""
function build_sigma!(sigma_vector::BSstate, ci_vector::BSstate, cluster_ops, clustered_ham; nbody=4)
    #={{{=#

    fold!(sigma_vector)
    fold!(ci_vector)
    for (fock_bra, configs_bra) in sigma_vector
        for (fock_ket, configs_ket) in ci_vector
            fock_trans = fock_bra - fock_ket

            # check if transition is connected by H
            haskey(clustered_ham, fock_trans) == true || continue

            for (config_bra, coeff_bra) in configs_bra
                for (config_ket, coeff_ket) in configs_ket
                

                    for term in clustered_ham[fock_trans]
                      
                        length(term.clusters) <= nbody || continue

                        FermiCG.form_sigma_block!(term, cluster_ops, fock_bra, config_bra, 
                                                  fock_ket, config_ket,
                                                  coeff_bra, coeff_ket)


                    end
                end
            end
        end
    end
    return 
    #=}}}=#
end
function form_sigma_block!(term::ClusteredTerm4B, 
    cluster_ops::Vector{ClusterOps{T}},
    fock_bra::FockConfig, bra::TuckerConfig, 
    fock_ket::FockConfig, ket::TuckerConfig,
    bra_coeffs, ket_coeffs) where {T}
#={{{=#
#display(term)
#println(bra, ket)

c1 = term.clusters[1]
c2 = term.clusters[2]
c3 = term.clusters[3]
c4 = term.clusters[4]
length(fock_bra) == length(fock_ket) || throw(Exception)
length(bra) == length(ket) || throw(Exception)
n_clusters = length(bra)
# 
# make sure inactive clusters are diagonal
for ci in 1:n_clusters
    ci != c1.idx || continue
    ci != c2.idx || continue
    ci != c3.idx || continue
    ci != c4.idx || continue

    fock_bra[ci] == fock_ket[ci] || throw(Exception)
    bra[ci] == ket[ci] || return 0.0 
end

# 
# make sure active clusters are correct transitions 
fock_bra[c1.idx] == fock_ket[c1.idx] .+ term.delta[1] || throw(Exception)
fock_bra[c2.idx] == fock_ket[c2.idx] .+ term.delta[2] || throw(Exception)
fock_bra[c3.idx] == fock_ket[c3.idx] .+ term.delta[3] || throw(Exception)
fock_bra[c4.idx] == fock_ket[c4.idx] .+ term.delta[4] || throw(Exception)

# 
# determine sign from rearranging clusters if odd number of operators
state_sign = compute_terms_state_sign(term, fock_ket) 


#
# op[IKMO,JLNP] = <I|p'|J> h(pqrs) <K|q|L> <M|r|N> <O|s|P>
gamma1 = cluster_ops[c1.idx][term.ops[1]][(fock_bra[c1.idx],fock_ket[c1.idx])][:,bra[c1.idx],ket[c1.idx]]
gamma2 = cluster_ops[c2.idx][term.ops[2]][(fock_bra[c2.idx],fock_ket[c2.idx])][:,bra[c2.idx],ket[c2.idx]]
gamma3 = cluster_ops[c3.idx][term.ops[3]][(fock_bra[c3.idx],fock_ket[c3.idx])][:,bra[c3.idx],ket[c3.idx]]
gamma4 = cluster_ops[c4.idx][term.ops[4]][(fock_bra[c4.idx],fock_ket[c4.idx])][:,bra[c4.idx],ket[c4.idx]]

op = Array{Float64}[]
@tensor begin
    op[J,L,N,P,I,K,M,O] := term.ints[p,q,r,s] * gamma1[p,I,J] * gamma2[q,K,L] * gamma3[r,M,N] * gamma4[s,O,P]  
end
#@tensor begin
#    op[q,r,I,J] := term.ints[p,q,r] * gamma1[p,I,J]
#    op[r,I,J,K,L] := op[q,r,I,J] * gamma2[q,K,L]  
#    op[J,L,N,I,K,M] := op[r,I,J,K,L] * gamma2[r,M,N]  
#end

# possibly cache some of these integrals
# compress this
#    opsize = size(op)
#    op = reshape(op, prod(size(op)[1:4]), prod(size(op)[5:8]))
#    F = svd(op)
#    #display(F.S)
#    for si in 1:length(F.S) 
#        if F.S[si] < 1e-3
#            F.S[si] = 0
#        end
#    end
#    op = F.U * Diagonal(F.S) * F.Vt
#    op = reshape(op,opsize)

# now transpose state vectors and multiply, also, try without transposing to compare
indices = collect(1:n_clusters+1)
indices[c1.idx] = 0
indices[c2.idx] = 0
indices[c3.idx] = 0
indices[c4.idx] = 0
perm,_ = bubble_sort(indices)

length(size(ket_coeffs)) == n_clusters + 1 || error(" tensors should be folded")

n_roots = last(size(ket_coeffs))
ket_coeffs2 = permutedims(ket_coeffs,perm)
bra_coeffs2 = permutedims(bra_coeffs,perm)

dim1 = size(ket_coeffs2)
ket_coeffs2 = reshape(ket_coeffs2, dim1[1]*dim1[2]*dim1[3]*dim1[4], prod(dim1[5:end]))

dim2 = size(bra_coeffs2)
bra_coeffs2 = reshape(bra_coeffs2, dim2[1]*dim2[2]*dim2[3]*dim2[4], prod(dim2[5:end]))

op = reshape(op, prod(size(op)[1:4]),prod(size(op)[5:8]))
if state_sign == 1
    bra_coeffs2 .+= op' * ket_coeffs2
elseif state_sign == -1
    bra_coeffs2 .-= op' * ket_coeffs2
else
    error()
end


ket_coeffs2 = reshape(ket_coeffs2, dim1)
bra_coeffs2 = reshape(bra_coeffs2, dim2)

# now untranspose
perm,_ = bubble_sort(perm)
ket_coeffs2 = permutedims(ket_coeffs2,perm)
bra_coeffs2 = permutedims(bra_coeffs2,perm)

bra_coeffs .= bra_coeffs2
return  
#=}}}=#
end
"""
    orth_dot(ts1::FermiCG.BSTstate, ts2::FermiCG.BSTstate)

Dot product between `ts2` and `ts1`

Warning: this assumes both `ts1` and `ts2` have the same tucker factors for each `TuckerConfig`
Returns vector of dot products
"""
function orth_dot(ts1::BSTstate{T,N,R}, ts2::BSTstate{T,N,R}) where {T,N,R}
    #={{{=#
    overlap = zeros(T,R) 
    for (fock,configs) in ts2
        haskey(ts1, fock) || continue
        for (config,coeffs) in configs
            haskey(ts1[fock], config) || continue
            for r in 1:R
                overlap[r] += sum(ts1[fock][config].core[r] .* ts2[fock][config].core[r])
            end
        end
    end
    return overlap
    #=}}}=#
end
function fill_p_space!(s::BSTstate{T,N,R}, na, nb) where {T,N,R}

    sectors = [] 
    for ci in s.clusters
        sectors_i = []
        for (fock, range) in s.p_spaces[ci.idx].data
            push!(sectors_i, fock)
        end
        push!(sectors, sectors_i)
    end
    for fconfig in Iterators.product(sectors...)
        fi = FockConfig([fconfig...])
        if n_elec_a(fi) == na && n_elec_b(fi) == nb 
            add_fockconfig!(s, fi)
            
            factors = []
            dims = []
            for ci in s.clusters
                dim = length(s.p_spaces[ci.idx][fi[ci.idx]])
                push!(factors, Matrix(1.0I, dim, dim))
                push!(dims, dim)
            end
            factors = tuple(factors...) 

            tconfig = TuckerConfig([s.p_spaces[ci.idx].data[fi[ci.idx]] for ci in s.clusters])
            core = tuple([reshape(ones(prod(dims)), tuple(dims...)) for r in 1:R]...)
            s[fi][tconfig] = Tucker{T,N,R}(core, factors)
        end
    end
end


"""
    build_compressed_1st_order_state(ket_cts::BSTstate{T,N}, cluster_ops, clustered_ham; 
        thresh=1e-7, 
        max_number=nothing, 
        nbody=4) where {T,N}
Apply the Hamiltonian to `v` expanding into the uncompressed space.
This is done only partially, where each term is recompressed after being computed.
Lots of overhead probably from compression, but never completely uncompresses.

#Arguments
- `cts::BSTstate`: input state
- `cluster_ops`:
- `clustered_ham`: Hamiltonian
- `thresh`: Threshold for each HOSVD 
- `max_number`: max number of tucker factors kept in each HOSVD
- `nbody`: allows one to limit (max 4body) terms in the Hamiltonian considered

#Returns
- `v1::BSTstate`

"""

function build_compressed_1st_order_state(ket_cts::BSTstate{T,N,R}, cluster_ops, clustered_ham; 
        thresh=1e-7, 
        max_number=nothing, 
        nbody=4) where {T,N,R}
#={{{=#
    #
    # Initialize data for our output sigma, which we will convert to a
    sig_cts = BSTstate(ket_cts.clusters, OrderedDict{FockConfig{N},OrderedDict{TuckerConfig{N},FermiCG.Tucker{T,N,R}} }(),  ket_cts.p_spaces, ket_cts.q_spaces)

    data = OrderedDict{FockConfig{N}, OrderedDict{TuckerConfig{N}, Vector{Tucker{T,N,R}} } }()

    lk = ReentrantLock()

    #
    #   2body:
    #       term: H(IK,I'K') = h(pq) G1(pII') G3(qKK')     
    #       ket: C(I'J'K')  = c(i'j'k') U1(I'i') U2(J'j') U3(K'k')
    #
    #       sigma: Σ(IJ'K) = h(pq) X1(pIi') U2(J'j') X3(qKk') c(i'j'k')    diagonal in j'
    #           
    #           sigma is quadratic in cluster dimension. We can reduce that sometimes by 
    #           compressing X
    #
    #       X1(pIi') = x1(pii') V1(Ii)   where V1(Ii) are the left singular vectors of X1(I,pi') 
    #                                    such that when dim(p)*dim(i') < dim(I) we get exact reduction
    #       X3(qKk') = x3(qkk') V3(Kk)   
    #                                   
    #       Σ(IJ'K) = σ(ij'k) V1(Ii) U2(J'j') V3(Kk)
    #
    #       at this point, Σ has the form of an hosvd with σ as teh core tensor
    #
    #       σ(ij'k) =  h(pq) x1(pii') x3(qkk') c(i'j'k')
    #
    #
    nscr = 10
    scr = Vector{Vector{Vector{Float64}} }()
    for tid in 1:Threads.nthreads()
        tmp = Vector{Vector{Float64}}() 
        [push!(tmp, zeros(Float64,10000)) for i in 1:nscr]
        push!(scr, tmp)
    end
       
    #for (fock_trans, terms) in clustered_ham
    keys_to_loop = [keys(clustered_ham.trans)...]
        
    @printf(" %-50s%10i\n", "Number of tasks: ", length(keys_to_loop))
    @printf(" %-50s", "Compute tasks: ")
    @time Threads.@threads for fock_trans in keys_to_loop
        for (ket_fock, ket_tconfigs) in ket_cts
            terms = clustered_ham[fock_trans]

            #
            # new fock sector configuration
            sig_fock = ket_fock + fock_trans

            #
            # check that each cluster doesn't have too many/few electrons
            ok = true
            for ci in ket_cts.clusters
                if sig_fock[ci.idx][1] > length(ci) || sig_fock[ci.idx][2] > length(ci)
                    ok = false
                end
                if sig_fock[ci.idx][1] < 0 || sig_fock[ci.idx][2] < 0
                    ok = false
                end
            end
            ok == true || continue

            for term in terms

                #
                # only proceed if current term acts on no more than our requested max number of clusters
                length(term.clusters) <= nbody || continue
                for (ket_tconfig, ket_tuck) in ket_tconfigs

                    #
                    # find the sig TuckerConfigs reached by applying current Hamiltonian term to ket_tconfig.
                    #
                    # For example:
                    #
                    #   [(p'q), I, I, (r's), I ] * |P,Q,P,Q,P>  --> |X, Q, P, X, P>  where X = {P,Q}
                    #
                    #   This this term, will couple to 4 distinct tucker blocks (assuming each of the active clusters
                    #   have both non-zero P and Q spaces within the current fock sector, "sig_fock".
                    #
                    # We will loop over all these destination TuckerConfig's by creating the cartesian product of their
                    # available spaces, this list of which we will keep in "available".
                    #

                    available = [] # list of lists of index ranges, the cartesian product is the set needed
                    #
                    # for current term, expand index ranges for active clusters
                    for ci in term.clusters
                        tmp = []
                        if haskey(ket_cts.p_spaces[ci.idx], sig_fock[ci.idx])
                            push!(tmp, ket_cts.p_spaces[ci.idx][sig_fock[ci.idx]])
                        end
                        if haskey(ket_cts.q_spaces[ci.idx], sig_fock[ci.idx])
                            push!(tmp, ket_cts.q_spaces[ci.idx][sig_fock[ci.idx]])
                        end
                        push!(available, tmp)
                    end


                    #
                    # Now loop over cartesian product of available subspaces (those in X above) and
                    # create the target TuckerConfig and then evaluate the associated terms
                    for prod in Iterators.product(available...)
                        sig_tconfig = [ket_tconfig.config...]
                        for cidx in 1:length(term.clusters)
                            ci = term.clusters[cidx]
                            sig_tconfig[ci.idx] = prod[cidx]
                        end
                        sig_tconfig = TuckerConfig(sig_tconfig)

                        #
                        # the `term` has now coupled our ket TuckerConfig, to a sig TuckerConfig
                        # let's compute the matrix element block, then compress, then add it to any existing compressed
                        # coefficient tensor for that sig TuckerConfig.
                        #
                        # Both the Compression and addition takes a fair amount of work.


#                        if check_term(term, sig_fock, sig_tconfig, ket_fock, ket_tconfig) == false
#       
#                            println()
#                            display(term.delta)
#                            display(sig_fock - ket_fock)
#                        end
                        check_term(term, sig_fock, sig_tconfig, ket_fock, ket_tconfig) || continue


                        bound = calc_bound(term, cluster_ops,
                                           sig_fock, sig_tconfig,
                                           ket_fock, ket_tconfig, ket_tuck,
                                           prescreen=thresh)
                        if bound < sqrt(thresh)
                            #continue
                        end
                        

                        sig_tuck = form_sigma_block_expand(term, cluster_ops,
                                                           sig_fock, sig_tconfig,
                                                           ket_fock, ket_tconfig, ket_tuck,
                                                           max_number=max_number,
                                                           prescreen=thresh)

                        #if (term isa ClusteredTerm2B) && false
                        #    @btime del = form_sigma_block_expand2($term, $cluster_ops,
                        #                                        $sig_fock, $sig_tconfig,
                        #                                        $ket_fock, $ket_tconfig, $ket_tuck,
                        #                                        $scr[Threads.threadid()],
                        #                                        max_number=$max_number,
                        #                                        prescreen=$thresh)
                        #    #del = form_sigma_block_expand2(term, cluster_ops,
                        #    #                                    sig_fock, sig_tconfig,
                        #    #                                    ket_fock, ket_tconfig, ket_tuck,
                        #    #                                    scr[Threads.threadid()],
                        #    #                                    max_number=max_number,
                        #    #                                    prescreen=thresh)
                        #end

                        if length(sig_tuck) == 0
                            continue
                        end
                        if norm(sig_tuck) < thresh 
                            continue
                        end
                       
                        sig_tuck = compress(sig_tuck, thresh=thresh)

    
                        #sig_tuck = compress(sig_tuck, thresh=1e-16, max_number=max_number)

                        length(sig_tuck) > 0 || continue


                        begin
                            lock(lk)
                            try
                                if haskey(data, sig_fock)
                                    if haskey(data[sig_fock], sig_tconfig)
                                        #
                                        # In this case, our sigma vector already has a compressed coefficient tensor.
                                        # Consequently, we need to add these two together

                                        push!(data[sig_fock][sig_tconfig], sig_tuck)
                                        #sig_tuck = add([sig_tuck, sig_cts[sig_fock][sig_tconfig]])
                                        ##sig_tuck = compress(sig_tuck, thresh=thresh, max_number=max_number)
                                        #sig_cts[sig_fock][sig_tconfig] = sig_tuck

                                    else
                                        data[sig_fock][sig_tconfig] = [sig_tuck]
                                        #sig_cts[sig_fock][sig_tconfig] = sig_tuck
                                    end
                                else
                                    #sig_cts[sig_fock] = OrderedDict(sig_tconfig => sig_tuck)
                                    data[sig_fock] = OrderedDict(sig_tconfig => [sig_tuck])
                                end
                            finally
                                unlock(lk)
                            end
                        end

                    end

                end
            end
        end
    end

    @printf(" %-50s", "Add results together: ")
    flush(stdout)
    @time for (fock,tconfigs) in data
        for (tconfig, tuck) in tconfigs
            if haskey(sig_cts, fock)
                sig_cts[fock][tconfig] = compress(nonorth_add(tuck), thresh=thresh)
            else
                sig_cts[fock] = OrderedDict(tconfig => nonorth_add(tuck))
            end
        end
    end
    flush(stdout)

#    # 
#    # project out A space
#    for (fock,tconfigs) in sig_cts 
#        for (tconfig, tuck) in tconfigs
#            if haskey(ket_cts, fock)
#                if haskey(ket_cts[fock], tconfig)
#                    ket_tuck_A = ket_cts[fock][tconfig]
#
#                    ovlp = nonorth_dot(tuck, ket_tuck_A) / nonorth_dot(ket_tuck_A, ket_tuck_A)
#                    tmp = scale(ket_tuck_A, -1.0 * ovlp)
#                    #sig_cts[fock][tconfig] = nonorth_add(tuck, tmp, thresh=1e-16)
#                end
#            end
#        end
#    end
   
  
    # now combine Tuckers, project out reference space and multiply by resolvents
    #prune_empty_TuckerConfigs!(sig_cts)
    #return compress(sig_cts, thresh=thresh)
    return sig_cts
#=}}}=#
end
    
function form_sigma_block!(term::ClusteredTerm4B, 
    cluster_ops::Vector{ClusterOps{T}},
    fock_bra::FockConfig, bra::TuckerConfig, 
    fock_ket::FockConfig, ket::TuckerConfig,
    bra_coeffs, ket_coeffs) where {T}
#={{{=#
#display(term)
#println(bra, ket)

c1 = term.clusters[1]
c2 = term.clusters[2]
c3 = term.clusters[3]
c4 = term.clusters[4]
length(fock_bra) == length(fock_ket) || throw(Exception)
length(bra) == length(ket) || throw(Exception)
n_clusters = length(bra)
# 
# make sure inactive clusters are diagonal
for ci in 1:n_clusters
    ci != c1.idx || continue
    ci != c2.idx || continue
    ci != c3.idx || continue
    ci != c4.idx || continue

    fock_bra[ci] == fock_ket[ci] || throw(Exception)
    bra[ci] == ket[ci] || return 0.0 
end

# 
# make sure active clusters are correct transitions 
fock_bra[c1.idx] == fock_ket[c1.idx] .+ term.delta[1] || throw(Exception)
fock_bra[c2.idx] == fock_ket[c2.idx] .+ term.delta[2] || throw(Exception)
fock_bra[c3.idx] == fock_ket[c3.idx] .+ term.delta[3] || throw(Exception)
fock_bra[c4.idx] == fock_ket[c4.idx] .+ term.delta[4] || throw(Exception)

# 
# determine sign from rearranging clusters if odd number of operators
state_sign = compute_terms_state_sign(term, fock_ket) 


#
# op[IKMO,JLNP] = <I|p'|J> h(pqrs) <K|q|L> <M|r|N> <O|s|P>
gamma1 = cluster_ops[c1.idx][term.ops[1]][(fock_bra[c1.idx],fock_ket[c1.idx])][:,bra[c1.idx],ket[c1.idx]]
gamma2 = cluster_ops[c2.idx][term.ops[2]][(fock_bra[c2.idx],fock_ket[c2.idx])][:,bra[c2.idx],ket[c2.idx]]
gamma3 = cluster_ops[c3.idx][term.ops[3]][(fock_bra[c3.idx],fock_ket[c3.idx])][:,bra[c3.idx],ket[c3.idx]]
gamma4 = cluster_ops[c4.idx][term.ops[4]][(fock_bra[c4.idx],fock_ket[c4.idx])][:,bra[c4.idx],ket[c4.idx]]

op = Array{Float64}[]
@tensor begin
    op[J,L,N,P,I,K,M,O] := term.ints[p,q,r,s] * gamma1[p,I,J] * gamma2[q,K,L] * gamma3[r,M,N] * gamma4[s,O,P]  
end
#@tensor begin
#    op[q,r,I,J] := term.ints[p,q,r] * gamma1[p,I,J]
#    op[r,I,J,K,L] := op[q,r,I,J] * gamma2[q,K,L]  
#    op[J,L,N,I,K,M] := op[r,I,J,K,L] * gamma2[r,M,N]  
#end

# possibly cache some of these integrals
# compress this
#    opsize = size(op)
#    op = reshape(op, prod(size(op)[1:4]), prod(size(op)[5:8]))
#    F = svd(op)
#    #display(F.S)
#    for si in 1:length(F.S) 
#        if F.S[si] < 1e-3
#            F.S[si] = 0
#        end
#    end
#    op = F.U * Diagonal(F.S) * F.Vt
#    op = reshape(op,opsize)

# now transpose state vectors and multiply, also, try without transposing to compare
indices = collect(1:n_clusters+1)
indices[c1.idx] = 0
indices[c2.idx] = 0
indices[c3.idx] = 0
indices[c4.idx] = 0
perm,_ = bubble_sort(indices)

length(size(ket_coeffs)) == n_clusters + 1 || error(" tensors should be folded")

n_roots = last(size(ket_coeffs))
ket_coeffs2 = permutedims(ket_coeffs,perm)
bra_coeffs2 = permutedims(bra_coeffs,perm)

dim1 = size(ket_coeffs2)
ket_coeffs2 = reshape(ket_coeffs2, dim1[1]*dim1[2]*dim1[3]*dim1[4], prod(dim1[5:end]))

dim2 = size(bra_coeffs2)
bra_coeffs2 = reshape(bra_coeffs2, dim2[1]*dim2[2]*dim2[3]*dim2[4], prod(dim2[5:end]))

op = reshape(op, prod(size(op)[1:4]),prod(size(op)[5:8]))
if state_sign == 1
    bra_coeffs2 .+= op' * ket_coeffs2
elseif state_sign == -1
    bra_coeffs2 .-= op' * ket_coeffs2
else
    error()
end


ket_coeffs2 = reshape(ket_coeffs2, dim1)
bra_coeffs2 = reshape(bra_coeffs2, dim2)

# now untranspose
perm,_ = bubble_sort(perm)
ket_coeffs2 = permutedims(ket_coeffs2,perm)
bra_coeffs2 = permutedims(bra_coeffs2,perm)

bra_coeffs .= bra_coeffs2
return  
#=}}}=#
end

function eye!(s::TPSCIstate{T,N,R}) where {T,N,R}
    set_vector!(s, Matrix{T}(I,size(s)))
end
function set_vector!(ts::TPSCIstate{T,N,R}, v::Vector{T}; root=1) where {T,N,R}

    nbasis=length(v)
    length(ts) == length(v) || throw(DimensionMismatch)

    idx = 1
    for (fock, tconfigs) in ts.data
        for (tconfig, coeffs) in tconfigs
            coeffs[root] = v[idx]
            idx += 1
        end
    end
    nbasis == idx-1 || error("huh?", nbasis, " ", idx)
    return
end
"""
    compute_pt2_energy(ci_vector::TPSCIstate{T,N,R}, cluster_ops, clustered_ham::ClusteredOperator; 
        nbody=4, 
        H0="Hcmf",
        E0=nothing, #pass in <0|H0|0>, or compute it
        thresh_foi=1e-8, 
        prescreen=true,
        verbose=1) where {T,N,R}
"""
function compute_pt2_energy(ci_vector_in::TPSCIstate{T,N,R}, cluster_ops, clustered_ham::ClusteredOperator; 
        nbody=4, 
        H0="Hcmf",
        E0=nothing, #pass in <0|H0|0>, or compute it
        thresh_foi=1e-9, 
        prescreen=true,
        verbose=1) where {T,N,R}
    #={{{=#

    println()
    println(" |........................do batched PT2............................")
    println(" thresh_foi    :", thresh_foi   ) 
    println(" prescreen     :", prescreen   ) 
    println(" H0            :", H0   ) 
    println(" nbody         :", nbody   ) 

    e2 = zeros(T,R)
   
    ci_vector = deepcopy(ci_vector_in)
    clusters = ci_vector.clusters
    norms = norm(ci_vector);
    @printf(" Norms of input states:\n")
    [@printf(" %12.8f\n",i) for i in norms]
    orthonormalize!(ci_vector)
    
    clustered_ham_0 = extract_1body_operator(clustered_ham, op_string = H0) 
    if E0 == nothing
        @printf(" %-50s", "Compute <0|H0|0>:")
        @time E0 = compute_expectation_value_parallel(ci_vector, cluster_ops, clustered_ham_0)
        #E0 = diag(E0)
        flush(stdout)
    end
    @printf(" %-50s", "Compute <0|H|0>:")
    @time Evar = compute_expectation_value_parallel(ci_vector, cluster_ops, clustered_ham)
    #Evar = diag(Evar)
    flush(stdout)


    # 
    # define batches (FockConfigs present in resolvant)
    jobs = Dict{FockConfig{N},Vector{Tuple}}()
    for (fock_ket, configs_ket) in ci_vector.data
        for (ftrans, terms) in clustered_ham
            fock_x = ftrans + fock_ket

            #
            # check to make sure this fock config doesn't have negative or too many electrons in any cluster
            all(f[1] >= 0 for f in fock_x) || continue 
            all(f[2] >= 0 for f in fock_x) || continue 
            all(f[1] <= length(clusters[fi]) for (fi,f) in enumerate(fock_x)) || continue 
            all(f[2] <= length(clusters[fi]) for (fi,f) in enumerate(fock_x)) || continue 
           
            job_input = (terms, fock_ket, configs_ket)
            if haskey(jobs, fock_x)
                push!(jobs[fock_x], job_input)
            else
                jobs[fock_x] = [job_input]
            end
            
        end
    end

    #
    # prepare scratch arrays to help cut down on allocation in the threads
    jobs_vec = []
    for (fock_x, job) in jobs
        push!(jobs_vec, (fock_x, job))
    end

    scr_f = Vector{Vector{Vector{T}} }()
    scr_i = Vector{Vector{Vector{Int16}} }()
    scr_m = Vector{Vector{MVector{N,Int16}} }()
    nscr = 20 

    scr1 = Vector{Vector{T}}()
    scr2 = Vector{Vector{T}}()
    scr3 = Vector{Vector{T}}()
    scr4 = Vector{Vector{T}}()
    tmp1 = Vector{MVector{N,Int16}}()
    tmp2 = Vector{MVector{N,Int16}}()

    e2_thread = Vector{Vector{T}}()
    for tid in 1:Threads.nthreads()
        push!(e2_thread, zeros(T, R))
        push!(scr1, zeros(T, 1000))
        push!(scr2, zeros(T, 1000))
        push!(scr3, zeros(T, 1000))
        push!(scr4, zeros(T, 1000))
        push!(tmp1, zeros(Int16,N))
        push!(tmp2, zeros(Int16,N))

        tmp = Vector{Vector{T}}() 
        [push!(tmp, zeros(T,10000)) for i in 1:nscr]
        push!(scr_f, tmp)

        tmp = Vector{Vector{Int16}}() 
        [push!(tmp, zeros(Int16,10000)) for i in 1:nscr]
        push!(scr_i, tmp)

        tmp = Vector{MVector{N,Int16}}() 
        [push!(tmp, zeros(Int16,N)) for i in 1:nscr]
        push!(scr_m, tmp)
    end



    println(" Number of jobs:    ", length(jobs_vec))
    println(" Number of threads: ", Threads.nthreads())
    BLAS.set_num_threads(1)
    flush(stdout)


    tmp = Int(round(length(jobs_vec)/100))
    verbose < 1 || println("   |----------------------------------------------------------------------------------------------------|")
    verbose < 1 || println("   |0%                                                                                              100%|")
    verbose < 1 || print("   |")
    #@profilehtml @Threads.threads for job in jobs_vec
    t = @elapsed begin
        #@qthreads for job in jobs_vec
        #@time for job in jobs_vec
        
        @Threads.threads for (jobi,job) in collect(enumerate(jobs_vec))
        #for (jobi,job) in collect(enumerate(jobs_vec))
            fock_bra = job[1]
            tid = Threads.threadid()
            e2_thread[tid] .+= _pt2_job(job[2], fock_bra, cluster_ops, nbody, thresh_foi,  
                                        scr_f[tid], scr_i[tid], scr_m[tid],  prescreen, verbose, 
                                        ci_vector, clustered_ham_0, E0)
            if verbose > 0
                if  jobi%tmp == 0
                    print("-")
                    flush(stdout)
                end
            end
        end
    end
    verbose < 1 || println("|")
    flush(stdout)
   
    @printf(" Time spent computing E2 %12.1f (s)\n",t)
    e2 = sum(e2_thread) 

    #BLAS.set_num_threads(Threads.nthreads())

    @printf(" %5s %12s %12s\n", "Root", "E(0)", "E(2)") 
    for r in 1:R
        @printf(" %5s %12.8f %12.8f\n",r, Evar[r], Evar[r] + e2[r])
    end
    println(" ..................................................................|")

    return e2
end
#=}}}=#
#e2 = FermiCG.compute_pt2_energy(v0b, cluster_ops, clustered_ham, thresh_foi=1e-8);