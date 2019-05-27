import Base.==
using LinearAlgebra: eigen, eigvals, eigvecs
using DataStructures: Queue, isempty, enqueue!, dequeue!

###############################################################################
# Permutation
###############################################################################
struct Permutation{T<:Integer}
    d::Vector{T}

    # identity
    function Permutation(n::T) where T<:Integer
        v = collect(T, 1:n)
        new{T}(v)
    end

    function Permutation(v::Vector{T}, check::Bool=true) where T<:Integer
        if check
            if !isperm(v)
                error("invalid sequence for Permutation: $v")
            end
        end
        new{T}(v)
    end
end

Base.getindex(g::Permutation{T}, n::T) where T<:Integer = Base.getindex(g.d, n)
Base.length(g::Permutation{T}) where T<:Integer = Base.length(g.d)
Base.hash(g::Permutation{T}, h::UInt) where T<:Integer = Base.hash(g.d, h)
==(g::Permutation{T}, h::Permutation{T}) where T<:Integer = g.d == h.d
inverse(g::Permutation) = Permutation(invperm(g.d))
conjugate(g::Permutation, h::Permutation) = h * g * inverse(h)

"""
    *(g::Permutation{T}, h::Permutation{T})

compute composition of two Permutations, gh(i) = g(h(i))
"""
function Base.:*(g::Permutation{T}, h::Permutation{T})::Permutation{T} where T<:Integer
    d = similar(g.d)
    @inbounds for i in 1:length(d)
        d[i] = g[h[i]]
    end
    return Permutation(d, false)
end

###############################################################################
# Permutaion Group
###############################################################################

struct PermutationGroup{T<:Integer}
    generators::Vector{Permutation{T}}
    elements::Vector{Permutation{T}}

    function PermutationGroup(generators::Vector{Permutation{T}}) where T<:Integer
        dim = length(generators[1])
        elements = []

        que = Queue{Permutation{T}}()
        enqueue!(que, Permutation(dim))
        while !isempty(que)
            p = dequeue!(que)
            if p in elements
                continue
            end
            push!(elements, p)
            for gen in generators
                enqueue!(que, gen * p)
            end
        end
        new{T}(generators, elements)
    end
end

Base.getindex(G::PermutationGroup{T}, n::T) where T<:Integer = Base.getindex(G.elements, n)

order(g::PermutationGroup) = length(g.elements)

function get_conjugate_classes(pg::PermutationGroup{T}) where T<:Integer
    classes = []
    visited = []

    for g in pg.elements
        if g in visited
            continue
        end
        g_class = unique([conjugate(g, h) for h in pg.elements])
        append!(visited, g_class)
        push!(classes, g_class)
    end

    return classes
end

function get_multiplication_matrix(classes)
    numclasses = length(classes)
    class_constants = zeros(Int64, (numclasses, numclasses, numclasses))
    for (i, ci) in enumerate(classes)
        for (j, cj) in enumerate(classes)
            list_perm = [permi * permj for permi in ci for permj in cj]
            for (k, ck) in enumerate(classes)
                class_constants[i, j, k] = count(x -> x == ck[1], list_perm)
            end
        end
    end
    return class_constants
end

###############################################################################
# Representation
###############################################################################
"""
    regular_representation(G)

Compute regular representation of group G, Gamma_{jk}(R_{i})

# Returns
reg: reg[i, j, k] == Gamma_{jk}(R_{i}) s.t. R_{i} R_{k} = sum_{j} R_{j} Gamma_{jk}(R_{i})
"""
function regular_representation(G::PermutationGroup{T}) where T <:Integer
    g = order(G)
    reg = zeros(T, (g, g, g))
    for i in 1:g
        for k in 1:g
            Rj = G[i] * G[k]
            for j in 1:g
                if G[j] == Rj
                    reg[i, j, k] = 1
                    break
                end
            end
        end
    end
    return reg
end


###############################################################################
# Character
###############################################################################
"""
    character_burnside(classes)

Compute character table for Irreps of finite group with Burnside's method

# Returns
characters: characters[alpha, i] is character of i-th class in alpha-th Irreps.
"""
function character_burnside(classes)
    class_constants = get_multiplication_matrix(classes)
    numclasses = size(class_constants)[1]
    eigenspaces = []
    # classes[1] is assumed to be identity
    for i in 2:numclasses
        for j in i:numclasses
            C = class_constants[i, :, :] + class_constants[j, :, :]
            vals, vecs = eigen(C)

            # TODO: more robust way to compare eigenvalues
            vals = round.(vals, digits=6)
            vals = map(x -> (abs(x) < 1e-8) ? 0 : x, vals)
            vecs = map(x -> (abs(x) < 1e-8) ? 0 : x, vecs)

            d = group_eigen(vals, vecs)

            for vec in values(d)
                # degenerated eigenspace
                if size(vec)[1] > 1
                    continue
                end
                # normalize eigenvector
                nvec = vec ./ vec[:, 1]
                push!(eigenspaces, nvec)
            end
        end
    end

    # dirac[alpha, i] is dirac character of i-th class in alpha-th Irreps.
    dirac = vcat(unique_approx(eigenspaces)...)
    @assert size(dirac)[1] == numclasses

    nc = map(length, classes)
    g = sum(nc)
    dims = zeros(Int64, numclasses)
    for alpha in 1:numclasses
        dims[alpha] = sqrt(round(g / sum(dirac[alpha, :] .* conj(dirac[alpha, :]) ./ nc)))
    end

    characters = dirac .* dims ./ reshape(nc, 1, numclasses)
    return characters
end

function group_eigen(vals, vecs, rtol::Real=sqrt(eps()))
    dim = size(vals)[1]
    d = Dict()
    for j in 1:dim
        v = vals[j]
        # vecs[:, j] is the j-th eigenvector
        vec = reshape(vecs[:, j], 1, :)

        isexists = false
        for key in keys(d)
            if isapprox(key, v, rtol=rtol)
                isexists = true
                d[key] = vcat(d[key], vec)
                break
            end
        end
        if !isexists
            d[v] = vec
        end
    end
    return d
end

function unique_approx(itr, rtol::Real=sqrt(eps()))
    ret = []
    for x in itr
        if isempty(ret)
            push!(ret, x)
        else
            if all(map(y -> !isapprox(y, x, rtol=rtol), ret))
                push!(ret, x)
            end
        end
    end
    return ret
end


function test_character()
    println("C3v")
    C3v = PermutationGroup([Permutation([2, 3, 1]), Permutation([2, 1, 3])])
    classes = get_conjugate_classes(C3v)
    characters = character_burnside(classes)
    for alpha in 1:size(classes)[1]
        println(characters[alpha, :])
    end

    println("C4v")
    C4v = PermutationGroup([Permutation([2, 3, 4, 1]),
                            Permutation([2, 1, 4, 3])])
    classes = get_conjugate_classes(C4v)
    characters = character_burnside(classes)
    for alpha in 1:size(classes)[1]
        println(characters[alpha, :])
    end

    println("C6v")
    C6v = PermutationGroup([Permutation([2, 3, 4, 5, 6, 1]),
                            Permutation([6, 5, 4, 3, 2, 1])])
    classes = get_conjugate_classes(C6v)
    characters = character_burnside(classes)
    for alpha in 1:size(classes)[1]
        println(characters[alpha, :])
    end

    println("D3h")
    D3h = PermutationGroup([Permutation([2, 3, 1, 5, 6, 4]),
                            Permutation([4, 6, 5, 1, 3, 2]),
                            Permutation([4, 5, 6, 1, 2, 3])])
    classes = get_conjugate_classes(D3h)
    characters = character_burnside(classes)
    for alpha in 1:size(classes)[1]
        println(characters[alpha, :])
    end

    println("T")
    T = PermutationGroup([Permutation([1, 3, 4, 2]),
                         Permutation([4, 3, 2, 1])])
    classes = get_conjugate_classes(T)
    characters = character_burnside(classes)
    for alpha in 1:size(classes)[1]
        println(characters[alpha, :])
    end

    println("Td")
    Td = PermutationGroup([Permutation([1, 3, 4, 2]),
                           Permutation([4, 3, 2, 1]),
                           Permutation([1, 3, 2, 4])])
    classes = get_conjugate_classes(Td)
    characters = character_burnside(classes)
    for alpha in 1:size(classes)[1]
        println(characters[alpha, :])
    end
end

function test_reg()
    println("C3v")
    C3v = PermutationGroup([Permutation([2, 3, 1]), Permutation([2, 1, 3])])
    reg = regular_representation(C3v)
    @show reg
end


test_character()
# test_reg()
