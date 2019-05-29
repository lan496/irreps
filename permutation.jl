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
