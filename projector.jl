using LinearAlgebra: eigen, normalize
include("permutation.jl")

"""
    irreps_projecton_operator(characters, rep)
# Attributes
characters: Array, (# of Irreps, order of group)
rep: Array, (order of group, dim, dim)

# Returns
proj: Array, (# of Irreps, dim, dim)
"""
function irreps_projecton_operator(characters::Array, rep::Array) where T<:Number
    nc, order = size(characters)
    dim = size(rep)[end]
    dimensions = characters[:, 1]  # assume characters[:, 1] is for identity
    proj = zeros(nc, dim, dim)
    for mu in 1:nc
        for i in 1:order
            proj[mu, :, :] += characters[mu, i] * rep[i, :, :]
        end
        proj[mu, :, :] *= dimensions[mu] / order
    end
    return proj
end

"""
    projector_eigenvectors(proj)

# Attributes
proj: Array, (# of Irreps, dim, dim)

# Returns
eigvecs: Array of Array{T, 2}
eigvecs[mu] is linealy independent symmetrized eigenvectors for mu-th Irreps
"""
function projector_eigenvectors(proj::Array)
    nc, dim, _ = size(proj)
    eigvecs = []
    for mu in 1:nc
        eigvecs_mu = []
        vals, vecs = eigen(proj[mu, :, :])
        for i in 1:dim
            if !isapprox(vals[i], 1)
                continue
            end
            vec = reshape(normalize(vecs[:, i], Inf), (1, :))
            # push!(eigvecs_mu, reshape(vecs[:, i], (1, :)))
            push!(eigvecs_mu, vec)
        end
        # eigvecs_mu = unique_approx(eigvecs_mu)
        push!(eigvecs, vcat(eigvecs_mu...))
        # push!(eigvecs, eigvecs_mu)
    end
    return eigvecs
end
