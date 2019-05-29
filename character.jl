include("permutation.jl")
using LinearAlgebra: eigen
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
