using Test
include("permutation.jl")
include("character.jl")

C3v = PermutationGroup([Permutation([2, 3, 1]), Permutation([2, 1, 3])])
C4v = PermutationGroup([Permutation([2, 3, 4, 1]),
                        Permutation([2, 1, 4, 3])])
C6v = PermutationGroup([Permutation([2, 3, 4, 5, 6, 1]),
                        Permutation([6, 5, 4, 3, 2, 1])])
D3h = PermutationGroup([Permutation([2, 3, 1, 5, 6, 4]),
                        Permutation([4, 6, 5, 1, 3, 2]),
                        Permutation([4, 5, 6, 1, 2, 3])])
T = PermutationGroup([Permutation([1, 3, 4, 2]),
                     Permutation([4, 3, 2, 1])])
Td = PermutationGroup([Permutation([1, 3, 4, 2]),
                       Permutation([4, 3, 2, 1]),
                       Permutation([1, 3, 2, 4])])

function test_character(G::PermutationGroup)
    classes = get_conjugate_classes(G)
    characters = character_burnside(classes)
    for alpha in 1:size(classes)[1]
        println(characters[alpha, :])
    end
    return characters
end

function test_reg(G::PermutationGroup)
    reg = regular_representation(G)
    @show reg
    return reg
end


println("Test Character Tables")
println("C3v")
@test_nowarn test_character(C3v)
println("C4v")
@test_nowarn test_character(C4v)
println("C6v")
@test_nowarn test_character(C6v)
println("D3h")
@test_nowarn test_character(D3h)
println("T")
@test_nowarn test_character(T)
println("Td")
@test_nowarn test_character(Td)

println("Test Regular Representation")
println("C3v")
@test_nowarn test_reg(C3v)
