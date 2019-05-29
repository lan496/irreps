using LinearAlgebra: diagm
include("projector.jl")
function test_H2_vibration()
    # (E, C2, sigma_v1, sigma_v2)
    characters = [
                  1 1 1 1;  # A_1
                  1 1 -1 -1;  # A_2
                  1 -1 1 -1;  # B_1
                  1 -1 -1 1;  # B_2
                 ]

    nc = 4
    rep = zeros(nc, 6, 6)  # (|G|, dim, dim)
    I = diagm(0 => [1, 1, 1])
    C2 = diagm(0 => [-1, -1, 1])
    sigma_x = diagm(0 => [-1, 1, 1])
    sigma_y = diagm(0 => [1, -1, 1])
    rep[1, :, :] = kron([1 0; 0 1], I)
    rep[2, :, :] = kron([0 1; 1 0], C2)
    rep[3, :, :] = kron([1 0; 0 1], sigma_x)
    rep[4, :, :] = kron([0 1; 1 0], sigma_y)
    proj = irreps_projecton_operator(characters, rep)
    eigu = projector_eigenvectors(proj)
    for u in eigu
        @show u
    end
end

function test_H2O_wavefunction()
    # (E, C2, sigma_v1, sigma_v2)
    characters = [
                  1 1 1 1;  # A_1
                  1 1 -1 -1;  # A_2
                  1 -1 1 -1;  # B_1
                  1 -1 -1 1;  # B_2
                 ]
    nc = 4
    # Hibert space spanned by {1s(H1), 1s(H2), 2px(O), 2py(O), 2pz(O)}
    rep = zeros(nc, 5, 5)
    rep[1, :, :] = diagm(0 => [1, 1, 1, 1, 1])
    rep[2, :, :] = [
                    0 1 0 0 0;
                    1 0 0 0 0;
                    0 0 -1 0 0;
                    0 0 0 -1 0;
                    0 0 0 0 1;
                   ]
    rep[3, :, :] = diagm(0 => [1, 1, -1, 1, 1])
    rep[4, :, :] = [
                    0 1 0 0 0;
                    1 0 0 0 0;
                    0 0 1 0 0;
                    0 0 0 -1 0;
                    0 0 0 0 1;
                   ]
    proj = irreps_projecton_operator(characters, rep)
    eigu = projector_eigenvectors(proj)
    for u in eigu
        @show u
    end
end

function test_NH3_vibration()
    # C_{3v}
    # (E, C3, C3^{2}, sigma_v1, sigma_v2, sigma_v3)
    characters = [
                  1 1 1 1 1 1;  # A_1
                  1 1 1 -1 -1 -1;  # A_2
                  2 -1 -1 0 0 0;  # E
                 ]
    # Phase space spanned by {x(H1), y(H1), x(H2), y(H2), x(H3), y(H3)}
    rep = zeros(6, 6, 6)
    I = diagm(0 => [1, 1])
    C3 = [-1 / 2 -sqrt(3) / 2; sqrt(3) / 2 -1 / 2]
    sigma_v1 = diagm(0 => [-1, 1])
    C3_inv = C3 * C3
    sigma_v2 = C3 * sigma_v1
    sigma_v3 = C3_inv * sigma_v1
    rep[1, :, :] = kron(diagm(0 => [1, 1, 1]), I)
    rep[2, :, :] = kron([0 1 0; 0 0 1; 1 0 0], C3)
    rep[3, :, :] = kron([0 0 1; 1 0 0; 0 1 0], C3_inv)
    rep[4, :, :] = kron([1 0 0; 0 0 1; 0 1 0], sigma_v1)
    rep[5, :, :] = kron([0 0 1; 0 1 0; 1 0 0], sigma_v2)
    rep[6, :, :] = kron([0 1 0; 1 0 0; 0 0 1], sigma_v3)
    proj = irreps_projecton_operator(characters, rep)
    eigu = projector_eigenvectors(proj)
    for u in eigu
        @show u
    end
end

test_H2_vibration()
test_H2O_wavefunction()
test_NH3_vibration()
