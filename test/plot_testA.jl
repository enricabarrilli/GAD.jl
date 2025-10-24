include("../src/GAD.jl")

using DynamicPolynomials
using Distributions
using LinearAlgebra
using Plots
using TensorDec


function random_tensor_gad(n::Int, d::Int, ks::Vector{Int})
    s = length(ks)
    X = (@polyvar x[0:n])[1]
    poly = zero(X[1])
    ells = []
    omegas = []

    for k in ks
        ell_coeffs = randn(n+1)
        ell_coeffs[1] = 1 
        ell = sum(ell_coeffs[i] * X[i] for i in 1:n+1)
        push!(ells, ell)

        monomials_k = monomials(X, k)
        coeffs = randn(length(monomials_k))
        omega = sum(coeffs[i] * monomials_k[i] for i in 1:length(monomials_k))

        while omega % ell == 0 && k > 0
            coeffs = randn(length(monomials_k))
            omega = sum(coeffs[i] * monomials_k[i] for i in 1:length(monomials_k))
        end

        push!(omegas, omega)
        poly += omega * ell^(d - k)
    end

    return poly, ells, omegas, s
end




function random_poly(F)
    X = variables(F)
    d = maxdegree(F)
    mons = monomials(X, d)
    coeffs = rand(Normal(0, 1), length(mons))
    return sum(coeffs[i] * mons[i] for i in 1:length(mons))
end


function stability_data(epsilons::Vector{Float64}; n, d, ks, n_trials=3)
    means = Float64[]
    stds  = Float64[]
    lmax = []
    lmin = []
    raw_data = Dict{Float64, Vector{Float64}}()
    F0, _, _, _ = random_tensor_gad(n, d, ks)
    for eps in epsilons
        deltas = Float64[]
        for _ in 1:n_trials
            R = random_poly(F0)
            R = R / norm_apolar(R)
            F = (F0 + eps * R)/norm_apolar(F0 + eps * R)
            try
                if gad == true
                    W, L, mu = gad_decompose(F)
                    println("gad ",mu)
                    T = reconstruct(W, L, maxdegree(F))
                else
                    W, L = decompose(F)
                    T = tensor(W,L,variables(F),d)
                end
                push!(deltas, norm_apolar(T - F)/ norm_apolar(F))
              
            catch
                push!(deltas, NaN)
            end
        end
        clean = filter(!isnan, deltas)
        raw_data[eps] = deltas
       if isempty(clean)
    push!(means, NaN)
    push!(stds, NaN)
    push!(lmax, NaN)
    push!(lmin, NaN)
else
    push!(means, median(clean))
    push!(stds, std(clean))
    push!(lmax, maximum(clean))
    push!(lmin, minimum(clean))
end
    end
    return (epsilons=epsilons, means=means, lmax=lmax, lmin=lmin, raw=raw_data)
end

gad=true

eps_vals = 10.0 .^ (-14:0.5:0)
data = stability_data(eps_vals; n=10, d=3, ks = [0,0,0,0,0], n_trials=10)

mask = .!isnan.(data.means)
eps_valid   = data.epsilons[mask]
means_valid = data.means[mask]
lmax_valid  = data.lmax[mask]
lmin_valid  = data.lmin[mask]


plot(eps_valid, means_valid;
     xscale=:log10, yscale=:log10,
     label="median", marker=:circle,
     xlabel="Perturbation size", ylabel="Relative reconstruction error",
     title="Stability of gad_decompose")

plot!(eps_valid, lmax_valid; label="max error")
plot!(eps_valid, lmin_valid; label="min error")
