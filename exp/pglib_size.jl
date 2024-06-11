using Base.Threads
using CSV
using DataFrames
using LinearAlgebra
using SparseArrays
using Statistics

using KLU

using PGLib
using PowerModels
const PM = PowerModels
PM.silence()

using Plots

gmean(x, s=one(eltype(x))) = exp(mean(log.(x .+ s))) - s

function main_pglib_size()

    PGLIB_CASES = filter(f -> isfile(f) && endswith(f, ".m"), readdir(PGLib.PGLib_opf, join=true))

    df = DataFrame(
        casename=String[],
        N=Int[],
        E=Int[],
        nnz_M=Int[],
        nnz_klu=Int[],
        nnz_ldlt=Int[],
    )
    L = ReentrantLock()
    BLAS.set_num_threads(1)
    @threads for fpath in PGLIB_CASES
        fname = basename(fpath)
        data = pglib(fpath)
        N = length(data["bus"])
        N <= 1000 && continue

        @info fname

        data = make_basic_network(data)

        # Re-export to JSON?
        N = length(data["bus"])
        E = length(data["branch"])

        # number of non-zeros in nodal susceptance matrix
        M = calc_basic_susceptance_matrix(data)
        i0 = reference_bus(data)["bus_i"]
        M[i0, :] .= 0
        M[:, i0] .= 0
        M[i0, i0] = -1
        F = klu(M)
        nnz_klu = nnz(F.L) + nnz(F.U)
        nnz_ldl = nnz(ldlt(M))

        lock(L) do
            push!(df, (fname, N, E, nnz(M), nnz_klu, nnz_ldl))
        end
    end

    sort!(df, :N)
    CSV.write(joinpath(@__DIR__, "pglib_size.csv"), df)

    γ_E = gmean(df.E ./ df.N)
    γ_F = gmean(df.nnz_ldlt ./ df.N)

    plt = plot(
        yscale=:log10,
        ylim=(1000, 1_000_000),
        yticks=(10 .^ (3:6), ["\$10^{$k}\$" for k in 3:6]),
        xscale=:log10,
        xlim=(1000, 100_000),
        xticks=(10 .^ (3:5), ["\$10^{$k}\$" for k in 3:5]),
        legend=:topleft,
        xlabel="Number of buses",
        ylabel="Number of non-zeros"
    )
    scatter!(plt, df.N, 2 .* df.E, label="|A|", color=:blue)
    plot!(plt, df.N, 2 .* γ_E .* df.N, label=nothing, color=:blue, linestyle=:dash)
    scatter!(plt, df.N, df.nnz_ldlt, label="|L|", color=:red)
    plot!(plt, df.N, γ_F .* df.N, label=nothing, color=:red, linestyle=:dash)

    savefig(plt, joinpath(@__DIR__, "pglib_size.pdf"))

    return df
end

if abspath(PROGRAM_FILE) == @__FILE__
    df = main_pglib_size()
    exit(0)
end
