# test/runtests.jl – versão sem o teste load_data
using Test, JSON3, Random

include(joinpath(@__DIR__, "..", "src/auxiliary_functions.jl"))
include(joinpath(@__DIR__, "..", "src/planner.jl"))

const TMPDIR = mktempdir()

# ---------------------------------------------------------
@testset "JSON validation – campo obrigatório" begin
    json_bad = joinpath(TMPDIR, "bad.json")
    JSON3.write(json_bad, Dict(
        "age_start"       => 30,
        "age_end"         => 31,
        "life_expectancy" => 32,
        "income_monthly"  => 2000
        # contrib_max ausente
    ))
    @test_throws ErrorException run(json_bad)
end

# ---------------------------------------------------------
@testset "initial_hmm – dimensões básicas" begin
    K, N = 3, 2
    hmm = initial_hmm(K, N, randn(120) .* 0.01)
    @test length(hmm.B) == K
    @test size(hmm.A, 1) == K
end
