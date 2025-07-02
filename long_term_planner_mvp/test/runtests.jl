# test/runtests.jl – versão sem o teste load_data
using Test, JSON3, Random
using Distributions, Statistics

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


@testset "load_log_ret_train_ret_test_ret – partições corretas" begin
    n        = 120               # 10 anos de dados mensais
    ret      = 1 .+ randn(n) .* 0.01     # série sintética > 0
    val_sz   = 12
    test_sz  = 24
    train, test, val = load_log_ret_train_ret_test_ret(val_sz, test_sz, ret)

    @test length(train) == n - val_sz - test_sz - 1
    @test length(test)  == test_sz
    @test length(val)   == val_sz
end

@testset "generate_sample – dimensões e positividade" begin
    mu    = [0.0, 0.01]           # dois estados
    sigma = [0.02, 0.03]
    n_s   = length(mu)
    S     = 50
    samples = generate_sample(mu, sigma, n_s, S)   # :contentReference[oaicite:2]{index=2}

    @test size(samples) == (2, S, n_s)
    @test all(samples .> 0)        # retornos multiplicativos devem ser positivos
    # primeira camada é exp(r); média aproximada deve refletir e^μ
    approx_means = [mean(samples[1, :, k]) for k in 1:n_s]
    @test isapprox.(approx_means, exp.(mu); atol = 0.1) |> all
end

@testset "create_violations_series – detecção de violações" begin
    hist = [1.02, 0.99, 1.05, 0.97]
    upper = [1.03, 1.03, 1.03, 1.03]
    lower = [0.98, 0.98, 0.98, 0.98]
    viol = create_violations_series(hist, upper, lower)   # :contentReference[oaicite:3]{index=3}

    @test viol == [0, 0, 1, 1]      # pontos 2 e 4 violam os limites
    @test sum(viol) == 2.0
end
