#!/usr/bin/env julia
# ===============================================
# Long-Term Investment Planner  –  MVP  v0.1
# ===============================================
#
# Uso (normal):
#   julia --project=. src/planner.jl input.json
#
# Uso (teste rápido):
#   julia --project=. src/planner.jl input.json --fast
#
# Saídas:
#   wealth_paths.png      (First Plot)
#   avg_contrib.png       (Second Plot)
#   total_contrib.png     (Third Plot)
# -----------------------------------------------

#################################################
# 0. Pacotes e includes do código já existente
#################################################
using JSON3, Statistics, Random, Printf, Logging
using Plots; gr()                                # backend leve
using CSV, DataFrames, Distributions, HMMBase
using JuMP, HiGHS, Ipopt
using Dates, Plots.PlotMeasures

# => inclui suas funções definidas nos arquivos enviados
include("auxiliary_functions.jl")    # load_data, generate_sample, etc.
include("model_functions.jl")        # experiment, simulation_previous_stages …
# Se alguma utilidade só existir em main.jl, inclua-o também:
# include("main.jl")

#################################################
# 1. Função principal
#################################################
function run(json_path::AbstractString; fast::Bool=false)
    #### 1.1 Ler e validar JSON ####
    p = JSON3.read(open(json_path, "r"), Dict{String,Any})
    required = ["age_start","age_end","retirement_age",
                "life_expectancy",
                "income_monthly","contrib_max"]
    foreach(k -> haskey(p,k) || error("Faltou o campo '$k' no JSON"), required)

    age_start, age_end       = p["age_start"],      p["age_end"]
    retirement_age           = p["retirement_age"]
    life_expectancy          = p["life_expectancy"]
    wealth0                  = 0
    income_monthly           = p["income_monthly"]
    contrib_max              = p["contrib_max"]

    @info "JSON lido com sucesso: idade $(age_start)→$(age_end)"

    #### 1.2 Variáveis derivadas ####
    years_simulation = age_end - age_start
    n_stages         = years_simulation * 12               # meses
    pension_months   = (life_expectancy - retirement_age) * 12
    PMT              = 0.70 * income_monthly               # 70 % da renda

    #### 1.3 Dados de mercado e taxa RF ####
    ret, libor = load_data(
        joinpath(@__DIR__, "..", "data", "historical-libor-rates-chart.csv"),
        joinpath(@__DIR__, "..", "data", "S&P 500 Historical Data.csv")
    )
    average_monthly_libor = mean(libor)
    rf_month = average_monthly_libor - 1                   # série vem como 1+taxa
    @info @sprintf("Taxa RF média (mensal): %.4f %%", rf_month*100)

    #### 1.4 Meta financeira (Goal) ####
    Goal = PMT * (1 - (1 + rf_month)^(-pension_months)) / rf_month
    @info "Meta (Goal) = $(round(Goal; digits=2))"

    ############################################################
    # 2. Preparar parâmetros para o modelo
    ############################################################
    n_states    = 4
    sample_size = 1000     # tamanho das amostras de retorno
    n_stages = years_simulation*12                            

    # Ajustar HMM com dados históricos

    test_size = Int(round(size(ret,1)*0.2))
    validation_size = 12*0
    train_ret, test_ret, validation_ret = load_log_ret_train_ret_test_ret(validation_size, test_size, ret)
    t_ret = append!(train_ret, test_ret)
    alpha_ = 0.1
    K = 4
    N = 2
    sig = var(t_ret)
    pre_hmm = HMM(randtransmat(K), [MvNormal(rand(N), sig) for k=1:K])

    hmm, forw = fit_hmm(pre_hmm, N, t_ret, :kmeans, 0)
    hmm, hist = fit_mle(hmm, t_ret; init = method)
    states = viterbi(hmm, t_ret)


    hmm = initial_hmm(K, N, train_ret)

    # Penalidades originais
    λ1, λ2 = 1, -1000/0.85

    n_stages = Int(n_stages)
    ############################################################
    # 3. Rodar as três políticas com a função 'experiment'
    ############################################################
    @time begin
    informations_hmm = experiment(wealth0, Goal, contrib_max,
                           n_stages, n_states, hmm,
                           sample_size, false, false,
                           λ1, λ2, rf_month)

    informations_fixed = experiment(wealth0, Goal, contrib_max,
                           n_stages, n_states, hmm,
                           sample_size, true,  false,
                           λ1, λ2, rf_month)

    informations_time = experiment(wealth0, Goal, contrib_max,
                           n_stages, n_states, hmm,
                           sample_size, false, true,
                           λ1, λ2, rf_month)
    end


    ######################################################################################
    # 4. Funções auxiliares para simulação de trajetórias de riqueza e de contribuição
    ######################################################################################
    function create_basic_arrays_for_simulation(years_simulation, n_stages)
        hmm_model_sim = Array{Float64}(undef, 1000, years_simulation*12)
        hmm_port_sim = Array{Float64}(undef, 1000, n_stages)
        hmm_contribution_sim  = Array{Float64}(undef, 1000, n_stages)
    
    
        fixed_model_sim = Array{Float64}(undef, 1000, years_simulation*12)
        fixed_port_sim  = Array{Float64}(undef, 1000, n_stages)
        fixed_contribution_sim  = Array{Float64}(undef, 1000, n_stages)
    
    
        time_model_sim  = Array{Float64}(undef, 1000, years_simulation*12)
        time_port_sim  = Array{Float64}(undef, 1000, n_stages)
        time_contribution_sim  = Array{Float64}(undef, 1000, n_stages)
    
        return hmm_model_sim, hmm_port_sim, hmm_contribution_sim, fixed_model_sim, fixed_port_sim, fixed_contribution_sim, time_model_sim, time_port_sim, time_contribution_sim 
    end
    
    function simulating_comparative_trajectories(hmm_model_sim, hmm_port_sim, hmm_contribution_sim, fixed_model_sim, fixed_port_sim, fixed_contribution_sim, time_model_sim, time_port_sim, time_contribution_sim, years_simulation, n_stages, n_states, hmm, num_scenarios, train_ret, sample_size, informations_hmm, informations_fixed, informations_time, max_cont, pv_rate,  Goal, rate, λ_1, λ_2)
        combination_line_fixed = [false, true, false]
        combination_line_policy = [false, false, true]
        for i in 1:3
            fixed = combination_line_fixed[i]
            time_policy = combination_line_policy[i]
            if (fixed == false) & (time_policy == false)
                print("Simulacao FAlSE FALSE")
                hmm_model_sim, hmm_port_sim, hmm_contribution_sim = simulating_individual_models(fixed, time_policy, years_simulation, n_stages, n_states, hmm, num_scenarios, train_ret, sample_size, informations_hmm, informations_fixed, informations_time, max_cont, pv_rate,  Goal, rate, λ_1, λ_2)
            elseif (fixed == true) & (time_policy == false)
                print("Simulacao TRUE FALSE")
                fixed_model_sim, fixed_port_sim, fixed_contribution_sim = simulating_individual_models(fixed, time_policy, years_simulation, n_stages, n_states, hmm, num_scenarios, train_ret, sample_size, informations_hmm, informations_fixed, informations_time, max_cont, pv_rate,  Goal, rate, λ_1, λ_2)
            elseif (fixed == false) & (time_policy == true)
                print("Simulacao FALSE TRUE")
                time_model_sim, time_port_sim, time_contribution_sim = simulating_individual_models(fixed, time_policy, years_simulation, n_stages, n_states, hmm, num_scenarios, train_ret, sample_size, informations_hmm, informations_fixed, informations_time, max_cont, pv_rate,  Goal, rate, λ_1, λ_2)
            end
        end
        return  hmm_model_sim, hmm_port_sim, hmm_contribution_sim, fixed_model_sim, fixed_port_sim, fixed_contribution_sim, time_model_sim, time_port_sim, time_contribution_sim
    end
    
    function simulating_individual_models(fixed, time_policy, years_simulation, n_stages, n_states, hmm, num_scenarios, train_ret, sample_size, informations_hmm, informations_fixed, informations_time, max_cont, pv_rate,  Goal, rate, λ_1, λ_2)
        model_sim = Array{Float64}(undef, 1000, years_simulation*12)
        port_sim = Array{Float64}(undef, 1000, n_stages)
        contribution_sim = Array{Float64}(undef, 1000, n_stages)
        mu = []
        sigma = []
        for k in 1:n_states
            mu = append!(mu, params(hmm.B[k])[1])
            sigma = append!(sigma, params(hmm.B[k])[2])
        end
        for sc in 1:num_scenarios
            scenario_wealth_stage, scenario_portfolio, scenario_contribution = simulating_scenarios(hmm, years_simulation, sc, train_ret, mu, sigma, n_states, sample_size, fixed, time_policy, informations_hmm, informations_fixed, informations_time, max_cont, pv_rate, n_stages, Goal,  rate, λ_1, λ_2)
            model_sim[sc,:] = scenario_wealth_stage[2:end]
            port_sim[sc,:] = scenario_portfolio
            contribution_sim[sc,:] = scenario_contribution
        end
        return model_sim, port_sim, contribution_sim
    
    end
    
    function simulating_scenarios(hmm, years_simulation, sc, train_ret, mu, sigma, n_states, sample_size, fixed, time_policy, informations_hmm, informations_fixed, informations_time, max_cont, pv_rate, n_stages, Goal,  rate, λ_1, λ_2)
        print("AA", sc)
        simulation_state, simulation_returns = rand(hmm, 12*years_simulation, seq = true)
        scenario_wealth_stage = [0.0]
        scenario_portfolio = [] 
        scenario_contribution = []
        scenario_observations = Float64[]
        for t in 1:years_simulation*12
            scenario_observations = append!(scenario_observations, simulation_returns[t])
            l_aux = vcat(train_ret, scenario_observations)
            state = findmax( forward(hmm, l_aux)[1][end, :] )[end]
            prob_to_state = hmm.A[state,:]
            retornos = generate_sample(mu, sigma, n_states, sample_size)
            if t < years_simulation*12
                next_stage_functions = Dict()
                for k in 1:n_states
                    if (fixed == false) & (time_policy == false)
                        setindex!(next_stage_functions, informations_hmm["$(t+1)"][string("state_", k)]["functions"], string("state_", k))
                    elseif (fixed == true) & (time_policy == false)
                        setindex!(next_stage_functions, informations_fixed["$(t+1)"][string("state_", k)]["functions"], string("state_", k))
                    elseif (fixed == false) & (time_policy == true)
                        setindex!(next_stage_functions, informations_time["$(t+1)"][string("state_", k)]["functions"], string("state_", k))
                    end
                end
            end
            if t < years_simulation*12
                cash_flow, allo_stock = simulation_previous_stages(last(scenario_wealth_stage), max_cont, retornos, pv_rate, n_states, prob_to_state, t, n_stages, next_stage_functions, fixed, time_policy)
            else
                cash_flow, allo_stock = simulation_last_stage(n_states, last(scenario_wealth_stage), Goal, retornos, rate, fixed, time_policy, λ_1, λ_2)
            end
            w_stg = last(scenario_wealth_stage) + cash_flow
            if isnan(allo_stock)
                allo_stock  = 0.0
            end
            scenario_contribution = append!(scenario_contribution, [cash_flow])
            scenario_portfolio = append!(scenario_portfolio, [allo_stock])
            if t !== years_simulation*12 
                    # w_stg = w_stg * exp(validation_ret[t])*allo_stock + w_stg*(1-allo_stock)*(1+0.05)^(1/12)
                    w_stg = w_stg * exp(simulation_returns[t])*allo_stock + w_stg*(1-allo_stock)*(1+0.05)^(1/12)
            end
            scenario_wealth_stage = push!(scenario_wealth_stage, w_stg)
        end
        return  scenario_wealth_stage, scenario_portfolio, scenario_contribution
    end
    
    hmm_model_sim, hmm_port_sim, hmm_contribution_sim, fixed_model_sim, fixed_port_sim, fixed_contribution_sim, time_model_sim, time_port_sim, time_contribution_sim  = create_basic_arrays_for_simulation(years_simulation, n_stages)
    num_scenarios = 100
    @time begin
    hmm_model_sim, hmm_port_sim, hmm_contribution_sim, fixed_model_sim, fixed_port_sim, fixed_contribution_sim, time_model_sim, time_port_sim, time_contribution_sim = simulating_comparative_trajectories(hmm_model_sim, hmm_port_sim, hmm_contribution_sim, fixed_model_sim, fixed_port_sim, fixed_contribution_sim, time_model_sim, time_port_sim, time_contribution_sim, years_simulation, n_stages, n_states, hmm, num_scenarios, train_ret, sample_size, informations_hmm, informations_fixed, informations_time,  contrib_max, rf_month,  Goal, rf_month, λ1, λ2)
    end

    df_wealth_hmm_model = DataFrame(hmm_model_sim, :auto)[1:10,:]
    df_contribution_hmm_model = DataFrame(hmm_contribution_sim, :auto)[1:10,:]

    df_wealth_hmm_fixed = DataFrame(fixed_model_sim, :auto)[1:10,:]
    df_contribution_hmm_fixed = DataFrame(fixed_contribution_sim, :auto)[1:10,:]

    df_wealth_hmm_time = DataFrame(time_model_sim, :auto)[1:10,:]
    df_contribution_hmm_time = DataFrame(time_contribution_sim, :auto)[1:10,:]


    matrix_wealth_hmm_model = Matrix(df_wealth_hmm_model)
    matrix_wealth_hmm_fixed = Matrix(df_wealth_hmm_fixed)
    matrix_wealth_hmm_time = Matrix(df_wealth_hmm_time)

    matrix_contribution_hmm_model = Matrix(df_contribution_hmm_model)
    matrix_contribution_hmm_fixed = Matrix(df_contribution_hmm_fixed)
    matrix_contribution_hmm_time  = Matrix(df_contribution_hmm_time)

    ##########################################################################
    # 5. FIRST PLOT — Wealth Trajectory Between Different Allocations 
    ##########################################################################
    qs = [0.25, 0.5, 0.75]
    quants_wealth_hmm_model = hcat([quantile(c, qs) for c in eachcol(matrix_wealth_hmm_model)]...)
    quants_wealth_hmm_fixed =  hcat([quantile(c, qs) for c in eachcol(matrix_wealth_hmm_fixed)]...)
    quants_wealth_hmm_time = hcat([quantile(c, qs) for c in eachcol(matrix_wealth_hmm_time)]...)
    x = 1:size(quants_wealth_hmm_time, 2)

    p1 = plot(x, quants_wealth_hmm_model', color = "blue", label = ["Model's Quantile" "" ""], title = "Wealth Quantiles 0.25-0.5-0.75", legend = :bottomright)
    plot!(x,quants_wealth_hmm_fixed', color = "orange", label = ["Fixed Allocation Quantile" "" ""])
    plot!(x,quants_wealth_hmm_time', color = "red", label = ["Time Allocation Quantile" "" ""] )
    savefig(p1, "plots/wealth_trajectory.png")
    ########################################################################################
    # 6. SECOND PLOT —  Average Contribution considering different allocations across stages
    ########################################################################################

    mean_contribution_per_stage_model = mean(matrix_contribution_hmm_model, dims = 1)
    mean_contribution_per_stage_fixed = mean(matrix_contribution_hmm_fixed, dims = 1)
    mean_contribution_per_stage_time = mean(matrix_contribution_hmm_time, dims = 1)

    total_contribution_model =  sum(Matrix(df_contribution_hmm_model))/num_scenarios
    total_contribution_fixed = sum(Matrix(df_contribution_hmm_fixed))/num_scenarios
    total_contribution_time = sum(Matrix(df_contribution_hmm_time))/num_scenarios

    p2 = scatter(1:length(mean_contribution_per_stage_model[1,:]), mean_contribution_per_stage_model[1,:], label = "Model Allocation", xlabel = "Stage", ylabel = "Contribution") 
    scatter!(1:length(mean_contribution_per_stage_fixed[1,:]), mean_contribution_per_stage_fixed[1,:], label = "Fixed Allocation", xlabel = "Stage", ylabel = "Contribution") 
    scatter!(1:length(mean_contribution_per_stage_time[1,:]), mean_contribution_per_stage_time[1,:], label = "Time Allocation", xlabel = "Stage", ylabel = "Contribution") 
    savefig(p2, "plots/contrib_across_stages.png")
    ############################################################
    # 7. THIRD PLOT — Total Contribution in the trajectory
    ############################################################
    p3 = bar(["Model's Allocation", "Fixed Allocation", "Time based Allocation"], [total_contribution_model, total_contribution_fixed, total_contribution_time], title = "Total Contribution", xlabel = "Allocation Type", color= [:blue, :orange, :red], label= "", ylabel = "x10^3 \$")
    savefig(p3, "plots/total_contrib.png")
end

#################################################
# 2. CLI simples
#################################################
if abspath(PROGRAM_FILE) == @__FILE__
    if length(ARGS) == 0
        println("Uso: julia --project=. src/planner.jl <arquivo_json> [--fast]")
        exit(1)
    end
    jsonfile = ARGS[1]
    fastflag = (length(ARGS) > 1) && ARGS[2] == "--fast"
    run(jsonfile; fast=fastflag)
end
