function last_stage(initial_wealth, Goal, retornos, n_states, fixed, time_policy, λ_1, λ_2)
    portfolio = []
    contributions = []
    arr_wealth = []
    duals = []
    objectives = []

    S = size(retornos,2)
    K = n_states

    #model = JuMP.direct_model(Gurobi.Optimizer())
    model = JuMP.direct_model(HiGHS.Optimizer())
    JuMP.set_silent(model)

    JuMP.@variable(model, wealth[1:S, 1:K] >= 0)
    JuMP.@variable(model, c_retirada >= 0)
    JuMP.@variable(model, c_investimento >= 0)
    JuMP.@variable(model, x[1:2] >= 0)

    if fixed == true
        JuMP.@constraint(model, cons_model_11, x[1]  == 0.6 * sum(x) ) 
    end
    
    if time_policy == true
        JuMP.@constraint(model, cons_12, x[2]  == sum(x)) #case with 1-t portfolio
    end

    JuMP.@constraint(model, cons_2, sum(x) - (c_investimento - c_retirada) == initial_wealth)
    JuMP.@constraint(model, Goal == sum(x) ) 

    JuMP.@objective(model, Min, λ_1*c_investimento - λ_2*(c_retirada*0.85))

    while initial_wealth <= 1.5*Goal
        JuMP.optimize!(model)
        
        push!(arr_wealth, initial_wealth)
        push!(contributions, value(c_investimento) - value(c_retirada)*0.85)
        push!(portfolio, JuMP.value.(x)./sum(JuMP.value.(x)))
        push!(duals, JuMP.dual(cons_2))
        push!(objectives, JuMP.objective_value(model))

        #rhs = MOI.get(model, Gurobi.ConstraintAttribute("SARHSUp"), cons_2)
        #initial_wealth = rhs + 1
        initial_wealth += 1000


        JuMP.set_normalized_rhs(cons_2, initial_wealth)
    
    end

    initial_wealth = Goal

    JuMP.set_normalized_rhs(cons_2, initial_wealth)
    JuMP.optimize!(model)

    push!(arr_wealth, initial_wealth)
    push!(contributions, value(c_investimento) - value(c_retirada)*0.85)
    push!(portfolio, value.(x)./sum(value.(x)))
    push!(duals, JuMP.dual(cons_2))
    push!(objectives, JuMP.objective_value(model))


    initial_wealth = 1.5*Goal

    JuMP.set_normalized_rhs(cons_2, initial_wealth)
    JuMP.optimize!(model)

    push!(arr_wealth, initial_wealth)
    push!(contributions, value(c_investimento) - value(c_retirada)*0.85)
    push!(portfolio, value.(x)./sum(value.(x)))
    push!(duals, JuMP.dual(cons_2))
    push!(objectives, JuMP.objective_value(model))


    functions = Array{Function}(undef, size(objectives,1)+1)
    for i = 1:size(objectives,1)
        functions[i] =  (x -> objectives[i] + duals[i]*(x - arr_wealth[i]))
    end
    functions[size(objectives,1)+1] =  (x->0)


    return functions, portfolio, contributions, arr_wealth, duals, objectives

end

function previous_stages(initial_wealth, max_cont, informations, retornos, rate,  n_states, prob_to_state, Goal, t, n_stages, fixed, time_policy)
    previous_portfolio = []
    previous_contributions = []
    previous_arr_wealth = []
    previous_duals = []
    previous_objectives = []

    S = size(retornos,2)
    K = n_states

    functions = Dict()
    for k in 1:n_states
        setindex!(functions, informations[string("state_", k)]["functions"], string("state_", k))
    end

    S = size(retornos,2)

    #model2 = JuMP.direct_model(Gurobi.Optimizer())
    model2 = JuMP.direct_model(HiGHS.Optimizer())
    JuMP.set_silent(model2)

    JuMP.@variable(model2, wealth[1:S, 1:K] >= 0)
    JuMP.@variable(model2, c_retirada >= 0)
    JuMP.@variable(model2, c_investimento >= 0)
    JuMP.@variable(model2, c)
    JuMP.@variable(model2, x[1:2] >= 0)
    JuMP.@variable(model2, θ[1:S, 1:K] >= 0)

    if fixed == true
        JuMP.@constraint(model2, cons_model2_11, x[1]  == 0.6 * sum(x) ) 
    end
    if time_policy == true
        JuMP.@constraint(model2, cons_model2_11, x[1] == ((n_stages - t)/n_stages) * sum(x)) #case with 1-t policy
    end
    
    
    JuMP.@constraint(model2, c == c_investimento - (c_retirada*0.85) )
    JuMP.@constraint(model2, cons_model2_2, sum(x) - (c_investimento - c_retirada) == initial_wealth)
    JuMP.@constraint(model2, [s=1:S, k=1:K], wealth[s,k] == sum(x[i]*(retornos[i, s, k]) for i = 1:2))
    JuMP.@constraint(model2, c <= max_cont)
    
    for k in 1:K
        JuMP.@constraint(model2, [l=1:S, j=1:size(functions[string("state_", k)],1)], θ[l,k] >= functions[string("state_", k)][j](wealth[l,k]) )
    end 
    
    JuMP.@objective(model2, Min, c + (1/(1+rate))* sum(sum(θ[s,k]/S for s in 1:S)*prob_to_state[k] for k in 1:K) )
    

    while initial_wealth < Goal
        
        JuMP.optimize!(model2)
    
        push!(previous_arr_wealth, initial_wealth)
        push!(previous_contributions, value(c_investimento) - value(c_retirada))
        push!(previous_portfolio, value.(x)./sum(value.(x)))
        push!(previous_duals, JuMP.dual(cons_model2_2))
        push!(previous_objectives, JuMP.objective_value(model2))

        #rhs = MOI.get(model2, Gurobi.ConstraintAttribute("SARHSUp"), cons_model2_2)
        initial_wealth += 1000

        JuMP.set_normalized_rhs(cons_model2_2, initial_wealth)
    end

    initial_wealth = Goal

    JuMP.set_normalized_rhs(cons_model2_2, initial_wealth)
    JuMP.optimize!(model2)
    push!(previous_arr_wealth, initial_wealth)
    push!(previous_contributions, value(c))
    push!(previous_portfolio, value.(x)./sum(value.(x)))
    push!(previous_duals, JuMP.dual(cons_model2_2))
    push!(previous_objectives, JuMP.objective_value(model2))

    initial_wealth = 1.5*Goal

    JuMP.set_normalized_rhs(cons_model2_2, initial_wealth)
    JuMP.optimize!(model2)
    
    push!(previous_arr_wealth, initial_wealth)
    push!(previous_contributions, value(c))
    push!(previous_portfolio, value.(x)./sum(value.(x)))
    push!(previous_duals, JuMP.dual(cons_model2_2))
    push!(previous_objectives, JuMP.objective_value(model2))

    
    previous_functions = Array{Function}(undef, size(previous_objectives,1)+1)
    for i = 1:size(previous_objectives,1)
        previous_functions[i] =  (x -> previous_objectives[i] + previous_duals[i]*(x - previous_arr_wealth[i]))
    end 
    previous_functions[size(previous_objectives,1)+1] = (x->0)

    return previous_functions, previous_portfolio, previous_contributions, previous_arr_wealth,
            previous_duals, previous_objectives
end

function experiment(initial_wealth, Goal, max_cont, n_stages, n_states, hmm, sample_size, fixed, time_policy, λ_1, λ_2, pv_rate)
    informations = Dict()
    mu = []
    sigma = []
    for k in 1:n_states
        mu = append!(mu, params(hmm.B[k])[1])
        sigma = append!(sigma, params(hmm.B[k])[2])
    end

    for t = n_stages:-1:1
        stage_information = Dict()
        for k in 1:n_states
            setindex!(stage_information, Dict(), string("state_", k))
        end

        if t == n_stages

            println("Starting ($t)th stage")
            for i in 1:n_states
                prob_to_state = hmm.A[i,:]
                retornos = generate_sample(mu, sigma, n_states, sample_size)

                functions, portfolio, contributions, arr_wealth,
                duals, objectives = last_stage(initial_wealth, Goal, retornos, n_states, fixed, time_policy, λ_1, λ_2);

                stage_information[string("state_", i)]["functions"] = functions
                stage_information[string("state_", i)]["portfolio"] = portfolio
                stage_information[string("state_", i)]["contributions"] = contributions
                stage_information[string("state_", i)]["arr_wealth"] = arr_wealth
                stage_information[string("state_", i)]["duals"] = duals
                stage_information[string("state_", i)]["objectives"] = objectives
            end
            println("Finished ($t)th stage")
        else
            println("Starting ($t)th stage")
            for j in 1:n_states
                prob_to_state = hmm.A[j,:]
                retornos = generate_sample(mu, sigma, n_states, sample_size)

                functions, portfolio, contributions, arr_wealth,
                duals, objectives = previous_stages(initial_wealth, max_cont, informations["$(t+1)"], retornos, pv_rate, n_states, prob_to_state, Goal, t, n_stages, fixed, time_policy);

                stage_information[string("state_", j)]["functions"] = functions
                stage_information[string("state_", j)]["portfolio"] = portfolio
                stage_information[string("state_", j)]["contributions"] = contributions
                stage_information[string("state_", j)]["arr_wealth"] = arr_wealth
                stage_information[string("state_", j)]["duals"] = duals
                stage_information[string("state_", j)]["objectives"] = objectives
            end
            println("Finished ($t)th stage")
            
        end
        
        informations["$t"] = stage_information

    end

    return informations
end

function simulation_previous_stages(initial_wealth, max_cont, retornos, rate, n_states, prob_to_state, t, n_stages, next_stage_functions, fixed, time_policy)

    S = size(retornos,2)
    K = n_states

    S = size(retornos,2)

    #model2 = JuMP.direct_model(Gurobi.Optimizer())
    model2 = JuMP.direct_model(HiGHS.Optimizer())

    JuMP.set_silent(model2)

    JuMP.@variable(model2, wealth[1:S, 1:K] >= 0)
    JuMP.@variable(model2, c_retirada >= 0)
    JuMP.@variable(model2, c_investimento >= 0)
    JuMP.@variable(model2, c)
    JuMP.@variable(model2, x[1:2] >= 0)
    JuMP.@variable(model2, θ[1:S, 1:K] >= 0)

    if fixed == true
        JuMP.@constraint(model2, cons_model2_11, x[1]  == 0.6 * sum(x) ) 
    end
    if time_policy == true
        JuMP.@constraint(model2, cons_model2_11, x[1] == ((n_stages - t)/n_stages) * sum(x)) #case with 1-t policy
    end
    
    JuMP.@constraint(model2, c == c_investimento - (c_retirada*0.85) )
    JuMP.@constraint(model2, cons_model2_2, sum(x) - (c_investimento - c_retirada) == initial_wealth)
    JuMP.@constraint(model2, [s=1:S, k=1:K], wealth[s,k] == sum(x[i]*(retornos[i, s, k]) for i = 1:2))
    JuMP.@constraint(model2, c <= max_cont)
    
    for k in 1:K
        JuMP.@constraint(model2, [l=1:S, j=1:size(next_stage_functions[string("state_", k)],1)], θ[l,k] >= next_stage_functions[string("state_", k)][j](wealth[l,k]) )
    end 
    
    JuMP.@objective(model2, Min, c + (1/(1+rate))* sum(sum(θ[s,k]/S for s in 1:S)*prob_to_state[k] for k in 1:K) )
    
    JuMP.optimize!(model2)
    
    cash_flow = value(c_investimento) - value(c_retirada)
    allo_stock = value.(x)[1]/sum(value.(x))

    return cash_flow, allo_stock
end

function simulation_last_stage(initial_wealth, Goal, retornos, rate, fixed, time_policy, λ_1, λ_2)
    portfolio = []
    contributions = []
    arr_wealth = []
    duals = []
    objectives = []

    S = size(retornos,2)

    #model = JuMP.direct_model(Gurobi.Optimizer())
    model = JuMP.direct_model(HiGHS.Optimizer())

    JuMP.set_silent(model)

    JuMP.@variable(model, wealth[1:S, 1:K] >= 0)
    JuMP.@variable(model, c_retirada >= 0)
    JuMP.@variable(model, c_investimento >= 0)
    JuMP.@variable(model, x[1:2] >= 0)

    if fixed == true
        JuMP.@constraint(model, cons_model_11, x[1]  == 0.6 * sum(x) ) 
    end
    
    if time_policy == true
        JuMP.@constraint(model, cons_12, x[2]  == sum(x)) #case with 1-t portfolio
    end

    JuMP.@constraint(model, cons_2, sum(x) - (c_investimento - c_retirada) == initial_wealth)
    JuMP.@constraint(model, Goal == sum(x) ) 

    JuMP.@objective(model, Min, λ_1*c_investimento - λ_2*(c_retirada*0.85))

    JuMP.optimize!(model)

    cash_flow = value(c_investimento) - value(c_retirada)
    allo_stock = value.(x)[1]/sum(value.(x))

    return cash_flow, allo_stock

end