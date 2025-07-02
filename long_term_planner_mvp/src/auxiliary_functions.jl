
using Distributions
using Optim

function f!(F, x, p, alpha, hmm, K)
    F[1] = sum(p[k]*cdf(hmm.B[k],x[1]) for k in 1:K) .- alpha
end

function find_quantile(N, p_next, hmm, alpha_1, alpha_2, quantile_prediction, K)
    for n in 1:N
        x0 = [sum(p_next[k]*mean(hmm.B[k]) for k in 1:K)]   #modificaçao para ter mais de um ativo
        result_1 = nlsolve((F,x) ->f!(F, x, p_next, alpha_1, hmm, K), x0)
        result_2 = nlsolve((F,x) ->f!(F, x, p_next, alpha_2, hmm, K), x0)
        quantile_prediction = push!(quantile_prediction,  (result_1.zero[1], result_2.zero[1]))
    end
    return quantile_prediction
end

function fit_hmm(hmm, N, ret, method, i)
    if N >= 2
        hmm, hist = fit_mle(hmm, ret[:, 1:end-i]; init = method)
        forw = forward(hmm, ret[:, 1:end-i])
    else
        hmm, hist = fit_mle(hmm, ret[1:end-i]; init = method)
        forw = forward(hmm, ret[1:end-i])
    end
    return hmm, forw
end

function one_step_quantile_predictions(N, K, test_size, hmm, ret, alpha)
    upper_one_step_quantile = zeros(N, test_size)
    lower_one_step_quantile = zeros(N, test_size)


    
    for i in  collect(test_size-1:-1:0)

        hmm, forw = fit_hmm(hmm, N, ret, :init, i)
        p_t = forw[1][end, :]
        p_next = [sum(hmm.A[i,j]* p_t[i] for i in 1:K) for j in 1:K ]
        quantile_prediction = []
        alpha_1 = alpha/2
        alpha_2 = 1 - alpha/2
        
        quantile_prediction = find_quantile(N, p_next, hmm, alpha_1, alpha_2, quantile_prediction, K)


        if N == 1  
            upper_one_step_quantile[end-i] = quantile_prediction[1][1]
            lower_one_step_quantile[end-i] = quantile_prediction[1][2]

        else for n in 1:N one_step_quantile[n, end-i] = quantile_prediction[n] end
        end  
    end
    one_step_quantile = [upper_one_step_quantile,  lower_one_step_quantile]
    return (one_step_quantile, hmm)
end

function generate_sample(mu, sigma, n_states, sample_size)
    retornos = zeros(2, sample_size, n_states)
    for k in 1:n_states
         d = Normal(mu[k], sigma[k]) 
         r = rand(d, sample_size)
         retornos[1, :, k] = exp.(r)
    end
    retornos[2, :, :] = (ones(sample_size, n_states).+0.05).^(1/12)
    return retornos
end

function simulate_stage_trajectory(hmm)
    l1 = repeat([1], round(Int, 1000*hmm.A[1,1]))
    l1 = append!(l1, repeat([2], round(Int, 1000*hmm.A[1,2])))
    l1 = append!(l1, repeat([3], round(Int, 1000*hmm.A[1,3])))
    l1 = append!(l1, repeat([4], round(Int, 1000*hmm.A[1,4])))

    l2 = repeat([1], round(Int, 1000*hmm.A[2,1]))
    l2 = append!(l2, repeat([2], round(Int, 1000*hmm.A[2,2])))
    l2 = append!(l2, repeat([3], round(Int, 1000*hmm.A[2,3])))
    l2 = append!(l2, repeat([4], round(Int, 1000*hmm.A[2,4])))

    l3 = repeat([1], round(Int, 1000*hmm.A[3,1]))
    l3 = append!(l3, repeat([2], round(Int, 1000*hmm.A[3,2])))
    l3 = append!(l3, repeat([3], round(Int, 1000*hmm.A[3,3])))
    l3 = append!(l3, repeat([4], round(Int, 1000*hmm.A[3,4])))

    l4 = repeat([1], round(Int, 1000*hmm.A[4,1]))
    l4 = append!(l4, repeat([2], round(Int, 1000*hmm.A[4,2])))
    l4 = append!(l4, repeat([3], round(Int, 1000*hmm.A[4,3])))
    l4 = append!(l4, repeat([4], round(Int, 1000*hmm.A[4,4])))

    stages_trajectories = []
    start_stage = findmax(hmm.a)[2]
    stages_trajectories = push!(stages_trajectories, start_stage)
    c = 1
    while c < n_stages

    if stages_trajectories[end] == 1
        stages_trajectories = push!(stages_trajectories, rand(l1))
    elseif stages_trajectories[end] == 2
        stages_trajectories = push!(stages_trajectories, rand(l2))
    elseif stages_trajectories[end] == 3
        stages_trajectories = push!(stages_trajectories, rand(l3))
    elseif stages_trajectories[end] == 4
        stages_trajectories = push!(stages_trajectories, rand(l4))
    end
    c+=1
    end

    return stages_trajectories
end





function create_violations_series(historical_data, upper_limit, lower_limit)
    violations_series = zeros(size(historical_data, 1))
    for i in 1:size(violations_series, 1)
        if historical_data[i] > upper_limit[i] || historical_data[i] < lower_limit[i]
            violations_series[i] = 1
        end
    end
    return violations_series
end

############# functions for Markov Test ############# 
function L(violations_series, x, T_1, T)    
    return x^(T_1)*(1-x)^(T-T_1)
end

function L(violations_series, pi_01, pi_11, T_0, T_01, T_1, T_11)
    return (1-pi_01)^(T_0 - T_01) * pi_01^(T_01) * (1-pi_11)^(T_1 - T_11) * (pi_11)^(T_11)
end

function Markov_test_statistics(violations_series, VaR_probability)
    T_01 = 0
    T_11 = 0
    T = size(violations_series, 1)
    T_1 = sum(violations_series)
    T_0 = T - T_1
    
    for i in 2:size(violations_series, 1)
        if violations_series[i-1] == 0 && violations_series[i] == 1  T_01 += 1 
        elseif violations_series[i-1] == 1 && violations_series[i] == 1  T_11 += 1
        end
    end

    pi_1 = T_1 / T
    pi_01 = T_01/ T_0
    pi_11 = T_11/ T_1
    
    LR_uc = 2*(log(L(violations_series, pi_1, T_1, T)) - log(L(violations_series, VaR_probability, T_1, T)) )
    LR_ind = 2*( log( L(violations_series, pi_01, pi_11, T_0, T_01, T_1, T_11) ) - log( L(violations_series, pi_1, T_1, T) ) )
    LR_cc = 2*( log( L(violations_series, pi_01, pi_11, T_0, T_01, T_1, T_11) ) - log( L(violations_series, VaR_probability, T_1, T) ) )

    return LR_uc, LR_ind, LR_cc
end

function p_value(LR, degrees_of_freedom)
    return ccdf(Chisq(degrees_of_freedom), LR)
end

function testing_HMM(N, K, test_ret, hmm_tb, ret, alpha)
    test_size = size(test_ret,1)
    VaR_probability = alpha
    (quantiles, pos_hmm) = one_step_quantile_predictions(N, K, test_size, hmm_tb, ret, alpha)
    lower_limit = quantiles[1]
    upper_limit = quantiles[2]
    violations_series = create_violations_series(test_ret, upper_limit, lower_limit)
    LR_uc, LR_ind, LR_cc = Markov_test_statistics(violations_series, VaR_probability)
    p_v = p_value(LR_cc, 2)
    return p_v, lower_limit, upper_limit
end 

################### functions for joint test ###################
function Kupiec_statistic_test(violations_series, VaR_probability)
    
    n = size(violations_series,1)
    I = sum(violations_series)
    estimated_alpha = I/n
    alpha = VaR_probability
    Kupiec_statisitic = 2*log( ((1-estimated_alpha)/(1-alpha))^(n-I) * (estimated_alpha/alpha)^I)
    return Kupiec_statisitic
end

function Joint_test(Kupiec_statisitic, violations_series)
    N_01 = 0 
    N_11 = 0
    N_00 = 0
    N_10 = 0

    for i in 2:size(violations_series, 1)
        if violations_series[i-1] == 0 && violations_series[i] == 1  N_01 += 1 
        elseif violations_series[i-1] == 1 && violations_series[i] == 1  N_11 += 1
        elseif violations_series[i-1] == 0 && violations_series[i] == 0  N_00 += 1
        elseif violations_series[i-1] == 1 && violations_series[i] == 0  N_10 += 1
        end
    end

    pi_0 = N_01/(N_00 + N_01)
    pi_1 = N_11/(N_10 + N_11)
    pi_ = pi_0 + pi_1

    LR_ind = -2*log((1-pi_)^(N_00+N_01) * (pi_)^(N_01+N_11)) + 2*log((1-pi_0)^N_00 * pi_0^N_01 * (1-pi_1)^N_10 * pi_1^N_11) #### CHECAR
    LR = LR_ind + Kupiec_statisitic
    return LR
end


############### Basic Script functions ###############
function load_data(libor_path, returns_path)

    df_libor = CSV.read(libor_path, DataFrame)
    returns = CSV.read(returns_path, DataFrame)
    sp = returns[!, "Change %"]
    
    s = replace.(sp, ['.','%'] => "")
    s = parse.(Int64, s)
    sp = s./10000 .+ 1
    
    sp_return = Float64[]
    for i in 1:size(sp,1)-1
         append!(sp_return, sp[417-i])
    end
    
    libor = (1 .+ df_libor[!," value"]/12 ./100)
    ret = (sp_return./libor)
    return ret, libor
end

function load_log_ret_train_ret_test_ret(validation_size, test_size, ret)

    ret = log.(ret)

    validation_ret = ret[end-(validation_size)+1:end]
    test_ret = ret[end-(validation_size)-test_size:end-(validation_size)-1]
    train_ret = ret[1:end-(validation_size)-test_size-1]
    
    
    
    return train_ret, test_ret, validation_ret

end

function initial_hmm(K, N, train_ret)

    sig = var(train_ret)
    if N >= 2 pre_hmm = HMM(randtransmat(K), [MvNormal(rand(N), sig) for k=1:K])
    else pre_hmm = HMM(randtransmat(K), [Normal(0, 1) for k=1:K])
    end

    hmm, forw = fit_hmm(pre_hmm, N, train_ret, :kmeans, 0)

    return hmm
end

function initial_hmm(K, N, train_ret, pre_hmm)

    hmm, forw = fit_hmm(pre_hmm, N, train_ret, :kmeans, 0)

    return hmm
end

