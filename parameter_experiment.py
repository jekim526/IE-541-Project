import sys
sys.path.append("..")
sys.path.append('/content/IE-541-Project')
import pandas as pd
import numpy as np
import knapsack_EA_functions as ea
import operators as op
import numpy
import random
import time
import matplotlib.pyplot as plt
import time




def search_parameter(url, evolution_specify_parameters, ga_type, rounds = 10, popsize = 100):
    ''' GET DATA '''
    df1_o = pd.read_csv(url)
    df1 = pd.DataFrame(df1_o).to_numpy() # DataFrame to nummpy array
    n_size = len(df1) - 1

    # Array of item value
    item_value = [0] * n_size
    for x in range(0,n_size):
        item_value[x] = df1[1,x+1]

    # Array of item weight
    item_weight = [0] * n_size
    for x in range(0,n_size):
        item_weight[x] = df1[0,x+1]

    # n x n array of joint profit
    joint_profit = np.zeros((n_size,n_size))
    for x in range(0, n_size-1):
        for y in range(0, n_size):
            joint_profit[x,y] = df1[x+2,y+1]

    item_weight = np.array(item_weight)
    item_value = np.array(item_value)


    # -------  NOTES: ------------------------------------------------------------------------------
    # instance_setings should be a tuple contains:
    # {item_value, item_weight, joint_profit, capacities}
    #                 capacities = {capacity1, capacity2, ...}
    #
    # evolution_general_parameters should be a tuple contains:
    # {popsize, swap_prob, mute_prob, punish_factor}, in which:
    #    swap_prob is independent probability for swap at each point in uniform crossover.
    #    mute_prob is independent probability for each attribute to be flipped in flip-bit mutation.
    #
    # evolution_specify_parameters should be a tuple contains: {CXPB, MUTPB, MAX_GEN, STOP_GEN},In which:
    #    CXPB is the probability with which two individuals
    #          are crossed
    #    MUTPB is the probability for mutating an individual
    #    MAX_GEN is the maximum generation threshold
    #    STOP_GEN is the threshold of no progress generations

    ''' the maximize covalue_case '''
    max_c = 0; max_c = max(max_c,max(np.sum(joint_profit,axis = 1))); max_c = max(max_c,max(np.sum(joint_profit,axis = 0)))
    max_i = max(item_value)

    ''' prepare this solver inputs: '''
    #initialize evolution parameter setting
    evolution_general_parameters = (popsize, 0.2, 0.02, max_i + max_c)
    # evolution_specify_parameters = (0.5, 0.25, 1000, 1000)
    num_of_knapsack = 5
    capacity = (sum(item_weight)/num_of_knapsack)*0.8 # 80% of the sum of all item weights divided by the number of knapsacks
    capacities = (capacity,)*num_of_knapsack
    #initialize instance setting
    instance_settings = (item_value, item_weight, joint_profit, capacities)

    #repeat experiment

    result = []

    if ga_type == 0:
        ga_fuc = ea.perform_GA_base
    elif ga_type == 1:
        ga_fuc = ea.perform_GA_tugba
    else:
        return 0

    objf_type = 1
    for i in range(rounds):
        best_ind,pop,num_gen,gen_log = ga_fuc(objf_type, instance_settings, evolution_general_parameters, evolution_specify_parameters, PRINT=False)
        result.append(best_ind.fitness.values)
    mean_obj = np.mean(result)
    return mean_obj

