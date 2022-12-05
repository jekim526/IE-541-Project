from scoop import futures
import multiprocessing
import random
from deap import base
from deap import creator
from deap import tools
import numpy
import pandas as pd
import numpy as np
import time
import matplotlib.pyplot as plt
import sys

sys.path.append("..")
sys.path.append('/content/IE-541-Project')
import knapsack_EA_functions as ea
import operators as op



def GA_base_normal(objf_type, instance_settings, evolution_general_parameters, evolution_specify_parameters):
    time_start = time.time()
    best_ind, pop, num_gen, gen_log = ea.perform_GA_base(objf_type, instance_settings, evolution_general_parameters,
                                                         evolution_specify_parameters, PRINT=False)
    # best_ind,pop,num_gen,gen_log = ea.perform_GA_tugba((objf_type, instance_settings, evolution_general_parameters,
    # evolution_specify_parameters, PRINT=False) print(best_ind)
    time_end = time.time()
    print('[normal]time cost', time_end - time_start, 's')
    print(best_ind.fitness.values)
    return


def GA_base_multi(objf_type, instance_settings, evolution_general_parameters, evolution_specify_parameters):
    time_start = time.time()
    best_ind, pop, num_gen, gen_log = ea.perform_GA_base_mt(objf_type, instance_settings, evolution_general_parameters,
                                                            evolution_specify_parameters, PRINT=False)
    # best_ind,pop,num_gen,gen_log = ea.perform_GA_tugba_mt((objf_type, instance_settings,
    # evolution_general_parameters, evolution_specify_parameters, PRINT=False) print(best_ind)
    time_end = time.time()
    print('[multi]time cost', time_end - time_start, 's')
    print(best_ind.fitness.values)
    return


if __name__ == "__main__":
    url = 'https://raw.githubusercontent.com/jekim526/IE-541-Project/main/data/Z_r_100_25_1.csv'  # Get the "Raw" link
    df1_o = pd.read_csv(url)
    df1 = pd.DataFrame(df1_o).to_numpy()  # DataFrame to nummpy array
    n_size = len(df1) - 1

    # Array of item value
    item_value = [0] * n_size
    for x in range(0, n_size):
        item_value[x] = df1[1, x + 1]

    # Array of item weight
    item_weight = [0] * n_size
    for x in range(0, n_size):
        item_weight[x] = df1[0, x + 1]

    # n x n array of joint profit
    joint_profit = np.zeros((n_size, n_size))
    for x in range(0, n_size - 1):
        for y in range(0, n_size):
            joint_profit[x, y] = df1[x + 2, y + 1]

    item_weight = np.array(item_weight)
    item_value = np.array(item_value)
    ''' the maximize covalue_case '''
    max_c = 0;
    max_c = max(max_c, max(np.sum(joint_profit, axis=1)));
    max_c = max(max_c, max(np.sum(joint_profit, axis=0)))
    max_i = max(item_value)

    ''' prepare this solver inputs: '''
    # initialize evolution parameter setting
    evolution_general_parameters = (100, 0.2, 0.02, max_i + max_c)
    evolution_specify_parameters = (0.5, 0.25, 20000, 100000)
    num_of_knapsack = 5
    capacity = (sum(item_weight) / num_of_knapsack) * 0.8  # 80% of the sum of all item weights divided by the number of knapsacks
    capacities = (capacity,) * num_of_knapsack
    objf_type = 1
    # initialize instance setting
    instance_settings = (item_value, item_weight, joint_profit, capacities)
    GA_base_normal(objf_type, instance_settings, evolution_general_parameters, evolution_specify_parameters)
    GA_base_multi(objf_type, instance_settings, evolution_general_parameters, evolution_specify_parameters)

