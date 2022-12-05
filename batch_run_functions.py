# Some more magic so that the notebook will reload external python modules;
# see http://stackoverflow.com/questions/1907993/autoreload-of-modules-in-ipython

import pandas as pd
import numpy as np
import time
import sys
sys.path.append("..")
sys.path.append('/content/IE-541-Project')
import knapsack_EA_functions as ea


# return item_value, item_weight, joint_profit, penalty_factor
def warp_getdata(url):
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
    #------------------------------------------------------------------------
    max_c = 0; max_c = max(max_c,max(np.sum(joint_profit,axis = 1))); max_c = max(max_c,max(np.sum(joint_profit,axis = 0)))
    max_i = max(item_value)
    penalty_factor = max_i+max_c
    return item_value, item_weight, joint_profit, penalty_factor

def result_writer(instance_name, running_results):
    # instance_name, num_of_knapsack,
    #       running_result1, running_result3):
    instance_meta_information = []
    decision_matrices = []
    for case in running_results:
        GA_type, num_of_knapsack, objc, objfValue, time_cost, num_gen, best_ind = case
        case_meta = (instance_name, objc, num_of_knapsack, objfValue, GA_type, time_cost, num_gen)
        ind_knap_location = [0] * len(best_ind)
        for x in range(0, len(best_ind)):
            i = 1
            if i in best_ind[x]:
                ind_knap_location[x] = tuple(best_ind[x]).index(1) + 1
            else:
                ind_knap_location[x] = 0
        print(ind_knap_location)
        decision_matrices.append(ind_knap_location)
        instance_meta_information.append(case_meta)
    df_decision_matrices = pd.DataFrame(decision_matrices)
    # df_decision_matrices = df_decision_matrices.T
    df_meta_info = pd.DataFrame(instance_meta_information)
    df_meta_info = df_meta_info.T
    # print(df_decision_matrices)
    # print(df_meta_info)

    df_meta_info = df_meta_info.append(df_decision_matrices.T, ignore_index = True)
    df_meta_info.to_csv("compareGA_"+instance_name+".csv")

def run_compare_GA(urls, instances):
    for i in range(len(urls)):
        ins = instances[i]
        url = urls[i]
        debug_result = []
        # 'https://raw.githubusercontent.com/jekim526/IE-541-Project/main/data/Z_r_100_25_1.csv'
        print(url)
        item_value, item_weight, joint_profit, penalty_factor = warp_getdata(url)
        running_results = []
        for num_of_knapsack in [3,5,10]:
            capacity = (sum(item_weight)/num_of_knapsack)*0.8 # 80% of the sum of all item weights divided by the number of knapsacks
            capacities = (capacity,)*num_of_knapsack
            instance_settings = (item_value, item_weight, joint_profit, capacities)
            for GA_type in ['base', 'tugba']:
                if GA_type == 'base':
                    GA_solver = ea.perform_GA_base
                    evolution_general_parameters = (100, 0.2, 0.02, penalty_factor)
                    # evolution_specify_parameters = (0.7, 0.5, 12000, 20000)
                    evolution_specify_parameters = (0.7, 0.5, 50, 20000)
                else:
                    GA_solver = ea.perform_GA_tugba
                    evolution_general_parameters = (50, 0.2, 0.02, penalty_factor)
                    # evolution_specify_parameters = (0.7, 0.3, 1000, 20000)
                    evolution_specify_parameters = (0.7, 0.3, 5, 20000)
                for objc in [1,3]:
                    time_start = time.time()
                    best_ind,_,num_gen,_ = GA_solver(objc, instance_settings, evolution_general_parameters, evolution_specify_parameters)
                    time_end = time.time()
                    time_cost = time_end - time_start
                    objfValue = best_ind.fitness.values
                    case = [GA_type, objc, num_of_knapsack, objfValue, time_cost, num_gen, best_ind]
                    running_results.append(case)
        debug_result.append(running_results)
        result_writer(ins, running_results)
