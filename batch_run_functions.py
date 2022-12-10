# Some more magic so that the notebook will reload external python modules;
# see http://stackoverflow.com/questions/1907993/autoreload-of-modules-in-ipython

import pandas as pd
import numpy as np
import time
import os
import sys

sys.path.append("..")
sys.path.append('/content/IE-541-Project')
import knapsack_EA_functions as ea
import operators as op
import objfuncs as objf


# return item_value, item_weight, joint_profit, penalty_factor
def warp_getdata(url):
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
    # ------------------------------------------------------------------------
    max_c = 0;
    max_c = max(max_c, max(np.sum(joint_profit, axis=1)));
    max_c = max(max_c, max(np.sum(joint_profit, axis=0)))
    max_i = max(item_value)
    penalty_factor = max_i + max_c
    return item_value, item_weight, joint_profit, penalty_factor


def result_writer(instance_name, running_results):
    # output the meta information (objfuction, instance_name, num_of_knapsack) and the solution (the best individual)
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
        decision_matrices.append(ind_knap_location)
        instance_meta_information.append(case_meta)
    df_decision_matrices = pd.DataFrame(decision_matrices)
    # df_decision_matrices = df_decision_matrices.T
    df_meta_info = pd.DataFrame(instance_meta_information)
    df_meta_info = df_meta_info.T
    # print(df_decision_matrices)
    # print(df_meta_info)

    df_meta_info = df_meta_info.append(df_decision_matrices.T, ignore_index=True)
    df_meta_info.to_csv("compareGA_" + instance_name + ".csv")


def result_writer2(instance_name, running_results):
    # output the meta information (objfuction, instance_name, num_of_knapsack) and the objf in each generation
    instance_meta_information = []
    log = []
    for case in running_results:
        GA_type, num_of_knapsack, objc, objfValue, time_cost, num_gen, gen_log = case
        case_meta = (instance_name, objc, num_of_knapsack, objfValue, GA_type, time_cost, num_gen)
        log.append(gen_log)
        instance_meta_information.append(case_meta)
    df_log = pd.DataFrame(log)
    # df_decision_matrices = df_decision_matrices.T
    df_meta_info = pd.DataFrame(instance_meta_information)
    df_meta_info = df_meta_info.T
    # print(df_decision_matrices)
    # print(df_meta_info)

    df_meta_info = df_meta_info.append(df_log.T, ignore_index=True)
    df_meta_info.to_csv("compareGA_" + instance_name + ".csv")


def run_compare_GA(urls, instances):
    print("running begins!...")
    for i in range(len(urls)):
        start_tt = time.time()
        ins = instances[i]
        url = urls[i]
        debug_result = []
        # 'https://raw.githubusercontent.com/jekim526/IE-541-Project/main/data/Z_r_100_25_1.csv'
        print(url)
        item_value, item_weight, joint_profit, penalty_factor = warp_getdata(url)
        running_results = []
        for num_of_knapsack in [3, 5, 10]:
            capacity = (
                               sum(item_weight) / num_of_knapsack) * 0.8  # 80% of the sum of all item weights divided by the number of knapsacks
            capacities = (capacity,) * num_of_knapsack
            instance_settings = (item_value, item_weight, joint_profit, capacities)
            for GA_type in ['base', 'tugba']:
                if GA_type == 'base':
                    GA_solver = ea.perform_GA_base
                    evolution_general_parameters = (100, 0.2, 0.02, penalty_factor)
                    evolution_specify_parameters = (0.7, 0.5, 6000, 20000)
                    # evolution_specify_parameters = (0.7, 0.5, 50, 20000)
                else:
                    GA_solver = ea.perform_GA_tugba
                    evolution_general_parameters = (50, 0.2, 0.02, penalty_factor)
                    evolution_specify_parameters = (0.7, 0.3, 500, 20000)
                    # evolution_specify_parameters = (0.7, 0.3, 5, 20000)
                for objc in [1, 3]:
                    time_start = time.time()
                    best_ind, _, num_gen, _ = GA_solver(objc, instance_settings, evolution_general_parameters,
                                                        evolution_specify_parameters)
                    time_end = time.time()
                    time_cost = time_end - time_start
                    objfValue = best_ind.fitness.values
                    case = [GA_type, objc, num_of_knapsack, objfValue, time_cost, num_gen, best_ind]
                    running_results.append(case)
        end_tt = time.time()
        print("the ", i, "th instance done. Time:", end_tt - start_tt, "s")
        result_writer(ins, running_results)


def run_compare_GA_rerun(urls, instances, GA_type, objs, knapsack_nums, t_gen, b_gen, tail):
    print("running begins!...")
    for i in range(len(urls)):
        start_tt = time.time()
        ins = instances[i]
        url = urls[i]
        debug_result = []
        # 'https://raw.githubusercontent.com/jekim526/IE-541-Project/main/data/Z_r_100_25_1.csv'
        print(url)
        item_value, item_weight, joint_profit, penalty_factor = warp_getdata(url)
        running_results = []
        for num_of_knapsack in knapsack_nums:
            capacity = (
                               sum(item_weight) / num_of_knapsack) * 0.8  # 80% of the sum of all item weights divided by the number of knapsacks
            capacities = (capacity,) * num_of_knapsack
            instance_settings = (item_value, item_weight, joint_profit, capacities)
            # for GA_type in ['base', 'tugba']:
            for GA_type in GA_type:
                if GA_type == 'base':
                    GA_solver = ea.perform_GA_base
                    evolution_general_parameters = (100, 0.2, 0.02, penalty_factor)
                    evolution_specify_parameters = (0.7, 0.5, b_gen, 20000)
                    # evolution_specify_parameters = (0.7, 0.5, 50, 20000)
                else:
                    GA_solver = ea.perform_GA_tugba
                    evolution_general_parameters = (50, 0.2, 0.02, penalty_factor)
                    evolution_specify_parameters = (0.7, 0.3, t_gen, 20000)
                    # evolution_specify_parameters = (0.7, 0.3, 5, 20000)
                for objc in objs:
                    time_start = time.time()
                    best_ind, _, num_gen, gen_log = GA_solver(objc, instance_settings, evolution_general_parameters,
                                                              evolution_specify_parameters)
                    time_end = time.time()
                    time_cost = time_end - time_start
                    objfValue = best_ind.fitness.values
                    case = [GA_type, objc, num_of_knapsack, objfValue, time_cost, num_gen, gen_log]
                    running_results.append(case)
        end_tt = time.time()
        print("the ", i, "th instance done. Time:", end_tt - start_tt, "s")
        ins = ins[:-4]
        ins = ins + tail
        result_writer2(ins, running_results)


def run_random_shit_old(instance_dir, sample_number):
    empty_pb = 0
    instances = os.listdir(instance_dir)
    start_tt = time.time()
    knapsack_nums = [3, 5, 10]
    # sample_number = 500000
    for ins in instances:
        print(" + start run instance: ", ins)
        count = 0
        path = instance_dir + ins
        start_tt = time.time()
        # 'https://raw.githubusercontent.com/jekim526/IE-541-Project/main/data/Z_r_100_25_1.csv'
        item_value, item_weight, joint_profit, penalty_factor = warp_getdata(path)
        item_num = item_weight.size
        random_results = [0] * sample_number * len(knapsack_nums)
        for knapsack_num in knapsack_nums:
            print("start run knapsack: ", knapsack_num)
            for i in range(sample_number):
                if knapsack_num == 3:
                    empty_pb = 0.25
                elif knapsack_num == 5:
                    empty_pb = 3 / 8
                elif knapsack_num == 10:
                    empty_pb = 3 / 8
                capacity = (
                                   sum(item_weight) / knapsack_num) * 0.8  # 80% of the sum of all item weights divided by the number of knapsacks
                capacities = (capacity,) * knapsack_num
                ind = np.zeros([item_num, knapsack_num])
                for i in range(item_num):
                    ind[i] = op.rand_oneHotVector(knapsack_num, empty_pb)
                # print(np.sum(ind)/ind.shape[0])
                # def objfuncs(decision_matrix: np.ndarray,  # n*m (m:number of knapsack, n: number of items)
                #              item_value: np.ndarray,  # n,
                #              item_weight: np.ndarray,  # n,
                #              joint_profit: np.ndarray,  # n*n
                #              ) -> (float, float, float):
                total_profit_value, total_weight_value, min_indiv_profit_value = objf.objfuncs(ind, item_value,
                                                                                               item_weight,
                                                                                               joint_profit)
                feasibility = op.exam_feasibility_fast(ind, item_weight, capacities)
                random_results[count] = [total_profit_value, -total_weight_value, min_indiv_profit_value, feasibility]
                count += 1
        df_random_results = pd.DataFrame(random_results)
        df_random_results.to_csv("RandomPoint_" + ins[:-4] + ".csv")
        print(" + output instance: ", ins)
    end_tt = time.time()
    print(end_tt - start_tt)


def run_random_shit(instance_dir, sample_number):
    empty_pb = 0
    instances = os.listdir(instance_dir)
    start_tt = time.time()
    knapsack_nums = [3, 5, 10]
    # sample_number = 500000
    for ins in instances:
        print(" + start run instance: ", ins)
        count = 0
        path = instance_dir + ins
        start_tt = time.time()
        # 'https://raw.githubusercontent.com/jekim526/IE-541-Project/main/data/Z_r_100_25_1.csv'
        item_value, item_weight, joint_profit, penalty_factor = warp_getdata(path)
        item_num = item_weight.size
        item_value = item_value.reshape(len(item_value), 1)
        item_weight = item_value.reshape(len(item_weight), 1)
        random_results = [0] * sample_number * len(knapsack_nums)
        for knapsack_num in knapsack_nums:
            if knapsack_num == 3:
                empty_pb = 0.25
            elif knapsack_num == 5:
                empty_pb = 3 / 8
            elif knapsack_num == 10:
                empty_pb = 3 / 8
            start_tk = time.time()
            capacity = (sum(item_weight) / knapsack_num) * 0.8  # 80% of the sum of all item weights
            capacities = (capacity,) * knapsack_num
            infeasible_tuple = (knapsack_num, -1, -1, -1, False)
            print("start run knapsack: ", knapsack_num)
            # nums_of_picked_item = np.random.standard_normal(sample_number)*(item_num/25)+(item_num*(1-empty_pb))
            nums_of_picked_item = np.random.binomial(item_num, p=(1 - empty_pb), size=sample_number)
            for i in range(sample_number):
                ind = np.zeros([item_num, knapsack_num])
                item_index = np.random.choice(item_num, size=int(nums_of_picked_item[i]), replace=False)
                knapsack_index = np.random.randint(knapsack_num, size=int(nums_of_picked_item[i]))
                ind[item_index, knapsack_index] = 1
                feasibility = op.exam_feasibility_fast(ind, item_weight, capacities)
                if feasibility:
                    total_profit_value, total_weight_value, min_indiv_profit_value = objf.objfuncs_fast(ind, item_value,
                                                                                                        item_weight,
                                                                                                        joint_profit)
                    random_results[count] = (
                        knapsack_num, total_profit_value, -total_weight_value, min_indiv_profit_value, feasibility)
                else:
                    random_results[count] = infeasible_tuple
                count += 1
                end_tk = time.time()
            print("finished ", knapsack_num, "th knapsack, time: ", end_tk - start_tk)
        df_random_results = pd.DataFrame(random_results)
        df_random_results.to_csv("RandomPoint_" + ins[:-4] + ".csv")
        print(" + output instance: ", ins)
    end_tt = time.time()
    print(end_tt - start_tt)
