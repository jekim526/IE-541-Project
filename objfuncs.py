# -*- coding: utf-8 -*-
"""
Created on Fri Nov  4 11:44:15 2022

@author: Karlz
"""

import math

import numpy as np

''' Punish '''


def punish(decision_matrix: np.ndarray,  # n*m (m:number of knapsack, n: number of items)
           item_weight: np.ndarray,  # n,
           capacities: list,  # m
           punish_factor: float
           ) -> float:
    punishment = 0
    for k in range(decision_matrix.shape[1]):
        kth_decision = decision_matrix[:, k]
        weight = np.dot(kth_decision, item_weight)
        if weight > capacities[k]:
            punishment += punish_factor * (weight - capacities[k])
    return punishment


''' objf 1 '''


def total_profit(decision_matrix: np.ndarray,  # n*m (m:number of knapsack, n: number of items)
                 item_value: np.ndarray,  # n,
                 item_weight: np.ndarray,  # n,
                 joint_profit: np.ndarray,  # n*n
                 ) -> float:
    total_profit_value = 0
    item_value = item_value.reshape(len(item_value), 1)
    for k in range(decision_matrix.shape[1]):
        kth_decision = decision_matrix[:, k]  # 1*n
        indiv_sum = np.dot(kth_decision, item_value)
        joint_sum = np.dot(np.dot(kth_decision, joint_profit), kth_decision.T)
        total_profit_value += (indiv_sum + joint_sum)
    return total_profit_value


''' objf 2 '''


def total_weight(decision_matrix: np.ndarray,  # n*m (m:number of knapsack, n: number of items)
                 item_value: np.ndarray,  # n,
                 item_weight: np.ndarray,  # n,
                 joint_profit: np.ndarray,  # n*n
                 ) -> float:
    # total_weight_value = 0
    item_weight = item_weight.reshape(len(item_weight), 1)
    # for k in range(decision_matrix.shape[1]):
    #     kth_decision = decision_matrix[:, k]
    #     total_weight_value += np.dot(kth_decision, item_weight)
    total_weight_value = np.dot(np.sum(decision_matrix, axis=1), item_weight)
    # if total_weight_value2 != total_weight_value:
    #     raise RuntimeError('>>>')
    return -total_weight_value


''' objf 3 '''


def min_indiv_profit(decision_matrix: np.ndarray,  # n*m (m:number of knapsack, n: number of items)
                     item_value: np.ndarray,  # n,
                     item_weight: np.ndarray,  # n,
                     joint_profit: np.ndarray,  # n*n
                     ) -> float:
    min_indiv_profit_value = math.inf
    item_value = item_value.reshape(len(item_value), 1)
    for k in range(decision_matrix.shape[1]):
        kth_decision = decision_matrix[:, k]  # 1*n
        indiv_sum = np.dot(kth_decision, item_value)
        joint_sum = np.dot(np.dot(kth_decision, joint_profit), kth_decision.T)
        min_indiv_profit_value = min(min_indiv_profit_value, indiv_sum + joint_sum)
    return min_indiv_profit_value


def objf_base(decision_matrix, objc, instance_settings, punish_factor=-100):
    # instance_setings should be a tuple contains: {item_value, item_weight, joint_profit, capacities}
    #                                                           capacities = {capacity1, capacity2, ...}
    if objc == 1:
        func = total_profit
    elif objc == 2:
        func = total_weight
    elif objc == 3:
        func = min_indiv_profit
    else:
        print("wrong input obj")
        return
    item_value, item_weight, joint_profit, capacities = instance_settings
    obj_value = func(decision_matrix, item_value, item_weight, joint_profit)
    punish_value = punish(decision_matrix, item_weight, capacities, punish_factor)
    return obj_value + punish_value


''' calculate all 3 objectives and return in a tuple '''


def objfuncs(decision_matrix: np.ndarray,  # n*m (m:number of knapsack, n: number of items)
             item_value: np.ndarray,  # n,
             item_weight: np.ndarray,  # n,
             joint_profit: np.ndarray,  # n*n
             ) -> (float, float, float):
    item_value = item_value.reshape(len(item_value), 1)
    item_weight = item_value.reshape(len(item_weight), 1)
    total_profit_value = 0;
    # total_weight = 0;
    min_indiv_profit_value = math.inf
    total_weight_value = np.dot(np.sum(decision_matrix, axis=1), item_weight)
    for k in range(decision_matrix.shape[1]):
        kth_decision = decision_matrix[:, k]  # 1*n
        indiv_sum = np.dot(kth_decision, item_value)
        joint_sum = np.dot(np.dot(kth_decision, joint_profit), kth_decision.T)
        profit_sum = indiv_sum + joint_sum
        total_profit_value += profit_sum
        # total_weight += np.dot(kth_decision, item_weight)
        min_indiv_profit_value = min(min_indiv_profit_value, profit_sum)
    return total_profit_value[0], -total_weight_value, min_indiv_profit_value[0]

def objfuncs_fast(decision_matrix: np.ndarray,  # n*m (m:number of knapsack, n: number of items)
             item_value: np.ndarray,  # n,
             item_weight: np.ndarray,  # n,
             joint_profit: np.ndarray,  # n*n
             ) -> (float, float, float):
    # item_value = item_value.reshape(len(item_value), 1)
    # item_weight = item_value.reshape(len(item_weight), 1)
    total_profit_value = 0;
    # total_weight = 0;
    min_indiv_profit_value = math.inf
    total_weight_value = np.dot(np.sum(decision_matrix, axis=1), item_weight)[0]
    for k in range(decision_matrix.shape[1]):
        kth_decision = decision_matrix[:, k]  # 1*n
        indiv_sum = np.dot(kth_decision, item_value)
        joint_sum = np.dot(np.dot(kth_decision, joint_profit), kth_decision.T)
        profit_sum = indiv_sum + joint_sum
        total_profit_value += profit_sum
        # total_weight += np.dot(kth_decision, item_weight)
        min_indiv_profit_value = min(min_indiv_profit_value, profit_sum)
    return total_profit_value[0], -total_weight_value, min_indiv_profit_value[0]


##
def objf_weight(decision_matrix, objc_weight_vector, instance_settings, punish_factor=-100):
    item_value, item_weight, joint_profit, capacities = instance_settings
    obj_value_vector = np.array(objfuncs(decision_matrix, item_value, item_weight, joint_profit))
    obj_value = np.dot(np.squeeze(obj_value_vector), objc_weight_vector)
    punish_value = punish(decision_matrix, item_weight, capacities, punish_factor)
    return [obj_value + punish_value]
