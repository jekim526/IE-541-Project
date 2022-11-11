# -*- coding: utf-8 -*-
"""
Created on Fri Nov  4 11:44:15 2022

@author: Karlz
"""

import math
import numpy as np

''' Punish '''


def punish(decision_matrix: np.ndarray,  # m*n (m:number of knapsack, n: number of items)
           item_weight: np.ndarray,  # n,
           capacity: list,  # m
           punish_factor: float
           ) -> float:
    punishment = 0
    for k in range(decision_matrix.shape[0]):
        kth_decision = decision_matrix[k]
        weight = np.dot(kth_decision, item_weight)[0]
        if weight > capacity[k]:
            punishment += punish_factor * (weight - capacity)
    return punishment


''' objf 1 '''


def total_profit(decision_matrix: np.ndarray,  # m*n (m:number of knapsack, n: number of items)
                 item_value: np.ndarray,  # n,
                 item_weight: np.ndarray,  # n,
                 joint_profit: np.ndarray,  # n*n
                 ) -> float:
    total_profit = 0
    item_value = item_value.reshape(len(item_value), 1)
    for k in range(decision_matrix.shape[0]):
        kth_decision = decision_matrix[k]  # 1*n
        indiv_sum = np.dot(kth_decision, item_value)[0]
        joint_sum = np.dot(np.dot(kth_decision, joint_profit), kth_decision.T)[0][0]
        total_profit += (indiv_sum + joint_sum)
    return total_profit


''' objf 2 '''


def total_weight(decision_matrix: np.ndarray,  # m*n (m:number of knapsack, n: number of items)
                 item_value: np.ndarray,  # n,
                 item_weight: np.ndarray,  # n,
                 joint_profit: np.ndarray,  # n*n
                 ) -> float:
    total_weight = 0
    item_weight = item_value.reshape(len(item_weight), 1)
    for k in range(decision_matrix.shape[0]):
        kth_decision = decision_matrix[k]
        total_weight += np.dot(kth_decision, item_weight)[0]
    return total_weight


''' objf 3 '''


def min_indiv_profit(decision_matrix: np.ndarray,  # m*n (m:number of knapsack, n: number of items)
                     item_value: np.ndarray,  # n,
                     item_weight: np.ndarray,  # n,
                     joint_profit: np.ndarray,  # n*n
                     ) -> float:
    min_indiv_profit = math.inf
    item_value = item_value.reshape(len(item_value), 1)
    for k in range(decision_matrix.shape[0]):
        kth_decision = decision_matrix[k]  # 1*n
        indiv_sum = np.dot(kth_decision, item_value)[0]
        joint_sum = np.dot(np.dot(kth_decision, joint_profit), kth_decision.T)[0][0]
        min_indiv_profit = min(min_indiv_profit, indiv_sum + joint_sum)
    return min_indiv_profit


''' calculate all 3 objectives and return in a tuple '''


def objfuncs(decision_matrix: np.ndarray,  # m*n (m:number of knapsack, n: number of items)
             item_value: np.ndarray,  # n,
             item_weight: np.ndarray,  # n,
             joint_profit: np.ndarray,  # n*n
             ) -> (float, float, float):
    total_profit = 0;
    total_weight = 0;
    min_indiv_profit = math.inf
    item_value = item_value.reshape(len(item_value), 1)
    item_weight = item_value.reshape(len(item_weight), 1)
    for k in range(decision_matrix.shape[0]):
        kth_decision = decision_matrix[k]  # 1*n
        indiv_sum = np.dot(kth_decision, item_value)[0]
        joint_sum = np.dot(np.dot(kth_decision, joint_profit), kth_decision.T)[0][0]
        profit_sum = indiv_sum + joint_sum
        total_profit += profit_sum
        total_weight += np.dot(kth_decision, item_weight)[0]
        min_indiv_profit = min(min_indiv_profit, profit_sum)
    return (total_profit, total_weight, min_indiv_profit)


