import math
import random
import numpy
from deap import base


def rand_oneHotVector(length, empty_pb=0.3):
    vec = numpy.zeros(length).astype(int)
    if random.random() > empty_pb:
        hot = random.randint(0, length - 1)
        vec[hot] = 1
    return vec


def cxUniform_free(ind1, ind2, prob=0.1):
    size = len(ind1)
    for i in range(size):
        if random.random() < prob:
            ind1[i], ind2[i] = ind2[i].copy(), ind1[i].copy()
    return ind1, ind2


def mutUniformVec_free(individual, indpb):
    item_num = individual.shape[0]
    knapsack_num = individual.shape[1]
    for i in range(item_num):
        if random.random() < indpb:
            individual[i] = rand_oneHotVector(knapsack_num)
    return individual,


def exam_feasibility(individual, instance_settings):
    result = []
    item_weight = instance_settings[1]
    capacities = instance_settings[3]
    for k in range(individual.shape[1]):
        kth_knapsack = individual[:, k]
        weight = numpy.dot(kth_knapsack, item_weight)
        if weight > capacities[k]:
            result.append(False)
        else:
            result.append(True)
    return result


'''
Mutation 1, improves on the solution though local exchanges. 
First, it considers all pairs of items assigned to different knapsacks and, 
if possible and if the total profit increases, interchanges them.
'''
def mutLocalSearch()