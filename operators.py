import math
import random
import numpy


def rand_oneHotVector(length, empty_pb=0.3):
    vec = numpy.zeros(length).astype(int)
    if random.random() > empty_pb:
        hot = random.randint(0, length - 1)
        vec[hot] = 1
    return vec


# TODO
def remove_to_feasible(pop, item_weight, capacities):
    pop_size = len(pop)
    item_num = pop[0].shape[0]
    knapsack_num = pop[0].shape[1]
    for ind in pop:
        if exam_feasibility(ind, item_weight, capacities).all():
            continue
        contained_items = numpy.where(numpy.sum(ind, axis=1) == 1)[0]
        random.shuffle(contained_items)
        for i in contained_items:
            ind[i] = numpy.zeros(knapsack_num)
            if exam_feasibility(ind, item_weight, capacities).all():
                break
    return pop


'''-----------------    CROSSOVERS    -----------------'''


def cxUniform_free(ind1, ind2, prob=0.1):
    for i in range(len(ind1)):
        if random.random() < prob:
            ind1[i], ind2[i] = ind2[i].copy(), ind1[i].copy()
    return ind1, ind2


def cxUniform_restrict(ind1, ind2, instance_settings, prob=0.1):
    item_weight = instance_settings[1]
    capacities = instance_settings[3]
    rcapcity1 = capacities - numpy.dot(ind1.T, item_weight)
    rcapcity2 = capacities - numpy.dot(ind2.T, item_weight)
    num_knapsack = len(ind1[0])
    diff_gen_list = numpy.where(ind1 != ind2)[0]
    random.shuffle(diff_gen_list)
    for i in diff_gen_list:
        if random.random() < prob:
            ctw = item_weight[i]
            change_ind1 = (ind2[i] - ind1[i]) * ctw
            change_ind2 = (ind1[i] - ind2[i]) * ctw
            preserve_1 = ind1[i].copy()
            preserve_2 = ind2[i].copy()
            if (change_ind1 <= rcapcity1).all():  # this change did not exceed capacity
                ind1[i] = preserve_2
                rcapcity1 += change_ind1
            else:
                ind1[i] = numpy.zeros(num_knapsack)
                rcapcity1 -= ind1[i] * ctw
            if (change_ind2 <= rcapcity2).all():  # this change did not exceed capacity
                ind2[i] = preserve_1
                rcapcity2 += change_ind2
            else:
                ind2[i] = numpy.zeros(num_knapsack)
                rcapcity2 -= ind2[i] * ctw
    return ind1, ind2


def random_complete(ind, item_weight, capacities):
    # item_weight = instance_settings[1]
    # capacities = instance_settings[3]
    rcapcity = capacities - numpy.dot(ind.T, item_weight)
    # ---------------------------------------------------------
    not_contained_items = numpy.where(numpy.sum(ind, axis=1) == 0)[0]
    random.shuffle(not_contained_items)
    # ---------------------------------------------------------
    num_knapsack = len(ind[0])
    knapsack_list = list(range(num_knapsack))
    random.shuffle(knapsack_list)
    for i in not_contained_items:
        for k in knapsack_list:
            if item_weight[i] < rcapcity[k]:
                ind[i] = numpy.zeros(num_knapsack)
                ind[i, k] = 1
                rcapcity[k] -= item_weight[i]
                break
    return ind


''' -----------------    MUTATIONS    -----------------'''


def mutUniformVec_free(individual, indpb):
    item_num = individual.shape[0]
    knapsack_num = individual.shape[1]
    for i in range(item_num):
        if random.random() < indpb:
            individual[i] = rand_oneHotVector(knapsack_num)
    return individual,


'''
Mutation 1, improves on the solution though local exchanges. 
First, it considers all pairs of items assigned to different knapsacks and, 
if possible and if the total profit increases, interchanges them.
'''


def mutLocalSearch(ind, item_weight, capacities, toolbox):
    rcapcity = capacities - numpy.dot(ind.T, item_weight)
    curr_obj = list(map(toolbox.evaluate, [ind]))[0][0]
    item_num = ind.shape[0]
    knapsack_num = ind.shape[1]
    change_pair = [0] * get_combination_num(knapsack_num, 2)
    kk = 0
    for i in range(knapsack_num):
        for j in range(i, knapsack_num):
            if j == i:
                continue
            change_pair[kk] = (i, j)
            kk += 1
    # If all are zero, no need to exchange
    for i in range(item_num):
        if sum(ind[i]) == 0:
            continue
        for m, n in change_pair:
            if ind[i, m] == 0 and ind[i, n] == 0:
                continue
            # If it makes the instance infeasible no need to exchange
            change_m = (- ind[i, m] + ind[i, n]) * item_weight[i]
            change_n = (- ind[i, n] + ind[i, m]) * item_weight[i]
            if rcapcity[m] + change_m < 0:
                continue
            if rcapcity[n] + change_n < 0:
                continue
            # if it not increases the joint_value, (maybe) no need to exchange
            # ....
            # Now try exchange, see if it increases the ObjValue
            ind[i, m], ind[i, n] = ind[i, n], ind[i, m]
            new_obj = list(map(toolbox.evaluate, [ind]))[0][0]
            if curr_obj > new_obj:  # did not increase the ObjValue, change back
                ind[i, m], ind[i, n] = ind[i, n], ind[i, m]
                continue
            curr_obj = new_obj
            rcapcity[m] -= change_m
            rcapcity[n] -= change_n
    return ind


def mutRandomRemove(ind, num_of_remove):
    item_num = ind.shape[0]
    knapsack_num = ind.shape[1]
    contained_items = numpy.where(numpy.sum(ind, axis=1) == 1)[0]
    random.shuffle(contained_items)
    for i in contained_items[0: min(len(contained_items), num_of_remove)]:
        ind[i] = numpy.zeros(knapsack_num)
    return ind


# def change_of_objValue(gen_old, gen_new, instance_seetings, objc)


def exam_feasibility(individual, item_weight, capacities):
    if (numpy.sum(individual, axis=1) > 1).any():
        raise RuntimeError('Serious Issue: Item in Multi Knapsack')
    result = []
    # item_weight = instance_settings[1]
    # capacities = instance_settings[3]
    for k in range(individual.shape[1]):

        kth_knapsack = individual[:, k]
        weight = numpy.dot(kth_knapsack, item_weight)
        if weight > capacities[k]:
            result.append(False)
        else:
            result.append(True)
    return result


import math


def get_combination_num(n, m):
    return math.factorial(n) // (math.factorial(m) * math.factorial(n - m))
