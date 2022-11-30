import math
import random
from deap import base
from deap import creator
from deap import tools
import numpy
import sys

sys.path.append("..")
import objfuncs as objf


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


def perform_GA_base(objc, instance_settings, evoluion_general_parameters, evolution_specify_parameters,
                    PRINT=False):
    # instance_setings should be a tuple contains: {item_value, item_weight, joint_profit, capacities}
    #                                                           capacities = {capacity1, capacity2, ...}
    # evolution_general_parameters should be a tuple contains: {popsize, swap_prob, mute_prob, punish_factor}, in which:
    #    swap_prob  is independent probability for swap at each point in uniform crossover.
    #    mute_prob  is independent probability for each attribute to be flipped in flip-bit mutation.
    #    punish_factor is value of punish_factor
    # evolution_specify_parameters should be a tuple contains: {CXPB, MUTPB},In which:
    #    CXPB  is the probability with which two individuals
    #          are crossed
    #    MUTPB is the probability for mutating an individual
    #    MAX_GEN is the maximum generation threshold
    #    STOP_GEN is the threshold of no progress generations

    # update the toolbox base on specified instances:
    item_value, item_weight, joint_profit, capacities = instance_settings
    NUM_ITEMS = item_value.shape[0]
    NUM_KNAPSACK = len(capacities)
    ''' create individuals '''
    creator.create("FitnessMax", base.Fitness, weights=(1.0,))
    creator.create("Individual", numpy.ndarray, fitness=creator.FitnessMax)

    ''' initialize population '''
    toolbox = base.Toolbox()
    # Attribute generator
    toolbox.register("attr_vector", rand_oneHotVector, NUM_KNAPSACK)
    # Structure initializers
    toolbox.register("individual", tools.initRepeat, creator.Individual, toolbox.attr_vector, NUM_ITEMS)
    toolbox.register("population", tools.initRepeat, list, toolbox.individual)

    ''' define select paradigm '''
    toolbox.register("select", tools.selTournament, tournsize=2)

    # update the toolbox base on specified evolution parameters:
    popsize, swap_prob, mute_prob, punish_factor = evoluion_general_parameters
    pop = toolbox.population(n=popsize)
    toolbox.register("mutate", mutUniformVec_free, indpb=mute_prob)  # Vanilla
    toolbox.register("mate", cxUniform_free, prob=swap_prob)  # Vanilla

    # def objf_base(decision_matrix, objc, instance_settings, punish_factor=-100):
    if type(objc) == int:
        toolbox.register("evaluate", objf.objf_base, objc=objc, instance_settings=instance_settings,
                         punish_factor=-punish_factor)
    elif type(objc) == tuple:
        objc = numpy.array(objc)
        toolbox.register("evaluate", objf.objf_weight, objc_weight_vector=objc, instance_settings=instance_settings,
                         punish_factor=-punish_factor)
    else:
        raise Exception('Incorrect input of objc')
    return GA_core(toolbox, pop, evolution_specify_parameters, PRINT)


# def perform_GA_better():
#     return


def GA_core(toolbox, pop, evolution_specify_parameters, PRINT):
    # update the toolbox base on specified evolution parameters:
    CXPB, MUTPB, MAX_GEN, STOP_GEN = evolution_specify_parameters
    # --------- begin the evolution ----------
    if PRINT:
        print("Start of evolution")
    # Evaluate the entire population
    fitnesses = list(map(toolbox.evaluate, pop))  # TODO tobe check if 'list' can be deleted?
    for ind, fit in zip(pop, fitnesses):
        ind.fitness.values = fit
    if PRINT:
        print("  Evaluated %i individuals" % len(pop))
    # Extracting all the fitness of
    fits = [ind.fitness.values[0] for ind in pop]

    # Variable keeping track of the number of generations
    g = 0
    g_hold = 0
    best_fit_sofar = -math.inf

    # Begin the evolution
    while g_hold < STOP_GEN and g < MAX_GEN:
        # stop at the stable best population or stop at the first best individual
        if max(fits) > best_fit_sofar:  # reach the best individual
            best_fit_sofar = max(fits)
        if max(fits) == best_fit_sofar:
            g_hold += 1
        else:
            g_hold = 0
        # A new generation
        g += 1
        if PRINT:
            print("-- Generation %i --" % g)

        # Select the next generation individuals
        offspring = toolbox.select(pop, len(pop))
        # Clone the selected individuals
        offspring = list(map(toolbox.clone, offspring))

        # Apply crossover and mutation on the offspring
        for child1, child2 in zip(offspring[::2], offspring[1::2]):

            # cross two individuals with probability CXPB
            if random.random() < CXPB:
                toolbox.mate(child1, child2)
                # fitness values of the children must be recalculated later
                del child1.fitness.values
                del child2.fitness.values

        for mutant in offspring:
            # mutate an individual with probability MUTPB
            if random.random() < MUTPB:
                toolbox.mutate(mutant)
                del mutant.fitness.values

        # Evaluate the individuals with an invalid fitness
        invalid_ind = [ind for ind in offspring if not ind.fitness.valid]
        fitnesses = map(toolbox.evaluate, invalid_ind)
        for ind, fit in zip(invalid_ind, fitnesses):
            ind.fitness.values = fit
        if PRINT:
            print("  Evaluated %i individuals" % len(invalid_ind))

        # The population is entirely replaced by the offspring
        pop[:] = offspring

        # Gather all the fitnesses in one list and print the stats
        fits = [ind.fitness.values[0] for ind in pop]

        if PRINT:
            length = len(pop)
            mean = sum(fits) / length
            sum2 = sum(x * x for x in fits)
            std = abs(sum2 / length - mean ** 2) ** 0.5
            print("  Min %s" % min(fits))
            print("  Max %s" % max(fits))
            print("  Avg %s" % mean)
            print("  Std %s" % std)

    best_ind = tools.selBest(pop, 1)[0]
    if PRINT:
        print("-- End of (successful) evolution --")
        print("Best individual is %s, %s" % (best_ind, best_ind.fitness.values))

    # return
    #   pop: the final round Population and the round
    #   g: the number of generation that reach the optimal
    return best_ind, pop, g
