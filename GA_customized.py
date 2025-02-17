import numpy as np
import random
import tensorflow as tf
import pickle
import psutil
import sys
import os
from keras import backend as K
from sklearn.metrics import mean_absolute_error, root_mean_squared_error, mean_squared_error

tf.experimental.numpy.experimental_enable_numpy_behavior()
ndim = 4
decoder_model = pickle.load(open("decoder_model_16500.sav", 'rb'))
LSTM_ = pickle.load(open("LSTM_model.sav", 'rb'))
M1_max = 37.4957
M2_max = 87.4904
Max_intervals = 53

w_M1 = 0.3 * 5000
# M2 to be fed in kg
w_M2 = 0.7 * 5000

# molar mass of monomer kg/mol
wm1 = 0.1
wm2 = 0.1

# moles of M1 to be fed
m_M1 = w_M1 / wm1
# moles of M2 to be fed
m_M2 = w_M2 / wm2


def pad(feed, intervals, max_interval):
    feed.extend([0] * (max_interval - intervals))
    return feed


def non_pad_size(feed):
    count = 0
    for x in feed:
        if x == 0:
            continue
        else:
            count = count + 1

    return count


def ks_distance_loss(y_true, y_pred):
    y_true_cdf = np.cumsum(y_true / np.sum(y_true))
    y_pred_cdf = np.cumsum(y_pred / np.sum(y_pred))
    ks_distance = np.max(np.abs(y_true_cdf - y_pred_cdf))
    return ks_distance


def init_population(init_population_size):
    features = []
    wm1 = 0.1
    wm2 = 0.1
    # M1 to be fed in kg
    w_M1 = 0.3 * 5000
    # M2 to be fed in kg
    w_M2 = 0.7 * 5000

    # moles of M1 to be fed
    m_M1 = w_M1 / wm1
    # moles of M2 to be fed
    m_M2 = w_M2 / wm2
    max_interval = 53
    timeseries_interval = 400
    for i in range(init_population_size):
        feature = []

        tfmax = 6 * 60 * 60  # maximum reaction time   # 6 HRS
        tfmin = 1 * 60 * 60  # minimum reaction time   # 1 HR
        tf = np.random.randint(tfmin, tfmax)

        Nintervals = int(np.floor(tf / timeseries_interval))
        temperature = np.random.randint(60, 80)
        t_feed = tf - 20 * 60
        Fm1av = m_M1 / t_feed  # gets an average feed rate based on the selected feed time
        Fm2av = m_M2 / t_feed

        m1_feed = list(abs(np.random.normal(1, 1.0, Nintervals)) * Fm1av)
        m2_feed = list(abs(np.random.normal(1, 1.0, Nintervals)) * Fm2av)

        """
        Fm1av = ((0.3 * 5000) / 0.1) / (intervals * 400)
        Fm2av = ((0.7 * 5000) / 0.1) / (intervals * 400)

        m1_feed = list(abs(np.random.normal(1, 0.3, intervals)) * Fm1av)
        m2_feed = list(abs(np.random.normal(1, 0.3, intervals)) * Fm2av)
        """

        m1_feed[-3:] = [
                           -1] * 3  # Last 20 min the reaction should be take place without any feeding. In the simulator, this is represented by setting the feed values to zero.
        m2_feed[-3:] = [
                           -1] * 3  # We use zero-value masking to ensure that all monomer feed rate sequences have the same length in the LSTM model.
        # To make the LSTM consider the 20-minute feed values, we replace them with -1,
        # as LSTM will otherwise ignore zero-masked values.

        # m1_feed = adjust_feed(m1_feed, M1_max)
        # m2_feed = adjust_feed(m2_feed, M2_max)

        m1_feed = adjust_feed(m1_feed, m_M1)
        m2_feed = adjust_feed(m2_feed, m_M2)

        print("M1 feed:", sum(m1_feed) + 3)
        print("M2 feed:", sum(m2_feed) + 3)

        # if intervals != max_interval:
        #    m1_feed = pad(m1_feed, intervals, max_interval)
        #    m2_feed = pad(m2_feed, intervals, max_interval)

        feature.append(temperature)
        feature.extend(m1_feed)
        feature.extend(m2_feed)

        features.append(feature)

    return features


def decoder(decoder_model, ndim, encoded):
    decoded_values = decoder_model(encoded)
    decoded_values = decoded_values.transpose()

    return decoded_values


def LSTM_model(LSTM_, ndim, features):
    features = tf.keras.preprocessing.sequence.pad_sequences(features, padding='post', dtype=float)
    # print(features)
    features = features.reshape(1, features.shape[0], features.shape[1])
    predicted = LSTM_.predict(features)
    return predicted


def fitness_function(target, individual):
    features = []
    temp = individual[0]
    m1_feed = individual[1:54]
    m2_feed = individual[54:]

    for i in range(len(m1_feed)):

        if m1_feed[i] == m2_feed[i] == 0:
            features.append([0, m1_feed[i], m2_feed[i]])
        else:
            features.append([temp, m1_feed[i], m2_feed[i]])

    features = np.array(features)
    encoded_value_pred = LSTM_model(LSTM_, 4, features)
    cluster_pred = decoder(decoder_model, 4, encoded_value_pred)

    # print(cluster_pred.shape)
    # fitness_rmse = root_mean_squared_error(target, cluster_pred)  # RMSE
    # fitness = mean_squared_error(cluster_pred, target)  # MSE
    # fitness_KS_1 = ks_distance_loss(target, cluster_pred)  # KS Distance
    # fitness_KS_2 = ks_distance_loss(target[29:], cluster_pred[29:])  # KS Distance
    # print("Fitness:",fitness)
    # fitness = (0.5 * fitness_MAE_1) + (0.5 * fitness_MAE_2)
    # return -fitness, cluster_pred

    fitness_MAE_1 = mean_absolute_error(target, cluster_pred)  # MAE
    return -fitness_MAE_1, cluster_pred


def selection(population, fitnesses, size):
    selected_indices = random.sample(range(len(population)), size)

    sorted_indices_by_fitness = sorted(selected_indices, key=lambda i: fitnesses[i], reverse=True)

    return [population[i] for i in sorted_indices_by_fitness[:2]]


def adjust_feed(feed, m_M):  # Monomer unit Normalization from the simulator
    timeseries_interval = 400
    feed[-3:] = [0] * 3
    feed = np.array(feed)
    print("Inside weight:", m_M)
    Feederror = np.sum(feed * timeseries_interval) - m_M
    while abs(Feederror / m_M) > 0.001:
        if Feederror < 0:
            feed = feed * 1.001
        if Feederror > 0:
            feed = feed * 0.999
        Feederror = np.sum(feed * timeseries_interval) - m_M

    feed = list(feed)
    feed[-3:] = [-1] * 3  # Masking purpose

    feed = pad(feed, len(feed), Max_intervals)
    return feed


""" # This is a normalization process to make sure total monomer unit is same for all the reactions 
def adjust_feed(feed, max_value):  # Changes required here.

    if len(feed) > 3:
        total = sum(feed[:-3])
        print("Total:", total)
        if total > max_value:
            diff = total - max_value
            remaining_indices = [i for i in range(len(feed) - 3) if feed[i] > 0]

            while diff > 0.1 and remaining_indices:
                # Calculate amount to subtract per element
                per_element_adjustment = diff / len(remaining_indices)
                i = 0
                while i < len(remaining_indices):
                    idx = remaining_indices[i]

                    if feed[idx] > per_element_adjustment:
                        feed[idx] -= per_element_adjustment
                        diff -= per_element_adjustment
                        i += 1  # Move to next element
                    else:
                        # Element can't be adjusted further, keep its original value
                        diff -= feed[idx] + 0.1
                        feed[idx] = 0.1
                        remaining_indices.pop(i)

        elif total < max_value:
            diff = max_value - total

            for i in range(len(feed) - 3):
                feed[i] += diff / (len(feed) - 3)
    else:
        print("Not a good option")

    feed = pad(feed, len(feed), Max_intervals)
    return feed
"""


def crossover(parent1, parent2, crossover_rate):
    # print(parent1)
    # print(parent2)

    temp_parent_1 = parent1[0]
    temp_parent_2 = parent2[0]
    m1_feed_parent1 = parent1[1:54]
    m1_feed_parent2 = parent2[1:54]
    m2_feed_parent1 = parent1[54:]
    m2_feed_parent2 = parent2[54:]

    m1_feed_parent1_count, m1_feed_parent2_count, m2_feed_parent1_count, m2_feed_parent2_count = non_pad_size(
        m1_feed_parent1), non_pad_size(m1_feed_parent2), non_pad_size(m2_feed_parent1), non_pad_size(
        m2_feed_parent2)
    if random.random() < crossover_rate:

        print("Parent Feed Count:", m1_feed_parent1_count, m1_feed_parent2_count, m2_feed_parent1_count,
              m2_feed_parent2_count)

        # m1_cross_over_point = random.randrange(1, min(m1_feed_parent1_count, m1_feed_parent2_count))
        # m2_cross_over_point = random.randrange(1, min(m2_feed_parent1_count, m2_feed_parent2_count))

        cross_over_point = random.randrange(1, min(m1_feed_parent1_count - 3, m1_feed_parent2_count - 3))

        # print("Crossover point:", m1_cross_over_point, m2_cross_over_point)

        # print("Crossover point:(conditioned)", m1_cross_over_point, m2_cross_over_point)
        m1_feed_child1 = m1_feed_parent1[:cross_over_point] + m1_feed_parent2[cross_over_point:]
        m2_feed_child1 = m2_feed_parent1[:cross_over_point] + m2_feed_parent2[cross_over_point:]

        m1_feed_child2 = m1_feed_parent2[:cross_over_point] + m1_feed_parent1[cross_over_point:]
        m2_feed_child2 = m2_feed_parent2[:cross_over_point] + m2_feed_parent1[cross_over_point:]

        # m1_feed_child1 = adjust_feed(m1_feed_child1[:non_pad_size(m1_feed_child1)], M1_max)
        m1_feed_child1 = adjust_feed(m1_feed_child1[:non_pad_size(m1_feed_child1)], m_M1)

        # m2_feed_child1 = adjust_feed(m2_feed_child1[:non_pad_size(m2_feed_child1)], M2_max)
        m2_feed_child1 = adjust_feed(m2_feed_child1[:non_pad_size(m2_feed_child1)], m_M2)

        # m1_feed_child2 = adjust_feed(m1_feed_child2[:non_pad_size(m1_feed_child2)], M1_max)
        m1_feed_child2 = adjust_feed(m1_feed_child2[:non_pad_size(m1_feed_child2)], m_M1)

        # m2_feed_child2 = adjust_feed(m2_feed_child2[:non_pad_size(m2_feed_child2)], M2_max)
        m2_feed_child2 = adjust_feed(m2_feed_child2[:non_pad_size(m2_feed_child2)], m_M2)

        print("Sum: ", sum(m1_feed_child1[:non_pad_size(m1_feed_child1)]) + 3,
              sum(m1_feed_child2[:non_pad_size(m1_feed_child2)]) + 3,
              sum(m2_feed_child2[:non_pad_size(m2_feed_child2)]) + 3,
              sum(m2_feed_child1[:non_pad_size(m2_feed_child1)]) + 3)

        child1 = []
        child2 = []

        child1.append(temp_parent_2)
        child1 = child1 + m1_feed_child1 + m2_feed_child1

        child2.append(temp_parent_1)
        child2 = child2 + m1_feed_child2 + m2_feed_child2

        m1_feed_parent1 = child1[1:54]
        m1_feed_parent2 = child2[1:54]
        m2_feed_parent1 = child1[54:]
        m2_feed_parent2 = child2[54:]

        m1_feed_parent1_count, m1_feed_parent2_count, m2_feed_parent1_count, m2_feed_parent2_count = non_pad_size(
            m1_feed_parent1), non_pad_size(m1_feed_parent2), non_pad_size(m2_feed_parent1), non_pad_size(
            m2_feed_parent2)

        print("Parent Feed Count child After:", m1_feed_parent1_count, m1_feed_parent2_count, m2_feed_parent1_count,
              m2_feed_parent2_count)

        print("Cross point:", cross_over_point)
        print("Child 1:", child1)
        print("Child 2:", child2)
        # print("*********************")

        return child1, child2
    else:
        return parent1, parent2


def mutate_shuffle(parent):
    child = []

    m1_feed_count = non_pad_size(parent[1:54])
    m2_feed_count = non_pad_size(parent[54:])

    print("Parent Feed Count shuf Before:", m1_feed_count, m2_feed_count)

    m1_feed_values = parent[1:m1_feed_count + 1]
    m2_feed_values = parent[54:54 + m2_feed_count]

    m1_feed_values = m1_feed_values[:-3]
    m2_feed_values = m2_feed_values[:-3]

    # Shuffling
    random.shuffle(m1_feed_values)
    random.shuffle(m2_feed_values)

    m1_feed_values = m1_feed_values + [-1] * 3
    m2_feed_values = m2_feed_values + [-1] * 3

    m1_feed_values = pad(m1_feed_values, len(m1_feed_values), Max_intervals)
    m2_feed_values = pad(m2_feed_values, len(m2_feed_values), Max_intervals)

    child.append(np.random.randint(60, 80))
    child = child + m1_feed_values + m2_feed_values

    m1_feed_parent1 = child[1:54]
    m2_feed_parent1 = child[54:]

    m1_feed_parent1_count, m2_feed_parent1_count = non_pad_size(m1_feed_parent1), non_pad_size(m2_feed_parent1)

    print("Parent Feed Count shuf (after):", m1_feed_parent1_count, m2_feed_parent1_count)

    mutation_switch = 1
    return child, mutation_switch


def mutate_ext(parent):
    wm1 = 0.1
    wm2 = 0.1
    # M1 to be fed in kg
    w_M1 = 0.3 * 5000
    # M2 to be fed in kg
    w_M2 = 0.7 * 5000

    # moles of M1 to be fed
    m_M1 = w_M1 / wm1
    # moles of M2 to be fed
    m_M2 = w_M2 / wm2

    child = []
    mutated = 0
    temp = parent[0]
    m1_feed_values = parent[1:54]
    m2_feed_values = parent[54:]

    m1_feed_count = non_pad_size(m1_feed_values)
    m2_feed_count = non_pad_size(m2_feed_values)

    print("Parent Feed Count ext  (Before):", m1_feed_count, m2_feed_count)

    if m1_feed_count > 3 and m2_feed_count > 3:
        mutation_position_m1 = random.sample(range(0, m1_feed_count - 3), 1)
        mutation_position_m2 = random.sample(range(0, m2_feed_count - 3), 1)
    else:
        return parent, mutated

    if m1_feed_count + 1 > Max_intervals or m2_feed_count + 1 > Max_intervals:
        return parent, mutated

    if m1_feed_count + len(mutation_position_m1) > Max_intervals or m2_feed_count + len(
            mutation_position_m2) > Max_intervals:
        return parent, mutated

    tf = m1_feed_count * 400
    t_feed = tf - 20 * 60
    Fm1av = m_M1 / t_feed  # gets an average feed rate based on the selected feed time
    Fm2av = m_M2 / t_feed

    # Fm1av = ((0.3 * 5000) / 0.1) / (m1_feed_count * 400)
    # Fm2av = ((0.7 * 5000) / 0.1) / (m2_feed_count * 400)

    m1_feed = list(abs(np.random.normal(1, 0.3, len(mutation_position_m1))) * Fm1av)
    m2_feed = list(abs(np.random.normal(1, 0.3, len(mutation_position_m2))) * Fm2av)

    for k, j in zip(mutation_position_m1, range(len(mutation_position_m1))):
        m1_feed_values.insert(k, m1_feed[j])  # extending
        m1_feed_values.pop()

    for k, j in zip(mutation_position_m2, range(len(mutation_position_m2))):
        m2_feed_values.insert(k, m2_feed[j])  # extending
        m2_feed_values.pop()

    # m1_feed_values = adjust_feed(m1_feed_values[:non_pad_size(m1_feed_values)], M1_max)
    m1_feed_values = adjust_feed(m1_feed_values[:non_pad_size(m1_feed_values)], m_M1)

    # m2_feed_values = adjust_feed(m2_feed_values[:non_pad_size(m2_feed_values)], M2_max)
    m2_feed_values = adjust_feed(m2_feed_values[:non_pad_size(m2_feed_values)], m_M2)

    print("Sum Mutation ext:", sum(m1_feed_values) + 3, sum(m2_feed_values) + 3)

    # child.append(np.random.randint(60, 80))
    child.append(temp)
    child.extend(m1_feed_values)
    child.extend(m2_feed_values)

    m1_feed_parent1 = child[1:54]
    m2_feed_parent1 = child[54:]
    m1_feed_parent1_count, m2_feed_parent1_count = non_pad_size(m1_feed_parent1), non_pad_size(m2_feed_parent1)

    print("Parent Feed Count ext (after):", m1_feed_parent1_count, m2_feed_parent1_count)

    mutated = 1
    return child, mutated


def mutate_rm(parent):
    child = []
    mutated = 0
    temp = parent[0]
    m1_feed_values = parent[1:54]
    m2_feed_values = parent[54:]

    m1_feed_count = non_pad_size(m1_feed_values)
    m2_feed_count = non_pad_size(m2_feed_values)

    print("Parent Feed Count ext  (Before):", m1_feed_count, m2_feed_count)

    if m1_feed_count > 3 and m2_feed_count > 3:
        mutation_position_m1 = random.sample(range(0, m1_feed_count - 3), 1)
        mutation_position_m2 = random.sample(range(0, m2_feed_count - 3), 1)
    else:
        return parent, mutated

    if m1_feed_count - 1 < 9 or m2_feed_count - 1 < 9:
        return parent, mutated

    if m1_feed_count - len(mutation_position_m1) < 9 or m2_feed_count - len(mutation_position_m2) < 9:
        return parent, mutated

    for k, j in zip(mutation_position_m1, range(len(mutation_position_m1))):
        m1_feed_values.pop(k)
        m1_feed_values.insert(len(m1_feed_values), 0)

    for k, j in zip(mutation_position_m2, range(len(mutation_position_m2))):
        m2_feed_values.pop(k)
        m2_feed_values.insert(len(m2_feed_values), 0)

    # m1_feed_values = adjust_feed(m1_feed_values[:non_pad_size(m1_feed_values)], M1_max)
    m1_feed_values = adjust_feed(m1_feed_values[:non_pad_size(m1_feed_values)], m_M1)

    # m2_feed_values = adjust_feed(m2_feed_values[:non_pad_size(m2_feed_values)], M2_max)
    m2_feed_values = adjust_feed(m2_feed_values[:non_pad_size(m2_feed_values)], m_M2)

    print("Sum Mutation rm:", sum(m1_feed_values) + 3, sum(m2_feed_values) + 3)

    # child.append(np.random.randint(60, 80))
    child.append(temp)
    child.extend(m1_feed_values)
    child.extend(m2_feed_values)

    m1_feed_parent1 = child[1:54]
    m2_feed_parent1 = child[54:]

    m1_feed_parent1_count, m2_feed_parent1_count = non_pad_size(m1_feed_parent1), non_pad_size(m2_feed_parent1)

    print("Parent Feed Count rm (after):", m1_feed_parent1_count, m2_feed_parent1_count)

    mutated = 1
    return child, mutated


def mutate(parent, mutation_rate):
    # Shuffling the genes
    child = []
    mutated = 0

    m1_feed_parent1 = parent[1:54]
    m2_feed_parent1 = parent[54:]

    m1_feed_parent1_count, m2_feed_parent1_count = non_pad_size(m1_feed_parent1), non_pad_size(m2_feed_parent1)

    print("Parent Feed Count start:", m1_feed_parent1_count, m2_feed_parent1_count)

    if random.random() < mutation_rate:  # Shuffling mutation

        child, mutated = mutate_shuffle(parent)

    if random.random() < mutation_rate:  # Inserting value mutation

        if mutated == 1:
            child, mutated = mutate_ext(child)
        else:
            child, mutated = mutate_ext(parent)

    if random.random() < mutation_rate:  # Removing value mutation
        if mutated == 1:
            child, mutated = mutate_rm(child)
        else:
            child, mutated = mutate_rm(parent)

    if mutated == 1:
        print("Mutated:", child)
        return child
    else:
        return parent


def genetic_algorithm(target, population_size, generations, sol_per_pop, crossover_rate=0.005, mutation_rate=0.005,
                      elitism_size=1):
    no_improvement_count = 0
    best_fitness_prev = -np.inf
    patience = 100

    population = init_population(population_size)

    # Initialize global best solution and fitness
    global_best_individual = None
    global_best_fitness = -np.inf
    best_fitnesses = []
    best_solns = []
    best_solns_AE_LSTM = []

    for generation in range(generations):

        new_population = []

        fitnesses, AE_LSTM_pred = np.array(
            [fitness_function(target, individual)[0] for individual in population]), np.array(
            [fitness_function(target, individual)[1] for individual in population])

        best_fitness = np.max(fitnesses)
        best_individual = population[np.argmax(fitnesses)]
        AE_LSTM_soln = AE_LSTM_pred[np.argmax(fitnesses)]

        print(f"Generation {generation + 1} - Best Fitness Value: {best_fitness:.6f}")
        print(f"Generation {generation + 1} - Best Fitness Solution: {best_individual}")

        best_fitnesses.append(best_fitness)
        best_solns.append(best_individual)
        best_solns_AE_LSTM.append(AE_LSTM_soln)

        if best_fitness > global_best_fitness:
            global_best_fitness = best_fitness
            global_best_individual = best_individual

        if best_fitness <= best_fitness_prev:
            no_improvement_count += 1
        else:
            no_improvement_count = 0
            best_fitness_prev = best_fitness

        if no_improvement_count > patience:
            break

        # Create a new population
        elite_indices = np.argsort(fitnesses)[-elitism_size:]
        elite_individuals = [population[i] for i in elite_indices]

        population = [ind for i, ind in enumerate(population) if i not in elite_indices]

        new_population = elite_individuals.copy()

        while len(new_population) < sol_per_pop:
            # Selection
            parent1, parent2 = selection(population, fitnesses, 2)

            # Crossover
            child1, child2 = crossover(parent1, parent2, crossover_rate)

            child1_ = mutate(child1, mutation_rate)

            child2_ = mutate(child2, mutation_rate)

            new_population.append(child1_)
            new_population.append(child2_)

        population = new_population[:sol_per_pop]

    return target, best_solns, best_solns_AE_LSTM, global_best_individual, global_best_fitness, best_fitnesses


opt_data = np.load("AE_4/weighted/GA/target.npy")

best_solutions = []
best_rmses = []
target_data = opt_data[int(sys.argv[1])]

target, best_soln_per_gen, best_solns_AE_LSTM, best_solution, best_rmse, best_fitness_values = genetic_algorithm(
    target_data,
    population_size=1000,
    sol_per_pop=500,
    generations=500,
    crossover_rate=0.6,
    mutation_rate=0.0001,
    elitism_size=100)

print("Best Solution: ", best_solution)
print("Best Fitness: ", best_rmse)

best_solutions.append(best_solution)
best_rmses.append(best_rmse * -1)

np.save(
    "Results/elitism_10_percent_mutation_MAE/target/target_" +
    sys.argv[1] + ".npy",
    np.array(target))

np.save(
    "Results/elitism_10_percent_mutation_MAE/Best_solution_per_gen/Best_solutions_" +
    sys.argv[1] + ".npy",
    np.array(best_soln_per_gen))

np.save(
    "Results/elitism_10_percent_mutation_MAE/Best_solution_per_gen_AE_LSTM/Best_solutions_AE_LSTM_" +
    sys.argv[1] + ".npy",
    np.array(best_solns_AE_LSTM))

np.save("Results/elitism_10_percent_mutation_MAE/Best_solution/Best_solutions_" +
        sys.argv[1] + ".npy",
        np.array(best_solutions))

np.save("Results/elitism_10_percent_mutation_MAE/RMSE/RMSE_" + sys.argv[1] + ".npy",
        np.array(best_rmses))

np.save("Results/elitism_10_percent_mutation_MAE/Fitness/Best_fitness_" + sys.argv[
    1] + ".npy",
        np.array(best_fitness_values))
