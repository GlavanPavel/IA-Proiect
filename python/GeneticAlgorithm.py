import random
import numpy as np

class GeneticAlgorithm:
    def __init__(self, nn_structure, pop_size=50, mutation_rate=0.05):
        self.nn = nn_structure
        self.pop_size = pop_size
        self.mutation_rate = mutation_rate


        self.population = []
        for _ in range(pop_size):
            individual = np.random.uniform(-1, 1, self.nn.num_weights)
            self.population.append(individual)

    def evaluate_fitness(self, inputs, targets):
        fitness_scores = []

        for individual_weights in self.population:
            predicted_output = self.nn.forward(inputs, individual_weights)

            mse = np.mean((targets - predicted_output) ** 2)

            score = 1 / (mse + 1e-5)
            fitness_scores.append(score)

        return fitness_scores

    def select_parents(self, fitness_scores):

        # alegem doi parinti random si il luam pe cel mai bun
        idx1 = random.randint(0, self.pop_size - 1)
        idx2 = random.randint(0, self.pop_size - 1)

        if fitness_scores[idx1] > fitness_scores[idx2]:
            return self.population[idx1]
        else:
            return self.population[idx2]

    def crossover(self, parent1, parent2):
        # punct de taietura random
        crossover_point = random.randint(1, len(parent1) - 1)

        child = np.concatenate((parent1[:crossover_point], parent2[crossover_point:]))
        return child

    def mutate(self, child):
        mutation_mask = np.random.rand(len(child)) < self.mutation_rate

        noise = np.random.normal(0, 0.5, size=len(child))

        child[mutation_mask] += noise[mutation_mask]

        return child

    def evolve(self, inputs, targets):
        scores = self.evaluate_fitness(inputs, targets)

        best_score = max(scores)
        best_mse = 1 / best_score

        # selectia
        new_population = []

        # elitism
        best_idx = np.argmax(scores)
        new_population.append(self.population[best_idx])

        # restul sunt copii
        while len(new_population) < self.pop_size:
            p1 = self.select_parents(scores)
            p2 = self.select_parents(scores)

            child = self.crossover(p1, p2)

            child = self.mutate(child)

            new_population.append(child)

        self.population = new_population
        return best_mse