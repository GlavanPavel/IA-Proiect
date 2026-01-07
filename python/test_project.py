import unittest
import numpy as np
from NeuralNetwork import NeuralNetwork
from GeneticAlgorithm import GeneticAlgorithm
from DataRead import normalize_data


class TestProject(unittest.TestCase):

    # ruleaza inainte de fiecare test
    def setUp(self):
        self.input_size = 5
        self.hidden_size = 4
        self.output_size = 1
        self.nn = NeuralNetwork(self.input_size, self.hidden_size, self.output_size)
        self.ga = GeneticAlgorithm(self.nn, pop_size=10, mutation_rate=0.1)

    # test daca reteaua returneaza forma corecta
    def test_neural_network_output_shape(self):
        dummy_input = np.random.rand(1, self.input_size)
        dummy_weights = np.random.uniform(-1, 1, self.nn.num_weights)

        output = self.nn.forward(dummy_input, dummy_weights)

        self.assertEqual(output.shape, (1, 1))

    # test daca functia de normalizare aduce datele in intervalul [0, 1]
    def test_normalization_logic(self):
        raw_data = np.array([
            [10, 200, 5],
            [20, 100, 6],
            [30, 300, 7]
        ])
        # ultima coloana e target, primele 2 inputs
        inputs, targets = normalize_data(raw_data)

        self.assertTrue(np.min(inputs) >= 0.0)
        self.assertTrue(np.max(inputs) <= 1.0)

        # 3 randuri
        self.assertEqual(inputs.shape[0], 3)
        # 2 coloane input
        self.assertEqual(inputs.shape[1], 2)

    # test daca populatia se initializeaza cu numarul corect de indivizi
    def test_genetic_population_init(self):
        self.assertEqual(len(self.ga.population), 10)
        # verificare lungimea cromozumului == (in*hid) + b1 + (hid*out) + b2
        expected_weights = (5 * 4) + 4 + (4 * 1) + 1
        self.assertEqual(len(self.ga.population[0]), expected_weights)

    # test daca incrucisarea pastreaza lungimea corecta a vectorului
    def test_crossover_length(self):
        parent1 = np.ones(self.nn.num_weights)
        parent2 = np.zeros(self.nn.num_weights)

        child = self.ga.crossover(parent1, parent2)

        self.assertEqual(len(child), len(parent1))

    # test daca mutatia modifica gena
    def test_mutation_changes_values(self):
        # pentru test rata de mutatie = 100%
        ga_high_mut = GeneticAlgorithm(self.nn, pop_size=2, mutation_rate=1.0)
        original = np.copy(ga_high_mut.population[0])

        mutated = ga_high_mut.mutate(ga_high_mut.population[0])

        self.assertFalse(np.array_equal(original, mutated))


if __name__ == '__main__':
    unittest.main()