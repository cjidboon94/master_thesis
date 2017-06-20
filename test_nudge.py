import unittest
import numpy as np

import nudge
from probability_distributions import ProbabilityArray

class TestNudge(unittest.TestCase):
    def setUp(self):
        self.distribution = np.array(
            [
                [
                    [0.1, 0.03, 0.05, 0.02],
                    [0.03, 0.001, 0.04, 0.009]
                ],
                [
                    [0.005, 0.005, 0, 0.1],
                    [0.2, 0.03, 0.03, 0.03]
                ],
                [
                    [0.007, 0.01, 0.1, 0.05],
                    [0.003, 0.02, 0.11, 0.02]
                ]
            ]
        )
        self.probability_array = ProbabilityArray(self.distribution) 

    def test1_nudge(self):
	distribution = np.array([0.001, 0.001, 0.003, 0.995])
	sum_nudged_distribution = np.zeros(distribution.shape)
	for i in range(10):
            nudged_distribution, nudged_states = nudge.nudge(distribution, 
                                                             0.01)
	    self.assertTrue(np.all(nudged_distribution >= 0))
	    self.assertTrue(np.all(nudged_distribution <= 1))
     
    def test2_nudge(self):
        distribution = np.array([0.3, 0.4, 0.3, 0.1])
        sum_nudged_distribution = np.zeros(distribution.shape)
        for i in range(1000):
            nudged_distribution, nudged_states = nudge.nudge(distribution, 
                                                             0.01)
            sum_nudged_distribution = (sum_nudged_distribution + 
                                       nudged_distribution)

        self.assertTrue(np.all((sum_nudged_distribution/1000)-distribution < 0.001))

    def test1_select_random_states(self):
        distribution = np.arange(40).reshape((4, 5, 2))
        states = nudge.select_random_states(distribution.shape, 1000)
        self.assertEqual(1000, len(states))
        #print(abs(np.mean([distribution[tuple(state)] for state in states])-19.5))
        self.assertTrue(abs(np.mean([distribution[tuple(state)] for state in states])-19.5) < 1)

    def test1_perform_nudge(self):
        arr = np.array(
            [
                [0.2, 0.1],
                [0.3, 0.4]
            ]
        )
        out = nudge.perform_nudge(arr, tuple([0, 0]), tuple([1, 0]), 0.01)
        self.assertEqual(out, 0.01)
        self.assertTrue(np.allclose(arr, np.array([[0.19, 0.1], [0.31, 0.4]])))

    def test2_perform_nudge(self):
        arr = np.array(
            [
                [0.01, 0.29],
                [0.3, 0.4]
            ]
        )
        out = nudge.perform_nudge(arr, tuple([0, 0]), tuple([1, 0]), 0.02)
        self.assertEqual(out, 0.01)
        self.assertTrue(np.allclose(arr, np.array([[0, 0.29], [0.31, 0.4]])))

    def test1_mutate_distribution(self):
        out = nudge.mutate_distribution(self.distribution, 0, 6, 0.002)
        self.assertEqual(out.shape, (3,2,4))
        marginal_input = ProbabilityArray(out).marginalize(set([1, 2]))
        marginal_output = ProbabilityArray(out).marginalize(set([0]))
        expected_marginal_input = self.probability_array.marginalize(set([1, 2]))
        expected_marginal_output = self.probability_array.marginalize(set([0]))
        self.assertTrue(np.allclose(marginal_input, expected_marginal_input))

        print("back in test class")
        print(marginal_output)
        print(expected_marginal_output)
        self.assertTrue(np.allclose(marginal_output, expected_marginal_output))



