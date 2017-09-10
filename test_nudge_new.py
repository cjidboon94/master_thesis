import unittest
import numpy as np
import nudge_new
from probability_distributions import ProbabilityArray

class TestNudge(unittest.TestCase):
    def setUp(self):
        pass

    def test_find_noise_vector(self):
        noise_sum = np.zeros(5)
        for i in range(100):
            noise_size = 0.01
            noise_vector = nudge_new.find_noise_vector(5, noise_size)
            noise_sum = noise_sum + noise_vector
            self.assertAlmostEqual(np.sum(noise_vector), 0)
            self.assertAlmostEqual(np.sum(np.absolute(noise_vector)), noise_size)
            #print("the noise vector {} \n".format(find_noise_vector(5, 0.01)))

        print(noise_sum)
        self.assertTrue(np.amin(noise_sum) > -0.08)
        self.assertTrue(np.amax(noise_sum) < 0.08) 

    def test_nudge_states(self):
        probabilities = np.array([0.1, 0.05, 0.2, 0.3, 0.25, 0.1])
        noise = np.array([0.02, -0.06, 0.01, 0.03, -0.04, 0.04])
        nudged_states = nudge_new.nudge_states(noise, probabilities)
        print(nudged_states)
        expected_nudged_states = np.array([0.118, 0, 0.209, 0.327, .21, 0.136])
        self.assertTrue(np.allclose(nudged_states, expected_nudged_states))

    def test1_nudge_impact(self):
        old_input = np.array([0.1, 0.2, 0.05, 0.3, 0.35])
        new_input = np.array([0.12, 0.18, 0.06, 0.31, 0.33])
        cond_output = np.array([
            [0.3, 0.4, 0.3],
            [0.2, 0.5, 0.3],
            [0.15, 0.45, 0.4],
            [0.46, 0.18, 0.36],
            [0.32, 0.43, 0.25]
        ])
        nudge_impact = nudge_new.find_nudge_impact(old_input, new_input, cond_output)
        self.assertAlmostEqual(nudge_impact, 0.0086)

    def test2_nudge_impact(self):
        old_input = np.array([[0.1, 0.2, 0.05], [0.3, 0.25, 0.1]])
        new_input = np.array([[0.12, 0.16, 0.06], [0.33, 0.245, 0.085]])
        cond_output = np.array([
            [
                [0.3, 0.4, 0.3],
                [0.2, 0.5, 0.3],
                [0.15, 0.45, 0.4]
            ],
            [
                [0.46, 0.18, 0.36],
                [0.32, 0.43, 0.25],
                [0.6, 0.2, 0.2]
            ]
        ])
        nudge_impact = nudge_new.find_nudge_impact(old_input, new_input, cond_output)
        self.assertAlmostEqual(nudge_impact, 0.0145)

    def test_local_non_causal_2vars(self):
        input_dist = np.array([
            [
                [0.2, 0.05, 0.1],
                [0.05, 0.1, 0.025],
            ],
            [
                [0.1, 0.08, 0.02],
                [0.05, 0.025, 0.2]
            ]
        ])
        new_dist = nudge_new.local_non_causal(input_dist, 0.01)
        print(new_dist)
        print(np.sum(np.absolute(new_dist-input_dist)))

    def test_global_non_causal_3vars(self):
        input_dist = np.array([
            [
                [0.2, 0.05, 0.1],
                [0.05, 0.1, 0.025],
            ],
            [
                [0.1, 0.08, 0.02],
                [0.05, 0.025, 0.2]
            ]
        ])
        #print(input_dist.shape)
        old_marginal1 = ProbabilityArray(input_dist).marginalize(set([0]))
        old_marginal2 = ProbabilityArray(input_dist).marginalize(set([1]))
        old_marginal3 = ProbabilityArray(input_dist).marginalize(set([2]))
        #print("old marginal 2 {}".format(old_marginal2))
        nudge_size = 0.01
        new_dist = nudge_new.global_non_causal(input_dist, [0,1], nudge_size)
        #print("new dist {}".format(new_dist))
        self.assertAlmostEqual(np.sum(np.absolute(new_dist-input_dist)), 
                               nudge_size)
        #print("new_marginals")
        #print(ProbabilityArray(new_dist).marginalize(set([0])))
        new_marginal1 = ProbabilityArray(new_dist).marginalize(set([0]))
        new_marginal2 = ProbabilityArray(new_dist).marginalize(set([1]))
        new_marginal3 = ProbabilityArray(new_dist).marginalize(set([2]))
        #print("new marginal 1 {}".format(new_marginal1))
        self.assertTrue(np.allclose(old_marginal3, new_marginal3))

