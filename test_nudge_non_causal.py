import unittest
import numpy as np
import nudge_non_causal as nudge
from probability_distributions import ProbabilityArray

class TestNudge(unittest.TestCase):
    def setUp(self):
        pass

    def test_find_noise_vector(self):
        noise_sum = np.zeros(5)
        for i in range(100):
            noise_size = 0.01
            noise_vector = nudge.find_noise_vector(5, noise_size)
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
        nudged_states = nudge.nudge_states(noise, probabilities)
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
        nudge_impact = nudge.find_nudge_impact(old_input, new_input, cond_output)
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
        nudge_impact = nudge.find_nudge_impact(old_input, new_input, cond_output)
        self.assertAlmostEqual(nudge_impact, 0.0145)

    def test_nudge_individual_makes_copy(self):
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
        input_dist_copy = np.copy(input_dist)
        new_dist = nudge.nudge_individual(input_dist, 0.01)
        self.assertTrue(np.all(input_dist==input_dist_copy))

        print(new_dist)
        print(np.sum(np.absolute(new_dist-input_dist)))

    def test_nudge_individual_correct_nudge_size(self):
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
        nudge_size = 0.01
        for _ in range(100):
            new_dist = nudge.nudge_individual(input_dist, 0.01)
            self.assertAlmostEqual(
                nudge_size, np.sum(np.absolute(new_dist-input_dist))
            )

    def test_nudge_individual_create_new_probability_dist(self):
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
        nudge_size = 0.01
        for _ in range(100):
            new_dist = nudge.nudge_individual(input_dist, 0.01)
            self.assertAlmostEqual(
                1, np.sum(new_dist)
            )
            self.assertTrue(np.all(new_dist >= 0))

        nudge_size = 0.01
        for _ in range(100):
            input_dist = np.random.dirichlet(np.array([5,5,5,5,5]))
            new_dist = nudge.nudge_individual(input_dist, 0.01)
            self.assertAlmostEqual(
                1, np.sum(new_dist)
            )
            self.assertTrue(np.all(new_dist >= 0))

    def test_nudge_individual_correct_nudge_size(self):
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
        nudge_size = 0.01
        for _ in range(100):
            new_dist = nudge.nudge_individual(input_dist, 0.01)
            self.assertAlmostEqual(
                nudge_size, np.sum(np.absolute(new_dist-input_dist))
            )
        for _ in range(100):
            input_dist = np.random.dirichlet(np.array([6,5,3,4,8]))
            new_dist = nudge.nudge_individual(input_dist, 0.01)
            self.assertAlmostEqual(
                nudge_size, np.sum(np.absolute(new_dist-input_dist))
            )

    def test_nudge_individual_correct_nudge_with_noise_vector(self):
        input_dist = np.array([
            [
                [0.2, 0.05, 0.1],
                [0.05, 0.1, 0.025],
            ],
            [
                [0.1, 0.09, 0.01],
                [0.05, 0.025, 0.2]
            ]
        ])
        noise_vector = np.array([0.1, -0.025, -0.075])
        nudge_size = 0.2
        new_dist = nudge.nudge_individual(input_dist, 0.01, noise_vector)
        expected_new_input_dist = np.array([
            [
                [0.235, 0.04125, 0.07375],
                [0.0675, 0.095625, 0.011875],
            ],
            [
                [0.115, 0.085, 0],
                [0.0775, 0.018125, 0.179375]
            ]
        ])
        self.assertTrue(np.allclose(new_dist, expected_new_input_dist))

        shape = [5,6,6,4,3]
        total_number_of_states = reduce(lambda x, y: x*y, shape)
        for _ in range(100):
            input_dist = np.random.dirichlet(np.array(total_number_of_states*[1]))
            input_dist = np.reshape(input_dist, shape)
            new_dist = nudge.nudge_individual(input_dist, 0.2, noise_vector)
            self.assertAlmostEqual(np.sum(new_dist), 1)
            self.assertTrue(np.all(new_dist >= 0))

    def test_nudge_local_makes_copy(self):
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
        input_dist_copy = np.copy(input_dist)
        new_dist = nudge.nudge_local(input_dist, [1, 2], 0.01)
        self.assertTrue(np.all(input_dist==input_dist_copy))

        print(new_dist)
        print(np.sum(np.absolute(new_dist-input_dist)))

    def test_nudge_local_correct_nudge_size(self):
        #create dirichlet different:
        #np.reshape(np.random.dirichlet(np.array(N*[1])), shape_dist)
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
        print("in the first loop")
        nudge_size = 0.01
        for _ in range(100):
            new_dist = nudge.nudge_local(input_dist, [0, 2], 0.01)
            self.assertAlmostEqual(
                nudge_size, np.sum(np.absolute(new_dist-input_dist))
            )

        print("in the second loop")
        shape = [5,6,6,4,5]
        total_number_of_states = reduce(lambda x, y: x*y, shape)
        for _ in range(20):
            input_dist = np.random.dirichlet(np.array(total_number_of_states*[1]))
            input_dist = np.reshape(input_dist, shape)
            new_dist = nudge.nudge_local(input_dist, [2, 0, 3], 0.01)
            self.assertAlmostEqual(
                nudge_size, np.sum(np.absolute(new_dist-input_dist))
            )

    def test_nudge_local_create_new_probability_dist(self):
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
        nudge_size = 0.01
        for _ in range(100):
            new_dist = nudge.nudge_local(input_dist, [1,2], 0.01)
            self.assertAlmostEqual(
                1, np.sum(new_dist)
            )
            self.assertTrue(np.all(new_dist >= 0))

        nudge_size = 0.01
        shape = [5,6,6,4,5]
        total_number_of_states = reduce(lambda x, y: x*y, shape)
        for _ in range(50):
            input_dist = np.random.dirichlet(np.array(total_number_of_states*[1]))
            input_dist = np.reshape(input_dist, shape)
            new_dist = nudge.nudge_local(input_dist, [2, 0, 3], 0.01)
            self.assertAlmostEqual(
                1, np.sum(new_dist)
            )
            self.assertTrue(np.all(new_dist >= 0))

    def test_nudge_local_3vars_marginal_constant(self):
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
        new_dist = nudge.nudge_local(input_dist, [0,1], nudge_size)
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

    def test_nudge_local_with_noise_vectors(self):
        noise_vectors = [
            np.array([-0.05, -0.05, 0.1]),
            np.array([0.02, -0.01, 0.02, 0.01, -0.03, -0.01]),
            np.array([0.03, 0.02, -0.1, 0.05])
        ]
        shape = [6,4,3]
        total_number_of_states = reduce(lambda x, y: x*y, shape)
        for _ in range(100):
            input_dist = np.random.dirichlet(np.array(total_number_of_states*[1]))
            input_dist = np.reshape(input_dist, shape)
            new_dist = nudge.nudge_local(input_dist, [2, 0, 1], 0.2, noise_vectors)
            self.assertAlmostEqual(np.sum(new_dist), 1)
            self.assertTrue(np.all(new_dist >= 0))

