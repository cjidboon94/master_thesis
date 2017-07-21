import unittest
import numpy as np

import nudge
import probability_distributions
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
        self.distribution2 = np.array(
            [
                [
                    [
                        [0.01, 0.03, 0.05],
                        [0.02, 0.01, 0.005],
                        [0.004, 0.006, 0.01],
                    ],
                    [
                        [0.06, 0.01, 0.003],
                        [0.002, 0.004, 0.001],
                        [0.1, 0.03, 0.02],
                    ]
                ],
                [
                    [
                        [0.02, 0.02, 0.07],
                        [0.02, 0.005, 0.01],
                        [0.003, 0.004, 0.003],
                    ],
                    [
                        [0.13, 0.01, 0.001],
                        [0.009, 0.01, 0.01],
                        [0.1, 0.15, 0.05],
                    ]
                ]
            ]
        )
        self.probability_array = ProbabilityArray(self.distribution) 
        self.probability_array2 = ProbabilityArray(self.distribution2)

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

    def test1_nudge_distribution_local_non_causal(self):
        dist = np.array(
            [
                [
                    [
                        [0.01, 0.03, 0.05],
                        [0.02, 0.01, 0.005],
                        [0.004, 0.006, 0.01],
                    ],
                    [
                        [0.06, 0.01, 0.003],
                        [0.002, 0.004, 0.001],
                        [0.1, 0.03, 0.02],
                    ]
                ],
                [
                    [
                        [0.02, 0.02, 0.07],
                        [0.02, 0.005, 0.01],
                        [0.003, 0.004, 0.003],
                    ],
                    [
                        [0.13, 0.01, 0.001],
                        [0.009, 0.01, 0.01],
                        [0.1, 0.15, 0.05],
                    ]
                ]
            ]
        )
        probability_array = ProbabilityArray(dist)
        number_of_states = reduce(lambda x,y: x*y, dist.shape)
        out = nudge.nudge_distribution_local_non_causal(
            dist, 0, 0.1/float(number_of_states), int(round(number_of_states/10))
        )
        marginal_other_out = ProbabilityArray(out).marginalize(set([1, 2, 3]))
        marginal_nudged_out = ProbabilityArray(out).marginalize(set([0]))
        expected_marginal_other = np.array(
            [
                [
                    [0.03, 0.05, 0.12],
                    [0.04, 0.015, 0.015],
                    [0.007, 0.010, 0.013],
                ],
                [
                    [0.19, 0.02, 0.004],
                    [0.011, 0.014, 0.011],
                    [0.2, 0.18, 0.07],
                ]
            ]
        )

        self.assertTrue(np.allclose(marginal_other_out, expected_marginal_other))
        #expected_abs_difference_nudged = 0.01
        #difference_nudged = marginal_nudged_out

    def test_select_subsets(self):
        arr = np.array([0.1, 0.05, 0.15, 0.2])
        threshold = 0.1
        for i in range(50):
            out = nudge.select_subset(arr, threshold)
            self.assertTrue(np.sum(arr[out!=0])>=threshold)
            self.assertFalse(np.all(out==1))

    def test_random_mutate_array_bigger_zero(self):
        arr = np.array([0.1, 0.05, 0.15, 0.2, 0.01])
        nudge_size = 0.1
        for i in range(10):
            out = nudge.mutate_array_bigger_zero(arr, nudge_size, 'random')
            self.assertTrue(np.all(arr==np.array([0.1, 0.05, 0.15, 0.2, 0.01])))
            self.assertAlmostEqual(np.sum(out), np.sum(arr))
            self.assertAlmostEqual(np.sum(np.absolute(out-arr)), 2*nudge_size)

    def test_proportional_mutate_array_bigger_zero(self):
        arr = np.array([0.1, 0.05, 0.15, 0.2, 0.01])
        nudge_size = 0.1
        for i in range(10):
            out = nudge.mutate_array_bigger_zero(arr, nudge_size, 'proportional')
            self.assertTrue(np.all(arr==np.array([0.1, 0.05, 0.15, 0.2, 0.01])))
            self.assertAlmostEqual(np.sum(out), np.sum(arr))
            self.assertAlmostEqual(np.sum(np.absolute(out-arr)), 2*nudge_size)

    def test_nudge_distribution_non_local_non_causal(self):
        dist = np.array(
            [
                [
                    [
                        [0.01, 0.03, 0.05],
                        [0.02, 0.01, 0.005],
                        [0.004, 0.006, 0.01],
                    ],
                    [
                        [0.06, 0.01, 0.003],
                        [0.002, 0.004, 0.001],
                        [0.1, 0.03, 0.02],
                    ]
                ],
                [
                    [
                        [0.02, 0.02, 0.07],
                        [0.02, 0.005, 0.01],
                        [0.003, 0.004, 0.003],
                    ],
                    [
                        [0.13, 0.01, 0.001],
                        [0.009, 0.01, 0.01],
                        [0.1, 0.15, 0.05],
                    ]
                ]
            ]
        )
        dist_copy = np.copy(dist)
        nudge_labels = [0, 3]
        nudge_size = 0.01
        out = nudge.nudge_distribution_non_local_non_causal(
            dist, nudge_labels, nudge_size, "random"
        )
        self.assertEqual(out.shape, dist.shape)
        out_arr = ProbabilityArray(out)
        dist_arr = ProbabilityArray(dist)
        self.assertTrue(np.allclose(out_arr.marginalize(set([1, 2])), dist_arr.marginalize(set([1, 2]))))
        self.assertTrue(np.all(dist==dist_copy))
        self.assertAlmostEqual(np.sum(np.absolute(out-dist)), 2*nudge_size)

    def test2_nudge_distribution_non_local_non_causal(self):
        dist = np.random.random((4,4))
        dist_copy = np.copy(dist)
        #print(dist.shape)
        nudge_labels = [0, 1]
        nudge_size = 0.01
        out = nudge.nudge_distribution_non_local_non_causal(
            dist, nudge_labels, nudge_size, "random"
        )

    def test1_mutate_distribution(self):
        print("hello world")
        out = nudge.mutate_distribution_with_fixed_marginals(self.distribution, 0, 6, 0.002)
        self.assertEqual(out.shape, (3,2,4))
        marginal_input = ProbabilityArray(out).marginalize(set([1, 2]))
        marginal_output = ProbabilityArray(out).marginalize(set([0]))
        expected_marginal_input = self.probability_array.marginalize(set([1, 2]))
        expected_marginal_output = self.probability_array.marginalize(set([0]))
        self.assertTrue(np.allclose(marginal_input, expected_marginal_input))

        #print("back in test class")
        #print(marginal_output)
        #print(expected_marginal_output)
        self.assertTrue(np.allclose(marginal_output, expected_marginal_output))

    def test_find_max_impact_nudge_to_output_state(self):
        cond_output = np.array(
            [
                [
                    [0.4, 0.2, 0.4],
                    [0.3, 0.5, 0.2]
                ],
                [
                    [0.3, 0.3, 0.4],
                    [0.2, 0.2, 0.6]
                ]
            ]
        )
        output_state = [1, -1, 1]
        input_arr = np.array([0.4, 0.1, 0.1, 0.4])
        nudge_size = 0.3
        out_nudge, out_impact = nudge.find_max_impact_nudge_to_output_state(
            cond_output, input_arr, nudge_size, output_state
        )
        expected_nudge = [[1, -0.10], [2, -0.10], [3, -0.10], [0, 0.3]]
        
        self.assertAlmostEqual(out_impact, 0.08)
        expected_nudge_sorted = sorted(expected_nudge, key=lambda x: x[0])
        out_nudge_sorted = sorted(out_nudge, key=lambda x: x[0])
        for count, nudge_info in enumerate(expected_nudge_sorted):
            self.assertEqual(nudge_info[0], out_nudge_sorted[count][0])
            self.assertAlmostEqual(nudge_info[1], out_nudge_sorted[count][1])

    def test2_find_max_impact_nudge_to_output_state(self):
        cond_output = np.array(
            [
                [
                    [0.8, 0.1, 0.1],
                    [0.18, 0.52, 0.3]
                ],
                [
                    [0.2, 0.6, 0.2],
                    [0.05, 0.1, 0.8]
                ]
            ]
        )
        output_state = [1, -1, -1]
        input_arr = np.array([0.2, 0.25, 0.35, 0.2])
        nudge_size = 0.3
        out_nudge, out_impact = nudge.find_max_impact_nudge_to_output_state(
            cond_output, input_arr, nudge_size, output_state
        )
        expected_nudge = [[1, -0.10], [3, -0.20], [0, 0.3]]
        
        self.assertAlmostEqual(out_impact, 0.3*0.6 - 0.2*(-0.85) - 0.1*(-0.64))
        expected_nudge_sorted = sorted(expected_nudge, key=lambda x: x[0])
        out_nudge_sorted = sorted(out_nudge, key=lambda x: x[0])
        for count, nudge_info in enumerate(expected_nudge_sorted):
            self.assertEqual(nudge_info[0], out_nudge_sorted[count][0])
            self.assertAlmostEqual(nudge_info[1], out_nudge_sorted[count][1])

    def test_find_max_impact_global_nudge(self):
        input_arr = np.array(
            [
                [0.1, 0.15, 0.05],
                [0.2, 0.1, 0.15],
                [0.1, 0.1, 0.05],
            ]
        )
        cond_output = np.array(
            [
                [
                    [0.3, 0.3, 0.4],
                    [0.5, 0.0, 0.5],
                    [0.4, 0.1, 0.5]
                ],
                [
                    [0.4, 0.2, 0.4],
                    [0.3, 0.35, 0.35],
                    [0.2, 0.4, 0.4]
                ],
                [
                    [0.1, 0.1, 0.8],
                    [0.5, 0.2, 0.3],
                    [0.1, 0.3, 0.6]
                ]
            ]
        )
        nudge_size = 0.2
        out_max_nudge, out_max_impact = nudge.find_max_impact_global_nudge(input_arr, cond_output, nudge_size)
        #print(out_max_nudge, out_max_impact)
