import unittest
import numpy as np
import probability_distributions
from scipy.stats import entropy
from probability_distributions import ProbabilityArray


class TestProbabilityArray(unittest.TestCase):
    def setUp(self):
        self.probability_array = np.array(
            [
                [0.2, 0.4],
                [0.15, 0.25]
            ]
        )
        self.probability_array2 = np.array(
            [
                [
                    [0.1, 0.15],
                    [0.05, 0.05]
                ],
                [
                    [0.15, 0.25],
                    [0.05, 0.2]
                ]
            ]
        )

    def tearDown(self):
        pass

    def test_marginalize(self):
        probability_arr = ProbabilityArray(self.probability_array)
        out = probability_arr.marginalize(set([0]))
        expected = np.array([0.6, 0.4])
        self.assertTrue(np.allclose(out, expected))

    def test_find_conditional1(self):
        probability_arr = ProbabilityArray(self.probability_array)
        out, labels1, labels2 = probability_arr.find_conditional(set([0]), set([1]))
        expected = np.array(
            [
                [0.2/0.35, 0.4/0.65],
                [0.15/0.35, 0.25/0.65]
            ]
        )
        self.assertTrue(np.allclose(out, expected))
        self.assertEqual(labels1, set([0]))
        self.assertEqual(labels2, set([1]))

    def test_find_joint_marginal1(self):
        probability_arr = ProbabilityArray(self.probability_array2)
        out_distribution, out_labels1, out_labels2 = (
            probability_arr.find_joint_marginal(set([0]), set([2]))
        )
        expected_distribution = np.array(
            [
                [0.15, 0.2],
                [0.2, 0.45]
            ]
        )
        expected_labels1 = set([0])
        expected_labels2 = set([1])
        self.assertTrue(np.allclose(out_distribution, expected_distribution))
        self.assertEqual(out_labels1, expected_labels1)
        self.assertEqual(out_labels2, expected_labels2)
    
    def test_find_joint_marginal2(self):
        probability_arr = ProbabilityArray(self.probability_array)
        out_distribution, out_labels1, out_labels2 = (
            probability_arr.find_joint_marginal(set([0]), set([1]))
        )
        expected_distribution = np.array(
            [
                [0.2, 0.4],
                [0.15, 0.25]
            ]
        )
        expected_labels1 = set([0])
        expected_labels2 = set([1])
        self.assertTrue(np.allclose(out_distribution, expected_distribution))
 
    def test_compute_joint1(self):
        marginal = np.array([0.6, 0.4])
        conditional = np.array(
            [
                [0.7, 0.3],
                [0.5, 0.5]
            ]
        )
        conditional_labels = set([0])
        out = probability_distributions.compute_joint(marginal, conditional, 
                                                      conditional_labels)
        expected = np.array([[0.42, 0.18], [0.2, 0.2]])
        self.assertTrue(np.allclose(out, expected))

    def test1_compute_joint_from_independent_marginals(self):
        a = np.array(
            [
                [0.2, 0.3],
                [0.1, 0.4]
            ]
        )

        b = np.array([0.6, 0.4])
        out = probability_distributions.compute_joint_from_independent_marginals(
            a, b, [1]        
        )    
        expected = np.array(
            [
                [
                    [0.12, 0.18],
                    [0.08, 0.12]
                ],
                [
                    [0.06, 0.24],
                    [0.04, 0.16]
                ]
            ]
        )
        self.assertTrue(np.allclose(out, expected))

    def test2_compute_joint_from_independent_marginals(self):
        # representing independent marginals (0.6, 0.4), (0.7, 0.3), 
        # (0.2, 0.8), (0.5, 0.3, 0.2) 
        independent_joint = np.array(
            [
                [
                    [
                        [0.084*0.5, 0.084*0.3, 0.084*0.2],
                        [0.336*0.5, 0.336*0.3, 0.336*0.2]
                    ],
                    [
                        [0.036*0.5, 0.036*0.3, 0.036*0.2],
                        [0.144*0.5, 0.144*0.3, 0.144*0.2],
                    ]
                ],
                [
                    [
                        [0.056*0.5, 0.056*0.3, 0.056*0.2],
                        [0.224*0.5, 0.224*0.3, 0.224*0.2],
                    ],
                    [
                        [0.024*0.5, 0.024*0.3, 0.024*0.2],
                        [0.096*0.5, 0.096*0.3, 0.096*0.2]
                    ]
                ]
            ]
        )
        self.assertAlmostEqual(np.sum(independent_joint), 1)
        marginal1 = ProbabilityArray(independent_joint).marginalize(set([0,3]))
        marginal2 = ProbabilityArray(independent_joint).marginalize(set([1,2]))
        computed_joint = probability_distributions.compute_joint_from_independent_marginals(marginal1, marginal2, [1,2])
        self.assertTrue(np.allclose(independent_joint, computed_joint))

    def test_overall_methods1(self):
        probability_arr = ProbabilityArray(self.probability_array2)
        marginal = probability_arr.marginalize(set([1]))
        conditional, marginal_labels, conditional_labels = (
            probability_arr.find_conditional(set([0, 2]), set([1]))
        )
        joint = probability_distributions.compute_joint(marginal, conditional,
                                                        conditional_labels)
        self.assertTrue(np.allclose(joint, self.probability_array2))

    
    def test1_find_conditional_accounting_for_zero_marginals(self):
        arr = np.array([
            [
                [0.3, 0.1, 0.01],
                [0.0, 0.0, 0.0]
            ],
            [
                [0.15, 0.05, 0.07],
                [0.0, 0.27, 0.05]
            ]
        ])
        probability_array = probability_distributions.ProbabilityArray(arr)
        def produce_conditional_states():
            yield np.array([0.5, 0.3, 0.2])
            yield np.array([1.0, 0.0, 0.0])

        generator = produce_conditional_states()

        marginal_variables = set([2])
        conditional_variables = set([0, 1])
        expected_conditional = np.array([
            [
                [0.3/0.41, 0.1/0.41, 0.01/0.41],
                [0.5, 0.3, 0.2]
            ],
            [
                [0.15/0.27, 0.05/0.27, 0.07/0.27],
                [0.0/0.32, 0.27/0.32, 0.05/0.32]
            ]
        ])       
        conditional, marginal_labels, conditional_labels = (
            probability_array.find_conditional_accounting_for_zero_marginals(
                marginal_variables, conditional_variables, 
                generator
            )
        )
        self.assertTrue(np.allclose(expected_conditional, conditional))

    def test_find_conditional_accounting_for_zero_marginals_normal_joint(self):
        probability_arr = ProbabilityArray(self.probability_array)
        def produce_conditional_states():
            yield np.array([0.5, 0.3])
            yield np.array([1.0, 0.0])

        generator = produce_conditional_states()
        out, labels1, labels2 = (
            probability_arr.find_conditional_accounting_for_zero_marginals(
                set([0]), set([1]), generator
            )
        )
        expected = np.array(
            [
                [0.2/0.35, 0.4/0.65],
                [0.15/0.35, 0.25/0.65]
            ]
        )
        self.assertTrue(np.allclose(out, expected))
        self.assertEqual(labels1, set([0]))
        self.assertEqual(labels2, set([1]))

    def test1_produce_distribution_with_entropy_evolutionary(self):
        shape = tuple([3,3,3,3,4,4])
        entropy_size = np.log2(3.0**4 * 4**2)/2
        #print("entropy size {}".format(entropy_size))
        number_of_trials = 1000
        #distribution = probability_distributions.produce_distribution_with_entropy_evolutionary(
        #    shape, entropy_size, number_of_trials, 
        #    population_size=50, number_of_children=100, generational=False,
        #    initial_dist='peaked'
        #)
        #print(distribution)
        #print(np.sum(distribution))
        #print("the distribution's entropy {}".format(entropy(distribution.flatten(), base=2)))

    def test1_generate_probability_distribution_with_certain_entropy(self):
        shape = tuple([5]*5)
        entropy_size = np.log2(5**5 * 0.1)
        #print("the entropy size is {}".format(entropy_size))
        out = probability_distributions.generate_probability_distribution_with_certain_entropy(
            shape, entropy_size, True
            )
        self.assertTrue(abs(entropy_size-entropy(out.flatten(), base=2)) < 0.01)

    def test_decrease_entropy(self):
        a = np.array([0.1, 0.2, 0.1, 0.05, 0.01, 0.04, 0.4, 0.1])
        initial_entropy = entropy(a, base=2)
        out = probability_distributions.decrease_entropy(a, 1, 5, 0.02)
        #print("decrease in entropy {}".format(out))
        self.assertAlmostEqual(initial_entropy-entropy(a, base=2), out)

    def test1_remove_zero_marginals(self):
        dist = np.array(
            [
                [
                    [0, 0, 0],
                    [0.3, 0.1, 0.2]
                ],
                [
                    [0, 0, 0],
                    [0.05, 0.15, 0.2]
                ]
            ]
        )
        out = probability_distributions.remove_zero_marginals(dist)
        self.assertTrue(np.all(out==np.array([[[0.3, 0.1, 0.2]], [[0.05, 0.15, 0.2]]])))

    def test2_remove_zero_marginals(self):
        dist = np.array(
            [
                [
                    [0, 0.1, 0.2],
                    [0, 0.1, 0.2]
                ],
                [
                    [0, 0.05, 0],
                    [0, 0.15, 0.2]
                ]
            ]
        )
        out = probability_distributions.remove_zero_marginals(dist)
        expected = np.array(
            [
                [
                    [0.1, 0.2],
                    [0.1, 0.2]
                ],
                [
                    [0.05, 0],
                    [0.15, 0.2]
                ]
            ]
        )
        self.assertTrue(np.all(out==expected))

    def test3_remove_zero_marginals(self):
        dist = np.array(
            [
                [
                    [0.1, 0.1, 0.1],
                    [0, 0.1, 0.2]
                ],
                [
                    [0, 0.05, 0],
                    [0, 0.15, 0.2]
                ]
            ]
        )
        out = probability_distributions.remove_zero_marginals(dist)
        expected = np.array(
            [
                [
                    [0.1, 0.1, 0.1],
                    [0, 0.1, 0.2]
                ],
                [
                    [0, 0.05, 0],
                    [0, 0.15, 0.2]
                ]
            ]
        )
        self.assertTrue(np.all(out==expected))

    def test4_remove_zero_marginals(self):
        dist = np.array(
            [
                [
                    [0, 0.1, 0.1],
                    [0, 0.1, 0.2]
                ],
                [
                    [0, 0.05, 0.1],
                    [0, 0.15, 0.2]
                ]
            ]
        )
        dist2 = np.copy(dist)
        out = probability_distributions.remove_zero_marginals(dist)
        self.assertEqual(dist.shape, dist2.shape )

