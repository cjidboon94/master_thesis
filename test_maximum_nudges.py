
import unittest
import numpy as np
import collections
import maximum_nudges

class TestMaximumControlNudges(unittest.TestCase):
    def test1_find_minimum_subset(self):
        weights = np.array([0.1, 0.1, 0.2, 0.5, 0.1])
        scores = np.array([0.8, -0.2, 0.3, 0.6, -0.3])
        total_size = 0.3
        states, sizes = maximum_nudges.find_minimum_subset(weights, scores, total_size)
        outcomes = sorted(list(zip(states, sizes)), key=lambda x: x[0])
        expected_outcomes = sorted(list(zip([1, 2, 4], [0.1, 0.1, 0.1])), key=lambda x: x[0])
        for count in range(len(states)):
            state, size = outcomes[count]
            expected_state, expected_size = expected_outcomes[count]
            self.assertEqual(state, expected_state)
            self.assertAlmostEqual(size, expected_size)

    def test1_find_max_control_impact(self):
        allignment_scores = np.array([0.60, 0, 0.4, 0.5])
        input_arr = np.array([0.4, 0.1, 0.1, 0.4])
        nudge_size = 0.3

        out_nudge_states, out_nudge_sizes, out_impact = maximum_nudges.find_max_nudge_impact(
            allignment_scores, input_arr, nudge_size
        )
        self.assertAlmostEqual(out_impact, 0.09)

        out_nudge = sorted(list(zip(out_nudge_states, out_nudge_sizes)),
                           key=lambda x: x[0])

        expected_nudge_states = [1, 2, 3, 0]
        expected_nudge_sizes = [-0.1, -0.1, -0.1, 0.3]
        expected_nudge = sorted(list(zip(expected_nudge_states, expected_nudge_sizes)),
                                key=lambda x: x[0])

        for count in range(len(out_nudge_states)):
            state, size = out_nudge[count]
            expected_state, expected_size = expected_nudge[count]
            self.assertEqual(state, expected_state)
            self.assertAlmostEqual(size, expected_size)

    def test2_find_max_control_impact(self):
        input_arr = np.array([0.2, 0.25, 0.35, 0.2])
        allignment_scores = np.array([0.6, -0.64, -0.6, -0.85])
        nudge_size = 0.3
        
        out_nudge_states, out_nudge_sizes, out_impact = maximum_nudges.find_max_nudge_impact(
            allignment_scores, input_arr, nudge_size
        )
        self.assertAlmostEqual(out_impact, 0.3*0.6 - (-0.85*0.2 - 0.64*0.1))

        out_nudge = sorted(list(zip(out_nudge_states, out_nudge_sizes)),
                           key=lambda x: x[0])

        expected_nudge_states = [1, 3, 0]
        expected_nudge_sizes = [-0.1, -0.2, 0.3]
        expected_nudge = sorted(list(zip(expected_nudge_states, expected_nudge_sizes)),
                                key=lambda x: x[0])

        for count in range(len(out_nudge_states)):
            state, size = out_nudge[count]
            expected_state, expected_size = expected_nudge[count]
            self.assertEqual(state, expected_state)
            self.assertAlmostEqual(size, expected_size)

    def test1_find_max_control_impact(self):
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
        out_nudge_states, out_nudge_sizes, out_max_impact = maximum_nudges.find_max_control_impact(
            input_arr, cond_output, nudge_size
        )
        out_nudge = sorted(list(zip(out_nudge_states, out_nudge_sizes)),
                           key=lambda x: x[0])
        #print(out_nudge_states, out_nudge_sizes, out_max_impact)
        expected_nudge_states = [7, 4, 6]
        expected_nudge_sizes = [-0.1, -0.1, 0.2]
        expected_nudge = sorted(list(zip(expected_nudge_states, expected_nudge_sizes)),
                                key=lambda x: x[0])

        for count in range(len(out_nudge_states)):
            state, size = out_nudge[count]
            expected_state, expected_size = expected_nudge[count]
            self.assertEqual(state, expected_state)
            self.assertAlmostEqual(size, expected_size)

    def test2_find_max_control_impact(self):
        input_arr = np.array(
            [
                [0.1, 0.15, 0.05],
                [0.15, 0.1, 0.15],
                [0.1, 0.1, 0.05],
            ]
        )
        cond_output = np.array(
            [
                [
                    [0.1, 0.3, 0.3, 0.3],
                    [0.1, 0.0, 0.5, 0.4],
                    [0.4, 0.1, 0.2, 0.3]
                ],
                [
                    [0.4, 0.3, 0.2, 0.1],
                    [0.2, 0.25, 0.25, 0.3],
                    [0.45, 0.15, 0.25, 0.15]
                ],
                [
                    [0.1, 0.2, 0.25, 0.45],
                    [0.15, 0.10, 0.4, 0.35],
                    [0.2, 0.2, 0.3, 0.3]
                ]
            ]
        )
        nudge_size = 0.3
        out_nudge_states, out_nudge_sizes, out_max_impact = maximum_nudges.find_max_control_impact(input_arr, cond_output, nudge_size)
        #print(out_nudge_states, out_nudge_sizes, out_max_impact)
        out_nudge = sorted(list(zip(out_nudge_states, out_nudge_sizes)),
                           key=lambda x: x[0])
        #print(out_nudge_states, out_nudge_sizes, out_max_impact)
        expected_nudge_states = [3, 5, 1]
        expected_nudge_sizes = [-0.15, -0.15, 0.3]
        expected_nudge = sorted(list(zip(expected_nudge_states, expected_nudge_sizes)),
                                key=lambda x: x[0])

        for count in range(len(out_nudge_states)):
            state, size = out_nudge[count]
            expected_state, expected_size = expected_nudge[count]
            self.assertEqual(state, expected_state)
            self.assertAlmostEqual(size, expected_size)


class TestMaximumNudges(unittest.TestCase):
    def setUp(self):
        pass

    def test1_create_route(self):
        states = [0.1, 0.3, 0.15]
        scores = [-0.05, 0.1, 0.05]
        nudge_size = 0.2
        positive_scores = [0.25, 0.1, -0.01]
        route = maximum_nudges.create_route(states, scores, nudge_size, positive_scores)
        print(route)
        expected_route = [
            {"length":0.1, "height": 0.1},
            {"length":0.05, "height": - 0.1},
            {"length":0.05, "height": -0.14}
        ]
        for count, track in enumerate(expected_route):
            self.assertAlmostEqual(track["length"], route[count]["length"])
            self.assertAlmostEqual(track["height"], route[count]["height"])

    def test2_create_route(self):
        states = [0.25, 0.3, 0.3]
        scores = [-0.05, 0.1, 0.05]
        nudge_size = 0.2
        positive_scores = [-0.1, 0.05, -0.02]
        route = maximum_nudges.create_route(states, scores, nudge_size, positive_scores)
        print(route)
        expected_route = [
            {"length":0.2, "height": 0.1},
        ]
        for count, track in enumerate(expected_route):
            self.assertAlmostEqual(track["length"], route[count]["length"])
            self.assertAlmostEqual(track["height"], route[count]["height"])

    def test3_create_route(self):
        states = [0.19, 0.1, 0.05]
        scores = [-0.15, 0.2, 0.05]
        nudge_size = 0.2
        positive_scores = [-0.1, 0.05, -0.02]
        route = maximum_nudges.create_route(states, scores, nudge_size, positive_scores)
        print(route)
        expected_route = [
            {"length":0.05, "height": 0.1},
            {"length":0.05, "height": 0.07},
            {"length":0.09, "height": -0.18},
            {"length":0.01, "height": 0.07}
        ]
        for count, track in enumerate(expected_route):
            self.assertAlmostEqual(track["length"], route[count]["length"])
            self.assertAlmostEqual(track["height"], route[count]["height"])

