import unittest
import collections
import maximum_nudges

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

