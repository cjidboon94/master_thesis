import unittest
import time
import numpy as np
from information_theory import info_theory

class TestInformationTheory(unittest.TestCase):
    def setUp(self):
        pass

    def test_calculate_entropy(self):
        dist = np.array([0.25, 0.5, 0.25])
        entropy = info_theory.calculate_entropy(dist)
        exp_entropy = 1.5
        self.assertAlmostEqual(entropy, exp_entropy)

    def test_calculate_mutual_information(self):
        dist1 = np.array([0.5, 0.5])
        dist2 = np.array([0.5, 0.5])
        mut_dist = np.array([0.25, 0.25, 0.25, 0.25])
        mutual_info = info_theory.calculate_mutual_information(
            dist1, dist2, mut_dist
        )

        exp_mutual_info = 0
        self.assertAlmostEqual(mutual_info, exp_mutual_info)

    def test_points_to_dist(self):
        start = time.time()
        for i in range(10000):
            points = np.random.randint(0, 2, size=(10000,15))
            info_theory.points_to_dist(points)

        print("using pandas {}".format(time.time()-start))
        start = time.time()
        for i in range(10000):
            points = np.random.randint(0, 2, size=(10000, 1))
            unique_out, counts_out = np.unique(points, return_counts=True)
            out_dist = counts_out/np.sum(counts_out) 

        print("using numpy on 1D {}".format(time.time()-start))
        print(info_theory.points_to_dist(points))
