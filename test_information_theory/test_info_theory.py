import unittest
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

