import unittest
import numpy as np

import nudge

class TestNudge(unittest.TestCase):
    def setUp(self):
        pass

    def test1_nudge(self):
	distribution = np.array([0.001, 0.001, 0.003, 0.995])
	sum_nudged_distribution = np.zeros(distribution.shape)
	for i in range(10):
	    self.assertTrue(np.all(nudge.nudge(distribution, 0.01) >= 0))
	    self.assertTrue(np.all(nudge.nudge(distribution, 0.01) <= 1))
     
    def test2_nudge(self):
        distribution = np.array([0.3, 0.4, 0.3, 0.1])
        sum_nudged_distribution = np.zeros(distribution.shape)
        for i in range(1000):
            sum_nudged_distribution = (sum_nudged_distribution + 
                                       nudge.nudge(distribution, 0.01))

        self.assertTrue(np.all((sum_nudged_distribution/1000)-distribution < 0.001))

