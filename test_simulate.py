import unittest
from collections import OrderedDict
import numpy as np
import simulate

class TestSimulate(unittest.TestCase):
    def setUp(self):
        self.tryout_dict = {
            'a': [1, 4, 6, 8, 9, 7],
            'b': [10, 2, 5, 3, 6, 1],
            'x': [8, 3, 1, 3, 6, 17],
            'e': [1, 7, 6, 5, 19, 7]
        }

    def test_calculate_batches_std(self):
        BATCH_SIZE = 2
        tryout_dict_sorted = OrderedDict(
            sorted(self.tryout_dict.items(), key=lambda x: x[0])
        )
        out = simulate.calculate_batch_std(tryout_dict_sorted, BATCH_SIZE)
        expected = []
        expected.append(np.std([2.5, 7, 8]))
        expected.append(np.std([6, 4, 3.5]))
        expected.append(np.std([4, 5.5, 13])) 
        expected.append(np.std([5.5, 2, 11.5]))

        self.assertTrue(np.allclose(np.array(out), np.array(expected)))


