import unittest
import numpy as np

from feedforward_ANN import cifar10_utils as cifar

class TestCifar10Utils(unittest.TestCase):
    def setUp(self):
        pass

    def test_transform_label_encoding_to_one_hot(self):
        x = np.array([3,3,2,1,0])
        num_classes = 4
        outcome = cifar.transform_label_encoding_to_one_hot(x, num_classes)
        out = np.array(
            [
                [0,0,0,1],
                [0,0,0,1],
                [0,0,1,0],
                [0,1,0,0],
                [1,0,0,0]
            ]
        )
        print(outcome)
        print(out)
        self.assertTrue(np.all(out==outcome))
