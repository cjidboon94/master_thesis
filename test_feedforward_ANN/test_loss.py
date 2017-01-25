import unittest
import numpy as np
from feedforward_ANN import loss

class TestLinearLayer(unittest.TestCase):
    def setUp(self):
        pass

    def test_right_output_dim(self):
        cross_entropy = loss.SoftmaxCrossEntropyLoss()

if __name__=="__main__":
    unittest.main()
