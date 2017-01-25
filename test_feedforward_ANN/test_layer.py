import unittest
import numpy as np
from feedforward_ANN import layer

class TestLinearLayer(unittest.TestCase):
    def setUp(self):
        pass

    def test_right_output_dim(self):
        input_dim, output_dim, batch_size = 2, 5, 3
        linear = layer.LinearLayer(input_dim, output_dim, batch_size)
        x = np.random.random((batch_size, input_dim))
        output = linear.forward(x)
        self.assertEqual((batch_size, output_dim), output.shape)

    def test_backward(self): 
        input_dim, output_dim, batch_size = 2, 5, 3
        linear = layer.LinearLayer(input_dim, output_dim, batch_size)
        x = np.random.random((batch_size, input_dim))
        linear.forward(x)
        out = linear.backward(np.random.random((batch_size, output_dim)))
        self.assertEqual(out.shape, (batch_size, input_dim))

if __name__=="__main__":
    unittest.main()
