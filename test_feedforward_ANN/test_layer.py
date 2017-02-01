import unittest
import numpy as np
from feedforward_ANN import layer

class TestLinearLayer(unittest.TestCase):
    def setUp(self):
        pass

    def test_forward(self):
        """test whether forward outputs the right dimensions""" 
        input_dim, output_dim, batch_size = 2, 5, 3
        linear = layer.LinearLayer(input_dim, output_dim, batch_size)
        x = np.random.random((batch_size, input_dim))
        output = linear.forward(x)
        self.assertEqual((batch_size, output_dim), output.shape)

    def test_backward_dim(self): 
        input_dim, output_dim, batch_size = 2, 5, 3
        linear = layer.LinearLayer(input_dim, output_dim, batch_size)
        x = np.random.random((batch_size, input_dim))
        linear.forward(x)
        out = linear.backward(np.random.random((batch_size, output_dim)))
        self.assertEqual(out.shape, (batch_size, input_dim))

    def test_back_first_example(self):
        input_dim, output_dim, batch_size = 2, 2, 2 
        linear = layer.LinearLayer(input_dim, output_dim, batch_size)
        x = np.array([[1,2],[0.5, -1]])
        linear.forward(x)
        prev_der = np.array([[5, 0.5], [2, -3]])
        out = linear.backward(prev_der)
        self.assertTrue(
            np.all(linear.dW == np.array([
                [
                    [5*x[0,0], 0.5*x[0,0]], 
                    [5*x[0,1], 0.5*x[0,1]]
                ], 
                [
                    [2*x[1,0], -3*x[1,0]], 
                    [2*x[1,1], -3*x[1,1]]
                ]
            ]))
        )
        self.assertTrue(np.all(out==prev_der @ np.transpose(linear.W)))

class TestReLuLayer(unittest.TestCase):
    def setUp(self):
        pass

    def test_forward(self):
        pass

    def test_backwards(self):
        pass

if __name__=="__main__":
    unittest.main()
