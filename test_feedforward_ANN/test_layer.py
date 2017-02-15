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
        input_dim, output_dim, batch_size = 3, 3, 2
        relu = layer.ReLuLayer(input_dim, batch_size)
        x = np.array(
            [
                [2, -3, 0],
                [3, 20, -5],
                [5, 6, 7],
            ]
        )
        pred_out = relu.forward(x)
        out = np.array(
            [
                [2, 0, 0],
                [3, 20, 0],
                [5, 6, 7]
            ]
            )
        self.assertTrue(np.all(out==pred_out))

    def test_backwards(self):
        input_dim, output_dim, batch_size = 3, 3, 2
        relu = layer.ReLuLayer(input_dim, batch_size)
        x = np.array(
            [
                [4, -9, 0],
                [5, 20, -5],
            ]
        )
        relu.forward(x)
        prev_der = np.array(
            [
                [1, 10, -4],
                [20, 25, 40]
            ]
        )

        pred_out = relu.backward(prev_der) 
        out = np.array(
            [
                [1, 0, 0],
                [20, 25, 0]
            ]
        )
        self.assertTrue(np.all(out==pred_out))

class TestSigmoidLayer(unittest.TestCase):
    def setUp(self):
        pass

    def test_forward(self):
        sigmoid = layer.SigmoidLayer(3, 2)
        x = np.array(
            [
                [4, -9, 0],
                [5, 20, -5],
            ]
        )
        pred_out = sigmoid.forward(x)
        out = np.array(
            [
                [
                    1/(1+np.exp(-x[0,0])), 
                    1/(1+np.exp(-x[0,1])),
                    1/(1+np.exp(-x[0,2]))
                ],
                [
                    1/(1+np.exp(-x[1,0])), 
                    1/(1+np.exp(-x[1,1])),
                    1/(1+np.exp(-x[1,2]))
                ],
            ]
        )
        self.assertTrue(np.allclose(out, pred_out))

    def test_backward(self):
        sigmoid = layer.SigmoidLayer(3, 2)
        x = np.array(
            [
                [4, -9, 0],
                [5, 20, -5],
            ]
        )
        forward_out = sigmoid.forward(x)
        prev_der = np.array(
            [
                [0, 5, 7],
                [-1, 5, -2]
            ]
        )

        pred_out = sigmoid.backward(prev_der)
        self.assertEqual(pred_out.shape, (2,3))
        #print(pred_out)
        #print(forward_out[1,1]*(1-forward_out[1,1]) * prev_der[1,1])
        self.assertAlmostEqual(pred_out[1,1], 
                         forward_out[1,1]*(1-forward_out[1,1]) * prev_der[1,1])
    
if __name__=="__main__":
    unittest.main()
