import unittest
import numpy as np
from feedforward_ANN import loss

class TestSoftMaxCrossEntropyLoss(unittest.TestCase):
    def setUp(self):
        pass

    def test_softmax(self):
        cross_entropy = loss.SoftmaxCrossEntropyLoss()
        x = np.array([
                [np.log(2), np.log(4), np.log(0.5)],
                [100, 200, 201],
                [np.log(1.0001), np.log(1.0002), np.log(1.000001)],
            ])
        out = cross_entropy.softmax(x)
        max_x1 = np.amax(x[1])
        softmax_x = np.array([
            [2/6.5, 4/6.5, 0.5/6.5],
            [np.exp(x[1,i]-max_x1) / sum([np.exp(entry-max_x1) for entry in x[1]])
             for i in range(3)],
            [
                1.0001/(1.0001+1.0002+1.000001), 
                1.0002/(1.0001+1.0002+1.000001), 
                1.000001/(1.0001+1.0002+1.000001)
            ]
        ]) 
        #print(out)
        #print(softmax_x)
        self.assertTrue(np.allclose(out, softmax_x))

    def test_compute_loss(self):
        cross_entropy = loss.SoftmaxCrossEntropyLoss()
        x1 = np.array(
            [
                [
                    np.log(10**(-11)), np.log(10**(-8)), 2*np.log(10**(-8))
                ],
                [
                    np.log(2), np.log(3), np.log(2.5)
                ],
            ]
        )
        y = np.array(
            [
                [
                    0, 0, 1
                ],
                [
                    1, 0, 0
                ]
            ]
        )
        x2 = cross_entropy.softmax(x1) + 10**(-11)
        loss_x1 = np.array([-np.log(x2[0, 2]), -np.log(x2[1,0])])
        pred_loss = cross_entropy.compute_loss(x1, y)
        #print(cross_entropy.soft)
        #print(x2)
        #print(pred_loss)
        #print(loss_x1)
        self.assertTrue(np.allclose(pred_loss, loss_x1))

    def test_compute_gradient(self):
        cross_entropy = loss.SoftmaxCrossEntropyLoss()
        x = np.array(
            [
                [np.log(2), np.log(5)],
                [np.log(100), np.log(250)]
            ]
        )
        y = np.array(
            [
                [0, 1],
                [1, 0]
            ]
        )
        first_loss = cross_entropy.compute_loss(x, y)
        grad = np.array(
            [
                [2/7, -(1 - (5/7))],
                [-(1 - (100/350)), 250/350],
            ]
        )
        pred_grad = cross_entropy.compute_gradient()
        self.assertTrue(np.allclose(grad, pred_grad))

if __name__=="__main__":
    unittest.main()
