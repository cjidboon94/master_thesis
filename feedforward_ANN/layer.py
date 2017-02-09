from abc import ABCMeta, abstractmethod
import numpy as np
from scipy.special import expit

class DimensionError(Exception): pass

class Layer():
    """represents an abstract layer in a neural network 
    
    note: The parameters (if any) of each layer are assumed to be matrices.
          Moreover, the input to each layer are assumed to be matrices, where
          every row represent one input in a batch, so the rows can be viewed
          independently. Every function outputs a matrix where the rows 
          represent the output of the same row in the matrix. 

    """

    __metaclass__ = ABCMeta
    
    def __init__(self, input_dim, output_dim, batch_size):
            self.batch_size = batch_size
            self.input_dim = input_dim
            self.output_dim = output_dim
            self.input = None
            self.output = None

    @abstractmethod
    def forward(self, x):
        """performs forward pass ANN
        
        params:
            x: should have dimension B times M, where B is the amount of 
               batches used and M is the output dimension
        """
        
        pass

    @abstractmethod
    def backward(self, prev_jac):
        """performs backward pass ANN
        
        params:
            prev_der: the jacobian of the previous layer with respect to 
                      its inputs, it should have dimensions B times M, where
                      B is the batch size and M the output dim of this layer

        note: This method performs two tasks. The first is to calculate the 
              gradient of the overall network with respect to the inputs of 
              this layer. This value is returned. It also should calculate
              the gradient(of the whole network) with respect to its 
              parameters and save the gradients of every input of the batch.
        
        returns: the gradient of the overall network with respect to the 
                 inputs of this layer.

        """

        pass

    @abstractmethod
    def update(self, learning_rate):
        """update the parameters of the module

        note: the updating always looks as: W = W - learning_rate*np.mean(dW)
              whether it is an minimazation/maximisation problem should be
              adjusted for with the sign of the learning rate 
        """
        pass

class LinearLayer(Layer):
    """linear map from dim batch_size*input_dim to batch_size*output_dim"""

    def __init__(self, input_dim, output_dim, batch_size, weight_scale_w=0.0001, mean_b=0.01, scale_b=0.001):
        super().__init__(input_dim, output_dim, batch_size)
        self.W = np.random.normal(loc=0, scale=weight_scale_w, size=(input_dim, output_dim))
        self.b = np.random.normal(loc=mean_b, scale=scale_b, size=output_dim)
        
    def forward(self, x):
        """computes forward go
        
        params:
            x: a np matrix with dimensions batch_size times input_size

        returns: numpy array with dimensions batch_size times output_size
        """

        self.input = x
        return x @ self.W + np.tile(self.b, (self.batch_size, 1))

    def backward(self, prev_der):
        """prev_der should have dimension batch_size times output_dim
        
        returns: derivative of whole network with respect to inputs this 
                 layer. It has dimensions batch_size times input_dim

        """
        
        #still add b
        out = np.zeros((self.batch_size, self.input_dim))
        W_T = np.transpose(self.W)
        for i in range(self.batch_size):
            out[i] = prev_der[i] @ W_T

        self.db = np.zeros((self.batch_size, self.output_dim))
        self.db = prev_der 

        self.dW = np.zeros((self.batch_size, self.input_dim, self.output_dim))
        for j in range(self.batch_size):
            for k in range(self.input_dim):
                for l in range(self.output_dim):
                    self.dW[j, k, l] = self.input[j, k] * prev_der[j, l]  

        if self.dW.shape[1:] != self.W.shape:
            raise DimensionError()
        return out

    def update(self, learning_rate, regularize=True, weight_decay=0.001):
        if regularize:
            self.W = ((1- learning_rate*weight_decay) * self.W -
                      learning_rate*np.mean(self.dW, 0))
        else:
            self.W = self.W - learning_rate*np.mean(self.dW, 0)

        self.b = self.b - learning_rate*np.mean(self.db, 0)

class ReLuLayer(Layer):
    def __init__(self, input_dim, output_dim, batch_size):
        if input_dim != output_dim:
            print("please specify the same input_dim as output_dim for ReLU")
            raise DimensionError()

        super().__init__(input_dim, output_dim, batch_size)

    def forward(self, x):
        """performs max(0, x) on every entry of x
        
        params: 
            x: a numpy array with dimensions batch size times input_dim
        """
        self.input = x
        return np.maximum(x, 0)

    def backward(self, prev_der):
        out = np.zeros((self.batch_size, self.input_dim))
        for i in range(self.batch_size):
            for j in range(self.input_dim):
                out[i,j] = prev_der[i, j] if self.input[i, j] > 0 else 0 

        return out

    def update(self, learning_rate, regularize=True, weight_decay=0.001):
        pass

class TanHLayer(Layer):
    def __init__(self, input_dim, output_dim, batch_size):
        super().__init__(input_dim, output_dim, batch_size)
        if input_dim != output_dim:
            print("please specify the same input_dim as output_dim for tanh")
            raise DimensionError()

    def forward(self, x):
        self.input = x
        self.output = np.tanh(x)
        return self.output
              
    def backward(self, prev_der):
        out = np.zeros((self.batch_size, self.input_dim))
        for i in range(self.batch_size):
            for j in range(self.input_dim):
                der = 1 - self.output[i, j]**2
                out[i,j] = prev_der[i, j] * der 

        return out

    def update(self, learning_rate, regularize=True, weight_decay=0.001):
        pass

class SigmoidLayer(Layer):
    def __init__(self, input_dim, output_dim, batch_size):
        super().__init__(input_dim, output_dim, batch_size)
        if input_dim != output_dim:
            print("please specify the same input_dim as output_dim for sigmoid")
            raise DimensionError()

    def forward(self, x):
        self.input = x
        self.output = expit(x)
        return self.output
              
    def backward(self, prev_der):
        out = np.zeros((self.batch_size, self.input_dim))
        for i in range(self.batch_size):
            for j in range(self.input_dim):
                der = self.output[i, j] * (1- self.output[i, j])
                out[i,j] = prev_der[i, j] * der 

        return out

    def update(self, learning_rate, regularize=True, weight_decay=0.001):
        pass

