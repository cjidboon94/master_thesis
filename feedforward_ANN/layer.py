from abc import ABCMeta, abstractmethod
import numpy as np

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
    
    def init(self, input_dim, output_dim, batch_size):
        self.data = { 
            "batch_size":batch_size,
            "input_dim": input_dim,
            "output_dim": output_dim,
            "input": None,
            "output": None
        }

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

    def update(self, learning_rate):
        """update the parameters of the module

        note: the updating always looks as: W = W - learning_rate*np.mean(dW)
              whether it is an minimazation/maximisation problem should be
              adjusted for with the sign of the learning rate 
        """
        pass

class LinearLayer(Layer):
    """linear map from dim batch_size*input_dim to batch_size*output_dim"""

    def init(self, input_dim, output_dim, batch_size):
        super().__init__(input_dim, output_dim, batch_size)    
        self.W = np.random.normal((input_dim, output_dim))

    def forward(self, x):
        """computes forward go
        
        params:
            x: a np matrix with dimensions batch_size times input_size

        returns: numpy array with dimensions batch_size times output_size
        """

        self.data["input"] = x
        return x @ W

    def backward(self, prev_der):
        """prev_der should have dimension batch_size times output_dim
        
        returns: derivative of whole network with respect to inputs this 
                 layer. It has dimensions batch_size times input_dim

        """
        out = np.zeros((batch_size, input_dim))
        W_T = np.transpose(self.W)
        for i in range(batch_size):
            out[i] = prev_der[i] @ W_T

        self.dW = np.zeros((batch_size, input_dim, output_dim))
        for j in range(batch_size):
            for k in range(self.data['input_dim']):
                for l in range(self.data['output_dim']):
                    self.dW[j, k, l] = self.data['input'][j, k] * prev_der[j, l]  

        if self.dW.shape[1:] != self.W.shape:
            raise DimensionError()
        return out

    def update(self, learning_rate):
        self.W = self.W - learning_rate*np.mean(self.dW, 0) 
        

class ReLuLayer(Layer):
    def init(self, input_dim, output_dim, batch_size):
        super().__init__(input_dim, output_dim, batch_size)

    def forward(self, x):
        self.input = x
        return np.maximum(x, 0)

    def backward(self, prev_der):
        der = np.copy(self.input)
        der[der>0] = 1
        der[der<0] = 0
        return prev_der @ der

class TanHLayer(Layer):
    def init(self, input_dim, output_dim, batch_size):
        super().__init__(input_dim, output_dim, batch_size)

    def forward(self, x):
        self.input = x
        return np.tanh(x)
              
    def backward(self, prev_der):
        return prev_der @ (1- (np.tanh(self.input) @ np.tanh(self.input)))



