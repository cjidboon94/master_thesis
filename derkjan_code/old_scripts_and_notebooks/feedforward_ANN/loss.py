import numpy as np

class SoftmaxCrossEntropyLoss():
    """ compute the combined softmax and cross-entropy on batches

    note: one hot encoding for y is assumed
    """

    def init(self):
        pass

    def compute_loss(self, x, y):
        """compute the softmax cross entropy loss
        
        params: 
            x: a numpy array with dimensions batch_size times input_dim
            y: a numpy array with dimensions batch_size times input_dim
               where for every batch (thus every row) a one hot encoding 
               is used
        
        returns: an array of batch_size representing the losses encountered
 
        """

        self.batch_size = x.shape[0]
        self.x = x
        self.y = y
        self.soft = self.softmax(x) + 10**(-11)
        out = np.zeros(self.batch_size)
        for i in range(self.batch_size):
            out[i] = -(y[i] @ np.log(self.soft[i]))

        return out

    def compute_gradient(self): 
        return -(self.y - self.soft)

    def softmax(self, x):
        """Compute softmax values for each sets of scores in x
        
        params:
            x: a numpy array with dim batch_size times input_dim 

        returns: a numpy array where the softmax function has been applied 
                 to each row of dim batch_size times input_dim
        """

        out = np.zeros(x.shape)
        for i in range(x.shape[0]):
            max_x = x[i] - np.max(x[i])
            out[i] = np.exp(max_x) / np.sum(np.exp(max_x), axis=0)

        return out



