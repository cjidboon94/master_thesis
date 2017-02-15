import numpy as np

class SGD():
    def __init__(self, network, X, y, batch_size):
        """train a feedforward ANN using backprop on data X, y
        
        params:
            X: a numpy matrix where every row is a datapoint 
            y: a numpy matrix where every row is a one hot encoding of the 
               right class of the corresponding datapoint in X    
        """
        self.X = X
        self.y = y
        self.network = network
        self.batch_size = batch_size

    def get_batch_data(self):
        """return a batch of data points and classes (of X and y)"""
        chosen_data_points = np.random.choice(self.X.shape[1], self.batch_size)
        return self.X[chosen_data_points], self.y[chosen_data_points]

    def train(self, iterations, learning_rate, verbose=True):
        loss_li = []
        for i in range(iterations):
            if verbose and i%50==0 and i!=0:
                print("{} iterations have passed".format(i))
            batch_data = self.get_batch_data()
            z = self.network.forward(batch_data[0])
            self.network.loss.compute_loss(z, batch_data[1])

            self.network.backward()
            self.network.update_weights(learning_rate)
