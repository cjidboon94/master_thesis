import numpy as np

class SGD():
    def __init__(self, network, X, y, learning_rate, iterations, batch_size):
        """train a feedforward ANN using backprop on data X, y
        
        params:
            X: a numpy matrix where every row is a datapoint 
            y: a numpy matrix where every row is a one hot encoding of the 
               right class of the corresponding datapoint in X    
        """
        self.X = X
        self.y = y
        self.network = network
        self.learning_rate = learning_rate
        self.iterations = iterations
        self.batch_size = batch_size

    def get_batch_data(self):
        """return a batch of data points and classes (of X and y)"""
        chosen_data_points = np.random.choice(X.shape[1], self.batch_size)
        return self.X[chosen_data_points], self.y[chosen_data_points]

    def train(self):
        loss_li = []
        for i in range(iterations):
            batch_data = self.get_batch_data()
            self.network.forward(batch_data)
            self.network.backward()
            self.network.update(self.learning_rate)
