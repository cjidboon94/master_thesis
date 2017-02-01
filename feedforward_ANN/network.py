import numpy as np

class DimensionError(ValueError):pass

class Network():
    def __init__(self, train_mode):
        """treat network as a stack"""
        self.train_mode = train_mode
        self.layers = []
        self.loss = None
        self.update = False

    def add_layer(self, layer):
        """adds layer to network
        
        layer: should implement the Layer class
        """
        #add check out dimension matches in dimension
        self.layers.append(self, layer)

    def remove_layer(self):
        self.layers.pop()

    def add_loss(self, loss):
        self.loss = loss

    def forward(self, x):
        """perform a forward pass through the network and return the loss"""
        if x.shape[0] != self.layers[0].input_dim:
            raise DimensionError()
        z = x
        for layer in self.layers:
            z = layer.forward(z) 
        
        return self.loss.compute_loss(z)

    def backward(self):
        """performs a backward pass throughout the network"""
        loss_der = self.loss.compute_gradient()
        for layer in reversed(self.layers):
            loss_der = layer.backward(loss_der)
            
        self.update = True
        return loss_der
    
    def update_weights(self, learning_rate):
        """update the weights of all layers in the network and return None"""
        if self.train_mode:
            raise ValueError("No updating allowed when not in train_mode")
        if self.update == False:
            print("WARNING: weights have already been used to update" +  
                  "network. Make sure that not using a backward pass first" +
                  "was intended"
            )

        for layer in self.layers:
            layer.update(learning_rate)

        self.update = False
       


