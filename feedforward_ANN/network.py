import numpy as np

class DimensionError(ValueError):pass

class Network():
    def __init__(self, batch_size, weight_decay, train_mode=True):
        """treat network as a stack"""
        self.train_mode = train_mode
        self.layers = []
        self.loss = None
        self.update = False
        self.weight_decay = weight_decay
        self.batch_size = batch_size

    def add_layer(self, layer):
        """adds layer to network
        
        layer: should implement the Layer class
        """
        if self.layers != []:
            if self.layers[-1].output_dim != layer.input_dim:
                error_message = ("the output dimension of the previous layer" +
                    "does not match the input dimension of the layer that" +
                    "is being added"
                )
                print(error_message)
                raise DimensionError()
        if layer.batch_size != self.batch_size:
            print("the layer to be added has batch size {}, it should be {}".format(
                layer.batch_size, self.batch_size
            ))
            raise DimensionError()

        self.layers.append(layer)

    def remove_layer(self):
        self.layers.pop()

    def add_loss(self, loss):
        self.loss = loss

    def forward(self, x):
        """perform a forward pass through the network and return the loss"""
        if x.shape[1] != self.layers[0].input_dim:
            print("the datapoints have dimension {} while {} was expected".format(
                x.shape[1], self.layers[0].input_dim
            ))
            raise DimensionError()
    
        z = x
        for layer in self.layers:
            z = layer.forward(z) 
        
        return z

    def backward(self):
        """performs a backward pass throughout the network"""
        loss_der = self.loss.compute_gradient()
        for layer in reversed(self.layers):
            loss_der = layer.backward(loss_der)
            
        self.update = True
        return loss_der

    def update_batch_size(self, batch_size):
        for layer in self.layers:
            layer.batch_size = batch_size

    def update_weights(self, learning_rate, regularize=True):
        """update the weights of all layers in the network and return None"""
        if not self.train_mode:
            raise ValueError("No updating allowed when not in train_mode")
        if self.update == False:
            print("WARNING: weights have already been used to update" +  
                  "network. Make sure that not using a backward pass first" +
                  "was intended"
            )

        for layer in self.layers:
            layer.update(learning_rate, regularize=regularize, 
                         weight_decay=self.weight_decay)

        self.update = False
       


