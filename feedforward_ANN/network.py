import numpy as np
from feedforward_ANN import layer

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
        self.layer_types = ["linear", "relu"]

    def add_layer(self, layer_type, dim_out=None, input_dim=None, params=None):
        """adds layer to network
        
        params:
            layer_type: the type of the layer, should be in self.layer_types
            dim_out: the dimension of the output
            input_size: only used for first layer
            params: the parameters used to initialise the layer if any
        """

        if self.layers != [] and input_dim is not None:
            print(("WARNING: input size given, since this is not the first" + 
                   "layer, the input_size will be determined automatically")) 

        if self.layers == [] and input_dim is None:
            raise TypeError("for the first layer an input size is needed")
        elif self.layers == []:
            dim_in = input_dim
        else:
            dim_in = self.layers[-1].output_dim    
          
        if layer_type == 'linear':
            if dim_out is None:
                raise TypeError("specify an output_dim to add a linear layer")
            self.layers.append(
                layer.LinearLayer(dim_in, dim_out, self.batch_size, **params)
            )
        elif layer_type == 'relu':
            self.layers.append(layer.ReLuLayer(dim_in, self.batch_size))

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
       


