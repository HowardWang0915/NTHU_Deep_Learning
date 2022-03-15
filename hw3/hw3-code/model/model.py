from base import BaseModel
from model.function import *
import numpy as np
import matplotlib.pyplot as plt

""" 
    This file contains all the models for this project
    Define your model with input, output dim here, inherit the base model class.
"""
class Fruit_CNN(BaseModel):
    def __init__(self, in_channel=3, out_shape=3):
        self.in_channels = in_channel
        self.out_shape = out_shape

        # Conv1
        self.conv1 = Conv2d(in_channels=self.in_channels, out_channels=8, kernel_size=3, strides=1, padding=(0, 0))
        self.relu1 = ReLU()
        # Conv 2
        self.conv2 = Conv2d(in_channels=8, out_channels=16, kernel_size=3, strides=1, padding=(0, 0))
        self.relu2 = ReLU()

        # Flatten the features
        self.flatten = Flatten()

        # Dense layer
        self.fc1 = Dense(in_dims=16*32*32, out_dims=64)
        self.relu3 = ReLU()
        self.fc2 = Dense(in_dims=64, out_dims=self.out_shape)

        self.softmax = softmax
        self.layers = [self.conv1, self.conv2, self.fc1, self.fc2]
        self.weights = self.__get_weights()
        self.n_weights = self.__cal_weights()
        super().__init__(input_dim=32*32*3, output_dim=3, n_weights=self.n_weights)

    def forward(self, x):
        """ 
            Forward pass of the model. For the architecture, refer to the 
            report.
        """
        x = self.conv1.forward(x)
        x = self.relu1.forward(x)
        x = self.conv2.forward(x)
        x = self.relu2.forward(x)

        # Flatten 
        x = self.flatten.forward(x)

        # Fully connected layer
        x = self.fc1.forward(x)
        x = self.relu3.forward(x)
        x = self.fc2.forward(x)

        # output
        output = self.softmax(x)

        return output

    def backward(self, L_bp):
        """
            Implement back prop process and compute the gradients
            :m_batch: mini batch size
            :L_bp: Backward pass of the loss
            :return: gradients
        """
        # Get the backward loss from softmax and cross-entropy loss
        dL = L_bp

        # Back prop the fc layers 
        dL = self.fc2.backward(dL)
        dL = self.relu3.backward(dL)
        dL = self.fc1.backward(dL)
        dL = self.flatten.backward(dL)

        # C2
        dL = self.relu2.backward(dL)
        dL = self.conv2.backward(dL)
        
        # C1
        dL = self.relu1.backward(dL)
        self.conv1.backward(dL)

    def __cal_weights(self):
        n_weights = 0
        for layer in self.layers:
            n_weights += layer.params['W'].size
            n_weights += layer.params['b'].size
        return n_weights

    def __get_weights(self):
        weights = {}
        for i, layer in enumerate(self.layers):
            weights['W' + str(i + 1)] = layer.params['W']
            weights['b' + str(i + 1)] = layer.params['b']
        return weights

    def resume(self, path):
        """
            Load in the stored parameters
        """
        parameters = np.load(path)
        for i, layer in enumerate(self.layers):
            layer.params['W'] = parameters['W' + str(i + 1)]
            layer.params['b'] = parameters['b' + str(i + 1)]

