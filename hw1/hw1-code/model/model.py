from base import BaseModel
import model.function as F
import model.nn as nn
import numpy as np

""" 
    This file contains all the models for this project
    Define your model with input, output dim here, inherit the base model class.
"""
class Shallow_Network(BaseModel):
    def __init__(self, input_dim=28*28, output_dim=10):

        # Initialize all model weights here
        self.fc1 = nn.Linear(input_dim, output_dim=300)
        self.fc2 = nn.Linear(input_dim=300, output_dim=output_dim)

        # Calculate trainable params for logging
        n_params = 0
        for i in self.fc1.values():
            temp = 1
            for j in i.shape:
                temp *= j
            n_params += temp
        for i in self.fc2.values():
            temp = 1
            for j in i.shape:
                temp *= j
            n_params += temp

        # Inherit the BaseModel class
        super().__init__(input_dim, output_dim, n_weights=n_params)

    def forward(self, x):
        """ 
            Forward pass of the model. For the architecture, refer to the 
            report.
        """

        # Input layer
        self.tensors['y0'] = x

        # Input layer -> fc1
        self.tensors['y1'] = np.matmul(self.fc1['W'], self.tensors['y0']) + self.fc1['b']
        
        # ReLU layer
        self.tensors['y2'] = F.relu(self.tensors['y1'])

        # Output Layer
        self.tensors['y3'] = np.matmul(self.fc2['W'], self.tensors['y2']) + self.fc2['b']

        # Output layer
        self.tensors['y'] = F.softmax(self.tensors['y3'])

        # return classified results
        return self.tensors['y']
    
    def backward(self, L_bp, m_batch):
        """
            Implement back prop process and compute the gradients
            :m_batch: mini batch size
            :L_bp: Backward pass of the loss
            :return: gradients
        """
        gradients = {}

        # The upstream gradient from the loss with softmax loss
        d_y = L_bp

        # Calculate the local gradients and multiply with upstream gradient
        gradients['dW2'] = (1. / m_batch) * np.matmul(d_y, self.tensors['y2'].T)
        gradients['db2'] = (1. / m_batch) * np.sum(d_y, axis=1, keepdims=True)

        # Calculate gradient of ReLU
        d_y2 = np.matmul(self.fc2['W'].T, d_y)
        d_y1 = np.multiply(d_y2, np.where(self.tensors['y1'] <= 0, 0, 1))

        gradients['dW1'] =  (1. / m_batch) * np.matmul(d_y1, self.tensors['y0'].T)
        gradients['db1'] =  (1. / m_batch) * np.sum(d_y1, axis=1, keepdims=True)

        return gradients

    def optimize(self, gradients):
        """
            Gradient update using GD
        """
        self.learning_rate = 0.1

        self.fc1['W'] -= self.learning_rate * gradients['dW1']
        self.fc1['b'] -= self.learning_rate * gradients['db1']

        self.fc2['W'] -= self.learning_rate * gradients['dW2']
        self.fc2['b'] -= self.learning_rate * gradients['db2']

    def resume(self, path):
        """
            Load in the stored parameters
        """
        parameters = np.load(path)
        self.fc1['W'] = parameters['w1']
        self.fc1['b'] = parameters['b1']
        self.fc2['W'] = parameters['w2']
        self.fc2['b'] = parameters['b2']

