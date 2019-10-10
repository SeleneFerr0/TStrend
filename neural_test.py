# -*- coding: utf-8 -*-
"""
Created on Wed Sep 25 11:24:04 2019

@author: seleneferro
"""

# coding:utf-8
import numpy as np 

class NeuralNetwork(): 
    # rand initial weight
    def __init__(self): 
        np.random.seed(1) 
        self.synaptic_weights = 2 * np.random.random((3, 1)) - 1 
    
    #sigmoid
    def sigmoid(self, x):  
        return 1 / (1 + np.exp(-x)) 
    
    def sigmoid_derivative(self, x): 
        return x * (1 - x)
    
    def train(self, training_inputs, training_outputs,learn_rate, training_iterations): 
        for iteration in range(training_iterations): 
            output = self.think(training_inputs) 
            error = training_outputs - output 
            adjustments = np.dot(training_inputs.T, error * self.sigmoid_derivative(output)) 
            self.synaptic_weights += learn_rate*adjustments 
    
    def think(self, inputs): 
        inputs = inputs.astype(float) 
        output = self.sigmoid(np.dot(inputs, self.synaptic_weights)) 
        return output 

if __name__ == "__main__": 
    neural_network = NeuralNetwork() 

    train_data=[[0,0,1], [1,1,1], [1,0,1], [0,1,1]]
    training_inputs = np.array(train_data) 

    training_outputs = np.array([[0,1,1,0]]).T 

    learn_rate=0.1

    epoch=150000
    neural_network.train(training_inputs, training_outputs, learn_rate, epoch) 

    pre_data=[0,0,1]
    print(neural_network.think(np.array(pre_data)))