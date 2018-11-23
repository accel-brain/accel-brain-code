# -*- coding: utf-8 -*-
import numpy as np
from pyqlearning.function_approximator import FunctionApproximator
from pydbm.cnn.convolutional_neural_network import ConvolutionalNeuralNetwork
from pydbm.cnn.layerablecnn.convolution_layer import ConvolutionLayer


class ConvolutionalNeuralNetworks(FunctionApproximator):
    '''
    Convolutional Neural Networks as a Function Approximator.
    '''
    
    def __init__(self):
        pass

    def learn_q(self, state_key_arr, action_key_arr, new_q):
        '''
        Infernce Q-Value.
        
        Args:
            state_key_arr:      `np.ndarray` of state.
            action_key_arr:     `np.ndarray` of action.
            new_q:              Q-Value.
        '''
        pass

    def inference_q(self, state_key_arr, next_action_arr):
        '''
        Infernce Q-Value.
        
        Args:
            state_key_arr:      `np.ndarray` of state.
            next_action_arr:     `np.ndarray` of action.
        
        Returns:
            `np.ndarray` of Q-Values.
        '''
        pass
