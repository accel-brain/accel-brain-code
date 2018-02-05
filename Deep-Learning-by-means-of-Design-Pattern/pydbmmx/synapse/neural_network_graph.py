# -*- coding: utf-8 -*-
import numpy as np
import pyximport; pyximport.install()
from pydbmmx.synapse_list import Synapse
from pydbmmx.activation.logistic_function import LogisticFunction


class NeuralNetworkGraph(Synapse):
    '''
    The graph for neural networks.
    '''

    # If True, `Self` consider itself as the neural networks in output layer.
    __output_layer_flag = False
    # Logistic function.
    __logistic_function = None
    # Momentum factor.
    __momentum_factor_arr = None

    def __init__(self, output_layer_flag=False):
        '''
        Initialize.
        '''
        if isinstance(output_layer_flag, bool) is False:
            raise TypeError()
        self.__output_layer_flag = output_layer_flag
        self.__logistic_function = LogisticFunction()

    def back_propagate(
        self,
        propagated_list,
        learning_rate=0.05,
        momentum_factor=0.1,
        back_nn_list=None,
        back_nn_index=0
    ):
        '''
        Back propagate.

        Args:
            propagated_list:    The list of back propagated feature points.
                                If this is in output layer, the values of the list are response variable.  
            learning_rate:      Learning rate.
            momentum_factor:    Momentum factor.
            back_nn_list:       The list of graph of neural networks.
            back_nn_index:      The index of graph of neural networks.

        Returns:
            Tuple(`The back propageted data points`, `The activity`)
        '''
        if self.__output_layer_flag is True:
            if len(self.deeper_neuron_list) != len(propagated_list):
                raise IndexError()
            diff_list = [self.deeper_neuron_list[j].activity - propagated_list[j] for j in range(len(self.deeper_neuron_list))]
        else:
            diff_list = propagated_list

        diff_list = list(np.nan_to_num(np.array(diff_list)))
        diff_arr = np.array([[diff_list[k]] * len(self.shallower_neuron_list) for k in range(len(diff_list))]).T
        if self.__momentum_factor_arr is not None:
            momentum_arr = self.__momentum_factor_arr * momentum_factor
        else:
            momentum_arr = np.ones(diff_arr.shape) * momentum_factor

        self.diff_weights_arr = (learning_rate * diff_arr) + momentum_arr
        self.__momentum_factor_arr = diff_arr
        self.weights_arr = np.nan_to_num(self.weights_arr)
        error_arr = diff_arr * self.weights_arr
        error_list = error_arr.sum(axis=1)
        back_propagated_list = [self.__logistic_function.derivative(self.shallower_neuron_list[i].activity) * error_list[i] for i in range(len(self.shallower_neuron_list))]

        # Normalize.
        if len(back_propagated_list) > 1 and sum(back_propagated_list) != 0:
            back_propagated_arr = np.array(back_propagated_list)
            back_propagated_arr = back_propagated_arr / back_propagated_arr.sum()
            back_propagated_arr = np.nan_to_num(back_propagated_arr)
            back_propagated_list = list(back_propagated_arr)

        # Update weights.
        self.learn_weights()

        [self.shallower_neuron_list[_i].update_bias(learning_rate) for _i in range(len(self.shallower_neuron_list))]
        [self.deeper_neuron_list[_j].update_bias(learning_rate) for _j in range(len(self.deeper_neuron_list))]

        # Recursive.
        if back_nn_list is not None:
            if back_nn_index < len(back_nn_list) - 1:
                back_nn_list[back_nn_index + 1].back_propagate(
                    propagated_list=back_propagated_list,
                    learning_rate=learning_rate,
                    momentum_factor=momentum_factor,
                    back_nn_list=back_nn_list,
                    back_nn_index=back_nn_index + 1
                )