# -*- coding: utf-8 -*-
import numpy as np
cimport numpy as np
from pydbm.dbm.interface.dbm_builder import DBMBuilder
from pydbm.dbm.dbm_director import DBMDirector
from pydbm.activation.interface.activating_function_interface import ActivatingFunctionInterface
from pydbm.approximation.interface.approximate_interface import ApproximateInterface
ctypedef np.float64_t DOUBLE_t


class DeepBoltzmannMachine(object):
    '''
    The `Client` in Builder Pattern,
    
    Build deep boltzmann machine.
    '''

    # The list of restricted boltzmann machines.
    __rbm_list = []
    # The dict of Hyper parameters.
    __hyper_param_dict = {}

    def __init__(
        self,
        dbm_builder,
        neuron_assign_list,
        activating_function_list,
        approximate_interface_list,
        double learning_rate,
        double dropout_rate=0.5
    ):
        '''
        Initialize deep boltzmann machine.

        Args:
            dbm_builder:            `    Concrete Builder` in Builder Pattern.
            neuron_assign_list:          The number of neurons in each layers.
            activating_function_list:    Activation function.
            approximate_interface_list:  The object of function approximation.
            learning_rate:               Learning rate.
            dropout_rate:                Dropout rate.
        '''
        dbm_builder.learning_rate = learning_rate
        dbm_builder.dropout_rate = dropout_rate
        dbm_director = DBMDirector(
            dbm_builder=dbm_builder
        )
        dbm_director.dbm_construct(
            neuron_assign_list=neuron_assign_list,
            activating_function_list=activating_function_list,
            approximate_interface_list=approximate_interface_list
        )
        self.__rbm_list = dbm_director.rbm_list

    def learn(
        self,
        np.ndarray[DOUBLE_t, ndim=2] observed_data_arr,
        int traning_count=1000
    ):
        '''
        Learning.

        Args:
            observed_data_arr:      The `np.ndarray` of observed data points.
            traning_count:          Training counts.
        '''
        cdef int i
        cdef int row_i = observed_data_arr.shape[0]
        cdef int j
        cdef np.ndarray[DOUBLE_t, ndim=1] data_arr
        cdef np.ndarray[DOUBLE_t, ndim=1] feature_point_arr
        for i in range(row_i):
            data_arr = observed_data_arr[i].copy()
            for j in range(len(self.__rbm_list)):
                rbm = self.__rbm_list[j]
                rbm.approximate_learning(data_arr, traning_count)
                feature_point_arr = self.get_feature_point(j)
                data_arr = feature_point_arr

    def get_feature_point(self, int layer_number=0):
        '''
        Extract the feature points.

        Args:
            layer_number:   The index of layers. 
                            For instance, 0 is visible layer, 1 is hidden or middle layer, and 2 is hidden layer in three layers.

        Returns:
            The list of feature points.
        '''
        feature_point_arr = self.__rbm_list[layer_number].graph.hidden_activity_arr
        return feature_point_arr

    def get_visible_activity_arr_list(self):
        '''
        Extract activity of neurons in each visible layers.

        Returns:
            Activity.
        '''
        visible_activity_arr_list = [self.__rbm_list[i].graph.visible_activity_arr for i in range(len(self.__rbm_list))]
        return visible_activity_arr_list

    def get_hidden_activity_arr_list(self):
        '''
        Extract activity of neurons in each hidden layers.

        Returns:
            Activity.
        '''
        hidden_activity_arr_list = [self.__rbm_list[i].graph.hidden_activity_arr for i in range(len(self.__rbm_list))]
        return hidden_activity_arr_list

    def get_visible_bias_arr_list(self):
        '''
        Extract bias in each visible layers.

        Returns:
            Bias.
        '''
        visible_bias_arr_list = [self.__rbm_list[i].graph.visible_bias_arr for i in range(len(self.__rbm_list))]
        return visible_bias_arr_list

    def get_hidden_bias_arr_list(self):
        '''
        Extract bias in each hidden layers.

        Returns:
            Bias.
        '''
        hidden_bias_arr_list = [self.__rbm_list[i].graph.hidden_bias_arr for i in range(len(self.__rbm_list))]
        return hidden_bias_arr_list

    def get_weight_arr_list(self):
        '''
        Extract weights of each links.

        Returns:
            The list of weights.
        '''
        weight_arr_list = [self.__rbm_list[i].graph.weights_arr for i in range(len(self.__rbm_list))]
        return weight_arr_list

    def get_reconstruct_error_arr(self, int layer_number=0):
        '''
        Extract reconsturction error rate.

        Returns:
            The np.ndarray.
        '''
        return np.array(self.__rbm_list[layer_number].get_reconstruct_error_list())
