# -*- coding: utf-8 -*-
import pyximport
import numpy as np
pyximport.install(setup_args={'include_dirs':[np.get_include()]}, inplace=True)
cimport numpy
from pydbm.approximation.interface.approximate_interface import ApproximateInterface


class ContrastiveDivergence(ApproximateInterface):
    '''
    Contrastive Divergence.
    
    Conceptually, the positive phase is to the negative phase what waking is to sleeping.
    '''

    # Graph of neurons.
    __graph = None
    # Learning rate.
    __learning_rate = 0.5

    def approximate_learn(
        self,
        graph,
        double learning_rate,
        numpy.ndarray observed_data_arr,
        int traning_count=1000
    ):
        '''
        learning with function approximation.

        Args:
            graph:                Graph of neurons.
            learning_rate:        Learning rate.
            observed_data_arr:    observed data points.
            traning_count:        Training counts.

        Returns:
            Graph of neurons.
        '''
        self.__graph = graph
        self.__learning_rate = learning_rate
        cdef int i
        [self.__wake_and_sleep(observed_data_arr[i]) for i in range(observed_data_arr.shape[0])]
        return self.__graph

    def __wake_and_sleep(self, numpy.ndarray observed_data_arr):
        '''
        Waking and sleeping.

        Args:
            observed_data_arr:      observed data points.
        '''
        self.__wake(observed_data_arr)
        self.__sleep()
        self.__learn()

    def __wake(self, numpy.ndarray observed_data_arr):
        '''
        Waking.

        Args:
            observed_data_list:      observed data points.
        '''
        cdef int k
        [self.__graph.visible_neuron_arr[k].observe_data_point(observed_data_arr[k]) for k in range(observed_data_arr.shape[0])]
        self.__update_hidden_spike()
        # so called `Hebb rule`.
        self.__graph.update(self.__learning_rate)
        cdef int i
        [self.__graph.visible_neuron_arr[i].update_bias(self.__learning_rate) for i in range(len(self.__graph.visible_neuron_arr))]
        cdef int j
        [self.__graph.hidden_neuron_arr[j].update_bias(self.__learning_rate) for j in range(len(self.__graph.hidden_neuron_arr))]

    def __sleep(self):
        '''
        Sleeping.
        '''
        self.__update_visible_spike()
        self.__update_hidden_spike()
        # so called `Hebb rule`.
        self.__graph.update((-1) * self.__learning_rate)
        cdef int i
        [self.__graph.visible_neuron_arr[i].update_bias((-1) * self.__learning_rate) for i in range(len(self.__graph.visible_neuron_arr))]
        cdef int j
        [self.__graph.hidden_neuron_arr[j].update_bias((-1) * self.__learning_rate) for j in range(len(self.__graph.hidden_neuron_arr))]

    def __update_visible_spike(self):
        '''
        Update activity of neurons in visible layer.
        '''
        cdef int j
        cdef numpy.ndarray hidden_activity_arr = np.array([[self.__graph.hidden_neuron_arr[j].activity] * self.__graph.weights_arr.T.shape[1] for j in range(len(self.__graph.hidden_neuron_arr))])
        cdef numpy.ndarray link_value_arr = self.__graph.weights_arr.T * hidden_activity_arr
        cdef numpy.ndarray visible_activity_arr = link_value_arr.sum(axis=0)
        cdef int i
        for i in range(visible_activity_arr.shape[0]):
            self.__graph.visible_neuron_arr[i].visible_update_state(visible_activity_arr[i])
            self.__graph.normalize_visible_bias()

    def __update_hidden_spike(self):
        '''
        Update activity of neurons in hidden layer.
        '''
        cdef int i
        cdef numpy.ndarray visible_activity_arr = np.array([[self.__graph.visible_neuron_arr[i].activity] * self.__graph.weights_arr.shape[1] for i in range(len(self.__graph.visible_neuron_arr))])
        cdef numpy.ndarray link_value_arr = self.__graph.weights_arr * visible_activity_arr
        cdef numpy.ndarray hidden_activity_arr = link_value_arr.sum(axis=0)
        cdef int j
        for j in range(hidden_activity_arr.shape[0]):
            self.__graph.hidden_neuron_arr[j].hidden_update_state(hidden_activity_arr[j])
            self.__graph.normalize_hidden_bias()

    def __learn(self):
        '''
        Learning the biases and weights.
        '''
        cdef int i
        [self.__graph.visible_neuron_arr[i].learn_bias() for i in range(len(self.__graph.visible_neuron_arr))]
        cdef int j
        [self.__graph.hidden_neuron_arr[j].learn_bias() for j in range(len(self.__graph.hidden_neuron_arr))]
        self.__graph.learn_weights()

    def recall(self, graph, numpy.ndarray observed_data_arr):
        '''
        Free association.

        Args:
            graph:                  Graph of neurons.
            observed_data_arr:      observed data points.

        Returns:
            Graph of neurons.

        '''
        self.__graph = graph
        cdef int k
        [self.__wake_and_sleep(observed_data_arr[k]) for k in range(observed_data_arr.shape[0])]
        return self.__graph
