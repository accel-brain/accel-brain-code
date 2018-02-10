# -*- coding: utf-8 -*-
import numpy as np
cimport numpy as np
cimport cython
from pydbm.approximation.interface.approximate_interface import ApproximateInterface
ctypedef np.float64_t DOUBLE_t


class ContrastiveDivergence(ApproximateInterface):
    '''
    Contrastive Divergence.
    
    Conceptually, the positive phase is to the negative phase what waking is to sleeping.
    '''

    # Graph of neurons.
    __graph = None
    # Learning rate.
    __learning_rate = 0.5
    # Dropout rate.
    __dropout_rate = 0.5
    # Particle normalized flag.
    __particle_normalize_flag = False

    __visible_bias = None
    __hidden_bias = None
    __hidden_activity_arr = None

    def approximate_learn(
        self,
        graph,
        double learning_rate,
        double dropout_rate,
        np.ndarray observed_data_arr,
        int traning_count=1000
    ):
        '''
        learning with function approximation.

        Args:
            graph:                Graph of neurons.
            learning_rate:        Learning rate.
            dropout_rate:         Dropout rate.
            observed_data_arr:    observed data points.
            traning_count:        Training counts.

        Returns:
            Graph of neurons.
        '''
        self.__graph = graph
        self.__learning_rate = learning_rate
        self.__dropout_rate = dropout_rate

        cdef int i
        cdef int _
        cdef int row_i = observed_data_arr.shape[0]

        for _ in range(traning_count):
            self.__wake_sleep_learn(observed_data_arr)

        return self.__graph

    def __wake_sleep_learn(self, np.ndarray[DOUBLE_t, ndim=1] observed_data_arr):
        '''
        Waking, sleeping, and learning.

        Standing on the premise that the settings of
        the activation function and weights operation are common.

        The binary activity is unsupported.

        Args:
            observed_data_list:      observed data points.
        '''
        # Waking.
        self.__graph.visible_activity_arr = observed_data_arr
        cdef int row_w = self.__graph.weights_arr.shape[0]
        cdef int col_w = self.__graph.weights_arr.shape[1]

        cdef np.ndarray[DOUBLE_t, ndim=2] link_value_arr = (self.__graph.weights_arr * self.__graph.visible_activity_arr.reshape(-1, 1)) + self.__graph.visible_bias_arr.reshape(-1, 1)
        link_value_arr = np.nan_to_num(link_value_arr)
        self.__graph.hidden_activity_arr = link_value_arr.sum(axis=0)

        cdef double hidden_activity_sum
        if self.__particle_normalize_flag is True:
            hidden_activity_sum = self.__graph.hidden_activity_arr.sum()
            if hidden_activity_sum != 0:
                self.__graph.hidden_activity_arr = self.__graph.hidden_activity_arr / hidden_activity_sum

        self.__graph.hidden_activity_arr = self.__graph.hidden_activating_function.activate(
            self.__graph.hidden_activity_arr
        )
        if self.__dropout_rate > 0:
            self.__graph.hidden_activity_arr = self.__dropout(self.__graph.hidden_activity_arr)

        self.__graph.diff_weights_arr = self.__graph.visible_activity_arr.reshape(-1, 1) * self.__graph.hidden_activity_arr.reshape(-1, 1).T * self.__learning_rate

        visible_diff_bias = self.__learning_rate * self.__graph.visible_activity_arr
        hidden_diff_bias = self.__learning_rate * self.__graph.hidden_activity_arr

        # Sleeping.
        link_value_arr = (self.__graph.weights_arr.T) * self.__graph.hidden_activity_arr.reshape(-1, 1) + self.__graph.hidden_bias_arr.reshape(-1, 1)
        link_value_arr = np.nan_to_num(link_value_arr)
        self.__graph.visible_activity_arr = link_value_arr.sum(axis=0)

        cdef double visible_activity_sum
        if self.__particle_normalize_flag is True:
            visible_activity_sum = self.__graph.visible_activity_arr.sum()
            if visible_activity_sum != 0:
                self.__graph.visible_activity_arr = self.__graph.visible_activity_arr / visible_activity_sum

        self.__graph.visible_activity_arr = self.__graph.visible_activating_function.activate(self.__graph.visible_activity_arr)

        if self.__dropout_rate > 0:
            self.__graph.visible_activity_arr = self.__dropout(self.__graph.visible_activity_arr)

        link_value_arr = (self.__graph.weights_arr * self.__graph.visible_activity_arr.reshape(-1, 1)) + self.__graph.visible_bias_arr.reshape(-1, 1)
        link_value_arr = np.nan_to_num(link_value_arr)
        self.__graph.hidden_activity_arr = link_value_arr.sum(axis=0)

        if self.__particle_normalize_flag is True:
            hidden_activity_sum = self.__graph.hidden_activity_arr.sum()
            if hidden_activity_sum != 0:
                self.__graph.hidden_activity_arr = self.__graph.hidden_activity_arr / hidden_activity_sum

        self.__graph.hidden_activity_arr = self.__graph.hidden_activating_function.activate(self.__graph.hidden_activity_arr)

        if self.__dropout_rate > 0:
            self.__graph.hidden_activity_arr = self.__dropout(self.__graph.hidden_activity_arr)

        self.__graph.diff_weights_arr += self.__graph.visible_activity_arr.reshape(-1, 1) * self.__graph.hidden_activity_arr.reshape(-1, 1).T * self.__learning_rate * (-1)

        visible_diff_bias += self.__learning_rate * self.__graph.visible_activity_arr * (-1)
        hidden_diff_bias += self.__learning_rate * self.__graph.hidden_activity_arr * (-1)

        cdef double visible_diff_bias_sum
        if self.__particle_normalize_flag is True:
            visible_diff_bias_sum = self.__graph.visible_activity_arr.sum()
            if visible_diff_bias_sum != 0:
                self.__graph.visible_activity_arr = self.__graph.visible_activity_arr / visible_diff_bias_sum

        cdef double hidden_diff_bias_sum
        if self.__particle_normalize_flag is True:
            hidden_diff_bias_sum = self.__graph.hidden_activity_arr.sum()
            if hidden_diff_bias_sum != 0:
                self.__graph.hidden_activity_arr = self.__graph.hidden_activity_arr / hidden_diff_bias_sum

        # Learning.
        self.__graph.visible_bias_arr += visible_diff_bias
        self.__graph.hidden_bias_arr += hidden_diff_bias
        self.__graph.learn_weights()

    def __dropout(self, np.ndarray[DOUBLE_t, ndim=1] activity_arr):
        '''
        Dropout.
        '''
        cdef int row = activity_arr.shape[0]
        cdef int col = activity_arr.shape[1]
        cdef np.ndarray[DOUBLE_t, ndim=1] dropout_rate_arr = np.random.uniform(0, 1, shape=(row, col))
        activity_arr = activity_arr * dropout_rate_arr.T
        return activity_arr

    def recall(self, graph, np.ndarray observed_data_arr):
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
