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

    __visible_bias = None
    __hidden_bias = None
    __hidden_activity_arr = None

    detail_setting_flag = True

    def approximate_learn(
        self,
        graph,
        double learning_rate,
        np.ndarray observed_data_arr,
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
        cdef int row_i = observed_data_arr.shape[0]

        if self.detail_setting_flag is True:
            for i in range(row_i):
                self.__detail_setting(observed_data_arr[i])
        else:
            for i in range(row_i):
                self.__wake_sleep_learn(observed_data_arr[i])
            row_k = len(self.__graph.visible_neuron_list)
            for k in range(row_k):
                self.__graph.visible_neuron_list[k].observe_data_point(
                    observed_data_arr[-1][k]
                )
                self.__graph.visible_neuron_list[k].bias = self.__visible_bias[k]
            row_j = len(self.__graph.hidden_neuron_list)
            for j in range(row_j):
                self.__graph.hidden_neuron_list[j].activity = self.__hidden_activity_arr[j]
                self.__graph.hidden_neuron_list[j].bias = self.__hidden_bias[j]

        return self.__graph

    def __detail_setting(self, np.ndarray[DOUBLE_t, ndim=1] observed_data_arr):
        '''
        Waking and sleeping.

        Args:
            observed_data_arr:      observed data points.
        '''
        self.__wake(observed_data_arr)
        self.__sleep()
        self.__learn()

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
        cdef np.ndarray[DOUBLE_t, ndim=1] visible_activity_arr = observed_data_arr
        cdef int row_w = self.__graph.weights_arr.shape[0]
        cdef int col_w = self.__graph.weights_arr.shape[1]
        cdef np.ndarray link_value_arr = self.__graph.weights_arr * (np.ones((row_w, col_w)) * visible_activity_arr) + self.__visible_bias
        cdef np.ndarray hidden_activity_arr = link_value_arr.sum(axis=0)
        self.__graph.diff_weights_arr = visible_activity_arr * hidden_activity_arr.T * self.__learning_rate
        visible_diff_bias = self.__learning_rate * visible_activity_arr
        hidden_diff_bias = self.__learning_rate * hidden_activity_arr

        # Sleeping.
        hidden_activity_arr = hidden_activity_arr.reshape(-1, 1)
        cdef np.ndarray _link_value_arr = self.__graph.weights_arr.T * (np.ones((col_w, row_w)) * hidden_activity_arr) + self.__hidden_bias
        cdef np.ndarray _visible_activity_arr = _link_value_arr.sum(axis=0)

        _visible_activity_arr = self.__graph.visible_neuron_list[0].activating_function.activate(
            _visible_activity_arr + visible_diff_bias
        )
        _visible_activity_arr = _visible_activity_arr / _visible_activity_arr.sum()

        cdef np.ndarray __link_value_arr = (self.__graph.weights_arr.T * _visible_activity_arr) + self.__visible_bias
        cdef np.ndarray _hidden_activity_arr = __link_value_arr.sum(axis=0)
        try:
            _hidden_activity_arr = self.__graph.hidden_neuron_list[0].activating_function.activate(
                _hidden_activity_arr + hidden_diff_bias
            )
        except AttributeError:
            _hidden_activity_arr = self.__graph.visible_neuron_list[0].activating_function.activate(
                _hidden_activity_arr + hidden_diff_bias
            )

        _hidden_activity_arr = _hidden_activity_arr / _hidden_activity_arr.sum()
        self.__hidden_activity_arr = _hidden_activity_arr
        self.__graph.diff_weights_arr += _visible_activity_arr * _hidden_activity_arr.T * self.__learning_rate * (-1)

        visible_diff_bias += self.__learning_rate * _visible_activity_arr * (-1)
        hidden_diff_bias += self.__learning_rate * _hidden_activity_arr * (-1)

        # Learning.
        if self.__visible_bias is None:
            self.__visible_bias = visible_diff_bias
        else:
            self.__visible_bias += visible_diff_bias
        if self.__hidden_bias is None:
            self.__hidden_bias = hidden_diff_bias
        else:
            self.__hidden_bias += hidden_diff_bias
        self.__graph.learn_weights()

    def __wake(self, np.ndarray[DOUBLE_t, ndim=1] observed_data_arr):
        '''
        Waking.

        Args:
            observed_data_list:      observed data points.
        '''
        cdef int k
        cdef int row_k = len(self.__graph.visible_neuron_list)
        [self.__graph.visible_neuron_list[k].observe_data_point(observed_data_arr[k]) for k in range(row_k)]
        self.__update_hidden_spike()
        # so called `Hebb rule`.
        self.__graph.update(self.__learning_rate)
        cdef int i
        cdef int row_i = len(self.__graph.visible_neuron_list)
        [self.__graph.visible_neuron_list[i].update_bias(self.__learning_rate) for i in range(row_i)]
        cdef int j
        cdef int row_j = len(self.__graph.hidden_neuron_list)
        [self.__graph.hidden_neuron_list[j].update_bias(self.__learning_rate) for j in range(row_j)]

    def __sleep(self):
        '''
        Sleeping.
        '''
        self.__update_visible_spike()
        self.__update_hidden_spike()
        # so called `Hebb rule`.
        self.__graph.update((-1) * self.__learning_rate)
        cdef int i
        cdef int row_i = len(self.__graph.visible_neuron_list)
        [self.__graph.visible_neuron_list[i].update_bias((-1) * self.__learning_rate) for i in range(row_i)]
        cdef int j
        cdef int row_j = len(self.__graph.hidden_neuron_list)
        [self.__graph.hidden_neuron_list[j].update_bias((-1) * self.__learning_rate) for j in range(row_j)]

    def __update_visible_spike(self):
        '''
        Update activity of neurons in visible layer.
        '''
        cdef int j
        cdef int row_j = len(self.__graph.hidden_neuron_list)
        activity_matrix = [None] * row_j
        cdef int T_col = self.__graph.weights_arr.T.shape[1]
        for j in range(row_j):
            activity_matrix[j] = [self.__graph.hidden_neuron_list[j].activity] * T_col

        cdef np.ndarray hidden_activity_arr = np.array(activity_matrix)
        cdef np.ndarray link_value_arr = self.__graph.weights_arr.T * hidden_activity_arr
        cdef np.ndarray visible_activity_arr = link_value_arr.sum(axis=0)
        cdef int i
        for i in range(visible_activity_arr.shape[0]):
            self.__graph.visible_neuron_list[i].visible_update_state(visible_activity_arr[i])
        self.__graph.normalize_visible_bias()

    def __update_hidden_spike(self):
        '''
        Update activity of neurons in hidden layer.
        '''
        cdef int i
        cdef int row_i = len(self.__graph.visible_neuron_list)
        activity_matrix = [None] * row_i
        cdef int col_i = self.__graph.weights_arr.shape[1]
        for i in range(row_i):
            activity_matrix[i] = [self.__graph.visible_neuron_list[i].activity] * col_i

        cdef np.ndarray visible_activity_arr = np.array(activity_matrix)
        cdef np.ndarray link_value_arr = self.__graph.weights_arr * visible_activity_arr
        cdef np.ndarray hidden_activity_arr = link_value_arr.sum(axis=0)
        cdef int j
        for j in range(hidden_activity_arr.shape[0]):
            self.__graph.hidden_neuron_list[j].hidden_update_state(hidden_activity_arr[j])
        self.__graph.normalize_hidden_bias()

    def __learn(self):
        '''
        Learning the biases and weights.
        '''
        cdef int i
        cdef int row_i = len(self.__graph.visible_neuron_list)
        for i in range(row_i):
            self.__graph.visible_neuron_list[i].learn_bias()
        cdef int j
        cdef int row_j = len(self.__graph.hidden_neuron_list)
        for j in range(row_j):
            self.__graph.hidden_neuron_list[j].learn_bias()
        self.__graph.learn_weights()

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
