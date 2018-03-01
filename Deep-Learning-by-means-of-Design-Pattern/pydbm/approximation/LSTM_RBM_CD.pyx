# -*- coding: utf-8 -*-
import numpy as np
cimport numpy as np
cimport cython
from pydbm.approximation.interface.approximate_interface import ApproximateInterface
ctypedef np.float64_t DOUBLE_t


class LSTMRBMCD(ApproximateInterface):
    '''
    Contrastive Divergence.
    
    Conceptually, the positive phase is to the negative phase what waking is to sleeping.
    '''

    # The list of the reconstruction error rate (MSE)
    __reconstruct_error_list = []

    def set_readonly(self, value):
        ''' setter '''
        raise TypeError("This is read-only.")

    def get_reconstrct_error_list(self):
        ''' getter '''
        return self.__reconstruct_error_list

    reconstruct_error_list = property(get_reconstrct_error_list, set_readonly)

    # Graph of neurons.
    __graph = None
    # Learning rate.
    __learning_rate = 0.5
    # Dropout rate.
    __dropout_rate = 0.5
    # Batch size in learning.
    __batch_size = 0
    # Batch step in learning.
    __batch_step = 0
    # Batch size in inference(recursive learning or not).
    __r_batch_size = 0
    # Batch step in inference(recursive learning or not).
    __r_batch_step = 0
    # weight matrix in `t-1`.
    __pre_weight_arr = None

    def __init__(self):
        '''
        Initialize.
        '''
        raise NotImplementedError()

    def approximate_learn(
        self,
        graph,
        double learning_rate,
        double dropout_rate,
        np.ndarray observed_data_arr,
        int traning_count=1000,
        int batch_size=200
    ):
        '''
        learning with function approximation.

        Args:
            graph:                Graph of neurons.
            learning_rate:        Learning rate.
            dropout_rate:         Dropout rate.
            observed_data_arr:    observed data points.
            traning_count:        Training counts.
            batch_size:           Batch size (0: not mini-batch)

        Returns:
            Graph of neurons.
        '''
        self.__graph = graph
        self.__learning_rate = learning_rate
        self.__dropout_rate = dropout_rate
        self.__batch_size = batch_size

        cdef int _
        for _ in range(traning_count):
            self.__batch_step += 1
            self.__wake_sleep_learn(observed_data_arr)

        return self.__graph

    def approximate_inference(
        self,
        graph,
        double learning_rate,
        double dropout_rate,
        np.ndarray observed_data_arr,
        int traning_count=1000,
        int r_batch_size=200
    ):
        '''
        Inference with function approximation.

        Args:
            graph:                Graph of neurons.
            learning_rate:        Learning rate.
            dropout_rate:         Dropout rate.
            observed_data_arr:    observed data points.
            traning_count:        Training counts.
            r_batch_size:         Batch size.
                                  If this value is `0`, the inferencing is a recursive learning.
                                  If this value is more than `0`, the inferencing is a mini-batch recursive learning.
                                  If this value is '-1', the inferencing is not a recursive learning.

        Returns:
            Graph of neurons.
        '''
        self.__graph = graph
        self.__learning_rate = learning_rate
        self.__dropout_rate = dropout_rate
        self.__r_batch_size = r_batch_size

        cdef int _
        for _ in range(traning_count):
            self.__r_batch_step += 1
            self.__sleep_wake_learn(observed_data_arr)

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
        self.__graph.visible_activity_arr = observed_data_arr.copy()

        cdef np.ndarray[DOUBLE_t, ndim=2] link_value_arr = (self.__graph.weights_arr * self.__graph.visible_activity_arr.reshape(-1, 1)) + self.__graph.visible_bias_arr.reshape(-1, 1)
        link_value_arr = np.nan_to_num(link_value_arr)
        self.__graph.hidden_activity_arr = link_value_arr.sum(axis=0)
        self.__graph.hidden_activity_arr = self.__graph.hidden_activating_function.activate(
            self.__graph.hidden_activity_arr
        )
        if self.__dropout_rate > 0:
            self.__graph.hidden_activity_arr = self.__dropout(self.__graph.hidden_activity_arr)

        self.__graph.diff_weights_arr += self.__graph.visible_activity_arr.reshape(-1, 1) * self.__graph.hidden_activity_arr.reshape(-1, 1).T * self.__learning_rate

        self.__graph.visible_diff_bias_arr += self.__learning_rate * self.__graph.visible_activity_arr
        self.__graph.hidden_diff_bias_arr += self.__learning_rate * self.__graph.hidden_activity_arr

        # Sleeping.
        self.__graph.pre_hidden_activity_arr = self.__graph.rnn_activating_function_interface.activate(
            (self.__graph.weights_arr * self.__graph.visible_activity_arr.reshape(-1, 1)) + (self.__graph.hidden_bias_weights_arr.T * self.__graph.pre_hidden_activity_arr.reshape(-1, 1).T) + self.__graph.hidden_activity_arr.reshape(-1, 1).T
        )
        self.__graph.pre_hidden_activity_arr = np.nan_to_num(self.__graph.pre_hidden_activity_arr)

        t_hidden_bias_arr = self.__graph.hidden_bias_arr.reshape(-1, 1) + self.__graph.hidden_bias_weights_arr.T * self.__graph.pre_hidden_activity_arr.reshape(-1, 1)
        t_visible_bias_arr = self.__graph.visible_bias_arr.reshape(-1, 1) + self.__graph.visible_bias_weights_arr * self.__graph.pre_hidden_activity_arr.reshape(-1, 1).T

        link_value_arr = self.__graph.visible_activity_arr.reshape(-1, 1).T * t_visible_bias_arr + self.__graph.visible_activity_arr.reshape(-1, 1).T * self.__graph.weights_arr.T * self.__graph.hidden_activity_arr.reshape(-1, 1)
        link_value_arr += (self.__graph.hidden_activity_arr.T * t_hidden_bias_arr) + (self.__graph.hidden_bias_weights_arr * self.__graph.pre_hidden_activity_arr.reshape(-1, 1))
        link_value_arr = np.nan_to_num(link_value_arr)
        link_value_arr = np.exp(link_value_arr)
        link_value_arr = link_value_arr / self.__graph.pre_hidden_activity_arr.sum()        
        self.__graph.visible_activity_arr = link_value_arr.sum(axis=1)
        '''
        link_value_arr = (self.__graph.weights_arr.T) * self.__graph.hidden_activity_arr.reshape(-1, 1) + self.__graph.hidden_bias_arr.reshape(-1, 1)
        link_value_arr = np.nan_to_num(link_value_arr)
        self.__graph.visible_activity_arr = link_value_arr.sum(axis=0)
        '''

        # Validation.
        self.compute_reconstruct_error(observed_data_arr, self.__graph.visible_activity_arr)

        if self.__dropout_rate > 0:
            self.__graph.visible_activity_arr = self.__dropout(self.__graph.visible_activity_arr)

        link_value_arr = (self.__graph.weights_arr * self.__graph.visible_activity_arr.reshape(-1, 1)) + self.__graph.visible_bias_arr.reshape(-1, 1)
        link_value_arr = np.nan_to_num(link_value_arr)
        self.__graph.hidden_activity_arr = link_value_arr.sum(axis=0)
        self.__graph.hidden_activity_arr = self.__graph.hidden_activating_function.activate(self.__graph.hidden_activity_arr)
        if self.__dropout_rate > 0:
            self.__graph.hidden_activity_arr = self.__dropout(self.__graph.hidden_activity_arr)

        self.__graph.diff_weights_arr += self.__graph.visible_activity_arr.reshape(-1, 1) * self.__graph.hidden_activity_arr.reshape(-1, 1).T * self.__learning_rate * (-1)

        self.__graph.visible_diff_bias_arr += self.__learning_rate * self.__graph.visible_activity_arr * (-1)
        self.__graph.hidden_diff_bias_arr += self.__learning_rate * self.__graph.hidden_activity_arr * (-1)

        # Learning.
        if self.__batch_size == 0 or self.__batch_step % self.__batch_size == 0:
            self.__graph.visible_bias_arr += self.__graph.visible_diff_bias_arr
            self.__graph.hidden_bias_arr += self.__graph.hidden_diff_bias_arr
            self.__graph.visible_diff_bias_arr = np.zeros(self.__graph.visible_bias_arr.shape)
            self.__graph.hidden_diff_bias_arr = np.zeros(self.__graph.hidden_bias_arr.shape)
            self.__graph.learn_weights()

        # Memorize.
        self.__graph.pre_hidden_activity_arr = self.__graph.hidden_activity_arr.copy()
        self.__pre_weight_arr = self.__graph.weights_arr.copy()

    def __sleep_wake_learn(self, np.ndarray[DOUBLE_t, ndim=1] observed_data_arr):
        '''
        Sleeping, waking, and learning.

        Args:
             observed_data_arr:      feature points.
        '''
        # Sleeping.
        self.__graph.hidden_activity_arr = observed_data_arr.copy()

        cdef np.ndarray[DOUBLE_t, ndim=2] link_value_arr = (self.__graph.weights_arr.T) * self.__graph.hidden_activity_arr.reshape(-1, 1) + self.__graph.hidden_bias_arr.reshape(-1, 1)
        link_value_arr = np.nan_to_num(link_value_arr)
        self.__graph.visible_activity_arr = link_value_arr.sum(axis=0)
        self.__graph.visible_activity_arr = self.__graph.visible_activating_function.activate(self.__graph.visible_activity_arr)

        link_value_arr = (self.__graph.weights_arr * self.__graph.visible_activity_arr.reshape(-1, 1)) + self.__graph.visible_bias_arr.reshape(-1, 1)
        link_value_arr = np.nan_to_num(link_value_arr)
        self.__graph.hidden_activity_arr = link_value_arr.sum(axis=0)
        self.__graph.hidden_activity_arr = self.__graph.hidden_activating_function.activate(self.__graph.hidden_activity_arr)

        if self.__r_batch_size != -1:
            self.__graph.diff_weights_arr += self.__graph.visible_activity_arr.reshape(-1, 1) * self.__graph.hidden_activity_arr.reshape(-1, 1).T * self.__learning_rate * (-1)
            self.__graph.visible_diff_bias_arr += self.__learning_rate * self.__graph.visible_activity_arr * (-1)
            self.__graph.hidden_diff_bias_arr += self.__learning_rate * self.__graph.hidden_activity_arr * (-1)

        # Waking.
        link_value_arr = (self.__graph.weights_arr * self.__graph.visible_activity_arr.reshape(-1, 1)) + self.__graph.visible_bias_arr.reshape(-1, 1)
        link_value_arr = np.nan_to_num(link_value_arr)
        self.__graph.hidden_activity_arr = link_value_arr.sum(axis=0)
        self.__graph.hidden_activity_arr = self.__graph.hidden_activating_function.activate(
            self.__graph.hidden_activity_arr
        )

        if self.__r_batch_size != -1:
            self.__graph.diff_weights_arr += self.__graph.visible_activity_arr.reshape(-1, 1) * self.__graph.hidden_activity_arr.reshape(-1, 1).T * self.__learning_rate
            self.__graph.visible_diff_bias_arr += self.__learning_rate * self.__graph.visible_activity_arr
            self.__graph.hidden_diff_bias_arr += self.__learning_rate * self.__graph.hidden_activity_arr

        # Learning.
        if self.__r_batch_size != -1:
            if self.__r_batch_size == 0 or self.__r_batch_step % self.__r_batch_size == 0:
                self.__graph.visible_bias_arr += self.__graph.visible_diff_bias_arr
                self.__graph.hidden_bias_arr += self.__graph.hidden_diff_bias_arr
                self.__graph.visible_diff_bias_arr = np.zeros(self.__graph.visible_bias_arr.shape)
                self.__graph.hidden_diff_bias_arr = np.zeros(self.__graph.hidden_bias_arr.shape)
                self.__graph.learn_weights()

    def __dropout(self, np.ndarray[DOUBLE_t, ndim=1] activity_arr):
        '''
        Dropout.
        '''
        cdef int row = activity_arr.shape[0]
        cdef np.ndarray[DOUBLE_t, ndim=1] dropout_rate_arr = np.random.uniform(0, 1, size=(row, ))
        activity_arr = activity_arr * dropout_rate_arr.T
        return activity_arr

    def compute_reconstruct_error(
        self,
        np.ndarray[DOUBLE_t, ndim=1] observed_data_arr, 
        np.ndarray[DOUBLE_t, ndim=1] reconstructed_arr
    ):
        '''
        Compute reconstruction error rate.
        '''
        cdef double reconstruct_error = np.sum((reconstructed_arr - observed_data_arr) ** 2) / observed_data_arr.shape[0]
        self.__reconstruct_error_list.append(reconstruct_error)
