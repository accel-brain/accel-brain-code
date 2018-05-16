# -*- coding: utf-8 -*-
import numpy as np
cimport numpy as np
cimport cython
from pydbm.approximation.interface.approximate_interface import ApproximateInterface
ctypedef np.float64_t DOUBLE_t


class RTRBMCD(ApproximateInterface):
    '''
    Recurrent Temporal Restricted Boltzmann Machines
    based on Contrastive Divergence.

    Conceptually, the positive phase is to the negative phase what waking is to sleeping.

    Parameters:
        graph.weights_arr:                $W$ (Connection between v^{(t)} and h^{(t)})
        graph.visible_bias_arr:           $b_v$ (Bias in visible layer)
        graph.hidden_bias_arr:            $b_h$ (Bias in hidden layer)
        graph.rnn_hidden_weights_arr:     $W'$ (Connection between h^{(t-1)} and b_h^{(t)})
        graph.rnn_visible_weights_arr:    $W''$ (Connection between h^{(t-1)} and b_v^{(t)})
        graph.hat_hidden_activity_arr:    $\hat{h}^{(t)}$ (RNN with hidden units)
        graph.pre_hidden_activity_arr:    $\hat{h}^{(t-1)}$
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
    graph = None
    # Learning rate.
    learning_rate = 0.5
    # Dropout rate.
    dropout_rate = 0.5
    # Batch size in learning.
    batch_size = 0
    # Batch step in learning.
    batch_step = 0
    # Batch size in inference(recursive learning or not).
    r_batch_size = 0
    # Batch step in inference(recursive learning or not).
    r_batch_step = 0
    # visible activity in negative phase.
    negative_visible_activity_arr = None
    # gradient in visible layer.
    b_v_c_diff_arr = np.array([])
    # gradient in hidden layer.
    b_h_c_diff_arr = np.array([])

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
        self.graph = graph
        self.learning_rate = learning_rate
        self.dropout_rate = dropout_rate
        self.batch_size = batch_size

        # RNN learning.
        self.rnn_learn(observed_data_arr)

        cdef int _
        for _ in range(traning_count):
            self.batch_step += 1
            self.wake_sleep_learn(self.graph.visible_activity_arr)

        # Memorizing.
        self.memorize_activity(
            observed_data_arr,
            self.graph.visible_activity_arr
        )

        # Back propagation.
        if self.batch_size == 0 or self.batch_step % self.batch_size == 0:
            self.back_propagation()

        return self.graph

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
        self.graph = graph
        self.learning_rate = learning_rate
        self.dropout_rate = dropout_rate
        self.r_batch_size = r_batch_size

        # RNN learning.
        self.rnn_learn(observed_data_arr)

        cdef int _
        for _ in range(traning_count):
            self.r_batch_step += 1
            self.wake_sleep_inference(observed_data_arr)

        # Memorizing.
        if self.r_batch_size != -1:
            self.memorize_activity(
                observed_data_arr,
                self.graph.visible_activity_arr
            )

            # Back propagation.
            if self.r_batch_size == 0 or self.r_batch_step % self.r_batch_size == 0:
                self.back_propagation()

        return self.graph

    def rnn_learn(self, np.ndarray[DOUBLE_t, ndim=1] observed_data_arr):
        '''
        Learning for RNN.

        Args:
            observed_data_list:      observed data points.
        '''
        self.graph.visible_activity_arr = observed_data_arr.copy()

        if self.graph.pre_hidden_activity_arr.shape[0] == 0:
            return
        if self.graph.hat_hidden_activity_arr.shape[0] == 0:
            return

        cdef np.ndarray[DOUBLE_t, ndim=2] link_value_arr = (self.graph.rnn_hidden_weights_arr * self.graph.hat_hidden_activity_arr.reshape(-1, 1)) + self.graph.hidden_bias_arr.reshape(-1, 1) * self.learning_rate
        link_value_arr = np.nan_to_num(link_value_arr)
        self.graph.rnn_hidden_bias_arr = link_value_arr.sum(axis=0)

        link_value_arr = (self.graph.rnn_visible_weights_arr.T * self.graph.hat_hidden_activity_arr.reshape(-1, 1)) + self.graph.visible_bias_arr.reshape(-1, 1).T * self.learning_rate
        link_value_arr = np.nan_to_num(link_value_arr)
        self.graph.rnn_visible_bias_arr = link_value_arr.sum(axis=0)

        link_value_arr = (self.graph.rnn_hidden_weights_arr * self.graph.pre_hidden_activity_arr.reshape(-1, 1)) + self.graph.hidden_bias_arr.reshape(-1, 1) * self.learning_rate
        link_value_arr = np.nan_to_num(link_value_arr)
        self.graph.hidden_activity_arr = link_value_arr.sum(axis=0)

        link_value_arr = (self.graph.rnn_visible_weights_arr * self.graph.pre_hidden_activity_arr.reshape(-1, 1).T) + self.graph.visible_bias_arr.reshape(-1, 1) * self.learning_rate
        link_value_arr = np.nan_to_num(link_value_arr)
        self.graph.visible_activity_arr = link_value_arr.sum(axis=1)

    def memorize_activity(
        self,
        np.ndarray[DOUBLE_t, ndim=1] observed_data_arr,
        np.ndarray[DOUBLE_t, ndim=1] negative_visible_activity_arr
    ):
        '''
        Memorize activity.

        Args:
            observed_data_arr:                Observed data points in positive phase.
            negative_visible_activity_arr:    visible acitivty in negative phase.
        '''
        self.graph.pre_hidden_activity_arr = self.graph.hat_hidden_activity_arr

        cdef np.ndarray[DOUBLE_t, ndim=2] link_value_arr = (self.graph.weights_arr * self.graph.visible_activity_arr.reshape(-1, 1)) + self.graph.hidden_bias_arr.reshape(-1, 1).T
        link_value_arr = np.nan_to_num(link_value_arr)
        self.graph.hat_hidden_activity_arr = link_value_arr.sum(axis=0)
        self.graph.hat_hidden_activity_arr = self.graph.rnn_activating_function.activate(
            self.graph.hat_hidden_activity_arr
        )

        cdef np.ndarray[DOUBLE_t, ndim=2] b_h_c_diff_arr = self.graph.rnn_activating_function.activate(
            (self.graph.weights_arr * (negative_visible_activity_arr.reshape(-1, 1)) - self.graph.hidden_bias_arr.reshape(-1, 1).T
        ) - self.graph.rnn_activating_function.activate(
            (self.graph.weights_arr * observed_data_arr.reshape(-1, 1))) - self.graph.hidden_bias_arr.reshape(-1, 1).T
        )
        cdef np.ndarray[DOUBLE_t, ndim=2] b_v_c_diff_arr = self.graph.rnn_activating_function.activate(
            (self.graph.weights_arr * (negative_visible_activity_arr.reshape(-1, 1)) - self.graph.visible_bias_arr.reshape(-1, 1)
        ) - self.graph.rnn_activating_function.activate(
            (self.graph.weights_arr * observed_data_arr.reshape(-1, 1))) - self.graph.visible_bias_arr.reshape(-1, 1)
        )
        b_h_c_diff_arr = b_h_c_diff_arr.T
        b_v_c_diff_arr = b_v_c_diff_arr.T
        if self.b_h_c_diff_arr.shape[0]:
            self.b_h_c_diff_arr += b_h_c_diff_arr
        else:
            self.b_h_c_diff_arr = b_h_c_diff_arr
        if self.b_v_c_diff_arr.shape[0]:
            self.b_v_c_diff_arr += b_v_c_diff_arr
        else:
            self.b_v_c_diff_arr = b_v_c_diff_arr

    def back_propagation(self):
        '''
        Details of the backpropagation through time algorithm.
        '''
        self.graph.rnn_hidden_weights_arr = self.b_h_c_diff_arr * self.graph.hat_hidden_activity_arr.reshape(-1, 1) * self.learning_rate
        self.graph.rnn_visible_weights_arr = (self.b_v_c_diff_arr * self.graph.hat_hidden_activity_arr.reshape(-1, 1)).T * self.learning_rate
        self.b_v_c_diff_arr = np.array([])
        self.b_h_c_diff_arr = np.array([])

    def wake_sleep_learn(self, np.ndarray[DOUBLE_t, ndim=1] observed_data_arr):
        '''
        Waking, sleeping, and learning.

        Standing on the premise that the settings of
        the activation function and weights operation are common.

        The binary activity is unsupported.

        Args:
            observed_data_list:      observed data points.
        '''
        self.graph.visible_activity_arr = observed_data_arr

        # Waking.
        link_value_arr = (self.graph.weights_arr * self.graph.visible_activity_arr.reshape(-1, 1)) + self.graph.visible_bias_arr.reshape(-1, 1)
        link_value_arr = np.nan_to_num(link_value_arr)
        self.graph.hidden_activity_arr = link_value_arr.sum(axis=0)
        self.graph.hidden_activity_arr = self.graph.hidden_activating_function.activate(
            self.graph.hidden_activity_arr
        )
        if self.dropout_rate > 0:
            self.graph.hidden_activity_arr = self.dropout(self.graph.hidden_activity_arr)

        self.graph.diff_weights_arr += self.graph.visible_activity_arr.reshape(-1, 1) * self.graph.hidden_activity_arr.reshape(-1, 1).T * self.learning_rate

        self.graph.visible_diff_bias_arr += self.learning_rate * self.graph.visible_activity_arr
        self.graph.hidden_diff_bias_arr += self.learning_rate * self.graph.hidden_activity_arr

        # Sleeping.
        link_value_arr = (self.graph.weights_arr.T) * self.graph.hidden_activity_arr.reshape(-1, 1) + self.graph.hidden_bias_arr.reshape(-1, 1)
        link_value_arr = np.nan_to_num(link_value_arr)

        self.graph.visible_activity_arr = link_value_arr.sum(axis=0)
        self.graph.visible_activity_arr = self.graph.visible_activating_function.activate(self.graph.visible_activity_arr)

        # Validation.
        self.compute_reconstruct_error(observed_data_arr, self.graph.visible_activity_arr)

        if self.dropout_rate > 0:
            self.graph.visible_activity_arr = self.dropout(self.graph.visible_activity_arr)

        link_value_arr = (self.graph.weights_arr * self.graph.visible_activity_arr.reshape(-1, 1)) + self.graph.visible_bias_arr.reshape(-1, 1)
        link_value_arr = np.nan_to_num(link_value_arr)
        self.graph.hidden_activity_arr = link_value_arr.sum(axis=0)
        self.graph.hidden_activity_arr = self.graph.hidden_activating_function.activate(self.graph.hidden_activity_arr)
        if self.dropout_rate > 0:
            self.graph.hidden_activity_arr = self.dropout(self.graph.hidden_activity_arr)

        self.graph.diff_weights_arr += self.graph.visible_activity_arr.reshape(-1, 1) * self.graph.hidden_activity_arr.reshape(-1, 1).T * self.learning_rate * (-1)

        self.graph.visible_diff_bias_arr += self.learning_rate * self.graph.visible_activity_arr * (-1)
        self.graph.hidden_diff_bias_arr += self.learning_rate * self.graph.hidden_activity_arr * (-1)

        # Learning.
        if self.batch_size == 0 or self.batch_step % self.batch_size == 0:
            self.graph.visible_bias_arr += self.graph.visible_diff_bias_arr
            self.graph.hidden_bias_arr += self.graph.hidden_diff_bias_arr
            self.graph.visible_diff_bias_arr = np.zeros(self.graph.visible_bias_arr.shape)
            self.graph.hidden_diff_bias_arr = np.zeros(self.graph.hidden_bias_arr.shape)
            self.graph.learn_weights()

    def wake_sleep_inference(self, np.ndarray[DOUBLE_t, ndim=1] observed_data_arr):
        '''
        Sleeping, waking, and inferencing.

        Args:
             observed_data_arr:      feature points.
        '''
        self.graph.visible_activity_arr = observed_data_arr

        # Waking.
        link_value_arr = (self.graph.weights_arr * self.graph.visible_activity_arr.reshape(-1, 1)) + self.graph.visible_bias_arr.reshape(-1, 1)
        link_value_arr = np.nan_to_num(link_value_arr)
        self.graph.hidden_activity_arr = link_value_arr.sum(axis=0)
        self.graph.hidden_activity_arr = self.graph.hidden_activating_function.activate(
            self.graph.hidden_activity_arr
        )
        if self.r_batch_size != -1:
            if self.dropout_rate > 0:
                self.graph.hidden_activity_arr = self.dropout(self.graph.hidden_activity_arr)

        if self.r_batch_size != -1:
            self.graph.diff_weights_arr += self.graph.visible_activity_arr.reshape(-1, 1) * self.graph.hidden_activity_arr.reshape(-1, 1).T * self.learning_rate
            self.graph.visible_diff_bias_arr += self.learning_rate * self.graph.visible_activity_arr
            self.graph.hidden_diff_bias_arr += self.learning_rate * self.graph.hidden_activity_arr

        # Sleeping.
        link_value_arr = (self.graph.weights_arr.T) * self.graph.hidden_activity_arr.reshape(-1, 1) + self.graph.hidden_bias_arr.reshape(-1, 1)
        link_value_arr = np.nan_to_num(link_value_arr)

        self.graph.visible_activity_arr = link_value_arr.sum(axis=0)
        self.graph.visible_activity_arr = self.graph.visible_activating_function.activate(self.graph.visible_activity_arr)

        # Validation.
        self.compute_reconstruct_error(observed_data_arr, self.graph.visible_activity_arr)

        if self.r_batch_size != -1:
            if self.dropout_rate > 0:
                self.graph.visible_activity_arr = self.dropout(self.graph.visible_activity_arr)

            link_value_arr = (self.graph.weights_arr * self.graph.visible_activity_arr.reshape(-1, 1)) + self.graph.visible_bias_arr.reshape(-1, 1)
            link_value_arr = np.nan_to_num(link_value_arr)
            self.graph.hidden_activity_arr = link_value_arr.sum(axis=0)
            self.graph.hidden_activity_arr = self.graph.hidden_activating_function.activate(self.graph.hidden_activity_arr)

            if self.dropout_rate > 0:
                self.graph.hidden_activity_arr = self.dropout(self.graph.hidden_activity_arr)

            self.graph.diff_weights_arr += self.graph.visible_activity_arr.reshape(-1, 1) * self.graph.hidden_activity_arr.reshape(-1, 1).T * self.learning_rate * (-1)
            self.graph.visible_diff_bias_arr += self.learning_rate * self.graph.visible_activity_arr * (-1)
            self.graph.hidden_diff_bias_arr += self.learning_rate * self.graph.hidden_activity_arr * (-1)

            # Learning.
            if self.r_batch_size == 0 or self.r_batch_step % self.r_batch_size == 0:
                self.graph.visible_bias_arr += self.graph.visible_diff_bias_arr
                self.graph.hidden_bias_arr += self.graph.hidden_diff_bias_arr
                self.graph.visible_diff_bias_arr = np.zeros(self.graph.visible_bias_arr.shape)
                self.graph.hidden_diff_bias_arr = np.zeros(self.graph.hidden_bias_arr.shape)
                self.graph.learn_weights()

    def dropout(self, np.ndarray[DOUBLE_t, ndim=1] activity_arr):
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
