# -*- coding: utf-8 -*-
import numpy as np
cimport numpy as np
import warnings
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

    def approximate_learn(
        self,
        graph,
        double learning_rate,
        double dropout_rate,
        np.ndarray observed_data_arr,
        int traning_count=-1,
        int batch_size=200,
        int training_count=1000
    ):
        '''
        learning with function approximation.

        Args:
            graph:                Graph of neurons.
            learning_rate:        Learning rate.
            dropout_rate:         Dropout rate.
            observed_data_arr:    observed data points.
            training_count:       Training counts.
            batch_size:           Batch size (0: not mini-batch)

        Returns:
            Graph of neurons.
        '''
        cdef int _
        cdef np.ndarray rand_index
        cdef np.ndarray[DOUBLE_t, ndim=3] batch_observed_arr
        cdef int batch_index
        cdef np.ndarray[DOUBLE_t, ndim=2] time_series_X

        if traning_count != -1:
            training_count = traning_count
            warnings.warn("`traning_count` will be removed in future version. Use `training_count`.", FutureWarning)

        self.graph = graph
        self.learning_rate = learning_rate
        self.dropout_rate = dropout_rate
        self.batch_size = batch_size

        # Learning.
        for _ in range(training_count):
            rand_index = np.random.choice(observed_data_arr.shape[0], size=batch_size)
            batch_observed_arr = observed_data_arr[rand_index]
            for batch_index in range(batch_observed_arr.shape[0]):
                time_series_X = batch_observed_arr[batch_index]
                for cycle_index in range(time_series_X.shape[0]):
                    # RNN learning.
                    self.rnn_learn(time_series_X[cycle_index])
                    # Memorizing.
                    self.memorize_activity(
                        time_series_X[cycle_index],
                        self.graph.visible_activity_arr
                    )
                # Wake and sleep.
                self.wake_sleep_learn(self.graph.visible_activity_arr)

            # Back propagation.
            self.back_propagation()

        return self.graph

    def approximate_inference(
        self,
        graph,
        double learning_rate,
        double dropout_rate,
        np.ndarray observed_data_arr,
        int traning_count=-1,
        int r_batch_size=200,
        int training_count=1000,
    ):
        '''
        Inference with function approximation.

        Args:
            graph:                Graph of neurons.
            learning_rate:        Learning rate.
            dropout_rate:         Dropout rate.
            observed_data_arr:    observed data points.
            training_count:       Training counts.
            r_batch_size:         Batch size.
                                  If this value is `0`, the inferencing is a recursive learning.
                                  If this value is more than `0`, the inferencing is a mini-batch recursive learning.
                                  If this value is '-1', the inferencing is not a recursive learning.

        Returns:
            Graph of neurons.
        '''
        cdef int _
        cdef np.ndarray rand_index
        cdef np.ndarray[DOUBLE_t, ndim=3] batch_observed_arr
        cdef int batch_index
        cdef np.ndarray[DOUBLE_t, ndim=2] time_series_X
        cdef np.ndarray inferenced_arr = np.array([])

        if traning_count != -1:
            training_count = traning_count
            warnings.warn("`traning_count` will be removed in future version. Use `training_count`.", FutureWarning)

        self.graph = graph
        self.learning_rate = learning_rate
        self.dropout_rate = dropout_rate
        self.r_batch_size = r_batch_size

        for _ in range(training_count):
            if r_batch_size > 0:
                rand_index = np.random.choice(observed_data_arr.shape[0], size=r_batch_size)
                batch_observed_arr = observed_data_arr[rand_index]
            else:
                batch_observed_arr = observed_data_arr

            for batch_index in range(batch_observed_arr.shape[0]):
                time_series_X = batch_observed_arr[batch_index]
                for cycle_index in range(time_series_X.shape[0]):
                    # RNN learning.
                    self.rnn_learn(time_series_X[cycle_index])
                    if r_batch_size != -1:
                        self.memorize_activity(
                            time_series_X[cycle_index],
                            self.graph.visible_activity_arr
                        )
                self.wake_sleep_inference(self.graph.visible_activity_arr)
                if inferenced_arr.shape[0] == 0:
                    inferenced_arr = self.graph.visible_activity_arr
                else:
                    inferenced_arr = np.vstack([inferenced_arr, self.graph.visible_activity_arr])

            if r_batch_size != -1:
                # Back propagation.
                self.back_propagation()

        self.graph.inferenced_arr = inferenced_arr
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
        self.graph.rnn_hidden_bias_arr = link_value_arr.sum(axis=1)

        link_value_arr = (self.graph.rnn_visible_weights_arr.T * self.graph.hat_hidden_activity_arr.reshape(-1, 1)) + self.graph.visible_bias_arr.reshape(-1, 1).T * self.learning_rate
        link_value_arr = np.nan_to_num(link_value_arr)
        self.graph.rnn_visible_bias_arr = link_value_arr.sum(axis=0)

        link_value_arr = (self.graph.rnn_hidden_weights_arr * self.graph.pre_hidden_activity_arr.reshape(-1, 1)) + self.graph.hidden_bias_arr.reshape(-1, 1) * self.learning_rate
        link_value_arr = np.nan_to_num(link_value_arr)
        self.graph.hidden_activity_arr = link_value_arr.sum(axis=0)
        self.graph.hidden_activity_arr = self.graph.hidden_activating_function.activate(
            self.graph.hidden_activity_arr
        )

        link_value_arr = (self.graph.rnn_visible_weights_arr * self.graph.pre_hidden_activity_arr.reshape(-1, 1).T) + self.graph.visible_bias_arr.reshape(-1, 1) * self.learning_rate
        link_value_arr = np.nan_to_num(link_value_arr)
        self.graph.visible_activity_arr = link_value_arr.sum(axis=1)
        self.graph.visible_activity_arr = self.graph.visible_activating_function.activate(
            self.graph.visible_activity_arr
        )

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

    def back_propagation(self):
        '''
        Details of the backpropagation through time algorithm.
        '''
        # Learning.
        self.graph.visible_bias_arr += self.graph.visible_diff_bias_arr
        self.graph.hidden_bias_arr += self.graph.hidden_diff_bias_arr
        self.graph.learn_weights()

        visible_step_arr = (self.graph.visible_activity_arr + self.graph.visible_diff_bias_arr).reshape(-1, 1)
        link_value_arr = (self.graph.weights_arr * visible_step_arr) - self.graph.hidden_bias_arr.reshape(-1, 1).T
        link_value_arr = np.nan_to_num(link_value_arr)
        visible_step_activity = link_value_arr.sum(axis=0)
        visible_step_activity = self.graph.rnn_activating_function.activate(visible_step_activity)
        visible_negative_arr = (self.graph.weights_arr * self.graph.visible_activity_arr.reshape(-1, 1)) - self.graph.hidden_bias_arr.reshape(-1, 1).T
        visible_negative_arr = visible_negative_arr.sum(axis=0)
        visible_negative_arr = self.graph.rnn_activating_function.activate(visible_negative_arr)
        self.graph.hidden_bias_arr += (visible_step_activity - visible_negative_arr) * self.learning_rate
        self.graph.weights_arr += ((visible_step_activity.reshape(-1, 1) * visible_step_arr.reshape(-1, 1).T).T - (visible_negative_arr.reshape(-1, 1).T * self.graph.visible_activity_arr.reshape(-1, 1))) * self.learning_rate

        self.graph.rnn_hidden_weights_arr += (visible_step_activity.reshape(-1, 1) - visible_negative_arr.reshape(-1, 1)) * self.graph.pre_hidden_activity_arr.reshape(-1, 1) * self.learning_rate
        self.graph.rnn_visible_weights_arr += self.graph.visible_diff_bias_arr.reshape(-1, 1) * self.graph.pre_hidden_activity_arr.reshape(-1, 1).T * self.learning_rate

        self.graph.visible_diff_bias_arr = np.zeros(self.graph.visible_bias_arr.shape)
        self.graph.hidden_diff_bias_arr = np.zeros(self.graph.hidden_bias_arr.shape)

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

    def dropout(self, np.ndarray[DOUBLE_t, ndim=1] activity_arr):
        '''
        Dropout.
        '''
        cdef int row = activity_arr.shape[0]
        cdef int dropout_flag = np.random.binomial(n=1, p=self.__dropout_rate, size=1).astype(int)
        cdef np.ndarray[DOUBLE_t, ndim=1] dropout_rate_arr

        if dropout_flag == 1:
            dropout_rate_arr = np.random.randint(0, 2, size=(row, )).astype(np.float64)
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
