# -*- coding: utf-8 -*-
import numpy as np
cimport numpy as np
import warnings
cimport cython
from pydbm.approximation.interface.approximate_interface import ApproximateInterface
from pydbm.optimization.opt_params import OptParams
from pydbm.optimization.optparams.sgd import SGD
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

    def __init__(self, opt_params=None):
        '''
        Init.
        
        Args:
            opt_params:     Optimization function.

        '''
        if opt_params is None:
            opt_params = SGD(momentum=0.0)

        if isinstance(opt_params, OptParams):
            self.__opt_params = opt_params
        else:
            raise TypeError()

    def approximate_learn(
        self,
        graph,
        double learning_rate,
        np.ndarray[DOUBLE_t, ndim=3] observed_data_arr,
        int traning_count=-1,
        int batch_size=200,
        int training_count=1000
    ):
        '''
        learning with function approximation.

        Args:
            graph:                Graph of neurons.
            learning_rate:        Learning rate.
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

        if traning_count != -1:
            training_count = traning_count
            warnings.warn("`traning_count` will be removed in future version. Use `training_count`.", FutureWarning)

        self.graph = graph
        self.learning_rate = learning_rate
        self.batch_size = batch_size

        # Learning.
        for _ in range(training_count):
            rand_index = np.random.choice(observed_data_arr.shape[0], size=batch_size)
            batch_observed_arr = observed_data_arr[rand_index]
            for cycle_index in range(batch_observed_arr.shape[1]):
                # RNN learning.
                self.rnn_learn(batch_observed_arr[:, cycle_index])
                # Wake and sleep.
                self.wake_sleep_learn(self.graph.visible_activity_arr)
                # Memorizing.
                self.memorize_activity(
                    batch_observed_arr[:, cycle_index],
                    self.graph.visible_activity_arr
                )

            # Back propagation.
            self.back_propagation()

        return self.graph

    def approximate_inference(
        self,
        graph,
        double learning_rate,
        np.ndarray[DOUBLE_t, ndim=3] observed_data_arr,
        int traning_count=-1,
        int r_batch_size=200,
        int training_count=1000
    ):
        '''
        Inference with function approximation.

        Args:
            graph:                Graph of neurons.
            learning_rate:        Learning rate.
            observed_data_arr:    observed data points.
            training_count:       Training counts.
            r_batch_size:         Batch size.
                                  If this value is `0`, the inferencing is a recursive learning.
                                  If this value is more than `0`, the inferencing is a mini-batch recursive learning.
                                  If this value is '-1', the inferencing is not a recursive learning.

        Returns:
            Graph of neurons.
        '''
        cdef int counter = 0
        cdef np.ndarray rand_index
        cdef np.ndarray[DOUBLE_t, ndim=3] batch_observed_arr
        cdef int batch_index
        cdef np.ndarray inferenced_arr = np.array([])
        cdef np.ndarray feature_points_arr = np.array([])

        if traning_count != -1:
            training_count = traning_count
            warnings.warn("`traning_count` will be removed in future version. Use `training_count`.", FutureWarning)

        self.graph = graph
        self.learning_rate = learning_rate
        self.r_batch_size = r_batch_size

        for counter in range(training_count):
            if r_batch_size > 0:
                rand_index = np.random.choice(observed_data_arr.shape[0], size=r_batch_size)
                batch_observed_arr = observed_data_arr[rand_index]
            else:
                batch_observed_arr = observed_data_arr

            for cycle_index in range(batch_observed_arr.shape[1]):
                # RNN learning.
                self.rnn_learn(batch_observed_arr[:, cycle_index])
                self.memorize_activity(
                    batch_observed_arr[:, cycle_index],
                    self.graph.visible_activity_arr
                )
                self.wake_sleep_inference(self.graph.visible_activity_arr)

            # Back propagation.
            self.back_propagation()

        inferenced_arr = self.graph.visible_activity_arr
        feature_points_arr = self.graph.hidden_activity_arr

        self.graph.inferenced_arr = inferenced_arr
        self.graph.feature_points_arr = feature_points_arr
        return self.graph

    def rnn_learn(self, np.ndarray[DOUBLE_t, ndim=2] observed_data_arr):
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

        self.graph.rnn_hidden_bias_arr = np.dot(
            self.graph.hat_hidden_activity_arr,
            self.graph.rnn_hidden_weights_arr
        ) + self.graph.hidden_bias_arr
        
        self.graph.rnn_visible_bias_arr = np.dot(
            self.graph.hat_hidden_activity_arr,
            self.graph.rnn_visible_weights_arr.T
        ) + self.graph.visible_bias_arr
        
        self.graph.hidden_activity_arr = self.graph.hidden_activating_function.activate(
            np.dot(
                self.graph.pre_hidden_activity_arr,
                self.graph.rnn_hidden_weights_arr
            ) + self.graph.hidden_bias_arr
        )
        
        self.graph.visible_activity_arr = self.graph.visible_activating_function.activate(
            np.dot(
                self.graph.pre_hidden_activity_arr,
                self.graph.rnn_visible_weights_arr.T
            ) + self.graph.visible_bias_arr
        )

    def memorize_activity(
        self,
        np.ndarray[DOUBLE_t, ndim=2] observed_data_arr,
        np.ndarray[DOUBLE_t, ndim=2] negative_visible_activity_arr
    ):
        '''
        Memorize activity.

        Args:
            observed_data_arr:                Observed data points in positive phase.
            negative_visible_activity_arr:    visible acitivty in negative phase.
        '''
        self.graph.pre_hidden_activity_arr = self.graph.hat_hidden_activity_arr
        
        self.graph.hat_hidden_activity_arr = self.graph.rnn_activating_function.activate(
            np.dot(
                self.graph.visible_activity_arr,
                self.graph.weights_arr
            ) + self.graph.hidden_bias_arr.T
        )

        self.graph.visible_diff_bias_arr += np.nansum((observed_data_arr - negative_visible_activity_arr), axis=0)

    def back_propagation(self):
        '''
        Details of the backpropagation through time algorithm.
        '''
        # Learning.
        cdef np.ndarray[DOUBLE_t, ndim=2] visible_step_arr = (
            self.graph.visible_activity_arr + self.graph.visible_diff_bias_arr
        )
        
        cdef np.ndarray[DOUBLE_t, ndim=2] visible_step_activity_arr = self.graph.rnn_activating_function.activate(
            np.dot(
                visible_step_arr,
                self.graph.weights_arr
            ) - self.graph.hidden_bias_arr
        )

        cdef np.ndarray[DOUBLE_t, ndim=2] visible_negative_arr = self.graph.rnn_activating_function.activate(
            np.dot(
                self.graph.visible_activity_arr,
                self.graph.weights_arr
            ) + self.graph.hidden_bias_arr.T
        )
        self.graph.hidden_diff_bias_arr += (
            np.nansum(visible_step_activity_arr, axis=0) - np.nansum(visible_negative_arr, axis=0)
        )

        cdef np.ndarray[DOUBLE_t, ndim=2] step_arr = np.dot(
            visible_step_activity_arr.T,
            visible_step_arr
        )
        
        cdef np.ndarray[DOUBLE_t, ndim=2] negative_arr = np.dot(
            visible_negative_arr.T,
            self.graph.visible_activity_arr
        )
        
        self.graph.diff_weights_arr += (step_arr - negative_arr).T

        cdef np.ndarray[DOUBLE_t, ndim=2] delta_rnn_hidden_weight_arr = np.dot(
            (visible_step_activity_arr - visible_negative_arr).T,
            self.graph.pre_hidden_activity_arr
        )

        delta_rnn_visible_weight_arr = np.dot(
            self.graph.visible_diff_bias_arr.reshape(-1, 1),
            np.nansum(self.graph.pre_hidden_activity_arr, axis=0).reshape(-1, 1).T
        )
        
        params_list = self.__opt_params.optimize(
            params_list=[
                self.graph.visible_bias_arr,
                self.graph.hidden_bias_arr,
                self.graph.weights_arr,
                self.graph.rnn_hidden_weights_arr,
                self.graph.rnn_visible_weights_arr
            ],
            grads_list=[
                self.graph.visible_diff_bias_arr,
                self.graph.hidden_diff_bias_arr,
                self.graph.diff_weights_arr,
                delta_rnn_hidden_weight_arr,
                delta_rnn_visible_weight_arr
            ],
            learning_rate=self.learning_rate
        )
        self.graph.visible_bias_arr = params_list[0]
        self.graph.hidden_bias_arr = params_list[1]
        self.graph.weights_arr = params_list[2]
        self.graph.rnn_hidden_weights_arr = params_list[3]
        self.graph.rnn_visible_weights_arr = params_list[4]

        self.graph.visible_diff_bias_arr = np.zeros(self.graph.visible_bias_arr.shape)
        self.graph.hidden_diff_bias_arr = np.zeros(self.graph.hidden_bias_arr.shape)
        self.graph.diff_weights_arr = np.zeros_like(self.graph.weights_arr, dtype=np.float64)

    def wake_sleep_learn(self, np.ndarray[DOUBLE_t, ndim=2] observed_data_arr):
        '''
        Waking, sleeping, and learning.

        Standing on the premise that the settings of
        the activation function and weights operation are common.

        The binary activity is unsupported.

        Args:
            observed_data_list:      observed data points.
        '''
        self.graph.visible_activity_arr = observed_data_arr

        self.graph.hidden_activity_arr = self.graph.hidden_activating_function.activate(
            np.dot(
                self.graph.visible_activity_arr,
                self.graph.weights_arr
            ) + self.graph.hidden_bias_arr
        )

        self.graph.hidden_activity_arr = self.__opt_params.dropout(self.graph.hidden_activity_arr)

        self.graph.diff_weights_arr += np.dot(
            self.graph.visible_activity_arr.T,
            self.graph.hidden_activity_arr
        )

        self.graph.visible_diff_bias_arr += np.nansum(self.graph.visible_activity_arr, axis=0)
        self.graph.hidden_diff_bias_arr += np.nansum(self.graph.hidden_activity_arr, axis=0)

        # Sleeping.
        self.graph.visible_activity_arr = self.graph.visible_activating_function.activate(
            np.dot(
                self.graph.hidden_activity_arr,
                self.graph.weights_arr.T
            ) + self.graph.visible_bias_arr
        )

        self.graph.visible_activity_arr = self.__opt_params.dropout(self.graph.visible_activity_arr)

        self.graph.hidden_activity_arr = self.graph.hidden_activating_function.activate(
            np.dot(
                self.graph.visible_activity_arr,
                self.graph.weights_arr
            ) + self.graph.hidden_bias_arr
        )

        self.graph.hidden_activity_arr = self.__opt_params.dropout(self.graph.hidden_activity_arr)

        self.graph.diff_weights_arr -= np.dot(
            self.graph.visible_activity_arr.T,
            self.graph.hidden_activity_arr
        )

        self.graph.visible_diff_bias_arr -= np.nansum(self.graph.visible_activity_arr, axis=0)
        self.graph.hidden_diff_bias_arr -= np.nansum(self.graph.hidden_activity_arr, axis=0)

    def wake_sleep_inference(self, np.ndarray[DOUBLE_t, ndim=2] observed_data_arr):
        '''
        Sleeping, waking, and inferencing.

        Args:
             observed_data_arr:      feature points.
        '''
        self.graph.visible_activity_arr = observed_data_arr
        
        self.graph.hidden_activity_arr = self.graph.hidden_activating_function.activate(
            np.dot(
                self.graph.visible_activity_arr,
                self.graph.weights_arr
            ) + self.graph.hidden_bias_arr
        )

        if self.r_batch_size != -1:
            self.graph.hidden_activity_arr = self.__opt_params.dropout(self.graph.hidden_activity_arr)
            self.graph.diff_weights_arr += np.dot(
                self.graph.visible_activity_arr.T,
                self.graph.hidden_activity_arr
            )
            self.graph.visible_diff_bias_arr += np.nansum(self.graph.visible_activity_arr, axis=0)
            self.graph.hidden_diff_bias_arr += np.nansum(self.graph.hidden_activity_arr, axis=0)

        # Sleeping.
        self.graph.visible_activity_arr = self.graph.visible_activating_function.activate(
            np.dot(
                self.graph.hidden_activity_arr,
                self.graph.weights_arr.T
            ) + self.graph.visible_bias_arr
        )

        if self.r_batch_size != -1:
            self.graph.visible_activity_arr = self.__opt_params.dropout(self.graph.visible_activity_arr)
            self.graph.hidden_activity_arr = self.graph.hidden_activating_function.activate(
                np.dot(
                    self.graph.visible_activity_arr,
                    self.graph.weights_arr
                ) + self.graph.hidden_bias_arr
            )
            self.graph.hidden_activity_arr = self.__opt_params.dropout(self.graph.hidden_activity_arr)

            self.graph.diff_weights_arr -= np.dot(
                self.graph.visible_activity_arr.T, 
                self.graph.hidden_activity_arr
            )
            self.graph.visible_diff_bias_arr -= np.nansum(self.graph.visible_activity_arr, axis=0)
            self.graph.hidden_diff_bias_arr -= np.nansum(self.graph.hidden_activity_arr, axis=0)

    def get_opt_params(self):
        ''' getter '''
        return self.__opt_params

    opt_params = property(get_opt_params, set_readonly)
