# -*- coding: utf-8 -*-
import numpy as np
cimport numpy as np
cimport cython
from pydbm.approximation.interface.approximate_interface import ApproximateInterface
ctypedef np.float64_t DOUBLE_t


class ShapeBMCD(ApproximateInterface):
    '''
    Contrastive Divergence for Shape-Boltzmann machine(Shape-BM).
    
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
    
    # the pair of layers.
    __v_h_flag = None
    
    def __init__(self, v_h_flag):
        '''
        Init.
        
        Args:
            v_h_flag:    If this value is `True`, the pair of layers is visible layer and hidden layer.
                         If this value is `False`, the pair of layers is hidden layer and hidden layer.
        '''
        if isinstance(v_h_flag, bool):
            self.__v_h_flag = v_h_flag
        else:
            raise TypeError()

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
            if self.__v_h_flag is True:
                self.__v_h_learn(observed_data_arr)
            else:
                self.__h_h_learn(observed_data_arr)

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
            if self.__v_h_flag is True:
                self.__v_h_inference(observed_data_arr)
            else:
                self.__h_h_inference(observed_data_arr)

        return self.__graph

    def __v_h_learn(self, np.ndarray[DOUBLE_t, ndim=1] observed_data_arr):
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
        cdef int left_num = np.floor(self.__graph.hidden_activity_arr.shape[0] / 2).astype(int)
        cdef int right_num = np.ceil(self.__graph.hidden_activity_arr.shape[0] / 2).astype(int)

        cdef np.ndarray[DOUBLE_t, ndim=2] left_link_value_arr = (self.__graph.weights_arr.T[:left_num, :]) * self.__graph.hidden_activity_arr[:left_num, :].reshape(-1, 1) + self.__graph.hidden_bias_arr[:left_num, :].reshape(-1, 1)
        left_link_value_arr = np.nan_to_num(left_link_value_arr)
        
        cdef np.ndarray[DOUBLE_t, ndim=2] right_link_value_arr = (self.__graph.weights_arr.T[right_num:, :]) * self.__graph.hidden_activity_arr[right_num:, :].reshape(-1, 1) + self.__graph.hidden_bias_arr[right_num:, :].reshape(-1, 1)
        right_link_value_arr = np.nan_to_num(right_link_value_arr)
        
        cdef np.ndarray[DOUBLE_t, ndim=1] left_visible_activity_arr = left_link_value_arr.sum(axis=0)
        cdef np.ndarray[DOUBLE_t, ndim=1] right_visible_activity_arr = right_link_value_arr.sum(axis=0)
        
        # Overlapping for Shape-BM.
        self.__graph.visible_activity_arr = np.r_[
            left_visible_activity_arr[:-1], 
            left_visible_activity_arr[-1] + right_visible_activity_arr[0],
            right_visible_activity_arr[1:]
        ]

        self.__graph.visible_activity_arr = self.__graph.visible_activating_function.activate(self.__graph.visible_activity_arr)

        # Validation.
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

    def __h_h_learn(self, np.ndarray[DOUBLE_t, ndim=1] observed_data_arr):
        '''
        Waking, sleeping, and learning.

        Standing on the premise that the settings of
        the activation function and weights operation are common.

        The binary activity is unsupported.

        Args:
            observed_data_list:      observed data points (feature points).
        '''
        # Waking.
        self.__graph.visible_activity_arr = observed_data_arr.copy()

        cdef int left_num = np.floor(self.__graph.visible_activity_arr.shape[0] / 2).astype(int)
        cdef int right_num = np.ceil(self.__graph.visible_activity_arr.shape[0] / 2).astype(int)

        cdef np.ndarray[DOUBLE_t, ndim=2] left_link_value_arr = (self.__graph.weights_arr[:left_num, :]) * self.__graph.visible_activity_arr[:left_num, :].reshape(-1, 1) + self.__graph.visible_bias_arr[:left_num, :].reshape(-1, 1)
        left_link_value_arr = np.nan_to_num(left_link_value_arr)
        
        cdef np.ndarray[DOUBLE_t, ndim=2] right_link_value_arr = (self.__graph.weights_arr[right_num:, :]) * self.__graph.visible_activity_arr[right_num:, :].reshape(-1, 1) + self.__graph.visible_bias_arr[right_num:, :].reshape(-1, 1)
        right_link_value_arr = np.nan_to_num(right_link_value_arr)

        cdef np.ndarray[DOUBLE_t, ndim=1] left_visible_activity_arr = left_link_value_arr.sum(axis=0)
        cdef np.ndarray[DOUBLE_t, ndim=1] right_visible_activity_arr = right_link_value_arr.sum(axis=0)

        # Overlapping for Shape-BM.
        self.__graph.hidden_activity_arr = np.r_[
            left_visible_activity_arr[:-1], 
            left_visible_activity_arr[-1] + right_visible_activity_arr[0],
            right_visible_activity_arr[1:]
        ]

        self.__graph.hidden_activity_arr = self.__graph.hidden_activating_function.activate(
            self.__graph.hidden_activity_arr
        )
        if self.__dropout_rate > 0:
            self.__graph.hidden_activity_arr = self.__dropout(self.__graph.hidden_activity_arr)

        self.__graph.diff_weights_arr += self.__graph.visible_activity_arr.reshape(-1, 1) * self.__graph.hidden_activity_arr.reshape(-1, 1).T * self.__learning_rate

        self.__graph.visible_diff_bias_arr += self.__learning_rate * self.__graph.visible_activity_arr
        self.__graph.hidden_diff_bias_arr += self.__learning_rate * self.__graph.hidden_activity_arr

        # Sleeping.
        cdef np.ndarray[DOUBLE_t, ndim=2] link_value_arr = (self.__graph.weights_arr.T) * self.__graph.hidden_activity_arr.reshape(-1, 1) + self.__graph.hidden_bias_arr.reshape(-1, 1)
        link_value_arr = np.nan_to_num(link_value_arr)
        self.__graph.visible_activity_arr = link_value_arr.sum(axis=0)
        self.__graph.visible_activity_arr = self.__graph.visible_activating_function.activate(self.__graph.visible_activity_arr)

        # Validation.
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

    def __v_h_inference(self, np.ndarray[DOUBLE_t, ndim=1] observed_data_arr):
        '''
        Sleeping, waking, and learning.

        Args:
             observed_data_arr:      feature points.
        '''
        self.__graph.hidden_activity_arr = observed_data_arr.copy()

        cdef int left_num = np.floor(self.__graph.hidden_activity_arr.shape[0] / 2).astype(int)
        cdef int right_num = np.ceil(self.__graph.hidden_activity_arr.shape[0] / 2).astype(int)

        cdef np.ndarray[DOUBLE_t, ndim=2] left_link_value_arr = (self.__graph.weights_arr.T[:left_num, :]) * self.__graph.hidden_activity_arr[:left_num, :].reshape(-1, 1) + self.__graph.hidden_bias_arr[:left_num, :].reshape(-1, 1)
        left_link_value_arr = np.nan_to_num(left_link_value_arr)
        
        cdef np.ndarray[DOUBLE_t, ndim=2] right_link_value_arr = (self.__graph.weights_arr.T[right_num:, :]) * self.__graph.hidden_activity_arr[right_num:, :].reshape(-1, 1) + self.__graph.hidden_bias_arr[right_num:, :].reshape(-1, 1)
        right_link_value_arr = np.nan_to_num(right_link_value_arr)
        
        cdef np.ndarray[DOUBLE_t, ndim=1] left_visible_activity_arr = left_link_value_arr.sum(axis=0)
        cdef np.ndarray[DOUBLE_t, ndim=1] right_visible_activity_arr = right_link_value_arr.sum(axis=0)
        
        # Overlapping for Shape-BM.
        self.__graph.visible_activity_arr = np.r_[
            left_visible_activity_arr[:-1], 
            left_visible_activity_arr[-1] + right_visible_activity_arr[0],
            right_visible_activity_arr[1:]
        ]

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

            self.__graph.diff_weights_arr += self.__graph.visible_activity_arr.reshape(-1, 1) * self.__graph.hidden_activity_arr.reshape(-1, 1).T * self.__learning_rate
            self.__graph.visible_diff_bias_arr += self.__learning_rate * self.__graph.visible_activity_arr
            self.__graph.hidden_diff_bias_arr += self.__learning_rate * self.__graph.hidden_activity_arr

            # Learning.
            if self.__r_batch_size == 0 or self.__r_batch_step % self.__r_batch_size == 0:
                self.__graph.visible_bias_arr += self.__graph.visible_diff_bias_arr
                self.__graph.hidden_bias_arr += self.__graph.hidden_diff_bias_arr
                self.__graph.visible_diff_bias_arr = np.zeros(self.__graph.visible_bias_arr.shape)
                self.__graph.hidden_diff_bias_arr = np.zeros(self.__graph.hidden_bias_arr.shape)
                self.__graph.learn_weights()

    def __h_h_inference(self, np.ndarray[DOUBLE_t, ndim=1] observed_data_arr):
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

        cdef int left_num = np.floor(self.__graph.visible_activity_arr.shape[0] / 2).astype(int)
        cdef int right_num = np.ceil(self.__graph.visible_activity_arr.shape[0] / 2).astype(int)

        cdef np.ndarray[DOUBLE_t, ndim=2] left_link_value_arr = (self.__graph.weights_arr[:left_num, :]) * self.__graph.visible_activity_arr[:left_num, :].reshape(-1, 1) + self.__graph.visible_bias_arr[:left_num, :].reshape(-1, 1)
        left_link_value_arr = np.nan_to_num(left_link_value_arr)
        
        cdef np.ndarray[DOUBLE_t, ndim=2] right_link_value_arr = (self.__graph.weights_arr[right_num:, :]) * self.__graph.visible_activity_arr[right_num:, :].reshape(-1, 1) + self.__graph.visible_bias_arr[right_num:, :].reshape(-1, 1)
        right_link_value_arr = np.nan_to_num(right_link_value_arr)

        cdef np.ndarray[DOUBLE_t, ndim=1] left_visible_activity_arr = left_link_value_arr.sum(axis=0)
        cdef np.ndarray[DOUBLE_t, ndim=1] right_visible_activity_arr = right_link_value_arr.sum(axis=0)

        # Overlapping for Shape-BM.
        self.__graph.hidden_activity_arr = np.r_[
            left_visible_activity_arr[:-1], 
            left_visible_activity_arr[-1] + right_visible_activity_arr[0],
            right_visible_activity_arr[1:]
        ]
        self.__graph.hidden_activity_arr = link_value_arr.sum(axis=0)
        self.__graph.hidden_activity_arr = self.__graph.hidden_activating_function.activate(
            self.__graph.hidden_activity_arr
        )

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

            self.__graph.diff_weights_arr += self.__graph.visible_activity_arr.reshape(-1, 1) * self.__graph.hidden_activity_arr.reshape(-1, 1).T * self.__learning_rate
            self.__graph.visible_diff_bias_arr += self.__learning_rate * self.__graph.visible_activity_arr
            self.__graph.hidden_diff_bias_arr += self.__learning_rate * self.__graph.hidden_activity_arr

            # Learning.
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
