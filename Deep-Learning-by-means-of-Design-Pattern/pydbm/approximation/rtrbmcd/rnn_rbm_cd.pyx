# -*- coding: utf-8 -*-
import numpy as np
cimport numpy as np
cimport cython
from pydbm.approximation.rt_rbm_cd. import RTRBMCD
ctypedef np.float64_t DOUBLE_t


class RNNRBMCD(RTRBMCD):
    '''
    Recurrent Neural Network Restricted Boltzmann Machines(RNN-RBM)
    based on Contrastive Divergence.

    Conceptually, the positive phase is to the negative phase what waking is to sleeping.

    Parameters:
        __graph.weights_arr:                $W$ (Connection between v^{(t)} and h^{(t)})
        __graph.visible_bias_arr:           $b_v$ (Bias in visible layer)
        __graph.hidden_bias_arr:            $b_h$ (Bias in hidden layer)
        __graph.rnn_hidden_weights_arr:     $W'$ (Connection between h^{(t-1)} and b_h^{(t)})
        __graph.rnn_visible_weights_arr:    $W''$ (Connection between h^{(t-1)} and b_v^{(t)})
        __graph.hat_hidden_activity_arr:    $\hat{h}^{(t)}$ (RNN with hidden units)
        __graph.pre_hidden_activity_arr:    $\hat{h}^{(t-1)}$
        __graph.v_hat_weights_arr:          $W_2$ (Connection between v^{(t)} and \hat{h}^{(t)})
        __graph.hat_weights_arr:            $W_3$ (Connection between \hat{h}^{(t-1)} and \hat{h}^{(t)})
        __graph.rnn_hidden_bias:            $b_{\hat{h}^{(t)}}$ (Bias of RNN hidden layers.)

    $$\hat{h}^{(t)} = \sig (W_2 v^{(t)} + W_3 \hat{h}^{(t-1)} + b_{\hat{h}})
    '''

    # gradient in RNN layer.
    h_p_diff_arr = np.array([])

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

        cdef np.ndarray[DOUBLE_t, ndim=2] link_value_arr = (self.__graph.v_hat_weights_arr * self.graph.visible_activity_arr.reshape(-1, 1)) + (self.__graph.hat_weights_arr * self.__graph.pre_hidden_activity_arr.reshape(-1, 1)) + self.__graph.rnn_hidden_bias.reshape(-1, 1)
        link_value_arr = np.nan_to_num(link_value_arr)
        self.graph.hat_hidden_activity_arr = link_value_arr.sum(axis=0)
        
        super().rnn_learn(observed_data_arr)

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
        
        cdef np.ndarray[DOUBLE_t, ndim=2] h_p_diff_arr = self.graph.hat_hidden_activity_arr - self.graph.pre_hidden_activity_arr
        if self.h_p_diff_arr.shape[0]:
            self.h_p_diff_arr += h_p_diff_arr
        else:
            self.h_p_diff_arr = h_p_diff_arr

        super().memorize_activity(observed_data_arr, negative_visible_activity_arr)

    def back_propagation(self):
        '''
        Details of the backpropagation through time algorithm.
        '''
        self.graph.rnn_hidden_weights_arr = self.b_h_c_diff_arr * self.graph.hat_hidden_activity_arr.reshape(-1, 1) * self.learning_rate
        self.graph.rnn_visible_weights_arr = (self.b_v_c_diff_arr * self.graph.hat_hidden_activity_arr.reshape(-1, 1)).T * self.learning_rate
        self.graph.rnn_hidden_bias = self.h_p_diff_arr * self.graph.hat_hidden_activity_arr.reshape(-1, 1) * self.learning_rate
        self.b_v_c_diff_arr = np.array([])
        self.b_h_c_diff_arr = np.array([])
        self.h_p_diff_arr = np.array([])
