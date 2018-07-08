# -*- coding: utf-8 -*-
import numpy as np
cimport numpy as np
cimport cython
from pydbm.approximation.rt_rbm_cd import RTRBMCD
ctypedef np.float64_t DOUBLE_t


class LSTMRTRBMCD(RTRBMCD):
    '''
    LSTM RTRBM based on Contrastive Divergence.

    Conceptually, the positive phase is to the negative phase what waking is to sleeping.

    Parameters:
        graph.weights_arr:                  $W$ (Connection between v^{(t)} and h^{(t)})
        graph.visible_bias_arr:             $b_v$ (Bias in visible layer)
        graph.hidden_bias_arr:              $b_h$ (Bias in hidden layer)
        graph.rnn_hidden_weights_arr:       $W'$ (Connection between h^{(t-1)} and b_h^{(t)})
        graph.rbm_hidden_weights_arr:       $W_{R}$ (Connection between h^{(t-1)} and h^{(t)})
        graph.hat_hidden_activity_arr:      $\hat{h}^{(t)}$ (RNN with hidden units)
        graph.hidden_activity_arr_list:     $\hat{h}^{(t)} \ (t = 0, 1, ...)$
        graph.v_hat_weights_arr:            $W_2$ (Connection between v^{(t)} and \hat{h}^{(t)})
        graph.hat_weights_arr:              $W_3$ (Connection between \hat{h}^{(t-1)} and \hat{h}^{(t)})
        graph.rnn_hidden_bias_arr:          $b_{\hat{h}^{(t)}}$ (Bias of RNN hidden layers.)

    $$\hat{h}^{(t)} = \sig (W_2 v^{(t)} + W_3 \hat{h}^{(t-1)} + b_{\hat{h}})
    '''

    def rnn_learn(self, np.ndarray[DOUBLE_t, ndim=1] observed_data_arr):
        '''
        Learning for RNN.

        Args:
            observed_data_list:      observed data points.
        '''
        self.graph.visible_activity_arr = observed_data_arr.copy()
        
        self.graph.pre_rbm_hidden_activity_arr = self.graph.hidden_activity_arr

        if self.graph.pre_hidden_activity_arr.shape[0] == 0:
            return
        if self.graph.hat_hidden_activity_arr.shape[0] == 0:
            return

        cdef np.ndarray[DOUBLE_t, ndim=2] rbm_link_value_arr
        cdef np.ndarray[DOUBLE_t, ndim=1] rbm_link_value

        cdef np.ndarray[DOUBLE_t, ndim=2] v_link_value_arr
        cdef np.ndarray[DOUBLE_t, ndim=2] h_link_value_arr
        cdef np.ndarray[DOUBLE_t, ndim=2] link_value_arr

        v_link_value_arr = self.graph.v_hat_weights_arr * self.graph.visible_activity_arr.reshape(-1, 1)
        v_link_value_arr = np.nan_to_num(v_link_value_arr)
        v_link_value_arr = v_link_value_arr.sum(axis=0).reshape(-1, 1)

        h_link_value_arr = self.graph.hat_weights_arr * self.graph.pre_hidden_activity_arr.reshape(-1, 1)
        h_link_value_arr = np.nan_to_num(h_link_value_arr)
        h_link_value_arr = h_link_value_arr.sum(axis=1).reshape(-1, 1)

        link_value_arr = v_link_value_arr + h_link_value_arr + self.graph.rnn_hidden_bias_arr.reshape(-1, 1)
        link_value_arr = np.nan_to_num(link_value_arr)
        cdef int row = link_value_arr.shape[0]
        self.graph.hat_hidden_activity_arr = link_value_arr.reshape(row, )
        self.graph.hat_hidden_activity_arr = self.graph.rnn_activating_function.activate(
            self.graph.hat_hidden_activity_arr
        )        

        link_value_arr = (self.graph.rnn_hidden_weights_arr * self.graph.hat_hidden_activity_arr.reshape(-1, 1)) + self.graph.hidden_bias_arr.reshape(-1, 1) * self.learning_rate
        link_value_arr = np.nan_to_num(link_value_arr)
        self.graph.rnn_hidden_bias_arr = link_value_arr.sum(axis=1)

        link_value_arr = (self.graph.rnn_visible_weights_arr.T * self.graph.hat_hidden_activity_arr.reshape(-1, 1)) + self.graph.visible_bias_arr.reshape(-1, 1).T * self.learning_rate
        link_value_arr = np.nan_to_num(link_value_arr)
        self.graph.rnn_visible_bias_arr = link_value_arr.sum(axis=0)

        rbm_link_value_arr = self.graph.rbm_hidden_weights_arr * self.graph.hidden_activity_arr.reshape(-1, 1)
        rbm_link_value_arr = np.nan_to_num(rbm_link_value_arr)
        rbm_link_value = rbm_link_value_arr.sum(axis=0) * self.learning_rate

        link_value_arr = (self.graph.rnn_hidden_weights_arr * self.graph.pre_hidden_activity_arr.reshape(-1, 1)) + self.graph.hidden_bias_arr.reshape(-1, 1) * self.learning_rate
        link_value_arr = np.nan_to_num(link_value_arr)

        self.graph.hidden_activity_arr = link_value_arr.sum(axis=1) + rbm_link_value
        self.graph.hidden_activity_arr = self.graph.hidden_activating_function.activate(
            self.graph.hidden_activity_arr
        )

        link_value_arr = (self.graph.rnn_visible_weights_arr * self.graph.pre_hidden_activity_arr.reshape(-1, 1).T) + self.graph.visible_bias_arr.reshape(-1, 1) * self.learning_rate
        link_value_arr = np.nan_to_num(link_value_arr)
        self.graph.visible_activity_arr = link_value_arr.sum(axis=1)
        self.graph.visible_activity_arr = self.graph.visible_activating_function.activate(
            self.graph.visible_activity_arr
        )

        self.graph.visible_bias_arr_list.append(self.graph.visible_bias_arr)
        self.graph.hidden_bias_arr_list.append(self.graph.hidden_bias_arr)

    def memorize_activity(
        self,
        np.ndarray[DOUBLE_t, ndim=1] observed_data_arr,
        np.ndarray[DOUBLE_t, ndim=1] negative_visible_activity_arr
    ):
        '''
        Memorize activity.
        
        Override.

        Args:
            observed_data_arr:                Observed data points in positive phase.
            negative_visible_activity_arr:    visible acitivty in negative phase.
        '''
        self.graph.pre_hidden_activity_arr_list.append(self.graph.hat_hidden_activity_arr)
        super().memorize_activity(observed_data_arr, negative_visible_activity_arr)

    def back_propagation(self):
        '''
        Details of the backpropagation through time algorithm.
        
        Override.
        '''
        # Learning.
        self.graph.visible_bias_arr += self.graph.visible_diff_bias_arr
        self.graph.hidden_bias_arr += self.graph.hidden_diff_bias_arr
        self.graph.learn_weights()

        self.graph.diff_visible_bias_arr_list.append(self.graph.visible_diff_bias_arr)
        
        cdef np.ndarray visible_step_arr
        cdef np.ndarray link_value_arr
        cdef np.ndarray visible_step_activity
        cdef np.ndarray visible_negative_arr
        cdef np.ndarray diff

        visible_step_arr = (self.graph.visible_activity_arr + self.graph.visible_diff_bias_arr).reshape(-1, 1)
        link_value_arr = (self.graph.weights_arr * visible_step_arr) - self.graph.hidden_bias_arr.reshape(-1, 1).T
        link_value_arr = np.nan_to_num(link_value_arr)
        visible_step_activity = link_value_arr.sum(axis=0)
        visible_step_activity = self.graph.rnn_activating_function.activate(visible_step_activity)
        visible_negative_arr = (self.graph.weights_arr * self.graph.visible_activity_arr.reshape(-1, 1)) - self.graph.hidden_bias_arr.reshape(-1, 1).T
        visible_negative_arr = visible_negative_arr.sum(axis=0)
        visible_negative_arr = self.graph.rnn_activating_function.activate(visible_negative_arr)
        diff = (visible_step_activity - visible_negative_arr) * self.learning_rate
        self.graph.diff_hidden_bias_arr_list.append(diff)
        self.graph.hidden_bias_arr += diff
        self.graph.weights_arr += ((visible_step_activity.reshape(-1, 1) * visible_step_arr.reshape(-1, 1).T).T - (visible_negative_arr.reshape(-1, 1).T * self.graph.visible_activity_arr.reshape(-1, 1))) * self.learning_rate

        self.graph.rnn_hidden_weights_arr += (visible_step_activity.reshape(-1, 1) - visible_negative_arr.reshape(-1, 1)) * self.graph.pre_hidden_activity_arr.reshape(-1, 1) * self.learning_rate

        self.graph.rbm_hidden_weights_arr += (visible_step_activity.reshape(-1, 1) - visible_negative_arr.reshape(-1, 1)) * self.graph.pre_rbm_hidden_activity_arr.reshape(-1, 1) * self.learning_rate

        if len(self.graph.diff_hidden_bias_arr_list) <= 1:
            return
        if len(self.graph.diff_visible_bias_arr_list) <= 1:
            return
        if len(self.graph.diff_hidden_bias_arr_list) <= 1:
            return

        hat_list = self.graph.diff_hidden_bias_arr_list[::-1]

        diff_v_b_list = self.graph.diff_visible_bias_arr_list[::-1]

        diff_h_b_list = self.graph.diff_hidden_bias_arr_list[::-1]

        diff_rnn_hidden_bias_arr = None
        diff_hat_weights_arr = None
        diff_v_hat_weights_arr = None
        for t in range(len(hat_list) - 1):
            hat_diff = self.graph.hat_weights_arr * ((hat_list[t] - hat_list[t+1]) * hat_list[t] * (1 - hat_list[t+1])).reshape(-1, 1)
            hat_diff += np.nan_to_num(
                self.graph.rnn_hidden_weights_arr * diff_h_b_list[t+1].reshape(-1, 1)
            ).sum(axis=1)

            hat_diff = hat_diff * self.learning_rate
            hat_diff_arr = hat_diff.sum(axis=1)
            if t == 0:
                self.graph.hat_hidden_activity_arr += hat_diff_arr

            diff = hat_diff_arr * self.graph.hat_hidden_activity_arr * (1 - self.graph.hat_hidden_activity_arr) * self.learning_rate
            if diff_rnn_hidden_bias_arr is None:
                diff_rnn_hidden_bias_arr = diff
            else:
                diff_rnn_hidden_bias_arr += diff

            diff = hat_diff_arr * self.graph.hat_hidden_activity_arr.reshape(-1, 1) * (1 - self.graph.hat_hidden_activity_arr).reshape(-1, 1) * self.graph.pre_hidden_activity_arr.reshape(-1, 1) * self.learning_rate
            if diff_hat_weights_arr is None:
                diff_hat_weights_arr = diff
            else:
                diff_hat_weights_arr += diff
            
            diff = hat_diff_arr.reshape(-1, 1) * self.graph.hat_hidden_activity_arr.reshape(-1, 1) * (1 - self.graph.hat_hidden_activity_arr).reshape(-1, 1) * self.graph.visible_activity_arr.reshape(-1, 1).T * self.learning_rate
            diff = diff.T
            if diff_v_hat_weights_arr is None:
                diff_v_hat_weights_arr = diff
            else:
                diff_v_hat_weights_arr += diff

        self.graph.rnn_hidden_bias_arr += diff_rnn_hidden_bias_arr

        self.graph.hat_weights_arr += diff_hat_weights_arr

        self.graph.v_hat_weights_arr += diff_v_hat_weights_arr

        self.graph.visible_diff_bias_arr = np.zeros(self.graph.visible_bias_arr.shape)
        self.graph.hidden_diff_bias_arr = np.zeros(self.graph.hidden_bias_arr.shape)

        self.graph.diff_hidden_bias_arr_list = []
        self.graph.diff_visible_bias_arr_list = []
        self.graph.diff_hidden_bias_arr_list = []
