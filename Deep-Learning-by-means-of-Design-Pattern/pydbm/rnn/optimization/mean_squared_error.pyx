# -*- coding: utf-8 -*-
import numpy as np
cimport numpy as np
from pydbm.rnn.optimization.interface.optimizable_loss import OptimizableLoss


class MeanSquaredError(OptimizableLoss):
    '''
    The mean squared error (MSE).
    '''

    def compute_loss(self, np.ndarray pred_arr, np.ndarray labeled_arr, axis=None):
        '''
        Return of result from this Cost function.

        Args:
            pred_arr:       Predicted data.
            labeled_arr:    Labeled data.
            axis:           None or int or tuple of ints, optional.
                            Axis or axes along which the means are computed.
                            The default is to compute the mean of the flattened array.

        Returns:
            Cost.
        '''
        return np.square(labeled_arr - pred_arr).mean(axis=axis)

    def optimize(
        self,
        graph,
        np.ndarray input_arr,
        np.ndarray rnn_arr,
        np.ndarray hidden_arr,
        np.ndarray output_arr,
        np.ndarray label_arr,
        double learning_rate
    ):
        '''
        Optimize.
        
        Args:
            loss:           Loss
            learning_rate:  Learning rate.

        '''
        cdef int samples_n = input_arr.shape[0]
        cdef int i = 0

        cdef np.ndarray delta_o
        cdef np.ndarray delta_h
        cdef np.ndarray delta_i

        cdef np.ndarray o_diff_arr
        cdef np.ndarray h_diff_arr

        cdef np.ndarray ho_diff_arr
        cdef np.ndarray hf_diff_arr
        cdef np.ndarray hi_diff_arr
        cdef np.ndarray hg_diff_arr

        cdef np.ndarray xo_diff_arr
        cdef np.ndarray xf_diff_arr
        cdef np.ndarray xi_diff_arr
        cdef np.ndarray xg_diff_arr

        for i in range(samples_n):
            delta_o = - (label_arr[i] - output_arr[i]) * output_arr[i] * (1 - output_arr[i])
            if i == 0:
                o_diff_arr = (learning_rate * delta_o * hidden_arr[i].reshape(-1, 1))
            else:
                o_diff_arr += (learning_rate * delta_o * hidden_arr[i].reshape(-1, 1))

            delta_h = np.sum(delta_o * graph.weights_hidden_output_arr) * (hidden_arr[i] * (1 - hidden_arr[i]))
            if i == 0:
                h_diff_arr = (learning_rate * delta_h * rnn_arr[i].reshape(-1, 1))
            else:
                h_diff_arr += (learning_rate * delta_h * rnn_arr[i].reshape(-1, 1))
            
            delta_rnn = np.sum(delta_h * graph.weights_hy_arr) * (rnn_arr[i].reshape(-1, 1))
            if i == 0:
                ho_diff_arr = (learning_rate * delta_rnn * np.dot(input_arr[i], graph.weights_xo_arr).reshape(-1, 1))
                hf_diff_arr = (learning_rate * delta_rnn * np.dot(input_arr[i], graph.weights_xf_arr).reshape(-1, 1))
                hi_diff_arr = (learning_rate * delta_rnn * np.dot(input_arr[i], graph.weights_xi_arr).reshape(-1, 1))
                hg_diff_arr = (learning_rate * delta_rnn * np.dot(input_arr[i], graph.weights_xg_arr).reshape(-1, 1))
            else:
                ho_diff_arr += (learning_rate * delta_rnn * np.dot(input_arr[i], graph.weights_xo_arr).reshape(-1, 1))
                hf_diff_arr += (learning_rate * delta_rnn * np.dot(input_arr[i], graph.weights_xf_arr).reshape(-1, 1))
                hi_diff_arr += (learning_rate * delta_rnn * np.dot(input_arr[i], graph.weights_xi_arr).reshape(-1, 1))
                hg_diff_arr += (learning_rate * delta_rnn * np.dot(input_arr[i], graph.weights_xg_arr).reshape(-1, 1))
            
            delta_i = np.sum(delta_rnn * graph.weights_hg_arr) * (input_arr[i].reshape(-1, 1))
            if i == 0:
                xo_diff_arr = (learning_rate * delta_i * input_arr[i].reshape(-1, 1))
                xf_diff_arr = (learning_rate * delta_i * input_arr[i].reshape(-1, 1))
                xg_diff_arr = (learning_rate * delta_i * input_arr[i].reshape(-1, 1))
                xi_diff_arr = (learning_rate * delta_i * input_arr[i].reshape(-1, 1))
            else:
                xo_diff_arr += (learning_rate * delta_i * input_arr[i].reshape(-1, 1))
                xf_diff_arr += (learning_rate * delta_i * input_arr[i].reshape(-1, 1))
                xg_diff_arr += (learning_rate * delta_i * input_arr[i].reshape(-1, 1))
                xi_diff_arr += (learning_rate * delta_i * input_arr[i].reshape(-1, 1))

        graph.weights_hidden_output_arr -= o_diff_arr

        graph.weights_hy_arr -= h_diff_arr

        graph.weights_ho_arr -= ho_diff_arr
        graph.weights_hf_arr -= hf_diff_arr
        graph.weights_hi_arr -= hi_diff_arr
        graph.weights_hg_arr -= hg_diff_arr

        graph.weights_xo_arr -= xo_diff_arr
        graph.weights_xf_arr -= xf_diff_arr
        graph.weights_xg_arr -= xg_diff_arr
        graph.weights_xi_arr -= xi_diff_arr

        return graph
