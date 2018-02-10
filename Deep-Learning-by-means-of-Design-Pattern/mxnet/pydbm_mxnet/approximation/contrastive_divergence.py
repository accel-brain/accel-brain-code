# -*- coding: utf-8 -*-
import mxnet as mx
from pydbm_mxnet.approximation.interface.approximate_interface import ApproximateInterface


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
    # Particle normalized flag
    __particle_normalize_flag = False

    def get_dropout_rate(self):
        ''' getter '''
        if isinstance(self.__dropout_rate, float) is False:
            raise TypeError()
        return self.__dropout_rate
    
    def set_dropout_rate(self, value):
        ''' setter '''
        if isinstance(value, float) is False:
            raise TypeError()
        self.__dropout_rate = value

    dropout_rate = property(get_dropout_rate, set_dropout_rate)

    def approximate_learn(
        self,
        graph,
        learning_rate,
        dropout_rate,
        observed_data_arr,
        traning_count=1000
    ):
        '''
        learning with function approximation.

        Args:
            graph:                Graph of neurons.
            learning_rate:        Learning rate.
            dropout_rate:         Dropout rate
            observed_data_arr:    observed data points.
            traning_count:        Training counts.

        Returns:
            Graph of neurons.
        '''
        self.__graph = graph
        self.__learning_rate = learning_rate
        self.__dropout_rate = dropout_rate

        for _ in range(traning_count):
            self.__wake_sleep_learn(observed_data_arr)

        return self.__graph

    def __wake_sleep_learn(self, observed_data_arr):
        '''
        Waking, sleeping, and learning.

        Standing on the premise that the settings of
        the activation function and weights operation are common.

        The binary activity is unsupported.

        Args:
            observed_data_arr:      observed data points.
        '''
        # Waking.
        self.__graph.visible_activity_arr = observed_data_arr

        link_value_arr = (self.__graph.weights_arr * mx.ndarray.reshape(self.__graph.visible_activity_arr, shape=(-1, 1))) + mx.ndarray.reshape(self.__graph.visible_bias_arr, shape=(-1, 1))
        self.__graph.hidden_activity_arr = mx.ndarray.nansum(link_value_arr, axis=0)

        if self.__particle_normalize_flag is True:
            hidden_activity_sum = self.__graph.hidden_activity_arr.sum()
            if hidden_activity_sum != 0:
                self.__graph.hidden_activity_arr = self.__graph.hidden_activity_arr / hidden_activity_sum
            else:
                raise ValueError("In waking, the sum of activity in hidden layer is zero.")

        self.__graph.hidden_activity_arr = self.__graph.hidden_activating_function.activate(
            self.__graph.hidden_activity_arr
        )
        if self.dropout_rate > 0:
            self.__graph.hidden_activity_arr = self.__dropout(self.__graph.hidden_activity_arr)

        self.__graph.diff_weights_arr = mx.ndarray.reshape(self.__graph.visible_activity_arr, shape=(-1, 1)) * mx.ndarray.reshape(self.__graph.hidden_activity_arr, shape=(-1, 1)).T * self.__learning_rate

        visible_diff_bias = self.__learning_rate * self.__graph.visible_activity_arr

        if self.__particle_normalize_flag is True:
            visible_diff_bias_sum = visible_diff_bias.sum()
            if visible_diff_bias_sum != 0:
                visible_diff_bias = visible_diff_bias / visible_diff_bias_sum
            else:
                raise ValueError("In waking, the sum of bias in visible layer is zero.")

        hidden_diff_bias = self.__learning_rate * self.__graph.hidden_activity_arr

        if self.__particle_normalize_flag is True:
            hidden_diff_bias_sum = hidden_diff_bias.sum()
            if hidden_diff_bias_sum != 0:
                hidden_diff_bias = hidden_diff_bias / hidden_diff_bias_sum
            else:
                raise ValueError("In waking, the sum of bias in hidden layer is zero.")

        # Sleeping.
        link_value_arr = (self.__graph.weights_arr.T * mx.ndarray.reshape(self.__graph.hidden_activity_arr, shape=(-1, 1))) + mx.ndarray.reshape(self.__graph.hidden_bias_arr, shape=(-1, 1))
        self.__graph.visible_activity_arr = mx.nd.nansum(link_value_arr, axis=0)

        if self.__particle_normalize_flag is True:
            visible_activity_sum = self.__graph.visible_activity_arr.sum()
            if visible_activity_sum != 0:
                self.__graph.visible_activity_arr = self.__graph.visible_activity_arr / visible_activity_sum
            else:
                raise ValueError("In sleeping, the sum of activity in visible layer is zero.")

        self.__graph.visible_activity_arr = self.__graph.visible_activating_function.activate(
            self.__graph.visible_activity_arr + visible_diff_bias
        )

        if self.dropout_rate > 0:
            self.__graph.visible_activity_arr = self.__dropout(self.__graph.visible_activity_arr)

        link_value_arr = (self.__graph.weights_arr * mx.ndarray.reshape(self.__graph.visible_activity_arr, shape=(-1, 1))) + mx.ndarray.reshape(self.__graph.visible_bias_arr, shape=(-1, 1))
        self.__graph.hidden_activity_arr = mx.nd.nansum(link_value_arr, axis=0)

        if self.__particle_normalize_flag is True:
            hidden_activity_sum = self.__graph.hidden_activity_arr.sum()
            if hidden_activity_sum != 0:
                self.__graph.hidden_activity_arr = self.__graph.hidden_activity_arr / hidden_activity_sum
            else:
                raise ValueError("In sleeping, the sum of activity in hidden layer is zero.")

        self.__graph.hidden_activity_arr = self.__graph.hidden_activating_function.activate(
            self.__graph.hidden_activity_arr + self.__graph.hidden_bias_arr
        )

        if self.dropout_rate > 0:
            self.__graph.hidden_activity_arr = self.__dropout(self.__graph.hidden_activity_arr)

        self.__graph.diff_weights_arr += mx.ndarray.reshape(self.__graph.visible_activity_arr, shape=(-1, 1)) * mx.ndarray.reshape(self.__graph.hidden_activity_arr, shape=(-1, 1)).T * self.__learning_rate * (-1)

        visible_diff_bias += self.__learning_rate * self.__graph.visible_activity_arr * (-1)

        if self.__particle_normalize_flag is True:
            visible_diff_bias_sum = mx.ndarray.nansum(visible_diff_bias)
            if visible_diff_bias_sum != 0:
                visible_diff_bias = visible_diff_bias / visible_diff_bias_sum
            else:
                raise ValueError("In sleeping, the sum of bias in visible layer is zero.")

        hidden_diff_bias += self.__learning_rate * self.__graph.hidden_activity_arr * (-1)

        if self.__particle_normalize_flag is True:
            hidden_diff_bias_sum = mx.ndarray.nansum(hidden_diff_bias)
            if hidden_diff_bias_sum != 0:
                hidden_diff_bias = hidden_diff_bias / hidden_diff_bias_sum
            else:
                raise ValueError("In sleeping, the sum of bias in hidden layer is zero.")

        # Learning.
        self.__graph.visible_bias_arr += visible_diff_bias
        self.__graph.hidden_bias_arr += hidden_diff_bias
        self.__graph.learn_weights()

    def __dropout(self, activity_arr):
        '''
        Dropout.
        '''
        dropout_rate_arr = mx.ndarray.random.uniform(0, 1, shape=activity_arr.shape) > 0.5
        activity_arr = activity_arr * dropout_rate_arr.T
        return activity_arr

    def recall(self, graph, observed_data_arr):
        '''
        Free association.

        Args:
            graph:                  Graph of neurons.
            observed_data_arr:      observed data points.

        Returns:
            Graph of neurons.

        '''
        self.__graph = graph
        [self.__wake_and_sleep(observed_data_arr[k]) for k in range(observed_data_arr.shape[0])]
        return self.__graph
