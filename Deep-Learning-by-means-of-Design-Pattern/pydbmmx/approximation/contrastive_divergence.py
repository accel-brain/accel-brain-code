# -*- coding: utf-8 -*-
import mxnet as mx
from pydbmmx.approximation.interface.approximate_interface import ApproximateInterface


class ContrastiveDivergence(ApproximateInterface):
    '''
    Contrastive Divergence.
    
    Conceptually, the positive phase is to the negative phase what waking is to sleeping.
    '''

    # Graph of neurons.
    __graph = None
    # Learning rate.
    __learning_rate = 0.5

    detail_setting_flag = True

    def approximate_learn(
        self,
        graph,
        learning_rate,
        observed_data_arr,
        traning_count=1000
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

        if self.detail_setting_flag is True:
            for i in range(row_i):
                self.__detail_setting(observed_data_arr[i, :])
        else:
            for i in range(row_i):
                self.__wake_sleep_learn(observed_data_arr[i, :])

        return self.__graph

    def __detail_setting(self, observed_data_arr):
        '''
        Waking and sleeping.

        Args:
            observed_data_arr:      observed data points.
        '''
        self.__wake(observed_data_arr)
        self.__sleep()
        self.__learn()

    def __wake_sleep_learn(self, observed_data_arr):
        '''
        Waking, sleeping, and learning.

        Standing on the premise that the settings of
        the activation function and weights operation are common.

        The binary activity is unsupported.

        Args:
            observed_data_list:      observed data points.
        '''
        # Waking.
        visible_activity_arr = observed_data_arr
        row_w = self.__graph.weights_arr.shape[0]
        col_w = self.__graph.weights_arr.shape[1]
        link_value_arr = self.__graph.weights_arr * (mx.nd.ones((row_w, col_w)) * visible_activity_arr) + self.__graph.visible_bias_arr
        hidden_activity_arr = mx.ndarray.nansum(link_value_arr, axis=0)
        self.__graph.diff_weights_arr = visible_activity_arr * hidden_activity_arr.T * self.__learning_rate
        visible_diff_bias = self.__learning_rate * visible_activity_arr
        hidden_diff_bias = self.__learning_rate * hidden_activity_arr

        # Sleeping.
        hidden_activity_arr = hidden_activity_arr.reshape(-1, 1)
        _link_value_arr = self.__graph.weights_arr.T * (mx.nd.ones((col_w, row_w)) * hidden_activity_arr) + self.__graph.hidden_bias_arr
        _visible_activity_arr = mx.nd.nansum(_link_value_arr.sum, axis=0)

        _visible_activity_arr = self.__graph.visible_neuron_list[0].activating_function.activate(
            _visible_activity_arr + visible_diff_bias
        )
        visible_activity_sum = mx.ndarray.sum(_visible_activity_arr)
        if visible_activity_sum != 0:
            _visible_activity_arr = _visible_activity_arr / visible_activity_sum

        __link_value_arr = (self.__graph.weights_arr.T * _visible_activity_arr) + self.__graph.visible_bias_arr
        _hidden_activity_arr = mx.nd.nansum(__link_value_arr, axis=0)
        try:
            _hidden_activity_arr = self.__graph.hidden_neuron_list[0].activating_function.activate(
                _hidden_activity_arr + hidden_diff_bias
            )
        except AttributeError:
            _hidden_activity_arr = self.__graph.visible_neuron_list[0].activating_function.activate(
                _hidden_activity_arr + hidden_diff_bias
            )

        hidden_activity_sum = mx.ndarray.sum(_hidden_activity_arr)
        if hidden_activity_sum != 0:
            _hidden_activity_arr = _hidden_activity_arr / hidden_activity_sum

        self.__graph.diff_weights_arr += _visible_activity_arr * _hidden_activity_arr.T * self.__learning_rate * (-1)

        visible_diff_bias += self.__learning_rate * _visible_activity_arr * (-1)
        hidden_diff_bias += self.__learning_rate * _hidden_activity_arr * (-1)

        # Learning.
        if self.__graph.visible_bias_arr is None:
            self.__graph.visible_bias_arr = visible_diff_bias
        else:
            self.__graph.visible_bias_arr += visible_diff_bias

        if self.__graph.hidden_bias_arr is None:
            self.__graph.hidden_bias_arr = hidden_diff_bias
        else:
            self.__graph.hidden_bias_arr += hidden_diff_bias

        self.__graph.learn_weights()

    def __wake(self, observed_data_arr):
        '''
        Waking.

        Args:
            observed_data_list:      observed data points.
        '''
        row_k = self.__graph.visible_activity_arr.shape[0]
        [self.__graph.visible_neuron_list[k].observe_data_point(observed_data_arr[k]) for k in range(row_k)]
        self.__update_hidden_spike()
        # so called `Hebb rule`.
        self.__graph.update(self.__learning_rate)
        row_i = self.__graph.visible_activity_arr.shape[0]
        [self.__graph.visible_neuron_list[i].update_bias(self.__learning_rate) for i in range(row_i)]
        row_j = self.__graph.hidden_activity_arr.shape[0]
        [self.__graph.hidden_neuron_list[j].update_bias(self.__learning_rate) for j in range(row_j)]

    def __sleep(self):
        '''
        Sleeping.
        '''
        self.__update_visible_spike()
        self.__update_hidden_spike()
        # so called `Hebb rule`.
        self.__graph.update((-1) * self.__learning_rate)
        row_i = self.__graph.visible_activity_arr.shape[0]
        [self.__graph.visible_neuron_list[i].update_bias((-1) * self.__learning_rate) for i in range(row_i)]
        row_j = self.__graph.hidden_activity_arr.shape[0]
        [self.__graph.hidden_neuron_list[j].update_bias((-1) * self.__learning_rate) for j in range(row_j)]

    def __update_visible_spike(self):
        '''
        Update activity of neurons in visible layer.
        '''
        hidden_activity_arr = self.__graph.hidden_activity_arr
        link_value_arr = self.__graph.weights_arr.T * hidden_activity_arr
        visible_activity_arr = mx.ndarray.sum(link_value_arr, axis=0)
        for i in range(visible_activity_arr.shape[0]):
            self.__graph.visible_neuron_list[i].visible_update_state(visible_activity_arr[i])
        self.__graph.normalize_visible_bias()

    def __update_hidden_spike(self):
        '''
        Update activity of neurons in hidden layer.
        '''
        visible_activity_arr = self.__graph.visible_activity_arr
        link_value_arr = self.__graph.weights_arr * visible_activity_arr
        hidden_activity_arr = mx.ndarray.sum(link_value_arr, axis=0)
        for j in range(hidden_activity_arr.shape[0]):
            self.__graph.hidden_neuron_list[j].hidden_update_state(hidden_activity_arr[j])
        self.__graph.normalize_hidden_bias()

    def __learn(self):
        '''
        Learning the biases and weights.
        '''
        row_i = self.__graph.visible_activity_arr.shape[0]
        for i in range(row_i):
            self.__graph.visible_neuron_list[i].learn_bias()
        row_j = self.__graph.hidden_activity_arr.shape[0]
        for j in range(row_j):
            self.__graph.hidden_neuron_list[j].learn_bias()
        self.__graph.learn_weights()

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
