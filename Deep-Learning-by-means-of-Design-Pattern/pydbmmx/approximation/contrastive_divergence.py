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

    __detail_setting_flag = False

    def get_detail_setting_flag(self):
        ''' getter '''
        if isinstance(self.__detail_setting_flag, bool) is False:
            raise TypeError()
        return self.__detail_setting_flag

    def set_detail_setting_flag(self, value):
        ''' setter '''
        if isinstance(value, bool) is False:
            raise TypeError()
        self.__detail_setting_flag = value

    detail_setting_flag = property(get_detail_setting_flag, set_detail_setting_flag)

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

        for ______ in range(traning_count):
            if self.detail_setting_flag is False:
                self.__wake_sleep_learn(observed_data_arr)
            else:
                self.__detail_wake_sleep_learn(observed_data_arr)

        return self.__graph

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

        link_value_arr = (self.__graph.weights_arr * mx.ndarray.reshape(visible_activity_arr, shape=(-1, 1))) + mx.ndarray.reshape(self.__graph.visible_bias_arr, shape=(-1, 1))
        hidden_activity_arr = mx.ndarray.nansum(link_value_arr, axis=0)

        hidden_activity_sum = hidden_activity_arr.sum()
        if hidden_activity_sum != 0:
            hidden_activity_arr = hidden_activity_arr / hidden_activity_sum
        else:
            raise ValueError("In waking, the sum of activity in hidden layer is zero.")

        try:
            hidden_activity_arr = self.__graph.hidden_neuron_list[0].activating_function.activate(
                hidden_activity_arr
            )
        except AttributeError:
            hidden_activity_arr = self.__graph.visible_neuron_list[0].activating_function.activate(
                hidden_activity_arr
            )

        self.__graph.diff_weights_arr = mx.ndarray.reshape(visible_activity_arr, shape=(-1, 1)) * mx.ndarray.reshape(hidden_activity_arr, shape=(-1, 1)).T * self.__learning_rate

        visible_diff_bias = self.__learning_rate * visible_activity_arr
        visible_diff_bias_sum = visible_diff_bias.sum()
        if visible_diff_bias_sum != 0:
            visible_diff_bias = visible_diff_bias / visible_diff_bias_sum
        else:
            raise ValueError("In waking, the sum of bias in visible layer is zero.")

        hidden_diff_bias = self.__learning_rate * hidden_activity_arr
        hidden_diff_bias_sum = hidden_diff_bias.sum()
        if hidden_diff_bias_sum != 0:
            hidden_diff_bias = hidden_diff_bias / hidden_diff_bias_sum
        else:
            raise ValueError("In waking, the sum of bias in hidden layer is zero.")

        # Sleeping.
        link_value_arr = (self.__graph.weights_arr.T * mx.ndarray.reshape(hidden_activity_arr, shape=(-1, 1))) + mx.ndarray.reshape(self.__graph.hidden_bias_arr, shape=(-1, 1))
        visible_activity_arr = mx.nd.nansum(link_value_arr, axis=0)
        visible_activity_sum = visible_activity_arr.sum()
        if visible_activity_sum != 0:
            visible_activity_arr = visible_activity_arr / visible_activity_sum
        else:
            raise ValueError("In sleeping, the sum of activity in visible layer is zero.")

        visible_activity_arr = self.__graph.visible_neuron_list[0].activating_function.activate(
            visible_activity_arr + visible_diff_bias
        )

        link_value_arr = (self.__graph.weights_arr * mx.ndarray.reshape(visible_activity_arr, shape=(-1, 1))) + mx.ndarray.reshape(self.__graph.visible_bias_arr, shape=(-1, 1))
        hidden_activity_arr = mx.nd.nansum(link_value_arr, axis=0)
        hidden_activity_sum = hidden_activity_arr.sum()
        if hidden_activity_sum != 0:
            hidden_activity_arr = hidden_activity_arr / hidden_activity_sum
        else:
            raise ValueError("In sleeping, the sum of activity in hidden layer is zero.")

        try:
            hidden_activity_arr = self.__graph.hidden_neuron_list[0].activating_function.activate(
                hidden_activity_arr + hidden_diff_bias
            )
        except AttributeError:
            hidden_activity_arr = self.__graph.visible_neuron_list[0].activating_function.activate(
                hidden_activity_arr + hidden_diff_bias
            )

        self.__graph.diff_weights_arr += mx.ndarray.reshape(visible_activity_arr, shape=(-1, 1)) * mx.ndarray.reshape(hidden_activity_arr, shape=(-1, 1)).T * self.__learning_rate * (-1)

        visible_diff_bias += self.__learning_rate * visible_activity_arr * (-1)
        visible_diff_bias_sum = mx.ndarray.nansum(visible_diff_bias)
        if visible_diff_bias_sum != 0:
            visible_diff_bias = visible_diff_bias / visible_diff_bias_sum
        else:
            raise ValueError("In sleeping, the sum of bias in visible layer is zero.")

        hidden_diff_bias += self.__learning_rate * hidden_activity_arr * (-1)
        hidden_diff_bias_sum = mx.ndarray.nansum(hidden_diff_bias)
        if hidden_diff_bias_sum != 0:
            hidden_diff_bias = hidden_diff_bias / hidden_diff_bias_sum
        else:
            raise ValueError("In sleeping, the sum of bias in hidden layer is zero.")

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

        # Memorizing.
        self.__graph.visible_activity_arr = visible_activity_arr
        self.__graph.hidden_activity_arr = hidden_activity_arr

    def __detail_wake_sleep_learn(self, observed_data_arr):
        '''
        Waking, sleeping, and learning.

        Standing on the premise that the settings of
        the activation function and weights operation are NOT common.

        Args:
            observed_data_list:      observed data points.
        '''
        # Waking.
        visible_activity_arr = observed_data_arr

        link_value_arr = (self.__graph.weights_arr * mx.ndarray.reshape(visible_activity_arr, shape=(-1, 1))) + mx.ndarray.reshape(self.__graph.visible_bias_arr, shape=(-1, 1))
        hidden_activity_arr = mx.ndarray.nansum(link_value_arr, axis=0)

        hidden_activity_sum = hidden_activity_arr.sum()
        if hidden_activity_sum != 0:
            hidden_activity_arr = hidden_activity_arr / hidden_activity_sum
        else:
            raise ValueError("In waking, the sum of activity in hidden layer is zero.")

        for i in range(len(self.__graph.hidden_neuron_list)):
            hidden_activity_arr[i] = self.__graph.hidden_neuron_list[i].activating_function.activate(
                hidden_activity_arr[i]
            )

        self.__graph.diff_weights_arr = mx.ndarray.reshape(visible_activity_arr, shape=(-1, 1)) * mx.ndarray.reshape(hidden_activity_arr, shape=(-1, 1)).T * self.__learning_rate

        visible_diff_bias = self.__learning_rate * visible_activity_arr
        visible_diff_bias_sum = visible_diff_bias.sum()
        if visible_diff_bias_sum != 0:
            visible_diff_bias = visible_diff_bias / visible_diff_bias_sum
        else:
            raise ValueError("In waking, the sum of bias in visible layer is zero.")

        hidden_diff_bias = self.__learning_rate * hidden_activity_arr
        hidden_diff_bias_sum = hidden_diff_bias.sum()
        if hidden_diff_bias_sum != 0:
            hidden_diff_bias = hidden_diff_bias / hidden_diff_bias_sum
        else:
            raise ValueError("In waking, the sum of bias in hidden layer is zero.")

        # Sleeping.
        link_value_arr = (self.__graph.weights_arr.T * mx.ndarray.reshape(hidden_activity_arr, shape=(-1, 1))) + mx.ndarray.reshape(self.__graph.hidden_bias_arr, shape=(-1, 1))
        visible_activity_arr = mx.nd.nansum(link_value_arr, axis=0)
        visible_activity_sum = visible_activity_arr.sum()
        if visible_activity_sum != 0:
            visible_activity_arr = visible_activity_arr / visible_activity_sum
        else:
            raise ValueError("In sleeping, the sum of activity in visible layer is zero.")

        for i in range(len(self.__graph.visible_neuron_list)):
            visible_activity_arr[i] = self.__graph.visible_neuron_list[i].activating_function.activate(
                visible_activity_arr[i] + visible_diff_bias[i]
            )

        link_value_arr = (self.__graph.weights_arr * mx.ndarray.reshape(visible_activity_arr, shape=(-1, 1))) + mx.ndarray.reshape(self.__graph.visible_bias_arr, shape=(-1, 1))
        hidden_activity_arr = mx.nd.nansum(link_value_arr, axis=0)
        hidden_activity_sum = hidden_activity_arr.sum()
        if hidden_activity_sum != 0:
            hidden_activity_arr = hidden_activity_arr / hidden_activity_sum
        else:
            raise ValueError("In sleeping, the sum of activity in hidden layer is zero.")

        for i in range(len(self.__graph.hidden_neuron_list)):
            hidden_activity_arr[i] = self.__graph.hidden_neuron_list[i].activating_function.activate(
                hidden_activity_arr[i] + hidden_diff_bias[i]
            )

        self.__graph.diff_weights_arr += mx.ndarray.reshape(visible_activity_arr, shape=(-1, 1)) * mx.ndarray.reshape(hidden_activity_arr, shape=(-1, 1)).T * self.__learning_rate * (-1)

        visible_diff_bias += self.__learning_rate * visible_activity_arr * (-1)
        visible_diff_bias_sum = mx.ndarray.nansum(visible_diff_bias)
        if visible_diff_bias_sum != 0:
            visible_diff_bias = visible_diff_bias / visible_diff_bias_sum
        else:
            raise ValueError("In sleeping, the sum of bias in visible layer is zero.")

        hidden_diff_bias += self.__learning_rate * hidden_activity_arr * (-1)
        hidden_diff_bias_sum = mx.ndarray.nansum(hidden_diff_bias)
        if hidden_diff_bias_sum != 0:
            hidden_diff_bias = hidden_diff_bias / hidden_diff_bias_sum
        else:
            raise ValueError("In sleeping, the sum of bias in hidden layer is zero.")

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

        # Memorizing.
        self.__graph.visible_activity_arr = visible_activity_arr
        self.__graph.hidden_activity_arr = hidden_activity_arr

        for i in range(len(self.__graph.visible_neuron_list)):
            self.__graph.visible_neuron_list[i].activity = visible_activity_arr[i]
            self.__graph.visible_neuron_list[i].bias = visible_bias_arr[i]
        for i in range(len(self.__graph.hidden_neuron_list)):
            self.__graph.hidden_neuron_list[i].activity = hidden_activity_arr[i]
            self.__graph.hidden_neuron_list[i].bias = hidden_bias_arr[i]

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
