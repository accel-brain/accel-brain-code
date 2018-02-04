# -*- coding: utf-8 -*-
from multipledispatch import dispatch
from pydbm.nn.nn_director import NNDirector
from pydbm.nn.interface.nn_builder import NNBuilder


class NeuralNetwork(object):
    '''
    The object of neural networks.
    '''

    # The list of graphs of neural networks.
    nn_list = []

    @dispatch(NNBuilder, int, int, int, list)
    def __init__(
        self,
        nn_builder,
        input_neuron_count,
        hidden_neuron_count,
        output_neuron_count,
        activating_function_list
    ):
        '''
        Initialize.

        Args:
            nn_builder:                 `Concrete Builder` in Builder Pattern.
            input_neuron_count:         The number of neurons in input layer.
            hidden_neuron_count:   　   The number of neurons in hidden layer.
            output_neuron_count:        The number of neurons in output layer.
            activating_function_list:   The list of activation function.
        '''
        nn_director = NNDirector(
            nn_builder=nn_builder
        )
        nn_director.nn_construct(
            neuron_assign_list=[input_neuron_count, hidden_neuron_count, output_neuron_count],
            activating_function_list=activating_function_list
        )

        self.nn_list = nn_director.nn_list

    @dispatch(NNBuilder, list, list)
    def __init__(
        self,
        nn_builder,
        neuron_assign_list,
        activating_function_list
    ):
        '''
        Initialize.

        Args:
            neuron_assign_list:         The list of the number of neurons in each layers.
                                        0: input layer.
                                        -1: output layer.
                                        others: hidden layer.
            activating_function_list:   The list of activation function.
        '''
        nn_director = NNDirector(
            nn_builder=nn_builder
        )
        nn_director.nn_construct(
            neuron_assign_list=neuron_assign_list,
            activating_function_list=activating_function_list
        )

        self.nn_list = nn_director.nn_list

    def learn(
        self,
        traning_data_matrix,
        class_data_matrix,
        learning_rate=0.5,
        momentum_factor=0.1,
        traning_count=1000,
        learning_rate_list=None
    ):
        '''
        Excute foward propagation and back propagation recursively.

        Args:
            traning_data_matrix:    Training data.
            class_data_matrix:      Class data.
            learning_rate:          Learning rate.
            momentum_factor:        Momentum factor.
            traning_count:          Training counts.
            learning_rate_list:     The list of learning rates.

        '''
        if len(traning_data_matrix) != len(class_data_matrix):
            raise ValueError()
        for i in range(traning_count):
            for j in range(len(traning_data_matrix)):
                self.forward_propagate(traning_data_matrix[j])
                if learning_rate_list is not None:
                    learning_rate = learning_rate_list[j]

                self.back_propagate(
                    test_data_list=class_data_matrix[j],
                    learning_rate=learning_rate,
                    momentum_factor=momentum_factor
                )

    def forward_propagate(self, input_data_list):
        '''
        Foward propagation.

        Args:
            input_data_list:  The list of input data.

        '''
        nn_from_input_to_hidden_layer = self.nn_list[0]
        nn_hidden_layer_list = self.nn_list[1:len(self.nn_list) - 1]
        nn_to_output_layer = self.nn_list[-1]
        [nn_from_input_to_hidden_layer.shallower_neuron_list[i].observe_data_point(input_data_list[i]) for i in range(len(input_data_list))]

        # In input layer.
        shallower_activity_arr = [[nn_from_input_to_hidden_layer.shallower_neuron_list[i].activity] * len(nn_from_input_to_hidden_layer.deeper_neuron_list) for i in range(len(nn_from_input_to_hidden_layer.shallower_neuron_list))]
        link_value_arr = shallower_activity_arr * nn_from_input_to_hidden_layer.weights_arr
        link_value_list = mx.ndarray.sum(link_value_arr, axis=0)
        [nn_from_input_to_hidden_layer.deeper_neuron_list[j].hidden_update_state(link_value_list[j]) for j in range(len(link_value_list))]
        nn_from_input_to_hidden_layer.normalize_visible_bias()
        nn_from_input_to_hidden_layer.normalize_hidden_bias()

        # In hidden layers.
        for nn_hidden_layer in nn_hidden_layer_list:
            shallower_activity_arr = [[nn_hidden_layer.shallower_neuron_list[i].activity] * len(nn_hidden_layer.deeper_neuron_list) for i in range(len(nn_hidden_layer.shallower_neuron_list))]
            link_value_arr = shallower_activity_arr * nn_hidden_layer.weights_arr
            link_value_list = mx.ndarray.sum(link_value_arr, axis=0)
            [nn_hidden_layer.deeper_neuron_list[j].hidden_update_state(link_value_list[j]) for j in range(len(link_value_list))]
            nn_hidden_layer.normalize_visible_bias()
            nn_hidden_layer.normalize_hidden_bias()

        # In output layer
        shallower_activity_arr = [[nn_to_output_layer.shallower_neuron_list[i].activity] * len(nn_to_output_layer.deeper_neuron_list) for i in range(len(nn_to_output_layer.shallower_neuron_list))]
        link_value_arr = shallower_activity_arr * nn_to_output_layer.weights_arr
        link_value_list = mx.ndarray.sum(link_value_arr, axis=0)
        [nn_to_output_layer.deeper_neuron_list[j].output_update_state(link_value_list[j]) for j in range(len(link_value_list))]
        nn_to_output_layer.normalize_visible_bias()

    def back_propagate(
        self,
        test_data_list,
        learning_rate=0.05,
        momentum_factor=0.1
    ):
        '''
        Back propagation.

        Args:
            test_data_list:     The list of test data.
            learning_rate:      Learning rate.
            momentum_factor:    Momentum factor.
        '''
        back_nn_list = [back_nn for back_nn in reversed(self.nn_list)]
        back_nn_list[0].back_propagate(
            propagated_list=test_data_list,
            learning_rate=learning_rate,
            momentum_factor=momentum_factor,
            back_nn_list=back_nn_list,
            back_nn_index=0
        )

    def predict(self, test_data_list):
        '''
        Predict.

        Args:
            test_data_matrix:   test data.

        Returns:
            Predicted result.
        '''

        output_data_list = []
        self.forward_propagate(test_data_list)
        for output_neuron in self.nn_list[-1].deeper_neuron_list:
            output_data_list.append(output_neuron.release())

        return output_data_list
