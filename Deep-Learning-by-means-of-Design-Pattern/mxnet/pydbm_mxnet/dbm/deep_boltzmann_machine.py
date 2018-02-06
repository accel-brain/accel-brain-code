# -*- coding: utf-8 -*-
import mxnet as mx
from multipledispatch import dispatch
from pydbm_mxnet.dbm.interface.dbm_builder import DBMBuilder
from pydbm_mxnet.dbm.dbm_director import DBMDirector
from pydbm_mxnet.activation.interface.activating_function_interface import ActivatingFunctionInterface
from pydbm_mxnet.approximation.interface.approximate_interface import ApproximateInterface


class DeepBoltzmannMachine(object):
    '''
    The `Client` in Builder Pattern,
    
    Build deep boltzmann machine.
    '''

    # The list of restricted boltzmann machines.
    __rbm_list = []
    # The dict of Hyper parameters.
    __hyper_param_dict = {}

    @dispatch(DBMBuilder, int, int, int, ActivatingFunctionInterface, ApproximateInterface, float)
    def __init__(
        self,
        dbm_builder,
        visible_neuron_count,
        feature_neuron_count,
        hidden_neuron_count,
        activating_function,
        approximate_interface,
        learning_rate
    ):
        '''
        Initialize deep boltzmann machine.

        Args:
            dbm_builder:            `Concrete Builder` in Builder Pattern.
            visible_neuron_count:   The number of neurons in visible layer.
            feature_neuron_count:   The number of feature neurons in visible layer.
            hidden_neuron_count:    The number of neurons in hidden layer.
            activating_function:    Activation function.
            approximate_interface:  The object of function approximation.
            learning_rate:          Learning rate.
        '''
        dbm_builder.learning_rate = learning_rate
        dbm_director = DBMDirector(
            dbm_builder=dbm_builder
        )
        dbm_director.dbm_construct(
            neuron_assign_list=[visible_neuron_count, feature_neuron_count, hidden_neuron_count],
            activating_function=activating_function,
            approximate_interface=approximate_interface
        )

        self.__rbm_list = dbm_director.rbm_list

    @dispatch(DBMBuilder, list, ActivatingFunctionInterface, ApproximateInterface, float)
    def __init__(
        self,
        dbm_builder,
        neuron_assign_list,
        activating_function,
        approximate_interface,
        learning_rate
    ):
        '''
        Initialize deep boltzmann machine.

        Args:
            dbm_builder:            `Concrete Builder` in Builder Pattern.
            neuron_assign_list:     The number of neurons in each layers.
            activating_function:    Activation function.
            approximate_interface:  The object of function approximation.
            learning_rate:          Learning rate.
        '''
        dbm_builder.learning_rate = learning_rate
        dbm_director = DBMDirector(
            dbm_builder=dbm_builder
        )
        dbm_director.dbm_construct(
            neuron_assign_list=neuron_assign_list,
            activating_function=activating_function,
            approximate_interface=approximate_interface
        )
        self.__rbm_list = dbm_director.rbm_list

    def learn(
        self,
        observed_data_arr,
        traning_count=1000
    ):
        '''
        Learning.

        Args:
            observed_data_arr:      The `np.ndarray` of observed data points.
            traning_count:          Training counts.
        '''
        if isinstance(observed_data_arr, mx.ndarray.ndarray.NDArray) is False:
            raise TypeError()

        for i in range(len(self.__rbm_list)):
            rbm = self.__rbm_list[i]
            rbm.approximate_learning(observed_data_arr, traning_count)
            observed_data_arr = self.get_feature_point(i)

    def get_feature_point(self, layer_number=0):
        '''
        Extract the feature points.

        Args:
            layer_number:   The index of layers. 
                            For instance, 0 is visible layer, 1 is hidden or middle layer, and 2 is hidden layer in three layers.

        Returns:
            The list of feature points.
        '''
        feature_point_arr = self.__rbm_list[layer_number].graph.hidden_activity_arr
        return feature_point_arr

    def get_visible_activity(self):
        '''
        Extract activity of neurons in visible layer.

        Returns:
            Activity.
        '''
        visible_activity_arr = self.__rbm_list[0].graph.visible_activity_arr
        return visible_activity_arr
