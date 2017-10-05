# -*- coding: utf-8 -*-
import pyximport
import numpy as np
pyximport.install(setup_args={'include_dirs':[np.get_include()]}, inplace=True)
cimport numpy
from multipledispatch import dispatch
from pydbm.dbm.interface.dbm_builder import DBMBuilder
from pydbm.dbm.dbm_director import DBMDirector
from pydbm.activation.interface.activating_function_interface import ActivatingFunctionInterface
from pydbm.approximation.interface.approximate_interface import ApproximateInterface


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
        int visible_neuron_count,
        int feature_neuron_count,
        int hidden_neuron_count,
        activating_function,
        approximate_interface,
        double learning_rate
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

        self.__hyper_param_dict = {
            "visible_neuron_count": visible_neuron_count,
            "feature_neuron_count": feature_neuron_count,
            "hidden_neuron_count": hidden_neuron_count,
            "learning_rate": learning_rate,
            "activating_function": str(type(activating_function)),
            "approximate_interface": str(type(approximate_interface))
        }

    @dispatch(DBMBuilder, list, ActivatingFunctionInterface, ApproximateInterface, float)
    def __init__(
        self,
        dbm_builder,
        neuron_assign_list,
        activating_function,
        approximate_interface,
        double learning_rate
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

        self.__hyper_param_dict = {
            "neuron_assign_list": neuron_assign_list,
            "learning_rate": learning_rate,
            "activating_function": str(type(activating_function)),
            "approximate_interface": str(type(approximate_interface))
        }

    def learn(
        self,
        numpy.ndarray observed_data_arr,
        int traning_count=1000
    ):
        '''
        Learning.

        Args:
            observed_data_arr:      The `np.ndarray` of observed data points.
            traning_count:          Training counts.
        '''
        if isinstance(observed_data_arr, np.ndarray) is False:
            raise TypeError()

        cdef int i
        for i in range(len(self.__rbm_list)):
            rbm = self.__rbm_list[i]
            rbm.approximate_learning(observed_data_arr, traning_count)

    def get_feature_point_list(self, int layer_number=0):
        '''
        Extract the feature points.

        Args:
            layer_number:   The index of layers. 
                            For instance, 0 is visible layer, 1 is hidden or middle layer, and 2 is hidden layer in three layers.

        Returns:
            The list of feature points.
        '''
        rbm = self.__rbm_list[layer_number]
        cdef int j
        feature_point_list = [rbm.graph.hidden_neuron_arr[j].activity for j in range(len(rbm.graph.hidden_neuron_arr))]
        feature_point_arr = np.array(feature_point_list)
        return feature_point_arr

    def get_visible_activity(self):
        '''
        Extract activity of neurons in visible layer.

        Returns:
            Activity.
        '''
        rbm = self.__rbm_list[0]
        cdef int i
        visible_activity_list = [rbm.graph.visible_neuron_arr[i].activity for i in range(len(rbm.graph.visible_neuron_arr))]
        return np.array(visible_activity_list)
