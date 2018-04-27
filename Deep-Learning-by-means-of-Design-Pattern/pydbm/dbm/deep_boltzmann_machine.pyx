# -*- coding: utf-8 -*-
import numpy as np
cimport numpy as np
from pydbm.dbm.interface.dbm_builder import DBMBuilder
from pydbm.dbm.dbm_director import DBMDirector
from pydbm.activation.interface.activating_function_interface import ActivatingFunctionInterface
from pydbm.approximation.interface.approximate_interface import ApproximateInterface
ctypedef np.float64_t DOUBLE_t


class DeepBoltzmannMachine(object):
    '''
    The `Client` in Builder Pattern,
    
    Build deep boltzmann machine.
    '''

    # The list of restricted boltzmann machines.
    __rbm_list = []
    
    def get_rbm_list(self):
        return self.__rbm_list
    
    def set_rbm_list(self, value):
        if isinstance(value, list):
            self.__rbm_list = value
        else:
            raise TypeError()

    rbm_list = property(get_rbm_list, set_rbm_list)

    # The dict of Hyper parameters.
    __hyper_param_dict = {}

    # Execute inferencing or not.
    __inferencing_flag = False
    # Inferencing plan. (`each` or `at_once`)
    __inferencing_plan = "each"

    def __init__(
        self,
        dbm_builder,
        neuron_assign_list,
        activating_function_list,
        approximate_interface_list,
        double learning_rate,
        double dropout_rate=0.5,
        inferencing_flag=True,
        inferencing_plan="each"
    ):
        '''
        Initialize deep boltzmann machine.

        Args:
            dbm_builder:            `    Concrete Builder` in Builder Pattern.
            neuron_assign_list:          The number of neurons in each layers.
            activating_function_list:    Activation function.
            approximate_interface_list:  The object of function approximation.
            learning_rate:               Learning rate.
            dropout_rate:                Dropout rate.
            inferencing_flag:            Execute inferencing or not. 
            inferencing_plan:            `each`:  Learn -> Inferece -> Learn -> ...
                                         `at_once`: All learn -> All inference   
        '''
        dbm_builder.learning_rate = learning_rate
        dbm_builder.dropout_rate = dropout_rate
        dbm_director = DBMDirector(
            dbm_builder=dbm_builder
        )
        dbm_director.dbm_construct(
            neuron_assign_list=neuron_assign_list,
            activating_function_list=activating_function_list,
            approximate_interface_list=approximate_interface_list
        )
        self.__rbm_list = dbm_director.rbm_list

        if isinstance(inferencing_flag, bool):
            self.__inferencing_flag = inferencing_flag
        else:
            raise TypeError()

        if isinstance(inferencing_plan, str):
            self.__inferencing_plan = inferencing_plan
        else:
            raise TypeError()

    def learn(
        self,
        np.ndarray[DOUBLE_t, ndim=2] observed_data_arr,
        int traning_count=1000,
        int batch_size=200,
        int r_batch_size=-1,
        sgd_flag=False
    ):
        '''
        Learning.

        Args:
            observed_data_arr:    The `np.ndarray` of observed data points.
            traning_count:        Training counts.
            batch_size:           Batch size.
            r_batch_size:         Batch size.
                                  If this value is `0`, the inferencing is a recursive learning.
                                  If this value is more than `0`, the inferencing is a mini-batch recursive learning.
                                  If this value is '-1', the inferencing is not a recursive learning.
            sgd_flag:             Learning with the stochastic gradient descent(SGD) or not.
        '''
        cdef int i
        cdef int row_i = observed_data_arr.shape[0]
        cdef int j
        cdef np.ndarray[DOUBLE_t, ndim=1] data_arr
        cdef np.ndarray[DOUBLE_t, ndim=1] feature_point_arr
        cdef int sgd_key

        if self.__inferencing_flag is False:
            for i in range(row_i):
                if sgd_flag is True:
                    sgd_key = np.random.randint(row_i)
                    data_arr = observed_data_arr[sgd_key]
                else:
                    data_arr = observed_data_arr[i].copy()

                for j in range(len(self.__rbm_list)):
                    self.__rbm_list[j].approximate_learning(
                        data_arr,
                        traning_count,
                        batch_size
                    )
                    feature_point_arr = self.get_feature_point(j)
                    data_arr = feature_point_arr
        else:
            if self.__inferencing_plan == "each":
                for i in range(row_i):
                    if sgd_flag is True:
                        sgd_key = np.random.randint(row_i)
                        data_arr = observed_data_arr[sgd_key]
                    else:
                        data_arr = observed_data_arr[i].copy()

                    for j in range(len(self.__rbm_list)):
                        self.__rbm_list[j].approximate_learning(
                            data_arr,
                            traning_count,
                            batch_size
                        )
                        feature_point_arr = self.get_feature_point(j)
                        data_arr = feature_point_arr

                    rbm_list = self.__rbm_list[::-1]

                    for j in range(len(rbm_list)):
                        data_arr = self.get_feature_point(len(rbm_list)-1-j)
                        rbm_list[j].approximate_inferencing(
                            data_arr,
                            traning_count,
                            r_batch_size
                        )

            elif self.__inferencing_plan == "at_once":
                for i in range(row_i):
                    if sgd_flag is True:
                        sgd_key = np.random.randint(row_i)
                        data_arr = observed_data_arr[sgd_key]
                    else:
                        data_arr = observed_data_arr[i].copy()

                    for j in range(len(self.__rbm_list)):
                        self.__rbm_list[j].approximate_learning(
                            data_arr,
                            traning_count,
                            batch_size
                        )
                        feature_point_arr = self.get_feature_point(j)
                        data_arr = feature_point_arr

                rbm_list = self.__rbm_list[::-1]
                for j in range(len(rbm_list)):
                    data_arr = self.get_feature_point(len(rbm_list)-1)

                    rbm_list[j].approximate_inferencing(
                        data_arr,
                        traning_count,
                        r_batch_size
                    )

    def get_feature_point(self, int layer_number=0):
        '''
        Extract the feature points.

        Args:
            layer_number:   The index of layers. 
                            For instance, 0 is visible layer, 1 is hidden or middle layer, and 2 is hidden layer in three layers.

        Returns:
            The np.ndarray of feature points.
        '''
        feature_point_arr = self.__rbm_list[layer_number].graph.hidden_activity_arr
        return feature_point_arr

    def get_visible_point(self, int layer_number=0):
        '''
        Extract the visible data points which is reconsturcted.

        Args:
            layer_number:    The index of layers.
                             For instance, 0 is visible layer, 1 is hidden or middle layer, and 2 is hidden layer in three layers.

        Returns:
            The np.ndarray of visible data points.
        '''
        visible_points_arr = self.__rbm_list[layer_number].graph.visible_activity_arr
        return visible_points_arr

    def get_visible_activity_arr_list(self):
        '''
        Extract activity of neurons in each visible layers.

        Returns:
            Activity.
        '''
        visible_activity_arr_list = [self.__rbm_list[i].graph.visible_activity_arr for i in range(len(self.__rbm_list))]
        return visible_activity_arr_list

    def get_hidden_activity_arr_list(self):
        '''
        Extract activity of neurons in each hidden layers.

        Returns:
            Activity.
        '''
        hidden_activity_arr_list = [self.__rbm_list[i].graph.hidden_activity_arr for i in range(len(self.__rbm_list))]
        return hidden_activity_arr_list

    def get_visible_bias_arr_list(self):
        '''
        Extract bias in each visible layers.

        Returns:
            Bias.
        '''
        visible_bias_arr_list = [self.__rbm_list[i].graph.visible_bias_arr for i in range(len(self.__rbm_list))]
        return visible_bias_arr_list

    def get_hidden_bias_arr_list(self):
        '''
        Extract bias in each hidden layers.

        Returns:
            Bias.
        '''
        hidden_bias_arr_list = [self.__rbm_list[i].graph.hidden_bias_arr for i in range(len(self.__rbm_list))]
        return hidden_bias_arr_list

    def get_weight_arr_list(self):
        '''
        Extract weights of each links.

        Returns:
            The list of weights.
        '''
        weight_arr_list = [self.__rbm_list[i].graph.weights_arr for i in range(len(self.__rbm_list))]
        return weight_arr_list

    def get_reconstruct_error_arr(self, int layer_number=0):
        '''
        Extract reconsturction error rate.

        Returns:
            The np.ndarray.
        '''
        return np.array(self.__rbm_list[layer_number].get_reconstruct_error_list())
