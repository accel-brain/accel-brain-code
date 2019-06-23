# -*- coding: utf-8 -*-
import numpy as np
cimport numpy as np
import warnings
from pydbm.dbm.interface.dbm_builder import DBMBuilder
from pydbm.dbm.dbm_director import DBMDirector
from pydbm.activation.interface.activating_function_interface import ActivatingFunctionInterface
from pydbm.approximation.interface.approximate_interface import ApproximateInterface
from pydbm.params_initializer import ParamsInitializer
ctypedef np.float64_t DOUBLE_t


class DeepBoltzmannMachine(object):
    '''
    The `Client` in Builder Pattern, to build Deep Boltzmann Machines.
    
    As is well known, DBM is composed of layers of RBMs 
    stacked on top of each other(Salakhutdinov, R., & Hinton, G. E. 2009). 
    This model is a structural expansion of Deep Belief Networks(DBN), 
    which is known as one of the earliest models of Deep Learning
    (Le Roux, N., & Bengio, Y. 2008). Like RBM, DBN places nodes in layers. 
    However, only the uppermost layer is composed of undirected edges, 
    and the other consists of directed edges.
    
    References:
        - https://github.com/chimera0/accel-brain-code/blob/master/Deep-Learning-by-means-of-Design-Pattern/demo/demo_stacked_auto_encoder.ipynb
        - Ackley, D. H., Hinton, G. E., & Sejnowski, T. J. (1985). A learning algorithm for Boltzmann machines. Cognitive science, 9(1), 147-169.
        - Hinton, G. E. (2002). Training products of experts by minimizing contrastive divergence. Neural computation, 14(8), 1771-1800.
        - Le Roux, N., & Bengio, Y. (2008). Representational power of restricted Boltzmann machines and deep belief networks. Neural computation, 20(6), 1631-1649.
        - Salakhutdinov, R., & Hinton, G. E. (2009). Deep boltzmann machines. InInternational conference on artificial intelligence and statistics (pp. 448-455).

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
        double learning_rate=1e-05,
        double learning_attenuate_rate=0.1,
        int attenuate_epoch=50,
        dropout_rate=None,
        inferencing_flag=True,
        inferencing_plan=None,
        scale=1.0,
        params_initializer=ParamsInitializer(),
        params_dict={"loc": 0.0, "scale": 1.0}
    ):
        '''
        Initialize deep boltzmann machine.

        Args:
            dbm_builder:                    `Concrete Builder` in Builder Pattern.
            neuron_assign_list:             The number of neurons in each layers.
            activating_function_list:       Activation function.
            approximate_interface_list:     The object of function approximation.
            learning_rate:                  Learning rate.
            learning_attenuate_rate:        Attenuate the `learning_rate` by a factor of this value every `attenuate_epoch`.
            attenuate_epoch:                Attenuate the `learning_rate` by a factor of `learning_attenuate_rate` every `attenuate_epoch`.
            inferencing_flag:               Execute inferencing or not. 
            inferencing_plan:               `each`:  Learn -> Inferece -> Learn -> ...
                                            `at_once`: All learn -> All inference

            scale:                          Scale of parameters which will be `ParamsInitializer`.
            params_initializer:             is-a `ParamsInitializer`.
            params_dict:                    `dict` of parameters other than `size` to be input to function `ParamsInitializer.sample_f`.

        '''
        if isinstance(params_initializer, ParamsInitializer) is False:
            raise TypeError("The type of `params_initializer` must be `ParamsInitializer`.")

        if dropout_rate is not None:
            warnings.warn("`dropout_rate` will be removed in future version. Use `OptParams`.", FutureWarning)

        if inferencing_plan is not None:
            warnings.warn("`inferencing_plan` will be removed in future version.", FutureWarning)
            
        dbm_builder.learning_rate = learning_rate
        dbm_builder.learning_attenuate_rate = learning_attenuate_rate
        dbm_builder.attenuate_epoch = attenuate_epoch
        dbm_director = DBMDirector(
            dbm_builder=dbm_builder
        )
        dbm_director.dbm_construct(
            neuron_assign_list=neuron_assign_list,
            activating_function_list=activating_function_list,
            approximate_interface_list=approximate_interface_list,
            scale=scale,
            params_initializer=params_initializer,
            params_dict=params_dict
        )
        self.__rbm_list = dbm_director.rbm_list

        if isinstance(inferencing_flag, bool):
            self.__inferencing_flag = inferencing_flag
        else:
            raise TypeError()

    def learn(
        self,
        np.ndarray[DOUBLE_t, ndim=2] observed_data_arr,
        int traning_count=-1,
        int batch_size=200,
        int r_batch_size=-1,
        sgd_flag=None,
        int training_count=1000
    ):
        '''
        Learning.

        Args:
            observed_data_arr:      The `np.ndarray` of observed data points.
            training_count:         Training counts.
            batch_size:             Batch size in learning.
            r_batch_size:           Batch size in inferencing.
                                    If this value is `0`, the inferencing is a recursive learning.
                                    If this value is more than `0`, the inferencing is a mini-batch recursive learning.
                                    If this value is '-1', the inferencing is not a recursive learning.

                                    If you do not want to execute the mini-batch training, 
                                    the value of `batch_size` must be `-1`. 
                                    And `r_batch_size` is also parameter to control the mini-batch training 
                                    but is refered only in inference and reconstruction. 
                                    If this value is more than `0`, 
                                    the inferencing is a kind of reccursive learning with the mini-batch training.
        '''
        if traning_count != -1:
            training_count = traning_count
            warnings.warn("`traning_count` will be removed in future version. Use `training_count`.", FutureWarning)

        if sgd_flag is not None:
            warnings.warn("`sgd_flag` will be removed in future version. All learning will be mini-batch training.", FutureWarning)

        cdef int i
        cdef np.ndarray[DOUBLE_t, ndim=2] data_arr
        cdef np.ndarray[DOUBLE_t, ndim=2] feature_point_arr

        if self.__inferencing_flag is False:
            data_arr = observed_data_arr.copy()
            for i in range(len(self.__rbm_list)):
                self.__rbm_list[i].approximate_learning(
                    data_arr,
                    training_count=training_count,
                    batch_size=batch_size
                )
                feature_point_arr = self.get_feature_point(i)
                data_arr = feature_point_arr
        else:
            data_arr = observed_data_arr.copy()
            for i in range(len(self.__rbm_list)):
                self.__rbm_list[i].approximate_learning(
                    data_arr,
                    training_count=training_count,
                    batch_size=batch_size
                )
                feature_point_arr = self.get_feature_point(i)
                data_arr = feature_point_arr

            rbm_list = self.__rbm_list[::-1]

            for i in range(len(rbm_list)):
                data_arr = self.get_feature_point(len(rbm_list)-1-i)
                rbm_list[i].approximate_inferencing(
                    data_arr,
                    training_count=training_count,
                    r_batch_size=r_batch_size
                )

    def get_feature_point(self, int layer_number=0):
        '''
        Extract the feature points.

        Args:
            layer_number:   The index of layers. 
                            For instance, `0` is visible layer, 
                            `1` is hidden or middle layer, 
                            and `2` is hidden layer in three layers.

        Returns:
            The np.ndarray of feature points.
        '''
        feature_point_arr = self.__rbm_list[layer_number].graph.hidden_activity_arr
        return feature_point_arr

    def get_visible_point(self, int layer_number=0):
        '''
        Extract the visible data points which is reconsturcted.

        Args:
            layer_number:       The index of layers.
                                For instance, `0` is visible layer, 
                                `1` is hidden or middle layer, 
                                and `2` is hidden layer in three layers.

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

    def save_pre_learned_params(self, dir_path=None, file_name=None):
        '''
        Save pre-learned parameters.
        
        Args:
            dir_path:   Path of dir. If `None`, the file is saved in the current directory.
            file_name:  The naming rule of files. If `None`, this value is `dbm`.
        '''
        file_path = ""
        if dir_path is not None:
            file_path += dir_path + "/"
        if file_name is not None:
            file_path += file_name
        else:
            file_path += "dbm"
        file_path += "_"
        
        for i in range(len(self.__rbm_list)):
            self.__rbm_list[i].graph.save_pre_learned_params(file_path + str(i) + ".npz")
