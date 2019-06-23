# -*- coding: utf-8 -*-
import numpy as np
cimport numpy as np
import warnings
from pydbm.synapse_list import Synapse
from pydbm.approximation.interface.approximate_interface import ApproximateInterface


class RestrictedBoltzmannMachine(object):
    '''
    Restricted Boltzmann Machine.
    
    According to graph theory, the structure of RBM corresponds to 
    a complete bipartite graph which is a special kind of bipartite 
    graph where every node in the visible layer is connected to every 
    node in the hidden layer. Based on statistical mechanics and 
    thermodynamics(Ackley, D. H., Hinton, G. E., & Sejnowski, T. J. 1985), 
    the state of this structure can be reflected by the energy function.
    
    References:
        - Ackley, D. H., Hinton, G. E., & Sejnowski, T. J. (1985). A learning algorithm for Boltzmann machines. Cognitive science, 9(1), 147-169.
        - Hinton, G. E. (2002). Training products of experts by minimizing contrastive divergence. Neural computation, 14(8), 1771-1800.
        - Le Roux, N., & Bengio, Y. (2008). Representational power of restricted Boltzmann machines and deep belief networks. Neural computation, 20(6), 1631-1649.
    '''
    # Complete bipartite graph.
    __graph = None
    # Learning rate.
    __learning_rate = 0.5
    # Attenuate the `learning_rate` by a factor of this value every `attenuate_epoch`.
    __learning_attenuate_rate = 0.1
    # Attenuate the `learning_rate` by a factor of `learning_attenuate_rate` every `attenuate_epoch`.
    __attenuate_epoch = 50
    # The object of function approximation.
    __approximate_interface = None

    def get_graph(self):
        ''' getter of graph '''
        return self.__graph

    def set_read_only(self, value):
        ''' setter of graph '''
        raise TypeError("Read Only.")

    graph = property(get_graph, set_read_only)

    def __init__(
        self,
        graph,
        double learning_rate=0.005,
        double learning_attenuate_rate=0.1,
        int attenuate_epoch=50,
        dropout_rate=None,
        approximate_interface=None
    ):
        '''
        Initialize.

        Args:
            graph:                          Synapse.
            learning_rate:                  Learning rate.
            learning_attenuate_rate:        Attenuate the `learning_rate` by a factor of this value every `attenuate_epoch`.
            attenuate_epoch:                Attenuate the `learning_rate` by a factor of `learning_attenuate_rate` every `attenuate_epoch`.
            dropout_rate:                   Dropout rate.
            approximate_interface:          The object of function approximation.

        '''
        if isinstance(graph, Synapse) is False:
            raise TypeError("Synapse")

        if isinstance(approximate_interface, ApproximateInterface) is False:
            if approximate_interface is not None:
                raise TypeError("ApproximateInterface")

        if dropout_rate is not None:
            warnings.warn("`dropout_rate` will be removed in future version. Use `OptParams`.", FutureWarning)

        self.__graph = graph
        self.__learning_rate = learning_rate
        self.__learning_attenuate_rate = learning_attenuate_rate
        self.__attenuate_epoch = attenuate_epoch
        self.__approximate_interface = approximate_interface

    def approximate_learning(
        self,
        np.ndarray observed_data_arr,
        int traning_count=-1, 
        int batch_size=200,
        int training_count=1000
    ):
        '''
        Learning with function approximation.

        Args:
            observed_data_arr:      The array of observed data points.
            traning_count:          Training counts.
            batch_size:             Batch size.
        '''
        if traning_count != -1:
            training_count = traning_count
            warnings.warn("`traning_count` will be removed in future version. Use `training_count`.", FutureWarning)

        self.__graph = self.__approximate_interface.approximate_learn(
            self.__graph,
            self.__learning_rate,
            self.__learning_attenuate_rate,
            self.__attenuate_epoch,
            observed_data_arr,
            training_count=training_count,
            batch_size=batch_size
        )

    def approximate_inferencing(
        self,
        np.ndarray observed_data_arr,
        int traning_count=-1,
        int r_batch_size=-1,
        int training_count=1000
    ):
        '''
        Learning with function approximation.

        Args:
            observed_data_arr:      The array of observed data points.
            traning_count:          Training counts.
            r_batch_size:           Batch size.
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

        self.__graph = self.__approximate_interface.approximate_inference(
            self.__graph,
            self.__learning_rate,
            self.__learning_attenuate_rate,
            self.__attenuate_epoch,
            observed_data_arr,
            training_count=training_count,
            r_batch_size=r_batch_size
        )

    def get_reconstruct_error_list(self):
        '''
        Extract reconstruction error.

        Returns:
            The list.
        '''
        return self.__approximate_interface.reconstruct_error_list
