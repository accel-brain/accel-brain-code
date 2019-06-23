# -*- coding: utf-8 -*-
import numpy as np
cimport numpy as np
from pydbm.dbm.interface.dbm_builder import DBMBuilder
from pydbm.approximation.contrastive_divergence import ContrastiveDivergence
from pydbm.synapse.complete_bipartite_graph import CompleteBipartiteGraph
from pydbm.dbm.restricted_boltzmann_machines import RestrictedBoltzmannMachine
from pydbm.params_initializer import ParamsInitializer
import warnings


class DBMMultiLayerBuilder(DBMBuilder):
    '''
    `Concrete Builder` in Builder Pattern.

    Compose three restricted boltzmann machines for building a deep boltzmann machine.

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
    # The list of neurons in visible layer.
    __visible_neuron_list = []
    # The list of neurons for feature points in `virtual` visible layer. 
    __feature_point_neuron = []
    # the list of neurons in hidden layer.
    __hidden_neuron_list = []
    # Complete bipartite graph
    __graph_list = []
    # The list of restricted boltzmann machines.
    __rbm_list = []
    # Learning rate.
    __learning_rate = 0.5
    # Attenuate the `learning_rate` by a factor of this value every `attenuate_epoch`.
    __learning_attenuate_rate = 0.1
    # Attenuate the `learning_rate` by a factor of `learning_attenuate_rate` every `attenuate_epoch`.
    __attenuate_epoch = 50

    def get_learning_rate(self):
        ''' getter '''
        if isinstance(self.__learning_rate, float) is False:
            raise TypeError()
        return self.__learning_rate

    def set_learning_rate(self, value):
        ''' setter '''
        if isinstance(value, float) is False:
            raise TypeError()
        self.__learning_rate = value

    learning_rate = property(get_learning_rate, set_learning_rate)

    def get_learning_attenuate_rate(self):
        ''' getter '''
        return self.__learning_attenuate_rate
    
    def set_learning_attenuate_rate(self, value):
        ''' setter '''
        if isinstance(value, float) is False:
            raise TypeError()
        self.__learning_attenuate_rate = value

    learning_attenuate_rate = property(get_learning_attenuate_rate, set_learning_attenuate_rate)

    def get_attenuate_epoch(self):
        ''' getter '''
        return self.__attenuate_epoch
    
    def set_attenuate_epoch(self, value):
        ''' setter '''
        if isinstance(value, int) is False:
            raise TypeError()
        self.__attenuate_epoch = value

    attenuate_epoch = property(get_attenuate_epoch, set_attenuate_epoch)

    def __init__(
        self,
        pre_learned_path_list=[],
        weights_arr_list=np.array([]),
        visible_bias_arr_list=np.array([]),
        hidden_bias_arr_list=np.array([])
    ):
        '''
        Initialize.
        
        Args:
            pre_learned_path_list:  `list` of file path that stores pre-learned parameters.
            weights_arr_list:       This will be removed in future version. Use `pre_learned_path_list`.
            visible_bias_arr_list:  This will be removed in future version. Use `pre_learned_path_list`.
            hidden_bias_arr_list:   This will be removed in future version. Use `pre_learned_path_list`.

        '''
        self.__visible_neuron_list = []
        self.__feature_point_neuron = []
        self.__hidden_neuron_list = []
        self.__graph_list = []
        self.__rbm_list = []

        if len(weights_arr_list) > 0:
            warnings.warn("`weights_arr_list` will be removed in future version. Use `pre_learned_path_list`.", FutureWarning)
        if len(visible_bias_arr_list) > 0:
            warnings.warn("`visible_bias_arr_list` will be removed in future version. Use `pre_learned_path_list`.", FutureWarning)
        if len(hidden_bias_arr_list) > 0:
            warnings.warn("`hidden_bias_arr_list` will be removed in future version. Use `pre_learned_path_list`.", FutureWarning)

        self.__weights_arr_list = weights_arr_list
        self.__visible_bias_arr_list = visible_bias_arr_list
        self.__hidden_bias_arr_list = hidden_bias_arr_list
        
        self.__pre_learned_path_list = pre_learned_path_list

    def visible_neuron_part(self, activating_function, int neuron_count):
        '''
        Build neurons in visible layer.

        Args:
            activating_function:    Activation function.
            neuron_count:           The number of neurons.
        '''
        self.__visible_activating_function = activating_function
        self.__visible_neuron_count = neuron_count

    def feature_neuron_part(self, activating_function_list, neuron_count_list):
        '''
        Build neurons for feature points in `virtual` visible layer.

        Build neurons in `n` layers.

        For associating with `n-1` layers, the object activate as neurons in hidden layer.
        On the other hand, for associating with `n+1` layers, the object activate as neurons in `virtual` visible layer.

        Args:
            activating_function_list:    The list of activation function.
            neuron_count_list:           The list of the number of neurons.
        '''
        self.__feature_activating_function_list = activating_function_list
        self.__feature_point_count_list = neuron_count_list

    def hidden_neuron_part(self, activating_function, int neuron_count):
        '''
        Build neurons in hidden layer.

        Args:
            activating_function:    Activation function
            neuron_count:           The number of neurons.
        '''
        self.__hidden_activating_function = activating_function
        self.__hidden_neuron_list = neuron_count

    def graph_part(
        self, 
        approximate_interface_list,
        scale=1.0,
        params_initializer=ParamsInitializer(),
        params_dict={"loc": 0.0, "scale": 1.0}
    ):
        '''
        Build complete bipartite graph.

        Args:
            approximate_interface_list:     The list of function approximation.
            scale:                          Scale of parameters which will be `ParamsInitializer`.
            params_initializer:             is-a `ParamsInitializer`.
            params_dict:                    `dict` of parameters other than `size` to be input to function `ParamsInitializer.sample_f`.
        '''
        self.__approximate_interface_list = approximate_interface_list

        complete_bipartite_graph = CompleteBipartiteGraph()

        if len(self.__pre_learned_path_list):
            complete_bipartite_graph.load_pre_learned_params(self.__pre_learned_path_list[0])
            complete_bipartite_graph.visible_activating_function = self.__visible_activating_function
            complete_bipartite_graph.hidden_activating_function = self.__feature_activating_function_list[0]

        elif len(self.__weights_arr_list):
            complete_bipartite_graph.create_node(
                self.__visible_neuron_count,
                self.__feature_point_count_list[0],
                self.__visible_activating_function,
                self.__feature_activating_function_list[0],
                self.__weights_arr_list[0],
                scale=scale,
                params_initializer=params_initializer,
                params_dict=params_dict
            )
        else:
            complete_bipartite_graph.create_node(
                self.__visible_neuron_count,
                self.__feature_point_count_list[0],
                self.__visible_activating_function,
                self.__feature_activating_function_list[0],
                scale=scale,
                params_initializer=params_initializer,
                params_dict=params_dict
            )

        if len(self.__visible_bias_arr_list):
            complete_bipartite_graph.visible_bias_arr = self.__visible_bias_arr_list[0]
        if len(self.__hidden_bias_arr_list):
            complete_bipartite_graph.hidden_bias_arr = self.__hidden_bias_arr_list[0]
        self.__graph_list.append(complete_bipartite_graph)

        cdef int i
        for i in range(1, len(self.__feature_point_count_list)):
            complete_bipartite_graph = CompleteBipartiteGraph()

            if len(self.__pre_learned_path_list):
                complete_bipartite_graph.load_pre_learned_params(self.__pre_learned_path_list[i])
                complete_bipartite_graph.visible_activating_function = self.__feature_activating_function_list[i - 1]
                complete_bipartite_graph.hidden_activating_function = self.__feature_activating_function_list[i]

            elif len(self.__weights_arr_list):
                complete_bipartite_graph.create_node(
                    self.__feature_point_count_list[i - 1],
                    self.__feature_point_count_list[i],
                    self.__feature_activating_function_list[i - 1],
                    self.__feature_activating_function_list[i],
                    self.__weights_arr_list[i],
                    scale=scale,
                    params_initializer=params_initializer,
                    params_dict=params_dict
                )
            else:
                complete_bipartite_graph.create_node(
                    self.__feature_point_count_list[i - 1],
                    self.__feature_point_count_list[i],
                    self.__feature_activating_function_list[i - 1],
                    self.__feature_activating_function_list[i],
                    scale=scale,
                    params_initializer=params_initializer,
                    params_dict=params_dict
                )

            if len(self.__visible_bias_arr_list):
                complete_bipartite_graph.visible_bias_arr = self.__visible_bias_arr_list[i]
            if len(self.__hidden_bias_arr_list):
                complete_bipartite_graph.hidden_bias_arr = self.__hidden_bias_arr_list[i]
            self.__graph_list.append(complete_bipartite_graph)

        complete_bipartite_graph = CompleteBipartiteGraph()

        if len(self.__pre_learned_path_list):
            complete_bipartite_graph.load_pre_learned_params(self.__pre_learned_path_list[-1])
            complete_bipartite_graph.visible_activating_function = self.__feature_activating_function_list[-1]
            complete_bipartite_graph.hidden_activating_function = self.__hidden_activating_function

        if len(self.__weights_arr_list):
            complete_bipartite_graph.create_node(
                self.__feature_point_count_list[-1],
                self.__hidden_neuron_list,
                self.__feature_activating_function_list[-1],
                self.__hidden_activating_function,
                self.__weights_arr_list[-1],
                scale=scale,
                params_initializer=params_initializer,
                params_dict=params_dict
            )
        else:
            complete_bipartite_graph.create_node(
                self.__feature_point_count_list[-1],
                self.__hidden_neuron_list,
                self.__feature_activating_function_list[-1],
                self.__hidden_activating_function,
                scale=scale,
                params_initializer=params_initializer,
                params_dict=params_dict
            )

        if len(self.__visible_bias_arr_list):
            complete_bipartite_graph.visible_bias_arr = self.__visible_bias_arr_list[-1]
        if len(self.__hidden_bias_arr_list):
            complete_bipartite_graph.hidden_bias_arr = self.__hidden_bias_arr_list[-1]
        self.__graph_list.append(complete_bipartite_graph)

    def get_result(self):
        '''
        Return builded restricted boltzmann machines.

        Returns:
            The list of restricted boltzmann machines.

        '''
        for i in range(len(self.__graph_list)):
            graph = self.__graph_list[i]
            rbm = RestrictedBoltzmannMachine(
                graph,
                learning_rate=self.__learning_rate,
                learning_attenuate_rate=self.__learning_attenuate_rate,
                attenuate_epoch=self.__attenuate_epoch,
                approximate_interface=self.__approximate_interface_list[i]
            )
            self.__rbm_list.append(rbm)

        return self.__rbm_list
