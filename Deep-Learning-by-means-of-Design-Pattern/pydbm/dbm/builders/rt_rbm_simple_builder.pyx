# -*- coding: utf-8 -*-
import numpy as np
cimport numpy as np
from pydbm.dbm.interface.rt_rbm_builder import RTRBMBuilder
from pydbm.synapse.recurrent_temporal_graph import RecurrentTemporalGraph
from pydbm.dbm.restrictedboltzmannmachines.rt_rbm import RTRBM
from pydbm.params_initializer import ParamsInitializer


class RTRBMSimpleBuilder(RTRBMBuilder):
    '''
    `Concrete Builder` in Builder Pattern.

    Compose restricted boltzmann machines for building a RTRBM.

    The RTRBM (Sutskever, I., et al. 2009) is a probabilistic 
    time-series model which can be viewed as a temporal stack of RBMs, 
    where each RBM has a contextual hidden state that is received 
    from the previous RBM and is used to modulate its hidden units bias.
    
    References:
        - Boulanger-Lewandowski, N., Bengio, Y., & Vincent, P. (2012). Modeling temporal dependencies in high-dimensional sequences: Application to polyphonic music generation and transcription. arXiv preprint arXiv:1206.6392.
        - Lyu, Q., Wu, Z., Zhu, J., & Meng, H. (2015, June). Modelling High-Dimensional Sequences with LSTM-RTRBM: Application to Polyphonic Music Generation. In IJCAI (pp. 4138-4139).
        - Lyu, Q., Wu, Z., & Zhu, J. (2015, October). Polyphonic music modelling with LSTM-RTRBM. In Proceedings of the 23rd ACM international conference on Multimedia (pp. 991-994). ACM.
        - Sutskever, I., Hinton, G. E., & Taylor, G. W. (2009). The recurrent temporal restricted boltzmann machine. In Advances in Neural Information Processing Systems (pp. 1601-1608).

    '''
    # The list of neurons in visible layer.
    __visible_neuron_count = 10
    # the list of neurons in hidden layer.
    __hidden_neuron_count = 10
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

    def __init__(self, pre_learned_path=None):
        '''
        Init.
        
        Args:
            pre_learned_path:  File path that stores pre-learned parameters.

        '''
        self.__pre_learned_path = pre_learned_path

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

    def visible_neuron_part(self, activating_function, int neuron_count):
        '''
        Build neurons in visible layer.

        Args:
            activating_function:    Activation function.
            neuron_count:           The number of neurons.
        '''
        self.__visible_activating_function = activating_function
        self.__visible_neuron_count = neuron_count

    def hidden_neuron_part(self, activating_function, int neuron_count):
        '''
        Build neurons in hidden layer.

        Args:
            activating_function:    Activation function
            neuron_count:           The number of neurons.
        '''
        self.__hidden_activating_function = activating_function
        self.__hidden_neuron_count = neuron_count

    def rnn_neuron_part(self, rnn_activating_function):
        '''
        Build neurons for RNN.

        Args:
            rnn_activating_function:    Activation function
        '''
        self.__rnn_activating_function = rnn_activating_function

    def graph_part(
        self, 
        approximate_interface,
        scale=1.0,
        params_initializer=ParamsInitializer(),
        params_dict={"loc": 0.0, "scale": 1.0}
    ):
        '''
        Build RTRBM graph.

        Args:
            approximate_interface:          The function approximation.
            scale:                          Scale of parameters which will be `ParamsInitializer`.
            params_initializer:             is-a `ParamsInitializer`.
            params_dict:                    `dict` of parameters other than `size` to be input to function `ParamsInitializer.sample_f`.

        '''
        self.__approximate_interface = approximate_interface
        self.__rt_graph = RecurrentTemporalGraph()
        self.__rt_graph.rnn_activating_function = self.__rnn_activating_function
        if self.__pre_learned_path is None:
            self.__rt_graph.create_node(
                self.__visible_neuron_count,
                self.__hidden_neuron_count,
                self.__visible_activating_function,
                self.__hidden_activating_function,
                scale=scale,
                params_initializer=params_initializer,
                params_dict=params_dict
            )
        else:
            self.__rt_graph.load_pre_learned_params(self.__pre_learned_path)
            self.__rt_graph.visible_activating_function = self.__visible_activating_function
            self.__rt_graph.hidden_activating_function = self.__hidden_activating_function

    def get_result(self):
        '''
        Return builded restricted boltzmann machines.

        Returns:
            The list of restricted boltzmann machines.

        '''
        rbm = RTRBM(
            self.__rt_graph,
            learning_rate=self.__learning_rate,
            learning_attenuate_rate=self.__learning_attenuate_rate,
            attenuate_epoch=self.__attenuate_epoch,
            approximate_interface=self.__approximate_interface
        )
        return rbm
