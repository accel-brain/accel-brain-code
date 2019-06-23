# -*- coding: utf-8 -*-
from pydbm.dbm.interface.rt_rbm_builder import RTRBMBuilder
from pydbm.approximation.interface.approximate_interface import ApproximateInterface
from pydbm.activation.interface.activating_function_interface import ActivatingFunctionInterface
from pydbm.params_initializer import ParamsInitializer


class RTRBMDirector(object):
    '''
    The `Director` in Builder Pattern.

    Compose RTRBM, RNN-RBM, or LSTM-RTRBM for building a object of 
    restricted boltzmann machine.
    
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

    # `Builder` in Builder Pattern.
    __rtrbm_builder = None
    # The restricted boltzmann machines.
    __rbm = None
    
    def get_rbm(self):
        ''' getter '''
        return self.__rbm

    def set_rbm(self, value):
        ''' setter '''
        self.__rbm = value
    
    rbm = property(get_rbm, set_rbm)
    
    def __init__(self, rtrbm_builder):
        '''
        Initialize `Builder` in Builder Pattern.

        Args:
            rtrbm_builder     `Concreat Builder` in Builder Pattern
        '''
        if isinstance(rtrbm_builder, RTRBMBuilder):
            self.__rtrbm_builder = rtrbm_builder
        else:
            raise TypeError()

        self.__rbm = None

    def rtrbm_construct(
        self,
        visible_num,
        hidden_num,
        visible_activating_function,
        hidden_activating_function,
        rnn_activating_function,
        approximate_interface,
        learning_rate=1e-05,
        learning_attenuate_rate=0.1,
        attenuate_epoch=50,
        scale=1.0,
        params_initializer=ParamsInitializer(),
        params_dict={"loc": 0.0, "scale": 1.0}
    ):
        '''
        Build deep boltzmann machine.

        Args:
            visible_num:                    The number of units in visible layer.
            hidden_num:                     The number of units in hidden layer.
            visible_activating_function:    The activation function in visible layer.
            hidden_activating_function:     The activation function in hidden layer.
            approximate_interface:          The function approximation.
            learning_rate:                  Learning rate.
            learning_attenuate_rate:        Attenuate the `learning_rate` by a factor of this value every `attenuate_epoch`.
            attenuate_epoch:                Attenuate the `learning_rate` by a factor of `learning_attenuate_rate` every `attenuate_epoch`.
            scale:                          Scale of parameters which will be `ParamsInitializer`.
            params_initializer:             is-a `ParamsInitializer`.
            params_dict:                    `dict` of parameters other than `size` to be input to function `ParamsInitializer.sample_f`.

        '''
        if isinstance(visible_activating_function, ActivatingFunctionInterface) is False:
            raise TypeError()
        if isinstance(hidden_activating_function, ActivatingFunctionInterface) is False:
            raise TypeError()
        if isinstance(rnn_activating_function, ActivatingFunctionInterface) is False:
            raise TypeError()
        if isinstance(approximate_interface, ApproximateInterface) is False:
            raise TypeError()

        # Learning rate.
        self.__rtrbm_builder.learning_rate = learning_rate
        # Attenuate the `learning_rate` by a factor of this value every `attenuate_epoch`.
        self.__rtrbm_builder.learning_attenuate_rate = learning_attenuate_rate
        # Attenuate the `learning_rate` by a factor of `learning_attenuate_rate` every `attenuate_epoch`.
        self.__rtrbm_builder.attenuate_epoch = attenuate_epoch
        # Set units in visible layer.
        self.__rtrbm_builder.visible_neuron_part(
            visible_activating_function,
            visible_num
        )
        # Set units in hidden layer.
        self.__rtrbm_builder.hidden_neuron_part(
            hidden_activating_function,
            hidden_num
        )
        # Set units in RNN layer.
        self.__rtrbm_builder.rnn_neuron_part(rnn_activating_function)
        # Set graph and approximation function, delegating `SGD` which is-a `OptParams`.
        self.__rtrbm_builder.graph_part(
            approximate_interface,
            scale=scale,
            params_initializer=params_initializer,
            params_dict=params_dict
        )
        # Building.
        self.__rbm = self.__rtrbm_builder.get_result()
