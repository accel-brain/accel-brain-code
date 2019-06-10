# -*- coding: utf-8 -*-
from pydbm.dbm.interface.dbm_builder import DBMBuilder
from pydbm.dbm.restricted_boltzmann_machines import RestrictedBoltzmannMachine
from pydbm.activation.interface.activating_function_interface import ActivatingFunctionInterface
from pydbm.approximation.interface.approximate_interface import ApproximateInterface
from pydbm.params_initializer import ParamsInitializer


class DBMDirector(object):
    '''
    The `Director` in Builder Pattern.
    
    Compose restricted boltzmann machines for building a object of deep boltzmann machine.

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

    # `Builder` in Builder Pattern.
    __dbm_builder = None
    # The list of restricted boltzmann machines.
    __rbm_list = []

    def get_rbm_list(self):
        ''' getter '''
        if isinstance(self.__rbm_list, list) is False:
            raise TypeError()

        for rbm in self.__rbm_list:
            if isinstance(rbm, RestrictedBoltzmannMachine) is False:
                raise TypeError()

        return self.__rbm_list

    def set_rbm_list(self, value):
        ''' setter '''
        if isinstance(value, list) is False:
            raise TypeError()

        for rbm in value:
            if isinstance(rbm, RestrictedBoltzmannMachine) is False:
                raise TypeError()

        self.__rbm_list = value

    rbm_list = property(get_rbm_list, set_rbm_list)

    def __init__(self, dbm_builder):
        '''
        Initialize `Builder` in Builder Pattern.

        Args:
            dbm_builder     `Concreat Builder` in Builder Pattern
        '''
        if isinstance(dbm_builder, DBMBuilder) is False:
            raise TypeError()

        self.__dbm_builder = dbm_builder

    def dbm_construct(
        self,
        neuron_assign_list,
        activating_function_list,
        approximate_interface_list,
        scale=1.0,
        params_initializer=ParamsInitializer(),
        params_dict={"loc": 0.0, "scale": 1.0}
    ):
        '''
        Build deep boltzmann machine.

        Args:
            neuron_assign_list:             The unit of neurons in each layers.
            activating_function_list:       The list of activation function,
            approximate_interface_list:     The list of function approximation.
            scale:                          Scale of parameters which will be `ParamsInitializer`.
            params_initializer:             is-a `ParamsInitializer`.
            params_dict:                    `dict` of parameters other than `size` to be input to function `ParamsInitializer.sample_f`.

        '''
        if isinstance(params_initializer, ParamsInitializer) is False:
            raise TypeError("The type of `params_initializer` must be `ParamsInitializer`.")

        for i in range(len(activating_function_list)):
            if isinstance(activating_function_list[i], ActivatingFunctionInterface) is False:
                raise TypeError()

        for i in range(len(approximate_interface_list)):
            if isinstance(approximate_interface_list[i], ApproximateInterface) is False:
                raise TypeError()

        visible_neuron_count = neuron_assign_list[0]
        visible_activating_function = activating_function_list[0]
        self.__dbm_builder.visible_neuron_part(visible_activating_function, visible_neuron_count)

        feature_neuron_count_list = neuron_assign_list[1:len(neuron_assign_list) - 1]
        feature_activating_function_list = activating_function_list[1:len(activating_function_list) - 1]
        self.__dbm_builder.feature_neuron_part(feature_activating_function_list, feature_neuron_count_list)

        hidden_neuron_count = neuron_assign_list[-1]
        hidden_acitivating_function = activating_function_list[-1]
        self.__dbm_builder.hidden_neuron_part(hidden_acitivating_function, hidden_neuron_count)

        self.__dbm_builder.graph_part(
            approximate_interface_list,
            scale=scale,
            params_initializer=params_initializer,
            params_dict=params_dict
        )
        self.rbm_list = self.__dbm_builder.get_result()
