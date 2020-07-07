# -*- coding: utf-8 -*-
import numpy as np
import mxnet.ndarray as nd
import mxnet as mx
from mxnet import gluon
from mxnet import MXNetError

import warnings
from accelbrainbase.observabledata._mxnet.restricted_boltzmann_machines import RestrictedBoltzmannMachines
from accelbrainbase.iteratable_data import IteratableData
from accelbrainbase._mxnet._exception.init_deferred_error import InitDeferredError
from logging import getLogger


class DeepBoltzmannMachines(RestrictedBoltzmannMachines):
    '''
    Deep Boltzmann Machines(DBM).
    
    As is well known, DBM is composed of layers of RBMs 
    stacked on top of each other(Salakhutdinov, R., & Hinton, G. E. 2009). 
    This model is a structural expansion of Deep Belief Networks(DBN), 
    which is known as one of the earliest models of Deep Learning
    (Le Roux, N., & Bengio, Y. 2008). Like RBM, DBN places nodes in layers. 
    However, only the uppermost layer is composed of undirected edges, 
    and the other consists of directed edges.

    According to graph theory, the structure of RBM corresponds to 
    a complete bipartite graph which is a special kind of bipartite 
    graph where every node in the visible layer is connected to every 
    node in the hidden layer. Based on statistical mechanics and 
    thermodynamics(Ackley, D. H., Hinton, G. E., & Sejnowski, T. J. 1985), 
    the state of this structure can be reflected by the energy function.

    In relation to RBM, the Contrastive Divergence(CD) is a method for 
    approximation of the gradients of the log-likelihood(Hinton, G. E. 2002).
    This algorithm draws a distinction between a positive phase and a 
    negative phase. Conceptually, the positive phase is to the negative 
    phase what waking is to sleeping.

    The procedure of this method is similar to Markov Chain Monte Carlo method(MCMC).
    However, unlike MCMC, the visbile variables to be set first in visible layer is 
    not randomly initialized but the observed data points in training dataset are set 
    to the first visbile variables. And, like Gibbs sampler, drawing samples from hidden 
    variables and visible variables is repeated k times. Empirically (and surprisingly), 
    `k` is considered to be `1`.

    **Note** that this class does not support a *Hybrid* of imperative and symbolic 
    programming. Only `mxnet.ndarray` is supported.

    References:
        - Ackley, D. H., Hinton, G. E., & Sejnowski, T. J. (1985). A learning algorithm for Boltzmann machines. Cognitive science, 9(1), 147-169.
        - Hinton, G. E. (2002). Training products of experts by minimizing contrastive divergence. Neural computation, 14(8), 1771-1800.
        - Le Roux, N., & Bengio, Y. (2008). Representational power of restricted Boltzmann machines and deep belief networks. Neural computation, 20(6), 1631-1649.
        - Salakhutdinov, R., & Hinton, G. E. (2009). Deep boltzmann machines. InInternational conference on artificial intelligence and statistics (pp. 448-455).

    '''

    # `bool` that means initialization in this class will be deferred or not.
    __init_deferred_flag = False

    def __init__(
        self, 
        computable_loss,
        rbm_list,
        initializer=None,
        optimizer_name="SGD",
        learning_rate=0.005,
        learning_attenuate_rate=1.0,
        attenuate_epoch=50,
        ctx=mx.gpu(),
        **kwargs
    ):
        '''
        Init.
        
        Args:
            computable_loss:            is-a `ComputableLoss`.
            rbm_list:                   `list` of `RestrictedBoltzmannMachines`s.
            initializer:                is-a `mxnet.initializer` for parameters of model. If `None`, it is drawing from the Xavier distribution.
            optimizer_name:             `str` of name of optimizer.
            learning_rate:              `float` of learning rate.
            learning_attenuate_rate:    `float` of attenuate the `learning_rate` by a factor of this value every `attenuate_epoch`.
            attenuate_epoch:            `int` of attenuate the `learning_rate` by a factor of `learning_attenuate_rate` every `attenuate_epoch`.
            ctx:                        `mx.gpu()` or `mx.cpu()`.

        '''
        for rbm in rbm_list:
            if isinstance(rbm, RestrictedBoltzmannMachines) is False:
                raise TypeError("The type of values of `rbm_list` must be `RestrictedBoltzmannMachines`.")
        self.__rbm_list = rbm_list

        init_deferred_flag = self.init_deferred_flag
        self.init_deferred_flag = True
        super().__init__(
            computable_loss=computable_loss,
            initializer=initializer,
            optimizer_name=optimizer_name,
            learning_rate=learning_rate,
            learning_attenuate_rate=learning_attenuate_rate,
            attenuate_epoch=attenuate_epoch,
            ctx=ctx,
            **kwargs
        )
        self.init_deferred_flag = init_deferred_flag

        if initializer is None:
            self.initializer = mx.initializer.Xavier(
                rnd_type="gaussian", 
                factor_type="in", 
                magnitude=1
            )
        else:
            if isinstance(initializer, mx.initializer.Initializer) is False:
                raise TypeError("The type of `initializer` must be `mxnet.initializer.Initializer`.")

            self.initializer = initializer

        if self.init_deferred_flag is False:
            try:
                self.collect_params().initialize(self.initializer, force_reinit=True, ctx=ctx)
                self.trainer = gluon.Trainer(self.collect_params(), optimizer_name, {"learning_rate": learning_rate})
            except InitDeferredError:
                self.__logger.debug("The initialization should be deferred.")

        logger = getLogger("accelbrainbase")
        self.__logger = logger

    def collect_params(self, select=None):
        '''
        Overrided `collect_params` in `mxnet.gluon.HybridBlok`.
        '''
        params_dict = self.__rbm_list[0].collect_params(select)
        for i in range(1, len(self.__rbm_list)):
            params_dict.update(self.__rbm_list[i].collect_params(select))
        return params_dict

    def inference(self, observed_arr):
        '''
        Inference samples drawn by `IteratableData.generate_inferenced_samples()`.

        Args:
            observed_arr:   rank-2 Array like or sparse matrix as the observed data points.
                            The shape is: (batch size, feature points)

        Returns:
            `mxnet.ndarray` of inferenced feature points.
        '''
        self.batch_size = observed_arr.shape[0]
        for i in range(len(self.__rbm_list)):
            self.__rbm_list[i].batch_size = self.batch_size
            _ = self.__rbm_list[i].inference(observed_arr)
            observed_arr = self.__rbm_list[i].feature_points_arr

        return observed_arr

    def regularize(self):
        '''
        Regularization.
        '''
        for i in range(len(self.__rbm_list)):
            self.__rbm_list[i].regularize()

    def set_readonly(self, value):
        ''' setter '''
        raise TypeError("This is read-only.")

    def get_feature_points_arr(self):
        ''' getter for `mxnet.ndarray` of feature points.'''
        return self.__hidden_activity_arr

    feature_points_arr = property(get_feature_points_arr, set_readonly)

    def get_init_deferred_flag(self):
        ''' getter for `bool` that means initialization in this class will be deferred or not.'''
        return self.__init_deferred_flag
    
    def set_init_deferred_flag(self, value):
        ''' setter for `bool` that means initialization in this class will be deferred or not.'''
        self.__init_deferred_flag = value

    init_deferred_flag = property(get_init_deferred_flag, set_init_deferred_flag)

    def get_rbm_list(self):
        ''' getter '''
        return self.__rbm_list
    
    def set_rbm_list(self, value):
        ''' setter '''
        self.__rbm_list = value
    
    rbm_list = property(get_rbm_list, set_rbm_list)
