# -*- coding: utf-8 -*-
from accelbrainbase.observabledata._mxnet.adversarialmodel.generative_model import GenerativeModel
from accelbrainbase.samplabledata.noise_sampler import NoiseSampler
from accelbrainbase._mxnet._exception.init_deferred_error import InitDeferredError
from accelbrainbase.samplabledata.condition_sampler import ConditionSampler

from mxnet import gluon
from mxnet import autograd
import numpy as np
import mxnet as mx
import mxnet.ndarray as nd
from mxnet import MXNetError
from logging import getLogger


class TransposedClusteringModel(GenerativeModel):
    '''
    Transposed Clasterer in the ClusterGAN.

    This is the beta version.

    Clasterer is a Generative model, which draws samples from 
    the fake distribution, and consturct a block diagnoal constraint 
    by self-paced learning algorithm of ClusterGAN.

    References:
        - Ghasedi, K., Wang, X., Deng, C., & Huang, H. (2019). Balanced self-paced learning for generative adversarial clustering network. In Proceedings of the IEEE Conference on Computer Vision and Pattern Recognition (pp. 4391-4400).

    '''
    # `bool` that means initialization in this class will be deferred or not.
    __init_deferred_flag = False

    def __init__(
        self, 
        model, 
        initializer=None,
        condition_sampler=None,
        learning_rate=1e-05,
        optimizer_name="SGD",
        hybridize_flag=True,
        scale=1.0, 
        ctx=mx.cpu(), 
        **kwargs
    ):
        '''
        Init.

        Args:
            model:                          is-a `mxnet.gluon.hybrid.hybridblock.HybridBlock`.
            initializer:                    is-a `mxnet.initializer`.
            condition_sampler:              is-a `ConditionSampler` of sampler to draw conditons from user-defined distributions.
            learning_rate:                  `float` of learning rate.
            optimizer_name:                 `str` of name of optimizer.
            hybridize_flag:                  Call `mxnet.gluon.HybridBlock.hybridize()` or not.
            scale:                          `float` of scaling factor for initial parameters.
            ctx:                            `mx.cpu()` or `mx.gpu()`.

        '''
        if initializer is None:
            self.initializer = mx.initializer.Xavier(
                rnd_type="gaussian", 
                factor_type="in", 
                magnitude=2
            )
        else:
            if isinstance(initializer, mx.initializer.Initializer) is False:
                raise TypeError("The type of `initializer` must be `mxnet.initializer.Initializer`.")
            self.initializer = initializer

        init_deferred_flag = self.init_deferred_flag
        self.init_deferred_flag = True

        super().__init__(
            noise_sampler=NoiseSampler(), 
            model=model, 
            initializer=initializer,
            condition_sampler=condition_sampler,
            conditonal_dim=1,
            learning_rate=learning_rate,
            optimizer_name=optimizer_name,
            hybridize_flag=hybridize_flag,
            scale=scale, 
            ctx=ctx, 
            **kwargs
        )
        self.init_deferred_flag = init_deferred_flag

        self.model = model
        logger = getLogger("accelbrainbase")
        self.__logger = logger

        if self.init_deferred_flag is False:
            try:
                self.collect_params().initialize(self.initializer, force_reinit=True, ctx=ctx)
                self.trainer = gluon.Trainer(
                    self.collect_params(),
                    optimizer_name,
                    {
                        "learning_rate": learning_rate
                    }
                )
                if hybridize_flag is True:
                    self.model.hybridize()
                    if condition_sampler is not None:
                        try:
                            if condition_sampler.observable_data is not None:
                                condition_sampler.observable_data.hybridize()
                        except AttributeError:
                            pass

            except InitDeferredError:
                self.__logger.debug("The initialization should be deferred.")

        if condition_sampler is not None:
            self.condition_sampler = condition_sampler

    def draw(self):
        '''
        Draw samples from the fake distribution.

        Returns:
            `Tuple` of `mxnet.ndarray`s.
            The shape is ... $(\hat{z}, x)$.
        '''
        condition_arr = self.condition_sampler.draw()
        if condition_arr.ndim == 5:
            seq_len = condition_arr.shape[1]
        else:
            seq_len = 1

        soft_assignment_arr = condition_arr
        if seq_len == 1:
            x_arr = self.model(condition_arr)
        else:
            soft_assignment_arr = nd.zeros_like(condition_arr)
            for i in range(seq_len):
                x_arr[:, i] = self.model(condition_arr[:, i])

        return x_arr, soft_assignment_arr

    # is-a `ConditionSampler`.
    __conditon_sampler = None

    def get_condition_sampler(self):
        ''' getter '''
        return self.__conditon_sampler
    
    def set_condition_sampler(self, value):
        ''' setter '''
        if isinstance(value, ConditionSampler) is False:
            raise TypeError("The type of `condition_sampler` must be `ConditionSampler`.")
        self.__conditon_sampler = value

    conditon_sampler = property(get_condition_sampler, set_condition_sampler)

    def get_init_deferred_flag(self):
        ''' getter '''
        return self.__init_deferred_flag
    
    def set_init_deferred_flag(self, value):
        ''' setter '''
        self.__init_deferred_flag = value

    init_deferred_flag = property(get_init_deferred_flag, set_init_deferred_flag)
