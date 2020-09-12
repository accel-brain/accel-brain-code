# -*- coding: utf-8 -*-
from accelbrainbase.observabledata._mxnet.adversarialmodel.discriminative_model import DiscriminativeModel
from accelbrainbase._mxnet._exception.init_deferred_error import InitDeferredError

from mxnet import gluon
from mxnet import autograd
import numpy as np
import mxnet as mx
from mxnet import MXNetError
from logging import getLogger


class MultimodalDiscriminativeModel(DiscriminativeModel):
    '''
    Discriminative model, which discriminates multi-modal observed data points.
    '''

    def __init__(
        self, 
        model_list, 
        final_model,
        initializer=None,
        learning_rate=1e-05,
        optimizer_name="SGD",
        hybridize_flag=True,
        scale=1.0, 
        ctx=mx.gpu(), 
        **kwargs
    ):
        '''
        Init.

        Args:
            model_list:                     `list` of `mxnet.gluon.hybrid.hybridblock.HybridBlock`s.
            final_model:                    is-a `mxnet.gluon.hybrid.hybridblock.HybridBlock`.
            initializer:                    is-a `mxnet.initializer` for parameters of model. If `None`, it is drawing from the Xavier distribution.
            learning_rate:                  `float` of learning rate.
            optimizer_name:                 `str` of name of optimizer.
            hybridize_flag:                  Call `mxnet.gluon.HybridBlock.hybridize()` or not.
            scale:                          `float` of scaling factor for initial parameters.
            ctx:                            `mx.cpu()` or `mx.gpu()`.

        '''
        init_deferred_flag = self.init_deferred_flag
        self.init_deferred_flag = True
        logger = getLogger("accelbrainbase")
        self.__logger = logger

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

        super().__init__(
            model=final_model,
            initializer=initializer,
            learning_rate=learning_rate,
            optimizer_name=optimizer_name,
            hybridize_flag=hybridize_flag,
            scale=scale, 
            ctx=ctx, 
            **kwargs
        )
        self.model = final_model
        self.model_list = model_list

        self.init_deferred_flag = init_deferred_flag
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
                    try:
                        for i in range(len(self.model_list)):
                            self.model[i].hybridize()

                    except AttributeError:
                        pass

            except InitDeferredError:
                self.__logger.debug("The initialization should be deferred.")

    def collect_params(self, select=None):
        '''
        Overrided `collect_params` in `mxnet.gluon.HybridBlok`.
        '''
        params_dict = self.model.collect_params(select)
        for i in range(len(self.model_list)):
            params_dict.update(self.model_list[i].collect_params(select))

        return params_dict

    def inference(self, observed_arr):
        '''
        Draw samples from the fake distribution.

        Args:
            observed_arr:       `mxnet.ndarray` or `mxnet.symbol` of observed data points.

        Returns:
            `Tuple` of `mxnet.ndarray`s.
        '''
        return self(observed_arr)

    def hybrid_forward(self, F, x):
        '''
        Hybrid forward with Gluon API.

        Args:
            F:      `mxnet.ndarray` or `mxnet.symbol`.
            x:      `mxnet.ndarray` of observed data points.
        
        Returns:
            `mxnet.ndarray` or `mxnet.symbol` of inferenced feature points.
        '''
        # rank-3
        return self.forward_propagation(F, x)

    def forward_propagation(self, F, x):
        '''
        Hybrid forward with Gluon API.

        Args:
            F:      `mxnet.ndarray` or `mxnet.symbol`.
            x:      `mxnet.ndarray` of observed data points.
        
        Returns:
            `mxnet.ndarray` or `mxnet.symbol` of inferenced feature points.
        '''
        inferenced_arr = self.model(x)
        return inferenced_arr
