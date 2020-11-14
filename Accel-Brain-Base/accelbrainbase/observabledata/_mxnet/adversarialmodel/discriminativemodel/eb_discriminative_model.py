# -*- coding: utf-8 -*-
from accelbrainbase._mxnet._exception.init_deferred_error import InitDeferredError
from accelbrainbase.observabledata._mxnet.adversarialmodel.discriminative_model import DiscriminativeModel

from mxnet.gluon.block import HybridBlock
from mxnet import gluon
from mxnet import autograd
import numpy as np
import mxnet as mx
from mxnet import MXNetError
from logging import getLogger


class EBDiscriminativeModel(DiscriminativeModel):
    '''
    Discriminative model, which discriminates true from fake,
    in the Energy-based Generative Adversarial Network(EBGAN).

    The Energy-based Generative Adversarial Network (EBGAN) model(Zhao, J., et al., 2016) which 
    views the discriminator as an energy function that attributes low energies to the regions 
    near the data manifold and higher energies to other regions. The Auto-Encoders have traditionally 
    been used to represent energy-based models. When trained with some regularization terms, 
    the Auto-Encoders have the ability to learn an energy manifold without supervision or negative examples. 
    This means that even when an energy-based Auto-Encoding model is trained to reconstruct a real sample, 
    the model contributes to discovering the data manifold by itself.

    References:
        - Zhao, J., Mathieu, M., & LeCun, Y. (2016). Energy-based generative adversarial network. arXiv preprint arXiv:1609.03126.
    '''

    def __init__(
        self, 
        model, 
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
            model:                          is-a `mxnet.gluon.hybrid.hybridblock.HybridBlock`.
            initializer:                    is-a `mxnet.initializer` for parameters of model. If `None`, it is drawing from the Xavier distribution.
            learning_rate:                  `float` of learning rate.
            optimizer_name:                 `str` of name of optimizer.
            hybridize_flag:                  Call `mxnet.gluon.HybridBlock.hybridize()` or not.
            scale:                          `float` of scaling factor for initial parameters.
            ctx:                            `mx.cpu()` or `mx.gpu()`.

        '''
        logger = getLogger("accelbrainbase")
        self.__logger = logger

        init_deferred_flag = self.init_deferred_flag
        self.init_deferred_flag = True

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

        super().__init__(
            model=model,
            initializer=initializer,
            learning_rate=learning_rate,
            optimizer_name=optimizer_name,
            hybridize_flag=hybridize_flag,
            scale=scale, 
            ctx=ctx, 
            **kwargs
        )
        self.model = model
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
                        self.model.encoder.hybridize()
                        self.model.decoder.hybridize()
                    except AttributeError:
                        pass

            except InitDeferredError:
                self.__logger.debug("The initialization should be deferred.")

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
        inferenced_arr = self.model.forward_propagation(F, x)
        inferenced_arr = F.reshape_like(inferenced_arr, x)
        mse_arr = F.square(x - inferenced_arr)
        return F.expand_dims(
            F.mean(
                mse_arr,
                axis=0,
                exclude=True
            ),
            axis=-1
        )
