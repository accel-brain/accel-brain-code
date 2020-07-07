# -*- coding: utf-8 -*-
from accelbrainbase._mxnet._exception.init_deferred_error import InitDeferredError
from accelbrainbase.observabledata._mxnet.function_approximator import FunctionApproximator as _FunctionApproximator

from mxnet.gluon.block import HybridBlock
from mxnet import gluon
from mxnet import autograd
import numpy as np
import mxnet as mx
from mxnet import MXNetError
from logging import getLogger


class FunctionApproximator(HybridBlock, _FunctionApproximator):
    '''
    The function approximator for the Deep Q-Learning.

    The convolutional neural networks(CNNs) are hierarchical models 
    whose convolutional layers alternate with subsampling layers, 
    reminiscent of simple and complex cells in the primary visual cortex.
    
    Mainly, this class demonstrates that a CNNs can solve generalisation problems to learn 
    successful control policies from observed data points in complex 
    Reinforcement Learning environments. The network is trained with a variant of 
    the Q-learning algorithm, with stochastic gradient descent to update the weights.

    But there is no need for the function approximator to be a CNNs.
    We probide this interface that implements various models as 
    function approximations, not limited to CNNs.

    References:
        - Dumoulin, V., & V,kisin, F. (2016). A guide to convolution arithmetic for deep learning. arXiv preprint arXiv:1603.07285.
        - Masci, J., Meier, U., Cireşan, D., & Schmidhuber, J. (2011, June). Stacked convolutional auto-encoders for hierarchical feature extraction. In International Conference on Artificial Neural Networks (pp. 52-59). Springer, Berlin, Heidelberg.
        - Mnih, V., Kavukcuoglu, K., Silver, D., Graves, A., Antonoglou, I., Wierstra, D., & Riedmiller, M. (2013). Playing atari with deep reinforcement learning. arXiv preprint arXiv:1312.5602.

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
        super(FunctionApproximator, self).__init__(**kwargs)

        self.model = model
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

            except InitDeferredError:
                self.__logger.debug("The initialization should be deferred.")

    def collect_params(self, select=None):
        '''
        Overrided `collect_params` in `mxnet.gluon.HybridBlok`.
        '''
        params_dict = self.model.collect_params(select)
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
        return self.model.forward_propagation(F, x)

    # is-a `mxnet.gluon.hybrid.hybridblock.HybridBlock`.
    __model = None

    def get_model(self):
        ''' getter for `mxnet.gluon.hybrid.hybridblock.HybridBlock`. '''
        return self.__model
    
    def set_model(self, value):
        ''' setter for `mxnet.gluon.hybrid.hybridblock.HybridBlock`. '''
        self.__model = value
    
    model = property(get_model, set_model)
