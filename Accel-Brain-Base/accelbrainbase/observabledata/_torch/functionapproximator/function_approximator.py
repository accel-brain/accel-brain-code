# -*- coding: utf-8 -*-
from accelbrainbase.observabledata._torch.function_approximator import FunctionApproximator as _FunctionApproximator
import numpy as np
from logging import getLogger
import torch
from torch import nn
from torch.optim.adam import Adam


class FunctionApproximator(nn.Module, _FunctionApproximator):
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
        learning_rate=1e-05,
        ctx="cpu", 
        not_init_flag=False,
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
        super(FunctionApproximator, self).__init__()

        self.model = model
        logger = getLogger("accelbrainbase")
        self.__logger = logger

        self.__not_init_flag = not_init_flag

    def inference(self, observed_arr):
        '''
        Draw samples from the fake distribution.

        Args:
            observed_arr:       `mxnet.ndarray` or `mxnet.symbol` of observed data points.

        Returns:
            `Tuple` of `mxnet.ndarray`s.
        '''
        return self(observed_arr)

    def forward(self, x):
        '''
        Hybrid forward with Gluon API.

        Args:
            F:      `mxnet.ndarray` or `mxnet.symbol`.
            x:      `mxnet.ndarray` of observed data points.
        
        Returns:
            `mxnet.ndarray` or `mxnet.symbol` of inferenced feature points.
        '''
        return self.model(x)
