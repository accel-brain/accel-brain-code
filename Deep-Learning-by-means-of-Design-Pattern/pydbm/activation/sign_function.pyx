# -*- coding: utf-8 -*-
from abc import abstractmethod
import numpy as np
cimport numpy as np
from pydbm.activation.interface.activating_function_interface import ActivatingFunctionInterface
ctypedef np.float64_t DOUBLE_t


class SignFunction(ActivatingFunctionInterface):
    '''
    Sign function (or Signam function).

    As the sign function is non-smooth and non-convex, its gradient is zero for all nonzero inputs,
    and is ill-defined at zero, which makes the standard back-propagation infeasible. This is a kind
    of the vanishing gradient problem(Song, J. et al., 2018).

    In order to tackle this problem setting, this library entries some methods such as
    a Straight-Through Estimator(Bengio, Y. et al., 2013) and a Binary Neurons(Oza, M., et al., 2018).

    The Straight-Through Estimator is a strategy to replace the non-differentiable functions, 
    which are used in the forward pass, by differentiable functions in the backward pass.
    In the backward pass, the Straight-Through Estimator simply treats the binary neurons 
    as identify functions and ignores their gradients. A variant of the Straight-Through Estimator 
    is the sigmoid-adjusted Straight-Through Estimator, which multiplies the gradients in the 
    backward pass by the derivative of the sigmoid function.

    The binary neurons are neurons that output binary valued predictions as a sign-like function. 
    This library draw a distinction between deterministic binary neurons and stochastic binary neurons. 
    The function of deterministic binary neurons is to act like neurons with hard thresholding functions 
    as their activation functions. The stochastic binary neurons, in contrast, binarize an input according 
    to a probability.

    This class is a abstract class in `Template Method Pattern`, which is also useful design method to 
    design the approximators in this library because this design pattern makes it possible to define the 
    skeleton of an algorithm in the approximations, deferring some steps to client subclasses such as 
    `DeterministicBinaryNeurons` and `StochasticBinaryNeurons`.

    References:
        - Bengio, Y., Léonard, N., & Courville, A. (2013). Estimating or propagating gradients through stochastic neurons for conditional computation. arXiv preprint arXiv:1308.3432.
        - Chung, J., Ahn, S., & Bengio, Y. (2016). Hierarchical multiscale recurrent neural networks. arXiv preprint arXiv:1609.01704.
        - Oza, M., Vaghela, H., & Srivastava, K. (2019). Progressive Generative Adversarial Binary Networks for Music Generation. arXiv preprint arXiv:1903.04722.
        - Song, J., He, T., Gao, L., Xu, X., Hanjalic, A., & Shen, H. T. (2018, April). Binary generative adversarial networks for image retrieval. In Thirty-Second AAAI Conference on Artificial Intelligence.
    '''

    # The length of memories.
    __memory_len = 50

    def __init__(self, memory_len=50):
        '''
        Init.

        Args:
            memory_len:     The number of memos of activities for derivative in backward.

        '''
        self.__activity_arr_list = []
        self.__memory_len = memory_len

    def get_activity_arr_list(self):
        ''' getter '''
        return self.__activity_arr_list
    
    def set_activity_arr_list(self, value):
        ''' setter '''
        self.__activity_arr_list = value
    
    activity_arr_list = property(get_activity_arr_list, set_activity_arr_list)

    def get_memory_len(self):
        ''' getter '''
        return self.__memory_len
    
    def set_memory_len(self, value):
        ''' setter '''
        self.__memory_len = value
    
    memory_len = property(get_memory_len, set_memory_len)
