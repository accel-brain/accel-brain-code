# -*- coding: utf-8 -*-
from accelbrainbase.observable_data import ObservableData
from abc import abstractproperty


class FunctionApproximator(ObservableData):
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

    @abstractproperty
    def model(self):
        ''' is-a `mxnet.gluon.hybrid.hybridblock.HybridBlock`. '''
        raise NotImplementedError()

    def learn(self, iteratable_data):
        '''
        Learn the observed data points
        for vector representation of the input images.

        Args:
            iteratable_data:     is-a `IteratableData`.

        '''
        self.model.learn(iteratable_data)

    def inference(self, observed_arr):
        '''
        Inference samples drawn by `IteratableData.generate_inferenced_samples()`.

        Args:
            observed_arr:   rank-2 Array like or sparse matrix as the observed data points.
                            The shape is: (batch size, feature points)

        Returns:
            `mxnet.ndarray` of inferenced feature points.
        '''
        return self.model.inference(observed_arr)

    # `bool` that means initialization in this class will be deferred or not.
    __init_deferred_flag = False

    def get_init_deferred_flag(self):
        ''' getter for `bool` that means initialization in this class will be deferred or not. '''
        return self.__init_deferred_flag
    
    def set_init_deferred_flag(self, value):
        ''' setter for `bool` that means initialization in this class will be deferred or not. '''
        self.__init_deferred_flag = value

    init_deferred_flag = property(get_init_deferred_flag, set_init_deferred_flag)

    # is-a `mxnet.initializer` for parameters of model. If `None`, it is drawing from the Xavier distribution.
    __initializer = None

    def get_initializer(self):
        ''' getter for `mxnet.initializer`. '''
        return self.__initializer
    
    def set_initializer(self, value):
        ''' setter for `mxnet.initializer`. '''
        self.__initializer = value
    
    initializer = property(get_initializer, set_initializer)
