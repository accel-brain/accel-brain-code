# -*- coding: utf-8 -*-
import numpy as np
cimport numpy as np
from abc import ABCMeta, abstractmethod


class OptParams(metaclass=ABCMeta):
    '''
    Abstract class of optimization functions.
    
    Note that this library underestimates effects and functions of weight decay regularizations
    and then disregards the possibilities of various variants of weight decay such as 
    *decoupling the weight decay from the gradient-based update* (Loshchilov, I., & Hutter, F., 2017).
    From the perspective of architecture design, the concept of weight decays are highly variable.
    This concept often tends to be described as obscuring the difference from *L2 regularization*.
    From the perspective of algorithm design, it is considered that *weight constraint* or so-called 
    *max-norm regularization* is more effective than weight decay. This regularization technic is 
    structurally easily loosely coupled to other regularization techniques such as dropout (Srivastava, N., et al., 2014).

    References:
        - Kingma, D. P., & Ba, J. (2014). Adam: A method for stochastic optimization. arXiv preprint arXiv:1412.6980.
        - Loshchilov, I., & Hutter, F. (2017). Fixing weight decay regularization in adam. arXiv preprint arXiv:1711.05101.
        - Pascanu, R., Mikolov, T., & Bengio, Y. (2012). Understanding the exploding gradient problem. CoRR, abs/1211.5063, 2.
        - Pascanu, R., Mikolov, T., & Bengio, Y. (2013, February). On the difficulty of training recurrent neural networks. In International conference on machine learning (pp. 1310-1318).
        - Srivastava, N., Hinton, G., Krizhevsky, A., Sutskever, I., & Salakhutdinov, R. (2014). Dropout: a simple way to prevent neural networks from overfitting. The Journal of Machine Learning Research, 15(1), 1929-1958.
        - Zaremba, W., Sutskever, I., & Vinyals, O. (2014). Recurrent neural network regularization. arXiv preprint arXiv:1409.2329.
    '''
    
    # Regularization for weights matrix
    # to repeat multiplying the weights matrix and `0.9`
    # until $\sum_{j=0}^{n}w_{ji}^2 < weight\_limit$.
    __weight_limit = 1e+10
    
    # Probability of dropout.
    __dropout_rate = 0.5

    # `np.ndarray` of dropout values.
    __dropout_rate_arr_list = []

    # Inferencing mode or not.
    __inferencing_mode = False

    # Threshold of the gradient clipping.
    __grad_clip_threshold = 10.0

    # lambda for weight decay.
    __weight_decay_lambda = 0.0

    @abstractmethod
    def optimize(self, params_list, np.ndarray grads_arr, double learning_rate):
        '''
        Return of result from this concrete optimization function.

        Args:
            params_dict:    `list` of parameters.
            grads_arr:      `np.ndarray` of gradation.
            learning_rate:  Learning rate.

        Returns:
            `list` of optimized parameters.
        '''
        raise NotImplementedError()

    def constrain_weight(self, np.ndarray weight_arr):
        '''
        So-called max-norm regularization.

        Regularization for weights matrix
        to repeat multiplying the weights matrix and `0.9`
        until $\sum_{j=0}^{n}w_{ji}^2 < weight\_limit$.
    
        Args:
            weight_arr:       wegiht matrix.
        
        Returns:
            weight matrix.
        '''
        cdef np.ndarray square_weight_arr = np.nanprod(
            np.array([
                np.expand_dims(weight_arr, axis=0),
                np.expand_dims(weight_arr, axis=0)
            ]),
            axis=0
        )[0]
        while np.nansum(square_weight_arr) > self.weight_limit:
            weight_arr = np.nanprod(
                np.array([
                    np.expand_dims(weight_arr, axis=0),
                    np.expand_dims(np.ones_like(weight_arr) * 0.9, axis=0)
                ]),
                axis=0
            )[0]

            square_weight_arr = np.nanprod(
                np.array([
                    np.expand_dims(weight_arr, axis=0),
                    np.expand_dims(weight_arr, axis=0)
                ]),
                axis=0
            )[0]

        return weight_arr

    def compute_weight_decay_delta(self, np.ndarray weight_arr):
        '''
        Compute delta of weight decay.

        Args:
            weight_arr:     `np.ndarray` of weight matrix.
        
        Returns:
            `np.ndarray` of delta.
        '''
        return self.weight_decay_lambda * weight_arr

    def compute_weight_decay(self, np.ndarray weight_arr):
        '''
        Compute penalty term of weight decay.

        Args:
            weight_arr:     `np.ndarray` of weight matrix.
        
        Returns:
            `np.ndarray` of delta.
        '''
        return 0.5 * self.weight_decay_lambda * np.nansum(np.square(weight_arr))

    def dropout(self, np.ndarray activity_arr):
        '''
        Dropout.
        
        Args:
            activity_arr:    The state of units.
        
        Returns:
            The state of units.
        '''
        if self.dropout_rate == 0.0:
            return activity_arr

        if self.inferencing_mode is True:
            return activity_arr * (1.0 - self.dropout_rate)

        cdef int row = activity_arr.shape[0]
        cdef int col
        cdef np.ndarray dropout_rate_arr

        if activity_arr.ndim == 1:
            dropout_rate_arr = np.random.binomial(n=1, p=1-self.dropout_rate, size=(row, )).astype(int)

        if activity_arr.ndim > 1:
            col = activity_arr.shape[1]
            dropout_rate_arr = np.random.binomial(
                n=1, 
                p=1-self.dropout_rate, 
                size=activity_arr.copy().shape
            ).astype(int)

        activity_arr = np.nanprod(
            np.array([
                np.expand_dims(activity_arr, axis=0),
                np.expand_dims(dropout_rate_arr, axis=0)
            ]),
            axis=0
        )[0]

        self.__dropout_rate_arr_list.insert(0, dropout_rate_arr)
        return activity_arr

    def de_dropout(self, np.ndarray delta_arr):
        '''
        Dropout.
        
        Args:
            activity_arr:    The state of delta.
        
        Returns:
            The state of delta.
        '''
        if self.dropout_rate == 0.0:
            return delta_arr
        
        if self.inferencing_mode is True:
            return delta_arr

        cdef np.ndarray dropout_rate_arr = self.__dropout_rate_arr_list.pop(0)
        delta_arr = np.nanprod(
            np.array([
                np.expand_dims(delta_arr, axis=0),
                np.expand_dims(dropout_rate_arr, axis=0)
            ]),
            axis=0
        )[0]
        return delta_arr

    def get_weight_limit(self):
        ''' getter '''
        return self.__weight_limit

    def set_weight_limit(self, value):
        ''' setter '''
        if isinstance(value, float):
            self.__weight_limit = value
        else:
            raise TypeError()

    weight_limit = property(get_weight_limit, set_weight_limit)

    def get_weight_decay_lambda(self):
        ''' getter '''
        return self.__weight_decay_lambda
    
    def set_weight_decay_lambda(self, value):
        ''' setter '''
        self.__weight_decay_lambda = value

    weight_decay_lambda = property(get_weight_decay_lambda, set_weight_decay_lambda)

    def get_dropout_rate(self):
        ''' getter '''
        return self.__dropout_rate

    def set_dropout_rate(self, value):
        ''' setter '''
        if isinstance(value, float):
            self.__dropout_rate = value
        else:
            raise TypeError()

    dropout_rate = property(get_dropout_rate, set_dropout_rate)

    def get_inferencing_mode(self):
        ''' getter '''
        return self.__inferencing_mode
    
    def set_inferencing_mode(self, value):
        ''' setter '''
        self.__inferencing_mode = value
    
    inferencing_mode = property(get_inferencing_mode, set_inferencing_mode)

    def get_grad_clip_threshold(self):
        '''
        getter for the threshold of the gradient clipping.
        '''
        return self.__grad_clip_threshold

    def set_grad_clip_threshold(self, value):
        '''
        setter for the threshold of the gradient clipping.
        '''
        self.__grad_clip_threshold = value
    
    grad_clip_threshold = property(get_grad_clip_threshold, set_grad_clip_threshold)
