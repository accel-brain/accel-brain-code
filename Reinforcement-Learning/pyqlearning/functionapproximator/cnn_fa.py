# -*- coding: utf-8 -*-
import numpy as np
from logging import getLogger, StreamHandler, NullHandler, DEBUG, ERROR

from pyqlearning.function_approximator import FunctionApproximator

from pydbm.cnn.convolutional_neural_network import ConvolutionalNeuralNetwork
from pydbm.cnn.layerablecnn.convolution_layer import ConvolutionLayer
from pydbm.cnn.layerable_cnn import LayerableCNN
from pydbm.cnn.feature_generator import FeatureGenerator
from pydbm.optimization.opt_params import OptParams
from pydbm.verification.interface.verificatable_result import VerificatableResult
from pydbm.loss.interface.computable_loss import ComputableLoss

# Loss function.
from pydbm.loss.mean_squared_error import MeanSquaredError
# Adam as a optimizer.
from pydbm.optimization.optparams.adam import Adam
# Verification.
from pydbm.verification.verificate_function_approximation import VerificateFunctionApproximation


class CNNFA(FunctionApproximator):
    '''
    Convolutional Neural Networks(CNNs) as a Function Approximator.

    CNNs are hierarchical models whose convolutional layers alternate with subsampling
    layers, reminiscent of simple and complex cells in the primary visual cortex.
    
    This class demonstrates that a CNNs can solve generalisation problems to learn 
    successful control policies from observed data points in complex 
    Reinforcement Learning environments. The network is trained with a variant of 
    the Q-learning algorithm, with stochastic gradient descent to update the weights.
    
    The Deconvolution also called transposed convolutions “work by swapping the forward and backward passes of a convolution.” (Dumoulin, V., & Visin, F. 2016, p20.)
    
    References:
        - Dumoulin, V., & V,kisin, F. (2016). A guide to convolution arithmetic for deep learning. arXiv preprint arXiv:1603.07285.
        - Masci, J., Meier, U., Cireşan, D., & Schmidhuber, J. (2011, June). Stacked convolutional auto-encoders for hierarchical feature extraction. In International Conference on Artificial Neural Networks (pp. 52-59). Springer, Berlin, Heidelberg.
        - Mnih, V., Kavukcuoglu, K., Silver, D., Graves, A., Antonoglou, I., Wierstra, D., & Riedmiller, M. (2013). Playing atari with deep reinforcement learning. arXiv preprint arXiv:1312.5602.
    '''
    
    def __init__(
        self,
        batch_size,
        layerable_cnn_list,
        learning_rate=1e-05,
        computable_loss=None,
        opt_params=None,
        verificatable_result=None,
        pre_learned_path_list=None,
        fc_w_arr=None,
        fc_activation_function=None,
        verbose_mode=False
    ):
        logger = getLogger("pydbm")
        handler = StreamHandler()
        if verbose_mode is True:
            handler.setLevel(DEBUG)
            logger.setLevel(DEBUG)
        else:
            handler.setLevel(ERROR)
            logger.setLevel(ERROR)
            
        logger.addHandler(handler)

        self.__logger = getLogger("pyqlearning")
        handler = StreamHandler()
        if verbose_mode is True:
            self.__logger.setLevel(DEBUG)
        else:
            self.__logger.setLevel(ERROR)
            
        self.__logger.addHandler(handler)

        if computable_loss is None:
            computable_loss = MeanSquaredError()
        if verificatable_result is None:
            verificatable_result = VerificateFunctionApproximation()
        if opt_params is None:
            opt_params = Adam()

        cnn = ConvolutionalNeuralNetwork(
            # The `list` of `ConvolutionLayer`.
            layerable_cnn_list=layerable_cnn_list,
            # The number of epochs in mini-batch training.
            epochs=200,
            # The batch size.
            batch_size=batch_size,
            # Learning rate.
            learning_rate=learning_rate,
            # Loss function.
            computable_loss=computable_loss,
            # Optimizer.
            opt_params=opt_params,
            # Verification.
            verificatable_result=verificatable_result,
            # Others.
            learning_attenuate_rate=0.1,
            attenuate_epoch=50
        )
        self.__cnn = cnn
        self.__batch_size = batch_size
        self.__computable_loss = computable_loss
        self.__learning_rate = learning_rate
        self.__verbose_mode = verbose_mode
        self.__fc_w_arr = fc_w_arr
        self.__fc_activation_function = fc_activation_function
        self.__q_shape = None
        self.__q_logs_list = []

    def learn_q(self, q, new_q):
        '''
        Infernce Q-Value.
        
        Args:
            q:                  Predicted Q-Value.
            new_q:              Real Q-Value.
        '''
        if self.__q_shape is None:
            raise ValueError("Before learning, You should execute `__inference_q`.")

        q_arr = np.array([q] * self.__batch_size).reshape(-1, 1)
        new_q_arr = np.array([new_q] * self.__batch_size).reshape(-1, 1)
        cost_arr = self.__computable_loss.compute_loss(q_arr, new_q_arr)
        delta_arr = self.__computable_loss.compute_delta(q_arr, new_q_arr)
        # This is a constant parameter and will be not updated.
        delta_arr = np.dot(delta_arr, self.__fc_w_arr.T)
        delta_arr = delta_arr / self.__batch_size

        delta_arr = delta_arr.reshape(self.__q_shape)
        delta_arr = self.__cnn.back_propagation(delta_arr)
        self.__cnn.optimize(self.__learning_rate, 1)
        
        self.__q_logs_list.append((q, new_q, cost_arr.mean()))

    def inference_q(self, next_action_arr):
        '''
        Infernce Q-Value.
        
        Args:
            next_action_arr:     `np.ndarray` of action.
        
        Returns:
            `np.ndarray` of Q-Values.
        '''
        q_arr = self.__cnn.inference(next_action_arr)
        self.__q_shape = q_arr.shape
        q_arr = q_arr.reshape((q_arr.shape[0], -1))
        
        if self.__fc_w_arr is None:
            self.__fc_w_arr = np.random.normal(size=(q_arr.shape[-1], 1)) * 0.01

        q_arr = np.dot(q_arr, self.__fc_w_arr)
        q_arr = self.__fc_activation_function.activate(q_arr)
        return q_arr

    def get_q_logs_list(self):
        ''' getter '''
        return self.__q_logs_list

    def set_q_logs_list(self, value):
        ''' setter '''
        self.__q_logs_list = value
    
    q_logs_list = property(get_q_logs_list, set_q_logs_list)
