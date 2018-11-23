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

# Logistic Function as activation function.
from pydbm.activation.logistic_function import LogisticFunction
# Loss function.
from pydbm.loss.mean_squared_error import MeanSquaredError
# Adam as a optimizer.
from pydbm.optimization.optparams.adam import Adam
# Verification.
from pydbm.verification.verificate_function_approximation import VerificateFunctionApproximation
# First convolution layer.
from pydbm.cnn.layerablecnn.convolution_layer import ConvolutionLayer as ConvolutionLayer1


class CNNFA(FunctionApproximator):
    '''
    Convolutional Neural Networks as a Function Approximator.
    '''
    
    def __init__(
        self,
        batch_size,
        layerable_cnn_list=[],
        learning_rate=1e-05,
        computable_loss=None,
        opt_params=None,
        verificatable_result=None,
        pre_learned_path_list=None,
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

        if len(layerable_cnn_list) == 0:
            # First convolution layer.
            conv1 = ConvolutionLayer1(
                # Computation graph for first convolution layer.
                ConvGraph1(
                    # Logistic function as activation function.
                    activation_function=LogisticFunction(),
                    # The number of `filter`.
                    filter_num=20,
                    # The number of channel.
                    channel=1,
                    # The size of kernel.
                    kernel_size=3,
                    # The filter scale.
                    scale=0.1,
                    # The nubmer of stride.
                    stride=1,
                    # The number of zero-padding.
                    pad=1
                )
            )
            layerable_cnn_list=[
                conv1, 
            ]
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
            # Optimizer.
            opt_params=opt_params,
            # Verification.
            verificatable_result=verificatable_result,
        )
        self.__cnn = cnn
        self.__computable_loss = computable_loss
        self.__learning_rate = learning_rate
        self.__verbose_mode = verbose_mode
        self.__fc_w_arr = np.random.normal(size=(batch_size, 1)) * 0.01

    def learn_q(self, action_key_arr, new_q):
        '''
        Infernce Q-Value.
        
        Args:
            action_key_arr:     `np.ndarray` of action.
            new_q:              Q-Value.
        '''
        q_arr = self.inference_q(action_key_arr)
        q_arr = np.dot(q_arr, self.__fc_w_arr)
        new_q_arr = np.array([new_q] * q_arr.shape[0]).reshape(-1, 1)
        cost_arr = self.__computable_loss.compute_loss(q_arr, new_q_arr)
        self.__logger.debug("Q-cost(min, mean, max): " + str((cost_arr.min(), cost_arr.mean(), cost_arr.max())))
        delta_arr = self.__computable_loss.compute_delta(q_arr, new_q_arr)
        # This is a constant parameter and will be not updated.
        delta_arr = np.dot(delta_arr, self.__fc_w_arr.T)
        delta_arr = delta_arr.reshape(q_arr.shape)
        delta_arr = self.__cnn.back_propagation(delta_arr)
        self.__cnn.optimize(self.__learning_rate, 1)

    def inference_q(self, next_action_arr):
        '''
        Infernce Q-Value.
        
        Args:
            next_action_arr:     `np.ndarray` of action.
        
        Returns:
            `np.ndarray` of Q-Values.
        '''
        q_arr = self.__cnn.inference(next_action_arr)
        return q_arr
