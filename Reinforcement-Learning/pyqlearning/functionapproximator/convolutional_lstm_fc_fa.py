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

from pydbm.rnn.lstm_model import LSTMModel

# Loss function.
from pydbm.loss.mean_squared_error import MeanSquaredError
# Adam as a optimizer.
from pydbm.optimization.optparams.adam import Adam
# Verification.
from pydbm.verification.verificate_function_approximation import VerificateFunctionApproximation


class ConvolutionalLSTMFCFA(FunctionApproximator):
    '''
    Convolutional LSTM Networks as a Function Approximator like CLDNN Architecture(Sainath, T. N, et al., 2015).
    
    This is a model of the function approximator which loosely coupled CNN and LSTM.
    Like CLDNN Architecture(Sainath, T. N, et al., 2015), this model uses CNNs to reduce 
    the spectral variation of the input feature of rewards, and then passes this to LSTM 
    layers to perform temporal modeling, and finally outputs this to DNN layers, 
    which produces a feature representation of Q-Values that is more easily separable.
    
    References:
        - https://code.accel-brain.com/Deep-Learning-by-means-of-Design-Pattern/pydbm.cnn.html
        - Sainath, T. N., Vinyals, O., Senior, A., & Sak, H. (2015, April). Convolutional, long short-term memory, fully connected deep neural networks. In Acoustics, Speech and Signal Processing (ICASSP), 2015 IEEE International Conference on (pp. 4580-4584). IEEE.
    '''
    
    __q_arr_list = []
    
    def __init__(
        self,
        batch_size,
        layerable_cnn_list,
        lstm_model,
        seq_len=10,
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

        if isinstance(lstm_model, LSTMModel) is False:
            raise TypeError()
        
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
        self.__lstm_model = lstm_model
        self.__seq_len = seq_len
        self.__batch_size = batch_size
        self.__computable_loss = computable_loss
        self.__learning_rate = learning_rate
        self.__verbose_mode = verbose_mode
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
        delta_arr = delta_arr / self.__batch_size
        delta_arr, lstm_output_grads_list = self.__lstm_model.output_back_propagate(
            q_arr,
            delta_arr
        )

        delta_arr, lstm_hidden_grads_list = self.__lstm_model.hidden_back_propagate(
            delta_arr
        )

        lstm_grads_list = lstm_output_grads_list
        lstm_grads_list.extend(lstm_hidden_grads_list)

        delta_arr = delta_arr[:, -1].reshape(self.__q_shape)
        delta_arr = self.__cnn.back_propagation(delta_arr)

        self.__lstm_model.optimize(lstm_grads_list, self.__learning_rate, 1)
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
        
        self.__q_arr_list.append(q_arr)
        while len(self.__q_arr_list) > self.__seq_len:
            self.__q_arr_list = self.__q_arr_list[1:]
        while len(self.__q_arr_list) < self.__seq_len:
            self.__q_arr_list.append(self.__q_arr_list[-1])
        q_arr = np.array(self.__q_arr_list)
        q_arr = q_arr.transpose((1, 0, 2))
        q_arr = self.__lstm_model.inference(q_arr)
        return q_arr[:, -1]

    def get_q_logs_list(self):
        ''' getter '''
        return self.__q_logs_list

    def set_q_logs_list(self, value):
        ''' setter '''
        self.__q_logs_list = value
    
    q_logs_list = property(get_q_logs_list, set_q_logs_list)
