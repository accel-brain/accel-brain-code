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
        cnn=None,
        verbose_mode=False
    ):
        '''
        Init.

        Args:
            batch_size:                     Batch size in mini-batch.
            layerable_cnn_list:             `list` of `LayerableCNN`.
            lstm_model:                     is-a `LSTMMode`.
            seq_len:                        The length of sequences.
            learning_rate:                  Learning rate.
            computable_loss:                is-a `ComputableLoss`.
            opt_params:                     is-a `OptParams`.
            verificatable_result:           is-a `VerificateFunctionApproximation`.
            pre_learned_path_list:          `list` of file path that stored pre-learned parameters.
                                            This parameters will be refered only when `cnn` is `None`.

            cnn:                            is-a `ConvolutionalNeuralNetwork` as a model in this class.
                                            If not `None`, `self.__cnn` will be overrided by this `cnn`.
                                            If `None`, this class initialize `ConvolutionalNeuralNetwork`
                                            by default hyper parameters.

            verbose_mode:                   Verbose mode or not.

        '''
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
            opt_params.weight_limit = 0.5
            opt_params.dropout_rate = 0.0

        if cnn is None:
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
                # Pre-learned parameters.
                pre_learned_path_list=pre_learned_path_list,
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
        self.__loss_list = []

    def learn_q(self, predicted_q_arr, real_q_arr):
        '''
        Infernce Q-Value.
        
        Args:
            predicted_q_arr:    `np.ndarray` of predicted Q-Values.
            real_q_arr:         `np.ndarray` of real Q-Values.
        '''
        if self.__q_shape is None:
            raise ValueError("Before learning, You should execute `__inference_q`.")

        loss = self.__computable_loss.compute_loss(predicted_q_arr, real_q_arr)
        delta_arr = self.__computable_loss.compute_delta(predicted_q_arr, real_q_arr)
        delta_arr, lstm_output_grads_list = self.__lstm_model.output_back_propagate(
            predicted_q_arr,
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
        self.__loss_list.append(loss)

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
        return q_arr[:, -1].reshape((q_arr.shape[0], 1))

    def get_model(self):
        '''
        `object` of model as a function approximator,
        which has `cnn` whose type is 
        `pydbm.cnn.pydbm.cnn.convolutional_neural_network.ConvolutionalNeuralNetwork`
        and `lstm_model` whose type is `pydbm.rnn.lstm_model.LSTMModel`.
        '''
        class Model(object):
            def __init__(self, cnn, lstm_model):
                self.cnn = cnn
                self.lstm_model = lstm_model

        return Model(self.__cnn, self.__lstm_model)

    def set_model(self, value):
        '''
        Model as a function approximator.
        '''
        raise TypeError("This property must be read-only.")

    model = property(get_model, set_model)

    def get_loss_list(self):
        ''' getter '''
        return self.__loss_list

    def set_loss_list(self, value):
        ''' setter '''
        self.__loss_list = value
    
    loss_list = property(get_loss_list, set_loss_list)
