# -*- coding: utf-8 -*-
import numpy as np
from logging import getLogger, StreamHandler, NullHandler, DEBUG, ERROR

from pyqlearning.function_approximator import FunctionApproximator

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


class LSTMFA(FunctionApproximator):
    '''
    LSTM Networks as a Function Approximator.
    '''
    
    __q_arr_list = []
    
    def __init__(
        self,
        batch_size,
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

        self.__lstm_model = lstm_model
        self.__seq_len = seq_len
        self.__batch_size = batch_size
        self.__computable_loss = computable_loss
        self.__learning_rate = learning_rate
        self.__verbose_mode = verbose_mode
        self.__q_logs_list = []

    def learn_q(self, q, new_q):
        '''
        Infernce Q-Value.
        
        Args:
            q:                  Predicted Q-Value.
            new_q:              Real Q-Value.
        '''
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

        self.__lstm_model.optimize(lstm_grads_list, self.__learning_rate, 1)
        self.__q_logs_list.append((q, new_q, cost_arr.mean()))

    def inference_q(self, next_action_arr):
        '''
        Infernce Q-Value.
        
        Args:
            next_action_arr:     `np.ndarray` of action.
        
        Returns:
            `np.ndarray` of Q-Values.
        '''
        q_arr = next_action_arr.reshape((next_action_arr.shape[0], -1))
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
