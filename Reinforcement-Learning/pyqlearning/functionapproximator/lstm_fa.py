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
    
    Originally, Long Short-Term Memory(LSTM) networks as a 
    special RNN structure has proven stable and powerful for 
    modeling long-range dependencies.

    The Key point of structural expansion is its memory cell 
    which essentially acts as an accumulator of the state information. 
    Every time observed data points are given as new information and input 
    to LSTM’s input gate, its information will be accumulated to the cell 
    if the input gate is activated. The past state of cell could be forgotten 
    in this process if LSTM’s forget gate is on. Whether the latest cell output 
    will be propagated to the final state is further controlled by the output gate.

    References:
        - Cho, K., Van Merriënboer, B., Gulcehre, C., Bahdanau, D., Bougares, F., Schwenk, H., & Bengio, Y. (2014). Learning phrase representations using RNN encoder-decoder for statistical machine translation. arXiv preprint arXiv:1406.1078.
        - Malhotra, P., Ramakrishnan, A., Anand, G., Vig, L., Agarwal, P., & Shroff, G. (2016). LSTM-based encoder-decoder for multi-sensor anomaly detection. arXiv preprint arXiv:1607.00148.
        - Zaremba, W., Sutskever, I., & Vinyals, O. (2014). Recurrent neural network regularization. arXiv preprint arXiv:1409.2329.

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
        verbose_mode=False
    ):
        '''
        Init.

        Args:
            batch_size:                     Batch size in mini-batch.
            lstm_model:                     is-a `LSTMMode`.
            seq_len:                        The length of sequences.
            learning_rate:                  Learning rate.
            computable_loss:                is-a `ComputableLoss`.
            opt_params:                     is-a `OptParams`.
            verificatable_result:           is-a `VerificateFunctionApproximation`.
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

        self.__lstm_model = lstm_model
        self.__seq_len = seq_len
        self.__batch_size = batch_size
        self.__computable_loss = computable_loss
        self.__learning_rate = learning_rate
        self.__verbose_mode = verbose_mode
        self.__loss_list = []

    def learn_q(self, predicted_q_arr, real_q_arr):
        '''
        Infernce Q-Value.
        
        Args:
            predicted_q_arr:    `np.ndarray` of predicted Q-Values.
            real_q_arr:         `np.ndarray` of real Q-Values.
        '''
        loss = self.__computable_loss.compute_loss(predicted_q_arr, real_q_arr)
        delta_arr = self.__computable_loss.compute_delta(predicted_q_arr, real_q_arr)
        delta_arr, lstm_output_grads_list = self.__lstm_model.output_back_propagate(
            np.expand_dims(predicted_q_arr, axis=1),
            np.expand_dims(delta_arr, axis=1)
        )
        delta_arr, _, lstm_hidden_grads_list = self.__lstm_model.hidden_back_propagate(
            delta_arr[:, -1]
        )
        lstm_grads_list = lstm_output_grads_list
        lstm_grads_list.extend(lstm_hidden_grads_list)
        self.__lstm_model.optimize(lstm_grads_list, self.__learning_rate, 1)
        self.__loss_list.append(loss)

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
        return q_arr[:, -1].reshape((q_arr.shape[0], 1))

    def get_model(self):
        '''
        `object` of model as a function approximator,
        which has `lstm_model` whose type is `pydbm.rnn.lstm_model.LSTMModel`.
        '''
        class Model(object):
            def __init__(self, lstm_model):
                self.lstm_model = lstm_model
        return Model(self.__lstm_model)

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
