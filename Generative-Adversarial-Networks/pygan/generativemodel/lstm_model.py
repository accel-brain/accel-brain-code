# -*- coding: utf-8 -*-
import numpy as np
from logging import getLogger, StreamHandler, NullHandler, DEBUG, ERROR

from pygan.generative_model import GenerativeModel

from pydbm.optimization.opt_params import OptParams
from pydbm.verification.interface.verificatable_result import VerificatableResult
from pydbm.loss.interface.computable_loss import ComputableLoss

from pydbm.rnn.lstm_model import LSTMModel

# Loss function.
from pydbm.loss.mean_squared_error import MeanSquaredError
# Adam as a optimizer.
from pydbm.optimization.optparams.sgd import SGD
# Verification.
from pydbm.verification.verificate_function_approximation import VerificateFunctionApproximation

from pygan.discriminative_model import DiscriminativeModel


class LSTMModel(GenerativeModel):
    '''
    LSTM as a Generator.

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

    def __init__(
        self,
        batch_size,
        lstm_model,
        seq_len=10,
        learning_rate=1e-05,
        computable_loss=None,
        opt_params=None,
        norm_mode="z_score",
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
            norm_mode:                      How to normalize generated values.
                                            - `z_score`: Z-Score normalization.
                                            - `min_max`: Min-max normalization.
                                            - `tanh`: Normalization by tanh function.

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

        if isinstance(lstm_model, LSTMModel) is False:
            raise TypeError()

        if computable_loss is None:
            computable_loss = MeanSquaredError()
        if verificatable_result is None:
            verificatable_result = VerificateFunctionApproximation()
        if opt_params is None:
            opt_params = SGD()
            opt_params.weight_limit = 0.5
            opt_params.dropout_rate = 0.0

        self.__lstm_model = lstm_model
        self.__seq_len = seq_len
        self.__batch_size = batch_size
        self.__computable_loss = computable_loss
        self.__learning_rate = learning_rate
        self.__verbose_mode = verbose_mode
        self.__loss_list = []
        self.__norm_mode = norm_mode

    def draw(self):
        '''
        Draws samples from the `fake` distribution.

        Returns:
            `np.ndarray` of samples.
        '''
        observed_arr = self.noise_sampler.generate()
        arr = self.inference(observed_arr)
        if self.__norm_mode == "z_score":
            for i in range(arr.shape[0]):
                for seq in range(arr.shape[1]):
                    arr[i, seq] = (arr[i, seq] - arr[i, seq].mean()) / arr[i, seq].std()
        elif self.__norm_mode == "min_max":
            for i in range(arr.shape[0]):
                for seq in range(arr.shape[1]):
                    arr[i, seq] = (arr[i, seq] - arr[i, seq].min()) / (arr[i, seq].max() - arr[i, seq].min())
        elif self.__norm_mode == "tanh":
            arr = np.tanh(arr)

        return arr

    def inference(self, observed_arr):
        '''
        Draws samples from the `true` distribution.

        Args:
            observed_arr:     `np.ndarray` of observed data points.
        
        Returns:
            `np.ndarray` of inferenced.
            `0` is to `1` what `fake` is to `true`.
        '''
        return self.__lstm_model.inference(observed_arr)

    def learn(self, grad_arr):
        '''
        Update this Discriminator by ascending its stochastic gradient.

        Args:
            grad_arr:   `np.ndarray` of gradients.
        
        '''
        delta_arr, lstm_hidden_grads_list = self.__lstm_model.hidden_back_propagate(
            grad_arr
        )
        self.__lstm_model.optimize(lstm_hidden_grads_list, self.__learning_rate, 1)
