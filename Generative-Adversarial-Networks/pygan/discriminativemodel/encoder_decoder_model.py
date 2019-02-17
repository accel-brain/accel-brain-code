# -*- coding: utf-8 -*-
import numpy as np
from logging import getLogger, StreamHandler, NullHandler, DEBUG, ERROR

from pydbm.rnn.encoder_decoder_controller import EncoderDecoderController

from pygan.discriminative_model import DiscriminativeModel


class EncoderDecoderModel(DiscriminativeModel):
    '''
    Encoder/Decoder based on LSTM as a Discriminator.

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
        encoder_decoder_controller,
        seq_len=10,
        learning_rate=1e-05,
        verbose_mode=False
    ):
        '''
        Init.

        Args:
            batch_size:                     Batch size in mini-batch.
            encoder_decoder_controller:     is-a `EncoderDecoderController`.
            seq_len:                        The length of sequences.
            learning_rate:                  Learning rate.
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

        if isinstance(encoder_decoder_controller, EncoderDecoderController) is False:
            raise TypeError()

        self.__encoder_decoder_controller = encoder_decoder_controller
        self.__seq_len = seq_len
        self.__learning_rate = learning_rate
        self.__verbose_mode = verbose_mode
        self.__loss_list = []

    def inference(self, observed_arr):
        '''
        Draws samples from the `true` distribution.

        Args:
            observed_arr:     `np.ndarray` of observed data points.
        
        Returns:
            `np.ndarray` of inferenced.
        '''
        return self.__encoder_decoder_controller.inference(observed_arr)

    def learn(self, grad_arr):
        '''
        Update this Discriminator by ascending its stochastic gradient.

        Args:
            grad_arr:   `np.ndarray` of gradients.
        
        '''
        decoder_grads_list, encoder_delta_arr, encoder_grads_list = self.__encoder_decoder_controller.back_propagation(
            grad_arr[:, -1]
        )
        self.__encoder_decoder_controller.optimize(
            decoder_grads_list,
            encoder_grads_list,
            self.__learning_rate,
            1
        )
