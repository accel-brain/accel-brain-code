# -*- coding: utf-8 -*-
import numpy as np
from logging import getLogger

from pygan.generativemodel.auto_encoder_model import AutoEncoderModel
from pygan.true_sampler import TrueSampler

from pydbm.rnn.encoder_decoder_controller import EncoderDecoderController

from pydbm.activation.relu_function import ReLuFunction
from pydbm.activation.tanh_function import TanhFunction
from pydbm.activation.logistic_function import LogisticFunction
from pydbm.loss.mean_squared_error import MeanSquaredError
from pydbm.optimization.optparams.adam import Adam
from pydbm.optimization.optparams.sgd import SGD
from pydbm.verification.verificate_function_approximation import VerificateFunctionApproximation


class EncoderDecoderModel(AutoEncoderModel):
    '''
    Encoder/Decoder based on LSTM as a Generator.

    This library regards the Encoder/Decoder based on LSTM as an Auto-Encoder.

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
        learning_rate=1e-10,
        join_io_flag=False
    ):
        '''
        Init.
        
        Args:
            encoder_decoder_controller:         is-a `EncoderDecoderController`.
            seq_len:                            The length of sequence.
            learning_rate:                      Learning rate.
            join_io_flag:                       If this value and value of `output_activating_function` is not `None`,
                                                This model outputs tensors combining observed data points and inferenced data
                                                in a series direction.

        '''
        if isinstance(encoder_decoder_controller, EncoderDecoderController) is False:
            raise TypeError()

        self.__encoder_decoder_controller = encoder_decoder_controller
        self.__seq_len = seq_len
        self.__learning_rate = learning_rate
        self.__join_io_flag = join_io_flag
        self.__epoch_counter = 0
        logger = getLogger("pygan")
        self.__logger = logger

    def pre_learn(self, true_sampler, epochs=1000):
        '''
        Pre learning.

        Args:
            true_sampler:       is-a `TrueSampler`.
            epochs:             Epochs.
        '''
        if isinstance(true_sampler, TrueSampler) is False:
            raise TypeError("The type of `true_sampler` must be `TrueSampler`.")
        
        pre_loss_list = []
        for epoch in range(epochs):
            try:
                observed_arr = true_sampler.draw()
                inferenced_arr = self.inference(observed_arr)
                if observed_arr.size != inferenced_arr.size:
                    raise ValueError("In pre-learning, the rank or shape of observed data points and feature points in last layer must be equivalent.")
                grad_arr = self.__encoder_decoder_controller.computable_loss.compute_delta(observed_arr, inferenced_arr)
                loss = self.__encoder_decoder_controller.computable_loss.compute_loss(observed_arr, inferenced_arr)
                pre_loss_list.append(loss)
                self.__logger.debug("Epoch: " + str(epoch) + " loss: " + str(loss))
                self.learn(grad_arr)
            except KeyboardInterrupt:
                self.__logger.debug("Interrupt.")
                break

        self.__pre_loss_arr = np.array(pre_loss_list)

    def draw(self):
        '''
        Draws samples from the `fake` distribution.

        Returns:
            `np.ndarray` of samples.
        '''
        observed_arr = self.noise_sampler.generate()
        _ = self.__encoder_decoder_controller.encoder.inference(observed_arr)
        inferenced_arr = self.__encoder_decoder_controller.encoder.get_feature_points()
        if self.__join_io_flag is False:
            return inferenced_arr
        else:
            return np.concatenate([observed_arr, inferenced_arr], axis=1)

    def inference(self, observed_arr):
        '''
        Draws samples from the `fake` distribution.

        Args:
            observed_arr:     `np.ndarray` of observed data points.
        
        Returns:
            `np.ndarray` of inferenced.
        '''
        inferenced_arr = self.__encoder_decoder_controller.inference(observed_arr)
        return inferenced_arr

    def learn(self, grad_arr):
        '''
        Update this Discriminator by ascending its stochastic gradient.

        Args:
            grad_arr:   `np.ndarray` of gradients.

        Returns:
            `np.ndarray` of delta or gradients.

        '''
        if self.__join_io_flag is True:
            grad_arr = grad_arr[:, self.__seq_len:]

        encoder_delta_arr, _, encoder_grads_list = self.__encoder_decoder_controller.encoder.hidden_back_propagate(
            grad_arr[:, -1]
        )
        encoder_grads_list.insert(0, None)
        encoder_grads_list.insert(0, None)

        self.__encoder_decoder_controller.encoder.optimize(
            encoder_grads_list, 
            self.__learning_rate,
            self.__epoch_counter
        )

        self.__epoch_counter += 1
        return encoder_delta_arr

    def update(self):
        '''
        Update the encoder and the decoder
        to minimize the reconstruction error of the inputs.

        Returns:
            `np.ndarray` of the reconstruction errors.
        '''
        observed_arr = self.noise_sampler.generate()
        inferenced_arr = self.inference(observed_arr)

        error_arr = self.__encoder_decoder_controller.computable_loss.compute_loss(
            observed_arr,
            inferenced_arr
        )
        delta_arr = self.__encoder_decoder_controller.computable_loss.compute_delta(
            observed_arr,
            inferenced_arr
        )
        decoder_grads_list, encoder_delta_arr, encoder_grads_list = self.__encoder_decoder_controller.back_propagation(
            delta_arr
        )
        self.__encoder_decoder_controller.optimize(
            decoder_grads_list,
            encoder_grads_list,
            self.__learning_rate, 
            self.__epoch_counter
        )

        self.__epoch_counter += 1
        return error_arr

    def get_encoder_decoder_controller(self):
        ''' getter '''
        return self.__encoder_decoder_controller
    
    def set_encoder_decoder_controller(self, value):
        ''' setter '''
        raise TypeError("This property must be read-only.")
    
    encoder_decoder_controller = property(get_encoder_decoder_controller, set_encoder_decoder_controller)

    def set_readonly(self, value):
        ''' setter '''
        raise TypeError("This property must be read-only.")
    
    def get_pre_loss_arr(self):
        ''' getter '''
        return self.__pre_loss_arr

    pre_loss_arr = property(get_pre_loss_arr, set_readonly)
