# -*- coding: utf-8 -*-
from logging import getLogger
import numpy as np
from pygan.discriminativemodel.auto_encoder_model import AutoEncoderModel
from pygan.true_sampler import TrueSampler
from pydbm.rnn.encoder_decoder_controller import EncoderDecoderController


class EncoderDecoderModel(AutoEncoderModel):
    '''
    Encoder/Decoder based on LSTM as a Discriminative Model
    which discriminates `true` from `fake`.

    The Energy-based GAN framework considers the discriminator as an energy function, 
    which assigns low energy values to real data and high to fake data. 
    The generator is a trainable parameterized function that produces 
    samples in regions to which the discriminator assigns low energy. 

    References:
        - Manisha, P., & Gujar, S. (2018). Generative Adversarial Networks (GANs): What it can generate and What it cannot?. arXiv preprint arXiv:1804.00140.
        - Zhao, J., Mathieu, M., & LeCun, Y. (2016). Energy-based generative adversarial network. arXiv preprint arXiv:1609.03126.
    '''

    def __init__(
        self, 
        encoder_decoder_controller,
        seq_len=10,
        learning_rate=1e-10,
        learning_attenuate_rate=0.1,
        attenuate_epoch=50,
    ):
        '''
        Init.
        
        Args:
            encoder_decoder_controller:     is-a `EncoderDecoderController`.
            seq_len:                        The length of sequence.
            learning_rate:                  Learning rate.
            learning_attenuate_rate:        Attenuate the `learning_rate` by a factor of this value every `attenuate_epoch`.
            attenuate_epoch:                Attenuate the `learning_rate` by a factor of `learning_attenuate_rate` every `attenuate_epoch`.
                                            Additionally, in relation to regularization,
                                            this class constrains weight matrixes every `attenuate_epoch`.

        '''
        if isinstance(encoder_decoder_controller, EncoderDecoderController) is False:
            raise TypeError()

        self.__encoder_decoder_controller = encoder_decoder_controller
        self.__seq_len = seq_len
        self.__learning_rate = learning_rate
        self.__attenuate_epoch = attenuate_epoch
        self.__learning_attenuate_rate = learning_attenuate_rate

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
        

        learning_rate = self.__learning_rate

        pre_loss_list = []
        for epoch in range(epochs):
            if (epoch + 1) % self.__attenuate_epoch == 0:
                learning_rate = learning_rate * self.__learning_attenuate_rate
            try:
                observed_arr = true_sampler.draw()
                _ = self.inference(observed_arr)
                pre_loss_list.append(self.__loss)
                self.__logger.debug("Epoch: " + str(epoch) + " loss: " + str(self.__loss))
                decoder_grads_list, _, encoder_grads_list = self.__encoder_decoder_controller.back_propagation(
                    self.__delta_arr
                )
                self.__encoder_decoder_controller.optimize(
                    decoder_grads_list,
                    encoder_grads_list,
                    learning_rate, 
                    epoch
                )

            except KeyboardInterrupt:
                self.__logger.debug("Interrupt.")
                break

        self.__pre_loss_arr = np.array(pre_loss_list)

    def inference(self, observed_arr):
        '''
        Draws samples from the `fake` distribution.

        Args:
            observed_arr:     `np.ndarray` of observed data points.
        
        Returns:
            `np.ndarray` of inferenced.
        '''
        self.__observed_arr = observed_arr
        inferenced_arr = self.__encoder_decoder_controller.inference(observed_arr)
        self.__delta_arr = self.__encoder_decoder_controller.get_reconstruction_error()
        self.__loss = (self.__delta_arr ** 2).mean()
        return np.nanmean(self.__delta_arr, axis=1).mean(axis=1)

    def learn(self, grad_arr, fix_opt_flag=False):
        '''
        Update this Discriminator by ascending its stochastic gradient.

        Args:
            grad_arr:       `np.ndarray` of gradients.
            fix_opt_flag:   If `False`, no optimization in this model will be done.
        
        Returns:
            `np.ndarray` of delta or gradients.
        '''
        grad_arr = np.repeat(
            grad_arr.reshape((grad_arr.shape[0], -1)), 
            repeats=self.__delta_arr.shape[1] * self.__delta_arr.shape[2], 
            axis=1
        )
        grad_arr = grad_arr.reshape(self.__delta_arr.shape)

        decoder_grads_list, encoder_delta_arr, encoder_grads_list = self.__encoder_decoder_controller.back_propagation(
            grad_arr
        )
        if fix_opt_flag is False:
            if ((self.__epoch_counter + 1) % self.__attenuate_epoch == 0):
                self.__learning_rate = self.__learning_rate * self.__learning_attenuate_rate

            self.__encoder_decoder_controller.optimize(
                decoder_grads_list,
                encoder_grads_list,
                self.__learning_rate, 
                self.__epoch_counter
            )

            self.__epoch_counter += 1
        return encoder_delta_arr

    def feature_matching_forward(self, observed_arr):
        '''
        Forward propagation in only first or intermediate layer
        for so-called Feature matching.

        Like C-RNN-GAN(Mogren, O. 2016), this model chooses 
        the last layer before the output layer in this Discriminator.

        Args:
            observed_arr:       `np.ndarray` of observed data points.

        Returns:
            `np.ndarray` of outputs.
        '''
        return self.__encoder_decoder_controller.encoder.hidden_forward_propagate(observed_arr)

    def feature_matching_backward(self, grad_arr):
        '''
        Back propagation in only first or intermediate layer
        for so-called Feature matching.

        Args:
            observed_arr:       `np.ndarray` of observed data points.

        Returns:
            `np.ndarray` of outputs.
        '''
        grad_arr, _, _ = self.__encoder_decoder_controller.encoder.hidden_back_propagate(grad_arr[:, -1])
        return grad_arr

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
