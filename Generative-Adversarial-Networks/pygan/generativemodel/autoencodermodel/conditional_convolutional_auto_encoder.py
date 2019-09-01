# -*- coding: utf-8 -*-
import numpy as np
from logging import getLogger
from pygan.generativemodel.auto_encoder_model import AutoEncoderModel
from pygan.true_sampler import TrueSampler
from pygan.generativemodel.conditionalgenerativemodel.conditional_convolutional_model import ConditionalConvolutionalModel
from pygan.truesampler.conditional_true_sampler import ConditionalTrueSampler
from pygan.true_sampler import TrueSampler


class ConditionalConvolutionalAutoEncoder(AutoEncoderModel):
    '''
    Conditional Convolutional Auto-Encoder(CCAE) as a `AutoEncoderModel`
    which has a `CondtionalConvolutionalModel`.

    A stack of Convolutional Auto-Encoder (Masci, J., et al., 2011) 
    forms a convolutional neural network(CNN), which are among the most successful models 
    for supervised image classification.  Each Convolutional Auto-Encoder is trained 
    using conventional on-line gradient descent without additional regularization terms.
    
    In this library, Convolutional Auto-Encoder is also based on Encoder/Decoder scheme.
    The encoder is to the decoder what the Convolution is to the Deconvolution.
    The Deconvolution also called transposed convolutions 
    "work by swapping the forward and backward passes of a convolution." (Dumoulin, V., & Visin, F. 2016, p20.)

    Also, this model has a so-called Deconvolutional Neural Network as a `Conditioner`,
    where the function of `Conditioner` is a conditional mechanism 
    to use previous knowledge to condition the generations, 
    incorporating information from previous observed data points to 
    itermediate layers of the `Generator`. In this method, this model 
    can "look back" without a recurrent unit as used in RNN or LSTM. 

    This model observes not only random noises but also any other prior
    information as a previous knowledge and outputs feature points.
    Due to the `Conditioner`, this model has the capacity to exploit
    whatever prior knowledge that is available and can be represented
    as a matrix or tensor.

    **Note** that this model defines the inputs as samples of conditions and so 
    the outputs as reconstructed samples of condtions, considering that
    those distributions, especially scales represented by biases, can be equivalents or similar.
    This definition assumes an *intuitive* implementation specific to this library.
    If you do not want to train in this way, you should use not this model but `ConditionalConvolutionalModel`.

    References:
        - Dumoulin, V., & V,kisin, F. (2016). A guide to convolution arithmetic for deep learning. arXiv preprint arXiv:1603.07285.
        - Masci, J., Meier, U., Cireşan, D., & Schmidhuber, J. (2011, June). Stacked convolutional auto-encoders for hierarchical feature extraction. In International Conference on Artificial Neural Networks (pp. 52-59). Springer, Berlin, Heidelberg.
        - Mirza, M., & Osindero, S. (2014). Conditional generative adversarial nets. arXiv preprint arXiv:1411.1784.
        - Yang, L. C., Chou, S. Y., & Yang, Y. H. (2017). MidiNet: A convolutional generative adversarial network for symbolic-domain music generation. arXiv preprint arXiv:1703.10847.

    '''

    def __init__(
        self,
        conditional_convolutional_model,
        batch_size=20,
        learning_rate=1e-10,
        learning_attenuate_rate=0.1,
        attenuate_epoch=50,
        opt_params=None
    ):
        '''
        Init.

        Args:
            conditional_convolutional_model:        is-a `ConditionalConvolutionalModel`.
            batch_size:                             Batch size.
            learning_rate:                          Learning rate.
            learning_attenuate_rate:                Attenuate the `learning_rate` by a factor of this value every `attenuate_epoch`.
            attenuate_epoch:                        Attenuate the `learning_rate` by a factor of `learning_attenuate_rate` every `attenuate_epoch`.
                                                    Additionally, in relation to regularization,
                                                    this class constrains weight matrixes every `attenuate_epoch`.
        '''
        if isinstance(conditional_convolutional_model, ConditionalConvolutionalModel) is False:
            raise TypeError("The type of `conditional_convolutional_model` must be `ConditionalConvolutionalModel`")
        
        self.__conditional_convolutional_model = conditional_convolutional_model

        self.__batch_size = batch_size
        self.__learning_rate = learning_rate
        self.__attenuate_epoch = attenuate_epoch
        self.__learning_attenuate_rate = learning_attenuate_rate

        logger = getLogger("pygan")
        self.__logger = logger

    def pre_learn(self, true_sampler, epochs=1000):
        '''
        Pre learning.

        Args:
            true_sampler:       is-a `TrueSampler`.
            epochs:             Epochs.
        '''
        if isinstance(true_sampler, ConditionalTrueSampler) is True:
            raise TypeError("The type of `true_sampler` must be not `ConditionalTrueSampler` but `TrueSampler`.")
        if isinstance(true_sampler, TrueSampler) is False:
            raise TypeError("The type of `true_sampler` must be `TrueSampler`.")

        learning_rate = self.__learning_rate
        pre_loss_list = []
        for epoch in range(epochs):
            if (epoch + 1) % self.__attenuate_epoch == 0:
                learning_rate = learning_rate * self.__learning_attenuate_rate

            try:
                observed_arr = true_sampler.draw()
                conv_arr = self.inference(observed_arr)
                if self.__conditional_convolutional_model.condition_noise_sampler is not None:
                    self.__conditional_convolutional_model.condition_noise_sampler.output_shape = conv_arr.shape
                    noise_arr = self.__conditional_convolutional_model.condition_noise_sampler.generate()
                    conv_arr += noise_arr

                deconv_arr = self.__conditional_convolutional_model.deconvolution_model.inference(conv_arr)

                grad_arr = self.__conditional_convolutional_model.cnn.computable_loss.compute_delta(observed_arr, deconv_arr)
                loss = self.__conditional_convolutional_model.cnn.computable_loss.compute_loss(observed_arr, deconv_arr)
                pre_loss_list.append(loss)
                self.__logger.debug("Epoch: " + str(epoch) + " loss: " + str(loss))

                grad_arr = self.__conditional_convolutional_model.deconvolution_model.learn(grad_arr)
                grad_arr = self.__conditional_convolutional_model.cnn.back_propagation(grad_arr)
                self.__conditional_convolutional_model.cnn.optimize(learning_rate, epoch)

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
        return self.__conditional_convolutional_model.draw()

    def inference(self, observed_arr):
        '''
        Draws samples from the `true` distribution.

        Args:
            observed_arr:     `np.ndarray` of observed data points.
        
        Returns:
            `np.ndarray` of inferenced.
        '''
        return self.__conditional_convolutional_model.inference(observed_arr)

    def learn(self, grad_arr):
        '''
        Update this Generator by ascending its stochastic gradient.

        Args:
            grad_arr:       `np.ndarray` of gradients.
        
        Returns:
            `np.ndarray` of delta or gradients.
        '''
        return self.__conditional_convolutional_model.learn(grad_arr)

    def update(self):
        '''
        Update the encoder and the decoder
        to minimize the reconstruction error of the inputs.

        This model defines the inputs as samples of conditions and so 
        the outputs as reconstructed samples of condtions, considering that
        those distributions, especially scales represented by biases, are equivalents or similar.

        Returns:
            `np.ndarray` of the reconstruction errors.
        '''
        if ((self.__conditional_convolutional_model.epoch_counter + 1) % self.__attenuate_epoch == 0):
            self.__learning_rate = self.__learning_rate * self.__learning_attenuate_rate

        observed_arr = self.__conditional_convolutional_model.extract_conditions()
        conv_arr = self.inference(observed_arr)
        if self.__conditional_convolutional_model.condition_noise_sampler is not None:
            self.__conditional_convolutional_model.condition_noise_sampler.output_shape = conv_arr.shape
            noise_arr = self.__conditional_convolutional_model.condition_noise_sampler.generate()
            conv_arr += noise_arr

        inferenced_arr = self.__conditional_convolutional_model.deconvolution_model.inference(conv_arr)

        error_arr = self.__conditional_convolutional_model.cnn.computable_loss.compute_loss(
            observed_arr,
            inferenced_arr
        )

        delta_arr = self.__conditional_convolutional_model.cnn.computable_loss.compute_delta(
            observed_arr,
            inferenced_arr
        )

        delta_arr = self.__conditional_convolutional_model.deconvolution_model.learn(delta_arr)
        delta_arr = self.__conditional_convolutional_model.cnn.back_propagation(delta_arr)
        self.__conditional_convolutional_model.cnn.optimize(self.__learning_rate, self.__conditional_convolutional_model.epoch_counter)

        self.__conditional_convolutional_model.epoch_counter += 1
        return error_arr

    def switch_inferencing_mode(self, inferencing_mode=True):
        '''
        Set inferencing mode in relation to concrete regularizations.

        Args:
            inferencing_mode:       Inferencing mode or not.
        '''
        self.__conditional_convolutional_model.switch_inferencing_mode(inferencing_mode)

    def get_conditional_convolutional_model(self):
        ''' getter '''
        return self.__conditional_convolutional_model
    
    def set_conditional_convolutional_model(self, value):
        ''' setter '''
        self.__conditional_convolutional_model = value

    conditional_convolutional_model = property(get_conditional_convolutional_model, set_conditional_convolutional_model)

    def set_readonly(self, value):
        ''' setter '''
        raise TypeError("This property must be read-only.")
    
    def get_pre_loss_arr(self):
        ''' getter '''
        return self.__pre_loss_arr

    pre_loss_arr = property(get_pre_loss_arr, set_readonly)
