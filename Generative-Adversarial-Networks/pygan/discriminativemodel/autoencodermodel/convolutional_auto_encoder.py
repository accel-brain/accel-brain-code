# -*- coding: utf-8 -*-
from logging import getLogger
import numpy as np
from pygan.discriminativemodel.auto_encoder_model import AutoEncoderModel
from pygan.true_sampler import TrueSampler
from pydbm.cnn.convolutionalneuralnetwork.convolutional_auto_encoder import ConvolutionalAutoEncoder as CAE
from pydbm.cnn.convolutionalneuralnetwork.convolutionalautoencoder.repelling_convolutional_auto_encoder import RepellingConvolutionalAutoEncoder
from pydbm.cnn.layerablecnn.convolution_layer import ConvolutionLayer as ConvolutionLayer1
from pydbm.cnn.layerablecnn.convolution_layer import ConvolutionLayer as ConvolutionLayer2
from pydbm.activation.tanh_function import TanhFunction
from pydbm.activation.logistic_function import LogisticFunction
from pydbm.loss.mean_squared_error import MeanSquaredError
from pydbm.optimization.optparams.adam import Adam
from pydbm.synapse.cnn_graph import CNNGraph as ConvGraph1
from pydbm.synapse.cnn_graph import CNNGraph as ConvGraph2
from pydbm.verification.verificate_function_approximation import VerificateFunctionApproximation


class ConvolutionalAutoEncoder(AutoEncoderModel):
    '''
    Stacked Convolutional Auto-Encoder as a Discriminative Model
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
        convolutional_auto_encoder=None,
        batch_size=10,
        channel=1,
        learning_rate=1e-10,
        learning_attenuate_rate=0.1,
        attenuate_epoch=50,
        opt_params=None,
        feature_matching_layer=0
    ):
        '''
        Init.
        
        Args:
            convolutional_auto_encoder:     is-a `pydbm.cnn.convolutionalneuralnetwork.convolutional_auto_encoder.ConvolutionalAutoEncoder`.
            learning_rate:                  Learning rate.
            batch_size:                     Batch size in mini-batch.
            learning_rate:                  Learning rate.
            learning_attenuate_rate:        Attenuate the `learning_rate` by a factor of this value every `attenuate_epoch`.
            attenuate_epoch:                Attenuate the `learning_rate` by a factor of `learning_attenuate_rate` every `attenuate_epoch`.
                                            Additionally, in relation to regularization,
                                            this class constrains weight matrixes every `attenuate_epoch`.

            opt_params:                     is-a `pydbm.optimization.opt_params.OptParams`.
            feature_matching_layer:         Key of layer number for feature matching forward/backward.

        '''
        if isinstance(convolutional_auto_encoder, CAE) is False and convolutional_auto_encoder is not None:
            raise TypeError("The type of `convolutional_auto_encoder` must be `pydbm.cnn.convolutionalneuralnetwork.convolutional_auto_encoder.ConvolutionalAutoEncoder`.")

        if opt_params is None:
            opt_params = Adam()
            opt_params.weight_limit = 1e+10
            opt_params.dropout_rate = 0.0

        if convolutional_auto_encoder is None:
            scale = 0.01
            conv1 = ConvolutionLayer1(
                ConvGraph1(
                    activation_function=TanhFunction(),
                    filter_num=batch_size,
                    channel=channel,
                    kernel_size=3,
                    scale=scale,
                    stride=1,
                    pad=1
                )
            )

            conv2 = ConvolutionLayer2(
                ConvGraph2(
                    activation_function=TanhFunction(),
                    filter_num=batch_size,
                    channel=batch_size,
                    kernel_size=3,
                    scale=scale,
                    stride=1,
                    pad=1
                )
            )

            convolutional_auto_encoder = RepellingConvolutionalAutoEncoder(
                layerable_cnn_list=[
                    conv1, 
                    conv2
                ],
                epochs=100,
                batch_size=batch_size,
                learning_rate=learning_rate,
                learning_attenuate_rate=learning_attenuate_rate,
                attenuate_epoch=attenuate_epoch,
                computable_loss=MeanSquaredError(),
                opt_params=opt_params,
                verificatable_result=VerificateFunctionApproximation(),
                test_size_rate=0.3,
                tol=1e-15,
                save_flag=False
            )

        self.__convolutional_auto_encoder = convolutional_auto_encoder
        self.__learning_rate = learning_rate
        self.__attenuate_epoch = attenuate_epoch
        self.__learning_attenuate_rate = learning_attenuate_rate

        self.__epoch_counter = 0
        self.__feature_matching_layer = feature_matching_layer
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
                _ = self.__convolutional_auto_encoder.back_propagation(
                    self.__delta_arr
                )
                self.__convolutional_auto_encoder.optimize(
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
        inferenced_arr = self.__convolutional_auto_encoder.inference(observed_arr)
        self.__delta_arr = self.__convolutional_auto_encoder.computable_loss.compute_delta(
            inferenced_arr, 
            observed_arr
        )
        self.__loss = self.__convolutional_auto_encoder.computable_loss.compute_loss(
            inferenced_arr, 
            observed_arr
        )
        return np.nanmean(self.__delta_arr, axis=1).mean(axis=1).mean(axis=1)

    def learn(self, grad_arr, fix_opt_flag=False):
        '''
        Update this Discriminator by ascending its stochastic gradient.

        Args:
            grad_arr:       `np.ndarray` of gradients.
            fix_opt_flag:   If `False`, no optimization in this model will be done.
        
        Returns:
            `np.ndarray` of delta or gradients.
        '''
        repeats = 1
        for i in range(1, self.__delta_arr.ndim):
            repeats *= self.__delta_arr.shape[i]

        grad_arr = np.repeat(
            grad_arr.reshape((grad_arr.shape[0], -1)), 
            repeats=repeats, 
            axis=1
        )
        grad_arr = grad_arr.reshape(self.__delta_arr.shape)

        grad_arr = self.__convolutional_auto_encoder.back_propagation(
            grad_arr
        )
        if fix_opt_flag is False:
            if ((self.__epoch_counter + 1) % self.__attenuate_epoch == 0):
                self.__learning_rate = self.__learning_rate * self.__learning_attenuate_rate

            self.__convolutional_auto_encoder.optimize(
                self.__learning_rate, 
                self.__epoch_counter
            )
            self.__epoch_counter += 1
        return grad_arr

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
        if self.__feature_matching_layer == 0:
            return self.__convolutional_auto_encoder.layerable_cnn_list[0].forward_propagate(observed_arr)
        else:
            for i in range(self.__feature_matching_layer):
                observed_arr = self.__convolutional_auto_encoder.layerable_cnn_list[i].forward_propagate(observed_arr)

        return observed_arr

    def feature_matching_backward(self, grad_arr):
        '''
        Back propagation in only first or intermediate layer
        for so-called Feature matching.

        Args:
            observed_arr:       `np.ndarray` of observed data points.

        Returns:
            `np.ndarray` of outputs.
        '''
        if self.__feature_matching_layer == 0:
            return self.__convolutional_auto_encoder.layerable_cnn_list[0].deconvolve(grad_arr)
        else:
            cnn_layer_list = self.__convolutional_auto_encoder.layerable_cnn_list[:self.__feature_matching_layer][::-1]
            for i in range(len(cnn_layer_list)):
                grad_arr = cnn_layer_list[i].deconvolve(grad_arr)
            return grad_arr

    def get_convolutional_auto_encoder(self):
        ''' getter '''
        return self.__convolutional_auto_encoder
    
    def set_convolutional_auto_encoder(self, value):
        ''' setter '''
        raise TypeError("This property must be read-only.")
    
    convolutional_auto_encoder = property(get_convolutional_auto_encoder, set_convolutional_auto_encoder)

    def set_readonly(self, value):
        ''' setter '''
        raise TypeError("This property must be read-only.")
    
    def get_pre_loss_arr(self):
        ''' getter '''
        return self.__pre_loss_arr

    pre_loss_arr = property(get_pre_loss_arr, set_readonly)

    def get_loss(self):
        ''' getter '''
        return self.__loss

    def set_loss(self, value):
        ''' setter '''
        self.__loss = value

    loss = property(get_loss, set_loss)
