# -*- coding: utf-8 -*-
import numpy as np
from logging import getLogger, StreamHandler, NullHandler, DEBUG, ERROR

from pygan.generativemodel.auto_encoder_model import AutoEncoderModel
from pydbm.cnn.convolutionalneuralnetwork.convolutional_auto_encoder import ConvolutionalAutoEncoder as CAE
from pydbm.cnn.layerablecnn.convolution_layer import ConvolutionLayer as ConvolutionLayer1
from pydbm.cnn.layerablecnn.convolution_layer import ConvolutionLayer as ConvolutionLayer2
from pydbm.synapse.cnn_graph import CNNGraph as ConvGraph1
from pydbm.synapse.cnn_graph import CNNGraph as ConvGraph2
from pydbm.activation.relu_function import ReLuFunction
from pydbm.activation.tanh_function import TanhFunction
from pydbm.activation.logistic_function import LogisticFunction
from pydbm.loss.mean_squared_error import MeanSquaredError
from pydbm.optimization.optparams.adam import Adam
from pydbm.optimization.optparams.sgd import SGD
from pydbm.verification.verificate_function_approximation import VerificateFunctionApproximation


class ConvolutionalAutoEncoder(AutoEncoderModel):
    '''
    Convolutional Auto-Encoder(CAE) as a `GenerativeModel`.

    A stack of Convolutional Auto-Encoder (Masci, J., et al., 2011) 
    forms a convolutional neural network(CNN), which are among the most successful models 
    for supervised image classification.  Each Convolutional Auto-Encoder is trained 
    using conventional on-line gradient descent without additional regularization terms.
    
    In this library, Convolutional Auto-Encoder is also based on Encoder/Decoder scheme.
    The encoder is to the decoder what the Convolution is to the Deconvolution.
    The Deconvolution also called transposed convolutions 
    "work by swapping the forward and backward passes of a convolution." (Dumoulin, V., & Visin, F. 2016, p20.)

    References:
        - Dumoulin, V., & V,kisin, F. (2016). A guide to convolution arithmetic for deep learning. arXiv preprint arXiv:1603.07285.
        - Masci, J., Meier, U., Cireşan, D., & Schmidhuber, J. (2011, June). Stacked convolutional auto-encoders for hierarchical feature extraction. In International Conference on Artificial Neural Networks (pp. 52-59). Springer, Berlin, Heidelberg.

    '''

    def __init__(
        self,
        batch_size=20,
        learning_rate=1e-10,
        convolutional_auto_encoder=None,
        gray_scale_flag=True,
        norm_mode="z_score",
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

        if convolutional_auto_encoder is None:
            if gray_scale_flag is True:
                channel = 1
            else:
                channel = 3
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

            convolutional_auto_encoder = CAE(
                layerable_cnn_list=[
                    conv1, 
                    conv2
                ],
                epochs=100,
                batch_size=batch_size,
                learning_rate=1e-05,
                learning_attenuate_rate=0.1,
                attenuate_epoch=25,
                computable_loss=MeanSquaredError(),
                opt_params=SGD(),
                verificatable_result=VerificateFunctionApproximation(),
                test_size_rate=0.3,
                tol=1e-15,
                save_flag=False
            )
        self.__convolutional_auto_encoder = convolutional_auto_encoder
        self.__learning_rate = learning_rate
        self.__verbose_mode = verbose_mode
        self.__norm_mode = norm_mode

    def draw(self):
        '''
        Draws samples from the `fake` distribution.

        Returns:
            `np.ndarray` of samples.
        '''
        observed_arr = self.noise_sampler.generate()
        _ = self.inference(observed_arr)
        arr = self.__convolutional_auto_encoder.extract_feature_points_arr()
        if self.__norm_mode == "z_score":
            for i in range(arr.shape[0]):
                arr[i] = (arr[i] - arr[i].mean()) / arr[i].std()
        elif self.__norm_mode == "min_max":
            for i in range(arr.shape[0]):
                arr[i] = (arr[i] - arr[i].min()) / (arr[i].max() - arr[i].min())
        elif self.__norm_mode == "tanh":
            arr = np.tanh(arr)

        return np.expand_dims(
            np.nansum(
                arr,
                axis=1
            ) / arr.shape[1],
            axis=1
        )

    def inference(self, observed_arr):
        '''
        Draws samples from the `true` distribution.

        Args:
            observed_arr:     `np.ndarray` of observed data points.
        
        Returns:
            `np.ndarray` of inferenced.
            `0` is to `1` what `fake` is to `true`.
        '''
        return self.__convolutional_auto_encoder.inference(observed_arr)

    def learn(self, grad_arr):
        '''
        Update this Discriminator by ascending its stochastic gradient.

        Args:
            grad_arr:   `np.ndarray` of gradients.
        
        '''
        delta_arr = grad_arr
        if self.__convolutional_auto_encoder.opt_params.dropout_rate > 0:
            hidden_activity_arr = delta_arr.reshape((delta_arr.shape[0], -1))
            hidden_activity_arr = self.__convolutional_auto_encoder.opt_params.de_dropout(hidden_activity_arr)
            delta_arr = hidden_activity_arr.reshape((
                delta_arr.shape[0],
                delta_arr.shape[1],
                delta_arr.shape[2],
                delta_arr.shape[3]
            ))

        layerable_cnn_list = self.__convolutional_auto_encoder.layerable_cnn_list[::-1]
        for i in range(len(layerable_cnn_list)):
            try:
                grad_arr = layerable_cnn_list[i].back_propagate(grad_arr)
            except:
                self.__logger.debug(
                    "Delta computation raised an error in CNN layer " + str(len(layerable_cnn_list) - i)
                )
                raise

        self.__convolutional_auto_encoder.optimize(self.__learning_rate, 1)

    def update(self):
        '''
        Update the encoder and the decoder
        to minimize the reconstruction error of the inputs.

        Returns:
            `np.ndarray` of the reconstruction errors.
        '''
        observed_arr = self.noise_sampler.generate()
        inferenced_arr = self.inference(observed_arr)

        error_arr = self.__convolutional_auto_encoder.computable_loss.compute_loss(
            observed_arr,
            inferenced_arr
        )        
        delta_arr = self.__convolutional_auto_encoder.computable_loss.compute_delta(
            observed_arr,
            inferenced_arr
        )
        delta_arr = self.__convolutional_auto_encoder.back_propagation(delta_arr)
        self.__convolutional_auto_encoder.optimize(self.__learning_rate, 1)

        return error_arr
