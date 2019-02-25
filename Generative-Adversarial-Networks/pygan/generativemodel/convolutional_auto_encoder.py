# -*- coding: utf-8 -*-
import numpy as np
from logging import getLogger, StreamHandler, NullHandler, DEBUG, ERROR

from pygan.generative_model import GenerativeModel

from pydbm.cnn.convolutionalneuralnetwork.convolutional_auto_encoder import ConvolutionalAutoEncoder as CAE
#from pydbm.cnn.convolutional_neural_network import ConvolutionalNeuralNetwork as CAE

from pydbm.cnn.layerablecnn.convolution_layer import ConvolutionLayer
from pydbm.cnn.layerable_cnn import LayerableCNN
from pydbm.cnn.feature_generator import FeatureGenerator
from pydbm.optimization.opt_params import OptParams
from pydbm.verification.interface.verificatable_result import VerificatableResult
from pydbm.loss.interface.computable_loss import ComputableLoss

from pydbm.cnn.layerablecnn.convolution_layer import ConvolutionLayer as ConvolutionLayer1
from pydbm.cnn.layerablecnn.convolution_layer import ConvolutionLayer as ConvolutionLayer2
from pydbm.synapse.cnn_graph import CNNGraph as ConvGraph1
from pydbm.synapse.cnn_graph import CNNGraph as ConvGraph2

# Tanh Function as activation function.
from pydbm.activation.tanh_function import TanhFunction
# Loss function.
from pydbm.loss.mean_squared_error import MeanSquaredError
# Adam as a optimizer.
from pydbm.optimization.optparams.sgd import SGD
# Verification.
from pydbm.verification.verificate_function_approximation import VerificateFunctionApproximation


class ConvolutionalAutoEncoder(GenerativeModel):
    '''
    Convolutional Auto-Encoder(CAE) as a `GenerativeModel`.

    CNNs are hierarchical models whose convolutional layers alternate with subsampling
    layers, reminiscent of simple and complex cells in the primary visual cortex.
    
    This class demonstrates that a CNNs can solve generalisation problems to learn 
    successful control policies from observed data points in complex 
    Reinforcement Learning environments. The network is trained with a variant of 
    the Q-learning algorithm, with stochastic gradient descent to update the weights.
    
    The Deconvolution also called transposed convolutions “work by swapping the forward and backward passes of a convolution.” (Dumoulin, V., & Visin, F. 2016, p20.)
    
    References:
        - Dumoulin, V., & V,kisin, F. (2016). A guide to convolution arithmetic for deep learning. arXiv preprint arXiv:1603.07285.
        - Masci, J., Meier, U., Cireşan, D., & Schmidhuber, J. (2011, June). Stacked convolutional auto-encoders for hierarchical feature extraction. In International Conference on Artificial Neural Networks (pp. 52-59). Springer, Berlin, Heidelberg.
        - Mnih, V., Kavukcuoglu, K., Silver, D., Graves, A., Antonoglou, I., Wierstra, D., & Riedmiller, M. (2013). Playing atari with deep reinforcement learning. arXiv preprint arXiv:1312.5602.

    '''

    def __init__(
        self,
        batch_size=20,
        learning_rate=1e-10,
        convolutional_auto_encoder=None,
        gray_scale_flag=True,
        verbose_mode=False
    ):
        '''
        Init.

        Args:
            batch_size:                     Batch size in mini-batch.
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
        self.__logger = logger

    def draw(self):
        '''
        Draws samples from the `fake` distribution.

        Returns:
            `np.ndarray` of samples.
        '''
        observed_arr = self.noise_sampler.generate()
        return self.inference(observed_arr)

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
