# -*- coding: utf-8 -*-
import numpy as np
from logging import getLogger, StreamHandler, NullHandler, DEBUG, ERROR

from pygan.generative_model import GenerativeModel

from pydbm.cnn.convolutional_neural_network import ConvolutionalNeuralNetwork
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

# Loss function.
from pydbm.loss.mean_squared_error import MeanSquaredError
# Adam as a optimizer.
from pydbm.optimization.optparams.sgd import SGD
# Verification.
from pydbm.verification.verificate_function_approximation import VerificateFunctionApproximation


class CNNModel(GenerativeModel):
    '''
    Convolutional Neural Network as a Generator.

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
        batch_size,
        layerable_cnn_list,
        learning_rate=1e-05,
        computable_loss=None,
        opt_params=None,
        norm_mode="z_score",
        verificatable_result=None,
        pre_learned_path_list=None,
        cnn=None,
        verbose_mode=False
    ):
        '''
        Init.

        Args:
            batch_size:                     Batch size in mini-batch.
            layerable_cnn_list:             `list` of `LayerableCNN`.
            learning_rate:                  Learning rate.
            computable_loss:                is-a `ComputableLoss`.
            opt_params:                     is-a `OptParams`.
            norm_mode:                      How to normalize generated values.
                                            - `z_score`: Z-Score normalization.
                                            - `min_max`: Min-max normalization.
                                            - `tanh`: Normalization by tanh function.

            verificatable_result:           is-a `VerificateFunctionApproximation`.
            pre_learned_path_list:          `list` of file path that stored pre-learned parameters.
                                            This parameters will be refered only when `cnn` is `None`.

            cnn:                            is-a `ConvolutionalNeuralNetwork` as a model in this class.
                                            If not `None`, `self.__cnn` will be overrided by this `cnn`.
                                            If `None`, this class initialize `ConvolutionalNeuralNetwork`
                                            by default hyper parameters.

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
        if computable_loss is None:
            computable_loss = MeanSquaredError()
        if verificatable_result is None:
            verificatable_result = VerificateFunctionApproximation()
        if opt_params is None:
            opt_params = SGD()
            opt_params.weight_limit = 0.5
            opt_params.dropout_rate = 0.0

        if cnn is None:
            cnn = ConvolutionalNeuralNetwork(
                # The `list` of `ConvolutionLayer`.
                layerable_cnn_list=layerable_cnn_list,
                # The number of epochs in mini-batch training.
                epochs=200,
                # The batch size.
                batch_size=batch_size,
                # Learning rate.
                learning_rate=learning_rate,
                # Loss function.
                computable_loss=computable_loss,
                # Optimizer.
                opt_params=opt_params,
                # Verification.
                verificatable_result=verificatable_result,
                # Pre-learned parameters.
                pre_learned_path_list=pre_learned_path_list,
                # Others.
                learning_attenuate_rate=0.1,
                attenuate_epoch=50
            )

        self.__cnn = cnn
        self.__batch_size = batch_size
        self.__computable_loss = computable_loss
        self.__learning_rate = learning_rate
        self.__verbose_mode = verbose_mode
        self.__q_shape = None
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
        return self.__cnn.inference(observed_arr)

    def learn(self, grad_arr):
        '''
        Update this Discriminator by ascending its stochastic gradient.

        Args:
            grad_arr:   `np.ndarray` of gradients.
        
        '''
        _ = self.__cnn.back_propagation(grad_arr)
        self.__cnn.optimize(self.__learning_rate, 1)
