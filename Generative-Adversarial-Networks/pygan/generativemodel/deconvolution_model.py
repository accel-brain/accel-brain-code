# -*- coding: utf-8 -*-
import numpy as np
from logging import getLogger, StreamHandler, NullHandler, DEBUG, ERROR
from pygan.generative_model import GenerativeModel

from pydbm.cnn.convolutionalneuralnetwork.convolutional_auto_encoder import ConvolutionalAutoEncoder as CAE
from pydbm.synapse.cnn_graph import CNNGraph as ConvGraph1
from pydbm.synapse.cnn_graph import CNNGraph as ConvGraph2

from pydbm.cnn.layerablecnn.convolutionlayer.deconvolution_layer import DeconvolutionLayer
from pydbm.synapse.cnn_graph import CNNGraph as DeCNNGraph

from pydbm.activation.relu_function import ReLuFunction
from pydbm.activation.tanh_function import TanhFunction
from pydbm.activation.logistic_function import LogisticFunction
from pydbm.loss.mean_squared_error import MeanSquaredError
from pydbm.optimization.optparams.adam import Adam
from pydbm.optimization.opt_params import OptParams
from pydbm.verification.verificate_function_approximation import VerificateFunctionApproximation


class DeconvolutionModel(GenerativeModel):
    '''
    So-called Deconvolutional Neural Network as a `GenerativeModel`.

    Deconvolution also called transposed convolutions
    "work by swapping the forward and backward passes of a convolution." (Dumoulin, V., & Visin, F. 2016, p20.)
    
    References:
        - Dumoulin, V., & V,kisin, F. (2016). A guide to convolution arithmetic for deep learning. arXiv preprint arXiv:1603.07285.
    '''

    def __init__(
        self,
        deconvolution_layer_list,
        opt_params=None,
        learning_rate=1e-05,
        verbose_mode=False
    ):
        '''
        Init.

        Args:
            deconvolution_layer_list:   `list` of `DeconvolutionLayer`.
            opt_params:                 is-a `OptParams`. If `None`, this value will be `Adam`.
            learning_rate:              Learning rate.
            verbose_mode:               Verbose mode or not.

        '''
        for deconvolution_layer in deconvolution_layer_list:
            if isinstance(deconvolution_layer, DeconvolutionLayer) is False:
                raise TypeError()

        if opt_params is None:
            opt_params = Adam()
            opt_params.dropout_rate = 0.0
        
        if isinstance(opt_params, OptParams) is False:
            raise TypeError()

        logger = getLogger("pydbm")
        handler = StreamHandler()
        if verbose_mode is True:
            handler.setLevel(DEBUG)
            logger.setLevel(DEBUG)
        else:
            handler.setLevel(ERROR)
            logger.setLevel(ERROR)

        logger.addHandler(handler)

        self.__deconvolution_layer_list = deconvolution_layer_list
        self.__learning_rate = learning_rate
        self.__attenuate_epoch = 50
        self.__opt_params = opt_params
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
        Draws samples from the `fake` distribution.

        Args:
            observed_arr:     `np.ndarray` of observed data points.
        
        Returns:
            `np.ndarray` of inferenced.
        '''
        for i in range(len(self.__deconvolution_layer_list)):
            try:
                observed_arr = self.__deconvolution_layer_list[i].forward_propagate(observed_arr)
            except:
                self.__logger.debug("Error raised in Deconvolution layer " + str(i + 1))
                raise

        return observed_arr

    def learn(self, grad_arr, fix_opt_flag=False):
        '''
        Update this Discriminator by ascending its stochastic gradient.

        Args:
            grad_arr:       `np.ndarray` of gradients.
            fix_opt_flag:   If `False`, no optimization in this model will be done.
        
        Returns:
            `np.ndarray` of delta or gradients.
        '''
        deconvolution_layer_list = self.__deconvolution_layer_list[::-1]
        for i in range(len(deconvolution_layer_list)):
            try:
                grad_arr = deconvolution_layer_list[i].back_propagate(grad_arr)
            except:
                self.__logger.debug("Error raised in Convolution layer " + str(i + 1))
                raise

        if fix_opt_flag is False:
            self.__optimize(self.__learning_rate, 1)
        
        return grad_arr

    def __optimize(self, learning_rate, epoch):
        '''
        Back propagation.
        
        Args:
            learning_rate:  Learning rate.
            epoch:          Now epoch.
            
        '''
        params_list = []
        grads_list = []

        for i in range(len(self.__deconvolution_layer_list)):
            if self.__deconvolution_layer_list[i].delta_weight_arr.shape[0] > 0:
                params_list.append(self.__deconvolution_layer_list[i].graph.weight_arr)
                grads_list.append(self.__deconvolution_layer_list[i].delta_weight_arr)

        for i in range(len(self.__deconvolution_layer_list)):
            if self.__deconvolution_layer_list[i].delta_bias_arr.shape[0] > 0:
                params_list.append(self.__deconvolution_layer_list[i].graph.bias_arr)
                grads_list.append(self.__deconvolution_layer_list[i].delta_bias_arr)

        params_list = self.__opt_params.optimize(
            params_list,
            grads_list,
            learning_rate
        )

        i = 0
        for i in range(len(self.__deconvolution_layer_list)):
            if self.__deconvolution_layer_list[i].delta_weight_arr.shape[0] > 0:
                self.__deconvolution_layer_list[i].graph.weight_arr = params_list.pop(0)
                if ((epoch + 1) % self.__attenuate_epoch == 0):
                    self.__deconvolution_layer_list[i].graph.weight_arr = self.__opt_params.constrain_weight(
                        self.__deconvolution_layer_list[i].graph.weight_arr
                    )

        for i in range(len(self.__deconvolution_layer_list)):
            if self.__deconvolution_layer_list[i].delta_bias_arr.shape[0] > 0:
                self.__deconvolution_layer_list[i].graph.bias_arr = params_list.pop(0)

        for i in range(len(self.__deconvolution_layer_list)):
            if self.__deconvolution_layer_list[i].delta_weight_arr.shape[0] > 0:
                if self.__deconvolution_layer_list[i].delta_bias_arr.shape[0] > 0:
                    self.__deconvolution_layer_list[i].reset_delta()

    def get_deconvolution_layer_list(self):
        ''' getter '''
        return self.__deconvolution_layer_list
    
    def set_deconvolution_layer_list(self, value):
        ''' setter '''
        raise TypeError("This property must be read-only.")

    deconvolution_layer_list = property(get_deconvolution_layer_list, set_deconvolution_layer_list)
