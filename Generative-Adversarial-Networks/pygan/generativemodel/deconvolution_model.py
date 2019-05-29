# -*- coding: utf-8 -*-
import numpy as np
from logging import getLogger
from pygan.generative_model import GenerativeModel
from pygan.true_sampler import TrueSampler

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
from pydbm.synapse.cnn_output_graph import CNNOutputGraph


class DeconvolutionModel(GenerativeModel):
    '''
    So-called Deconvolutional Neural Network as a `GenerativeModel`.

    Deconvolution also called transposed convolutions
    "work by swapping the forward and backward passes of a convolution." (Dumoulin, V., & Visin, F. 2016, p20.)
    
    References:
        - Dumoulin, V., & V,kisin, F. (2016). A guide to convolution arithmetic for deep learning. arXiv preprint arXiv:1603.07285.
    '''

    # Computation graph which is-a `CNNOutputGraph` to compute parameters in output layer.
    __cnn_output_graph = None

    def __init__(
        self,
        deconvolution_layer_list,
        computable_loss=None,
        cnn_output_graph=None,
        opt_params=None,
        learning_rate=1e-05
    ):
        '''
        Init.

        Args:
            deconvolution_layer_list:   `list` of `DeconvolutionLayer`.
            computable_loss:            Loss function.
            cnn_output_graph:           is-a `CNNOutputGraph`.
            opt_params:                 is-a `OptParams`. If `None`, this value will be `Adam`.
            learning_rate:              Learning rate.

        '''
        for deconvolution_layer in deconvolution_layer_list:
            if isinstance(deconvolution_layer, DeconvolutionLayer) is False:
                raise TypeError()

        if cnn_output_graph is not None and isinstance(cnn_output_graph, CNNOutputGraph) is False:
            raise TypeError("The type of `cnn_output_graph` must be `CNNOutputGraph`.")

        if opt_params is None:
            opt_params = Adam()
            opt_params.dropout_rate = 0.0
        
        if isinstance(opt_params, OptParams) is False:
            raise TypeError()

        self.__deconvolution_layer_list = deconvolution_layer_list
        self.__computable_loss = computable_loss
        self.__cnn_output_graph = cnn_output_graph
        self.__learning_rate = learning_rate
        self.__attenuate_epoch = 50
        self.__opt_params = opt_params
        self.__epoch_counter = 0
        logger = getLogger("pygan")
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

        if self.__cnn_output_graph is not None:
            return self.output_forward_propagate(observed_arr)
        else:
            return observed_arr

    def output_forward_propagate(self, pred_arr):
        '''
        Forward propagation in output layer.
        
        Args:
            pred_arr:            `np.ndarray` of predicted data points.

        Returns:
            `np.ndarray` of propagated data points.
        '''
        if self.__cnn_output_graph is not None:
            _pred_arr = self.__cnn_output_graph.activating_function.activate(
                np.dot(pred_arr.reshape((pred_arr.shape[0], -1)), self.__cnn_output_graph.weight_arr) + self.__cnn_output_graph.bias_arr
            )
            self.__cnn_output_graph.hidden_arr = pred_arr
            self.__cnn_output_graph.output_arr = _pred_arr
            return _pred_arr
        else:
            return pred_arr

    def learn(self, grad_arr):
        '''
        Update this Generator by ascending its stochastic gradient.

        Args:
            grad_arr:       `np.ndarray` of gradients.
        
        Returns:
            `np.ndarray` of delta or gradients.
        '''
        if self.__cnn_output_graph is not None:
            if grad_arr.ndim != 2:
                grad_arr = grad_arr.reshape((grad_arr.shape[0], -1))

            grad_arr, output_grads_list = self.output_back_propagate(
                self.__cnn_output_graph.output_arr, 
                grad_arr
            )
            grad_arr = grad_arr.reshape((
                self.__cnn_output_graph.hidden_arr.shape[0],
                self.__cnn_output_graph.hidden_arr.shape[1],
                self.__cnn_output_graph.hidden_arr.shape[2],
                self.__cnn_output_graph.hidden_arr.shape[3]
            ))
            self.__cnn_output_graph.output_grads_list = output_grads_list

        deconvolution_layer_list = self.__deconvolution_layer_list[::-1]
        for i in range(len(deconvolution_layer_list)):
            try:
                grad_arr = deconvolution_layer_list[i].back_propagate(grad_arr)
            except:
                self.__logger.debug("Error raised in Convolution layer " + str(i + 1))
                raise

        self.__optimize(self.__learning_rate, self.__epoch_counter)
        self.__epoch_counter += 1

        return grad_arr

    def output_back_propagate(self, pred_arr, delta_arr):
        '''
        Back propagation in output layer.

        Args:
            pred_arr:            `np.ndarray` of predicted data points.
            delta_output_arr:    Delta.
        
        Returns:
            Tuple data.
            - `np.ndarray` of Delta, 
            - `list` of gradations.
        '''
        _delta_arr = np.dot(
            delta_arr,
            self.__cnn_output_graph.weight_arr.T
        )
        delta_weights_arr = np.dot(pred_arr.T, _delta_arr).T
        delta_bias_arr = np.sum(delta_arr, axis=0)

        grads_list = [
            delta_weights_arr,
            delta_bias_arr
        ]
        
        return (_delta_arr, grads_list)

    def __optimize(self, learning_rate, epoch):
        '''
        Back propagation.
        
        Args:
            learning_rate:  Learning rate.
            epoch:          Now epoch.
            
        '''
        params_list = []
        grads_list = []

        if self.__cnn_output_graph is not None:
            params_list.append(self.__cnn_output_graph.weight_arr)
            params_list.append(self.__cnn_output_graph.bias_arr)
            grads_list.append(self.__cnn_output_graph.output_grads_list[0])
            grads_list.append(self.__cnn_output_graph.output_grads_list[1])

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

        if self.__cnn_output_graph is not None:
            self.__cnn_output_graph.weight_arr = params_list.pop(0)
            self.__cnn_output_graph.bias_arr = params_list.pop(0)

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
