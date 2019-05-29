# -*- coding: utf-8 -*-
import numpy as np
from logging import getLogger

from pygan.discriminative_model import DiscriminativeModel

from pydbm.optimization.opt_params import OptParams
from pydbm.verification.interface.verificatable_result import VerificatableResult
from pydbm.loss.interface.computable_loss import ComputableLoss

from pydbm.cnn.convolutional_neural_network import ConvolutionalNeuralNetwork
from pydbm.cnn.layerable_cnn import LayerableCNN
from pydbm.synapse.cnn_graph import CNNGraph as ConvGraph1
from pydbm.synapse.cnn_graph import CNNGraph as ConvGraph2
from pydbm.activation.relu_function import ReLuFunction
from pydbm.activation.tanh_function import TanhFunction
from pydbm.activation.logistic_function import LogisticFunction
from pydbm.loss.mean_squared_error import MeanSquaredError
from pydbm.optimization.optparams.adam import Adam
from pydbm.optimization.optparams.sgd import SGD
from pydbm.verification.verificate_function_approximation import VerificateFunctionApproximation


class CNNModel(DiscriminativeModel):
    '''
    Convolutional Neural Network as a Discriminator.
    '''

    def __init__(
        self,
        batch_size,
        layerable_cnn_list,
        cnn_output_graph,
        learning_rate=1e-05,
        computable_loss=None,
        opt_params=None,
        verificatable_result=None,
        cnn=None,
        feature_matching_layer=0
    ):
        '''
        Init.

        Args:
            batch_size:                     Batch size in mini-batch.
            layerable_cnn_list:             `list` of `LayerableCNN`.
            cnn_output_graph:               is-a `CNNOutputGraph`.
            learning_rate:                  Learning rate.
            computable_loss:                is-a `ComputableLoss`.
                                            This parameters will be refered only when `cnn` is `None`.

            opt_params:                     is-a `OptParams`.
                                            This parameters will be refered only when `cnn` is `None`.

            verificatable_result:           is-a `VerificateFunctionApproximation`.
                                            This parameters will be refered only when `cnn` is `None`.

            cnn:                            is-a `ConvolutionalNeuralNetwork` as a model in this class.
                                            If not `None`, `self.__cnn` will be overrided by this `cnn`.
                                            If `None`, this class initialize `ConvolutionalNeuralNetwork`
                                            by default hyper parameters.

            feature_matching_layer:         Key of layer number for feature matching forward/backward.

        '''
        for layerable_cnn in layerable_cnn_list:
            if isinstance(layerable_cnn, LayerableCNN) is False:
                raise TypeError()


        self.__layerable_cnn_list = layerable_cnn_list
        self.__learning_rate = learning_rate
        self.__opt_params = opt_params

        if cnn is None:
            if computable_loss is None:
                computable_loss = MeanSquaredError()
            if isinstance(computable_loss, ComputableLoss) is False:
                raise TypeError()

            if verificatable_result is None:
                verificatable_result = VerificateFunctionApproximation()
            if isinstance(verificatable_result, VerificatableResult) is False:
                raise TypeError()

            if opt_params is None:
                opt_params = Adam()
                opt_params.weight_limit = 0.5
                opt_params.dropout_rate = 0.0
            if isinstance(opt_params, OptParams) is False:
                raise TypeError()

            cnn = ConvolutionalNeuralNetwork(
                layerable_cnn_list=layerable_cnn_list,
                computable_loss=computable_loss,
                opt_params=opt_params,
                verificatable_result=verificatable_result,
                epochs=100,
                batch_size=batch_size,
                learning_rate=learning_rate,
                learning_attenuate_rate=0.1,
                test_size_rate=0.3,
                tol=1e-15,
                tld=100.0,
                save_flag=False,
                pre_learned_path_list=None
            )
            cnn.setup_output_layer(cnn_output_graph)

        self.__cnn = cnn
        self.__batch_size = batch_size
        self.__computable_loss = computable_loss
        self.__learning_rate = learning_rate
        self.__q_shape = None
        self.__loss_list = []
        self.__feature_matching_layer = feature_matching_layer
        self.__epoch_counter = 0
        logger = getLogger("pygan")
        self.__logger = logger

    def inference(self, observed_arr):
        '''
        Draws samples from the `true` distribution.

        Args:
            observed_arr:     `np.ndarray` of observed data points.
        
        Returns:
            `np.ndarray` of inferenced.
        '''
        return self.__cnn.inference(observed_arr)

    def learn(self, grad_arr, fix_opt_flag=False):
        '''
        Update this Discriminator by ascending its stochastic gradient.

        Args:
            grad_arr:       `np.ndarray` of gradients.
            fix_opt_flag:   If `False`, no optimization in this model will be done.
        
        Returns:
            `np.ndarray` of delta or gradients.
        '''
        if grad_arr.ndim != 2:
            grad_arr = grad_arr.reshape((grad_arr.shape[0], -1))
        delta_arr = self.__cnn.back_propagation(grad_arr)
        if fix_opt_flag is False:
            self.__cnn.optimize(self.__learning_rate, self.__epoch_counter)
            self.__epoch_counter += 1
        return delta_arr

    def feature_matching_forward(self, observed_arr):
        '''
        Forward propagation in only first or intermediate layer
        for so-called Feature matching.

        Args:
            observed_arr:       `np.ndarray` of observed data points.

        Returns:
            `np.ndarray` of outputs.
        '''
        if self.__feature_matching_layer == 0:
            return self.__cnn.layerable_cnn_list[0].forward_propagate(observed_arr)
        else:
            for i in range(self.__feature_matching_layer):
                observed_arr = self.__cnn.layerable_cnn_list[i].forward_propagate(observed_arr)

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
            return self.__cnn.layerable_cnn_list[0].deconvolve(grad_arr)
        else:
            cnn_layer_list = self.__cnn.layerable_cnn_list[:self.__feature_matching_layer][::-1]
            for i in range(len(cnn_layer_list)):
                grad_arr = cnn_layer_list[i].deconvolve(grad_arr)
            return grad_arr

    def get_cnn(self):
        ''' getter '''
        return self.__cnn
    
    def set_cnn(self, value):
        ''' setter '''
        raise TypeError("This property must be read-only.")
    
    cnn = property(get_cnn, set_cnn)
