# -*- coding: utf-8 -*-
import numpy as np
from logging import getLogger, StreamHandler, NullHandler, DEBUG, ERROR

from pygan.discriminative_model import DiscriminativeModel

from pydbm.nn.neural_network import NeuralNetwork
from pydbm.nn.nn_layer import NNLayer
from pydbm.optimization.opt_params import OptParams
from pydbm.verification.interface.verificatable_result import VerificatableResult
from pydbm.loss.interface.computable_loss import ComputableLoss

from pydbm.cnn.layerablecnn.convolution_layer import ConvolutionLayer
from pydbm.synapse.nn_graph import NNGraph

# Loss function.
from pydbm.loss.mean_squared_error import MeanSquaredError
# Adam as a optimizer.
from pydbm.optimization.optparams.adam import Adam
# Verification.
from pydbm.verification.verificate_function_approximation import VerificateFunctionApproximation


class NNModel(DiscriminativeModel):
    '''
    Neural Network as a Discriminator.
    '''

    def __init__(
        self,
        batch_size,
        nn_layer_list,
        learning_rate=1e-05,
        computable_loss=None,
        opt_params=None,
        verificatable_result=None,
        nn=None,
        verbose_mode=False
    ):
        '''
        Init.

        Args:
            batch_size:                     Batch size in mini-batch.
            nn_layer_list:                  `list` of `NNLayer`.
            learning_rate:                  Learning rate.
            computable_loss:                is-a `ComputableLoss`.
                                            This parameters will be refered only when `nn` is `None`.

            opt_params:                     is-a `OptParams`.
                                            This parameters will be refered only when `nn` is `None`.

            verificatable_result:           is-a `VerificateFunctionApproximation`.
                                            This parameters will be refered only when `nn` is `None`.

            nn:                             is-a `NeuralNetwork` as a model in this class.
                                            If not `None`, `self.__nn` will be overrided by this `nn`.
                                            If `None`, this class initialize `NeuralNetwork`
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

        if nn is None:
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

            nn = NeuralNetwork(
                # The `list` of `ConvolutionLayer`.
                nn_layer_list=nn_layer_list,
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
                pre_learned_path_list=None,
                # Others.
                learning_attenuate_rate=0.1,
                attenuate_epoch=50
            )

        self.__nn = nn
        self.__batch_size = batch_size
        self.__learning_rate = learning_rate
        self.__verbose_mode = verbose_mode
        self.__q_shape = None
        self.__loss_list = []

    def inference(self, observed_arr):
        '''
        Draws samples from the `true` distribution.

        Args:
            observed_arr:     `np.ndarray` of observed data points.
        
        Returns:
            `np.ndarray` of inferenced.
        '''
        if observed_arr.ndim != 2:
            observed_arr = observed_arr.reshape((observed_arr.shape[0], -1))
        return self.__nn.inference(observed_arr)

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
        delta_arr = self.__nn.back_propagation(grad_arr)
        if fix_opt_flag is False:
            self.__nn.optimize(self.__learning_rate, 1)
        return delta_arr

    def get_nn(self):
        ''' getter '''
        return self.__nn
    
    def set_nn(self, value):
        ''' setter '''
        raise TypeError("This property must be read-only.")
    
    nn = property(get_nn, set_nn)
