# -*- coding: utf-8 -*-
import numpy as np
from logging import getLogger

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
        learning_attenuate_rate=0.1,
        attenuate_epoch=50,
        computable_loss=None,
        opt_params=None,
        verificatable_result=None,
        nn=None,
        feature_matching_layer=0
    ):
        '''
        Init.

        Args:
            batch_size:                     Batch size in mini-batch.
            nn_layer_list:                  `list` of `NNLayer`.
            learning_rate:                  Learning rate.
            learning_attenuate_rate:        Attenuate the `learning_rate` by a factor of this value every `attenuate_epoch`.
            attenuate_epoch:                Attenuate the `learning_rate` by a factor of `learning_attenuate_rate` every `attenuate_epoch`.
                                            Additionally, in relation to regularization,
                                            this class constrains weight matrixes every `attenuate_epoch`.

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

            feature_matching_layer:         Key of layer number for feature matching forward/backward.

        '''
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
                opt_params.weight_limit = 1e+10
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
                learning_attenuate_rate=learning_attenuate_rate,
                attenuate_epoch=attenuate_epoch
            )

        self.__nn = nn
        self.__batch_size = batch_size
        self.__learning_rate = learning_rate
        self.__attenuate_epoch = attenuate_epoch
        self.__learning_attenuate_rate = learning_attenuate_rate

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
            if ((self.__epoch_counter + 1) % self.__attenuate_epoch == 0):
                self.__learning_rate = self.__learning_rate * self.__learning_attenuate_rate

            self.__nn.optimize(self.__learning_rate, self.__epoch_counter)
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
        if observed_arr.ndim != 2:
            observed_arr = observed_arr.reshape((observed_arr.shape[0], -1))

        if self.__feature_matching_layer == 0:
            return self.__nn.nn_layer_list[0].forward_propagate(observed_arr)
        else:
            for i in range(self.__feature_matching_layer):
                observed_arr = self.__nn.nn_layer_list[i].forward_propagate(observed_arr)

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
        if grad_arr.ndim != 2:
            grad_arr = grad_arr.reshape((grad_arr.shape[0], -1))

        if self.__feature_matching_layer == 0:
            return np.dot(grad_arr, self.__nn.nn_layer_list[0].graph.weight_arr.T)
        else:
            nn_layer_list = self.__nn.nn_layer_list[:self.__feature_matching_layer][::-1]
            for i in range(len(nn_layer_list)):
                grad_arr = np.dot(grad_arr, nn_layer_list[i].graph.weight_arr.T)
            return grad_arr

    def get_nn(self):
        ''' getter '''
        return self.__nn
    
    def set_nn(self, value):
        ''' setter '''
        raise TypeError("This property must be read-only.")
    
    nn = property(get_nn, set_nn)
