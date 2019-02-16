# -*- coding: utf-8 -*-
import numpy as np
from logging import getLogger, StreamHandler, NullHandler, DEBUG, ERROR

from pyqlearning.function_approximator import FunctionApproximator

from pydbm.cnn.convolutional_neural_network import ConvolutionalNeuralNetwork
from pydbm.cnn.layerablecnn.convolution_layer import ConvolutionLayer
from pydbm.cnn.layerable_cnn import LayerableCNN
from pydbm.cnn.feature_generator import FeatureGenerator
from pydbm.optimization.opt_params import OptParams
from pydbm.verification.interface.verificatable_result import VerificatableResult
from pydbm.loss.interface.computable_loss import ComputableLoss
from pydbm.synapse.cnn_output_graph import CNNOutputGraph

# Loss function.
from pydbm.loss.mean_squared_error import MeanSquaredError
# Adam as a optimizer.
from pydbm.optimization.optparams.adam import Adam
# Verification.
from pydbm.verification.verificate_function_approximation import VerificateFunctionApproximation


class CNNFA(FunctionApproximator):
    '''
    Convolutional Neural Networks(CNNs) as a Function Approximator.

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
        cnn_output_graph,
        learning_rate=1e-05,
        computable_loss=None,
        opt_params=None,
        verificatable_result=None,
        pre_learned_path_list=None,
        pre_learned_output_path=None,
        cnn=None,
        verbose_mode=False
    ):
        '''
        Init.

        Args:
            batch_size:                     Batch size in mini-batch.
            layerable_cnn_list:             `list` of `LayerableCNN`.
            cnn_output_graph:               Computation graph which is-a `CNNOutputGraph` to compute parameters in output layer.
            learning_rate:                  Learning rate.
            computable_loss:                is-a `ComputableLoss`.
            opt_params:                     is-a `OptParams`.
            verificatable_result:           is-a `VerificateFunctionApproximation`.
            pre_learned_path_list:          `list` of file path that stored pre-learned parameters.
                                            This parameters will be refered only when `cnn` is `None`.

            pre_learned_output_path:        File path that stores pre-learned parameters.

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

        self.__logger = getLogger("pyqlearning")
        handler = StreamHandler()
        if verbose_mode is True:
            self.__logger.setLevel(DEBUG)
        else:
            self.__logger.setLevel(ERROR)

        self.__logger.addHandler(handler)

        if computable_loss is None:
            computable_loss = MeanSquaredError()
        if verificatable_result is None:
            verificatable_result = VerificateFunctionApproximation()
        if opt_params is None:
            opt_params = Adam()
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
            cnn.setup_output_layer(cnn_output_graph, pre_learned_output_path)

        self.__cnn = cnn
        self.__batch_size = batch_size
        self.__computable_loss = computable_loss
        self.__learning_rate = learning_rate
        self.__verbose_mode = verbose_mode
        self.__loss_list = []

    def learn_q(self, predicted_q_arr, real_q_arr):
        '''
        Infernce Q-Value.
        
        Args:
            predicted_q_arr:    `np.ndarray` of predicted Q-Values.
            real_q_arr:         `np.ndarray` of real Q-Values.
        '''
        """
        if self.__q_shape is None:
            raise ValueError("Before learning, You should execute `__inference_q`.")
        """

        loss = self.__computable_loss.compute_loss(predicted_q_arr, real_q_arr)
        delta_arr = self.__computable_loss.compute_delta(predicted_q_arr, real_q_arr)
        delta_arr = self.__cnn.back_propagation(delta_arr)
        self.__cnn.optimize(self.__learning_rate, 1)
        self.__loss_list.append(loss)

    def inference_q(self, next_action_arr):
        '''
        Infernce Q-Value.
        
        Args:
            next_action_arr:     `np.ndarray` of action.
        
        Returns:
            `np.ndarray` of Q-Values.
        '''
        q_arr = self.__cnn.inference(next_action_arr)
        return q_arr

    def get_model(self):
        '''
        `object` of model as a function approximator,
        which has `cnn` whose type is 
        `pydbm.cnn.pydbm.cnn.convolutional_neural_network.ConvolutionalNeuralNetwork`.
        '''
        class Model(object):
            def __init__(self, cnn):
                self.cnn = cnn

        return Model(self.__cnn)

    def set_model(self, value):
        '''
        `object` of model as a function approximator.
        '''
        raise TypeError("This property must be read-only.")

    model = property(get_model, set_model)

    def get_loss_list(self):
        ''' getter '''
        return self.__loss_list

    def set_loss_list(self, value):
        ''' setter '''
        self.__loss_list = value
    
    loss_list = property(get_loss_list, set_loss_list)
