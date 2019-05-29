# -*- coding: utf-8 -*-
import numpy as np
from logging import getLogger, StreamHandler, NullHandler, DEBUG, ERROR

from pygan.generativemodel.conditional_generative_model import ConditionalGenerativeModel
from pygan.generativemodel.deconvolution_model import DeconvolutionModel
from pygan.true_sampler import TrueSampler

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


class ConditionalConvolutionalModel(ConditionalGenerativeModel):
    '''
    Convolutional Neural Network as a `GenerativeModel`.
    
    This model has a so-called Deconvolutional Neural Network as a `Conditioner`,
    where the function of `Conditioner` is a conditional mechanism 
    to use previous knowledge to condition the generations, 
    incorporating information from previous observed data points to 
    itermediate layers of the `Generator`. In this method, this model 
    can "look back" without a recurrent unit as used in RNN or LSTM. 

    This model observes not only random noises but also any other prior
    information as a previous knowledge and outputs feature points.
    Due to the `Conditioner`, this model has the capacity to exploit
    whatever prior knowledge that is available and can be represented
    as a matrix or tensor.

    Deconvolution in this class is a transposed convolutions which
    "work by swapping the forward and backward passes of a convolution." (Dumoulin, V., & Visin, F. 2016, p20.)
    
    References:
        - Dumoulin, V., & V,kisin, F. (2016). A guide to convolution arithmetic for deep learning. arXiv preprint arXiv:1603.07285.
        - Mirza, M., & Osindero, S. (2014). Conditional generative adversarial nets. arXiv preprint arXiv:1411.1784.
        - Yang, L. C., Chou, S. Y., & Yang, Y. H. (2017). MidiNet: A convolutional generative adversarial network for symbolic-domain music generation. arXiv preprint arXiv:1703.10847.

    '''

    def __init__(
        self,
        deconvolution_model,
        batch_size,
        layerable_cnn_list,
        learning_rate=1e-05,
        computable_loss=None,
        opt_params=None,
        verificatable_result=None,
        cnn=None,
        condition_noise_sampler=None
    ):
        '''
        Init.

        Args:
            deconvolution_model:            is-a `DeconvolutionModel`.
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

            condition_noise_sampler:         is-a `NoiseSampler` to add noise to outputs from `Conditioner`.

        '''
        if isinstance(deconvolution_model, DeconvolutionModel) is False:
            raise TypeError()
        self.__deconvolution_model = deconvolution_model

        if cnn is None:
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

        self.__cnn = cnn
        self.__condition_noise_sampler = condition_noise_sampler
        self.__batch_size = batch_size
        self.__computable_loss = computable_loss
        self.__learning_rate = learning_rate
        self.__q_shape = None
        self.__loss_list = []
        self.__epoch_counter = 0
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
        
        pre_loss_list = []
        for epoch in range(epochs):
            try:
                observed_arr = true_sampler.draw()
                channel = observed_arr.shape[1]
                inferenced_arr = self.inference(observed_arr[:, :channel//2])
                if observed_arr.size != inferenced_arr.size:
                    raise ValueError("In pre-learning, the rank or shape of observed data points and feature points in last layer must be equivalent.")
                grad_arr = self.__computable_loss.compute_delta(observed_arr[:, :channel//2], inferenced_arr)
                loss = self.__computable_loss.compute_loss(observed_arr[:, :channel//2], inferenced_arr)
                pre_loss_list.append(loss)
                self.__logger.debug("Epoch: " + str(epoch) + " loss: " + str(loss))
                self.learn(grad_arr)
            except KeyboardInterrupt:
                self.__logger.debug("Interrupt.")
                break

        self.__pre_loss_arr = np.array(pre_loss_list)

    def draw(self):
        '''
        Draws samples from the `fake` distribution.

        Returns:
            `np.ndarray` of samples.
        '''
        observed_arr = self.extract_conditions()
        conv_arr = self.inference(observed_arr)

        if self.__condition_noise_sampler is not None:
            self.__condition_noise_sampler.output_shape = conv_arr.shape
            noise_arr = self.__condition_noise_sampler.generate()
            conv_arr += noise_arr

        deconv_arr = self.__deconvolution_model.inference(conv_arr)
        return np.concatenate((deconv_arr, observed_arr), axis=1)

    def inference(self, observed_arr):
        '''
        Draws samples from the `true` distribution.

        Args:
            observed_arr:     `np.ndarray` of observed data points.
        
        Returns:
            `np.ndarray` of inferenced.
        '''
        return self.__cnn.inference(observed_arr)

    def learn(self, grad_arr):
        '''
        Update this Generator by ascending its stochastic gradient.

        Args:
            grad_arr:       `np.ndarray` of gradients.
        
        Returns:
            `np.ndarray` of delta or gradients.
        '''
        channel = grad_arr.shape[1] // 2
        grad_arr = self.__deconvolution_model.learn(grad_arr[:, :channel])
        delta_arr = self.__cnn.back_propagation(grad_arr)
        self.__cnn.optimize(self.__learning_rate, self.__epoch_counter)
        self.__epoch_counter += 1

        return delta_arr

    def get_cnn(self):
        ''' getter '''
        return self.__cnn
    
    def set_cnn(self, value):
        ''' setter '''
        raise TypeError("This property must be read-only.")
    
    cnn = property(get_cnn, set_cnn)

    def extract_conditions(self):
        '''
        Extract samples of conditions.
        
        Returns:
            `np.ndarray` of samples.
        '''
        observed_arr = self.noise_sampler.generate()
        return observed_arr

    def get_deconvolution_model(self):
        ''' getter '''
        return self.__deconvolution_model
    
    def set_readonly(self, value):
        ''' setter '''
        raise TypeError()
    
    deconvolution_model = property(get_deconvolution_model, set_readonly)

    def get_epoch_counter(self):
        ''' getter '''
        return self.__epoch_counter
    
    def set_epoch_counter(self, value):
        ''' setter '''
        self.__epoch_counter = value
    
    epoch_counter = property(get_epoch_counter, set_epoch_counter)
