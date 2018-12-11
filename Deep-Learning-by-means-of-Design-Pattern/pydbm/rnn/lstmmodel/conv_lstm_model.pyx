# -*- coding: utf-8 -*-
from logging import getLogger
import numpy as np
cimport numpy as np

from pydbm.synapse_list import Synapse
from pydbm.cnn.layerablecnn.convolution_layer import ConvolutionLayer

from pydbm.synapse.cnn_graph import CNNGraph as GivenGraph
from pydbm.synapse.cnn_graph import CNNGraph as InputGraph
from pydbm.synapse.cnn_graph import CNNGraph as ForgotGraph
from pydbm.synapse.cnn_graph import CNNGraph as OutputGraph


from pydbm.activation.logistic_function import LogisticFunction
from pydbm.activation.tanh_function import TanhFunction

from pydbm.cnn.feature_generator import FeatureGenerator
from pydbm.verification.interface.verificatable_result import VerificatableResult
from pydbm.loss.interface.computable_loss import ComputableLoss
from pydbm.optimization.opt_params import OptParams
from pydbm.rnn.interface.reconstructable_model import ReconstructableModel

ctypedef np.float64_t DOUBLE_t


class ConvLSTMModel(ReconstructableModel):
    '''
    Conv LSTM.
    '''
    # is-a `Synapse`.
    __graph = None
    
    def get_graph(self):
        ''' getter '''
        if isinstance(self.__graph, Synapse) is False:
            raise TypeError()

        return self.__graph

    def set_graph(self, value):
        ''' setter '''
        if isinstance(value, Synapse) is False:
            raise TypeError()
        self.__graph = value
    
    graph = property(get_graph, set_graph)

    # Verification function.
    __verificatable_result = None

    # The list of paramters to be differentiated.
    __learned_params_list = []
    
    # Latest loss
    __latest_loss = None
    
    # Latest length of sequneces.
    __cycle_len = 1
    
    # Width and height of input images in the first layer.
    __width = None
    __height = None

    def __init__(
        self,
        graph,
        int epochs,
        int batch_size,
        double learning_rate,
        double learning_attenuate_rate,
        int attenuate_epoch,
        given_conv=None,
        input_conv=None,
        forgot_conv=None,
        output_conv=None,
        int filter_num=20,
        int channel=3,
        width=None,
        height=None,
        int kernel_size=3,
        double scale=0.1,
        int stride=1,
        int pad=1,
        int bptt_tau=16,
        double test_size_rate=0.3,
        computable_loss=None,
        opt_params=None,
        verificatable_result=None,
        tol=1e-04,
        tld=100.0,
        pre_learned_dir=None
    ):
        '''
        Init for building LSTM networks.

        Args:
            graph:                          is-a `Synapse`.
            epochs:                         Epochs of Mini-batch.
            bath_size:                      Batch size of Mini-batch.
            learning_rate:                  Learning rate.
            learning_attenuate_rate:        Attenuate the `learning_rate` by a factor of this value every `attenuate_epoch`.
            attenuate_epoch:                Attenuate the `learning_rate` by a factor of `learning_attenuate_rate` every `attenuate_epoch`.
                                            Additionally, in relation to regularization,
                                            this class constrains weight matrixes every `attenuate_epoch`.

            given_conv:                     is-a `ConvolutionLayer` for LSTM hidden layer.
                                            If `None`, this class instantiates `ConvolutionLayer` based on default settings.

            input_conv:                     is-a `ConvolutionLayer` for input gate in LSTM hidden layer.
                                            If `None`, this class instantiates `ConvolutionLayer` based on default settings.

            forgot_conv:                    is-a `ConvolutionLayer` for forgot gate in LSTM hidden layer.
                                            If `None`, this class instantiates `ConvolutionLayer` based on default settings.

            output_conv:                    is-a `ConvolutionLayer` for output gate in LSTM hidden layer.
                                            If `None`, this class instantiates `ConvolutionLayer` based on default settings.
            
            filter_num:                     The number of filter in default settings.
            channel:                        The channel in default settings.
            width:                          The width of input images in the first layer.
            height:                         The height of input images in the first layer.
            kernel_size:                    The kernel size in default settings.
            scale:                          Scale of weights and bias vector in convolution layer in default settings.
            stride:                         The stride in default settings.
            pad:                            The pad in default settings.

            bptt_tau:                       Refereed maxinum step `t` in Backpropagation Through Time(BPTT).
                                            If `0`, this class referes all past data in BPTT.

            test_size_rate:                 Size of Test data set. If this value is `0`, the validation will not be executed.
            computable_loss:                Loss function.
            opt_params:                     Optimization function.
            verificatable_result:           Verification function.
            tol:                            Tolerance for the optimization.
                                            When the loss or score is not improving by at least tol 
                                            for two consecutive iterations, convergence is considered 
                                            to be reached and training stops.

            tld:                            Tolerance for deviation of loss.

            pre_learned_dir:                Path to directory that stores pre-learned parameters.

        '''
        self.graph = graph
        if pre_learned_dir is not None:
            self.graph.load_pre_learned_params(pre_learned_dir + "conv_lstm_graph.npz")

        self.__height = height
        self.__width = width
        self.__channel = channel

        if given_conv is None:
            self.__given_conv = ConvolutionLayer(
                # Computation graph for first convolution layer.
                GivenGraph(
                    # Logistic function as activation function.
                    activation_function=TanhFunction(),
                    # The number of `filter`.
                    filter_num=filter_num,
                    # The number of channel.
                    channel=channel,
                    # The size of kernel.
                    kernel_size=kernel_size,
                    # The filter scale.
                    scale=scale,
                    # The nubmer of stride.
                    stride=stride,
                    # The number of zero-padding.
                    pad=pad
                )
            )
            if pre_learned_dir is not None:
                self.__given_conv.graph.load_pre_learned_params(
                    pre_learned_dir + "conv_lstm_given_graph.npz"
                )
        else:
            self.__given_conv = given_conv

        if input_conv is None:
            self.__input_conv = ConvolutionLayer(
                # Computation graph for first convolution layer.
                InputGraph(
                    # Logistic function as activation function.
                    activation_function=LogisticFunction(),
                    # The number of `filter`.
                    filter_num=filter_num,
                    # The number of channel.
                    channel=channel,
                    # The size of kernel.
                    kernel_size=kernel_size,
                    # The filter scale.
                    scale=scale,
                    # The nubmer of stride.
                    stride=stride,
                    # The number of zero-padding.
                    pad=pad
                )
            )
            if pre_learned_dir is not None:
                self.__input_conv.graph.load_pre_learned_params(
                    pre_learned_dir + "conv_lstm_input_graph.npz"
                )
        else:
            self.__input_conv = input_conv

        if forgot_conv is None:
            self.__forgot_conv = ConvolutionLayer(
                # Computation graph for first convolution layer.
                ForgotGraph(
                    # Logistic function as activation function.
                    activation_function=LogisticFunction(),
                    # The number of `filter`.
                    filter_num=filter_num,
                    # The number of channel.
                    channel=channel,
                    # The size of kernel.
                    kernel_size=kernel_size,
                    # The filter scale.
                    scale=scale,
                    # The nubmer of stride.
                    stride=stride,
                    # The number of zero-padding.
                    pad=pad
                )
            )
            if pre_learned_dir is not None:
                self.__forgot_conv.graph.load_pre_learned_params(
                    pre_learned_dir + "conv_lstm_forgot_graph.npz"
                )
        else:
            self.__forgot_conv = forgot_conv

        if output_conv is None:
            self.__output_conv = ConvolutionLayer(
                # Computation graph for first convolution layer.
                OutputGraph(
                    # Logistic function as activation function.
                    activation_function=TanhFunction(),
                    # The number of `filter`.
                    filter_num=filter_num,
                    # The number of channel.
                    channel=channel,
                    # The size of kernel.
                    kernel_size=kernel_size,
                    # The filter scale.
                    scale=scale,
                    # The nubmer of stride.
                    stride=stride,
                    # The number of zero-padding.
                    pad=pad
                )
            )
            if pre_learned_dir is not None:
                self.__output_conv.graph.load_pre_learned_params(
                    pre_learned_dir + "conv_lstm_output_graph.npz"
                )
        else:
            self.__output_conv = output_conv

        if isinstance(computable_loss, ComputableLoss):
            self.__computable_loss = computable_loss
        else:
            raise TypeError()

        if isinstance(opt_params, OptParams):
            self.__opt_params = opt_params
            self.__dropout_rate = self.__opt_params.dropout_rate
        else:
            raise TypeError()

        if isinstance(verificatable_result, VerificatableResult):
            self.__verificatable_result = verificatable_result
        else:
            raise TypeError()

        self.__epochs = epochs
        self.__batch_size = batch_size

        self.__learning_rate = learning_rate
        self.__learning_attenuate_rate = learning_attenuate_rate
        self.__attenuate_epoch = attenuate_epoch

        self.__bptt_tau = bptt_tau

        self.__test_size_rate = test_size_rate
        self.__tol = tol
        self.__tld = tld

        self.__memory_tuple_list = []
        self.__cycle_len = 1

        logger = getLogger("pydbm")
        self.__logger = logger

    def learn(self, np.ndarray[DOUBLE_t, ndim=5] observed_arr, np.ndarray target_arr=np.array([])):
        '''
        Learn the observed data points
        for vector representation of the input time-series.

        Override.

        Args:
            observed_arr:    Array like or sparse matrix as the observed data points.
            target_arr:      Array like or sparse matrix as the target data points.
                             To learn as Auto-encoder, this value must be `None` or equivalent to `observed_arr`.
        '''
        self.__logger.debug("pydbm.rnn.lstm_model.learn is started. ")

        cdef double learning_rate = self.__learning_rate
        cdef int epoch
        cdef int batch_index

        cdef int row_o = observed_arr.shape[0]
        cdef int row_t = target_arr.shape[0]

        cdef np.ndarray train_index
        cdef np.ndarray test_index
        cdef np.ndarray[DOUBLE_t, ndim=5] train_observed_arr
        cdef np.ndarray train_target_arr
        cdef np.ndarray[DOUBLE_t, ndim=5] test_observed_arr
        cdef np.ndarray test_target_arr

        cdef np.ndarray rand_index
        cdef np.ndarray[DOUBLE_t, ndim=5] batch_observed_arr
        cdef np.ndarray batch_target_arr

        if row_t != 0 and row_t != row_o:
            raise ValueError("The row of `target_arr` must be equivalent to the row of `observed_arr`.")

        if row_t == 0:
            target_arr = observed_arr.copy()
        else:
            if target_arr.ndim == 2:
                target_arr = target_arr.reshape((target_arr.shape[0], 1, target_arr.shape[1]))

        if self.__test_size_rate > 0:
            train_index = np.random.choice(observed_arr.shape[0], round(self.__test_size_rate * observed_arr.shape[0]), replace=False)
            test_index = np.array(list(set(range(observed_arr.shape[0])) - set(train_index)))
            train_observed_arr = observed_arr[train_index]
            test_observed_arr = observed_arr[test_index]
            train_target_arr = target_arr[train_index]
            test_target_arr = target_arr[test_index]
        else:
            train_observed_arr = observed_arr
            train_target_arr = observed_arr

        cdef double loss
        cdef double test_loss
        cdef np.ndarray[DOUBLE_t, ndim=2] pred_arr
        cdef np.ndarray[DOUBLE_t, ndim=2] test_pred_arr
        cdef np.ndarray[DOUBLE_t, ndim=2] delta_arr

        best_params_list = []
        try:
            self.__memory_tuple_list = []
            loss_list = []
            min_loss = None
            eary_stop_flag = False
            for epoch in range(self.__epochs):
                self.__opt_params.dropout_rate = self.__dropout_rate

                if ((epoch + 1) % self.__attenuate_epoch == 0):
                    learning_rate = learning_rate / self.__learning_attenuate_rate

                rand_index = np.random.choice(train_observed_arr.shape[0], size=self.__batch_size)
                batch_observed_arr = train_observed_arr[rand_index]
                batch_target_arr = train_target_arr[rand_index]

                try:
                    pred_arr = self.forward_propagation(batch_observed_arr)
                    ver_pred_arr = pred_arr.copy()
                    if pred_arr.ndim == batch_target_arr[:, -1].ndim:
                        loss = self.__computable_loss.compute_loss(
                            pred_arr,
                            batch_target_arr[:, -1]
                        )
                    else:
                        loss = self.__computable_loss.compute_loss(
                            pred_arr,
                            batch_target_arr[:, -1].reshape((
                                batch_target_arr[:, -1].shape[0],
                                -1
                            ))
                        )
                        
                    remember_flag = False
                    if len(loss_list) > 0:
                        if abs(loss - (sum(loss_list)/len(loss_list))) > self.__tld:
                            remember_flag = True

                    if remember_flag is True:
                        self.__remember_best_params(best_params_list)
                        # Re-try.
                        pred_arr = self.forward_propagation(batch_observed_arr)
                        ver_pred_arr = pred_arr.copy()
                        if pred_arr.ndim == batch_target_arr[:, -1].ndim:
                            loss = self.__computable_loss.compute_loss(
                                pred_arr,
                                batch_target_arr[:, -1]
                            )
                        else:
                            loss = self.__computable_loss.compute_loss(
                                pred_arr,
                                batch_target_arr[:, -1].reshape((
                                    batch_target_arr[:, -1].shape[0],
                                    -1
                                ))
                            )

                    if pred_arr.ndim == batch_target_arr[:, -1].ndim:
                        delta_arr = self.__computable_loss.compute_delta(
                            pred_arr,
                            batch_target_arr[:, -1]
                        )
                    else:
                        delta_arr = self.__computable_loss.compute_delta(
                            pred_arr,
                            batch_target_arr[:, -1].reshape((
                                test_batch_target_arr[:, -1].shape[0],
                                -1
                            ))
                        )

                    _delta_arr, grads_list = self.back_propagation(pred_arr, delta_arr)
                    self.optimize(grads_list, learning_rate, epoch)
                    self.graph.hidden_activity_arr = np.array([])
                    self.graph.rnn_activity_arr = np.array([])

                    if min_loss is None or min_loss > loss:
                        min_loss = loss
                        best_params_list = [
                            self.graph.weights_output_arr,
                            self.graph.output_bias_arr,
                            self.__given_conv.graph.weight_arr,
                            self.__input_conv.graph.weight_arr,
                            self.__forgot_conv.graph.weight_arr,
                            self.__output_conv.graph.weight_arr,
                            self.__given_conv.graph.bias_arr,
                            self.__input_conv.graph.bias_arr,
                            self.__forgot_conv.graph.bias_arr,
                            self.__output_conv.graph.bias_arr
                        ]
                        self.__logger.debug("Best params are updated.")

                except FloatingPointError:
                    if epoch > int(self.__epochs * 0.7):
                        self.__logger.debug(
                            "Underflow occurred when the parameters are being updated. Because of early stopping, this error is catched and the parameter is not updated."
                        )
                        eary_stop_flag = True
                        break
                    else:
                        raise

                if self.__test_size_rate > 0:
                    self.__opt_params.dropout_rate = 0.0
                    rand_index = np.random.choice(test_observed_arr.shape[0], size=self.__batch_size)
                    test_batch_observed_arr = test_observed_arr[rand_index]
                    test_batch_target_arr = test_target_arr[rand_index]

                    test_pred_arr = self.forward_propagation(test_batch_observed_arr)

                    if test_pred_arr.ndim == test_batch_target_arr[:, -1].ndim:
                        test_loss = self.__computable_loss.compute_loss(
                            test_pred_arr,
                            test_batch_target_arr[:, -1]
                        )
                    else:
                        test_loss = self.__computable_loss.compute_loss(
                            test_pred_arr,
                            test_batch_target_arr[:, -1].reshape((
                                test_batch_target_arr[:, -1].shape[0],
                                -1
                            ))
                        )

                    remember_flag = False
                    if len(loss_list) > 0:
                        if abs(test_loss - (sum(loss_list)/len(loss_list))) > self.__tld:
                            remember_flag = True

                    if remember_flag is True:
                        self.__remember_best_params(best_params_list)
                        # Re-try.
                        test_pred_arr = self.forward_propagation(test_batch_observed_arr)

                    if self.__verificatable_result is not None:
                        if self.__test_size_rate > 0:
                            if ver_pred_arr.ndim == batch_target_arr[:, -1]:
                                train_label_arr = batch_target_arr[:, -1].reshape((
                                    batch_target_arr[:, -1].shape[0],
                                    -1
                                ))
                            else:
                                train_label_arr = batch_target_arr[:, -1]

                            if test_pred_arr.ndim == test_batch_target_arr[:, -1].ndim:
                                test_label_arr = test_batch_target_arr[:, -1].reshape((
                                    test_batch_target_arr[:, -1].shape[0],
                                    -1
                                ))
                            else:
                                test_label_arr = test_batch_target_arr[:, -1]

                            self.__verificatable_result.verificate(
                                self.__computable_loss,
                                train_pred_arr=ver_pred_arr, 
                                train_label_arr=train_label_arr,
                                test_pred_arr=test_pred_arr,
                                test_label_arr=test_label_arr
                            )

                if epoch > 1 and abs(loss - loss_list[-1]) < self.__tol:
                    eary_stop_flag = True
                    break
                loss_list.append(loss)

        except KeyboardInterrupt:
            self.__logger.debug("Interrupt.")

        if eary_stop_flag is True:
            self.__logger.debug("Eary stopping.")
            eary_stop_flag = False

        self.__remember_best_params(best_params_list)
        self.__logger.debug("end. ")

    def learn_generated(self, feature_generator):
        '''
        Learn features generated by `FeatureGenerator`.
        
        Args:
            feature_generator:    is-a `FeatureGenerator`.

        '''
        if isinstance(feature_generator, FeatureGenerator) is False:
            raise TypeError("The type of `feature_generator` must be `FeatureGenerator`.")

        cdef double learning_rate = self.__learning_rate
        cdef int epoch
        cdef int batch_index

        cdef np.ndarray train_index
        cdef np.ndarray test_index
        cdef np.ndarray[DOUBLE_t, ndim=5] train_observed_arr
        cdef np.ndarray train_target_arr
        cdef np.ndarray[DOUBLE_t, ndim=5] test_observed_arr
        cdef np.ndarray test_target_arr

        cdef np.ndarray rand_index
        cdef np.ndarray[DOUBLE_t, ndim=5] batch_observed_arr
        cdef np.ndarray batch_target_arr

        cdef double loss
        cdef double test_loss
        cdef np.ndarray[DOUBLE_t, ndim=2] pred_arr
        cdef np.ndarray[DOUBLE_t, ndim=2] test_pred_arr
        cdef np.ndarray[DOUBLE_t, ndim=2] delta_arr

        best_params_list = []
        try:
            self.__memory_tuple_list = []
            loss_list = []
            min_loss = None
            eary_stop_flag = False
            epoch = 0
            for batch_observed_arr, batch_target_arr, test_batch_observed_arr, test_batch_target_arr in feature_generator.generate():
                epoch += 1
                self.__opt_params.dropout_rate = self.__dropout_rate

                if ((epoch + 1) % self.__attenuate_epoch == 0):
                    learning_rate = learning_rate / self.__learning_attenuate_rate

                try:
                    pred_arr = self.forward_propagation(batch_observed_arr)
                    ver_pred_arr = pred_arr.copy()
                    if pred_arr.ndim == batch_target_arr[:, -1].ndim:
                        loss = self.__computable_loss.compute_loss(
                            pred_arr,
                            batch_target_arr[:, -1]
                        )
                    else:
                        loss = self.__computable_loss.compute_loss(
                            pred_arr,
                            batch_target_arr[:, -1].reshape((
                                batch_target_arr[:, -1].shape[0],
                                -1
                            ))
                        )

                    remember_flag = False
                    if len(loss_list) > 0:
                        if abs(loss - (sum(loss_list)/len(loss_list))) > self.__tld:
                            remember_flag = True

                    if remember_flag is True:
                        self.__remember_best_params(best_params_list)
                        # Re-try.
                        pred_arr = self.forward_propagation(batch_observed_arr)
                        ver_pred_arr = pred_arr.copy()

                        if pred_arr.ndim == batch_target_arr[:, -1].ndim:
                            loss = self.__computable_loss.compute_loss(
                                pred_arr,
                                batch_target_arr[:, -1]
                            )
                        else:
                            loss = self.__computable_loss.compute_loss(
                                pred_arr,
                                batch_target_arr[:, -1].reshape((
                                    batch_target_arr[:, -1].shape[0],
                                    -1
                                ))
                            )

                    if pred_arr.ndim == batch_target_arr[:, -1].ndim:
                        delta_arr = self.__computable_loss.compute_delta(
                            pred_arr,
                            batch_target_arr[:, -1]
                        )
                    else:
                        delta_arr = self.__computable_loss.compute_delta(
                            pred_arr,
                            batch_target_arr[:, -1].reshape((
                                batch_target_arr[:, -1].shape[0],
                                -1
                            ))
                        )

                    _delta_arr, grads_list = self.back_propagation(pred_arr, delta_arr)
                    self.optimize(grads_list, learning_rate, epoch)
                    self.graph.hidden_activity_arr = np.array([])
                    self.graph.rnn_activity_arr = np.array([])

                    if min_loss is None or min_loss > loss:
                        min_loss = loss
                        best_params_list = [
                            self.graph.weights_output_arr,
                            self.graph.output_bias_arr,
                            self.__given_conv.graph.weight_arr,
                            self.__input_conv.graph.weight_arr,
                            self.__forgot_conv.graph.weight_arr,
                            self.__output_conv.graph.weight_arr,
                            self.__given_conv.graph.bias_arr,
                            self.__input_conv.graph.bias_arr,
                            self.__forgot_conv.graph.bias_arr,
                            self.__output_conv.graph.bias_arr
                        ]
                        self.__logger.debug("Best params are updated.")

                except FloatingPointError:
                    if epoch > int(self.__epochs * 0.7):
                        self.__logger.debug(
                            "Underflow occurred when the parameters are being updated. Because of early stopping, this error is catched and the parameter is not updated."
                        )
                        eary_stop_flag = True
                        break
                    else:
                        raise

                if self.__test_size_rate > 0:
                    self.__opt_params.dropout_rate = 0.0
                    test_pred_arr = self.forward_propagation(test_batch_observed_arr)

                    if test_pred_arr.ndim == test_batch_target_arr[:, -1].ndim:
                        test_loss = self.__computable_loss.compute_loss(
                            test_pred_arr,
                            test_batch_target_arr[:, -1]
                        )
                    else:
                        test_loss = self.__computable_loss.compute_loss(
                            test_pred_arr,
                            test_batch_target_arr[:, -1].reshape((
                                test_batch_target_arr[:, -1].shape[0],
                                -1
                            ))
                        )

                    remember_flag = False
                    if len(loss_list) > 0:
                        if abs(test_loss - (sum(loss_list)/len(loss_list))) > self.__tld:
                            remember_flag = True

                    if remember_flag is True:
                        self.__remember_best_params(best_params_list)
                        # Re-try.
                        test_pred_arr = self.forward_propagation(test_batch_observed_arr)

                    if self.__verificatable_result is not None:
                        if self.__test_size_rate > 0:
                            if ver_pred_arr.ndim == batch_target_arr[:, -1].ndim:
                                train_label_arr = batch_target_arr[:, -1]
                            else:
                                train_label_arr = batch_target_arr[:, -1].reshape((
                                    batch_target_arr[:, -1].shape[0],
                                    -1
                                ))

                            if test_pred_arr.ndim == test_batch_target_arr[:, -1].ndim:
                                test_label_arr = test_batch_target_arr[:, -1]
                            else:
                                test_label_arr = test_batch_target_arr[:, -1].reshape((
                                    test_batch_target_arr[:, -1].shape[0],
                                    -1
                                ))

                            self.__verificatable_result.verificate(
                                self.__computable_loss,
                                train_pred_arr=ver_pred_arr, 
                                train_label_arr=train_label_arr,
                                test_pred_arr=test_pred_arr,
                                test_label_arr=test_label_arr
                            )

                if epoch > 1 and abs(loss - loss_list[-1]) < self.__tol:
                    eary_stop_flag = True
                    break
                loss_list.append(loss)

        except KeyboardInterrupt:
            self.__logger.debug("Interrupt.")

        if eary_stop_flag is True:
            self.__logger.debug("Eary stopping.")
            eary_stop_flag = False

        self.__remember_best_params(best_params_list)
        self.__logger.debug("end. ")

    def __remember_best_params(self, best_params_list):
        '''
        Remember best parameters.
        
        Args:
            best_params_list:    `list` of parameters.

        '''
        if len(best_params_list) > 0:
            self.graph.weights_output_arr = best_params_list[0]
            self.graph.output_bias_arr = best_params_list[1]
            self.__given_conv.graph.weight_arr = best_params_list[2]
            self.__input_conv.graph.weight_arr = best_params_list[3]
            self.__forgot_conv.graph.weight_arr = best_params_list[4]
            self.__output_conv.graph.weight_arr = best_params_list[5]
            self.__given_conv.graph.bias_arr = best_params_list[6]
            self.__input_conv.graph.bias_arr = best_params_list[7]
            self.__forgot_conv.graph.bias_arr = best_params_list[8]
            self.__output_conv.graph.bias_arr = best_params_list[9]
            self.__logger.debug("Best params are saved.")

    def forward_propagation(self, np.ndarray batch_observed_arr):
        '''
        Forward propagation.
        
        Args:
            batch_observed_arr:    Array like or sparse matrix as the observed data points.
        
        Returns:
            Array like or sparse matrix as the predicted data points.
        '''
        if batch_observed_arr.ndim == 3 and self.__height is not None and self.__width is not None:
             batch_observed_arr = batch_observed_arr.reshape((
                 batch_observed_arr.shape[0],
                 batch_observed_arr.shape[1],
                 self.__channel,
                 self.__height,
                 self.__width
             ))

        cdef np.ndarray[DOUBLE_t, ndim=5] hidden_activity_arr = self.hidden_forward_propagate(
            batch_observed_arr
        )
        self.graph.hidden_activity_arr = hidden_activity_arr
        """
        cdef np.ndarray[DOUBLE_t, ndim=2] pred_arr = self.output_forward_propagate(
            hidden_activity_arr
        )
        return pred_arr
        """
        return hidden_activity_arr

    def back_propagation(self, np.ndarray[DOUBLE_t, ndim=4] pred_arr, np.ndarray[DOUBLE_t, ndim=4] delta_arr):
        r'''
        Back propagation.

        Args:
            pred_arr:            `np.ndarray` of predicted data points.
            delta_output_arr:    Delta.
        
        Returns:
            Tuple(
                `np.ndarray` of Delta, 
                `list` of gradations
            )
        '''
        """
        delta_arr, grads_list = self.output_back_propagate(pred_arr, delta_arr)
        """
        _delta_arr = self.hidden_back_propagate(delta_arr)
        """
        return (_delta_arr, grads_list)
        """
        return _delta_arr, []

    def optimize(
        self,
        grads_list,
        double learning_rate,
        int epoch
    ):
        '''
        Back propagation.
        
        Args:
            grads_list:     `list` of graduations.
            learning_rate:  Learning rate.
            epoch:          Now epoch.
            
        '''
        params_list = [
            self.graph.weights_output_arr,
            self.graph.output_bias_arr,
            self.__given_conv.graph.weight_arr,
            self.__input_conv.graph.weight_arr,
            self.__forgot_conv.graph.weight_arr,
            self.__output_conv.graph.weight_arr,
            self.__given_conv.graph.bias_arr,
            self.__input_conv.graph.bias_arr,
            self.__forgot_conv.graph.bias_arr,
            self.__output_conv.graph.bias_arr
        ]
        grads_list.extend([
            self.__given_conv.delta_weight_arr,
            self.__input_conv.delta_weight_arr,
            self.__forgot_conv.delta_weight_arr,
            self.__output_conv.delta_weight_arr,
            self.__given_conv.delta_bias_arr,
            self.__input_conv.delta_bias_arr,
            self.__forgot_conv.delta_bias_arr,
            self.__output_conv.delta_bias_arr
        ])

        params_list = self.__opt_params.optimize(
            params_list,
            grads_list,
            learning_rate
        )

        self.graph.weights_output_arr = params_list[0]
        self.graph.output_bias_arr = params_list[1]

        self.__given_conv.graph.weight_arr = params_list[2]
        self.__input_conv.graph.weight_arr = params_list[3]
        self.__forgot_conv.graph.weight_arr = params_list[4]
        self.__output_conv.graph.weight_arr = params_list[5]

        self.__given_conv.graph.bias_arr = params_list[6]
        self.__input_conv.graph.bias_arr = params_list[7]
        self.__forgot_conv.graph.bias_arr = params_list[8]
        self.__output_conv.graph.bias_arr = params_list[9]

        if ((epoch + 1) % self.__attenuate_epoch == 0):
            self.graph.weights_output_arr = self.__opt_params.constrain_weight(
                self.graph.weights_output_arr
            )
            self.__given_conv.graph.weight_arr = self.__opt_params.constrain_weight(
                self.__given_conv.graph.weight_arr
            )
            self.__input_conv.graph.weight_arr = self.__opt_params.constrain_weight(
                self.__input_conv.graph.weight_arr
            )
            self.__forgot_conv.graph.weight_arr = self.__opt_params.constrain_weight(
                self.__forgot_conv.graph.weight_arr
            )
            self.__output_conv.graph.weight_arr = self.__opt_params.constrain_weight(
                self.__output_conv.graph.weight_arr
            )

        self.__given_conv.reset_delta()
        self.__input_conv.reset_delta()
        self.__forgot_conv.reset_delta()
        self.__output_conv.reset_delta()

    def inference(
        self,
        np.ndarray observed_arr,
        np.ndarray hidden_activity_arr=None,
        np.ndarray rnn_activity_arr=None
    ):
        r'''
        Inference the feature points to reconstruct the time-series.

        Override.

        Args:
            observed_arr:           Array like or sparse matrix as the observed data points.
            hidden_activity_arr:    Array like or sparse matrix as the state in hidden layer.
            rnn_activity_arr:       Array like or sparse matrix as the state in RNN.

        Returns:
            Tuple(
                Array like or sparse matrix of reconstructed instances of time-series,
                Array like or sparse matrix of the state in hidden layer,
                Array like or sparse matrix of the state in RNN
            )
        '''
        cdef int sample_n = observed_arr.shape[0]
        cdef int cycle_len = observed_arr.shape[1]

        if hidden_activity_arr is None:
            self.graph.hidden_activity_arr = np.array([])
        else:
            self.graph.hidden_activity_arr = hidden_activity_arr

        if rnn_activity_arr is None:
            self.graph.rnn_activity_arr = np.array([])
        else:
            self.graph.rnn_activity_arr = rnn_activity_arr

        self.__opt_params.dropout_rate = 0.0
        cdef np.ndarray pred_arr = self.forward_propagation(observed_arr)
        self.__opt_params.dropout_rate = self.__dropout_rate

        return pred_arr

    def get_feature_points(self):
        '''
        Extract the activities in hidden layer and reset it, 
        considering this method will be called per one cycle in instances of time-series.

        Returns:
            The `list` of array like or sparse matrix of feature points or virtual visible observed data points.
        '''
        return self.graph.hidden_activity_arr

    def hidden_forward_propagate(self, np.ndarray[DOUBLE_t, ndim=5] observed_arr):
        '''
        Forward propagation in LSTM gate.
        
        Args:
            observed_arr:    `np.ndarray` of observed data points.
        
        Returns:
            Predicted data points.
        '''
        cdef int sample_n = observed_arr.shape[0]
        cdef int cycle_len = observed_arr.shape[1]
        cdef int channel = observed_arr.shape[2]
        cdef int width = observed_arr.shape[3]
        cdef int height = observed_arr.shape[4]
        
        self.__cycle_len = cycle_len

        cdef np.ndarray[DOUBLE_t, ndim=5] pred_arr = None
        cdef int cycle
        for cycle in range(cycle_len):
            self.graph.hidden_activity_arr, self.graph.rnn_activity_arr = self.__lstm_forward(
                observed_arr[:, cycle, :, :, :],
                self.graph.hidden_activity_arr,
                self.graph.rnn_activity_arr
            )
            
            if pred_arr is None:
                pred_arr = np.zeros(
                    (
                        self.graph.hidden_activity_arr.shape[0],
                        cycle_len,
                        self.graph.hidden_activity_arr.shape[1],
                        self.graph.hidden_activity_arr.shape[2],
                        self.graph.hidden_activity_arr.shape[3]
                    ),
                    dtype=np.float64
                )

            pred_arr[:, cycle] = self.graph.hidden_activity_arr

        return pred_arr

    def output_forward_propagate(self, np.ndarray[DOUBLE_t, ndim=5] pred_arr):
        '''
        Forward propagation in output layer.
        
        Args:
            pred_arr:            `np.ndarray` of predicted data points.

        Returns:
            `np.ndarray` of propagated data points.
        '''
        cdef np.ndarray hidden_pred_arr = pred_arr[:, -1].reshape((pred_arr[:, -1].shape[0], -1))
        cdef np.ndarray[DOUBLE_t, ndim=2] _pred_arr = self.graph.output_activating_function.activate(
            np.dot(hidden_pred_arr, self.graph.weights_output_arr) + self.graph.output_bias_arr
        )
        return _pred_arr

    def output_back_propagate(self, np.ndarray[DOUBLE_t, ndim=2] pred_arr, np.ndarray[DOUBLE_t, ndim=2] delta_arr):
        r'''
        Back propagation in output layer.

        Args:
            pred_arr:            `np.ndarray` of predicted data points.
            delta_output_arr:    Delta.
        
        Returns:
            Tuple(
                `np.ndarray` of Delta, 
                `list` of gradations
            )
        '''
        cdef np.ndarray[DOUBLE_t, ndim=2] _delta_arr = np.dot(delta_arr, self.graph.weights_output_arr.T)
        cdef np.ndarray[DOUBLE_t, ndim=2] delta_weights_arr = np.dot(pred_arr.T, _delta_arr).T
        cdef np.ndarray[DOUBLE_t, ndim=1] delta_bias_arr = np.sum(delta_arr, axis=0)

        grads_list = [
            delta_weights_arr,
            delta_bias_arr
        ]
        
        return (_delta_arr, grads_list)

    def hidden_back_propagate(self, np.ndarray[DOUBLE_t, ndim=4] delta_output_arr):
        r'''
        Back propagation in hidden layer.
        
        @TODO(chimera0): BPTT.
        
        Args:
            delta_output_arr:    Delta.
        
        Returns:
            Tuple(
                `np.ndarray` of Delta, 
                `list` of gradations
            )
        '''
        cdef int sample_n = delta_output_arr.shape[0]
        # not BPTT.
        cdef int cycle_len = 1
        cdef int channel = delta_output_arr.shape[1]
        cdef int width = delta_output_arr.shape[2]
        cdef int height = delta_output_arr.shape[3]

        cdef np.ndarray[DOUBLE_t, ndim=5] delta_arr = np.empty(
            (
                sample_n,
                cycle_len,
                channel,
                width,
                height
            ),
            dtype=np.float64
        )

        cdef np.ndarray[DOUBLE_t, ndim=4] _delta_hidden_arr
        cdef np.ndarray[DOUBLE_t, ndim=4] delta_hidden_arr
        cdef np.ndarray delta_rnn_arr = np.array([])

        cdef int bp_count = 0
        cdef int cycle
        for cycle in reversed(range(cycle_len)):
            if bp_count == 0:
                _delta_hidden_arr = delta_output_arr
            else:
                _delta_hidden_arr = delta_hidden_arr

            delta_hidden_arr, delta_rnn_arr = self.lstm_backward(
                _delta_hidden_arr,
                delta_rnn_arr,
                cycle
            )
            delta_arr[:, cycle] = delta_hidden_arr

            if bp_count >= self.__bptt_tau:
                break
            bp_count += 1

        self.__memory_tuple_list = []
        return delta_arr, []

    def __lstm_forward(
        self,
        np.ndarray[DOUBLE_t, ndim=4] observed_arr,
        np.ndarray hidden_activity_arr,
        np.ndarray rnn_activity_arr
    ):
        '''
        Forward propagate in LSTM gate.
        
        Args:
            observed_arr:           `np.ndarray` of observed data points.
            hidden_activity_arr:    `np.ndarray` of activities in hidden layer.
            rnn_activity_arr:       `np.ndarray` of activities in LSTM gate.
        
        Returns:
            Tuple(
                `np.ndarray` of activities in hidden layer,
                `np.ndarray` of activities in LSTM gate.
            )
        '''
        cdef np.ndarray[DOUBLE_t, ndim=4] given_activity_arr = self.__given_conv.forward_propagate(
            observed_arr
        )

        cdef np.ndarray[DOUBLE_t, ndim=4] input_gate_activity_arr = self.__input_conv.forward_propagate(
            observed_arr
        )

        cdef np.ndarray[DOUBLE_t, ndim=4] forget_gate_activity_arr = self.__forgot_conv.forward_propagate(
            observed_arr
        )

        cdef np.ndarray[DOUBLE_t, ndim=4] output_gate_activity_arr = self.__output_conv.forward_propagate(observed_arr)

        if rnn_activity_arr is None or rnn_activity_arr.shape[0] == 0:
            rnn_activity_arr = np.zeros_like(forget_gate_activity_arr)

        cdef np.ndarray[DOUBLE_t, ndim=4] _rnn_activity_arr = np.nansum(
            np.array([
                np.nanprod(
                    np.array([
                        np.expand_dims(given_activity_arr, axis=0),
                        np.expand_dims(input_gate_activity_arr, axis=0)
                    ]),
                    axis=0
                ),
                np.nanprod(
                    np.array([
                        np.expand_dims(forget_gate_activity_arr, axis=0),
                        np.expand_dims(rnn_activity_arr, axis=0)
                    ]),
                    axis=0
                )
            ]),
            axis=0
        )[0]

        cdef np.ndarray[DOUBLE_t, ndim=4] _hidden_activity_arr = np.nanprod(
            np.array([
                np.expand_dims(output_gate_activity_arr, axis=0),
                np.expand_dims(self.graph.hidden_activating_function.activate(_rnn_activity_arr), axis=0)
            ]),
            axis=0
        )[0]

        _hidden_activity_arr = self.__opt_params.dropout(_hidden_activity_arr)

        self.__memory_tuple_list.append((
            observed_arr, 
            hidden_activity_arr, 
            rnn_activity_arr, 
            given_activity_arr, 
            input_gate_activity_arr, 
            forget_gate_activity_arr, 
            output_gate_activity_arr, 
            _rnn_activity_arr,
            _hidden_activity_arr
        ))

        return (_hidden_activity_arr, _rnn_activity_arr)

    def lstm_backward(
        self,
        np.ndarray[DOUBLE_t, ndim=4] delta_hidden_arr,
        np.ndarray delta_rnn_arr,
        int cycle
    ):
        r'''
        Back propagation in LSTM gate.
        
        Args:
            delta_hidden_arr:   Delta from output layer to hidden layer.
            delta_rnn_arr:      Delta in LSTM gate.
            cycle:              Now cycle or time.

        Returns:
            Tuple(
                Delta from hidden layer to input layer,
                Delta in hidden layer at previous time,
                Delta in LSTM gate at previous time,
                `list` of gradations.
            )
        '''
        cdef np.ndarray[DOUBLE_t, ndim=4] observed_arr = self.__memory_tuple_list[cycle][0]
        cdef np.ndarray[DOUBLE_t, ndim=4] pre_rnn_activity_arr = self.__memory_tuple_list[cycle][2]
        cdef np.ndarray[DOUBLE_t, ndim=4] given_activity_arr = self.__memory_tuple_list[cycle][3]
        cdef np.ndarray[DOUBLE_t, ndim=4] input_gate_activity_arr = self.__memory_tuple_list[cycle][4]
        cdef np.ndarray[DOUBLE_t, ndim=4] forget_gate_activity_arr = self.__memory_tuple_list[cycle][5]
        cdef np.ndarray[DOUBLE_t, ndim=4] output_gate_activity_arr = self.__memory_tuple_list[cycle][6]
        cdef np.ndarray[DOUBLE_t, ndim=4] rnn_activity_arr = self.__memory_tuple_list[cycle][7]

        cdef np.ndarray[DOUBLE_t, ndim=4] _rnn_activity_arr = self.graph.hidden_activating_function.activate(
            rnn_activity_arr
        )

        if delta_rnn_arr.shape[0] == 0:
            delta_rnn_arr = np.zeros_like(delta_hidden_arr)

        cdef np.ndarray[DOUBLE_t, ndim=4] delta_top_arr = np.nanprod(
            np.array([
                delta_rnn_arr,
                np.nansum(
                    np.array([
                        delta_hidden_arr,
                        output_gate_activity_arr,
                        self.graph.hidden_activating_function.derivative(rnn_activity_arr)
                    ]),
                    axis=0
                )
            ]),
            axis=0
        )
        cdef np.ndarray[DOUBLE_t, ndim=4] delta_pre_rnn_arr = np.nanprod(
            np.array([delta_top_arr, forget_gate_activity_arr]),
            axis=0
        )
        cdef np.ndarray[DOUBLE_t, ndim=4] delta_output_gate_arr = np.nanprod(
            np.array([
                delta_hidden_arr,
                rnn_activity_arr,
                self.graph.output_gate_activating_function.derivative(output_gate_activity_arr)]
            ),
            axis=0
        )
        cdef np.ndarray[DOUBLE_t, ndim=4] delta_forget_gate_arr = np.nanprod(
            np.array([
                delta_top_arr,
                delta_pre_rnn_arr,
                self.graph.forget_gate_activating_function.derivative(forget_gate_activity_arr)
            ]),
            axis=0
        )
        cdef np.ndarray[DOUBLE_t, ndim=4] delta_input_gate_arr = np.nanprod(
            np.array([
                delta_top_arr,
                given_activity_arr,
                self.graph.input_gate_activating_function.derivative(input_gate_activity_arr)
            ]),
            axis=0
        )
        cdef np.ndarray[DOUBLE_t, ndim=4] delta_given_arr = np.nanprod(
            np.array([
                delta_top_arr,
                input_gate_activity_arr,
                self.graph.observed_activating_function.derivative(given_activity_arr)
            ]),
            axis=0
        )
        
        delta_output_gate_arr = self.__output_conv.back_propagate(delta_output_gate_arr)
        delta_forget_gate_arr = self.__forgot_conv.back_propagate(delta_forget_gate_arr)
        delta_input_gate_arr = self.__input_conv.back_propagate(delta_input_gate_arr)
        delta_given_arr = self.__given_conv.back_propagate(delta_given_arr)

        cdef np.ndarray[DOUBLE_t, ndim=4] delta_pre_hidden_arr = np.nansum(
            np.array([
                delta_output_gate_arr,
                delta_forget_gate_arr,
                delta_input_gate_arr,
                delta_given_arr
            ]),
            axis=0
        )

        return (delta_pre_hidden_arr, delta_pre_rnn_arr)

    def save_pre_learned_params(self, dir_name, file_name=None):
        '''
        Save pre-learned parameters.
        
        Args:
            dir_name:   Path of dir. If `None`, the file is saved in the current directory.
            file_name:  File name.
        '''
        if dir_name[-1] != "/":
            dir_name = dir_name + "/"
        if file_name is None:
            file_name = "conv_lstm_graph"
        elif ".npz" in file_name:
            file_name = file_name.replace(".npz", "")

        self.graph.save_pre_learned_params(dir_name + file_name + ".npz")
        self.__given_conv.graph.save_pre_learned_params(
            dir_name + "given_" + file_name + ".npz"
        )
        self.__input_conv.graph.save_pre_learned_params(
            dir_name + "input_" + file_name + ".npz"
        )
        self.__forgot_conv.graph.save_pre_learned_params(
            dir_name + "forgot_" + file_name + ".npz"
        )
        self.__output_conv.graph.save_pre_learned_params(
            dir_name + "output_" + file_name + ".npz"
        )

    def load_pre_learned_params(self, dir_name, file_name=None):
        '''
        Load pre-learned parameters.
        
        Args:
            dir_name:   Path of dir. If `None`, the file is saved in the current directory.
            file_name:  File name.
        '''
        if dir_name[-1] != "/":
            dir_name = dir_name + "/"
        if file_name is None:
            file_name = "conv_lstm_graph"
        elif ".npz" in file_name:
            file_name = file_name.replace(".npz", "")

        self.graph.load_pre_learned_params(dir_name + file_name + ".npz")
        self.__given_conv.graph.load_pre_learned_params(
            dir_name + "given_" + file_name + ".npz"
        )
        self.__input_conv.graph.load_pre_learned_params(
            dir_name + "input_" + file_name + ".npz"
        )
        self.__forgot_conv.graph.load_pre_learned_params(
            dir_name + "forgot_" + file_name + ".npz"
        )
        self.__output_conv.graph.load_pre_learned_params(
            dir_name + "output_" + file_name + ".npz"
        )

    def get_verificatable_result(self):
        ''' getter '''
        if isinstance(self.__verificatable_result, VerificatableResult):
            return self.__verificatable_result
        else:
            raise TypeError()

    def set_verificatable_result(self, value):
        ''' setter '''
        if isinstance(value, VerificatableResult):
            self.__verificatable_result = value
        else:
            raise TypeError()
    
    verificatable_result = property(get_verificatable_result, set_verificatable_result)
