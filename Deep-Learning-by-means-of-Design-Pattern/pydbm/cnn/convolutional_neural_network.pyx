# -*- coding: utf-8 -*-
from logging import getLogger
from pydbm.synapse.cnn_output_graph import CNNOutputGraph
from pydbm.cnn.layerable_cnn import LayerableCNN
from pydbm.cnn.feature_generator import FeatureGenerator
from pydbm.optimization.opt_params import OptParams
from pydbm.verification.interface.verificatable_result import VerificatableResult
from pydbm.loss.interface.computable_loss import ComputableLoss
import numpy as np
cimport numpy as np
ctypedef np.float64_t DOUBLE_t


class ConvolutionalNeuralNetwork(object):
    '''
    Convolutional Neural Network.
    '''
    # is-a `ComputableLoss`.
    __computable_loss = None
    # is-a `OptParams`.
    __opt_params = None

    # Computation graph which is-a `CNNOutputGraph` to compute parameters in output layer.
    __cnn_output_graph = None

    # Penalty term of weight decay.
    __weight_decay_term = 0.0

    def __init__(
        self,
        layerable_cnn_list,
        computable_loss,
        opt_params,
        verificatable_result,
        int epochs=100,
        int batch_size=100,
        double learning_rate=1e-05,
        double learning_attenuate_rate=0.1,
        int attenuate_epoch=50,
        double test_size_rate=0.3,
        tol=1e-15,
        tld=100.0,
        save_flag=False,
        pre_learned_path_list=None
    ):
        '''
        Init.
        
        Args:
            layerable_cnn_list:             The `list` of `LayerableCNN`.
            computable_loss:                Loss function.
            opt_params:                     Optimization function.
            verificatable_result:           Verification function.

            epochs:                         Epochs of Mini-batch.
            bath_size:                      Batch size of Mini-batch.
            learning_rate:                  Learning rate.
            learning_attenuate_rate:        Attenuate the `learning_rate` by a factor of this value every `attenuate_epoch`.
            attenuate_epoch:                Attenuate the `learning_rate` by a factor of `learning_attenuate_rate` every `attenuate_epoch`.
                                            Additionally, in relation to regularization,
                                            this class constrains weight matrixes every `attenuate_epoch`.

            test_size_rate:                 Size of Test data set. If this value is `0`, the validation will not be executed.
            tol:                            Tolerance for the optimization.
            tld:                            Tolerance for deviation of loss.
            save_flag:                      If `True`, save `np.ndarray` of inferenced test data in training.
            pre_learned_path_list:          `list` of pre-learned parameters.
        '''
        for layerable_cnn in layerable_cnn_list:
            if isinstance(layerable_cnn, LayerableCNN) is False:
                raise TypeError("The type of value of `layerable_cnn` must be `LayerableCNN`.")
        
        if pre_learned_path_list is not None:
            if len(pre_learned_path_list) != len(layerable_cnn_list):
                raise ValueError("The number of files that store pre-learned parameters must be same as the number of `layerable_cnn_list`.")
            for i in range(len(pre_learned_path_list)):
                layerable_cnn_list[i].graph.load_pre_learned_params(pre_learned_path_list[i])

        self.__layerable_cnn_list = layerable_cnn_list

        if isinstance(computable_loss, ComputableLoss):
            self.__computable_loss = computable_loss
        else:
            raise TypeError("The type of `computable_loss` must be `ComputableLoss`.")

        if isinstance(opt_params, OptParams):
            self.__opt_params = opt_params
            self.__dropout_rate = self.__opt_params.dropout_rate
        else:
            raise TypeError("The type of `opt_params` must be `OptParams`.")

        if isinstance(verificatable_result, VerificatableResult):
            self.__verificatable_result = verificatable_result
        else:
            raise TypeError("The type of `verificatable_result` must be `VerificatableResult`.")

        self.__epochs = epochs
        self.__batch_size = batch_size

        self.__learning_rate = learning_rate
        self.__learning_attenuate_rate = learning_attenuate_rate
        self.__attenuate_epoch = attenuate_epoch

        self.__test_size_rate = test_size_rate
        self.__tol = tol
        self.__tld = tld

        self.__memory_tuple_list = []
        
        self.__save_flag = save_flag

        logger = getLogger("pydbm")
        self.__logger = logger
        
        self.__logger.debug("Setup CNN layers and the parameters.")

    def setup_output_layer(
        self,
        cnn_output_graph,
        pre_learned_path=None
    ):
        '''
        Setup output layer.

        Args:
            cnn_output_graph:           Computation graph which is-a `CNNOutputGraph` to compute parameters in output layer.
            pre_learned_path:           File path that stores pre-learned parameters.
        '''
        if isinstance(cnn_output_graph, CNNOutputGraph) is False:
            raise TypeError("The type of `cnn_output_graph` must be `CNNOutputGraph`.")

        self.__cnn_output_graph = cnn_output_graph
        if self.__cnn_output_graph is not None and pre_learned_path is not None:
            self.__cnn_output_graph.load_pre_learned_params(pre_learned_path)

    def learn(
        self,
        np.ndarray[DOUBLE_t, ndim=4] observed_arr,
        np.ndarray target_arr=None
    ):
        '''
        Learn.
        
        Args:
            observed_arr:   `np.ndarray` of observed data points.
            target_arr:     `np.ndarray` of labeled data.
                            If `None`, the function of this cnn model is equivalent to Convolutional Auto-Encoder.

        '''
        self.__logger.debug("CNN starts learning.")

        cdef double learning_rate = self.__learning_rate
        cdef int epoch
        cdef int batch_index

        cdef int row_o = observed_arr.shape[0]
        cdef int row_t = 0
        if target_arr is not None:
            row_t = target_arr.shape[0]

        cdef np.ndarray train_index
        cdef np.ndarray test_index
        cdef np.ndarray[DOUBLE_t, ndim=4] train_observed_arr
        cdef np.ndarray train_target_arr
        cdef np.ndarray[DOUBLE_t, ndim=4] test_observed_arr
        cdef np.ndarray test_target_arr

        cdef np.ndarray rand_index
        cdef np.ndarray[DOUBLE_t, ndim=4] batch_observed_arr
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
            train_target_arr = target_arr

        cdef double loss
        cdef double test_loss
        cdef np.ndarray pred_arr
        cdef np.ndarray test_pred_arr
        cdef np.ndarray delta_arr
        
        best_weight_params_list = []
        best_bias_params_list = []

        try:
            self.__memory_tuple_list = []
            loss_list = []
            min_loss = None
            eary_stop_flag = False
            for epoch in range(self.__epochs):
                self.__opt_params.dropout_rate = self.__dropout_rate
                self.__opt_params.inferencing_mode = False

                if ((epoch + 1) % self.__attenuate_epoch == 0):
                    learning_rate = learning_rate * self.__learning_attenuate_rate

                rand_index = np.random.choice(train_observed_arr.shape[0], size=self.__batch_size)
                batch_observed_arr = train_observed_arr[rand_index]
                batch_target_arr = train_target_arr[rand_index]

                try:
                    pred_arr = self.inference(batch_observed_arr)
                    ver_pred_arr = pred_arr.copy()
                    train_weight_decay = self.__weight_decay_term
                    loss = self.__computable_loss.compute_loss(
                        pred_arr,
                        batch_target_arr
                    )
                    loss = loss + train_weight_decay

                    remember_flag = False
                    if len(loss_list) > 0:
                        if abs(loss - (sum(loss_list)/len(loss_list))) > self.__tld:
                            remember_flag = True

                    if remember_flag is True:
                        self.__remember_best_params(best_weight_params_list, best_bias_params_list)
                        # Re-try.
                        pred_arr = self.inference(batch_observed_arr)
                        ver_pred_arr = pred_arr.copy()
                        train_weight_decay = self.__weight_decay_term
                        loss = self.__computable_loss.compute_loss(
                            pred_arr,
                            batch_target_arr
                        )
                        loss = loss + train_weight_decay

                    delta_arr = self.__computable_loss.compute_delta(
                        pred_arr,
                        batch_target_arr
                    )
                    delta_arr = self.back_propagation(delta_arr)
                    self.optimize(learning_rate, epoch)

                    if min_loss is None or min_loss > loss:
                        min_loss = loss
                        best_weight_params_list = []
                        best_bias_params_list = []

                        for i in range(len(self.__layerable_cnn_list)):
                            best_weight_params_list.append(self.__layerable_cnn_list[i].graph.weight_arr)
                            best_bias_params_list.append(self.__layerable_cnn_list[i].graph.bias_arr)
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
                    self.__opt_params.inferencing_mode = True
                    rand_index = np.random.choice(test_observed_arr.shape[0], size=self.__batch_size)
                    test_batch_observed_arr = test_observed_arr[rand_index]
                    test_batch_target_arr = test_target_arr[rand_index]

                    test_pred_arr = self.forward_propagation(
                        test_batch_observed_arr
                    )
                    test_weight_decay = self.__weight_decay_term
                    test_loss = self.__computable_loss.compute_loss(
                        test_pred_arr,
                        test_batch_target_arr
                    )
                    test_loss = test_loss + test_weight_decay

                    remember_flag = False
                    if len(loss_list) > 0:
                        if abs(test_loss - (sum(loss_list)/len(loss_list))) > self.__tld:
                            remember_flag = True

                    if remember_flag is True:
                        self.__remember_best_params(best_weight_params_list, best_bias_params_list)
                        # Re-try
                        test_pred_arr = self.forward_propagation(
                            test_batch_observed_arr
                        )

                    if self.__save_flag is True:
                        np.save("test_pred_arr_" + str(epoch), test_pred_arr)

                    if self.__verificatable_result is not None:
                        if self.__test_size_rate > 0:
                            self.__verificatable_result.verificate(
                                self.__computable_loss,
                                train_pred_arr=ver_pred_arr, 
                                train_label_arr=batch_target_arr,
                                test_pred_arr=test_pred_arr,
                                test_label_arr=test_batch_target_arr,
                                train_penalty=train_weight_decay,
                                test_penalty=test_weight_decay
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

        self.__remember_best_params(best_weight_params_list, best_bias_params_list)
        self.__logger.debug("end. ")

    def learn_generated(self, feature_generator):
        '''
        Learn features generated by `FeatureGenerator`.
        
        Args:
            feature_generator:    is-a `FeatureGenerator`.

        '''
        if isinstance(feature_generator, FeatureGenerator) is False:
            raise TypeError("The type of `feature_generator` must be `FeatureGenerator`.")

        self.__logger.debug("CNN starts learning.")

        cdef double learning_rate = self.__learning_rate
        cdef int epoch
        cdef int batch_index

        cdef np.ndarray train_index
        cdef np.ndarray test_index
        cdef np.ndarray[DOUBLE_t, ndim=4] train_observed_arr
        cdef np.ndarray train_target_arr
        cdef np.ndarray[DOUBLE_t, ndim=4] test_observed_arr
        cdef np.ndarray test_target_arr

        cdef np.ndarray rand_index
        cdef np.ndarray[DOUBLE_t, ndim=4] batch_observed_arr
        cdef np.ndarray batch_target_arr

        cdef double loss
        cdef double test_loss
        cdef np.ndarray pred_arr
        cdef np.ndarray test_pred_arr
        cdef np.ndarray delta_arr

        best_weight_params_list = []
        best_bias_params_list = []

        try:
            self.__memory_tuple_list = []
            loss_list = []
            min_loss = None
            eary_stop_flag = False
            epoch = 0
            for batch_observed_arr, batch_target_arr, test_batch_observed_arr, test_batch_target_arr in feature_generator.generate():
                epoch += 1
                self.__opt_params.dropout_rate = self.__dropout_rate
                self.__opt_params.inferencing_mode = False

                if ((epoch + 1) % self.__attenuate_epoch == 0):
                    learning_rate = learning_rate * self.__learning_attenuate_rate

                try:
                    pred_arr = self.inference(batch_observed_arr)
                    ver_pred_arr = pred_arr.copy()
                    train_weight_decay = self.__weight_decay_term
                    loss = self.__computable_loss.compute_loss(
                        pred_arr,
                        batch_target_arr
                    )
                    loss = loss + train_weight_decay

                    remember_flag = False
                    if len(loss_list) > 0:
                        if abs(loss - (sum(loss_list)/len(loss_list))) > self.__tld:
                            remember_flag = True

                    if remember_flag is True:
                        self.__remember_best_params(best_weight_params_list, best_bias_params_list)
                        # Re-try.
                        pred_arr = self.inference(batch_observed_arr)
                        ver_pred_arr = pred_arr.copy()
                        train_weight_decay = self.__weight_decay_term
                        loss = self.__computable_loss.compute_loss(
                            pred_arr,
                            batch_target_arr
                        )
                        loss = loss + train_weight_decay

                    delta_arr = self.__computable_loss.compute_delta(
                        pred_arr,
                        batch_target_arr
                    )
                    delta_arr = self.back_propagation(delta_arr)
                    self.optimize(learning_rate, epoch)
                    if min_loss is None or min_loss > loss:
                        min_loss = loss
                        best_weight_params_list = []
                        best_bias_params_list = []

                        for i in range(len(self.__layerable_cnn_list)):
                            best_weight_params_list.append(self.__layerable_cnn_list[i].graph.weight_arr)
                            best_bias_params_list.append(self.__layerable_cnn_list[i].graph.bias_arr)
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
                    self.__opt_params.inferencing_mode = True
                    test_pred_arr = self.forward_propagation(
                        test_batch_observed_arr
                    )
                    test_weight_decay = self.__weight_decay_term
                    test_loss = self.__computable_loss.compute_loss(
                        test_pred_arr,
                        test_batch_target_arr
                    )
                    test_loss = test_loss + test_weight_decay

                    remember_flag = False
                    if len(loss_list) > 0:
                        if abs(test_loss - (sum(loss_list)/len(loss_list))) > self.__tld:
                            remember_flag = True

                    if remember_flag is True:
                        self.__remember_best_params(best_weight_params_list, best_bias_params_list)
                        # Re-try
                        test_pred_arr = self.forward_propagation(
                            test_batch_observed_arr
                        )

                    if self.__save_flag is True:
                        np.save("test_pred_arr_" + str(epoch), test_pred_arr)

                    if self.__verificatable_result is not None:
                        if self.__test_size_rate > 0:
                            self.__verificatable_result.verificate(
                                self.__computable_loss,
                                train_pred_arr=ver_pred_arr, 
                                train_label_arr=batch_target_arr,
                                test_pred_arr=test_pred_arr,
                                test_label_arr=test_batch_target_arr,
                                train_penalty=train_weight_decay,
                                test_penalty=test_weight_decay
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

        self.__remember_best_params(best_weight_params_list, best_bias_params_list)
        self.__logger.debug("end. ")

    def __remember_best_params(self, best_weight_params_list, best_bias_params_list):
        '''
        Remember best parameters.
        
        Args:
            best_weight_params_list:    `list` of weight parameters.
            best_bias_params_list:      `list` of bias parameters.

        '''
        if len(best_weight_params_list) and len(best_bias_params_list):
            for i in range(len(self.__layerable_cnn_list)):
                if self.__layerable_cnn_list[i].graph.constant_flag is False:
                    self.__layerable_cnn_list[i].graph.weight_arr = best_weight_params_list[i]
                    self.__layerable_cnn_list[i].graph.bias_arr = best_bias_params_list[i]
            self.__logger.debug("Best params are saved.")

    def inference(self, np.ndarray[DOUBLE_t, ndim=4] observed_arr):
        '''
        Inference the feature points to reconstruct the time-series.

        Override.

        Args:
            observed_arr:           Array like or sparse matrix as the observed data points.

        Returns:
            Predicted array like or sparse matrix.
        '''
        cdef np.ndarray pred_arr = self.forward_propagation(
            observed_arr
        )
        return pred_arr

    def forward_propagation(self, np.ndarray[DOUBLE_t, ndim=4] img_arr):
        '''
        Forward propagation in CNN.
        
        Args:
            img_arr:    `np.ndarray` of image file array.
        
        Returns:
            Propagated `np.ndarray`.
        '''
        cdef int i = 0
        self.__weight_decay_term = 0.0
        for i in range(len(self.__layerable_cnn_list)):
            try:
                img_arr = self.__layerable_cnn_list[i].forward_propagate(img_arr)
            except:
                self.__logger.debug("Error raised in CNN layer " + str(i + 1))
                raise
            if self.__layerable_cnn_list[i].graph.constant_flag is False:
                self.__weight_decay_term += self.__opt_params.compute_weight_decay(
                    self.__layerable_cnn_list[i].graph.weight_arr
                )

        if self.__opt_params.dropout_rate > 0:
            hidden_activity_arr = img_arr.reshape((img_arr.shape[0], -1))
            hidden_activity_arr = self.__opt_params.dropout(hidden_activity_arr)
            img_arr = hidden_activity_arr.reshape((
                img_arr.shape[0],
                img_arr.shape[1],
                img_arr.shape[2],
                img_arr.shape[3]
            ))

        if self.__cnn_output_graph is not None:
            return self.output_forward_propagate(img_arr)
        else:
            return img_arr

    def output_forward_propagate(self, np.ndarray[DOUBLE_t, ndim=4] pred_arr):
        '''
        Forward propagation in output layer.
        
        Args:
            pred_arr:            `np.ndarray` of predicted data points.

        Returns:
            `np.ndarray` of propagated data points.
        '''
        cdef np.ndarray[DOUBLE_t, ndim=2] _pred_arr
        if self.__cnn_output_graph is not None:
            _pred_arr = self.__cnn_output_graph.activating_function.activate(
                np.dot(
                    pred_arr.reshape((pred_arr.shape[0], -1)), self.__cnn_output_graph.weight_arr) + self.__cnn_output_graph.bias_arr
            )
            self.__cnn_output_graph.hidden_arr = pred_arr
            self.__cnn_output_graph.output_arr = _pred_arr

            self.__weight_decay_term += self.__opt_params.compute_weight_decay(
                self.__cnn_output_graph.weight_arr
            )

            return _pred_arr
        else:
            return pred_arr

    def back_propagation(self, np.ndarray delta_arr):
        '''
        Back propagation in CNN.
        
        Args:
            Delta.
        
        Returns.
            Delta.
        '''
        cdef np.ndarray[DOUBLE_t, ndim=2] _delta_arr
        if self.__cnn_output_graph is not None:
            if delta_arr.ndim != 2:
                _delta_arr = delta_arr.reshape((delta_arr.shape[0], -1))
            else:
                _delta_arr = delta_arr

            _delta_arr, output_grads_list = self.output_back_propagate(
                self.__cnn_output_graph.output_arr.reshape((self.__cnn_output_graph.output_arr.shape[0], -1)), 
                _delta_arr.reshape((_delta_arr.shape[0], -1))
            )
            delta_arr = _delta_arr.reshape((
                self.__cnn_output_graph.hidden_arr.shape[0],
                self.__cnn_output_graph.hidden_arr.shape[1],
                self.__cnn_output_graph.hidden_arr.shape[2],
                self.__cnn_output_graph.hidden_arr.shape[3]
            ))
            self.__cnn_output_graph.output_grads_list = output_grads_list

        if self.__opt_params.dropout_rate > 0:
            hidden_activity_arr = delta_arr.reshape((delta_arr.shape[0], -1))
            hidden_activity_arr = self.__opt_params.de_dropout(hidden_activity_arr)
            delta_arr = hidden_activity_arr.reshape((
                delta_arr.shape[0],
                delta_arr.shape[1],
                delta_arr.shape[2],
                delta_arr.shape[3]
            ))

        cdef int i = 0
        layerable_cnn_list = self.__layerable_cnn_list[::-1]

        for i in range(len(layerable_cnn_list)):
            try:
                delta_arr = layerable_cnn_list[i].back_propagate(delta_arr)
            except:
                self.__logger.debug(
                    "Delta computation raised an error in CNN layer " + str(len(layerable_cnn_list) - i)
                )
                raise

        return delta_arr

    def output_back_propagate(
        self, 
        np.ndarray[DOUBLE_t, ndim=2] pred_arr, 
        np.ndarray[DOUBLE_t, ndim=2] delta_arr
    ):
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
        cdef np.ndarray[DOUBLE_t, ndim=2] _delta_arr = np.dot(
            delta_arr,
            self.__cnn_output_graph.weight_arr.T
        )
        cdef np.ndarray[DOUBLE_t, ndim=2] delta_weights_arr = np.dot(
            self.__cnn_output_graph.hidden_arr.reshape((
                self.__cnn_output_graph.hidden_arr.shape[0],
                -1
            )).T, 
            delta_arr
        )
        cdef np.ndarray[DOUBLE_t, ndim=1] delta_bias_arr = np.sum(delta_arr, axis=0)

        grads_list = [
            delta_weights_arr,
            delta_bias_arr
        ]
        
        return (_delta_arr, grads_list)

    def optimize(self, double learning_rate, int epoch):
        '''
        Back propagation.
        
        Args:
            learning_rate:  Learning rate.
            epoch:          Now epoch.
            
        '''
        params_list = []
        grads_list = []

        if self.__cnn_output_graph is not None:
            self.__cnn_output_graph.output_grads_list[0] += self.__opt_params.compute_weight_decay_delta(
                self.__cnn_output_graph.weight_arr
            )
            params_list.append(self.__cnn_output_graph.weight_arr)
            params_list.append(self.__cnn_output_graph.bias_arr)
            grads_list.append(self.__cnn_output_graph.output_grads_list[0])
            grads_list.append(self.__cnn_output_graph.output_grads_list[1])

        for i in range(len(self.__layerable_cnn_list)):
            if self.__layerable_cnn_list[i].graph.constant_flag is False:
                if self.__layerable_cnn_list[i].delta_weight_arr.shape[0] > 0:
                    self.__layerable_cnn_list[i].delta_weight_arr += self.__opt_params.compute_weight_decay_delta(
                        self.__layerable_cnn_list[i].graph.weight_arr
                    )
                    params_list.append(self.__layerable_cnn_list[i].graph.weight_arr)
                    grads_list.append(self.__layerable_cnn_list[i].delta_weight_arr)

        for i in range(len(self.__layerable_cnn_list)):
            if self.__layerable_cnn_list[i].graph.constant_flag is False:
                if self.__layerable_cnn_list[i].delta_bias_arr.shape[0] > 0:
                    params_list.append(self.__layerable_cnn_list[i].graph.bias_arr)
                    grads_list.append(self.__layerable_cnn_list[i].delta_bias_arr)

        for i in range(len(self.__layerable_cnn_list)):
            if self.__layerable_cnn_list[i].graph.constant_flag is False:
                if self.__layerable_cnn_list[i].graph.activation_function.batch_norm is not None:
                    params_list.append(
                        self.__layerable_cnn_list[i].graph.activation_function.batch_norm.beta_arr
                    )
                    grads_list.append(
                        self.__layerable_cnn_list[i].graph.activation_function.batch_norm.delta_beta_arr
                    )
                    params_list.append(
                        self.__layerable_cnn_list[i].graph.activation_function.batch_norm.gamma_arr
                    )
                    grads_list.append(
                        self.__layerable_cnn_list[i].graph.activation_function.batch_norm.delta_gamma_arr
                    )

        params_list = self.__opt_params.optimize(
            params_list,
            grads_list,
            learning_rate
        )
        
        if self.__cnn_output_graph is not None:
            self.__cnn_output_graph.weight_arr = params_list.pop(0)
            self.__cnn_output_graph.bias_arr = params_list.pop(0)

        i = 0
        for i in range(len(self.__layerable_cnn_list)):
            if self.__layerable_cnn_list[i].graph.constant_flag is False:
                if self.__layerable_cnn_list[i].delta_weight_arr.shape[0] > 0:
                    self.__layerable_cnn_list[i].graph.weight_arr = params_list.pop(0)
                    if ((epoch + 1) % self.__attenuate_epoch == 0):
                        self.__layerable_cnn_list[i].graph.weight_arr = self.__opt_params.constrain_weight(
                            self.__layerable_cnn_list[i].graph.weight_arr
                        )

        for i in range(len(self.__layerable_cnn_list)):
            if self.__layerable_cnn_list[i].graph.constant_flag is False:
                if self.__layerable_cnn_list[i].delta_bias_arr.shape[0] > 0:
                    self.__layerable_cnn_list[i].graph.bias_arr = params_list.pop(0)

        for i in range(len(self.__layerable_cnn_list)):
            if self.__layerable_cnn_list[i].graph.constant_flag is False:
                if self.__layerable_cnn_list[i].graph.activation_function.batch_norm is not None:
                    self.__layerable_cnn_list[i].graph.activation_function.batch_norm.gamma_arr = params_list.pop(0)
                    self.__layerable_cnn_list[i].graph.activation_function.batch_norm.beta_arr = params_list.pop(0)

        for i in range(len(self.__layerable_cnn_list)):
            if self.__layerable_cnn_list[i].graph.constant_flag is False:
                if self.__layerable_cnn_list[i].delta_weight_arr.shape[0] > 0:
                    if self.__layerable_cnn_list[i].delta_bias_arr.shape[0] > 0:
                        self.__layerable_cnn_list[i].reset_delta()

    def save_pre_learned_params(self, dir_path=None, file_name=None):
        '''
        Save pre-learned parameters.
        
        Args:
            dir_path:   Path of dir. If `None`, the file is saved in the current directory.
            file_name:  The naming rule of files. If `None`, this value is `cnn`.
        '''
        file_path = ""
        if dir_path is not None:
            file_path += dir_path + "/"
        if file_name is not None:
            file_path += file_name
        else:
            file_path += "cnn"
        file_path += "_"

        if self.__cnn_output_graph is not None:
            self.__cnn_output_graph.save_pre_learned_params(file_path + "_output_layer.npz")

        for i in range(len(self.layerable_cnn_list)):
            if self.__layerable_cnn_list[i].graph.constant_flag is False:
                self.layerable_cnn_list[i].graph.save_pre_learned_params(file_path + str(i) + ".npz")

    def get_layerable_cnn_list(self):
        ''' getter '''
        return self.__layerable_cnn_list

    def set_layerable_cnn_list(self, value):
        ''' setter '''
        for layerable_cnn in value:
            if isinstance(layerable_cnn, LayerableCNN) is False:
                raise TypeError()

        self.__layerable_cnn_list = value

    layerable_cnn_list = property(get_layerable_cnn_list, set_layerable_cnn_list)

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

    def get_opt_params(self):
        ''' getter '''
        return self.__opt_params
    
    def set_opt_params(self, value):
        ''' setter '''
        self.__opt_params = value
    
    opt_params = property(get_opt_params, set_opt_params)

    def get_computable_loss(self):
        ''' getter '''
        return self.__computable_loss
    
    def set_computable_loss(self, value):
        ''' setter '''
        self.__computable_loss = value

    computable_loss = property(get_computable_loss, set_computable_loss)

    def get_weight_decay_term(self):
        ''' getter '''
        return self.__weight_decay_term
    
    def set_weight_decay_term(self, value):
        ''' setter '''
        self.__weight_decay_term = value
    
    weight_decay_term = property(get_weight_decay_term, set_weight_decay_term)
