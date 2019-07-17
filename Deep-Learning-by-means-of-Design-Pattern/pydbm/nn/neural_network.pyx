# -*- coding: utf-8 -*-
from logging import getLogger
from pydbm.nn.nn_layer import NNLayer
from pydbm.optimization.opt_params import OptParams
from pydbm.verification.interface.verificatable_result import VerificatableResult
from pydbm.loss.interface.computable_loss import ComputableLoss
import numpy as np
cimport numpy as np
ctypedef np.float64_t DOUBLE_t


class NeuralNetwork(object):
    '''
    Neural Network.

    References:
        - Kamyshanska, H., & Memisevic, R. (2014). The potential energy of an autoencoder. IEEE transactions on pattern analysis and machine intelligence, 37(6), 1261-1273.
    '''
    # is-a `ComputableLoss`.
    __computable_loss = None
    # is-a `OptParams`.
    __opt_params = None

    def __init__(
        self,
        nn_layer_list,
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
        pre_learned_path_list=None
    ):
        '''
        Init.
        
        Args:
            nn_layer_list:                  The `list` of `NNLayer`.
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
            pre_learned_path_list:          `list` of file path that stores pre-learned parameters.
        '''
        for nn_layer in nn_layer_list:
            if isinstance(nn_layer, NNLayer) is False:
                raise TypeError("The type of value of `nn_layer_list` must be `NNLayer`.")
        
        if pre_learned_path_list is not None:
            if len(pre_learned_path_list) != len(nn_layer_list):
                raise ValueError("The number of files that store pre-learned parameters must be same as the number of `nn_layer_list`.")
            for i in range(len(pre_learned_path_list)):
                nn_layer_list[i].graph.load_pre_learned_params(pre_learned_path_list[i])

        self.__nn_layer_list = nn_layer_list

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

        logger = getLogger("pydbm")
        self.__logger = logger
        
        self.__logger.debug("Setup NN layers and the parameters.")

    def learn(
        self,
        np.ndarray[DOUBLE_t, ndim=2] observed_arr,
        np.ndarray target_arr=None
    ):
        '''
        Learn.
        
        Args:
            observed_arr:   `np.ndarray` of observed data points.
            target_arr:     `np.ndarray` of labeled data.
                            If `None`, the function of this NN model is equivalent to Convolutional Auto-Encoder.

        '''
        self.__logger.debug("NN starts learning.")

        cdef double learning_rate = self.__learning_rate
        cdef int epoch
        cdef int batch_index

        cdef int row_o = observed_arr.shape[0]
        cdef int row_t = 0
        if target_arr is not None:
            row_t = target_arr.shape[0]

        cdef np.ndarray train_index
        cdef np.ndarray test_index
        cdef np.ndarray[DOUBLE_t, ndim=2] train_observed_arr
        cdef np.ndarray train_target_arr
        cdef np.ndarray[DOUBLE_t, ndim=2] test_observed_arr
        cdef np.ndarray test_target_arr

        cdef np.ndarray rand_index
        cdef np.ndarray[DOUBLE_t, ndim=2] batch_observed_arr
        cdef np.ndarray batch_target_arr

        if row_t != 0 and row_t != row_o:
            raise ValueError("The row of `target_arr` must be equivalent to the row of `observed_arr`.")

        if row_t == 0:
            target_arr = observed_arr.copy()

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

                        for i in range(len(self.__nn_layer_list)):
                            best_weight_params_list.append(self.__nn_layer_list[i].graph.weight_arr)
                            best_bias_params_list.append(self.__nn_layer_list[i].graph.bias_arr)
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
            for i in range(len(self.__nn_layer_list)):
                self.__nn_layer_list[i].graph.weight_arr = best_weight_params_list[i]
                self.__nn_layer_list[i].graph.bias_arr = best_bias_params_list[i]
            self.__logger.debug("Best params are saved.")

    def inference(self, np.ndarray[DOUBLE_t, ndim=2] observed_arr):
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

    def forward_propagation(self, np.ndarray[DOUBLE_t, ndim=2] observed_arr):
        '''
        Forward propagation in NN.
        
        Args:
            observed_arr:    `np.ndarray` of image file array.
        
        Returns:
            Propagated `np.ndarray`.
        '''
        cdef int i = 0
        self.__weight_decay_term = 0.0

        for i in range(len(self.__nn_layer_list)):
            try:
                observed_arr = self.__nn_layer_list[i].forward_propagate(observed_arr)
            except:
                self.__logger.debug("Error raised in NN layer " + str(i + 1))
                raise

            self.__weight_decay_term += self.__opt_params.compute_weight_decay(
                self.__nn_layer_list[i].graph.weight_arr
            )

        if self.__opt_params.dropout_rate > 0:
            observed_arr = self.__opt_params.dropout(observed_arr)

        return observed_arr

    def back_propagation(self, np.ndarray delta_arr):
        '''
        Back propagation in NN.
        
        Args:
            Delta.
        
        Returns.
            Delta.
        '''
        if self.__opt_params.dropout_rate > 0:
            delta_arr = self.__opt_params.de_dropout(delta_arr)

        cdef int i = 0
        nn_layer_list = self.__nn_layer_list[::-1]

        for i in range(len(nn_layer_list)):
            try:
                delta_arr = nn_layer_list[i].back_propagate(delta_arr)
            except:
                self.__logger.debug(
                    "Delta computation raised an error in NN layer " + str(len(nn_layer_list) - i)
                )
                raise

        return delta_arr

    def optimize(self, double learning_rate, int epoch):
        '''
        Back propagation.
        
        Args:
            learning_rate:  Learning rate.
            epoch:          Now epoch.
            
        '''
        params_list = []
        grads_list = []

        for i in range(len(self.__nn_layer_list)):
            if self.__nn_layer_list[i].delta_weight_arr.shape[0] > 0:
                self.__nn_layer_list[i].delta_weight_arr += self.__opt_params.compute_weight_decay_delta(
                    self.__nn_layer_list[i].graph.weight_arr
                )
                params_list.append(self.__nn_layer_list[i].graph.weight_arr)
                grads_list.append(self.__nn_layer_list[i].delta_weight_arr)

        for i in range(len(self.__nn_layer_list)):
            if self.__nn_layer_list[i].delta_bias_arr.shape[0] > 0:
                params_list.append(self.__nn_layer_list[i].graph.bias_arr)
                grads_list.append(self.__nn_layer_list[i].delta_bias_arr)

        for i in range(len(self.__nn_layer_list)):
            if self.__nn_layer_list[i].graph.activation_function.batch_norm is not None:
                params_list.append(
                    self.__nn_layer_list[i].graph.activation_function.batch_norm.beta_arr
                )
                grads_list.append(
                    self.__nn_layer_list[i].graph.activation_function.batch_norm.delta_beta_arr
                )
                params_list.append(
                    self.__nn_layer_list[i].graph.activation_function.batch_norm.gamma_arr
                )
                grads_list.append(
                    self.__nn_layer_list[i].graph.activation_function.batch_norm.delta_gamma_arr
                )

        params_list = self.__opt_params.optimize(
            params_list,
            grads_list,
            learning_rate
        )
        
        for i in range(len(self.__nn_layer_list)):
            if self.__nn_layer_list[i].delta_weight_arr.shape[0] > 0:
                self.__nn_layer_list[i].graph.weight_arr = params_list.pop(0)
                if ((epoch + 1) % self.__attenuate_epoch == 0):
                    self.__nn_layer_list[i].graph.weight_arr = self.__opt_params.constrain_weight(
                        self.__nn_layer_list[i].graph.weight_arr
                    )

        for i in range(len(self.__nn_layer_list)):
            if self.__nn_layer_list[i].delta_bias_arr.shape[0] > 0:
                self.__nn_layer_list[i].graph.bias_arr = params_list.pop(0)

        for i in range(len(self.__nn_layer_list)):
            if self.__nn_layer_list[i].graph.activation_function.batch_norm is not None:
                self.__nn_layer_list[i].graph.activation_function.batch_norm.beta_arr = params_list.pop(0)
                self.__nn_layer_list[i].graph.activation_function.batch_norm.gamma_arr = params_list.pop(0)

        for i in range(len(self.__nn_layer_list)):
            if self.__nn_layer_list[i].delta_weight_arr.shape[0] > 0:
                if self.__nn_layer_list[i].delta_bias_arr.shape[0] > 0:
                    self.__nn_layer_list[i].reset_delta()

    def save_pre_learned_params(self, dir_path=None, file_name=None):
        '''
        Save pre-learned parameters.
        
        Args:
            dir_path:   Path of dir. If `None`, the file is saved in the current directory.
            file_name:  The naming rule of files. If `None`, this value is `nn`.
        '''
        file_path = ""
        if dir_path is not None:
            file_path += dir_path + "/"
        if file_name is not None:
            file_path += file_name
        else:
            file_path += "nn"
        file_path += "_"

        for i in range(len(self.nn_layer_list)):
            self.nn_layer_list[i].graph.save_pre_learned_params(file_path + str(i) + ".npz")

    def get_nn_layer_list(self):
        ''' getter '''
        return self.__nn_layer_list

    def set_nn_layer_list(self, value):
        ''' setter '''
        for nn_layer in value:
            if isinstance(nn_layer, NNLayer) is False:
                raise TypeError()

        self.__nn_layer_list = value

    nn_layer_list = property(get_nn_layer_list, set_nn_layer_list)

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
