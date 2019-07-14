# -*- coding: utf-8 -*-
from logging import getLogger
from pydbm.nn.simple_auto_encoder import SimpleAutoEncoder
from pydbm.activation.logistic_function import LogisticFunction
from pydbm.params_initializer import ParamsInitializer
import numpy as np
cimport numpy as np
ctypedef np.float64_t DOUBLE_t


class LadderNetworks(SimpleAutoEncoder):
    '''
    Ladder Networks.

    References:
        - Bengio, Y., Lamblin, P., Popovici, D., & Larochelle, H. (2007). Greedy layer-wise training of deep networks. In Advances in neural information processing systems (pp. 153-160).
        - Erhan, D., Bengio, Y., Courville, A., Manzagol, P. A., Vincent, P., & Bengio, S. (2010). Why does unsupervised pre-training help deep learning?. Journal of Machine Learning Research, 11(Feb), 625-660.
        - Erhan, D., Courville, A., & Bengio, Y. (2010). Understanding representations learned in deep architectures. Department dInformatique et Recherche Operationnelle, University of Montreal, QC, Canada, Tech. Rep, 1355, 1.
        - Goodfellow, I., Bengio, Y., & Courville, A. (2016). Deep learning (adaptive computation and machine learning series). Adaptive Computation and Machine Learning series, 800.
        - Rasmus, A., Berglund, M., Honkala, M., Valpola, H., & Raiko, T. (2015). Semi-supervised learning with ladder networks. In Advances in neural information processing systems (pp. 3546-3554).
        - Valpola, H. (2015). From neural PCA to deep unsupervised learning. In Advances in Independent Component Analysis and Learning Machines (pp. 143-171). Academic Press.
    '''

    def __init__(
        self,
        encoder,
        decoder,
        computable_loss,
        verificatable_result,
        int epochs=100,
        int batch_size=100,
        double learning_rate=1e-05,
        double learning_attenuate_rate=0.1,
        int attenuate_epoch=50,
        double test_size_rate=0.3,
        tol=1e-15,
        tld=100.0,
        pre_learned_path_tuple=None,
        alpha_weight=1e-05,
        sigma_weight=0.7,
        mu_weight=0.7,
        params_initializer=ParamsInitializer(),
        params_dict={"loc": 0.0, "scale": 1.0}
    ):
        '''
        Init.
        
        Args:
            encoder:                        is-a `NeuralNetwork`.
            decoder:                        is-a `NeuralNetwork`.
            computable_loss:                Loss function.
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

            alpha_weight:                   Weight of alpha cost.
            sigma_weight:                   Weight of sigma cost.
            mu_weight:                      Weight of mu cost.

            params_initializer:             is-a `ParamsInitializer`. This class will noise 
                                            observed data points and hidden units by using this 
                                            `params_initializer`.

            params_dict:                     `dict` of parameters other than `size` to be input to function `ParamsInitializer.sample_f`.

        '''
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

        super().__init__(
            encoder=encoder,
            decoder=decoder,
            computable_loss=computable_loss,
            verificatable_result=verificatable_result,
            epochs=epochs,
            batch_size=batch_size,
            learning_rate=learning_rate,
            learning_attenuate_rate=learning_attenuate_rate,
            attenuate_epoch=attenuate_epoch,
            test_size_rate=test_size_rate,
            tol=tol,
            tld=tld,
            pre_learned_path_tuple=pre_learned_path_tuple
        )

        self.__alpha_weight = alpha_weight
        self.__sigma_weight = sigma_weight
        self.__mu_weight = mu_weight
        self.__params_initializer = params_initializer
        self.__params_dict = params_dict

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
        
        best_encoder_weight_params_list = []
        best_encoder_bias_params_list = []
        best_decoder_weight_params_list = []
        best_decoder_bias_params_list = []

        try:
            self.__memory_tuple_list = []
            loss_list = []
            alpha_list = []
            sigma_list = []
            mu_list = []
            min_loss = None
            eary_stop_flag = False
            for epoch in range(self.__epochs):
                self.encoder.opt_params.inferencing_mode = False
                self.decoder.opt_params.inferencing_mode = False

                if ((epoch + 1) % self.__attenuate_epoch == 0):
                    learning_rate = learning_rate * self.__learning_attenuate_rate

                rand_index = np.random.choice(train_observed_arr.shape[0], size=self.__batch_size)
                batch_observed_arr = train_observed_arr[rand_index]
                batch_target_arr = train_target_arr[rand_index]

                try:
                    pred_arr = self.inference(batch_observed_arr)
                    ver_pred_arr = pred_arr.copy()
                    train_weight_decay = self.encoder.weight_decay_term + self.decoder.weight_decay_term
                    train_alpha_loss = self.compute_alpha_loss()
                    train_sigma_loss = self.compute_sigma_loss()
                    train_mu_loss = self.compute_mu_loss()
                    loss = self.computable_loss.compute_loss(
                        pred_arr,
                        batch_target_arr
                    )
                    loss = loss + train_alpha_loss + train_sigma_loss + train_mu_loss + train_weight_decay

                    remember_flag = False
                    if len(loss_list) > 0:
                        if abs(loss - (sum(loss_list)/len(loss_list))) > self.__tld:
                            remember_flag = True

                    if remember_flag is True:
                        self.__remember_best_params(
                            best_encoder_weight_params_list, 
                            best_encoder_bias_params_list,
                            best_decoder_weight_params_list, 
                            best_decoder_bias_params_list
                        )
                        # Re-try.
                        pred_arr = self.inference(batch_observed_arr)
                        ver_pred_arr = pred_arr.copy()
                        train_weight_decay = self.encoder.weight_decay_term + self.decoder.weight_decay_term
                        train_alpha_loss = self.compute_alpha_loss()
                        train_sigma_loss = self.compute_sigma_loss()
                        train_mu_loss = self.compute_mu_loss()
                        loss = self.computable_loss.compute_loss(
                            pred_arr,
                            batch_target_arr
                        )
                        loss = loss + train_alpha_loss + train_sigma_loss + train_mu_loss + train_weight_decay

                    delta_arr = self.computable_loss.compute_delta(
                        pred_arr,
                        batch_target_arr
                    )
                    delta_arr = self.back_propagation(delta_arr)
                    self.optimize(learning_rate, epoch)

                    if min_loss is None or min_loss > loss:
                        min_loss = loss
                        best_encoder_weight_params_list = []
                        best_encoder_bias_params_list = []
                        best_decoder_weight_params_list = []
                        best_decoder_bias_params_list = []

                        for i in range(len(self.encoder.nn_layer_list)):
                            best_encoder_weight_params_list.append(self.encoder.nn_layer_list[i].graph.weight_arr)
                            best_encoder_bias_params_list.append(self.encoder.nn_layer_list[i].graph.bias_arr)
                        for i in range(len(self.decoder.nn_layer_list)):
                            best_decoder_weight_params_list.append(self.decoder.nn_layer_list[i].graph.weight_arr)
                            best_decoder_bias_params_list.append(self.decoder.nn_layer_list[i].graph.bias_arr)

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
                    self.encoder.opt_params.inferencing_mode = True
                    self.decoder.opt_params.inferencing_mode = True

                    rand_index = np.random.choice(test_observed_arr.shape[0], size=self.__batch_size)
                    test_batch_observed_arr = test_observed_arr[rand_index]
                    test_batch_target_arr = test_target_arr[rand_index]

                    test_pred_arr = self.forward_propagation(
                        test_batch_observed_arr
                    )
                    test_weight_decay = self.encoder.weight_decay_term + self.decoder.weight_decay_term
                    test_alpha_loss = self.compute_alpha_loss()
                    test_sigma_loss = self.compute_sigma_loss()
                    test_mu_loss = self.compute_mu_loss()
                    test_loss = self.computable_loss.compute_loss(
                        test_pred_arr + self.encoder.weight_decay_term + self.decoder.weight_decay_term,
                        test_batch_target_arr
                    )
                    test_loss = test_loss + test_alpha_loss + test_sigma_loss + test_mu_loss + test_weight_decay

                    remember_flag = False
                    if len(loss_list) > 0:
                        if abs(test_loss - (sum(loss_list)/len(loss_list))) > self.__tld:
                            remember_flag = True

                    if remember_flag is True:
                        self.__remember_best_params(
                            best_encoder_weight_params_list, 
                            best_encoder_bias_params_list,
                            best_decoder_weight_params_list, 
                            best_decoder_bias_params_list
                        )
                        # Re-try
                        test_pred_arr = self.forward_propagation(
                            test_batch_observed_arr
                        )

                    if self.verificatable_result is not None:
                        if self.__test_size_rate > 0:
                            self.verificatable_result.verificate(
                                self.computable_loss,
                                train_pred_arr=ver_pred_arr, 
                                train_label_arr=batch_target_arr,
                                test_pred_arr=test_pred_arr,
                                test_label_arr=test_batch_target_arr,
                                train_penalty=train_alpha_loss + train_sigma_loss + train_mu_loss + train_weight_decay,
                                test_penalty=test_alpha_loss + test_sigma_loss + test_mu_loss + test_weight_decay
                            )
                            self.__logger.debug("-" * 100)
                            self.__logger.debug("Train alpha: " + str(train_alpha_loss) + " Test alpha: " + str(test_alpha_loss))
                            self.__logger.debug("Train sigma: " + str(train_sigma_loss) + " Test sigma: " + str(test_sigma_loss))
                            self.__logger.debug("Train mu: " + str(train_mu_loss) + " Test mu: " + str(test_mu_loss))
                            self.__logger.debug("-" * 100)

                            alpha_list.append((train_alpha_loss, test_alpha_loss))
                            sigma_list.append((train_sigma_loss, test_sigma_loss))
                            mu_list.append((train_mu_loss, test_mu_loss))

                if epoch > 1 and abs(loss - loss_list[-1]) < self.__tol:
                    eary_stop_flag = True
                    break
                loss_list.append(loss)

        except KeyboardInterrupt:
            self.__logger.debug("Interrupt.")

        if eary_stop_flag is True:
            self.__logger.debug("Eary stopping.")
            eary_stop_flag = False

        self.__remember_best_params(
            best_encoder_weight_params_list, 
            best_encoder_bias_params_list,
            best_decoder_weight_params_list, 
            best_decoder_bias_params_list
        )

        self.__alpha_loss_arr = np.array(alpha_list)
        self.__sigma_loss_arr = np.array(sigma_list)
        self.__mu_loss_arr = np.array(mu_list)

        self.__logger.debug("end. ")

    def __remember_best_params(
        self, 
        best_encoder_weight_params_list, 
        best_encoder_bias_params_list,
        best_decoder_weight_params_list, 
        best_decoder_bias_params_list
    ):
        '''
        Remember best parameters.
        
        Args:
            best_encoder_weight_params_list:    `list` of weight parameters in encoder.
            best_encoder_bias_params_list:      `list` of bias parameters in encoder.
            best_decoder_weight_params_list:    `list` of weight parameters in decoder.
            best_decoder_bias_params_list:      `list` of bias parameters in decoder.

        '''
        if len(best_encoder_weight_params_list) and len(best_encoder_bias_params_list):
            for i in range(len(self.encoder.nn_layer_list)):
                self.encoder.nn_layer_list[i].graph.weight_arr = best_encoder_weight_params_list[i]
                self.encoder.nn_layer_list[i].graph.bias_arr = best_encoder_bias_params_list[i]
            self.__logger.debug("Encoder's best params are saved.")
        if len(best_decoder_weight_params_list) and len(best_decoder_bias_params_list):
            for i in range(len(self.decoder.nn_layer_list)):
                self.decoder.nn_layer_list[i].graph.weight_arr = best_decoder_weight_params_list[i]
                self.decoder.nn_layer_list[i].graph.bias_arr = best_decoder_bias_params_list[i]
            self.__logger.debug("Decoder's best params are saved.")

    def forward_propagation(self, np.ndarray[DOUBLE_t, ndim=2] observed_arr):
        '''
        Forward propagation in NN.
        
        Args:
            observed_arr:    `np.ndarray` of image file array.
        
        Returns:
            Propagated `np.ndarray`.
        '''
        self.encoder.weight_decay_term = 0.0
        self.decoder.weight_decay_term = 0.0

        self.__encoder_delta_arr_list = []
        self.__decoder_delta_arr_list = []
        self.__encoder_sigma_arr_list = []
        self.__decoder_sigma_arr_list = []
        self.__encoder_mu_arr_list = []
        self.__decoder_mu_arr_list = []

        cdef np.ndarray[DOUBLE_t, ndim=2] nosied_observed_arr = observed_arr + self.__params_initializer.sample(
            size=observed_arr.copy().shape,
            **self.__params_dict
        )
        cdef np.ndarray[DOUBLE_t, ndim=2] delta_arr
        cdef np.ndarray[DOUBLE_t, ndim=2] sigma_arr
        for i in range(len(self.encoder.nn_layer_list)):
            try:
                observed_arr = self.encoder.nn_layer_list[i].forward_propagate(observed_arr)
                nosied_observed_arr = self.encoder.nn_layer_list[i].forward_propagate(nosied_observed_arr)
                nosied_observed_arr = nosied_observed_arr + self.__params_initializer.sample(
                    size=nosied_observed_arr.copy().shape,
                    **self.__params_dict
                )
                delta_arr = self.computable_loss.compute_delta(
                    nosied_observed_arr,
                    observed_arr
                )
                self.__encoder_delta_arr_list.append(delta_arr)
                sigma_arr = np.dot(observed_arr, observed_arr.T)
                sigma_arr = sigma_arr / observed_arr.shape[1]
                self.__encoder_sigma_arr_list.append(sigma_arr)
                self.__encoder_mu_arr_list.append(observed_arr)
            except:
                self.__logger.debug("Error raised in NN layer " + str(i + 1))
                raise

            self.encoder.weight_decay_term += self.encoder.opt_params.compute_weight_decay(
                self.encoder.nn_layer_list[i].graph.weight_arr
            )

        if self.encoder.opt_params.dropout_rate > 0:
            observed_arr = self.encoder.opt_params.dropout(observed_arr)

        for i in range(len(self.decoder.nn_layer_list)):
            try:
                observed_arr = self.decoder.nn_layer_list[i].forward_propagate(observed_arr)
                nosied_observed_arr = self.decoder.nn_layer_list[i].forward_propagate(nosied_observed_arr)
                nosied_observed_arr = nosied_observed_arr + self.__params_initializer.sample(
                    size=nosied_observed_arr.copy().shape,
                    **self.__params_dict
                )
                delta_arr = self.computable_loss.compute_delta(
                    nosied_observed_arr,
                    observed_arr
                )
                self.__decoder_delta_arr_list.append(delta_arr)
                sigma_arr = np.dot(observed_arr, observed_arr.T)
                sigma_arr = sigma_arr / observed_arr.shape[1]
                self.__decoder_sigma_arr_list.append(sigma_arr)
                self.__decoder_mu_arr_list.append(observed_arr)
            except:
                self.__logger.debug("Error raised in NN layer " + str(i + 1))
                raise

            self.decoder.weight_decay_term += self.decoder.opt_params.compute_weight_decay(
                self.decoder.nn_layer_list[i].graph.weight_arr
            )

        if self.decoder.opt_params.dropout_rate > 0:
            observed_arr = self.decoder.opt_params.dropout(observed_arr)

        return observed_arr

    def back_propagation(self, np.ndarray delta_arr):
        '''
        Back propagation in NN.
        
        Args:
            Delta.
        
        Returns.
            Delta.
        '''
        nn_layer_list = self.decoder.nn_layer_list[::-1]
        hidden_delta_arr_list = self.__decoder_delta_arr_list[::-1]
        sigma_arr_list = self.__decoder_sigma_arr_list[::-1]
        mu_arr_list = self.__decoder_mu_arr_list[::-1]

        if self.decoder.opt_params.dropout_rate > 0:
            delta_arr = self.decoder.opt_params.de_dropout(delta_arr)

        cdef np.ndarray hidden_delta_arr
        cdef np.ndarray sigma_arr
        cdef np.ndarray mu_arr

        for i in range(len(nn_layer_list)):
            try:
                hidden_delta_arr = hidden_delta_arr_list[i]
                sigma_arr = sigma_arr_list[i]
                sigma_arr[sigma_arr == 0] += 1e-08
                sigma_arr = np.eye(sigma_arr.shape[0]) - np.power(sigma_arr, -1)
                sigma_arr = np.nanmean(sigma_arr, axis=0).reshape((sigma_arr.shape[0], 1))
                mu_arr = mu_arr_list[i]
                delta_arr = delta_arr + (self.__alpha_weight * hidden_delta_arr) + (self.__sigma_weight * sigma_arr) + (self.__mu_weight * mu_arr)
                delta_arr = nn_layer_list[i].back_propagate(delta_arr)
            except:
                self.__logger.debug(
                    "Delta computation raised an error in decoder's NN layer " + str(len(nn_layer_list) - i)
                )
                raise

        if self.encoder.opt_params.dropout_rate > 0:
            delta_arr = self.encoder.opt_params.de_dropout(delta_arr)

        nn_layer_list = self.encoder.nn_layer_list[::-1]
        hidden_delta_arr_list = self.__encoder_delta_arr_list[::-1]
        sigma_arr_list = self.__encoder_sigma_arr_list[::-1]
        mu_arr_list = self.__encoder_mu_arr_list[::-1]

        for i in range(len(nn_layer_list)):
            try:
                hidden_delta_arr = hidden_delta_arr_list[i]
                sigma_arr = sigma_arr_list[i]
                sigma_arr[sigma_arr == 0] += 1e-08
                sigma_arr = np.eye(sigma_arr.shape[0]) - np.power(sigma_arr, -1)
                sigma_arr = np.nanmean(sigma_arr, axis=0).reshape((sigma_arr.shape[0], 1))
                mu_arr = mu_arr_list[i]
                delta_arr = delta_arr + (self.__alpha_weight * hidden_delta_arr) + (self.__sigma_weight * sigma_arr) + (self.__mu_weight * mu_arr)
                delta_arr = nn_layer_list[i].back_propagate(delta_arr)
            except:
                self.__logger.debug(
                    "Delta computation raised an error in encoder's NN layer " + str(len(nn_layer_list) - i)
                )
                raise

        return delta_arr

    def compute_alpha_loss(self):
        '''
        Compute denoising loss weighted alpha.

        Returns:
            loss.
        '''
        loss = 0.0
        for arr in self.__encoder_delta_arr_list:
            loss = loss + np.nansum(np.nanmean(np.square(arr), axis=0))
        for arr in self.__decoder_delta_arr_list:
            loss = loss + np.nansum(np.nanmean(np.square(arr), axis=0))

        return loss * self.__alpha_weight

    def compute_sigma_loss(self):
        '''
        Compute sigma loss weighted sigma.

        Returns:
            loss.
        '''
        sigma_arr = self.__encoder_sigma_arr_list[0]
        sigma_arr = np.diag(sigma_arr - np.ma.log(sigma_arr) - np.eye(sigma_arr.copy().shape[0])).reshape(1, sigma_arr.shape[1])
        sigma = np.mean(np.nanmean(sigma_arr, axis=1))
        for i in range(1, len(self.__encoder_sigma_arr_list)):
            sigma_arr = self.__encoder_sigma_arr_list[i]
            sigma_arr = np.diag(sigma_arr - np.ma.log(sigma_arr) - np.eye(sigma_arr.copy().shape[0])).reshape(1, sigma_arr.shape[1])
            sigma = sigma + np.mean(np.nanmean(sigma_arr, axis=1))

        for i in range(len(self.__decoder_sigma_arr_list)):
            sigma_arr = self.__decoder_sigma_arr_list[i]
            sigma_arr = np.diag(sigma_arr - np.ma.log(sigma_arr) - np.eye(sigma_arr.copy().shape[0])).reshape(1, sigma_arr.shape[1])
            sigma += sigma + np.mean(np.nanmean(sigma_arr, axis=1))

        sigma = sigma / (len(self.__encoder_sigma_arr_list) + len(self.__decoder_sigma_arr_list))
        sigma = sigma * self.__sigma_weight
        return sigma

    def compute_mu_loss(self):
        '''
        Compute mu loss weighted mu.

        Returns:
            loss.
        '''
        mu = np.nanmean(np.square(self.__encoder_mu_arr_list[0]))
        for i in range(1, len(self.__encoder_mu_arr_list)):
            mu += np.nanmean(np.square(self.__encoder_mu_arr_list[i]))
        for i in range(len(self.__decoder_mu_arr_list)):
            mu += np.nanmean(np.square(self.__decoder_mu_arr_list[i]))

        mu = mu / (len(self.__encoder_mu_arr_list) + len(self.__decoder_mu_arr_list))
        mu = mu * self.__mu_weight
        return mu

    def set_readonly(self, value):
        ''' setter '''
        raise TypeError("This property must be read-only.")

    def get_alpha_loss_arr(self):
        ''' getter '''
        return self.__alpha_loss_arr

    alpha_loss_arr = property(get_alpha_loss_arr, set_readonly)

    def get_sigma_loss_arr(self):
        ''' getter '''
        return self.__sigma_loss_arr
    
    sigma_loss_arr = property(get_sigma_loss_arr, set_readonly)

    def get_mu_loss_arr(self):
        ''' getter '''
        return self.__mu_loss_arr
    
    mu_loss_arr = property(get_mu_loss_arr, set_readonly)
