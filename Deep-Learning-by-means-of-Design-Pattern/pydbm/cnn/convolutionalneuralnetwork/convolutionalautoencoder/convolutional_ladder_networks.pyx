# -*- coding: utf-8 -*-
from logging import getLogger
from copy import deepcopy
from pydbm.cnn.convolutionalneuralnetwork.convolutional_auto_encoder import ConvolutionalAutoEncoder
from pydbm.cnn.layerablecnn.convolution_layer import ConvolutionLayer
from pydbm.params_initializer import ParamsInitializer
from pydbm.cnn.feature_generator import FeatureGenerator
import numpy as np
cimport numpy as np
ctypedef np.float64_t DOUBLE_t


class ConvolutionalLadderNetworks(ConvolutionalAutoEncoder):
    '''
    Ladder Networks with a Stacked convolutional Auto-Encoder.

    References:
        - Bengio, Y., Lamblin, P., Popovici, D., & Larochelle, H. (2007). Greedy layer-wise training of deep networks. In Advances in neural information processing systems (pp. 153-160).
        - Dumoulin, V., & V,kisin, F. (2016). A guide to convolution arithmetic for deep learning. arXiv preprint arXiv:1603.07285.
        - Erhan, D., Bengio, Y., Courville, A., Manzagol, P. A., Vincent, P., & Bengio, S. (2010). Why does unsupervised pre-training help deep learning?. Journal of Machine Learning Research, 11(Feb), 625-660.
        - Erhan, D., Courville, A., & Bengio, Y. (2010). Understanding representations learned in deep architectures. Department dInformatique et Recherche Operationnelle, University of Montreal, QC, Canada, Tech. Rep, 1355, 1.
        - Goodfellow, I., Bengio, Y., & Courville, A. (2016). Deep learning (adaptive computation and machine learning series). Adaptive Computation and Machine Learning series, 800.
        - Masci, J., Meier, U., Cireşan, D., & Schmidhuber, J. (2011, June). Stacked convolutional auto-encoders for hierarchical feature extraction. In International Conference on Artificial Neural Networks (pp. 52-59). Springer, Berlin, Heidelberg.
        - Rasmus, A., Berglund, M., Honkala, M., Valpola, H., & Raiko, T. (2015). Semi-supervised learning with ladder networks. In Advances in neural information processing systems (pp. 3546-3554).
        - Valpola, H. (2015). From neural PCA to deep unsupervised learning. In Advances in Independent Component Analysis and Learning Machines (pp. 143-171). Academic Press.
    '''
    # Feature points.
    __feature_points_arr = None

    # Delta.
    __delta_deconvolved_bias_arr = None

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
        pre_learned_path_list=None,
        output_no_bias_flag=True,
        alpha_weight=1e-05,
        sigma_weight=0.7,
        mu_weight=0.7,
        params_initializer=ParamsInitializer(),
        params_dict={"loc": 0.0, "scale": 1.0}
    ):
        '''
        Init.
        
        Override.
        
        Args:
            layerable_cnn_list:             The `list` of `ConvolutionLayer`.
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
            pre_learned_path_list:          `list` of file path that stores pre-learned parameters.
            output_no_bias_flag:            Output with no bias or not.

            sigma_weight:                   Weight of sigma cost.
            mu_weight:                      Weight of mu cost.

            params_initializer:             is-a `ParamsInitializer`. This class will noise 
                                            observed data points and hidden units by using this 
                                            `params_initializer`.

            params_dict:                     `dict` of parameters other than `size` to be input to function `ParamsInitializer.sample_f`.

        '''
        super().__init__(
            layerable_cnn_list=layerable_cnn_list,
            computable_loss=computable_loss,
            opt_params=opt_params,
            verificatable_result=verificatable_result,
            epochs=epochs,
            batch_size=batch_size,
            learning_rate=learning_rate,
            learning_attenuate_rate=learning_attenuate_rate,
            attenuate_epoch=attenuate_epoch,
            test_size_rate=test_size_rate,
            tol=tol,
            tld=tld,
            save_flag=save_flag,
            pre_learned_path_list=pre_learned_path_list
        )
        self.__epochs = epochs
        self.__batch_size = batch_size
        self.opt_params = opt_params
        self.__deconv_opt_params = deepcopy(self.opt_params)

        self.__learning_rate = learning_rate
        self.__learning_attenuate_rate = learning_attenuate_rate
        self.__attenuate_epoch = attenuate_epoch

        self.__test_size_rate = test_size_rate
        self.__tol = tol
        self.__tld = tld

        self.__memory_tuple_list = []
        
        self.__save_flag = save_flag

        self.__output_no_bias_flag = output_no_bias_flag

        logger = getLogger("pydbm")
        self.__logger = logger
        self.__learn_mode = True

        self.__alpha_weight = alpha_weight
        self.__sigma_weight = sigma_weight
        self.__mu_weight = mu_weight
        self.__params_initializer = params_initializer
        self.__params_dict = params_dict

        self.__dropout_rate = opt_params.dropout_rate

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
            train_target_arr = observed_arr

        cdef double loss
        cdef double test_loss
        cdef np.ndarray[DOUBLE_t, ndim=4] pred_arr
        cdef np.ndarray[DOUBLE_t, ndim=4] test_pred_arr
        cdef np.ndarray[DOUBLE_t, ndim=4] delta_arr
        
        best_weight_params_list = []
        best_bias_params_list = []

        try:
            self.__memory_tuple_list = []
            loss_list = []
            alpha_list = []
            sigma_list = []
            mu_list = []
            min_loss = None
            eary_stop_flag = False
            for epoch in range(self.__epochs):
                self.opt_params.dropout_rate = self.__dropout_rate
                self.opt_params.inferencing_mode = False

                if ((epoch + 1) % self.__attenuate_epoch == 0):
                    learning_rate = learning_rate * self.__learning_attenuate_rate

                rand_index = np.random.choice(train_observed_arr.shape[0], size=self.__batch_size)
                batch_observed_arr = train_observed_arr[rand_index]
                batch_target_arr = train_target_arr[rand_index]

                try:
                    pred_arr = self.inference(batch_observed_arr)
                    ver_pred_arr = pred_arr.copy()
                    train_weight_decay = self.weight_decay_term
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
                        self.__remember_best_params(best_weight_params_list, best_bias_params_list)
                        # Re-try.
                        pred_arr = self.inference(batch_observed_arr)
                        ver_pred_arr = pred_arr.copy()
                        train_weight_decay = self.weight_decay_term
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
                        best_weight_params_list = []
                        best_bias_params_list = []

                        for i in range(len(self.layerable_cnn_list)):
                            best_weight_params_list.append(self.layerable_cnn_list[i].graph.weight_arr)
                            best_bias_params_list.append(self.layerable_cnn_list[i].graph.bias_arr)
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
                    self.opt_params.inferencing_mode = True
                    rand_index = np.random.choice(test_observed_arr.shape[0], size=self.__batch_size)
                    test_batch_observed_arr = test_observed_arr[rand_index]
                    test_batch_target_arr = test_target_arr[rand_index]

                    test_pred_arr = self.forward_propagation(
                        test_batch_observed_arr
                    )
                    test_weight_decay = self.weight_decay_term
                    test_alpha_loss = self.compute_alpha_loss()
                    test_sigma_loss = self.compute_sigma_loss()
                    test_mu_loss = self.compute_mu_loss()
                    test_loss = self.computable_loss.compute_loss(
                        test_pred_arr,
                        test_batch_target_arr
                    )
                    test_loss = test_loss + test_alpha_loss + test_sigma_loss + test_mu_loss + test_weight_decay

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

        self.__remember_best_params(best_weight_params_list, best_bias_params_list)

        self.__alpha_loss_arr = np.array(alpha_list)
        self.__sigma_loss_arr = np.array(sigma_list)
        self.__mu_loss_arr = np.array(mu_list)

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
        cdef np.ndarray[DOUBLE_t, ndim=4] pred_arr
        cdef np.ndarray[DOUBLE_t, ndim=4] test_pred_arr
        cdef np.ndarray[DOUBLE_t, ndim=4] delta_arr

        best_weight_params_list = []
        best_bias_params_list = []

        try:
            self.__memory_tuple_list = []
            loss_list = []
            alpha_list = []
            sigma_list = []
            mu_list = []

            min_loss = None
            eary_stop_flag = False
            epoch = 0
            for batch_observed_arr, batch_target_arr, test_batch_observed_arr, test_batch_target_arr in feature_generator.generate():
                epoch += 1
                self.opt_params.dropout_rate = self.__dropout_rate
                self.opt_params.inferencing_mode = False

                if ((epoch + 1) % self.__attenuate_epoch == 0):
                    learning_rate = learning_rate * self.__learning_attenuate_rate

                try:
                    pred_arr = self.inference(batch_observed_arr)
                    ver_pred_arr = pred_arr.copy()
                    train_weight_decay = self.weight_decay_term
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
                        self.__remember_best_params(best_weight_params_list, best_bias_params_list)
                        # Re-try.
                        pred_arr = self.inference(batch_observed_arr)
                        ver_pred_arr = pred_arr.copy()
                        train_weight_decay = self.weight_decay_term
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
                        best_weight_params_list = []
                        best_bias_params_list = []

                        for i in range(len(self.layerable_cnn_list)):
                            best_weight_params_list.append(self.layerable_cnn_list[i].graph.weight_arr)
                            best_bias_params_list.append(self.layerable_cnn_list[i].graph.bias_arr)
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
                    self.opt_params.inferencing_mode = True
                    test_pred_arr = self.forward_propagation(
                        test_batch_observed_arr
                    )
                    test_weight_decay = self.weight_decay_term
                    test_alpha_loss = self.compute_alpha_loss()
                    test_sigma_loss = self.compute_sigma_loss()
                    test_mu_loss = self.compute_mu_loss()
                    test_loss = self.computable_loss.compute_loss(
                        test_pred_arr,
                        test_batch_target_arr
                    )
                    test_loss = test_loss + test_alpha_loss + test_sigma_loss + test_mu_loss + test_weight_decay

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

        self.__remember_best_params(best_weight_params_list, best_bias_params_list)

        self.__alpha_loss_arr = np.array(alpha_list)
        self.__sigma_loss_arr = np.array(sigma_list)
        self.__mu_loss_arr = np.array(mu_list)

        self.__logger.debug("end. ")

    def __remember_best_params(self, best_weight_params_list, best_bias_params_list):
        '''
        Remember best parameters.
        
        Args:
            best_weight_params_list:    `list` of weight parameters.
            best_bias_params_list:      `list` of bias parameters.

        '''
        if len(best_weight_params_list) and len(best_bias_params_list):
            for i in range(len(self.layerable_cnn_list)):
                if self.layerable_cnn_list[i].graph.constant_flag is False:
                    self.layerable_cnn_list[i].graph.weight_arr = best_weight_params_list[i]
                    self.layerable_cnn_list[i].graph.bias_arr = best_bias_params_list[i]
            self.__logger.debug("Best params are saved.")

    def forward_propagation(self, np.ndarray[DOUBLE_t, ndim=4] img_arr):
        '''
        Forward propagation in Convolutional Auto-Encoder.
        
        Override.
        
        Args:
            img_arr:    `np.ndarray` of image file array.
        
        Returns:
            Propagated `np.ndarray`.
        '''
        self.__encoder_delta_arr_list = []
        self.__decoder_delta_arr_list = []
        self.__encoder_sigma_arr_list = []
        self.__decoder_sigma_arr_list = []
        self.__encoder_mu_arr_list = []
        self.__decoder_mu_arr_list = []

        cdef np.ndarray[DOUBLE_t, ndim=4] nosied_observed_arr = img_arr + self.__params_initializer.sample(
            size=img_arr.copy().shape,
            **self.__params_dict
        )
        cdef np.ndarray[DOUBLE_t, ndim=4] delta_arr
        cdef np.ndarray[DOUBLE_t, ndim=2] sigma_arr

        cdef np.ndarray[DOUBLE_t, ndim=2] hidden_activity_arr
        cdef int i = 0
        self.weight_decay_term = 0.0

        for i in range(len(self.layerable_cnn_list)):
            try:
                img_arr = self.layerable_cnn_list[i].convolve(img_arr)
                nosied_observed_arr = self.layerable_cnn_list[i].convolve(nosied_observed_arr)
                if self.layerable_cnn_list[i].graph.constant_flag is False:
                    img_arr = self.layerable_cnn_list[i].graph.activation_function.activate(img_arr)
                    nosied_observed_arr = self.layerable_cnn_list[i].graph.activation_function.activate(nosied_observed_arr)

                nosied_observed_arr = nosied_observed_arr + self.__params_initializer.sample(
                    size=nosied_observed_arr.copy().shape,
                    **self.__params_dict
                )
                delta_arr = self.computable_loss.compute_delta(
                    nosied_observed_arr,
                    img_arr
                )
                self.__encoder_delta_arr_list.append(delta_arr)
                hidden_activity_arr = img_arr.reshape((
                    img_arr.shape[0],
                    -1
                ))
                sigma_arr = np.dot(hidden_activity_arr, hidden_activity_arr.T)
                sigma_arr = sigma_arr / hidden_activity_arr.shape[1]
                self.__encoder_sigma_arr_list.append(sigma_arr)
                self.__encoder_mu_arr_list.append(img_arr)
            except:
                self.__logger.debug("Error raised in Convolution layer " + str(i + 1))
                raise
            
            if self.layerable_cnn_list[i].graph.constant_flag is False:
                self.weight_decay_term += self.opt_params.compute_weight_decay(
                    self.layerable_cnn_list[i].graph.weight_arr
                )

        self.__feature_points_arr = img_arr.copy()

        if self.opt_params.dropout_rate > 0:
            hidden_activity_arr = img_arr.reshape((img_arr.shape[0], -1))
            hidden_activity_arr = self.opt_params.dropout(hidden_activity_arr)
            img_arr = hidden_activity_arr.reshape((
                img_arr.shape[0],
                img_arr.shape[1],
                img_arr.shape[2],
                img_arr.shape[3]
            ))

        layerable_cnn_list = self.layerable_cnn_list[::-1]
        for i in range(len(layerable_cnn_list)):
            try:
                img_arr = layerable_cnn_list[i].graph.activation_function.backward(img_arr)
                img_arr = layerable_cnn_list[i].deconvolve(img_arr)
                nosied_observed_arr = layerable_cnn_list[i].graph.activation_function.backward(nosied_observed_arr)
                nosied_observed_arr = layerable_cnn_list[i].deconvolve(nosied_observed_arr)
                if layerable_cnn_list[i].graph.deconvolved_bias_arr is not None and self.__output_no_bias_flag is False:
                    img_arr += layerable_cnn_list[i].graph.deconvolved_bias_arr.reshape((
                        1,
                        img_arr.shape[1],
                        img_arr.shape[2],
                        img_arr.shape[3]
                    ))
                    nosied_observed_arr += layerable_cnn_list[i].graph.deconvolved_bias_arr.reshape((
                        1, 
                        nosied_observed_arr.shape[1],
                        nosied_observed_arr.shape[2],
                        nosied_observed_arr.shape[3]
                    ))
                img_arr = layerable_cnn_list[i].graph.deactivation_function.activate(img_arr)
                nosied_observed_arr = layerable_cnn_list[i].graph.deactivation_function.activate(nosied_observed_arr)

                nosied_observed_arr = nosied_observed_arr + self.__params_initializer.sample(
                    size=nosied_observed_arr.copy().shape,
                    **self.__params_dict
                )
                delta_arr = self.computable_loss.compute_delta(
                    nosied_observed_arr,
                    img_arr
                )
                self.__decoder_delta_arr_list.append(delta_arr)
                hidden_activity_arr = img_arr.reshape((
                    img_arr.shape[0],
                    -1
                ))
                sigma_arr = np.dot(hidden_activity_arr, hidden_activity_arr.T)
                sigma_arr = sigma_arr / hidden_activity_arr.shape[1]
                self.__decoder_sigma_arr_list.append(sigma_arr)
                self.__decoder_mu_arr_list.append(img_arr)

            except:
                self.__logger.debug("Error raised in Deconvolution layer " + str(i + 1))
                raise

        return img_arr

    def back_propagation(self, np.ndarray[DOUBLE_t, ndim=4] delta_arr):
        '''
        Back propagation in CNN.
        
        Override.
        
        Args:
            Delta.
        
        Returns.
            Delta.
        '''
        cdef int i = 0
        cdef int sample_n = delta_arr.shape[0]
        cdef int kernel_height
        cdef int kernel_width
        cdef int img_sample_n
        cdef int img_channel
        cdef int img_height
        cdef int img_width

        cdef np.ndarray[DOUBLE_t, ndim=2] _delta_arr
        cdef np.ndarray[DOUBLE_t, ndim=3] delta_bias_arr
        cdef np.ndarray[DOUBLE_t, ndim=2] reshaped_img_arr
        cdef np.ndarray[DOUBLE_t, ndim=2] delta_weight_arr
        cdef np.ndarray[DOUBLE_t, ndim=4] _delta_weight_arr

        cdef np.ndarray[DOUBLE_t, ndim=4] hidden_delta_arr
        cdef np.ndarray[DOUBLE_t, ndim=2] sigma_arr
        cdef np.ndarray[DOUBLE_t, ndim=4] _sigma_arr
        cdef np.ndarray[DOUBLE_t, ndim=4] mu_arr

        hidden_delta_arr_list = self.__decoder_delta_arr_list[::-1]
        sigma_arr_list = self.__decoder_sigma_arr_list[::-1]
        mu_arr_list = self.__decoder_mu_arr_list[::-1]

        for i in range(len(self.layerable_cnn_list)):
            try:
                hidden_delta_arr = hidden_delta_arr_list[i]
                sigma_arr = sigma_arr_list[i]
                sigma_arr[sigma_arr == 0] += 1e-08
                sigma_arr = np.eye(sigma_arr.shape[0]) - np.power(sigma_arr, -1)
                _sigma_arr = np.nanmean(sigma_arr, axis=0).reshape((
                    hidden_delta_arr.shape[0], 
                    1,
                    1,
                    1
                ))
                mu_arr = mu_arr_list[i]

                delta_arr = delta_arr + (self.__alpha_weight * hidden_delta_arr) + (self.__sigma_weight * _sigma_arr) + (self.__mu_weight * mu_arr)

                img_sample_n = delta_arr.shape[0]
                img_channel = delta_arr.shape[1]
                img_height = delta_arr.shape[2]
                img_width = delta_arr.shape[3]

                if self.layerable_cnn_list[i].graph.constant_flag is False:
                    kernel_height = self.layerable_cnn_list[i].graph.weight_arr.shape[2]
                    kernel_width = self.layerable_cnn_list[i].graph.weight_arr.shape[3]
                    reshaped_img_arr = self.layerable_cnn_list[i].affine_to_matrix(
                        delta_arr,
                        kernel_height, 
                        kernel_width, 
                        self.layerable_cnn_list[i].graph.stride, 
                        self.layerable_cnn_list[i].graph.pad
                    )
                    if self.__output_no_bias_flag is False:
                        delta_bias_arr = delta_arr.sum(axis=0)

                delta_arr = self.layerable_cnn_list[i].convolve(delta_arr, no_bias_flag=True)
                channel = delta_arr.shape[1]
                _delta_arr = delta_arr.reshape(-1, sample_n)
                if self.layerable_cnn_list[i].graph.constant_flag is False:
                    delta_weight_arr = np.dot(reshaped_img_arr.T, _delta_arr)
                    delta_weight_arr = delta_weight_arr.transpose(1, 0)
                    _delta_weight_arr = delta_weight_arr.reshape(
                        sample_n,
                        kernel_height,
                        kernel_width,
                        -1
                    )
                    _delta_weight_arr = _delta_weight_arr.transpose((0, 3, 1, 2))

                    if self.__output_no_bias_flag is False:
                        if self.layerable_cnn_list[i].graph.delta_deconvolved_bias_arr is None:
                            self.layerable_cnn_list[i].graph.delta_deconvolved_bias_arr = delta_bias_arr.reshape(1, -1)
                        else:
                            self.layerable_cnn_list[i].graph.delta_deconvolved_bias_arr += delta_bias_arr.reshape(1, -1)

                        if self.layerable_cnn_list[i].graph.deconvolved_bias_arr is None:
                            self.layerable_cnn_list[i].graph.deconvolved_bias_arr = np.zeros((
                                1, 
                                img_channel * img_height * img_width
                            ))

                    if self.layerable_cnn_list[i].delta_weight_arr is None:
                        self.layerable_cnn_list[i].delta_weight_arr = _delta_weight_arr
                    else:
                        self.layerable_cnn_list[i].delta_weight_arr += _delta_weight_arr

            except:
                self.__logger.debug("Backward raised error in Convolution layer " + str(i + 1))
                raise

        cdef np.ndarray[DOUBLE_t, ndim=2] hidden_activity_arr
        if self.opt_params.dropout_rate > 0:
            hidden_activity_arr = delta_arr.reshape((delta_arr.shape[0], -1))
            hidden_activity_arr = self.opt_params.de_dropout(hidden_activity_arr)
            delta_arr = hidden_activity_arr.reshape((
                delta_arr.shape[0],
                delta_arr.shape[1],
                delta_arr.shape[2],
                delta_arr.shape[3]
            ))

        hidden_delta_arr_list = self.__encoder_delta_arr_list[::-1]
        sigma_arr_list = self.__encoder_sigma_arr_list[::-1]
        mu_arr_list = self.__encoder_mu_arr_list[::-1]

        layerable_cnn_list = self.layerable_cnn_list[::-1]
        for i in range(len(layerable_cnn_list)):
            try:
                hidden_delta_arr = hidden_delta_arr_list[i]
                sigma_arr = sigma_arr_list[i]
                sigma_arr[sigma_arr == 0] += 1e-08
                sigma_arr = np.eye(sigma_arr.shape[0]) - np.power(sigma_arr, -1)
                _sigma_arr = np.nanmean(sigma_arr, axis=0).reshape((
                    hidden_delta_arr.shape[0], 
                    1,
                    1,
                    1
                ))
                mu_arr = mu_arr_list[i]

                delta_arr = delta_arr + (self.__alpha_weight * hidden_delta_arr) + (self.__sigma_weight * _sigma_arr) + (self.__mu_weight * mu_arr)

                delta_arr = layerable_cnn_list[i].back_propagate(delta_arr)
                delta_arr = layerable_cnn_list[i].graph.deactivation_function.forward(delta_arr)
            except:
                self.__logger.debug(
                    "Delta computation raised an error in CNN layer " + str(len(layerable_cnn_list) - i)
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
        params_list = [
            self.layerable_cnn_list[i].graph.deconvolved_bias_arr for i in range(len(self.layerable_cnn_list))
        ]
        grads_list = [
            self.layerable_cnn_list[i].graph.delta_deconvolved_bias_arr for i in range(len(self.layerable_cnn_list))
        ]
        params_list = self.__deconv_opt_params.optimize(
            params_list,
            grads_list,
            learning_rate
        )
        for i in range(len(self.layerable_cnn_list)):
            self.layerable_cnn_list[i].graph.deconvolved_bias_arr = params_list[i]
            self.layerable_cnn_list[i].graph.delta_deconvolved_bias_arr = None

        super().optimize(learning_rate, epoch)

    def extract_feature_points_arr(self):
        '''
        Extract feature points.

        Returns:
            `np.ndarray` of feature points in hidden layer
            which means the encoded data.
        '''
        return self.__feature_points_arr

    def compute_alpha_loss(self):
        '''
        Compute denoising loss weighted alpha.

        Returns:
            loss.
        '''
        loss = 0.0
        for arr in self.__encoder_delta_arr_list:
            loss = loss + np.nanmean(np.nanmean(np.square(arr), axis=0))
        for arr in self.__decoder_delta_arr_list:
            loss = loss + np.nanmean(np.nanmean(np.square(arr), axis=0))

        return loss * self.__alpha_weight

    def compute_sigma_loss(self):
        '''
        Compute sigma loss weighted sigma.

        Returns:
            loss.
        '''
        sigma_arr = self.__encoder_sigma_arr_list[0]
        sigma = np.mean(np.nanmean(sigma_arr, axis=1))
        for i in range(1, len(self.__encoder_sigma_arr_list)):
            sigma = sigma + np.mean(np.nanmean(self.__encoder_sigma_arr_list[i], axis=1))

        for i in range(len(self.__decoder_sigma_arr_list)):
            sigma += sigma + np.mean(np.nanmean(self.__decoder_sigma_arr_list[i], axis=1))

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
