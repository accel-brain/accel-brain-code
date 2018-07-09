# -*- coding: utf-8 -*-
from logging import getLogger
import numpy as np
cimport numpy as np
from pydbm.synapse_list import Synapse
from pydbm.rnn.optimization.interface.optimizable_loss import OptimizableLoss
from pydbm.rnn.interface.reconstructable_feature import ReconstructableFeature
from pydbm.rnn.verification.interface.verificatable_result import VerificatableResult
ctypedef np.float64_t DOUBLE_t


class LSTMModel(ReconstructableFeature):
    '''
    Long short term memory(LSTM) networks.
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

    # Loss function.
    __optimizable_loss = None
    
    # Verification function.
    __verificatable_result = None

    # The list of inferenced feature points.
    __feature_points_arr = None

    # The list of paramters to be differentiated.
    __learned_params_list = []

    # The list of tuple of losses data.
    __learned_result_list = []

    def __init__(
        self,
        graph,
        optimizable_loss,
        int epochs,
        int batch_size,
        double learning_rate,
        double learning_attenuate_rate,
        int attenuate_epoch,
        double test_size_rate=0.3,
        verificatable_result=None
    ):
        '''
        Init for building LSTM networks.

        Args:
            graph:                          is-a `Synapse`.
            optimizable_loss:               is-a `OptimizableLoss`.
            epochs:                         Epochs of Mini-batch.
            bath_size:                      Batch size of Mini-batch.
            learning_rate:                  Learning rate.
            learning_attenuate_rate:        Attenuate the `learning_rate` by a factor of this value every `attenuate_epoch`.
            attenuate_epoch:                Attenuate the `learning_rate` by a factor of `learning_attenuate_rate` every `attenuate_epoch`.
            loss_function:                  Loss function for training LSTM model.
            test_size_rate:                 Size of Test data set. If this value is `0`, the validation will not be executed.
            verificatable_result:           Verification function.
        '''
        self.graph = graph
        if isinstance(optimizable_loss, OptimizableLoss):
            self.__optimizable_loss = optimizable_loss
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

        self.__test_size_rate = test_size_rate

        logger = getLogger("pydbm")
        self.__logger = logger
        self.__logger.debug("pydbm.rnn.lstm_model is started. ")

    def learn(self, np.ndarray[DOUBLE_t, ndim=3] observed_arr, np.ndarray target_arr=np.array([])):
        '''
        Learn the observed data points.
        for vector representation of the input time-series.

        In Encoder-Decode scheme, usecase of this method may be pre-training with `__learned_params_dict`.

        Override.

        Args:
            observed_arr:    Array like or sparse matrix as the observed data ponts.
            target_arr:      Array like or sparse matrix as the target data points.
                             To learn as Auto-encoder, this value must be `None` or equivalent to `observed_arr`.

        '''
        self.__logger.debug("pydbm.rnn.lstm_model.learn is started. ")

        cdef int row_o = observed_arr.shape[0]
        cdef int row_t = target_arr.shape[0]

        cdef np.ndarray train_index
        cdef np.ndarray test_index
        cdef np.ndarray[DOUBLE_t, ndim=3] train_observed_arr
        cdef np.ndarray train_target_arr
        cdef np.ndarray[DOUBLE_t, ndim=3] test_observed_arr
        cdef np.ndarray test_target_arr

        cdef double moving_loss = 0.0
        cdef double learning_rate = self.__learning_rate

        cdef int epoch
        cdef np.ndarray rand_index
        cdef np.ndarray[DOUBLE_t, ndim=3] batch_observed_arr
        cdef np.ndarray batch_target_arr
        cdef np.ndarray[DOUBLE_t, ndim=2] hidden_activity_arr
        cdef np.ndarray[DOUBLE_t, ndim=2] rnn_activity_arr
        cdef np.ndarray input_arr
        cdef np.ndarray rnn_arr
        cdef np.ndarray output_arr
        cdef np.ndarray label_arr
        cdef int batch_index
        cdef np.ndarray[DOUBLE_t, ndim=2] time_series_X

        cdef double loss
        cdef np.ndarray test_output_arr
        cdef np.ndarray test_label_arr
        cdef double test_loss
        cdef double test_moving_loss

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

        for epoch in range(self.__epochs):
            self.__logger.debug("Epoch: " + str(epoch+1))

            if ((epoch + 1) % self.__attenuate_epoch == 0):
                learning_rate = learning_rate / self.__learning_attenuate_rate

            rand_index = np.random.choice(train_observed_arr.shape[0], size=self.__batch_size)
            batch_observed_arr = train_observed_arr[rand_index]
            batch_target_arr = train_target_arr[rand_index]

            hidden_activity_arr = self.graph.hidden_activity_arr
            rnn_activity_arr = self.graph.rnn_activity_arr

            input_arr = None
            rnn_arr = None
            hidden_arr = None
            output_arr = None
            label_arr = None
            for batch_index in range(batch_observed_arr.shape[0]):
                time_series_X = batch_observed_arr[batch_index]
                output_arr_list, _hidden_activity_arr, _rnn_activity_arr = self.rnn_learn(time_series_X, hidden_activity_arr, rnn_activity_arr)

                if input_arr is None:
                    input_arr = time_series_X[-1]
                else:
                    input_arr = np.vstack([input_arr, time_series_X[-1]])

                if hidden_arr is None:
                    hidden_arr = _hidden_activity_arr
                else:
                    hidden_arr = np.vstack([hidden_arr, _hidden_activity_arr])
                
                if rnn_arr is None:
                    rnn_arr = _rnn_activity_arr
                else:
                    rnn_arr = np.vstack([rnn_arr, _rnn_activity_arr])

                if output_arr is None:
                    output_arr = output_arr_list[-1]
                else:
                    output_arr = np.vstack([output_arr, output_arr_list[-1]])

                target_time_series_X = batch_target_arr[batch_index]
                
                if label_arr is None:
                    label_arr = target_time_series_X[-1]
                else:
                    label_arr = np.vstack([label_arr, target_time_series_X[-1]])

            loss = self.optimizable_loss.compute_loss(output_arr, label_arr)
            if epoch == 0:
                self.__logger.debug("Computing loss is end.")

            self.back_propagation(
                input_arr,
                rnn_arr,
                hidden_arr,
                output_arr,
                label_arr,
                learning_rate
            )

            if epoch == 0:
                self.__logger.debug("Optimization is end.")

            if self.__test_size_rate > 0:
                rand_index = np.random.choice(test_observed_arr.shape[0], size=self.__batch_size)
                batch_observed_arr = test_observed_arr[rand_index]
                batch_target_arr = test_target_arr[rand_index]

                test_output_arr = None
                test_label_arr = None
                for batch_index in range(batch_observed_arr.shape[0]):
                    time_series_X = batch_observed_arr[batch_index]
                    test_output_arr_list = self.inference(time_series_X)

                    if test_output_arr is None:
                        test_output_arr = test_output_arr_list[-1]
                    else:
                        test_output_arr = np.vstack([test_output_arr, test_output_arr_list[-1]])

                    target_time_series_X = batch_target_arr[batch_index]
                    if test_label_arr is None:
                        test_label_arr = target_time_series_X[-1]
                    else:
                        test_label_arr = np.vstack([test_label_arr, target_time_series_X[-1]])

                test_loss = self.optimizable_loss.compute_loss(test_output_arr, test_label_arr)

            # Keeping a moving average of the losses.
            if epoch == 0:
                moving_loss = np.mean(loss)
                if self.__test_size_rate > 0:
                    test_moving_loss = np.mean(test_loss)
            else:
                moving_loss = .99 * moving_loss + .01 * np.mean(loss)
                if self.__test_size_rate > 0:
                    test_moving_loss = .99 * test_moving_loss + .01 * np.mean(test_loss)

            if self.__test_size_rate > 0:
                self.__learned_result_list.append((epoch, moving_loss, test_moving_loss))
            else:
                self.__learned_result_list.append((epoch, moving_loss))

            if self.__verificatable_result is not None:
                if self.__test_size_rate > 0:
                    self.__verificatable_result.verificate(
                        train_pred_arr=output_arr,
                        train_label_arr=label_arr,
                        test_pred_arr=test_output_arr,
                        test_label_arr=test_label_arr
                    )

        self.__logger.debug("end. ")

    def inference(self, np.ndarray[DOUBLE_t, ndim=2] time_series_X_arr):
        '''
        Inference the feature points to reconstruct the time-series.

        Override.

        Args:
            time_series_X_arr:    Array like or sparse matrix as the observed data ponts.
        
        Returns:
            Array like or sparse matrix of reconstructed instances of time-series.
        '''
        result_list = [None] * time_series_X_arr.shape[0]

        cdef np.ndarray hidden_activity_arr = np.array([])
        cdef np.ndarray rnn_activity_arr = np.array([])

        cdef np.ndarray _hidden_activity_arr
        cdef np.ndarray _rnn_activity_arr

        output_arr_list, _hidden_activity_arr, _rnn_activity_arr = self.rnn_learn(
            time_series_X_arr,
            hidden_activity_arr,
            rnn_activity_arr
        )

        if self.__feature_points_arr is not None and self.__feature_points_arr.shape[0] > 0:
            self.__feature_points_arr = np.r_[self.__feature_points_arr, _hidden_activity_arr.T]
        else:
            self.__feature_points_arr = _hidden_activity_arr.T

        return output_arr_list

    def get_feature_points(self):
        '''
        Extract the activities in hidden layer and reset it, 
        considering this method will be called per one cycle in instances of time-series.

        Returns:
            The `list` of array like or sparse matrix of feature points or virtual visible observed data points.
        '''
        feature_points_arr = self.__feature_points_arr
        self.__feature_points_arr = np.array([])
        return feature_points_arr

    def rnn_learn(
        self,
        np.ndarray[DOUBLE_t, ndim=2] time_series_X,
        np.ndarray h=np.array([]),
        np.ndarray c=np.array([])
    ):
        '''
        Encoder model.

        Args:
            time_series_X:          A time-series $X = \{x^{(1)}, x^{(2)}, ..., x^{(L)}\}$ in observed data points.
            h:                      The initial state in hidden layer.
            c:                      The initial state in hidden layer.
        
        Returns:
            Tuple(
                `The list of inferenced data points`,
                `The final state in hidden layer`,
                `The parameter of hidden layer`
            )
        '''
        cdef int default_row_h = h.shape[0]
        cdef int default_row_c = c.shape[0]

        cdef int row_h = self.graph.hidden_activity_arr.shape[0]
        cdef int col_h = self.graph.hidden_activity_arr.shape[1]
        cdef int row_c = self.graph.rnn_activity_arr.shape[0]
        cdef int col_c = self.graph.rnn_activity_arr.shape[1]

        if default_row_h == 0:
            h = np.zeros((row_h, col_h))

        if default_row_c == 0:
            c = np.zeros((row_c, col_c))

        cdef np.ndarray[DOUBLE_t, ndim=2] g
        cdef np.ndarray[DOUBLE_t, ndim=2] i
        cdef np.ndarray[DOUBLE_t, ndim=2] f
        cdef np.ndarray[DOUBLE_t, ndim=2] o
        cdef np.ndarray[DOUBLE_t, ndim=2] yhat_linear
        cdef np.ndarray[DOUBLE_t, ndim=2] yhat

        output_arr_list = []
        for X in time_series_X:
            g = self.graph.observed_activating_function.activate(
                np.dot(X, self.graph.weights_xg_arr) + np.dot(h, self.graph.weights_hg_arr) + self.graph.input_bias_arr
            )
            i = self.graph.input_activating_function.activate(
                np.dot(X, self.graph.weights_xi_arr) + np.dot(h, self.graph.weights_hi_arr) + self.graph.hidden_bias_arr
            )
            f = self.graph.forget_activating_function.activate(
                np.dot(X, self.graph.weights_xf_arr) + np.dot(h, self.graph.weights_hf_arr) + self.graph.forget_bias_arr
            )
            o = self.graph.rnn_output_activating_function.activate(
                np.dot(X, self.graph.weights_xo_arr) + np.dot(h, self.graph.weights_ho_arr) + self.graph.rnn_output_bias_arr
            )
            c = f * c + i * g
            h = o * self.graph.hidden_activating_function.activate(c)
            yhat = self.graph.linear_activating_function.activate(
                np.dot(h, self.graph.weights_hy_arr) + self.graph.linear_bias_arr
            )
            output_arr_list.append(yhat)
        return (output_arr_list, h, c)

    def back_propagation(
        self,
        np.ndarray[DOUBLE_t, ndim=2] input_arr,
        #np.ndarray[DOUBLE_t, ndim=2] rnn_arr,
        rnn_arr,
        np.ndarray[DOUBLE_t, ndim=2] hidden_arr,
        np.ndarray[DOUBLE_t, ndim=2] output_arr,
        np.ndarray[DOUBLE_t, ndim=2] label_arr,
        double learning_rate
    ):
        #cdef np.ndarray[DOUBLE_t, ndim=2] delta_o_arr
        #cdef np.ndarray[DOUBLE_t, ndim=2] delta_hy_arr
        cdef np.ndarray[DOUBLE_t, ndim=2] delta_ho_arr
        cdef np.ndarray[DOUBLE_t, ndim=2] delta_hf_arr
        cdef np.ndarray[DOUBLE_t, ndim=2] delta_hi_arr
        cdef np.ndarray[DOUBLE_t, ndim=2] delta_hg_arr

        #cdef np.ndarray[DOUBLE_t, ndim=2] weights_hy_diff_arr

        cdef np.ndarray[DOUBLE_t, ndim=2] weights_ho_diff_arr
        cdef np.ndarray[DOUBLE_t, ndim=2] weights_hf_diff_arr
        cdef np.ndarray[DOUBLE_t, ndim=2] weights_hi_diff_arr
        cdef np.ndarray[DOUBLE_t, ndim=2] weights_hg_diff_arr

        delta_o_arr = (label_arr - output_arr) * self.graph.linear_activating_function.derivative(output_arr)
        delta_hy_arr = delta_o_arr.dot(
            self.graph.weights_hy_arr.T
        ) * self.graph.hidden_activating_function.derivative(hidden_arr)
        weights_hy_diff_arr = rnn_arr.T.dot(delta_o_arr)

        delta_ho_arr = delta_hy_arr.dot(
            self.graph.weights_ho_arr.T
        ) * self.graph.rnn_output_activating_function.derivative(rnn_arr)
        weights_ho_diff_arr = np.dot(input_arr, self.graph.weights_xo_arr).T.dot(delta_ho_arr)

        delta_hf_arr = delta_ho_arr.dot(
            self.graph.weights_hf_arr.T
        ) * self.graph.forget_activating_function.derivative(rnn_arr)
        weights_hf_diff_arr = np.dot(input_arr, self.graph.weights_xf_arr).T.dot(delta_hf_arr)

        delta_hi_arr = delta_hf_arr.dot(
            self.graph.weights_hi_arr.T
        ) * self.graph.input_activating_function.derivative(rnn_arr)
        weights_hi_diff_arr = np.dot(input_arr, self.graph.weights_xi_arr).T.dot(delta_hi_arr)

        delta_hg_arr = delta_hi_arr.dot(
            self.graph.weights_hg_arr.T
        ) * self.graph.observed_activating_function.derivative(rnn_arr)
        weights_hg_diff_arr = np.dot(input_arr, self.graph.weights_xg_arr).T.dot(delta_hg_arr)

        #self.graph.weights_hy_arr += weights_hy_diff_arr * learning_rate
        self.graph.weights_ho_arr += weights_ho_diff_arr * learning_rate
        self.graph.weights_hf_arr += weights_hf_diff_arr * learning_rate
        self.graph.weights_hi_arr += weights_hi_diff_arr * learning_rate
        self.graph.weights_hg_arr += weights_hg_diff_arr * learning_rate
    
    def set_readonly(self, value):
        raise TypeError("This property must be read-only.")

    def get_optimizable_loss(self):
        ''' getter '''
        return self.__optimizable_loss

    optimizable_loss = property(get_optimizable_loss, set_readonly)

    def get_learned_params_dict(self):
        return self.__learned_params_dict

    def set_learned_params_dict(self, value):
        self.__learned_params_dict = value

    learned_params_dict = property(get_learned_params_dict, set_learned_params_dict)

    def get_learned_result_list(self):
        return self.__learned_result_list

    learned_result_list = property(get_learned_result_list, set_readonly)

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
