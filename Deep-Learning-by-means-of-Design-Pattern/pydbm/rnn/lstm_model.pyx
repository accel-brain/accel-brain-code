# -*- coding: utf-8 -*-
from logging import getLogger
import numpy as np
cimport numpy as np
from pydbm.synapse_list import Synapse
from pydbm.rnn.interface.reconstructable_feature import ReconstructableFeature
from pydbm.rnn.verification.interface.verificatable_result import VerificatableResult
ctypedef np.float64_t DOUBLE_t


class LSTMModel(ReconstructableFeature):
    '''
    Long short term memory(LSTM) networks.
    
    Alpha version.
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

    # The list of inferenced feature points.
    __feature_points_arr = None

    # The list of paramters to be differentiated.
    __learned_params_list = []

    def __init__(
        self,
        graph,
        int epochs,
        int batch_size,
        double learning_rate,
        double learning_attenuate_rate,
        int attenuate_epoch,
        int bptt_tau=16,
        double test_size_rate=0.3,
        verificatable_result=None
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
            bptt_tau:                       Refereed maxinum step `t` in BPTT. If `0`, this class referes all past data in BPTT.
            test_size_rate:                 Size of Test data set. If this value is `0`, the validation will not be executed.
            verificatable_result:           Verification function.
        '''
        self.graph = graph
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

        self.__delta_weights_output_arr = None
        self.__delta_weights_hidden_arr = None
        self.__delta_weights_output_gate_arr = None
        self.__delta_weights_output_gate_arr = None
        self.__delta_weights_forget_gate_arr = None
        self.__delta_weights_input_gate_arr = None
        self.__delta_weights_given_arr = None

        self.__delta_output_bias_arr = None
        self.__delta_hidden_bias_arr = None
        self.__delta_output_gate_bias_arr = None
        self.__delta_forget_gate_bias_arr = None
        self.__delta_input_gate_bias_arr = None
        self.__delta_given_bias_arr = None

        logger = getLogger("pydbm")
        self.__logger = logger
        self.__logger.debug("pydbm.rnn.lstm_model is started. ")

    def learn(self, np.ndarray[DOUBLE_t, ndim=3] observed_arr, np.ndarray target_arr=np.array([])):
        '''
        Learn the observed data points
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

        cdef double learning_rate = self.__learning_rate

        cdef int epoch
        cdef np.ndarray rand_index
        cdef np.ndarray[DOUBLE_t, ndim=3] batch_observed_arr
        cdef np.ndarray batch_target_arr
        cdef np.ndarray[DOUBLE_t, ndim=1] hidden_activity_arr
        cdef np.ndarray[DOUBLE_t, ndim=1] rnn_activity_arr

        cdef np.ndarray[DOUBLE_t, ndim=1] _output_arr
        cdef np.ndarray[DOUBLE_t, ndim=1] _hidden_activity_arr
        cdef np.ndarray[DOUBLE_t, ndim=1] _rnn_activity_arr

        cdef np.ndarray input_arr
        cdef np.ndarray rnn_arr
        cdef np.ndarray output_arr
        cdef np.ndarray label_arr
        cdef int batch_index
        cdef np.ndarray[DOUBLE_t, ndim=2] time_series_X

        cdef np.ndarray test_output_arr
        cdef np.ndarray test_label_arr

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

        try:
            for epoch in range(self.__epochs):
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
                    _output_arr, _hidden_activity_arr, _rnn_activity_arr = self.rnn_learn(time_series_X, hidden_activity_arr, rnn_activity_arr)

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
                        output_arr = _output_arr
                    else:
                        output_arr = np.vstack([output_arr, _output_arr])

                    target_time_series_X = batch_target_arr[batch_index]

                    if label_arr is None:
                        label_arr = target_time_series_X[-1]
                    else:
                        label_arr = np.vstack([label_arr, target_time_series_X[-1]])

                    self.back_propagation(target_time_series_X[-1])

                self.update(learning_rate)

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
                        _test_output_arr = self.inference(time_series_X)

                        if test_output_arr is None:
                            test_output_arr = _test_output_arr
                        else:
                            test_output_arr = np.vstack([test_output_arr, _test_output_arr])

                        target_time_series_X = batch_target_arr[batch_index]
                        if test_label_arr is None:
                            test_label_arr = target_time_series_X[-1]
                        else:
                            test_label_arr = np.vstack([test_label_arr, target_time_series_X[-1]])

                if self.__verificatable_result is not None:
                    if self.__test_size_rate > 0:
                        self.__verificatable_result.verificate(
                            train_pred_arr=output_arr,
                            train_label_arr=label_arr,
                            test_pred_arr=test_output_arr,
                            test_label_arr=test_label_arr
                        )

        except KeyboardInterrupt:
            self.__logger.debug("Interrupt.")

        self.__logger.debug("end. ")

    def inference(self, np.ndarray[DOUBLE_t, ndim=2] time_series_arr):
        '''
        Inference the feature points to reconstruct the time-series.

        Override.

        Args:
            time_series_arr:    Array like or sparse matrix as the observed data ponts.
        
        Returns:
            Array like or sparse matrix of reconstructed instances of time-series.
        '''
        result_list = [None] * time_series_arr.shape[0]

        cdef np.ndarray hidden_activity_arr = np.array([])
        cdef np.ndarray rnn_activity_arr = np.array([])

        cdef np.ndarray _hidden_activity_arr
        cdef np.ndarray _rnn_activity_arr

        output_arr, _hidden_activity_arr, _rnn_activity_arr = self.rnn_learn(
            time_series_arr,
            hidden_activity_arr,
            rnn_activity_arr
        )

        if self.__feature_points_arr is not None and self.__feature_points_arr.shape[0] > 0:
            self.__feature_points_arr = np.r_[self.__feature_points_arr, _hidden_activity_arr.T]
        else:
            self.__feature_points_arr = _hidden_activity_arr.T

        return output_arr

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
        np.ndarray[DOUBLE_t, ndim=2] time_series_arr,
        np.ndarray hidden_activity_arr=np.array([]),
        np.ndarray rnn_activity_arr=np.array([])
    ):
        '''
        Learn in RNN layer.

        Args:
            time_series_arr:        A time-series $X = \{x^{(1)}, x^{(2)}, ..., x^{(L)}\}$ in observed data points.
            hidden_activity_arr:    The initial state in hidden layer.
            rnn_activity_arr:       The initial state in hidden layer.
        
        Returns:
            Tuple(
                `Inferenced data points`,
                `The final state in hidden layer`,
                `The parameter of hidden layer`
            )
        '''
        cdef int default_row_h = 0
        if hidden_activity_arr is not None:
            default_row_h = hidden_activity_arr.shape[0]
        cdef int default_row_r = 0
        if rnn_activity_arr is not None:
            default_row_r = rnn_activity_arr.shape[0]

        cdef int row_h = self.graph.hidden_activity_arr.shape[0]
        cdef int row_r = self.graph.rnn_activity_arr.shape[0]

        if default_row_h == 0:
            hidden_activity_arr = np.zeros((row_h, ))

        if default_row_r == 0:
            rnn_activity_arr = np.zeros((row_r, ))
        
        cdef np.ndarray[DOUBLE_t, ndim=1] given_activity_arr
        cdef np.ndarray[DOUBLE_t, ndim=1] input_gate_arr
        cdef np.ndarray[DOUBLE_t, ndim=1] forget_gate_arr
        cdef np.ndarray[DOUBLE_t, ndim=1] output_gate_arr
        cdef np.ndarray[DOUBLE_t, ndim=1] output_arr

        self.__observed_arr_list = []
        self.__given_activity_arr_list = []
        self.__input_gate_arr_list = []
        self.__forget_gate_arr_list = []
        self.__output_gate_arr_list = []
        self.__rnn_activity_arr_list = []
        self.__output_arr_list = []
        self.__hidden_activity_arr_list = []

        for arr in time_series_arr:
            self.__observed_arr_list.append(arr.copy())

            given_activity_arr = self.graph.observed_activating_function.activate(
                np.dot(arr, self.graph.weights_given_arr) + self.graph.given_bias_arr
            )
            self.__given_activity_arr_list.append(given_activity_arr)
            input_gate_arr = self.graph.input_gate_activating_function.activate(
                np.dot(arr, self.graph.weights_input_gate_arr) + self.graph.input_gate_bias_arr
            )
            self.__input_gate_arr_list.append(input_gate_arr)
            forget_gate_arr = self.graph.forget_gate_activating_function.activate(
                np.dot(arr, self.graph.weights_forget_gate_arr) + self.graph.forget_gate_bias_arr
            )
            self.__forget_gate_arr_list.append(forget_gate_arr)
            output_gate_arr = self.graph.output_gate_activating_function.activate(
                np.dot(arr, self.graph.weights_output_gate_arr) + self.graph.output_gate_bias_arr
            )
            self.__output_gate_arr_list.append(output_gate_arr)
            rnn_activity_arr = given_activity_arr * input_gate_arr + rnn_activity_arr * forget_gate_arr
            self.__rnn_activity_arr_list.append(rnn_activity_arr)
            hidden_activity_arr = self.graph.hidden_activating_function.activate(
                output_gate_arr * (np.dot(rnn_activity_arr, self.graph.weights_hidden_arr) + self.graph.hidden_bias_arr)
            )
            self.__hidden_activity_arr_list.append(hidden_activity_arr)
            output_arr = self.graph.output_activating_function.activate(
                np.dot(hidden_activity_arr, self.graph.weights_output_arr) + self.graph.output_bias_arr
            )
            self.__output_arr_list.append(output_arr)

        return (output_arr, hidden_activity_arr, rnn_activity_arr)

    def back_propagation(self, np.ndarray[DOUBLE_t, ndim=1] label_arr):
        '''
        Simple back propagation.
        
        Args:
            label_arr:             Labeled data.

        '''
        cdef np.ndarray[DOUBLE_t, ndim=2] delta_weights_output_arr
        cdef np.ndarray[DOUBLE_t, ndim=2] delta_weights_hidden_arr
        cdef np.ndarray[DOUBLE_t, ndim=2] delta_weights_output_gate_arr
        cdef np.ndarray[DOUBLE_t, ndim=2] delta_weights_forget_gate_arr
        cdef np.ndarray[DOUBLE_t, ndim=2] delta_weights_input_gate_arr
        cdef np.ndarray[DOUBLE_t, ndim=2] delta_weights_given_arr
        
        cdef np.ndarray[DOUBLE_t, ndim=1] delta_output_bias_arr
        cdef np.ndarray[DOUBLE_t, ndim=1] delta_hidden_bias_arr
        cdef np.ndarray[DOUBLE_t, ndim=1] delta_output_gate_bias_arr
        cdef np.ndarray[DOUBLE_t, ndim=1] delta_forget_gate_bias_arr
        cdef np.ndarray[DOUBLE_t, ndim=1] delta_input_gate_bias_arr
        cdef np.ndarray[DOUBLE_t, ndim=1] delta_given_bias_arr


        self.__output_arr_list = self.__output_arr_list[::-1]
        self.__hidden_activity_arr_list = self.__hidden_activity_arr_list[::-1]
        self.__rnn_activity_arr_list = self.__rnn_activity_arr_list[::-1]
        self.__output_gate_arr_list = self.__output_gate_arr_list[::-1]
        self.__forget_gate_arr_list = self.__forget_gate_arr_list[::-1]
        self.__input_gate_arr_list = self.__input_gate_arr_list[::-1]
        self.__given_activity_arr_list = self.__given_activity_arr_list[::-1]
        
        cdef np.ndarray[DOUBLE_t, ndim=1] delta_output_arr = (self.__output_arr_list[0] - label_arr) * self.graph.output_activating_function.derivative(
            self.__output_arr_list[0]
        )
        cdef np.ndarray[DOUBLE_t, ndim=1] delta_hidden_arr = self.graph.hidden_activating_function.derivative(
            self.__hidden_activity_arr_list[0]
        ) * np.dot(delta_output_arr, self.graph.weights_output_arr.T)

        cdef np.ndarray[DOUBLE_t, ndim=1] delta_output_gate_arr = self.graph.output_gate_activating_function.derivative(
            self.__output_gate_arr_list[0]
        ) * np.dot(delta_hidden_arr, self.graph.weights_hidden_arr.T)

        cdef np.ndarray[DOUBLE_t, ndim=1] delta_forget_gate_arr = self.graph.forget_gate_activating_function.derivative(
            self.__forget_gate_arr_list[0]
        ) * np.dot(delta_hidden_arr, self.graph.weights_hidden_arr.T)

        cdef np.ndarray[DOUBLE_t, ndim=1] delta_input_gate_arr = self.graph.input_gate_activating_function.derivative(
            self.__input_gate_arr_list[0]
        ) * np.dot(delta_hidden_arr, self.graph.weights_hidden_arr.T)

        cdef np.ndarray[DOUBLE_t, ndim=1] delta_given_arr = self.graph.observed_activating_function.derivative(
            self.__given_activity_arr_list[0]
        ) * np.dot(delta_hidden_arr, self.graph.weights_hidden_arr.T)

        delta_weights_output_arr = delta_output_arr * self.__hidden_activity_arr_list[0].reshape(-1, 1)
        delta_weights_hidden_arr = delta_hidden_arr * self.__rnn_activity_arr_list[0].reshape(-1, 1)
        delta_weights_output_gate_arr = delta_output_gate_arr * self.__observed_arr_list[0].reshape(-1, 1)
        delta_weights_forget_gate_arr = delta_forget_gate_arr * self.__observed_arr_list[0].reshape(-1, 1)
        delta_weights_input_gate_arr = delta_input_gate_arr * self.__observed_arr_list[0].reshape(-1, 1)
        delta_weights_given_arr = delta_given_arr * self.__observed_arr_list[0].reshape(-1, 1)
        
        delta_output_bias_arr = delta_output_arr
        delta_hidden_bias_arr = delta_hidden_arr
        delta_output_gate_bias_arr = delta_output_gate_arr
        delta_forget_gate_bias_arr = delta_forget_gate_arr
        delta_input_gate_bias_arr = delta_input_gate_arr
        delta_given_bias_arr = delta_given_arr

        if self.__delta_weights_output_arr is None:
            self.__delta_weights_output_arr = np.zeros((delta_weights_output_arr.shape[0], delta_weights_output_arr.shape[1]))
        if self.__delta_weights_hidden_arr is None:
            self.__delta_weights_hidden_arr = np.zeros((delta_weights_hidden_arr.shape[0], delta_weights_hidden_arr.shape[1]))

        if self.__delta_weights_output_gate_arr is None:
            self.__delta_weights_output_gate_arr = np.zeros((delta_weights_output_gate_arr.shape[0], delta_weights_output_gate_arr.shape[1]))

        if self.__delta_weights_forget_gate_arr is None:
            self.__delta_weights_forget_gate_arr = np.zeros((delta_weights_forget_gate_arr.shape[0], delta_weights_forget_gate_arr.shape[1]))
        if self.__delta_weights_input_gate_arr is None:
            self.__delta_weights_input_gate_arr = np.zeros((delta_weights_input_gate_arr.shape[0], delta_weights_input_gate_arr.shape[1]))
        if self.__delta_weights_given_arr is None:
            self.__delta_weights_given_arr = np.zeros((delta_weights_given_arr.shape[0], delta_weights_given_arr.shape[1]))

        if self.__delta_output_bias_arr is None:
            self.__delta_output_bias_arr = np.zeros(delta_output_arr.shape[0])
        if self.__delta_hidden_bias_arr is None:
            self.__delta_hidden_bias_arr = np.zeros(delta_hidden_arr.shape[0])
        if self.__delta_output_gate_bias_arr is None:
            self.__delta_output_gate_bias_arr = np.zeros(delta_output_gate_arr.shape[0])
        if self.__delta_forget_gate_bias_arr is None:
            self.__delta_forget_gate_bias_arr = np.zeros(delta_forget_gate_arr.shape[0])
        if self.__delta_input_gate_bias_arr is None:
            self.__delta_input_gate_bias_arr = np.zeros(delta_input_gate_arr.shape[0])
        if self.__delta_given_bias_arr is None:
            self.__delta_given_bias_arr = np.zeros(delta_given_arr.shape[0])

        if self.__bptt_tau == 0:
            tau = self.__output_arr_list.shape[0]
        else:
            tau = self.__bptt_tau

        for i in range(1, len(self.__output_arr_list[:tau])):
            delta_hidden_arr = self.__hidden_activity_arr_list[i-1] * np.dot(
                self.graph.hidden_activating_function.derivative(self.__hidden_activity_arr_list[i]),
                self.graph.weights_hidden_arr
            )
            delta_output_gate_arr = self.graph.output_gate_activating_function.derivative(
                self.__output_gate_arr_list[i]
            ) * np.dot(delta_hidden_arr, self.graph.weights_hidden_arr.T)

            delta_forget_gate_arr = self.graph.forget_gate_activating_function.derivative(
                self.__forget_gate_arr_list[i]
            ) * np.dot(delta_hidden_arr, self.graph.weights_hidden_arr.T)

            delta_input_gate_arr = self.graph.input_gate_activating_function.derivative(
                self.__input_gate_arr_list[i]
            ) * np.dot(delta_hidden_arr, self.graph.weights_hidden_arr.T)

            delta_given_arr = self.graph.observed_activating_function.derivative(
                self.__given_activity_arr_list[i]
            ) * np.dot(delta_hidden_arr, self.graph.weights_hidden_arr.T)

            delta_weights_output_arr += delta_output_arr * self.__hidden_activity_arr_list[i].reshape(-1, 1)
            delta_weights_hidden_arr += delta_hidden_arr * self.__rnn_activity_arr_list[i].reshape(-1, 1)
            delta_weights_output_gate_arr += delta_output_gate_arr * self.__observed_arr_list[i].reshape(-1, 1)
            delta_weights_forget_gate_arr += delta_forget_gate_arr * self.__observed_arr_list[i].reshape(-1, 1)
            delta_weights_input_gate_arr += delta_input_gate_arr * self.__observed_arr_list[i].reshape(-1, 1)
            delta_weights_given_arr += delta_given_arr * self.__observed_arr_list[i].reshape(-1, 1)

            delta_output_bias_arr += delta_output_arr
            delta_hidden_bias_arr += delta_hidden_arr
            delta_output_gate_bias_arr += delta_output_gate_arr
            delta_forget_gate_bias_arr += delta_forget_gate_arr
            delta_input_gate_bias_arr += delta_input_gate_arr
            delta_given_bias_arr += delta_given_arr

        self.__delta_weights_output_arr += delta_weights_output_arr
        self.__delta_weights_hidden_arr += delta_weights_hidden_arr
        self.__delta_weights_output_gate_arr += delta_weights_output_gate_arr
        self.__delta_weights_forget_gate_arr += delta_weights_forget_gate_arr
        self.__delta_weights_input_gate_arr += delta_weights_input_gate_arr
        self.__delta_weights_given_arr += delta_weights_given_arr

        self.__delta_output_bias_arr += delta_output_bias_arr
        self.__delta_hidden_bias_arr += delta_hidden_bias_arr
        self.__delta_output_gate_bias_arr += delta_output_gate_bias_arr
        self.__delta_forget_gate_bias_arr += delta_forget_gate_bias_arr
        self.__delta_input_gate_bias_arr += delta_input_gate_bias_arr
        self.__delta_given_bias_arr += delta_given_bias_arr

    def update(self, double learning_rate):
        ''' Init. '''

        self.graph.weights_output_arr -= learning_rate * self.__delta_weights_output_arr / self.__batch_size
        self.graph.weights_hidden_arr -= learning_rate * self.__delta_weights_hidden_arr / self.__batch_size
        self.graph.weights_output_gate_arr -= learning_rate * self.__delta_weights_output_gate_arr / self.__batch_size
        self.graph.weights_forget_gate_arr -= learning_rate * self.__delta_weights_forget_gate_arr / self.__batch_size
        self.graph.weights_input_gate_arr -= learning_rate * self.__delta_weights_input_gate_arr / self.__batch_size
        self.graph.weights_given_arr -= learning_rate * self.__delta_weights_given_arr / self.__batch_size
        
        self.graph.output_bias_arr -= learning_rate * self.__delta_output_bias_arr / self.__batch_size
        self.graph.hidden_bias_arr -= learning_rate * self.__delta_hidden_bias_arr / self.__batch_size
        self.graph.output_gate_bias_arr -= learning_rate * self.__delta_output_gate_bias_arr / self.__batch_size
        self.graph.forget_gate_bias_arr -= learning_rate * self.__delta_forget_gate_bias_arr / self.__batch_size
        self.graph.input_gate_bias_arr -= learning_rate * self.__delta_input_gate_bias_arr / self.__batch_size
        self.graph.given_bias_arr -= learning_rate * self.__delta_given_bias_arr / self.__batch_size

        self.__delta_weights_output_arr = None
        self.__delta_weights_hidden_arr = None
        self.__delta_weights_output_gate_arr = None
        self.__delta_weights_forget_gate_arr = None
        self.__delta_weights_input_gate_arr = None
        self.__delta_weights_given_arr = None

        self.__delta_output_bias_arr = None
        self.__delta_hidden_bias_arr = None
        self.__delta_output_gate_bias_arr = None
        self.__delta_forget_gate_bias_arr = None
        self.__delta_input_gate_bias_arr = None
        self.__delta_given_bias_arr = None

    def set_readonly(self, value):
        raise TypeError("This property must be read-only.")

    def get_learned_params_dict(self):
        return self.__learned_params_dict

    def set_learned_params_dict(self, value):
        self.__learned_params_dict = value

    learned_params_dict = property(get_learned_params_dict, set_learned_params_dict)

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
