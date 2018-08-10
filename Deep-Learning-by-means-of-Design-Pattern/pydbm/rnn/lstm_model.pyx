# -*- coding: utf-8 -*-
from logging import getLogger
import numpy as np
cimport numpy as np
from pydbm.synapse_list import Synapse
from pydbm.rnn.verification.interface.verificatable_result import VerificatableResult
from pydbm.rnn.loss.interface.computable_loss import ComputableLoss
from pydbm.rnn.optimization.opt_params import OptParams
from pydbm.rnn.interface.reconstructable_model import ReconstructableModel
ctypedef np.float64_t DOUBLE_t


class LSTMModel(ReconstructableModel):
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
    
    # Verification function.
    __verificatable_result = None

    # The list of paramters to be differentiated.
    __learned_params_list = []
    
    # Latest loss
    __latest_loss = None

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
        computable_loss=None,
        opt_params=None,
        verificatable_result=None,
        tol=1e-04
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
        '''
        self.graph = graph

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

        self.__memory_tuple_list = []

        logger = getLogger("pydbm")
        self.__logger = logger

    def learn(self, np.ndarray[DOUBLE_t, ndim=3] observed_arr, np.ndarray target_arr=np.array([])):
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
        cdef np.ndarray[DOUBLE_t, ndim=3] train_observed_arr
        cdef np.ndarray train_target_arr
        cdef np.ndarray[DOUBLE_t, ndim=3] test_observed_arr
        cdef np.ndarray test_target_arr

        cdef np.ndarray rand_index
        cdef np.ndarray[DOUBLE_t, ndim=3] batch_observed_arr
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

        try:
            self.__memory_tuple_list = []
            loss_list = []
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
                    loss = self.__computable_loss.compute_loss(
                        pred_arr,
                        batch_target_arr[:, -1, :]
                    )
                    delta_arr = self.__computable_loss.compute_delta(
                        pred_arr,
                        batch_target_arr[:, -1, :]
                    )
                    delta_arr, output_grads_list = self.output_back_propagate(pred_arr, delta_arr)
                    _delta_arr, lstm_grads_list = self.hidden_back_propagate(delta_arr)
                    grads_list = output_grads_list
                    grads_list.extend(lstm_grads_list)
                    self.optimize(grads_list, learning_rate, epoch)
                    self.graph.hidden_activity_arr = np.array([])
                    self.graph.rnn_activity_arr = np.array([])

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
                    if self.__verificatable_result is not None:
                        if self.__test_size_rate > 0:
                            self.__verificatable_result.verificate(
                                self.__computable_loss,
                                train_pred_arr=pred_arr, 
                                train_label_arr=batch_target_arr[:, -1, :],
                                test_pred_arr=test_pred_arr,
                                test_label_arr=test_batch_target_arr[:, -1, :]
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

        self.__logger.debug("end. ")

    def forward_propagation(self, np.ndarray[DOUBLE_t, ndim=3] batch_observed_arr):
        '''
        Forward propagation.
        
        Args:
            batch_observed_arr:    Array like or sparse matrix as the observed data points.
        
        Returns:
            Array like or sparse matrix as the predicted data points.
        '''
        cdef np.ndarray[DOUBLE_t, ndim=3] hidden_activity_arr = self.hidden_forward_propagate(
            batch_observed_arr
        )
        self.graph.hidden_activity_arr = hidden_activity_arr
        cdef np.ndarray[DOUBLE_t, ndim=2] pred_arr = self.output_forward_propagate(
            hidden_activity_arr
        )
        return pred_arr

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
        params_list = self.__opt_params.optimize(
            [
                self.graph.weights_output_arr,
                self.graph.output_bias_arr,
                self.graph.weights_lstm_hidden_arr,
                self.graph.weights_lstm_observed_arr,
                self.graph.lstm_bias_arr
            ],
            grads_list,
            learning_rate
        )
        self.graph.weights_output_arr = params_list[0]
        self.graph.output_bias_arr = params_list[1]
        self.graph.weights_lstm_hidden_arr = params_list[2]
        self.graph.weights_lstm_observed_arr = params_list[3]
        self.graph.lstm_bias_arr = params_list[4]

        if ((epoch + 1) % self.__attenuate_epoch == 0):
            self.graph.weights_output_arr = self.__opt_params.constrain_weight(self.graph.weights_output_arr)
            self.graph.weights_lstm_hidden_arr = self.__opt_params.constrain_weight(self.graph.weights_lstm_hidden_arr)
            self.graph.weights_lstm_observed_arr = self.__opt_params.constrain_weight(self.graph.weights_lstm_observed_arr)

    def inference(
        self,
        np.ndarray[DOUBLE_t, ndim=3] observed_arr,
        np.ndarray[DOUBLE_t, ndim=2] hidden_activity_arr=None,
        np.ndarray[DOUBLE_t, ndim=2] rnn_activity_arr=None
    ):
        '''
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
        cdef int feature_n = observed_arr.shape[2]
        cdef int hidden_n = self.graph.weights_lstm_hidden_arr.shape[0]

        if hidden_activity_arr is None:
            self.graph.hidden_activity_arr = np.zeros((sample_n, hidden_n), dtype=np.float64)
        else:
            self.graph.hidden_activity_arr = hidden_activity_arr

        if rnn_activity_arr is None:
            self.graph.rnn_activity_arr = np.zeros((sample_n, hidden_n), dtype=np.float64)
        else:
            self.graph.rnn_activity_arr = rnn_activity_arr

        self.__opt_params.dropout_rate = 0.0
        cdef np.ndarray[DOUBLE_t, ndim=2] pred_arr = self.forward_propagation(observed_arr)
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

    def hidden_forward_propagate(self, np.ndarray[DOUBLE_t, ndim=3] observed_arr):
        '''
        Forward propagation in LSTM gate.
        
        Args:
            observed_arr:    `np.ndarray` of observed data points.
        
        Returns:
            Predicted data points.
        '''
        cdef int sample_n = observed_arr.shape[0]
        cdef int cycle_len = observed_arr.shape[1]
        cdef int feature_n = observed_arr.shape[2]
        cdef int hidden_n = self.graph.weights_lstm_hidden_arr.shape[0]

        cdef np.ndarray[DOUBLE_t, ndim=3] pred_arr = np.zeros((sample_n, cycle_len, hidden_n), dtype=np.float64)

        if self.graph.hidden_activity_arr is None or self.graph.hidden_activity_arr.shape[0] == 0:
            self.graph.hidden_activity_arr = np.zeros((sample_n, hidden_n), dtype=np.float64)

        if self.graph.rnn_activity_arr is None or self.graph.rnn_activity_arr.shape[0] == 0:
            self.graph.rnn_activity_arr = np.zeros((sample_n, hidden_n), dtype=np.float64)

        cdef int cycle
        for cycle in range(cycle_len):
            if self.graph.hidden_activity_arr.ndim == 2:
                self.graph.hidden_activity_arr, self.graph.rnn_activity_arr = self.__lstm_forward(
                    observed_arr[:, cycle, :],
                    self.graph.hidden_activity_arr,
                    self.graph.rnn_activity_arr
                )
            elif self.graph.hidden_activity_arr.ndim == 3:
                self.graph.hidden_activity_arr, self.graph.rnn_activity_arr = self.__lstm_forward(
                    observed_arr[:, cycle, :],
                    self.graph.hidden_activity_arr[:, cycle, :],
                    self.graph.rnn_activity_arr
                )
            else:
                raise ValueError("The shape of hidden activity array is invalid.")
            pred_arr[:, cycle, :] = self.graph.hidden_activity_arr

        return pred_arr

    def output_forward_propagate(self, np.ndarray[DOUBLE_t, ndim=3] pred_arr):
        '''
        Forward propagation in output layer.
        
        Args:
            pred_arr:            `np.ndarray` of predicted data points.

        Returns:
            `np.ndarray` of propagated data points.
        '''
        cdef np.ndarray[DOUBLE_t, ndim=2] _pred_arr = self.graph.output_activating_function.activate(
            np.dot(pred_arr[:, -1, :], self.graph.weights_output_arr) + self.graph.output_bias_arr
        )
        return _pred_arr

    def output_back_propagate(self, np.ndarray[DOUBLE_t, ndim=2] pred_arr, np.ndarray[DOUBLE_t, ndim=2] delta_arr):
        '''
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

    def hidden_back_propagate(self, np.ndarray[DOUBLE_t, ndim=2] delta_output_arr):
        '''
        Back propagation in hidden layer.
        
        Args:
            delta_output_arr:    Delta.
        
        Returns:
            Tuple(
                `np.ndarray` of Delta, 
                `list` of gradations
            )
        '''
        cdef int sample_n = delta_output_arr.shape[0]
        cdef int cycle_len = len(self.__memory_tuple_list)
        cdef int dim = self.graph.weights_lstm_observed_arr.shape[0]

        cdef np.ndarray[DOUBLE_t, ndim=3] delta_arr = np.empty((sample_n, cycle_len, dim), dtype=np.float64)

        grads_list = [0, 0, 0]
        cdef np.ndarray[DOUBLE_t, ndim=2] _delta_hidden_arr
        cdef np.ndarray[DOUBLE_t, ndim=2] delta_observed_arr
        cdef np.ndarray[DOUBLE_t, ndim=2] delta_hidden_arr
        cdef np.ndarray delta_rnn_arr = np.array([])

        cdef int bp_count = 0
        cdef int cycle
        for cycle in reversed(range(cycle_len)):
            if bp_count == 0:
                _delta_hidden_arr = delta_output_arr
            else:
                _delta_hidden_arr = delta_hidden_arr

            delta_observed_arr, delta_hidden_arr, delta_rnn_arr, grad_list = self.lstm_backward(
                _delta_hidden_arr,
                delta_rnn_arr,
                cycle
            )
            delta_arr[:, cycle, :] = delta_observed_arr
            for i in range(len(grad_list)):
                grads_list[i] += grad_list[i]

            if bp_count >= self.__bptt_tau:
                break
            bp_count += 1

        self.__memory_tuple_list = []
        return (delta_arr, grads_list)

    def __lstm_forward(
        self,
        np.ndarray[DOUBLE_t, ndim=2] observed_arr,
        np.ndarray[DOUBLE_t, ndim=2] hidden_activity_arr,
        np.ndarray[DOUBLE_t, ndim=2] rnn_activity_arr
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
        cdef int h_col = int(self.graph.weights_lstm_hidden_arr.shape[1] / 4)
        cdef np.ndarray[DOUBLE_t, ndim=2] lstm_matrix = np.dot(
            observed_arr,
            self.graph.weights_lstm_observed_arr
        ) + np.dot(
            hidden_activity_arr, 
            self.graph.weights_lstm_hidden_arr
        ) + self.graph.lstm_bias_arr

        cdef np.ndarray[DOUBLE_t, ndim=2] given_activity_arr = lstm_matrix[:, :h_col]
        cdef np.ndarray[DOUBLE_t, ndim=2] input_gate_activity_arr = lstm_matrix[:, h_col:h_col * 2]
        cdef np.ndarray[DOUBLE_t, ndim=2] forget_gate_activity_arr = lstm_matrix[:, h_col * 2:h_col * 3]
        cdef np.ndarray[DOUBLE_t, ndim=2] output_gate_activity_arr = lstm_matrix[:, h_col * 3:]

        given_activity_arr = self.graph.observed_activating_function.activate(given_activity_arr)
        input_gate_activity_arr = self.graph.input_gate_activating_function.activate(input_gate_activity_arr)
        forget_gate_activity_arr = self.graph.forget_gate_activating_function.activate(forget_gate_activity_arr)
        output_gate_activity_arr = self.graph.output_gate_activating_function.activate(output_gate_activity_arr)

        cdef np.ndarray[DOUBLE_t, ndim=2] _rnn_activity_arr = given_activity_arr * input_gate_activity_arr + forget_gate_activity_arr * rnn_activity_arr
        cdef np.ndarray[DOUBLE_t, ndim=2] _hidden_activity_arr = output_gate_activity_arr * self.graph.hidden_activating_function.activate(_rnn_activity_arr)

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
        np.ndarray[DOUBLE_t, ndim=2] delta_hidden_arr,
        np.ndarray delta_rnn_arr,
        int cycle
    ):
        '''
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
        cdef np.ndarray[DOUBLE_t, ndim=2] observed_arr = self.__memory_tuple_list[cycle][0]
        cdef np.ndarray[DOUBLE_t, ndim=2] pre_hidden_activity_arr = self.__memory_tuple_list[cycle][1]
        cdef np.ndarray[DOUBLE_t, ndim=2] pre_rnn_activity_arr = self.__memory_tuple_list[cycle][2]
        cdef np.ndarray[DOUBLE_t, ndim=2] given_activity_arr = self.__memory_tuple_list[cycle][3]
        cdef np.ndarray[DOUBLE_t, ndim=2] input_gate_activity_arr = self.__memory_tuple_list[cycle][4]
        cdef np.ndarray[DOUBLE_t, ndim=2] forget_gate_activity_arr = self.__memory_tuple_list[cycle][5]
        cdef np.ndarray[DOUBLE_t, ndim=2] output_gate_activity_arr = self.__memory_tuple_list[cycle][6]
        cdef np.ndarray[DOUBLE_t, ndim=2] rnn_activity_arr = self.__memory_tuple_list[cycle][7]

        cdef np.ndarray[DOUBLE_t, ndim=2] _rnn_activity_arr = self.graph.hidden_activating_function.activate(rnn_activity_arr)

        if delta_rnn_arr.shape[0] == 0:
            delta_rnn_arr = np.zeros((delta_hidden_arr.shape[0], delta_hidden_arr.shape[1]))

        cdef np.ndarray[DOUBLE_t, ndim=2] delta_top_arr
        cdef np.ndarray[DOUBLE_t, ndim=2] delta_pre_rnn_arr
        cdef np.ndarray[DOUBLE_t, ndim=2] delta_output_gate_arr
        cdef np.ndarray[DOUBLE_t, ndim=2] delta_forget_gate_arr
        cdef np.ndarray[DOUBLE_t, ndim=2] delta_input_gate_arr
        cdef np.ndarray[DOUBLE_t, ndim=2] delta_given_arr

        delta_top_arr = delta_rnn_arr + (delta_hidden_arr * output_gate_activity_arr) * self.graph.hidden_activating_function.derivative(rnn_activity_arr)
        delta_pre_rnn_arr = delta_top_arr * forget_gate_activity_arr
        delta_output_gate_arr = delta_hidden_arr * rnn_activity_arr * self.graph.output_gate_activating_function.derivative(output_gate_activity_arr)
        delta_forget_gate_arr = delta_top_arr * delta_pre_rnn_arr * self.graph.forget_gate_activating_function.derivative(forget_gate_activity_arr)
        delta_input_gate_arr = delta_top_arr * given_activity_arr * self.graph.input_gate_activating_function.derivative(input_gate_activity_arr)
        delta_given_arr = delta_top_arr * input_gate_activity_arr * self.graph.observed_activating_function.derivative(given_activity_arr)

        cdef np.ndarray[DOUBLE_t, ndim=2] delta_lstm_matrix = np.hstack([
            delta_output_gate_arr,
            delta_forget_gate_arr,
            delta_input_gate_arr,
            delta_given_arr
        ])

        cdef np.ndarray[DOUBLE_t, ndim=2] delta_weights_h_arr = np.dot(pre_hidden_activity_arr.T, delta_lstm_matrix)
        cdef np.ndarray[DOUBLE_t, ndim=2] delta_weights_x_arr = np.dot(observed_arr.T, delta_lstm_matrix)
        cdef np.ndarray[DOUBLE_t, ndim=1] delta_bias_arr = delta_lstm_matrix.sum(axis=0)

        grad_list = [
            delta_weights_h_arr,
            delta_weights_x_arr,
            delta_bias_arr
        ]
        cdef np.ndarray[DOUBLE_t, ndim=2] delta_observed_arr = np.dot(delta_lstm_matrix, delta_weights_x_arr.T)
        cdef np.ndarray[DOUBLE_t, ndim=2] delta_pre_hidden_arr = np.dot(delta_lstm_matrix, delta_weights_h_arr.T)

        return (delta_observed_arr, delta_pre_hidden_arr, delta_pre_rnn_arr, grad_list)

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
