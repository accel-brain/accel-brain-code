# -*- coding: utf-8 -*-
from logging import getLogger
import numpy as np
cimport numpy as np
from pydbm.synapse_list import Synapse
from pydbm.verification.interface.verificatable_result import VerificatableResult
from pydbm.loss.interface.computable_loss import ComputableLoss
from pydbm.optimization.opt_params import OptParams
from pydbm.rnn.interface.reconstructable_model import ReconstructableModel
ctypedef np.float64_t DOUBLE_t


class LSTMModel(ReconstructableModel):
    '''
    Long short term memory(LSTM) networks.
    
    Originally, Long Short-Term Memory(LSTM) networks as a 
    special RNN structure has proven stable and powerful for 
    modeling long-range dependencies.
    
    The Key point of structural expansion is its memory cell 
    which essentially acts as an accumulator of the state information. 
    Every time observed data points are given as new information and 
    input to LSTM's input gate, its information will be accumulated to 
    the cell if the input gate is activated. The past state of cell 
    could be forgotten in this process if LSTM's forget gate is on.
    Whether the latest cell output will be propagated to the final state 
    is further controlled by the output gate.
    
    References:
        - Cho, K., Van Merriënboer, B., Gulcehre, C., Bahdanau, D., Bougares, F., Schwenk, H., & Bengio, Y. (2014). Learning phrase representations using RNN encoder-decoder for statistical machine translation. arXiv preprint arXiv:1406.1078.
        - Malhotra, P., Ramakrishnan, A., Anand, G., Vig, L., Agarwal, P., & Shroff, G. (2016). LSTM-based encoder-decoder for multi-sensor anomaly detection. arXiv preprint arXiv:1607.00148.
        - Zaremba, W., Sutskever, I., & Vinyals, O. (2014). Recurrent neural network regularization. arXiv preprint arXiv:1409.2329.

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
    
    # is-a `OptParams`.
    __opt_params = None

    # Verification function.
    __verificatable_result = None

    # The list of paramters to be differentiated.
    __learned_params_list = []
    
    # Latest loss
    __latest_loss = None

    # Penalty term of weight decay.
    __weight_decay_term = 0.0

    def __init__(
        self,
        graph,
        computable_loss,
        opt_params,
        verificatable_result,
        int epochs=100,
        int batch_size=100,
        double learning_rate=1e-05,
        double learning_attenuate_rate=0.1,
        int attenuate_epoch=50,
        int seq_len=0,
        int bptt_tau=16,
        double test_size_rate=0.3,
        tol=1e-04,
        tld=100.0
    ):
        '''
        Init for building LSTM networks.

        Args:
            graph:                          is-a `Synapse`.
            computable_loss:                Loss function.
            opt_params:                     Optimization function.
            verificatable_result:           Verification function.
            epochs:                         Epochs of mini-batch.
            batch_size:                     Batch size of mini-batch.
            learning_rate:                  Learning rate.
            learning_attenuate_rate:        Attenuate the `learning_rate` by a factor of this value every `attenuate_epoch`.
            attenuate_epoch:                Attenuate the `learning_rate` by a factor of `learning_attenuate_rate` every `attenuate_epoch`.
                                            Additionally, in relation to regularization,
                                            this class constrains weight matrixes every `attenuate_epoch`.

            seq_len:                        The length of sequences.
                                            This means refereed maxinum step `t` in feedforward.
                                            If `0`, this model will reference all series elements included 
                                            in observed data points.
                                            If not `0`, only first sequence will be observed by this model 
                                            and will be feedfowarded as feature points.
                                            This parameter enables you to build this class as `Decoder` in
                                            Sequence-to-Sequence(Seq2seq) scheme.

            bptt_tau:                       Refereed maxinum step `t` in Backpropagation Through Time(BPTT).
                                            If `0`, this class referes all past data in BPTT.

            test_size_rate:                 Size of Test data set. If this value is `0`, the validation will not be executed.
            tol:                            Tolerance for the optimization.
                                            When the loss or score is not improving by at least tol 
                                            for two consecutive iterations, convergence is considered 
                                            to be reached and training stops.

            tld:                            Tolerance for deviation of loss.
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

        self.__seq_len = seq_len
        self.__bptt_tau = bptt_tau

        self.__test_size_rate = test_size_rate
        self.__tol = tol
        self.__tld = tld

        self.__memory_tuple_list = []
        self.__observed_cycle_len = 0

        logger = getLogger("pydbm")
        self.__logger = logger

    def learn(self, np.ndarray[DOUBLE_t, ndim=3] observed_arr, np.ndarray target_arr=np.array([])):
        '''
        Learn the observed data points
        for vector representation of the input time-series.

        Override.

        Args:
            observed_arr:   Array like or sparse matrix as the observed data points.
                            The shape is: (batch size, the length of sequences, feature points)

            target_arr:     Array like or sparse matrix as the target data points.
                            To learn as Auto-encoder, this value must be `None` or equivalent to `observed_arr`.
                            The shape is: (batch size, labeled data)
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
        cdef np.ndarray[DOUBLE_t, ndim=3] pred_arr
        cdef np.ndarray[DOUBLE_t, ndim=3] test_pred_arr
        cdef np.ndarray delta_arr

        best_params_list = []
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
                    pred_arr = self.forward_propagation(batch_observed_arr)
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
                        self.__remember_best_params(best_params_list)
                        # Re-try.
                        pred_arr = self.forward_propagation(batch_observed_arr)
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
                    delta_arr, grads_list = self.back_propagation(pred_arr, delta_arr)
                    self.optimize(grads_list, learning_rate, epoch)
                    self.graph.hidden_activity_arr = np.array([])
                    self.graph.cec_activity_arr = np.array([])

                    if min_loss is None or min_loss > loss:
                        min_loss = loss
                        best_params_list = [
                            self.graph.weights_output_arr,
                            self.graph.output_bias_arr,
                            self.graph.weights_lstm_hidden_arr,
                            self.graph.weights_lstm_observed_arr,
                            self.graph.lstm_bias_arr
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
                    self.__opt_params.inferencing_mode = True
                    rand_index = np.random.choice(test_observed_arr.shape[0], size=self.__batch_size)
                    test_batch_observed_arr = test_observed_arr[rand_index]
                    test_batch_target_arr = test_target_arr[rand_index]

                    test_pred_arr = self.forward_propagation(test_batch_observed_arr)

                    test_weight_decay = self.__weight_decay_term
                    test_loss = self.__computable_loss.compute_loss(
                        test_pred_arr + self.__weight_decay_term,
                        test_batch_target_arr
                    )
                    test_loss = test_loss + test_weight_decay

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
            self.graph.weights_lstm_hidden_arr = best_params_list[2]
            self.graph.weights_lstm_observed_arr = best_params_list[3]
            self.graph.lstm_bias_arr = best_params_list[4]
            self.__logger.debug("Best params are saved.")

    def forward_propagation(self, np.ndarray[DOUBLE_t, ndim=3] batch_observed_arr):
        '''
        Forward propagation.
        
        Args:
            batch_observed_arr:    Array like or sparse matrix as the observed data points.
        
        Returns:
            Array like or sparse matrix as the predicted data points.
        '''
        self.__weight_decay_term = 0.0
        self.__weight_decay_term += self.__opt_params.compute_weight_decay(
            self.graph.weights_output_arr
        )
        self.__weight_decay_term += self.__opt_params.compute_weight_decay(
            self.graph.weights_lstm_hidden_arr
        )
        self.__weight_decay_term += self.__opt_params.compute_weight_decay(
            self.graph.weights_lstm_observed_arr
        )
        self.__weight_decay_term += self.__opt_params.compute_weight_decay(
            self.graph.weights_input_cec_arr
        )
        self.__weight_decay_term += self.__opt_params.compute_weight_decay(
            self.graph.weights_forget_cec_arr
        )
        self.__weight_decay_term += self.__opt_params.compute_weight_decay(
            self.graph.weights_output_cec_arr
        )

        cdef np.ndarray[DOUBLE_t, ndim=3] hidden_activity_arr = self.hidden_forward_propagate(
            batch_observed_arr
        )

        cdef np.ndarray[DOUBLE_t, ndim=2] arr
        if self.__opt_params.dropout_rate > 0:
            arr = self.__opt_params.dropout(
                hidden_activity_arr.reshape((hidden_activity_arr.shape[0], -1))
            )
            hidden_activity_arr = arr.reshape((
                hidden_activity_arr.shape[0], 
                hidden_activity_arr.shape[1], 
                hidden_activity_arr.shape[2]
            ))

        self.graph.hidden_activity_arr = hidden_activity_arr
        cdef np.ndarray[DOUBLE_t, ndim=3] pred_arr = self.output_forward_propagate(
            hidden_activity_arr
        )
        return pred_arr

    def back_propagation(
        self,
        np.ndarray[DOUBLE_t, ndim=3] pred_arr,
        np.ndarray[DOUBLE_t, ndim=3] delta_arr
    ):
        '''
        Back propagation.

        Args:
            pred_arr:            `np.ndarray` of predicted data points.
            delta_output_arr:    Delta.
        
        Returns:
            Tuple data.
            - `np.ndarray` of Delta, 
            - `list` of gradations
        '''
        delta_arr, output_grads_list = self.output_back_propagate(pred_arr, delta_arr)
        _delta_arr, delta_hidden_arr, lstm_grads_list = self.hidden_back_propagate(delta_arr[:, -1])
        cdef np.ndarray[DOUBLE_t, ndim=2] arr
        if self.__opt_params.dropout_rate > 0:
            arr = self.__opt_params.dropout(
                _delta_arr.reshape((delta_arr.shape[0], -1))
            )
            _delta_arr = arr.reshape((
                _delta_arr.shape[0], 
                _delta_arr.shape[1], 
                _delta_arr.shape[2]
            ))

        grads_list = output_grads_list
        grads_list.extend(lstm_grads_list)
        return (_delta_arr, grads_list)

    def optimize(
        self,
        grads_list,
        double learning_rate,
        int epoch
    ):
        '''
        Optimization.

        Args:
            grads_list:     `list` of graduations.
            learning_rate:  Learning rate.
            epoch:          Now epoch.
            
        '''
        grads_list[0] = self.__opt_params.compute_weight_decay_delta(
            self.graph.weights_output_arr
        )
        grads_list[2] = self.__opt_params.compute_weight_decay_delta(
            self.graph.weights_lstm_hidden_arr
        )
        grads_list[3] = self.__opt_params.compute_weight_decay_delta(
            self.graph.weights_lstm_observed_arr
        )
        grads_list[5] = self.__opt_params.compute_weight_decay_delta(
            self.graph.weights_input_cec_arr
        )
        grads_list[6] = self.__opt_params.compute_weight_decay_delta(
            self.graph.weights_forget_cec_arr
        )
        grads_list[7] = self.__opt_params.compute_weight_decay_delta(
            self.graph.weights_output_cec_arr
        )

        params_list = [
                self.graph.weights_output_arr,
                self.graph.output_bias_arr,
                self.graph.weights_lstm_hidden_arr,
                self.graph.weights_lstm_observed_arr,
                self.graph.lstm_bias_arr,
                self.graph.weights_input_cec_arr,
                self.graph.weights_forget_cec_arr,
                self.graph.weights_output_cec_arr
        ]
        if self.graph.observed_activating_function.batch_norm is not None:
            params_list.append(
                self.graph.observed_activating_function.batch_norm.beta_arr
            )
            grads_list.append(
                self.graph.observed_activating_function.batch_norm.delta_beta_arr
            )
            params_list.append(
                self.graph.observed_activating_function.batch_norm.gamma_arr
            )
            grads_list.append(
                self.graph.observed_activating_function.batch_norm.delta_gamma_arr
            )
        if self.graph.input_gate_activating_function.batch_norm is not None:
            params_list.append(
                self.graph.input_gate_activating_function.batch_norm.beta_arr
            )
            grads_list.append(
                self.graph.input_gate_activating_function.batch_norm.delta_beta_arr
            )
            params_list.append(
                self.graph.input_gate_activating_function.batch_norm.gamma_arr
            )
            grads_list.append(
                self.graph.input_gate_activating_function.batch_norm.delta_gamma_arr
            )
        if self.graph.forget_gate_activating_function.batch_norm is not None:
            params_list.append(
                self.graph.forget_gate_activating_function.batch_norm.beta_arr
            )
            grads_list.append(
                self.graph.forget_gate_activating_function.batch_norm.delta_beta_arr
            )
            params_list.append(
                self.graph.forget_gate_activating_function.batch_norm.gamma_arr
            )
            grads_list.append(
                self.graph.forget_gate_activating_function.batch_norm.delta_gamma_arr
            )
        if self.graph.output_gate_activating_function.batch_norm is not None:
            params_list.append(
                self.graph.output_gate_activating_function.batch_norm.beta_arr
            )
            grads_list.append(
                self.graph.output_gate_activating_function.batch_norm.delta_beta_arr
            )
            params_list.append(
                self.graph.output_gate_activating_function.batch_norm.gamma_arr
            )
            grads_list.append(
                self.graph.output_gate_activating_function.batch_norm.delta_gamma_arr
            )
        if self.graph.hidden_activating_function.batch_norm is not None:
            params_list.append(
                self.graph.hidden_activating_function.batch_norm.beta_arr
            )
            grads_list.append(
                self.graph.hidden_activating_function.batch_norm.delta_beta_arr
            )
            params_list.append(
                self.graph.hidden_activating_function.batch_norm.gamma_arr
            )
            grads_list.append(
                self.graph.hidden_activating_function.batch_norm.delta_gamma_arr
            )
        if self.graph.output_activating_function.batch_norm is not None:
            params_list.append(
                self.graph.output_activating_function.batch_norm.beta_arr
            )
            grads_list.append(
                self.graph.output_activating_function.batch_norm.delta_beta_arr
            )
            params_list.append(
                self.graph.output_activating_function.batch_norm.gamma_arr
            )
            grads_list.append(
                self.graph.output_activating_function.batch_norm.delta_gamma_arr
            )

        params_list = self.__opt_params.optimize(
            params_list,
            grads_list,
            learning_rate
        )
        self.graph.weights_output_arr = params_list.pop(0)
        self.graph.output_bias_arr = params_list.pop(0)
        self.graph.weights_lstm_hidden_arr = params_list.pop(0)
        self.graph.weights_lstm_observed_arr = params_list.pop(0)
        self.graph.lstm_bias_arr = params_list.pop(0)
        self.graph.weights_input_cec_arr = params_list.pop(0)
        self.graph.weights_forget_cec_arr = params_list.pop(0)
        self.graph.weights_output_cec_arr = params_list.pop(0)

        if self.graph.observed_activating_function.batch_norm is not None:
            self.graph.observed_activating_function.batch_norm.beta_arr = params_list.pop(0)
            self.graph.observed_activating_function.batch_norm.gamma_arr = params_list.pop(0)
        if self.graph.input_gate_activating_function.batch_norm is not None:
            self.graph.input_gate_activating_function.batch_norm.beta_arr = params_list.pop(0)
            self.graph.input_gate_activating_function.batch_norm.gamma_arr = params_list.pop(0)
        if self.graph.forget_gate_activating_function.batch_norm is not None:
            self.graph.forget_gate_activating_function.batch_norm.beta_arr = params_list.pop(0)
            self.graph.forget_gate_activating_function.batch_norm.gamma_arr = params_list.pop(0)
        if self.graph.output_gate_activating_function.batch_norm is not None:
            self.graph.output_gate_activating_function.batch_norm.beta_arr = params_list.pop(0)
            self.graph.output_gate_activating_function.batch_norm.gamma_arr = params_list.pop(0)
        if self.graph.hidden_activating_function.batch_norm is not None:
            self.graph.hidden_activating_function.batch_norm.beta_arr = params_list.pop(0)
            self.graph.hidden_activating_function.batch_norm.gamma_arr = params_list.pop(0)
        if self.graph.output_activating_function.batch_norm is not None:
            self.graph.output_activating_function.batch_norm.beta_arr = params_list.pop(0)
            self.graph.output_activating_function.batch_norm.gamma_arr = params_list.pop(0)

        if ((epoch + 1) % self.__attenuate_epoch == 0):
            self.graph.weights_output_arr = self.__opt_params.constrain_weight(self.graph.weights_output_arr)
            self.graph.weights_lstm_hidden_arr = self.__opt_params.constrain_weight(self.graph.weights_lstm_hidden_arr)
            self.graph.weights_lstm_observed_arr = self.__opt_params.constrain_weight(self.graph.weights_lstm_observed_arr)
            self.graph.weights_input_cec_arr = self.__opt_params.constrain_weight(self.graph.weights_input_cec_arr)
            self.graph.weights_forget_cec_arr = self.__opt_params.constrain_weight(self.graph.weights_forget_cec_arr)
            self.graph.weights_output_cec_arr = self.__opt_params.constrain_weight(self.graph.weights_output_cec_arr)

    def inference(
        self,
        np.ndarray observed_arr,
        np.ndarray hidden_activity_arr=None,
        np.ndarray cec_activity_arr=None
    ):
        '''
        Inference the feature points to reconstruct the time-series.

        Override.

        Args:
            observed_arr:           Array like or sparse matrix as the observed data points.
            hidden_activity_arr:    Array like or sparse matrix as the state in hidden layer.
            cec_activity_arr:       Array like or sparse matrix as the state in the constant error carousel.

        Returns:
            Tuple data.
            - Array like or sparse matrix of reconstructed instances of time-series,
            - Array like or sparse matrix of the state in hidden layer,
            - Array like or sparse matrix of the state in RNN.
        '''
        if observed_arr.ndim != 3:
            observed_arr = observed_arr.reshape(
                (
                    observed_arr.shape[0], 
                    observed_arr.shape[1], 
                    -1
                )
            )

        cdef int sample_n = observed_arr.shape[0]
        cdef int cycle_len = observed_arr.shape[1]
        cdef int feature_n = observed_arr.shape[2]
        cdef int hidden_n = self.graph.weights_lstm_hidden_arr.shape[0]

        if hidden_activity_arr is None:
            self.graph.hidden_activity_arr = np.zeros((sample_n, hidden_n), dtype=np.float64)
        else:
            self.graph.hidden_activity_arr = hidden_activity_arr

        if cec_activity_arr is None:
            self.graph.cec_activity_arr = np.zeros((sample_n, hidden_n), dtype=np.float64)
        else:
            self.graph.cec_activity_arr = cec_activity_arr

        cdef np.ndarray[DOUBLE_t, ndim=3] pred_arr = self.forward_propagation(observed_arr)
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
        cdef int cycle_len
        if self.__seq_len == 0:
            cycle_len = observed_arr.shape[1]
            self.__observed_cycle_len = observed_arr.shape[1]
        else:
            cycle_len = self.__seq_len
            self.__observed_cycle_len = self.__seq_len

        cdef int hidden_n = self.graph.weights_lstm_hidden_arr.shape[0]

        cdef np.ndarray[DOUBLE_t, ndim=3] pred_arr = np.zeros((sample_n, cycle_len, hidden_n), dtype=np.float64)

        if self.graph.hidden_activity_arr is None or self.graph.hidden_activity_arr.shape[0] == 0:
            self.graph.hidden_activity_arr = np.zeros((sample_n, hidden_n), dtype=np.float64)

        if self.graph.cec_activity_arr is None or self.graph.cec_activity_arr.shape[0] == 0:
            self.graph.cec_activity_arr = np.zeros((sample_n, hidden_n), dtype=np.float64)

        cdef int cycle
        for cycle in range(cycle_len):
            if self.graph.hidden_activity_arr.ndim == 2:
                if self.__seq_len == 0 or cycle == 0:
                    self.graph.hidden_activity_arr, self.graph.cec_activity_arr = self.__lstm_forward(
                        observed_arr[:, cycle, :],
                        self.graph.hidden_activity_arr,
                        self.graph.cec_activity_arr
                    )
                else:
                    if observed_arr.shape[1] > cycle:
                        self.graph.hidden_activity_arr, self.graph.cec_activity_arr = self.__lstm_forward(
                            observed_arr[:, cycle, :],
                            self.graph.hidden_activity_arr,
                            self.graph.cec_activity_arr
                        )
                    else:
                        self.graph.hidden_activity_arr, self.graph.cec_activity_arr = self.__lstm_forward(
                            pred_arr[:, cycle-1, :],
                            self.graph.hidden_activity_arr,
                            self.graph.cec_activity_arr
                        )

            elif self.graph.hidden_activity_arr.ndim == 3:
                if self.__seq_len == 0 or cycle == 0:
                    self.graph.hidden_activity_arr, self.graph.cec_activity_arr = self.__lstm_forward(
                        observed_arr[:, cycle, :],
                        self.graph.hidden_activity_arr[:, cycle, :],
                        self.graph.cec_activity_arr
                    )
                else:
                    if observed_arr.shape[1] > cycle:
                        self.graph.hidden_activity_arr, self.graph.cec_activity_arr = self.__lstm_forward(
                            observed_arr[:, cycle, :],
                            self.graph.hidden_activity_arr[:, cycle, :],
                            self.graph.cec_activity_arr
                        )
                    else:
                        self.graph.hidden_activity_arr, self.graph.cec_activity_arr = self.__lstm_forward(
                            pred_arr[:, cycle-1, :],
                            self.graph.hidden_activity_arr[:, cycle, :],
                            self.graph.cec_activity_arr
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
        cdef np.ndarray[DOUBLE_t, ndim=3] _pred_arr = self.graph.output_activating_function.activate(
            np.dot(
                pred_arr.reshape((
                    pred_arr.shape[0] * pred_arr.shape[1],
                    -1
                )), 
                self.graph.weights_output_arr
            ) + self.graph.output_bias_arr
        ).reshape((
            pred_arr.shape[0],
            pred_arr.shape[1],
            -1
        ))
        return _pred_arr

    def output_back_propagate(self, np.ndarray[DOUBLE_t, ndim=3] pred_arr, np.ndarray[DOUBLE_t, ndim=3] delta_arr):
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
        cdef int batch_size = delta_arr.shape[0]
        cdef int seq_len = delta_arr.shape[1]

        delta_arr = self.graph.output_activating_function.derivative(
            delta_arr.reshape((
                batch_size * seq_len,
                -1
            ))
        ).reshape((
            batch_size,
            seq_len,
            -1
        ))

        cdef np.ndarray[DOUBLE_t, ndim=2] delta_2d_arr = delta_arr.reshape((
            delta_arr.shape[0] * delta_arr.shape[1],
            -1
        ))

        cdef np.ndarray[DOUBLE_t, ndim=2] _delta_arr = np.dot(
            delta_2d_arr,
            self.graph.weights_output_arr.T
        )

        cdef np.ndarray[DOUBLE_t, ndim=2] delta_weights_arr = np.dot(
            pred_arr.reshape((
                pred_arr.shape[0] * pred_arr.shape[1],
                -1
            )).T, 
            _delta_arr
        ).T
        cdef np.ndarray[DOUBLE_t, ndim=1] delta_bias_arr = np.sum(
            delta_2d_arr, 
            axis=0
        )

        grads_list = [
            delta_weights_arr,
            delta_bias_arr
        ]

        delta_arr = _delta_arr.reshape((
            delta_arr.shape[0],
            delta_arr.shape[1],
            -1
        ))
        return (delta_arr, grads_list)

    def hidden_back_propagate(self, np.ndarray[DOUBLE_t, ndim=2] delta_output_arr):
        '''
        Back propagation in hidden layer.
        
        Args:
            delta_output_arr:    Delta.
        
        Returns:
            Tuple data.
            - `np.ndarray` of Delta in observed data points,
            - `np.ndarray` of Delta in hidden units, 
            - `list` of gradations.
        '''
        cdef int sample_n = delta_output_arr.shape[0]
        cdef int cycle_len = self.__observed_cycle_len
        cdef int dim = self.graph.weights_lstm_observed_arr.shape[0]

        cdef np.ndarray[DOUBLE_t, ndim=3] delta_arr = np.empty((sample_n, cycle_len, dim), dtype=np.float64)

        grads_list = [0, 0, 0, 0, 0, 0]
        cdef np.ndarray[DOUBLE_t, ndim=2] _delta_hidden_arr
        cdef np.ndarray[DOUBLE_t, ndim=2] delta_observed_arr
        cdef np.ndarray[DOUBLE_t, ndim=2] delta_hidden_arr
        cdef np.ndarray delta_cec_arr = np.array([])

        cdef int bp_count = 0
        cdef int cycle
        for cycle in reversed(range(cycle_len)):
            if bp_count == 0:
                _delta_hidden_arr = delta_output_arr
                _delta_hidden_arr = self.__z_score(_delta_hidden_arr)
            else:
                _delta_hidden_arr = delta_hidden_arr

            delta_observed_arr, delta_hidden_arr, delta_cec_arr, grad_list = self.lstm_backward(
                _delta_hidden_arr,
                delta_cec_arr,
                cycle
            )

            delta_arr[:, cycle, :] = delta_observed_arr
            for i in range(len(grad_list)):
                if isinstance(grads_list[i], int) and grads_list[i] == 0:
                    grads_list[i] = grad_list[i]
                else:
                    grads_list[i] = np.nansum(
                        np.array([
                            np.expand_dims(grads_list[i], axis=0),
                            np.expand_dims(grad_list[i], axis=0)
                        ]),
                        axis=0
                    )[0]

            if bp_count >= self.__bptt_tau:
                break
            bp_count += 1

        self.__memory_tuple_list = []
        return (delta_arr, delta_hidden_arr, grads_list)

    def __z_score(self, np.ndarray delta_arr):
        std = delta_arr.std()
        std += 1e-08
        delta_arr = (delta_arr - delta_arr.mean()) / std
        return delta_arr

    def __lstm_forward(
        self,
        np.ndarray[DOUBLE_t, ndim=2] observed_arr,
        np.ndarray[DOUBLE_t, ndim=2] hidden_activity_arr,
        np.ndarray[DOUBLE_t, ndim=2] cec_activity_arr
    ):
        '''
        Forward propagate in LSTM gate.
        
        Args:
            observed_arr:           `np.ndarray` of observed data points.
            hidden_activity_arr:    `np.ndarray` of activities in hidden layer.
            cec_activity_arr:       `np.ndarray` of activities in the constant error carousel.
        
        Returns:
            Tuple data.
            - `np.ndarray` of activities in hidden layer,
            - `np.ndarray` of activities in LSTM gate.
        '''
        cdef int h_col = int(self.graph.weights_lstm_hidden_arr.shape[1] / 4)
        cdef np.ndarray[DOUBLE_t, ndim=2] lstm_matrix = np.dot(
            observed_arr,
            self.graph.weights_lstm_observed_arr
        ) + np.dot(
            hidden_activity_arr, 
            self.graph.weights_lstm_hidden_arr
        ) + self.graph.lstm_bias_arr

        cdef np.ndarray[DOUBLE_t, ndim=2] no_cec_i_arr = lstm_matrix[:, h_col:h_col * 2]
        cdef np.ndarray[DOUBLE_t, ndim=2] no_cec_f_arr = lstm_matrix[:, h_col * 2:h_col * 3]
        cdef np.ndarray[DOUBLE_t, ndim=2] no_cec_o_arr = lstm_matrix[:, h_col * 3:]

        cdef np.ndarray[DOUBLE_t, ndim=2] given_activity_arr = lstm_matrix[:, :h_col]
        cdef np.ndarray[DOUBLE_t, ndim=2] input_gate_activity_arr = lstm_matrix[:, h_col:h_col * 2]
        cdef np.ndarray[DOUBLE_t, ndim=2] forget_gate_activity_arr = lstm_matrix[:, h_col * 2:h_col * 3]
        cdef np.ndarray[DOUBLE_t, ndim=2] output_gate_activity_arr = lstm_matrix[:, h_col * 3:]

        input_gate_activity_arr += np.dot(cec_activity_arr, self.graph.weights_input_cec_arr)
        forget_gate_activity_arr += np.dot(cec_activity_arr, self.graph.weights_forget_cec_arr)

        given_activity_arr = self.graph.observed_activating_function.activate(given_activity_arr)
        input_gate_activity_arr = self.graph.input_gate_activating_function.activate(input_gate_activity_arr)
        forget_gate_activity_arr = self.graph.forget_gate_activating_function.activate(forget_gate_activity_arr)

        cdef np.ndarray[DOUBLE_t, ndim=2] _cec_activity_arr = np.nansum(
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
                        np.expand_dims(cec_activity_arr, axis=0)
                    ]),
                    axis=0
                )
            ]),
            axis=0
        )[0]

        output_gate_activity_arr += np.dot(_cec_activity_arr, self.graph.weights_output_cec_arr)
        output_gate_activity_arr = self.graph.output_gate_activating_function.activate(output_gate_activity_arr)

        cdef np.ndarray[DOUBLE_t, ndim=2] _hidden_activity_arr = np.nanprod(
            np.array([
                np.expand_dims(output_gate_activity_arr, axis=0),
                np.expand_dims(self.graph.hidden_activating_function.activate(_cec_activity_arr), axis=0)
            ]),
            axis=0
        )[0]

        self.__memory_tuple_list.append((
            observed_arr, 
            hidden_activity_arr, 
            cec_activity_arr, 
            given_activity_arr, 
            input_gate_activity_arr, 
            forget_gate_activity_arr, 
            output_gate_activity_arr, 
            _cec_activity_arr,
            _hidden_activity_arr,
            no_cec_i_arr,
            no_cec_f_arr,
            no_cec_o_arr
        ))
        return (_hidden_activity_arr, _cec_activity_arr)

    def lstm_backward(
        self,
        np.ndarray[DOUBLE_t, ndim=2] delta_hidden_arr,
        np.ndarray delta_cec_arr,
        int cycle
    ):
        '''
        Back propagation in LSTM gate.
        
        Args:
            delta_hidden_arr:   Delta from output layer to hidden layer.
            delta_cec_arr:      Delta in LSTM gate.
            cycle:              Now cycle or time.

        Returns:
            Tuple data.
            - Delta from hidden layer to input layer,
            - Delta in hidden layer at previous time,
            - Delta in LSTM gate at previous time,
            - `list` of gradations.
        '''
        cdef np.ndarray[DOUBLE_t, ndim=2] observed_arr = self.__memory_tuple_list[cycle][0]
        cdef np.ndarray[DOUBLE_t, ndim=2] pre_hidden_activity_arr = self.__memory_tuple_list[cycle][1]
        cdef np.ndarray[DOUBLE_t, ndim=2] pre_cec_activity_arr = self.__memory_tuple_list[cycle][2]
        cdef np.ndarray[DOUBLE_t, ndim=2] given_activity_arr = self.__memory_tuple_list[cycle][3]
        cdef np.ndarray[DOUBLE_t, ndim=2] input_gate_activity_arr = self.__memory_tuple_list[cycle][4]
        cdef np.ndarray[DOUBLE_t, ndim=2] forget_gate_activity_arr = self.__memory_tuple_list[cycle][5]
        cdef np.ndarray[DOUBLE_t, ndim=2] output_gate_activity_arr = self.__memory_tuple_list[cycle][6]
        cdef np.ndarray[DOUBLE_t, ndim=2] cec_activity_arr = self.__memory_tuple_list[cycle][7]
        cdef np.ndarray[DOUBLE_t, ndim=2] no_cec_i_arr = self.__memory_tuple_list[cycle][8]
        cdef np.ndarray[DOUBLE_t, ndim=2] no_cec_f_arr = self.__memory_tuple_list[cycle][9]
        cdef np.ndarray[DOUBLE_t, ndim=2] no_cec_o_arr = self.__memory_tuple_list[cycle][10]

        if delta_cec_arr.shape[0] == 0:
            delta_cec_arr = np.zeros((delta_hidden_arr.shape[0], delta_hidden_arr.shape[1]))
        
        delta_cec_arr = self.__z_score(delta_cec_arr)
        delta_hidden_arr = self.__z_score(delta_hidden_arr)
        cec_activity_arr = self.graph.hidden_activating_function.derivative(cec_activity_arr)
        cec_activity_arr = self.__z_score(cec_activity_arr)

        cdef np.ndarray[DOUBLE_t, ndim=2] delta_top_arr = np.nanprod(
            np.array([
                delta_cec_arr,
                np.nansum(
                    np.array([
                        delta_hidden_arr,
                        output_gate_activity_arr,
                        cec_activity_arr
                    ]),
                    axis=0
                )
            ]),
            axis=0
        )
        delta_top_arr = self.__z_score(delta_top_arr)
        cdef np.ndarray[DOUBLE_t, ndim=2] delta_pre_rnn_arr = np.nanprod(
            np.array([delta_top_arr, forget_gate_activity_arr]),
            axis=0
        )

        output_gate_activity_arr = self.graph.output_gate_activating_function.derivative(output_gate_activity_arr)
        output_gate_activity_arr = self.__z_score(output_gate_activity_arr)
        cdef np.ndarray[DOUBLE_t, ndim=2] delta_output_gate_arr = np.nanprod(
            np.array([
                delta_hidden_arr,
                cec_activity_arr,
                output_gate_activity_arr
            ]),
            axis=0
        )
        delta_pre_rnn_arr = self.__z_score(delta_pre_rnn_arr)
        forget_gate_activity_arr = self.graph.forget_gate_activating_function.derivative(forget_gate_activity_arr)
        forget_gate_activity_arr = self.__z_score(forget_gate_activity_arr)
        cdef np.ndarray[DOUBLE_t, ndim=2] delta_forget_gate_arr = np.nanprod(
            np.array([
                delta_top_arr,
                delta_pre_rnn_arr,
                forget_gate_activity_arr
            ]),
            axis=0
        )
        input_gate_activity_arr = self.graph.input_gate_activating_function.derivative(input_gate_activity_arr)
        input_gate_activity_arr = self.__z_score(input_gate_activity_arr)
        cdef np.ndarray[DOUBLE_t, ndim=2] delta_input_gate_arr = np.nanprod(
            np.array([
                delta_top_arr,
                given_activity_arr,
                input_gate_activity_arr
            ]),
            axis=0
        )
        given_activity_arr = self.graph.observed_activating_function.derivative(given_activity_arr)
        given_activity_arr = self.__z_score(given_activity_arr)
        cdef np.ndarray[DOUBLE_t, ndim=2] delta_given_arr = np.nanprod(
            np.array([
                delta_top_arr,
                input_gate_activity_arr,
                given_activity_arr
            ]),
            axis=0
        )
        delta_output_gate_arr = self.__z_score(delta_output_gate_arr)
        cdef np.ndarray[DOUBLE_t, ndim=2] delta_weights_o_cec_arr = np.dot(
            no_cec_o_arr.T, delta_output_gate_arr 
        )
        delta_forget_gate_arr = self.__z_score(delta_forget_gate_arr)
        cdef np.ndarray[DOUBLE_t, ndim=2] delta_weights_f_cec_arr = np.dot(
            no_cec_f_arr.T, delta_forget_gate_arr 
        )
        delta_input_gate_arr = self.__z_score(delta_input_gate_arr)
        cdef np.ndarray[DOUBLE_t, ndim=2] delta_weights_i_cec_arr = np.dot(
            no_cec_i_arr.T, delta_input_gate_arr 
        )
        delta_given_arr = self.__z_score(delta_given_arr)
        cdef np.ndarray[DOUBLE_t, ndim=2] delta_lstm_matrix = np.hstack([
            delta_output_gate_arr,
            delta_forget_gate_arr,
            delta_input_gate_arr,
            delta_given_arr
        ])

        cdef np.ndarray[DOUBLE_t, ndim=2] delta_weights_h_arr = np.dot(pre_hidden_activity_arr.T, delta_lstm_matrix) / 4
        cdef np.ndarray[DOUBLE_t, ndim=2] delta_weights_x_arr = np.dot(observed_arr.T, delta_lstm_matrix) / 4
        cdef np.ndarray[DOUBLE_t, ndim=1] delta_bias_arr = delta_lstm_matrix.sum(axis=0)

        delta_weights_h_arr = self.__z_score(delta_weights_h_arr)
        delta_weights_x_arr = self.__z_score(delta_weights_x_arr)
        delta_bias_arr = self.__z_score(delta_bias_arr)
        grad_list = [
            delta_weights_h_arr,
            delta_weights_x_arr,
            delta_bias_arr,
            delta_weights_i_cec_arr,
            delta_weights_f_cec_arr,
            delta_weights_o_cec_arr
        ]
        cdef np.ndarray[DOUBLE_t, ndim=2] delta_observed_arr = np.dot(delta_lstm_matrix, delta_weights_x_arr.T)
        cdef np.ndarray[DOUBLE_t, ndim=2] delta_pre_hidden_arr = np.dot(delta_lstm_matrix, delta_weights_h_arr.T)
        delta_observed_arr = self.__z_score(delta_observed_arr)
        delta_pre_hidden_arr = self.__z_score(delta_pre_hidden_arr)
        return (delta_observed_arr, delta_pre_hidden_arr, delta_pre_rnn_arr, grad_list)

    def get_opt_params(self):
        ''' getter '''
        if isinstance(self.__opt_params, OptParams):
            return self.__opt_params
        else:
            raise TypeError()
    
    def set_opt_params(self, value):
        ''' setter '''
        if isinstance(value, OptParams):
            self.__opt_params = value
        else:
            raise TypeError()

    opt_params = property(get_opt_params, set_opt_params)

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
            file_name = "lstm_graph.npz"
        else:
            if ".npz" not in file_name:
                file_name += ".npz"

        self.graph.save_pre_learned_params(dir_name + file_name)

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
            file_name = "lstm_graph.npz"
        else:
            if ".npz" not in file_name:
                file_name += ".npz"

        self.graph.load_pre_learned_params(dir_name + file_name)

    def get_weight_decay_term(self):
        ''' getter '''
        return self.__weight_decay_term
    
    def set_weight_decay_term(self, value):
        ''' setter '''
        self.__weight_decay_term = value
    
    weight_decay_term = property(get_weight_decay_term, set_weight_decay_term)
