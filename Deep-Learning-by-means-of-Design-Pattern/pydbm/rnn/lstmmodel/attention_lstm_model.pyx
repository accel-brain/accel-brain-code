# -*- coding: utf-8 -*-
from logging import getLogger
import numpy as np
cimport numpy as np
from pydbm.synapse_list import Synapse
from pydbm.activation.softmax_function import SoftmaxFunction
from pydbm.verification.interface.verificatable_result import VerificatableResult
from pydbm.loss.interface.computable_loss import ComputableLoss
from pydbm.optimization.opt_params import OptParams
from pydbm.rnn.lstm_model import LSTMModel
ctypedef np.float64_t DOUBLE_t


class AttentionLSTMModel(LSTMModel):
    '''
    Attention model of Long short term memory(LSTM) networks.

    The function of this class is to behave as decoder of Encoder/Decoder.
    This decoder model has a mechanism of attention so as to decides parts of the source
    sequences to pay attention to. This mechanism enalbes the encoder to reduce the burden 
    of having to encode all information in the source sequence into a fixed-length context vector. 
    With this new approach the information can be spread throughout the sequence of annotations, 
    which can be selectively retrieved by the decoder accordingly.

    References:
        - Bahdanau, D., Cho, K., & Bengio, Y. (2014). Neural machine translation by jointly learning to align and translate. arXiv preprint arXiv:1409.0473.
        - Vaswani, A., Shazeer, N., Parmar, N., Uszkoreit, J., Jones, L., Gomez, A. N., ... & Polosukhin, I. (2017). Attention is all you need. In Advances in Neural Information Processing Systems (pp. 5998-6008).
    '''

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
        super().__init__(
            graph=graph,
            computable_loss=computable_loss,
            opt_params=opt_params,
            verificatable_result=verificatable_result,
            epochs=epochs,
            batch_size=batch_size,
            learning_rate=learning_rate,
            learning_attenuate_rate=learning_attenuate_rate,
            attenuate_epoch=attenuate_epoch,
            seq_len=seq_len,
            bptt_tau=bptt_tau,
            test_size_rate=test_size_rate,
            tol=tol,
            tld=tld
        )
        self.__softmax_function = SoftmaxFunction()
        self.__attenuate_epoch = attenuate_epoch
        self.__opt_params = opt_params

    def forward_propagation(self, np.ndarray[DOUBLE_t, ndim=3] batch_observed_arr):
        '''
        Forward propagation.
        
        Args:
            batch_observed_arr:    Array like or sparse matrix as the observed data points.
        
        Returns:
            Array like or sparse matrix as the predicted data points.
        '''
        self.weight_decay_term = 0.0
        self.weight_decay_term += self.__opt_params.compute_weight_decay(
            self.graph.attention_output_weight_arr
        )
        self.weight_decay_term += self.__opt_params.compute_weight_decay(
            self.graph.weights_lstm_hidden_arr
        )
        self.weight_decay_term += self.__opt_params.compute_weight_decay(
            self.graph.weights_lstm_observed_arr
        )
        self.weight_decay_term += self.__opt_params.compute_weight_decay(
            self.graph.weights_input_cec_arr
        )
        self.weight_decay_term += self.__opt_params.compute_weight_decay(
            self.graph.weights_forget_cec_arr
        )
        self.weight_decay_term += self.__opt_params.compute_weight_decay(
            self.graph.weights_output_cec_arr
        )

        cdef np.ndarray[DOUBLE_t, ndim=3] hidden_activity_arr = self.hidden_forward_propagate(
            batch_observed_arr
        )

        cdef np.ndarray[DOUBLE_t, ndim=2] arr
        if self.opt_params.dropout_rate > 0:
            arr = self.opt_params.dropout(
                hidden_activity_arr.reshape((hidden_activity_arr.shape[0], -1))
            )
            hidden_activity_arr = arr.reshape((
                hidden_activity_arr.shape[0], 
                hidden_activity_arr.shape[1], 
                hidden_activity_arr.shape[2]
            ))

        cdef np.ndarray[DOUBLE_t, ndim=3] context_weight_arr = np.empty_like(hidden_activity_arr)
        cdef int cycle_len = context_weight_arr.shape[1]
        cdef np.ndarray[DOUBLE_t, ndim=2] weight_activity_arr
        cdef np.ndarray[DOUBLE_t, ndim=2] context_arr

        for cycle in range(cycle_len):
            weight_activity_arr = self.weight_forward(
                batch_observed_arr,
                hidden_activity_arr[:, cycle, :]
            )
            context_arr = self.context_forward(
                batch_observed_arr,
                weight_activity_arr
            )
            context_weight_arr[:, cycle, :] = context_arr

        context_weight_arr = np.concatenate((context_weight_arr, hidden_activity_arr), axis=2)
        self.graph.hidden_activity_arr = hidden_activity_arr
        cdef np.ndarray[DOUBLE_t, ndim=3] pred_arr = self.output_forward_propagate(
            context_weight_arr
        )

        self.__cycle_len = cycle_len
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
                self.graph.attention_output_weight_arr
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
        ).reshape((batch_size, seq_len, -1))

        cdef np.ndarray[DOUBLE_t, ndim=2] delta_2d_arr = delta_arr.reshape((
            delta_arr.shape[0] * delta_arr.shape[1],
            -1
        ))

        cdef np.ndarray[DOUBLE_t, ndim=2] _delta_arr = np.dot(
            delta_2d_arr,
            self.graph.attention_output_weight_arr.T
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

        cdef np.ndarray[DOUBLE_t, ndim=3] delta_hidden_arr
        cdef np.ndarray[DOUBLE_t, ndim=3] _delta_hidden_arr
        cdef np.ndarray[DOUBLE_t, ndim=2] delta_observed_arr
        cdef np.ndarray[DOUBLE_t, ndim=2] _delta_observed_arr

        cdef np.ndarray[DOUBLE_t, ndim=3] delta_context_hidden_arr = None

        cdef int sample_n = delta_arr.shape[0]
        cdef int cycle_len = self.__cycle_len
        cdef int dim = self.graph.weights_lstm_hidden_arr.shape[1] // 4

        cdef int hidden_n = delta_arr.shape[2] // 2
        cdef np.ndarray[DOUBLE_t, ndim=3] delta_pure_hidden_arr = delta_arr[:, :, :hidden_n]
        cdef np.ndarray[DOUBLE_t, ndim=3] delta_attention_hidden_arr = delta_arr[:, :, hidden_n:]
        cdef np.ndarray[DOUBLE_t, ndim=2] _delta_pure_hidden_arr

        cdef np.ndarray[DOUBLE_t, ndim=3] delta_context_weight_arr = np.empty((
            sample_n,
            cycle_len,
            dim
        ))
        cdef int bp_count = 0
        cdef int cycle
        cdef np.ndarray[DOUBLE_t, ndim=2] bp_arr
        for cycle in reversed(range(cycle_len)):
            if bp_count == 0:
                bp_arr = delta_pure_hidden_arr[:, -1]
            else:
                bp_arr = _delta_observed_arr
            delta_hidden_arr, delta_observed_arr = self.context_backward(bp_arr)
            _delta_hidden_arr, _delta_observed_arr = self.weight_backward(delta_observed_arr)
            delta_hidden_arr = delta_hidden_arr + _delta_hidden_arr
            if delta_context_hidden_arr is None:
                delta_context_hidden_arr = delta_hidden_arr
            else:
                delta_context_hidden_arr += delta_hidden_arr
            delta_context_weight_arr[:, cycle, :] = _delta_observed_arr

        cdef np.ndarray[DOUBLE_t, ndim=2] arr
        if self.opt_params.dropout_rate > 0:
            arr = self.opt_params.de_dropout(
                delta_context_weight_arr.reshape((delta_context_weight_arr.shape[0], -1))
            )
            delta_context_weight_arr = arr.reshape((
                delta_context_weight_arr.shape[0], 
                delta_context_weight_arr.shape[1], 
                delta_context_weight_arr.shape[2]
            ))

        delta_attention_hidden_arr += delta_context_weight_arr
        _delta_arr, _delta_pure_hidden_arr, lstm_grads_list = self.hidden_back_propagate(
            delta_attention_hidden_arr[:, -1]
        )
        delta_context_hidden_arr[:, -1] += _delta_pure_hidden_arr

        grads_list = output_grads_list
        grads_list.extend(lstm_grads_list)

        return (delta_context_hidden_arr, grads_list)

    def weight_forward(
        self,
        np.ndarray[DOUBLE_t, ndim=3] observed_arr,
        np.ndarray hidden_activity_arr,
    ):
        cdef int batch_size = observed_arr.shape[0]
        cdef int seq_len = observed_arr.shape[1]
        cdef int feature_n = observed_arr.shape[2]

        cdef np.ndarray[DOUBLE_t, ndim=3] _hidden_activity_arr
        if hidden_activity_arr.ndim == 2:
            _hidden_activity_arr = hidden_activity_arr.reshape((
                hidden_activity_arr.shape[0],
                1,
                hidden_activity_arr.shape[1]
            ))
        else:
            _hidden_activity_arr = hidden_activity_arr[:, -1, :].reshape((
                hidden_activity_arr.shape[0],
                1,
                hidden_activity_arr.shape[2]
            ))

        cdef np.ndarray[DOUBLE_t, ndim=2] weight_activity_arr = np.sum(
            observed_arr * _hidden_activity_arr,
            axis=2
        )
        weight_activity_arr = self.__softmax_function.activate(weight_activity_arr)
        self.__weight_tuple = (observed_arr, _hidden_activity_arr)
        return weight_activity_arr

    def weight_backward(
        self,
        np.ndarray[DOUBLE_t, ndim=2] delta_arr,
    ):
        cdef np.ndarray[DOUBLE_t, ndim=3] observed_arr = self.__weight_tuple[0]
        cdef np.ndarray[DOUBLE_t, ndim=3] hidden_activity_arr = self.__weight_tuple[1]
        cdef int batch_size = observed_arr.shape[0]
        cdef int seq_len = observed_arr.shape[1]
        cdef int feature_n = observed_arr.shape[2]

        delta_arr = self.__softmax_function.derivative(delta_arr)
        cdef np.ndarray[DOUBLE_t, ndim=3] _delta_arr = delta_arr.reshape((
            batch_size,
            seq_len,
            1
        )).repeat(feature_n, axis=2)
        cdef np.ndarray[DOUBLE_t, ndim=3] delta_observed_arr = _delta_arr * observed_arr
        cdef np.ndarray[DOUBLE_t, ndim=3] delta_hidden_arr = _delta_arr * hidden_activity_arr
        cdef np.ndarray[DOUBLE_t, ndim=2] _delta_observed_arr = np.sum(
            delta_observed_arr,
            axis=1
        )
        delta_hidden_arr = delta_hidden_arr.reshape((
            batch_size,
            seq_len,
            -1
        ))
        return delta_hidden_arr, _delta_observed_arr

    def context_forward(
        self,
        np.ndarray[DOUBLE_t, ndim=3] observed_arr,
        np.ndarray[DOUBLE_t, ndim=2] hidden_activity_arr,
    ):
        cdef int batch_size = observed_arr.shape[0]
        cdef int seq_len = observed_arr.shape[1]
        cdef int feature_n = observed_arr.shape[2]
        cdef np.ndarray[DOUBLE_t, ndim=3] _hidden_activity_arr = hidden_activity_arr.reshape((
            batch_size, 
            seq_len, 
            1
        ))
        cdef np.ndarray[DOUBLE_t, ndim=3] context_activity_arr = observed_arr * _hidden_activity_arr
        cdef np.ndarray[DOUBLE_t, ndim=2] context_arr = np.sum(
            context_activity_arr,
            axis=1
        )
        self.__context_tuple = (observed_arr, _hidden_activity_arr)
        return context_arr

    def context_backward(
        self,
        np.ndarray delta_arr
    ):
        cdef np.ndarray[DOUBLE_t, ndim=3] observed_arr = self.__context_tuple[0]
        cdef np.ndarray[DOUBLE_t, ndim=3] hidden_activity_arr = self.__context_tuple[1]
        cdef int batch_size = observed_arr.shape[0]
        cdef int seq_len = observed_arr.shape[1]
        cdef int feature_n = observed_arr.shape[2]
        cdef np.ndarray[DOUBLE_t, ndim=3] _delta_arr = delta_arr.reshape((
            batch_size, 
            1,
            -1
        )).repeat(seq_len, axis=1)
        cdef np.ndarray[DOUBLE_t, ndim=3] delta_observed_arr = _delta_arr * observed_arr
        cdef np.ndarray[DOUBLE_t, ndim=3] delta_hidden_arr = _delta_arr * hidden_activity_arr
        cdef np.ndarray[DOUBLE_t, ndim=2] _delta_observed_arr = np.sum(delta_observed_arr, axis=2)

        return delta_hidden_arr, _delta_observed_arr

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
            self.graph.attention_output_weight_arr
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
            self.graph.attention_output_weight_arr,
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

        params_list = self.opt_params.optimize(
            params_list,
            grads_list,
            learning_rate
        )
        self.graph.attention_output_weight_arr = params_list.pop(0)
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
            self.graph.attention_output_weight_arr = self.opt_params.constrain_weight(self.graph.attention_output_weight_arr)
            self.graph.weights_lstm_hidden_arr = self.opt_params.constrain_weight(self.graph.weights_lstm_hidden_arr)
            self.graph.weights_lstm_observed_arr = self.opt_params.constrain_weight(self.graph.weights_lstm_observed_arr)
            self.graph.weights_input_cec_arr = self.opt_params.constrain_weight(self.graph.weights_input_cec_arr)
            self.graph.weights_forget_cec_arr = self.opt_params.constrain_weight(self.graph.weights_forget_cec_arr)
            self.graph.weights_output_cec_arr = self.opt_params.constrain_weight(self.graph.weights_output_cec_arr)
