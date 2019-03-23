# -*- coding: utf-8 -*-
from logging import getLogger
import numpy as np

from pysummarization.abstractable_semantics import AbstractableSemantics
from pysummarization.vectorizable_token import VectorizableToken

# LSTM Graph which is-a `Synapse`.
from pydbm.synapse.recurrenttemporalgraph.lstm_graph import LSTMGraph as EncoderGraph
from pydbm.synapse.recurrenttemporalgraph.lstm_graph import LSTMGraph as DecoderGraph
from pydbm.synapse.recurrenttemporalgraph.lstm_graph import LSTMGraph as ReEncoderGraph

# Loss function.
from pydbm.loss.mean_squared_error import MeanSquaredError

# Adam as a Loss function.
from pydbm.optimization.optparams.adam import Adam as EncoderAdam
from pydbm.optimization.optparams.adam import Adam as DecoderAdam
from pydbm.optimization.optparams.adam import Adam as ReEncoderAdam
# Verification.
from pydbm.verification.verificate_function_approximation import VerificateFunctionApproximation
# LSTM model.
from pydbm.rnn.lstm_model import LSTMModel
from pydbm.rnn.lstm_model import LSTMModel as Encoder
from pydbm.rnn.lstm_model import LSTMModel as Decoder
from pydbm.rnn.lstm_model import LSTMModel as ReEncoder
# Logistic Function as activation function.
from pydbm.activation.logistic_function import LogisticFunction
# Tanh Function as activation function.
from pydbm.activation.tanh_function import TanhFunction
# Softmax Function as activation function.
from pydbm.activation.softmax_function import SoftmaxFunction
# Encoder/Decoder
from pydbm.rnn.encoder_decoder_controller import EncoderDecoderController
# Iterator/Generator.
from pydbm.cnn.feature_generator import FeatureGenerator


class ReSeq2Seq(AbstractableSemantics):
    '''
    A retrospective sequence-to-sequence learning(re-seq2seq).

    The concept of the re-seq2seq(Zhang, K. et al., 2018) provided inspiration to this library.
    This model is a new sequence learning model mainly in the field of Video Summarizations.
    "The key idea behind re-seq2seq is to measure how well the machine-generated summary 
    is similar to the original video in an abstract semantic space" (Zhang, K. et al., 2018, p3).

    The encoder of a seq2seq model observes the original video and output feature points
    which represents the semantic meaning of the observed data points.
    Then the feature points is observed by the decoder of this model.
    Additionally, in the re-seq2seq model, the outputs of the decoder is propagated
    to a retrospective encoder, which infers feature points to represent the 
    semantic meaning of the summary. "If the summary preserves the important and 
    relevant information in the original video, then we should expect that the 
    two embeddings are similar (e.g. in Euclidean distance)" (Zhang, K. et al., 2018, p3).

    This library refers to this intuitive insight above to apply the model to text summarizations.
    Like videos, semantic feature representation based on representation learning of manifolds 
    is also possible in text summarizations.
    
    The intuition in the design of their loss function is also suggestive.
    "The intuition behind our modeling is that the outputs should convey 
    the same amount of information as the inputs. For summarization, 
    this is precisely the goal: a good summary should be such that after viewing 
    the summary, users would get about the same amount of information as if they 
    had viewed the original video" (Zhang, K. et al., 2018, p7).

    But the model in this library and Zhang, K. et al.(2018) are different in some respects
    from the relation with the specification of the Deep Learning library: [pydbm](https://github.com/chimera0/accel-brain-code/tree/master/Deep-Learning-by-means-of-Design-Pattern).
    First, Encoder/Decoder based on LSTM is not designed as a hierarchical structure. 
    Second, it is possible to introduce regularization techniques which are not discussed in 
    Zhang, K. et al.(2018) such as the dropout, the gradient clipping, and limitation of weights.
    Third, the regression loss function for matching summaries is simplified in terms of 
    calculation efficiency in this library.

    References:
        - Zhang, K., Grauman, K., & Sha, F. (2018). Retrospective Encoders for Video Summarization. In Proceedings of the European Conference on Computer Vision (ECCV) (pp. 383-399).
    '''

    # Logs of accuracy.
    __logs_tuple_list = []

    def __init__(
        self, 
        margin_param=0.01,
        retrospective_lambda=0.5,
        retrospective_eta=0.5,
        encoder_decoder_controller=None,
        retrospective_encoder=None,
        input_neuron_count=20,
        hidden_neuron_count=20,
        weight_limit=0.5,
        dropout_rate=0.5,
        pre_learning_epochs=1000,
        epochs=100,
        batch_size=20,
        learning_rate=1e-05,
        learning_attenuate_rate=0.1,
        attenuate_epoch=50,
        grad_clip_threshold=1e+10,
        seq_len=8,
        bptt_tau=8,
        test_size_rate=0.3,
        tol=0.0,
        tld=100.0
    ):
        '''
        Init.

        Args:
            margin_param:                   A margin parameter for the mismatched pairs penalty.
            retrospective_lambda:           Tradeoff parameter for loss function.
            retrospective_eta:              Tradeoff parameter for loss function.
            encoder_decoder_controller:     is-a `EncoderDecoderController`.
            retrospective_encoder:          is-a `LSTMModel` as a retrospective encoder(or re-encoder).
            input_neuron_count:             The number of units in input layers.
            hidden_neuron_count:            The number of units in hidden layers.
            weight_limit:                   Regularization for weights matrix to repeat multiplying 
                                            the weights matrix and `0.9` until $\sum_{j=0}^{n}w_{ji}^2 < weight\_limit$.

            dropout_rate:                   Probability of dropout.
            pre_learning_epochs:            The epochs in mini-batch pre-learning Encoder/Decoder.
                                            If this value is `0`, no pre-learning will be executed
                                            in this class's method `learn`. In this case, you should 
                                            do pre-learning before calling `learn`.

            epochs:                         The epochs in mini-batch training Encoder/Decoder and retrospective encoder.
            batch_size:                     Batch size.
            learning_rate:                  Learning rate.
            learning_attenuate_rate:        Attenuate the `learning_rate` by a factor of this value every `attenuate_epoch`.
            attenuate_epoch:                Attenuate the `learning_rate` by a factor of `learning_attenuate_rate` every `attenuate_epoch`.
                                            Additionally, in relation to regularization,
                                            this class constrains weight matrixes every `attenuate_epoch`.

            grad_clip_threshold:            Threshold of the gradient clipping.
            seq_len:                        The length of sequneces in Decoder with Attention model.
            bptt_tau:                       Refereed maxinum step `t` in Backpropagation Through Time(BPTT).
                                            If `0`, this class referes all past data in BPTT.

            test_size_rate:                 Size of Test data set. If this value is `0`, the validation will not be executed.
            tol:                            Tolerance for the optimization.
                                            When the loss or score is not improving by at least tol 
                                            for two consecutive iterations, convergence is considered 
                                            to be reached and training stops.

            tld:                            Tolerance for deviation of loss.

        '''
        if isinstance(margin_param, float) is False:
            raise TypeError("The type of `margin_param` must be `float`.")
        if margin_param <= 0:
            raise ValueError("The value of `margin_param` must be more than `0`.")

        self.__margin_param = margin_param

        if isinstance(retrospective_lambda, float) is False or isinstance(retrospective_eta, float) is False:
            raise TypeError("The type of `retrospective_lambda` and `retrospective_eta` must be `float`.")

        if retrospective_lambda < 0 or retrospective_eta < 0:
            raise ValueError("The values of `retrospective_lambda` and `retrospective_eta` must be more then `0`.")
        if retrospective_lambda + retrospective_eta != 1:
            raise ValueError("The sum of `retrospective_lambda` and `retrospective_eta` must be `1`.")

        self.__retrospective_lambda = retrospective_lambda
        self.__retrospective_eta = retrospective_eta

        if encoder_decoder_controller is None:
            encoder_decoder_controller = self.__build_encoder_decoder_controller(
                input_neuron_count=input_neuron_count,
                hidden_neuron_count=hidden_neuron_count,
                weight_limit=weight_limit,
                dropout_rate=dropout_rate,
                epochs=pre_learning_epochs,
                batch_size=batch_size,
                learning_rate=learning_rate,
                attenuate_epoch=attenuate_epoch,
                learning_attenuate_rate=learning_attenuate_rate,
                seq_len=seq_len,
                bptt_tau=bptt_tau,
                test_size_rate=test_size_rate,
                tol=tol,
                tld=tld
            )
        else:
            if isinstance(encoder_decoder_controller, EncoderDecoderController) is False:
                raise TypeError()

        if retrospective_encoder is None:
            retrospective_encoder = self.__build_retrospective_encoder(
                input_neuron_count=input_neuron_count,
                hidden_neuron_count=hidden_neuron_count,
                weight_limit=weight_limit,
                dropout_rate=dropout_rate,
                batch_size=batch_size,
                learning_rate=learning_rate,
                bptt_tau=bptt_tau
            )
        else:
            if isinstance(retrospective_encoder, LSTMModel) is False:
                raise TypeError()

        self.__encoder_decoder_controller = encoder_decoder_controller
        self.__retrospective_encoder = retrospective_encoder
        self.__epochs = epochs
        self.__batch_size = batch_size
        self.__learning_rate = learning_rate
        self.__attenuate_epoch = attenuate_epoch
        self.__learning_attenuate_rate = learning_attenuate_rate
        self.__grad_clip_threshold = grad_clip_threshold
        self.__test_size_rate = test_size_rate
        self.__pre_learning_epochs = pre_learning_epochs
        self.__tol = tol
        self.__tld = tld

        self.__input_neuron_count = input_neuron_count
        self.__hidden_neuron_count = hidden_neuron_count

        logger = getLogger("pysummarization")
        self.__logger = logger
        self.__logs_tuple_list = []

    def __build_encoder_decoder_controller(
        self,
        input_neuron_count=20,
        hidden_neuron_count=20,
        weight_limit=0.5,
        dropout_rate=0.5,
        epochs=1000,
        batch_size=20,
        learning_rate=1e-05,
        attenuate_epoch=50,
        learning_attenuate_rate=0.1,
        seq_len=8,
        bptt_tau=8,
        test_size_rate=0.3,
        tol=1e-10,
        tld=100.0
    ):
        encoder_graph = EncoderGraph()

        encoder_graph.observed_activating_function = LogisticFunction()
        encoder_graph.input_gate_activating_function = LogisticFunction()
        encoder_graph.forget_gate_activating_function = LogisticFunction()
        encoder_graph.output_gate_activating_function = LogisticFunction()
        encoder_graph.hidden_activating_function = LogisticFunction()
        encoder_graph.output_activating_function = LogisticFunction()

        encoder_graph.create_rnn_cells(
            input_neuron_count=input_neuron_count,
            hidden_neuron_count=hidden_neuron_count,
            output_neuron_count=1
        )
        encoder_opt_params = EncoderAdam()
        encoder_opt_params.weight_limit = weight_limit
        encoder_opt_params.dropout_rate = dropout_rate

        encoder = Encoder(
            graph=encoder_graph,
            epochs=100,
            batch_size=batch_size,
            learning_rate=learning_rate,
            learning_attenuate_rate=0.1,
            attenuate_epoch=50,
            bptt_tau=8,
            test_size_rate=0.3,
            computable_loss=MeanSquaredError(),
            opt_params=encoder_opt_params,
            verificatable_result=VerificateFunctionApproximation(),
            tol=tol,
            tld=tld
        )

        decoder_graph = DecoderGraph()

        decoder_graph.observed_activating_function = LogisticFunction()
        decoder_graph.input_gate_activating_function = LogisticFunction()
        decoder_graph.forget_gate_activating_function = LogisticFunction()
        decoder_graph.output_gate_activating_function = LogisticFunction()
        decoder_graph.hidden_activating_function = LogisticFunction()
        decoder_graph.output_activating_function = SoftmaxFunction()

        decoder_graph.create_rnn_cells(
            input_neuron_count=hidden_neuron_count,
            hidden_neuron_count=hidden_neuron_count,
            output_neuron_count=input_neuron_count
        )
        decoder_opt_params = DecoderAdam()
        decoder_opt_params.weight_limit = weight_limit
        decoder_opt_params.dropout_rate = dropout_rate

        decoder = Decoder(
            graph=decoder_graph,
            epochs=100,
            batch_size=batch_size,
            learning_rate=learning_rate,
            learning_attenuate_rate=0.1,
            attenuate_epoch=50,
            seq_len=seq_len,
            bptt_tau=bptt_tau,
            test_size_rate=0.3,
            computable_loss=MeanSquaredError(),
            opt_params=decoder_opt_params,
            verificatable_result=VerificateFunctionApproximation()
        )

        encoder_decoder_controller = EncoderDecoderController(
            encoder=encoder,
            decoder=decoder,
            epochs=epochs,
            batch_size=batch_size,
            learning_rate=learning_rate,
            learning_attenuate_rate=learning_attenuate_rate,
            attenuate_epoch=attenuate_epoch,
            test_size_rate=test_size_rate,
            computable_loss=MeanSquaredError(),
            verificatable_result=VerificateFunctionApproximation(),
            tol=tol,
            tld=tld
        )

        return encoder_decoder_controller

    def __build_retrospective_encoder(
        self,
        input_neuron_count=20,
        hidden_neuron_count=20,
        weight_limit=0.5,
        dropout_rate=0.5,
        batch_size=20,
        learning_rate=1e-05,
        bptt_tau=8
    ):
        encoder_graph = ReEncoderGraph()

        encoder_graph.observed_activating_function = TanhFunction()
        encoder_graph.input_gate_activating_function = LogisticFunction()
        encoder_graph.forget_gate_activating_function = LogisticFunction()
        encoder_graph.output_gate_activating_function = LogisticFunction()
        encoder_graph.hidden_activating_function = LogisticFunction()
        encoder_graph.output_activating_function = LogisticFunction()

        encoder_graph.create_rnn_cells(
            input_neuron_count=input_neuron_count,
            hidden_neuron_count=hidden_neuron_count,
            output_neuron_count=1
        )
        encoder_opt_params = EncoderAdam()
        encoder_opt_params.weight_limit = weight_limit
        encoder_opt_params.dropout_rate = dropout_rate

        encoder = ReEncoder(
            graph=encoder_graph,
            epochs=100,
            batch_size=batch_size,
            learning_rate=learning_rate,
            learning_attenuate_rate=0.1,
            attenuate_epoch=50,
            bptt_tau=bptt_tau,
            test_size_rate=0.3,
            computable_loss=MeanSquaredError(),
            opt_params=encoder_opt_params,
            verificatable_result=VerificateFunctionApproximation()
        )

        return encoder

    def learn(self, observed_arr, target_arr):
        '''
        Training the model.

        Args:
            observed_arr:       `np.ndarray` of observed data points.
            target_arr:         `np.ndarray` of target labeled data.
        '''
        # Pre-learning.
        if self.__pre_learning_epochs > 0:
            self.__encoder_decoder_controller.learn(observed_arr, observed_arr)

        learning_rate = self.__learning_rate
        row_o = observed_arr.shape[0]
        row_t = target_arr.shape[0]
        if row_t != 0 and row_t != row_o:
            raise ValueError("The row of `target_arr` must be equivalent to the row of `observed_arr`.")

        if row_t == 0:
            target_arr = observed_arr.copy()
        else:
            if target_arr.ndim == 2:
                target_arr = target_arr.reshape((target_arr.shape[0], 1, target_arr.shape[1]))

        if self.__test_size_rate > 0:
            train_index = np.random.choice(observed_arr.shape[0], round((1 - self.__test_size_rate) * observed_arr.shape[0]), replace=False)
            test_index = np.array(list(set(range(observed_arr.shape[0])) - set(train_index)))
            train_observed_arr = observed_arr[train_index]
            test_observed_arr = observed_arr[test_index]
            train_target_arr = target_arr[train_index]
            test_target_arr = target_arr[test_index]
        else:
            train_observed_arr = observed_arr
            train_target_arr = target_arr

        encoder_best_params_list = []
        decoder_best_params_list = []
        re_encoder_best_params_list = []
        try:
            self.__change_inferencing_mode(False)
            self.__memory_tuple_list = []
            eary_stop_flag = False
            loss_list = []
            min_loss = None
            for epoch in range(self.__epochs):
                if ((epoch + 1) % self.__attenuate_epoch == 0):
                    learning_rate = learning_rate / self.__learning_attenuate_rate

                rand_index = np.random.choice(train_observed_arr.shape[0], size=self.__batch_size)
                batch_observed_arr = train_observed_arr[rand_index]
                batch_target_arr = train_target_arr[rand_index]
                try:
                    _ = self.inference(batch_observed_arr)
                    delta_arr, _, loss = self.compute_retrospective_loss()

                    remember_flag = False
                    if len(loss_list) > 0:
                        if abs(loss - (sum(loss_list)/len(loss_list))) > self.__tld:
                            remember_flag = True

                    if remember_flag is True:
                        self.__remember_best_params(
                            encoder_best_params_list,
                            decoder_best_params_list,
                            re_encoder_best_params_list
                        )
                        # Re-try.
                        _ = self.inference(batch_observed_arr)
                        delta_arr, _, loss = self.compute_retrospective_loss()

                    re_encoder_grads_list, decoder_grads_list, encoder_delta_arr, encoder_grads_list = self.back_propagation(delta_arr)
                    self.optimize(
                        re_encoder_grads_list,
                        decoder_grads_list, 
                        encoder_grads_list, 
                        learning_rate, 
                        epoch
                    )

                    if min_loss is None or min_loss > loss:
                        min_loss = loss
                        
                        encoder_best_params_list = [
                            self.__encoder_decoder_controller.encoder.graph.weights_lstm_hidden_arr,
                            self.__encoder_decoder_controller.encoder.graph.weights_lstm_observed_arr,
                            self.__encoder_decoder_controller.encoder.graph.lstm_bias_arr
                        ]
                        decoder_best_params_list = [
                            self.__encoder_decoder_controller.decoder.graph.weights_lstm_hidden_arr,
                            self.__encoder_decoder_controller.decoder.graph.weights_lstm_observed_arr,
                            self.__encoder_decoder_controller.decoder.graph.lstm_bias_arr
                        ]
                        re_encoder_best_params_list = [
                            self.__retrospective_encoder.graph.weights_lstm_hidden_arr,
                            self.__retrospective_encoder.graph.weights_lstm_observed_arr,
                            self.__retrospective_encoder.graph.lstm_bias_arr
                        ]
                        self.__logger.debug("Best params are updated.")

                    self.__encoder_decoder_controller.encoder.graph.hidden_activity_arr = np.array([])
                    self.__encoder_decoder_controller.encoder.graph.cec_activity_arr = np.array([])
                    self.__encoder_decoder_controller.decoder.graph.hidden_activity_arr = np.array([])
                    self.__encoder_decoder_controller.decoder.graph.cec_activity_arr = np.array([])
                    self.__retrospective_encoder.graph.hidden_activity_arr = np.array([])
                    self.__retrospective_encoder.graph.cec_activity_arr = np.array([])

                except FloatingPointError:
                    if epoch > int(self.__epochs * 0.7):
                        self.__logger.debug(
                            "Underflow occurred when the parameters are being updated. Because of early stopping, this error is catched and the parameter is not updated."
                        )
                        eary_stop_flag = True
                        break
                    else:
                        self.__logger.debug(
                            "Underflow occurred when the parameters are being updated."
                        )
                        raise

                if self.__test_size_rate > 0:
                    rand_index = np.random.choice(test_observed_arr.shape[0], size=self.__batch_size)
                    test_batch_observed_arr = test_observed_arr[rand_index]
                    test_batch_target_arr = test_target_arr[rand_index]

                    self.__change_inferencing_mode(True)
                    _ = self.inference(test_batch_observed_arr)
                    _, _, test_loss = self.compute_retrospective_loss()

                    remember_flag = False
                    if len(loss_list) > 0:
                        if abs(test_loss - (sum(loss_list)/len(loss_list))) > self.__tld:
                            remember_flag = True

                    if remember_flag is True:
                        self.__remember_best_params(
                            encoder_best_params_list, 
                            decoder_best_params_list,
                            re_encoder_best_params_list
                        )
                        # Re-try.
                        _ = self.inference(test_batch_observed_arr)
                        _, _, test_loss = self.compute_retrospective_loss()

                    self.__change_inferencing_mode(False)

                    self.__verificate_retrospective_loss(loss, test_loss)

                self.__encoder_decoder_controller.encoder.graph.hidden_activity_arr = np.array([])
                self.__encoder_decoder_controller.encoder.graph.cec_activity_arr = np.array([])
                self.__encoder_decoder_controller.decoder.graph.hidden_activity_arr = np.array([])
                self.__encoder_decoder_controller.decoder.graph.cec_activity_arr = np.array([])

                if epoch > 1 and abs(loss - loss_list[-1]) < self.__tol:
                    eary_stop_flag = True
                    break
                loss_list.append(loss)

        except KeyboardInterrupt:
            self.__logger.debug("Interrupt.")

        if eary_stop_flag is True:
            self.__logger.debug("Early stopping.")
            eary_stop_flag = False
        
        self.__remember_best_params(
            encoder_best_params_list, 
            decoder_best_params_list,
            re_encoder_best_params_list
        )
        self.__change_inferencing_mode(True)
        self.__logger.debug("end. ")

    def learn_generated(self, feature_generator):
        '''
        Learn features generated by `FeatureGenerator`.
        
        Args:
            feature_generator:    is-a `FeatureGenerator`.

        '''
        if isinstance(feature_generator, FeatureGenerator) is False:
            raise TypeError("The type of `feature_generator` must be `FeatureGenerator`.")

        # Pre-learning.
        if self.__pre_learning_epochs > 0:
            self.__encoder_decoder_controller.learn_generated(feature_generator)

        learning_rate = self.__learning_rate

        encoder_best_params_list = []
        decoder_best_params_list = []
        re_encoder_best_params_list = []
        try:
            self.__change_inferencing_mode(False)
            self.__memory_tuple_list = []
            eary_stop_flag = False
            loss_list = []
            min_loss = None
            epoch = 0
            for batch_observed_arr, batch_target_arr, test_batch_observed_arr, test_batch_target_arr in feature_generator.generate():
                epoch += 1
                if ((epoch + 1) % self.__attenuate_epoch == 0):
                    learning_rate = learning_rate / self.__learning_attenuate_rate

                try:
                    _ = self.inference(batch_observed_arr)
                    delta_arr, _, loss = self.compute_retrospective_loss()

                    remember_flag = False
                    if len(loss_list) > 0:
                        if abs(loss - (sum(loss_list)/len(loss_list))) > self.__tld:
                            remember_flag = True

                    if remember_flag is True:
                        self.__remember_best_params(
                            encoder_best_params_list,
                            decoder_best_params_list,
                            re_encoder_best_params_list
                        )
                        # Re-try.
                        _ = self.inference(batch_observed_arr)
                        delta_arr, _, loss = self.compute_retrospective_loss()

                    re_encoder_grads_list, decoder_grads_list, encoder_delta_arr, encoder_grads_list = self.back_propagation(delta_arr)
                    self.optimize(
                        re_encoder_grads_list,
                        decoder_grads_list, 
                        encoder_grads_list, 
                        learning_rate, 
                        epoch
                    )

                    if min_loss is None or min_loss > loss:
                        min_loss = loss
                        
                        encoder_best_params_list = [
                            self.__encoder_decoder_controller.encoder.graph.weights_lstm_hidden_arr,
                            self.__encoder_decoder_controller.encoder.graph.weights_lstm_observed_arr,
                            self.__encoder_decoder_controller.encoder.graph.lstm_bias_arr
                        ]
                        decoder_best_params_list = [
                            self.__encoder_decoder_controller.decoder.graph.weights_lstm_hidden_arr,
                            self.__encoder_decoder_controller.decoder.graph.weights_lstm_observed_arr,
                            self.__encoder_decoder_controller.decoder.graph.lstm_bias_arr
                        ]
                        re_encoder_best_params_list = [
                            self.__retrospective_encoder.graph.weights_lstm_hidden_arr,
                            self.__retrospective_encoder.graph.weights_lstm_observed_arr,
                            self.__retrospective_encoder.graph.lstm_bias_arr
                        ]
                        self.__logger.debug("Best params are updated.")

                    self.__encoder_decoder_controller.encoder.graph.hidden_activity_arr = np.array([])
                    self.__encoder_decoder_controller.encoder.graph.cec_activity_arr = np.array([])
                    self.__encoder_decoder_controller.decoder.graph.hidden_activity_arr = np.array([])
                    self.__encoder_decoder_controller.decoder.graph.cec_activity_arr = np.array([])
                    self.__retrospective_encoder.graph.hidden_activity_arr = np.array([])
                    self.__retrospective_encoder.graph.cec_activity_arr = np.array([])

                except FloatingPointError:
                    if epoch > int(self.__epochs * 0.7):
                        self.__logger.debug(
                            "Underflow occurred when the parameters are being updated. Because of early stopping, this error is catched and the parameter is not updated."
                        )
                        eary_stop_flag = True
                        break
                    else:
                        self.__logger.debug(
                            "Underflow occurred when the parameters are being updated."
                        )
                        raise

                if self.__test_size_rate > 0:
                    self.__change_inferencing_mode(True)
                    _ = self.inference(test_batch_observed_arr)
                    _, _, test_loss = self.compute_retrospective_loss()

                    remember_flag = False
                    if len(loss_list) > 0:
                        if abs(test_loss - (sum(loss_list)/len(loss_list))) > self.__tld:
                            remember_flag = True

                    if remember_flag is True:
                        self.__remember_best_params(
                            encoder_best_params_list, 
                            decoder_best_params_list,
                            re_encoder_best_params_list
                        )
                        # Re-try.
                        _ = self.inference(test_batch_observed_arr)
                        _, _, test_loss = self.compute_retrospective_loss()

                    self.__change_inferencing_mode(False)
                    self.__verificate_retrospective_loss(loss, test_loss)

                self.__encoder_decoder_controller.encoder.graph.hidden_activity_arr = np.array([])
                self.__encoder_decoder_controller.encoder.graph.cec_activity_arr = np.array([])
                self.__encoder_decoder_controller.decoder.graph.hidden_activity_arr = np.array([])
                self.__encoder_decoder_controller.decoder.graph.cec_activity_arr = np.array([])

                if epoch > 1 and abs(loss - loss_list[-1]) < self.__tol:
                    eary_stop_flag = True
                    break
                loss_list.append(loss)

        except KeyboardInterrupt:
            self.__logger.debug("Interrupt.")

        if eary_stop_flag is True:
            self.__logger.debug("Early stopping.")
            eary_stop_flag = False
        
        self.__remember_best_params(
            encoder_best_params_list, 
            decoder_best_params_list,
            re_encoder_best_params_list
        )
        self.__change_inferencing_mode(True)
        self.__logger.debug("end. ")

    def inference(self, observed_arr):
        '''
        Infernece by the model.

        Args:
            observed_arr:       `np.ndarray` of observed data points.

        Returns:
            `np.ndarray` of inferenced feature points.
        '''
        decoded_arr = self.__encoder_decoder_controller.inference(observed_arr)
        encoded_arr = self.__encoder_decoder_controller.get_feature_points()
        _ = self.__retrospective_encoder.inference(decoded_arr)
        re_encoded_arr = self.__retrospective_encoder.get_feature_points()
        self.__inferenced_tuple = (observed_arr, encoded_arr, decoded_arr, re_encoded_arr)
        return re_encoded_arr

    def summarize(self, test_arr, vectorizable_token, sentence_list, limit=5):
        '''
        Summarize input document.

        Args:
            test_arr:               `np.ndarray` of observed data points..
            vectorizable_token:     is-a `VectorizableToken`.
            sentence_list:          `list` of all sentences.
            limit:                  The number of selected abstract sentence.
        
        Returns:
            `list` of `str` of abstract sentences.
        '''
        if isinstance(vectorizable_token, VectorizableToken) is False:
            raise TypeError()

        _ = self.inference(test_arr)
        _, loss_arr, _ = self.compute_retrospective_loss()

        loss_list = loss_arr.tolist()

        abstract_list = []
        for i in range(limit):
            key = loss_arr.argmin()
            _ = loss_list.pop(key)
            loss_arr = np.array(loss_list)

            seq_arr = test_arr[key]
            token_arr = vectorizable_token.tokenize(seq_arr.tolist())
            s = " ".join(token_arr.tolist())

            for sentence in sentence_list:
                if s in sentence:
                    abstract_list.append(sentence)
                    abstract_list = list(set(abstract_list))

            if len(abstract_list) >= limit:
                break

        return abstract_list

    def back_propagation(self, delta_arr):
        '''
        Back propagation.

        Args:
            delta_output_arr:    Delta.
        
        Returns:
            Tuple data.
            - decoder's `list` of gradations,
            - encoder's `np.ndarray` of Delta, 
            - encoder's `list` of gradations.
        '''
        re_encoder_delta_arr, delta_hidden_arr, re_encoder_grads_list = self.__retrospective_encoder.hidden_back_propagate(
            delta_arr[:, -1]
        )
        re_encoder_grads_list.insert(0, None)
        re_encoder_grads_list.insert(0, None)

        observed_arr, encoded_arr, decoded_arr, re_encoded_arr = self.__inferenced_tuple
        delta_arr = self.__encoder_decoder_controller.computable_loss.compute_delta(
            decoded_arr, 
            observed_arr
        )
        delta_arr[:, -1] += re_encoder_delta_arr[:, -1]

        decoder_grads_list, encoder_delta_arr, encoder_grads_list = self.__encoder_decoder_controller.back_propagation(
            delta_arr
        )
        return re_encoder_grads_list, decoder_grads_list, encoder_delta_arr, encoder_grads_list

    def optimize(
        self,
        re_encoder_grads_list,
        decoder_grads_list,
        encoder_grads_list,
        learning_rate,
        epoch
    ):
        '''
        Back propagation.
        
        Args:
            re_encoder_grads_list:  re-encoder's `list` of graduations.
            decoder_grads_list:     decoder's `list` of graduations.
            encoder_grads_list:     encoder's `list` of graduations.
            learning_rate:          Learning rate.
            epoch:                  Now epoch.
        '''
        self.__retrospective_encoder.optimize(re_encoder_grads_list, learning_rate, epoch)
        self.__encoder_decoder_controller.optimize(
            decoder_grads_list,
            encoder_grads_list,
            learning_rate,
            epoch
        )

    def compute_retrospective_loss(self):
        '''
        Compute retrospective loss.

        Returns:
            The tuple data.
            - `np.ndarray` of delta.
            - `np.ndarray` of losses of each batch.
            - float of loss of all batch.

        '''
        observed_arr, encoded_arr, decoded_arr, re_encoded_arr = self.__inferenced_tuple
        batch_size = observed_arr.shape[0]
        if self.__input_neuron_count == self.__hidden_neuron_count:
            target_arr = encoded_arr - np.expand_dims(observed_arr.mean(axis=2), axis=2)
            summary_delta_arr = np.sqrt(np.power(decoded_arr - target_arr, 2))
        else:
            # For each batch, draw a samples from the Uniform distribution.
            if self.__input_neuron_count > self.__hidden_neuron_count:
                all_dim_arr = np.arange(self.__input_neuron_count)
                np.random.shuffle(all_dim_arr)
                choiced_dim_arr = all_dim_arr[:self.__hidden_neuron_count]
                target_arr = encoded_arr - np.expand_dims(observed_arr[:, :, choiced_dim_arr].mean(axis=2), axis=2)
                summary_delta_arr = np.sqrt(np.power(decoded_arr[:, :, choiced_dim_arr] - target_arr, 2))
            else:
                all_dim_arr = np.arange(self.__hidden_neuron_count)
                np.random.shuffle(all_dim_arr)
                choiced_dim_arr = all_dim_arr[:self.__input_neuron_count]
                target_arr = encoded_arr[:, :, choiced_dim_arr] - np.expand_dims(observed_arr.mean(axis=2), axis=2)
                summary_delta_arr = np.sqrt(np.power(decoded_arr - target_arr, 2))

        summary_delta_arr = np.nan_to_num(summary_delta_arr)
        summary_delta_arr = (summary_delta_arr - summary_delta_arr.mean()) / (summary_delta_arr.std() + 1e-08)

        match_delta_arr = np.sqrt(np.power(encoded_arr[:, -1] - re_encoded_arr[:, -1], 2))
        match_delta_arr = np.nan_to_num(match_delta_arr)
        match_delta_arr = (match_delta_arr - match_delta_arr.mean()) / (match_delta_arr.std() + 1e-08)

        other_encoded_delta_arr = np.nansum(
            np.sqrt(
                np.power(
                    np.maximum(
                        0,
                        encoded_arr[:, :-1] - re_encoded_arr[:, -1].reshape(
                            re_encoded_arr[:, -1].shape[0], 
                            1, 
                            re_encoded_arr[:, -1].shape[1]
                        )
                    ),
                    2
                )
            ) + self.__margin_param,
            axis=1
        )
        other_encoded_delta_arr = np.nan_to_num(other_encoded_delta_arr)
        other_encoded_delta_arr = (other_encoded_delta_arr - other_encoded_delta_arr.mean()) / (other_encoded_delta_arr.std() + 1e-08)

        other_re_encoded_delta_arr = np.nansum(
            np.sqrt(
                np.power(
                    np.maximum(
                        0, 
                        encoded_arr[:, -1].reshape(
                            encoded_arr[:, -1].shape[0],
                            1,
                            encoded_arr[:, -1].shape[1]
                        ) - re_encoded_arr[:, :-1], 
                    ),
                    2
                )
            ) + self.__margin_param,
            axis=1
        )
        other_encoded_delta_arr = np.nan_to_num(other_encoded_delta_arr)
        other_re_encoded_delta_arr = (other_re_encoded_delta_arr - other_re_encoded_delta_arr.mean()) / (other_re_encoded_delta_arr.std() + 1e-08)

        mismatch_delta_arr = (match_delta_arr - other_encoded_delta_arr) + (match_delta_arr - other_re_encoded_delta_arr)

        delta_arr = summary_delta_arr + np.expand_dims(self.__retrospective_lambda * match_delta_arr, axis=1) + np.expand_dims(self.__retrospective_eta * mismatch_delta_arr, axis=1)

        v = np.linalg.norm(delta_arr)
        if v > self.__grad_clip_threshold:
            delta_arr = delta_arr * self.__grad_clip_threshold / v

        loss = np.square(delta_arr).mean()
        loss_arr = np.square(delta_arr).sum(axis=1).mean(axis=1)
        return delta_arr, loss_arr, loss

    def __change_inferencing_mode(self, inferencing_mode):
        '''
        Change dropout rate in Encoder/Decoder.
        
        Args:
            dropout_rate:   The probalibity of dropout.
        '''
        self.__encoder_decoder_controller.decoder.opt_params.inferencing_mode = inferencing_mode
        self.__encoder_decoder_controller.encoder.opt_params.inferencing_mode = inferencing_mode
        self.__retrospective_encoder.opt_params.inferencing_mode = inferencing_mode

    def __remember_best_params(self, encoder_best_params_list, decoder_best_params_list, re_encoder_best_params_list):
        '''
        Remember best parameters.
        
        Args:
            encoder_best_params_list:    `list` of encoder's parameters.
            decoder_best_params_list:    `list` of decoder's parameters.
            re_encoder_best_params_list: `list` of re-decoder's parameters.

        '''
        if len(encoder_best_params_list) > 0 and len(decoder_best_params_list) > 0:
            self.__encoder_decoder_controller.encoder.graph.weights_lstm_hidden_arr = encoder_best_params_list[0]
            self.__encoder_decoder_controller.encoder.graph.weights_lstm_observed_arr = encoder_best_params_list[1]
            self.__encoder_decoder_controller.encoder.graph.lstm_bias_arr = encoder_best_params_list[2]

            self.__encoder_decoder_controller.decoder.graph.weights_lstm_hidden_arr = decoder_best_params_list[0]
            self.__encoder_decoder_controller.decoder.graph.weights_lstm_observed_arr = decoder_best_params_list[1]
            self.__encoder_decoder_controller.decoder.graph.lstm_bias_arr = decoder_best_params_list[2]

            self.__retrospective_encoder.graph.weights_lstm_hidden_arr = re_encoder_best_params_list[0]
            self.__retrospective_encoder.graph.weights_lstm_observed_arr = re_encoder_best_params_list[1]
            self.__retrospective_encoder.graph.lstm_bias_arr = re_encoder_best_params_list[2]

            self.__logger.debug("Best params are saved.")

    def __verificate_retrospective_loss(self, train_loss, test_loss):
        self.__logger.debug("Epoch: " + str(len(self.__logs_tuple_list) + 1))

        self.__logger.debug("Loss: ")
        self.__logger.debug(
            "Training: " + str(train_loss) + " Test: " + str(test_loss)
        )        
        self.__logs_tuple_list.append(
            (
                train_loss,
                test_loss
            )
        )

    def get_logs_arr(self):
        ''' getter '''
        return np.array(
            self.__logs_tuple_list,
        )

    def set_readonly(self, value):
        ''' setter '''
        raise TypeError()
    
    logs_arr = property(get_logs_arr, set_readonly)

    def get_encoder_decoder_controller(self):
        ''' getter '''
        return self.__encoder_decoder_controller
    
    encoder_decoder_controller = property(get_encoder_decoder_controller, set_readonly)

    def get_retrospective_encoder(self):
        ''' getter '''
        return self.__retrospective_encoder
    
    retrospective_encoder = property(get_retrospective_encoder, set_readonly)
