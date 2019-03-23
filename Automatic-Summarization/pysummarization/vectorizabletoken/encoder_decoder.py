# -*- coding: utf-8 -*-
from logging import getLogger
import numpy as np
from pysummarization.vectorizable_token import VectorizableToken

# LSTM Graph which is-a `Synapse`.
from pydbm.synapse.recurrenttemporalgraph.lstm_graph import LSTMGraph
from pydbm.synapse.recurrenttemporalgraph.lstm_graph import LSTMGraph as EncoderGraph
from pydbm.synapse.recurrenttemporalgraph.lstm_graph import LSTMGraph as DecoderGraph

# Loss function.
from pydbm.loss.mean_squared_error import MeanSquaredError
# Adam as a Loss function.
from pydbm.optimization.optparams.adam import Adam as EncoderAdam
from pydbm.optimization.optparams.adam import Adam as DecoderAdam
# Verification.
from pydbm.verification.verificate_function_approximation import VerificateFunctionApproximation
# LSTM model.
from pydbm.rnn.lstm_model import LSTMModel as Encoder
from pydbm.rnn.lstm_model import LSTMModel as Decoder
# Logistic Function as activation function.
from pydbm.activation.logistic_function import LogisticFunction
# Tanh Function as activation function.
from pydbm.activation.tanh_function import TanhFunction
# Encoder/Decoder
from pydbm.rnn.encoder_decoder_controller import EncoderDecoderController


class EncoderDecoder(VectorizableToken):
    '''
    Vectorize tokens by Encoder/Decoder based on LSTM.

    This library provides Encoder/Decoder based on LSTM, 
    which is a reconstruction model and makes it possible to 
    extract series features embedded in deeper layers. 
    The LSTM encoder learns a fixed length vector of time-series 
    observed data points and the LSTM decoder uses this representation 
    to reconstruct the time-series using the current hidden state 
    and the value inferenced at the previous time-step.

    References:
        - https://github.com/chimera0/accel-brain-code/blob/master/Deep-Learning-by-means-of-Design-Pattern/demo/demo_sine_wave_prediction_by_LSTM_encoder_decoder.ipynb
        - https://github.com/chimera0/accel-brain-code/blob/master/Deep-Learning-by-means-of-Design-Pattern/demo/demo_anomaly_detection_by_enc_dec_ad.ipynb
        - Cho, K., Van Merriënboer, B., Gulcehre, C., Bahdanau, D., Bougares, F., Schwenk, H., & Bengio, Y. (2014). Learning phrase representations using RNN encoder-decoder for statistical machine translation. arXiv preprint arXiv:1406.1078.
        - Malhotra, P., Ramakrishnan, A., Anand, G., Vig, L., Agarwal, P., & Shroff, G. (2016). LSTM-based encoder-decoder for multi-sensor anomaly detection. arXiv preprint arXiv:1607.00148.

    '''
    
    def __init__(self):
        ''' Init. '''
        logger = getLogger("pysummarization")
        self.__logger = logger

    def vectorize(self, token_list):
        '''
        Tokenize token list.
        
        Args:
            token_list:   The list of tokens..
        
        Returns:
            [vector of token, vector of token, vector of token, ...]
        '''
        sentence_list = [token_list]
        test_observed_arr = self.__setup_dataset(sentence_list, self.__token_master_list)
        pred_arr = self.__controller.inference(test_observed_arr)
        return self.__controller.get_feature_points()

    def learn(
        self,
        sentence_list,
        token_master_list,        
        hidden_neuron_count=200,
        epochs=100,
        batch_size=100,
        learning_rate=1e-05,
        learning_attenuate_rate=0.1,
        attenuate_epoch=50,
        bptt_tau=8,
        weight_limit=0.5,
        dropout_rate=0.5,
        test_size_rate=0.3
    ):
        '''
        Init.
        
        Args:
            sentence_list:                  The list of tokenized sentences.
                                            [[`token`, `token`, `token`, ...],
                                            [`token`, `token`, `token`, ...],
                                            [`token`, `token`, `token`, ...]]

            token_master_list:              Unique `list` of tokens.
            hidden_neuron_count:            The number of units in hidden layer.
            epochs:                         Epochs of Mini-batch.
            batch_size:                     Batch size of Mini-batch.
            learning_rate:                  Learning rate.
            learning_attenuate_rate:        Attenuate the `learning_rate` by a factor of this value every `attenuate_epoch`.
            attenuate_epoch:                Attenuate the `learning_rate` by a factor of `learning_attenuate_rate` every `attenuate_epoch`.
                                            Additionally, in relation to regularization,
                                            this class constrains weight matrixes every `attenuate_epoch`.

            bptt_tau:                       Refereed maxinum step `t` in Backpropagation Through Time(BPTT).
            weight_limit:                   Regularization for weights matrix
                                            to repeat multiplying the weights matrix and `0.9`
                                            until $\sum_{j=0}^{n}w_{ji}^2 < weight\_limit$.

            dropout_rate:                   The probability of dropout.
            test_size_rate:                 Size of Test data set. If this value is `0`, the 
        '''
        observed_arr = self.__setup_dataset(sentence_list, token_master_list)
        
        self.__logger.debug("Shape of observed data points:")
        self.__logger.debug(observed_arr.shape)

        # Init.
        encoder_graph = EncoderGraph()

        # Activation function in LSTM.
        encoder_graph.observed_activating_function = LogisticFunction()
        encoder_graph.input_gate_activating_function = LogisticFunction()
        encoder_graph.forget_gate_activating_function = LogisticFunction()
        encoder_graph.output_gate_activating_function = LogisticFunction()
        encoder_graph.hidden_activating_function = LogisticFunction()
        encoder_graph.output_activating_function = LogisticFunction()

        # Initialization strategy.
        # This method initialize each weight matrices and biases in Gaussian distribution: `np.random.normal(size=hoge) * 0.01`.
        encoder_graph.create_rnn_cells(
            input_neuron_count=observed_arr.shape[-1],
            hidden_neuron_count=hidden_neuron_count,
            output_neuron_count=1
        )

        # Init.
        decoder_graph = DecoderGraph()

        # Activation function in LSTM.
        decoder_graph.observed_activating_function = LogisticFunction()
        decoder_graph.input_gate_activating_function = LogisticFunction()
        decoder_graph.forget_gate_activating_function = LogisticFunction()
        decoder_graph.output_gate_activating_function = LogisticFunction()
        decoder_graph.hidden_activating_function = LogisticFunction()
        decoder_graph.output_activating_function = LogisticFunction()

        # Initialization strategy.
        # This method initialize each weight matrices and biases in Gaussian distribution: `np.random.normal(size=hoge) * 0.01`.
        decoder_graph.create_rnn_cells(
            input_neuron_count=hidden_neuron_count,
            hidden_neuron_count=hidden_neuron_count,
            output_neuron_count=observed_arr.shape[-1]
        )

        encoder_opt_params = EncoderAdam()
        encoder_opt_params.weight_limit = weight_limit
        encoder_opt_params.dropout_rate = dropout_rate

        encoder = Encoder(
            # Delegate `graph` to `LSTMModel`.
            graph=encoder_graph,
            # The number of epochs in mini-batch training.
            epochs=epochs,
            # The batch size.
            batch_size=batch_size,
            # Learning rate.
            learning_rate=learning_rate,
            # Attenuate the `learning_rate` by a factor of this value every `attenuate_epoch`.
            learning_attenuate_rate=learning_attenuate_rate,
            # Attenuate the `learning_rate` by a factor of `learning_attenuate_rate` every `attenuate_epoch`.
            attenuate_epoch=attenuate_epoch,
            # Refereed maxinum step `t` in BPTT. If `0`, this class referes all past data in BPTT.
            bptt_tau=bptt_tau,
            # Size of Test data set. If this value is `0`, the validation will not be executed.
            test_size_rate=test_size_rate,
            # Loss function.
            computable_loss=MeanSquaredError(),
            # Optimizer.
            opt_params=encoder_opt_params,
            # Verification function.
            verificatable_result=VerificateFunctionApproximation(),
            tol=0.0
        )

        decoder_opt_params = DecoderAdam()
        decoder_opt_params.weight_limit = weight_limit
        decoder_opt_params.dropout_rate = dropout_rate

        decoder = Decoder(
            # Delegate `graph` to `LSTMModel`.
            graph=decoder_graph,
            # The number of epochs in mini-batch training.
            epochs=epochs,
            # The batch size.
            batch_size=batch_size,
            # Learning rate.
            learning_rate=learning_rate,
            # Attenuate the `learning_rate` by a factor of this value every `attenuate_epoch`.
            learning_attenuate_rate=learning_attenuate_rate,
            # Attenuate the `learning_rate` by a factor of `learning_attenuate_rate` every `attenuate_epoch`.
            attenuate_epoch=attenuate_epoch,
            # The length of sequences.
            seq_len=observed_arr.shape[1],
            # Refereed maxinum step `t` in BPTT. If `0`, this class referes all past data in BPTT.
            bptt_tau=bptt_tau,
            # Size of Test data set. If this value is `0`, the validation will not be executed.
            test_size_rate=test_size_rate,
            # Loss function.
            computable_loss=MeanSquaredError(),
            # Optimizer.
            opt_params=decoder_opt_params,
            # Verification function.
            verificatable_result=VerificateFunctionApproximation(),
            tol=0.0
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
            tol=0.0
        )

        # Learning.
        encoder_decoder_controller.learn(observed_arr, observed_arr)
        
        self.__controller = encoder_decoder_controller
        self.__token_master_list = token_master_list

    def __setup_dataset(self, sentence_list, token_master_list):
        sentence_len_list = [0] * len(sentence_list)
        for i in range(len(sentence_list)):
            sentence_len_list[i] = len(sentence_list[i])
        sentence_mean_len = int(sum(sentence_len_list) / len(sentence_len_list))

        observed_list = [None] * len(sentence_list)
        for i in range(len(sentence_list)):
            arr_list = [None] * sentence_mean_len
            for j in range(sentence_mean_len):
                arr = np.zeros(len(token_master_list))
                try:
                    token = sentence_list[i][j]
                    arr[token_master_list.index(token)] = 1
                except IndexError:
                    pass
                finally:
                    arr = arr.astype(np.float64)
                    arr_list[j] = arr
            observed_list[i] = arr_list
        observed_arr = np.array(observed_list)
        return observed_arr

    def get_controller(self):
        ''' getter '''
        return self.__controller

    def set_readonly(self, value):
        ''' setter '''
        raise TypeError("This property must be read-only.")
    
    controller = property(get_controller, set_readonly)
