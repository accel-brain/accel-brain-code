# -*- coding: utf-8 -*-
from logging import getLogger
import numpy as np

from pysummarization.abstractable_semantics import AbstractableSemantics
from pysummarization.vectorizable_token import VectorizableToken

# LSTM Graph which is-a `Synapse`.
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
from pydbm.rnn.lstm_model import LSTMModel
from pydbm.rnn.lstm_model import LSTMModel as Encoder
from pydbm.rnn.lstm_model import LSTMModel as Decoder
# Logistic Function as activation function.
from pydbm.activation.logistic_function import LogisticFunction
# Tanh Function as activation function.
from pydbm.activation.tanh_function import TanhFunction
# Softmax Function as activation function.
from pydbm.activation.softmax_function import SoftmaxFunction
# Encoder/Decoder
from pydbm.rnn.encoder_decoder_controller import EncoderDecoderController


class EncDecAD(AbstractableSemantics):
    '''
    LSTM-based Encoder/Decoder scheme for Anomaly Detection (EncDec-AD).

    This library applies the Encoder-Decoder scheme for Anomaly Detection (EncDec-AD)
    to text summarizations by intuition. In this scheme, LSTM-based Encoder/Decoder 
    or so-called the sequence-to-sequence(Seq2Seq) model learns to reconstruct normal time-series behavior,
    and thereafter uses reconstruction error to detect anomalies.
    
    Malhotra, P., et al. (2016) showed that EncDecAD paradigm is robust and 
    can detect anomalies from predictable, unpredictable, periodic, aperiodic, 
    and quasi-periodic time-series. Further, they showed that the paradigm is able to 
    detect anomalies from short time-series (length as small as 30) as well as long 
    time-series (length as large as 500).

    This library refers to the intuitive insight in relation to the use case of 
    reconstruction error to detect anomalies above to apply the model to text summarization.
    As exemplified by Seq2Seq paradigm, document and sentence which contain tokens of text
    can be considered as time-series features. The anomalies data detected by EncDec-AD 
    should have to express something about the text.

    From the above analogy, this library introduces two conflicting intuitions. On the one hand,
    the anomalies data may catch observer's eye from the viewpoints of rarity or amount of information
    as the indicator of natural language processing like TF-IDF shows. On the other hand,
    the anomalies data may be ignorable noise as mere outlier.
    
    In any case, this library deduces the function and potential of EncDec-AD in text summarization
    is to draw the distinction of normal and anomaly texts and is to filter the one from the other.

    Note that the model in this library and Malhotra, P., et al. (2016) are different in some respects
    from the relation with the specification of the Deep Learning library: [pydbm](https://github.com/chimera0/accel-brain-code/tree/master/Deep-Learning-by-means-of-Design-Pattern).
    First, weight matrix of encoder and decoder is not shered. Second, it is possible to 
    introduce regularization techniques which are not discussed in Malhotra, P., et al. (2016) 
    such as the dropout, the gradient clipping, and limitation of weights. 
    Third, the loss function for reconstruction error is not limited to the L2 norm.

    References:
        - Malhotra, P., Ramakrishnan, A., Anand, G., Vig, L., Agarwal, P., & Shroff, G. (2016). LSTM-based encoder-decoder for multi-sensor anomaly detection. arXiv preprint arXiv:1607.00148.
    '''

    # Logs of accuracy.
    __logs_tuple_list = []

    def __init__(
        self, 
        normal_prior_flag=False,
        encoder_decoder_controller=None,
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
        seq_len=8,
        bptt_tau=8,
        test_size_rate=0.3,
        tol=0.0,
        tld=100.0
    ):
        '''
        Init.

        Args:
            normal_prior_flag:              If `True`, this class will select abstract sentence
                                            from sentences with low reconstruction error.

            encoder_decoder_controller:     is-a `EncoderDecoderController`.
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
        self.__normal_prior_flag = normal_prior_flag
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

        self.__encoder_decoder_controller = encoder_decoder_controller

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
            input_neuron_count=input_neuron_count,
            hidden_neuron_count=hidden_neuron_count,
            output_neuron_count=1
        )
        encoder_opt_params = EncoderAdam()
        encoder_opt_params.weight_limit = weight_limit
        encoder_opt_params.dropout_rate = dropout_rate

        encoder = Encoder(
            # Delegate `graph` to `LSTMModel`.
            graph=encoder_graph,
            # The number of epochs in mini-batch training.
            epochs=100,
            # The batch size.
            batch_size=batch_size,
            # Learning rate.
            learning_rate=learning_rate,
            # Attenuate the `learning_rate` by a factor of this value every `attenuate_epoch`.
            learning_attenuate_rate=0.1,
            # Attenuate the `learning_rate` by a factor of `learning_attenuate_rate` every `attenuate_epoch`.
            attenuate_epoch=50,
            # Refereed maxinum step `t` in BPTT. If `0`, this class referes all past data in BPTT.
            bptt_tau=8,
            # Size of Test data set. If this value is `0`, the validation will not be executed.
            test_size_rate=0.3,
            # Loss function.
            computable_loss=MeanSquaredError(),
            # Optimizer.
            opt_params=encoder_opt_params,
            # Verification function.
            verificatable_result=VerificateFunctionApproximation()
        )

        # Init.
        decoder_graph = DecoderGraph()

        # Activation function in LSTM.
        decoder_graph.observed_activating_function = LogisticFunction()
        decoder_graph.input_gate_activating_function = LogisticFunction()
        decoder_graph.forget_gate_activating_function = LogisticFunction()
        decoder_graph.output_gate_activating_function = LogisticFunction()
        decoder_graph.hidden_activating_function = LogisticFunction()
        decoder_graph.output_activating_function = SoftmaxFunction()

        # Initialization strategy.
        # This method initialize each weight matrices and biases in Gaussian distribution: `np.random.normal(size=hoge) * 0.01`.
        decoder_graph.create_rnn_cells(
            input_neuron_count=hidden_neuron_count,
            hidden_neuron_count=hidden_neuron_count,
            output_neuron_count=input_neuron_count
        )
        decoder_opt_params = DecoderAdam()
        decoder_opt_params.weight_limit = weight_limit
        decoder_opt_params.dropout_rate = dropout_rate

        decoder = Decoder(
            # Delegate `graph` to `LSTMModel`.
            graph=decoder_graph,
            # The number of epochs in mini-batch training.
            epochs=100,
            # The batch size.
            batch_size=batch_size,
            # Learning rate.
            learning_rate=learning_rate,
            # Attenuate the `learning_rate` by a factor of this value every `attenuate_epoch`.
            learning_attenuate_rate=0.1,
            # Attenuate the `learning_rate` by a factor of `learning_attenuate_rate` every `attenuate_epoch`.
            attenuate_epoch=50,
            # The length of sequences.
            seq_len=seq_len,
            # Refereed maxinum step `t` in BPTT. If `0`, this class referes all past data in BPTT.
            bptt_tau=bptt_tau,
            # Size of Test data set. If this value is `0`, the validation will not be executed.
            test_size_rate=0.3,
            # Loss function.
            computable_loss=MeanSquaredError(),
            # Optimizer.
            opt_params=decoder_opt_params,
            # Verification function.
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

    def learn(self, observed_arr, target_arr):
        '''
        Training the model.

        Args:
            observed_arr:       `np.ndarray` of observed data points.
            target_arr:         `np.ndarray` of target labeled data.
        '''
        self.__encoder_decoder_controller.learn(observed_arr, observed_arr)

    def inference(self, observed_arr):
        '''
        Infernece by the model.

        Args:
            observed_arr:       `np.ndarray` of observed data points.

        Returns:
            `np.ndarray` of inferenced feature points.
        '''
        return self.__encoder_decoder_controller.inference(observed_arr)

    def summarize(self, test_arr, vectorizable_token, sentence_list, limit=5):
        '''
        Summarize input document.

        Args:
            test_arr:               `np.ndarray` of observed data points..
            vectorizable_token:     is-a `VectorizableToken`.
            sentence_list:          `list` of all sentences.
            limit:                  The number of selected abstract sentence.
        
        Returns:
            `np.ndarray` of scores.
        '''
        if isinstance(vectorizable_token, VectorizableToken) is False:
            raise TypeError()

        _ = self.inference(test_arr)
        score_arr = self.__encoder_decoder_controller.get_reconstruction_error()
        score_arr = score_arr.reshape((
            score_arr.shape[0],
            -1
        )).mean(axis=1)

        score_list = score_arr.tolist()

        abstract_list = []
        for i in range(limit):
            if self.__normal_prior_flag is True:
                key = score_arr.argmin()
            else:
                key = score_arr.argmax()

            score = score_list.pop(key)
            score_arr = np.array(score_list)

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

    def set_readonly(self, value):
        ''' setter '''
        raise TypeError()

    def get_encoder_decoder_controller(self):
        ''' getter '''
        return self.__encoder_decoder_controller

    encoder_decoder_controller = property(get_encoder_decoder_controller, set_readonly)
