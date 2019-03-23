# -*- coding: utf-8 -*-
from logging import getLogger
import numpy as np
cimport numpy as np
from copy import copy

# The interfaces for type-hinting.
from pydbm.optimization.opt_params import OptParams
from pydbm.loss.interface.computable_loss import ComputableLoss
from pydbm.verification.interface.verificatable_result import VerificatableResult
from pydbm.activation.interface.activating_function_interface import ActivatingFunctionInterface


# Adam as a optimizer.
from pydbm.optimization.optparams.adam import Adam
# Logistic Function as activation function.
from pydbm.activation.logistic_function import LogisticFunction
# Tanh Function as activation function.
from pydbm.activation.tanh_function import TanhFunction
# ReLu Function as activation function.
from pydbm.activation.relu_function import ReLuFunction

# The function of reconsturction error.
from pydbm.loss.mean_squared_error import MeanSquaredError

# Verification.
from pydbm.verification.verificate_function_approximation import VerificateFunctionApproximation
from pydbm.verification.verificate_softmax import VerificateSoftmax

# LSTM Graph which is-a `Synapse`.
from pydbm.synapse.recurrenttemporalgraph.lstm_graph import LSTMGraph as EncoderGraph
from pydbm.synapse.recurrenttemporalgraph.lstm_graph import LSTMGraph as DecoderGraph

# LSTM model.
from pydbm.rnn.lstm_model import LSTMModel as Encoder
from pydbm.rnn.lstm_model import LSTMModel as Decoder

# Encoder/Decoder
from pydbm.rnn.encoder_decoder_controller import EncoderDecoderController

# not logging but logger.
from logging import getLogger, StreamHandler, NullHandler, DEBUG, ERROR


class FacadeEncoderDecoder(object):
    '''
    `Facade` for casual user of Encoder/Decoder based on LSTM networks.

    This library provides Encoder/Decoder based on LSTM, 
    which is a reconstruction model and makes it possible to extract 
    series features embedded in deeper layers. The LSTM encoder learns 
    a fixed length vector of time-series observed data points and the 
    LSTM decoder uses this representation to reconstruct the time-series 
    using the current hidden state and the value inferenced at the previous time-step.
    
    One interesting application example is the Encoder/Decoder for 
    Anomaly Detection (EncDec-AD) paradigm (Malhotra, P., et al. 2016).
    This reconstruction model learns to reconstruct normal time-series behavior, 
    and thereafter uses reconstruction error to detect anomalies. 
    Malhotra, P., et al. (2016) showed that EncDec-AD paradigm is robust 
    and can detect anomalies from predictable, unpredictable, periodic, aperiodic, 
    and quasi-periodic time-series. Further, they showed that the paradigm is able 
    to detect anomalies from short time-series (length as small as 30) as well as 
    long time-series (length as large as 500).

    References:
        - https://github.com/chimera0/accel-brain-code/blob/master/Deep-Learning-by-means-of-Design-Pattern/demo/demo_sine_wave_prediction_by_LSTM_encoder_decoder.ipynb
        - https://github.com/chimera0/accel-brain-code/blob/master/Deep-Learning-by-means-of-Design-Pattern/demo/demo_anomaly_detection_by_enc_dec_ad.ipynb
        - Cho, K., Van Merriënboer, B., Gulcehre, C., Bahdanau, D., Bougares, F., Schwenk, H., & Bengio, Y. (2014). Learning phrase representations using RNN encoder-decoder for statistical machine translation. arXiv preprint arXiv:1406.1078.
        - Malhotra, P., Ramakrishnan, A., Anand, G., Vig, L., Agarwal, P., & Shroff, G. (2016). LSTM-based encoder-decoder for multi-sensor anomaly detection. arXiv preprint arXiv:1607.00148.

    '''
    
    def __init__(
        self,
        input_neuron_count,
        hidden_neuron_count=200,
        epochs=200,
        batch_size=20,
        learning_rate=1e-05,
        learning_attenuate_rate=0.1,
        attenuate_epoch=50,
        hidden_activating_function=None,
        output_activating_function=None,
        computable_loss=None,
        opt_params=None,
        seq_len=8,
        bptt_tau=8,
        test_size_rate=0.3,
        tol=0.0,
        tld=1.0,
        verificatable_result=None,
        encoder_pre_learned_file_path=None,
        decoder_pre_learned_file_path=None,
        verbose_flag=False
    ):
        '''
        Init.
        
        Args:
            input_neuron_count:             The number of units in input layers.
            hidden_neuron_count:            The number of units in hidden layers.
            epochs:                         Epochs of Mini-batch.
            bath_size:                      Batch size of Mini-batch.
            learning_rate:                  Learning rate.
            learning_attenuate_rate:        Attenuate the `learning_rate` by a factor of this value every `attenuate_epoch`.
            attenuate_epoch:                Attenuate the `learning_rate` by a factor of `learning_attenuate_rate` every `attenuate_epoch`.
                                            Additionally, in relation to regularization,
                                            this class constrains weight matrixes every `attenuate_epoch`.

            hidden_activating_function:     Activation function in hidden layers.
            output_activating_function:     Activation function in output layers.

            computable_loss:                Loss function.
            opt_params:                     Optimizer.

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

            verificatable_result:           Verification function.
            encoder_pre_learned_file_path:  File path that stored Encoder's pre-learned parameters.
            decoder_pre_learned_file_path:  File path that stored Decoder's pre-learned parameters.
            tol:                            Tolerance for the optimization.
                                            When the loss or score is not improving by at least tol 
                                            for two consecutive iterations, convergence is considered 
                                            to be reached and training stops.

            tld:                            Tolerance for deviation of loss.

        '''
        logger = getLogger("pydbm")
        handler = StreamHandler()
        if verbose_flag is True:
            handler.setLevel(DEBUG)
            logger.setLevel(DEBUG)
        else:
            handler.setLevel(ERROR)
            logger.setLevel(ERROR)
        logger.addHandler(handler)

        if computable_loss is None:
            computable_loss = MeanSquaredError()
        else:
            if isinstance(computable_loss, ComputableLoss) is False:
                raise TypeError()

        if opt_params is None:
            encoder_opt_params = Adam()
            decoder_opt_params = Adam()
        else:
            if isinstance(opt_params, OptParams) is False:
                raise TypeError()

            encoder_opt_params = opt_params
            decoder_opt_params = copy(opt_params)

        if verificatable_result is None:
            verificatable_result = VerificateFunctionApproximation()
        else:
            if isinstance(verificatable_result, VerificatableResult) is False:
                raise TypeError()

        # Init.
        encoder_graph = EncoderGraph()

        # Activation function in LSTM.
        encoder_graph.observed_activating_function = TanhFunction()
        encoder_graph.input_gate_activating_function = LogisticFunction()
        encoder_graph.forget_gate_activating_function = LogisticFunction()
        encoder_graph.output_gate_activating_function = LogisticFunction()
        if hidden_activating_function is None:
            encoder_graph.hidden_activating_function = TanhFunction()
        else:
            if isinstance(hidden_activating_function, ActivatingFunctionInterface) is False:
                raise TypeError()

            encoder_graph.hidden_activating_function = hidden_activating_function
        if output_activating_function is None:
            encoder_graph.output_activating_function = LogisticFunction()
        else:
            if isinstance(output_activating_function, ActivatingFunctionInterface) is False:
                raise TypeError()

            encoder_graph.output_activating_function = output_activating_function

        # Initialization strategy.
        # This method initialize each weight matrices and biases in Gaussian distribution: `np.random.normal(size=hoge) * 0.01`.
        encoder_graph.create_rnn_cells(
            input_neuron_count=input_neuron_count,
            hidden_neuron_count=hidden_neuron_count,
            output_neuron_count=1
        )
        
        if encoder_pre_learned_file_path is not None:
            encoder_graph.load_pre_learned_params(encoder_pre_learned_file_path)

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
            computable_loss=computable_loss,
            # Optimizer.
            opt_params=encoder_opt_params,
            # Verification function.
            verificatable_result=verificatable_result,
            # Tolerance for the optimization.
            # When the loss or score is not improving by at least tol 
            # for two consecutive iterations, convergence is considered 
            # to be reached and training stops.
            tol=tol
        )

        # Init.
        decoder_graph = DecoderGraph()

        # Activation function in LSTM.
        decoder_graph.observed_activating_function = TanhFunction()
        decoder_graph.input_gate_activating_function = LogisticFunction()
        decoder_graph.forget_gate_activating_function = LogisticFunction()
        decoder_graph.output_gate_activating_function = LogisticFunction()
        if hidden_activating_function is None:
            decoder_graph.hidden_activating_function = TanhFunction()
        else:
            if isinstance(hidden_activating_function, ActivatingFunctionInterface) is False:
                raise TypeError()

            decoder_graph.hidden_activating_function = hidden_activating_function
        if output_activating_function is None:
            decoder_graph.output_activating_function = LogisticFunction()
        else:
            if isinstance(output_activating_function, ActivatingFunctionInterface) is False:
                raise TypeError()

            decoder_graph.output_activating_function = output_activating_function

        # Initialization strategy.
        # This method initialize each weight matrices and biases in Gaussian distribution: `np.random.normal(size=hoge) * 0.01`.
        decoder_graph.create_rnn_cells(
            input_neuron_count=hidden_neuron_count,
            hidden_neuron_count=hidden_neuron_count,
            output_neuron_count=input_neuron_count
        )

        if decoder_pre_learned_file_path is not None:
            decoder_graph.load_pre_learned_params(decoder_pre_learned_file_path)

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
            # This means refereed maxinum step `t` in feedforward.
            seq_len=seq_len,
            # Refereed maxinum step `t` in BPTT. If `0`, this class referes all past data in BPTT.
            bptt_tau=bptt_tau,
            # Size of Test data set. If this value is `0`, the validation will not be executed.
            test_size_rate=test_size_rate,
            # Loss function.
            computable_loss=computable_loss,
            # Optimizer.
            opt_params=decoder_opt_params,
            # Verification function.
            verificatable_result=verificatable_result,
            # Tolerance for the optimization.
            # When the loss or score is not improving by at least tol 
            # for two consecutive iterations, convergence is considered 
            # to be reached and training stops.
            tol=tol
        )

        encoder_decoder_controller = EncoderDecoderController(
            # is-a LSTM model.
            encoder=encoder,
            # is-a LSTM model.
            decoder=decoder,
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
            # Size of Test data set. If this value is `0`, the validation will not be executed.
            test_size_rate=test_size_rate,
            # Loss function.
            computable_loss=computable_loss,
            # Verification function.
            verificatable_result=verificatable_result,
            # Tolerance for the optimization.
            # When the loss or score is not improving by at least tol 
            # for two consecutive iterations, convergence is considered 
            # to be reached and training stops.
            tol=tol,
            tld=tld
        )
        self.__encoder_decoder_controller = encoder_decoder_controller

    def learn(self, observed_arr, target_arr):
        '''
        Learn the observed data points
        for vector representation of the input time-series.

        Args:
            observed_arr:    Array like or sparse matrix as the observed data ponts.
            target_arr:      Array like or sparse matrix as the target data points.
                             To learn as Auto-encoder, this value must be `None` or equivalent to `observed_arr`.
        '''
        # Learning.
        self.__encoder_decoder_controller.learn(observed_arr, target_arr)

    def infernece(self, test_arr):
        '''
        Inference the feature points to reconstruct the time-series.

        Args:
            observed_arr:           Array like or sparse matrix as the observed data ponts.
            hidden_activity_arr:    Array like or sparse matrix as the state in hidden layer.
            cec_activity_arr:       Array like or sparse matrix as the state in RNN.

        Returns:
            Tuple data.
            - Array like or sparse matrix of reconstructed instances of time-series,
            - Array like or sparse matrix of the state in hidden layer,
            - Array like or sparse matrix of the state in RNN.
        '''
        # Execute recursive learning.
        inferenced_arr = self.__encoder_decoder_controller.inference(test_arr)
        return inferenced_arr

    def get_feature_points(self):
        '''
        Extract the activities in hidden layer and reset it, 
        considering this method will be called per one cycle in instances of time-series.

        Returns:
            The array like or sparse matrix of feature points.
        '''
        return self.__encoder_decoder_controller.get_feature_points()

    def get_reconstruction_error(self):
        '''
        Extract the reconstructed error in inferencing.
        
        Returns:
            The array like or sparse matrix of reconstruction error. 
        '''
        return self.__encoder_decoder_controller.get_reconstruction_error()

    def save_pre_learned_params(self, encoder_file_path, decoder_file_path):
        '''
        Save pre-learned parameters.
        
        Args:
            encoder_file_path:      File path.
            decoder_file_path:      File path.

        '''
        self.__encoder_decoder_controller.encoder.graph.save_pre_learned_params(encoder_file_path)
        self.__encoder_decoder_controller.decoder.graph.save_pre_learned_params(decoder_file_path)
