# -*- coding: utf-8 -*-
from logging import getLogger
import numpy as np
from pycomposer.inferable_pitch import InferablePitch

# LSTM Graph which is-a `Synapse`.
from pydbm.synapse.recurrenttemporalgraph.lstm_graph import LSTMGraph
from pydbm.synapse.recurrenttemporalgraph.lstm_graph import LSTMGraph as EncoderGraph
from pydbm.synapse.recurrenttemporalgraph.lstm_graph import LSTMGraph as DecoderGraph

# Loss function.
from pydbm.rnn.loss.mean_squared_error import MeanSquaredError
# Adam as a Loss function.
from pydbm.rnn.optimization.optparams.adam import Adam as EncoderAdam
from pydbm.rnn.optimization.optparams.adam import Adam as DecoderAdam
from pydbm.rnn.optimization.optparams.sgd import SGD as EncoderSGD
from pydbm.rnn.optimization.optparams.sgd import SGD as DecoderSGD

# Verification.
from pydbm.rnn.verification.verificate_function_approximation import VerificateFunctionApproximation

# LSTM model.
from pydbm.rnn.lstm_model import LSTMModel as Encoder
from pydbm.rnn.lstm_model import LSTMModel as Decoder
# Logistic Function as activation function.
from pydbm.activation.logistic_function import LogisticFunction
# Tanh Function as activation function.
from pydbm.activation.tanh_function import TanhFunction
# ReLu Function as activation function.
from pydbm.activation.relu_function import ReLuFunction
# Encoder/Decoder
from pydbm.rnn.encoder_decoder_controller import EncoderDecoderController


class EncoderDecoder(InferablePitch):
    '''
    Inference MIDI by Encoder/Decoder based on LSTM.
    '''
    
    def __init__(self, cycle_len=30):
        '''
        Init.
        
        Args:
            cycle_len:    Length of one cycle.
        '''
        self.__cycle_len = cycle_len
        logger = getLogger("pycomposer")
        self.__logger = logger

    def inferance(self, midi_arr):
        '''
        Inferance and select next pitch of `pre_pitch` from the values of `midi_arr`.
        
        Args:
            midi_arr:    `np.ndarray` of pitch.
        
        Returns:
            `np.ndarray` of pitch.
        '''
        midi_arr[:, 0] = (midi_arr[:, 0] - self.__min_pitch) / (self.__max_pitch - self.__min_pitch)
        midi_arr[:, 1] = (midi_arr[:, 1] - self.__min_start) / (self.__max_start - self.__min_start)
        midi_arr[:, 2] = (midi_arr[:, 2] - self.__min_duration) / (self.__max_duration - self.__min_duration)
        midi_arr[:, 3] = (midi_arr[:, 3] - self.__min_velocity) / (self.__max_velocity - self.__min_velocity)
        midi_arr *= 0.1

        observed_arr, target_arr = self.__setup_dataset(midi_arr, self.__cycle_len)
        midi_arr = self.__controller.inference(observed_arr)[:, -1, :]
        #midi_arr = self.__controller.get_feature_points()
        
        midi_arr *= 10.0
        midi_arr[:, 0] = ((self.__max_pitch - self.__min_pitch) * (midi_arr[:, 0] - midi_arr[:, 0].min()) / (midi_arr[:, 0].max() - midi_arr[:, 0].min())) + self.__min_pitch

        midi_arr[:, 1] = ((self.__max_start - self.__min_start) * (midi_arr[:, 1] - midi_arr[:, 1].min()) / (midi_arr[:, 1].max() - midi_arr[:, 1].min())) + self.__min_start

        midi_arr[:, 2] = ((self.__max_duration - self.__min_duration) * (midi_arr[:, 2] - midi_arr[:, 2].min()) / (midi_arr[:, 2].max() - midi_arr[:, 2].min())) + self.__min_duration

        midi_arr[:, 3] = ((self.__max_velocity - self.__min_velocity) * (midi_arr[:, 3] - midi_arr[:, 3].min()) / (midi_arr[:, 3].max() - midi_arr[:, 3].min())) + self.__min_velocity
        return midi_arr

    def learn(
        self,
        midi_arr,
        hidden_neuron_count=200,
        epochs=1000,
        batch_size=50,
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
            midi_arr:                      `np.ndarray` of MIDI file.
                                            The shape is (`pitch`, `duration`, `velocity`).

            cycle_len:                      One cycle length.
            hidden_neuron_count:            The number of units in hidden layer.
            epochs:                         Epochs of Mini-batch.
            bath_size:                      Batch size of Mini-batch.
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
        # (`pitch`, `duration`, `velocity`)
        self.__min_pitch = midi_arr[:, 0].min()
        self.__min_start = midi_arr[:, 1].min()
        self.__min_duration = midi_arr[:, 2].min()
        self.__min_velocity = midi_arr[:, 3].min()

        self.__max_pitch = midi_arr[:, 0].max()
        self.__max_start = midi_arr[:, 1].max()
        self.__max_duration = midi_arr[:, 2].max()
        self.__max_velocity = midi_arr[:, 3].max()

        midi_arr[:, 0] = (midi_arr[:, 0] - self.__min_pitch) / (self.__max_pitch - self.__min_pitch)
        midi_arr[:, 1] = (midi_arr[:, 1] - self.__min_start) / (self.__max_start - self.__min_start)
        midi_arr[:, 2] = (midi_arr[:, 2] - self.__min_duration) / (self.__max_duration - self.__min_duration)
        midi_arr[:, 3] = (midi_arr[:, 3] - self.__min_velocity) / (self.__max_velocity - self.__min_velocity)
        midi_arr *= 0.1

        observed_arr, target_arr = self.__setup_dataset(midi_arr, self.__cycle_len)

        self.__logger.debug("Value of observed data points:")
        self.__logger.debug(observed_arr[0, :5, :])
        
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
            output_neuron_count=observed_arr.shape[-1]
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
            hidden_neuron_count=observed_arr.shape[-1],
            output_neuron_count=hidden_neuron_count
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
        encoder_decoder_controller.learn(observed_arr, target_arr)

        self.__controller = encoder_decoder_controller

    def __setup_dataset(self, midi_arr, cycle_len):
        observed_arr_list = []
        for i in range(midi_arr.shape[0] - cycle_len - 1):
            cycle_list = [None] * cycle_len
            for j in range(cycle_len):
                cycle_list[j] = midi_arr[i + j]
            observed_arr_list.append(cycle_list)
        observed_arr = np.array(observed_arr_list)
        target_arr = observed_arr[1:]
        observed_arr = observed_arr[:-1]
        
        self.__logger.debug("The shape of observed data points:")
        self.__logger.debug(observed_arr.shape)
        return (observed_arr, target_arr)

    def get_controller(self):
        ''' getter '''
        return self.__controller

    def set_readonly(self, value):
        ''' setter '''
        raise TypeError("This property must be read-only.")
    
    controller = property(get_controller, set_readonly)
