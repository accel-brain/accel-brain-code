# -*- coding: utf-8 -*-
from logging import getLogger
import numpy as np
import pandas as pd

from pycomposer.inferable_melody import InferableMelody

# LSTM Graph which is-a `Synapse`.
from pydbm.synapse.recurrenttemporalgraph.lstm_graph import LSTMGraph
from pydbm.synapse.recurrenttemporalgraph.lstm_graph import LSTMGraph as EncoderGraph
from pydbm.synapse.recurrenttemporalgraph.lstm_graph import LSTMGraph as DecoderGraph

# Loss function.
from pydbm.loss.mean_squared_error import MeanSquaredError
from pydbm.loss.cross_entropy import CrossEntropy
# Adam as a Loss function.
from pydbm.optimization.optparams.adam import Adam as EncoderAdam
from pydbm.optimization.optparams.adam import Adam as DecoderAdam
from pydbm.optimization.optparams.sgd import SGD as EncoderSGD
from pydbm.optimization.optparams.sgd import SGD as DecoderSGD

# Verification.
from pydbm.verification.verificate_function_approximation import VerificateFunctionApproximation

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


class EncoderDecoder(InferableMelody):
    '''
    Inference melody by Encoder/Decoder based on LSTM.
    '''
    
    def __init__(
        self,
        cycle_len=30,
        time_step_range=1.0
    ):
        '''
        Init.
        
        Args:
            time_step_range:                The range of time step.
            cycle_len:                      One cycle length.

        '''
        logger = getLogger("pycomposer")
        self.__logger = logger
        self.__cycle_len = cycle_len
        self.__time_step_range = time_step_range
        self.__logistic_function = LogisticFunction(binary_flag=True)

    def inferance(self, midi_df, octave=6):
        '''
        Inferance next melody.
        
        Args:
            midi_df:    `pd.DataFrame` of MIDI file.
            octave:     Octave.
        
        Returns:
            `pd.DataFrame` of MIDI file.
        '''
        observed_arr = self.__setup_dataset(midi_df)
        midi_arr = self.__controller.inference(observed_arr)[-1, :, :]
        
        midi_arr = self.__logistic_function.activate(midi_arr)

        midi_tuple_list = []
        for i in range(midi_arr.shape[0]):
            index_tuple = np.where(midi_arr[i] == 1)
            for index in index_tuple[0]:
                pitch = index + (12 * octave)
                start = i * self.__time_step_range
                end = (i + 1) * self.__time_step_range
                velocity = 100
                midi_tuple_list.append((
                    pitch,
                    start,
                    end,
                    velocity
                ))
        melody_df = pd.DataFrame(midi_tuple_list, columns=["pitch", "start", "end", "velocity"])
        return melody_df

    def learn(
        self,
        midi_df,
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
            midi_df:                       `pd.DataFrame` of MIDI file.

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
        observed_arr = self.__setup_dataset(midi_df)
        observed_arr = np.nan_to_num(observed_arr)
        target_arr = observed_arr.copy()

        self.__logger.debug("Value of observed data points:")
        self.__logger.debug(observed_arr[0, :5, :])
        
        self.__logger.debug("Shape of observed data points:")
        self.__logger.debug(observed_arr.shape)

        # Init.
        encoder_graph = EncoderGraph()

        # Activation function in LSTM.
        encoder_graph.observed_activating_function = TanhFunction()
        encoder_graph.input_gate_activating_function = LogisticFunction()
        encoder_graph.forget_gate_activating_function = LogisticFunction()
        encoder_graph.output_gate_activating_function = LogisticFunction()
        encoder_graph.hidden_activating_function = TanhFunction()
        encoder_graph.output_activating_function = TanhFunction()

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
        decoder_graph.observed_activating_function = TanhFunction()
        decoder_graph.input_gate_activating_function = LogisticFunction()
        decoder_graph.forget_gate_activating_function = LogisticFunction()
        decoder_graph.output_gate_activating_function = LogisticFunction()
        decoder_graph.hidden_activating_function = LogisticFunction(binary_flag=True)
        decoder_graph.output_activating_function = LogisticFunction(binary_flag=True)

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
        encoder_decoder_controller.learn(observed_arr)

        self.__controller = encoder_decoder_controller

    def __setup_dataset(self, midi_df):
        observed_arr_list = []
        
        start_t = 0.0
        end_t = 0.0
        prog_flag = True
        arr_list = []
        while prog_flag:
            start_t += self.__time_step_range
            df = midi_df[midi_df.start >= start_t]
            if df.shape[0] == 0:
                prog_flag = False
                break

            end_t += self.__time_step_range + self.__time_step_range
            df = df[df.end <= end_t]
            
            if df.shape[0] == 0:
                continue

            arr = np.zeros(12)
            for pitch in df.pitch.drop_duplicates():
                arr[int(pitch % 12)] = 1
            arr_list.append(arr)

        arr = np.array(arr_list)
        for i in range(arr.shape[0] - self.__cycle_len - 1):
            cycle_list = [None] * self.__cycle_len
            for j in range(self.__cycle_len):
                cycle_list[j] = arr[i + j]
            observed_arr_list.append(cycle_list)
        observed_arr = np.array(observed_arr_list)
        observed_arr = observed_arr.astype(np.float64)
        
        self.__logger.debug("The shape of observed data points:")
        self.__logger.debug(observed_arr.shape)
        return observed_arr

    def get_controller(self):
        ''' getter '''
        return self.__controller

    def set_readonly(self, value):
        ''' setter '''
        raise TypeError("This property must be read-only.")
    
    controller = property(get_controller, set_readonly)
