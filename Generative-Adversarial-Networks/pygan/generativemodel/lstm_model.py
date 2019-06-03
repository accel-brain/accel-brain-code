# -*- coding: utf-8 -*-
import numpy as np
from logging import getLogger

from pygan.generative_model import GenerativeModel
from pygan.true_sampler import TrueSampler

# LSTM Graph which is-a `Synapse`.
from pydbm.synapse.recurrenttemporalgraph.lstm_graph import LSTMGraph
# Loss function.
from pydbm.loss.interface.computable_loss import ComputableLoss
from pydbm.loss.mean_squared_error import MeanSquaredError
# SGD.
from pydbm.optimization.optparams.sgd import SGD
# Verification.
from pydbm.verification.verificate_function_approximation import VerificateFunctionApproximation

from pydbm.activation.interface.activating_function_interface import ActivatingFunctionInterface
# Logistic Function as activation function.
from pydbm.activation.logistic_function import LogisticFunction
# Tanh Function as activation function.
from pydbm.activation.tanh_function import TanhFunction

from pydbm.rnn.lstm_model import LSTMModel as LSTM


class LSTMModel(GenerativeModel):
    '''
    LSTM as a Generator.

    Originally, Long Short-Term Memory(LSTM) networks as a 
    special RNN structure has proven stable and powerful for 
    modeling long-range dependencies.

    The Key point of structural expansion is its memory cell 
    which essentially acts as an accumulator of the state information. 
    Every time observed data points are given as new information and input 
    to LSTM’s input gate, its information will be accumulated to the cell 
    if the input gate is activated. The past state of cell could be forgotten 
    in this process if LSTM’s forget gate is on. Whether the latest cell output 
    will be propagated to the final state is further controlled by the output gate.

    References:
        - Cho, K., Van Merriënboer, B., Gulcehre, C., Bahdanau, D., Bougares, F., Schwenk, H., & Bengio, Y. (2014). Learning phrase representations using RNN encoder-decoder for statistical machine translation. arXiv preprint arXiv:1406.1078.
        - Malhotra, P., Ramakrishnan, A., Anand, G., Vig, L., Agarwal, P., & Shroff, G. (2016). LSTM-based encoder-decoder for multi-sensor anomaly detection. arXiv preprint arXiv:1607.00148.
        - Zaremba, W., Sutskever, I., & Vinyals, O. (2014). Recurrent neural network regularization. arXiv preprint arXiv:1409.2329.

    '''

    def __init__(
        self,
        lstm_model=None,
        computable_loss=None,
        batch_size=20,
        input_neuron_count=100,
        hidden_neuron_count=300,
        observed_activating_function=None,
        input_gate_activating_function=None,
        forget_gate_activating_function=None,
        output_gate_activating_function=None,
        hidden_activating_function=None,
        output_activating_function=None,
        seq_len=10,
        join_io_flag=False,
        learning_rate=1e-05
    ):
        '''
        Init.

        Args:
            lstm_model:                         is-a `lstm_model`.
            computable_loss:                    is-a `ComputableLoss`.

            batch_size:                         Batch size.
                                                This parameters will be refered only when `lstm_model` is `None`.

            input_neuron_count:                 The number of input units.
                                                This parameters will be refered only when `lstm_model` is `None`.

            hidden_neuron_count:                The number of hidden units.
                                                This parameters will be refered only when `lstm_model` is `None`.

            observed_activating_function:       is-a `ActivatingFunctionInterface` in hidden layer.
                                                This parameters will be refered only when `lstm_model` is `None`.
                                                If `None`, this value will be `TanhFunction`.

            input_gate_activating_function:     is-a `ActivatingFunctionInterface` in hidden layer.
                                                This parameters will be refered only when `lstm_model` is `None`.
                                                If `None`, this value will be `LogisticFunction`.

            forget_gate_activating_function:    is-a `ActivatingFunctionInterface` in hidden layer.
                                                This parameters will be refered only when `lstm_model` is `None`.
                                                If `None`, this value will be `LogisticFunction`.

            output_gate_activating_function:    is-a `ActivatingFunctionInterface` in hidden layer.
                                                This parameters will be refered only when `lstm_model` is `None`.
                                                If `None`, this value will be `LogisticFunction`.

            hidden_activating_function:         is-a `ActivatingFunctionInterface` in hidden layer.
                                                This parameters will be refered only when `lstm_model` is `None`.

            output_activating_function:         is-a `ActivatingFunctionInterface` in output layer.
                                                This parameters will be refered only when `lstm_model` is `None`.
                                                If `None`, this model outputs from LSTM's hidden layer in inferencing.

            seq_len:                            The length of sequences.
                                                This means refereed maxinum step `t` in feedforward.

            join_io_flag:                       If this value and value of `output_activating_function` is not `None`,
                                                This model outputs tensors combining observed data points and inferenced data
                                                in a series direction.

            learning_rate:                      Learning rate.
        '''
        if computable_loss is None:
            computable_loss = MeanSquaredError()

        if lstm_model is not None:
            if isinstance(lstm_model, LSTM) is False:
                raise TypeError()
        else:
            # Init.
            graph = LSTMGraph()

            # Activation function in LSTM.
            if observed_activating_function is None:
                graph.observed_activating_function = TanhFunction()
            else:
                if isinstance(observed_activating_function, ActivatingFunctionInterface) is False:
                    raise TypeError()
                graph.observed_activating_function = observed_activating_function
            
            if input_gate_activating_function is None:
                graph.input_gate_activating_function = LogisticFunction()
            else:
                if isinstance(input_gate_activating_function, ActivatingFunctionInterface) is False:
                    raise TypeError()
                graph.input_gate_activating_function = input_gate_activating_function
            
            if forget_gate_activating_function is None:
                graph.forget_gate_activating_function = LogisticFunction()
            else:
                if isinstance(forget_gate_activating_function, ActivatingFunctionInterface) is False:
                    raise TypeError()
                graph.forget_gate_activating_function = forget_gate_activating_function
            
            if output_gate_activating_function is None:
                graph.output_gate_activating_function = LogisticFunction()
            else:
                if isinstance(output_gate_activating_function, ActivatingFunctionInterface) is False:
                    raise TypeError()
                graph.output_gate_activating_function = output_gate_activating_function

            if hidden_activating_function is None:
                graph.hidden_activating_function = TanhFunction()
            else:
                if isinstance(hidden_activating_function, ActivatingFunctionInterface) is False:
                    raise TypeError()
                graph.hidden_activating_function = hidden_activating_function

            if output_activating_function is None:
                graph.output_activating_function = TanhFunction()
                self.__output_flag = False
                output_neuron_count = 1
            else:
                graph.output_activating_function = output_activating_function
                self.__output_flag = True
                output_neuron_count = hidden_neuron_count

            # Initialization strategy.
            # This method initialize each weight matrices and biases in Gaussian distribution: `np.random.normal(size=hoge) * 0.01`.
            graph.create_rnn_cells(
                input_neuron_count=input_neuron_count,
                hidden_neuron_count=hidden_neuron_count,
                output_neuron_count=output_neuron_count
            )

            opt_params = SGD()
            opt_params.weight_limit = 0.5
            opt_params.dropout_rate = 0.0

            lstm_model = LSTM(
                # Delegate `graph` to `LSTMModel`.
                graph=graph,
                # The number of epochs in mini-batch training.
                epochs=100,
                # The batch size.
                batch_size=batch_size,
                # Learning rate.
                learning_rate=1e-05,
                # Attenuate the `learning_rate` by a factor of this value every `attenuate_epoch`.
                learning_attenuate_rate=0.1,
                # Attenuate the `learning_rate` by a factor of `learning_attenuate_rate` every `attenuate_epoch`.
                attenuate_epoch=50,
                # The length of sequences.
                seq_len=seq_len,
                # Refereed maxinum step `t` in BPTT. If `0`, this class referes all past data in BPTT.
                bptt_tau=seq_len,
                # Size of Test data set. If this value is `0`, the validation will not be executed.
                test_size_rate=0.3,
                # Loss function.
                computable_loss=computable_loss,
                # Optimizer.
                opt_params=opt_params,
                # Verification function.
                verificatable_result=VerificateFunctionApproximation(),
                tol=0.0
            )

        self.__lstm_model = lstm_model
        self.__seq_len = seq_len
        self.__learning_rate = learning_rate
        self.__join_io_flag = join_io_flag
        self.__computable_loss = computable_loss
        self.__loss_list = []
        self.__epoch_counter = 0
        logger = getLogger("pygan")
        self.__logger = logger

    def draw(self):
        '''
        Draws samples from the `fake` distribution.

        Returns:
            `np.ndarray` of samples.
        '''
        observed_arr = self.noise_sampler.generate()
        arr = self.inference(observed_arr)
        return arr

    def inference(self, observed_arr):
        '''
        Draws samples from the `fake` distribution.

        Args:
            observed_arr:     `np.ndarray` of observed data points.
        
        Returns:
            `np.ndarray` of inferenced data.
        '''
        inferenced_arr = self.__lstm_model.inference(observed_arr)
        if self.__output_flag is True:
            self.__inferenced_arr = inferenced_arr
            if self.__join_io_flag is False:
                return inferenced_arr
            else:
                return np.concatenate([observed_arr, inferenced_arr], axis=1)
        else:
            return self.__lstm_model.get_feature_points()

    def learn(self, grad_arr):
        '''
        Update this Discriminator by ascending its stochastic gradient.

        Args:
            grad_arr:   `np.ndarray` of gradients.

        Returns:
            `np.ndarray` of delta or gradients.

        '''
        if self.__output_flag is True:
            if self.__join_io_flag is False:
                delta_arr, grads_list = self.__lstm_model.back_propagation(self.__inferenced_arr, grad_arr)
            else:
                grad_arr = grad_arr[:, self.__seq_len:]
                delta_arr, grads_list = self.__lstm_model.back_propagation(self.__inferenced_arr, grad_arr)
        else:
            if grad_arr.ndim > 3:
                grad_arr = grad_arr.reshape((
                    grad_arr.shape[0],
                    grad_arr.shape[1],
                    -1
                ))
                grad_arr = grad_arr[:, -1]
            elif grad_arr.ndim == 3:
                grad_arr = grad_arr[:, -1]

            delta_arr, _, grads_list = self.__lstm_model.hidden_back_propagate(grad_arr)
            grads_list.insert(0, None)
            grads_list.insert(0, None)

        self.__lstm_model.optimize(
            grads_list,
            self.__learning_rate,
            self.__epoch_counter
        )
        self.__epoch_counter += 1

        return delta_arr

    def get_lstm_model(self):
        ''' getter '''
        return self.__lstm_model
    
    def set_lstm_model(self, value):
        ''' setter '''
        raise TypeError("This property must be read-only.")
    
    lstm_model = property(get_lstm_model, set_lstm_model)
