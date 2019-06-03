# -*- coding: utf-8 -*-
import numpy as np
from logging import getLogger

from pygan.discriminative_model import DiscriminativeModel

# LSTM Graph which is-a `Synapse`.
from pydbm.synapse.recurrenttemporalgraph.lstm_graph import LSTMGraph
# Loss function.
from pydbm.loss.mean_squared_error import MeanSquaredError
# SGD as a Loss function.
from pydbm.optimization.optparams.sgd import SGD
# Verification.
from pydbm.verification.verificate_function_approximation import VerificateFunctionApproximation

from pydbm.activation.interface.activating_function_interface import ActivatingFunctionInterface
# Logistic Function as activation function.
from pydbm.activation.logistic_function import LogisticFunction
# Tanh Function as activation function.
from pydbm.activation.tanh_function import TanhFunction

from pydbm.rnn.lstm_model import LSTMModel as LSTM


class LSTMModel(DiscriminativeModel):
    '''
    LSTM as a Discriminator.

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
        - Mogren, O. (2016). C-RNN-GAN: Continuous recurrent neural networks with adversarial training. arXiv preprint arXiv:1611.09904.
        - Zaremba, W., Sutskever, I., & Vinyals, O. (2014). Recurrent neural network regularization. arXiv preprint arXiv:1409.2329.

    '''

    def __init__(
        self,
        lstm_model=None,
        batch_size=20,
        input_neuron_count=100,
        hidden_neuron_count=300,
        observed_activating_function=None,
        input_gate_activating_function=None,
        forget_gate_activating_function=None,
        output_gate_activating_function=None,
        hidden_activating_function=None,
        seq_len=10,
        learning_rate=1e-05
    ):
        '''
        Init.

        Args:
            lstm_model:                         is-a `lstm_model`.
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
                                                If `None`, this value will be `TanhFunction`.

            seq_len:                            The length of sequences.
                                                This means refereed maxinum step `t` in feedforward.

            learning_rate:                      Learning rate.
        '''
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

            graph.output_activating_function = LogisticFunction()

            # Initialization strategy.
            # This method initialize each weight matrices and biases in Gaussian distribution: `np.random.normal(size=hoge) * 0.01`.
            graph.create_rnn_cells(
                input_neuron_count=input_neuron_count,
                hidden_neuron_count=hidden_neuron_count,
                output_neuron_count=1
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
                computable_loss=MeanSquaredError(),
                # Optimizer.
                opt_params=opt_params,
                # Verification function.
                verificatable_result=VerificateFunctionApproximation(),
                tol=0.0
            )

        self.__lstm_model = lstm_model
        self.__seq_len = seq_len
        self.__learning_rate = learning_rate
        self.__loss_list = []
        self.__epoch_counter = 0
        logger = getLogger("pygan")
        self.__logger = logger

    def inference(self, observed_arr):
        '''
        Draws samples from the `true` distribution.

        Args:
            observed_arr:     `np.ndarray` of observed data points.
        
        Returns:
            `np.ndarray` of inferenced.
        '''
        self.__pred_arr = self.__lstm_model.inference(observed_arr)
        return self.__pred_arr

    def learn(self, grad_arr, fix_opt_flag=False):
        '''
        Update this Discriminator by ascending its stochastic gradient.

        Args:
            grad_arr:       `np.ndarray` of gradients.
            fix_opt_flag:   If `False`, no optimization in this model will be done.
        
        Returns:
            `np.ndarray` of delta or gradients.
        '''
        if grad_arr.ndim > 3:
            grad_arr = grad_arr.reshape((
                grad_arr.shape[0],
                grad_arr.shape[1],
                -1
            ))

        delta_arr, grads_list = self.__lstm_model.back_propagation(self.__pred_arr, grad_arr)

        if fix_opt_flag is False:
            self.__lstm_model.optimize(
                grads_list,
                self.__learning_rate,
                self.__epoch_counter
            )
            self.__epoch_counter += 1

        return delta_arr

    def feature_matching_forward(self, observed_arr):
        '''
        Forward propagation in only first or intermediate layer
        for so-called Feature matching.

        Like C-RNN-GAN(Mogren, O. 2016), this model chooses 
        the last layer before the output layer in this Discriminator.

        Args:
            observed_arr:       `np.ndarray` of observed data points.

        Returns:
            `np.ndarray` of outputs.
        '''
        return self.__lstm_model.hidden_forward_propagate(observed_arr)

    def feature_matching_backward(self, grad_arr):
        '''
        Back propagation in only first or intermediate layer
        for so-called Feature matching.

        Args:
            observed_arr:       `np.ndarray` of observed data points.

        Returns:
            `np.ndarray` of outputs.
        '''
        grad_arr, _, _ = self.__lstm_model.hidden_back_propagate(grad_arr[:, -1])
        return grad_arr

    def get_lstm_model(self):
        ''' getter '''
        return self.__lstm_model
    
    def set_lstm_model(self, value):
        ''' setter '''
        raise TypeError("This property must be read-only.")
    
    lstm_model = property(get_lstm_model, set_lstm_model)
