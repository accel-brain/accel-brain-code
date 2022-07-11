# -*- coding: utf-8 -*-
from accelbrainbase.observable_data import ObservableData
from accelbrainbase.iteratable_data import IteratableData
from accelbrainbase.regularizatable_data import RegularizatableData
from accelbrainbase.computable_loss import ComputableLoss
import numpy as np
from logging import getLogger
import torch
from torch import nn
import torch.nn.functional as F
from torch.optim.optimizer import Optimizer, required
from torch.optim.adam import Adam


class LSTMNetworks(nn.Module, ObservableData):
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

    # `bool` that means initialization in this class will be deferred or not.
    __init_deferred_flag = False

    def __init__(
        self,
        computable_loss,
        initializer_f=None,
        optimizer_f=None,
        learning_rate=1e-05,
        seq_len=None,
        hidden_n=200,
        output_n=1,
        dropout_rate=0.5,
        input_adjusted_flag=True,
        observed_activation=torch.nn.Tanh(),
        input_gate_activation=torch.nn.Sigmoid(),
        forget_gate_activation=torch.nn.Sigmoid(),
        output_gate_activation=torch.nn.Sigmoid(),
        hidden_activation=torch.nn.Tanh(),
        output_activation=torch.nn.Tanh(),
        output_layer_flag=True,
        output_no_bias_flag=False,
        output_nn=None,
        ctx="cpu",
        regularizatable_data_list=[],
        not_init_flag=False,
    ):
        '''
        Init.

        Args:
            computable_loss:                is-a `ComputableLoss` or `mxnet.gluon.loss`.
            initializer:                    is-a `mxnet.initializer.Initializer` for parameters of model. If `None`, it is drawing from the Xavier distribution.
            batch_size:                     `int` of batch size of mini-batch.
            learning_rate:                  `float` of learning rate.
            learning_attenuate_rate:        `float` of attenuate the `learning_rate` by a factor of this value every `attenuate_epoch`.
            attenuate_epoch:                `int` of attenuate the `learning_rate` by a factor of `learning_attenuate_rate` every `attenuate_epoch`.
                                            

            seq_len:                        `int` of the length of sequences.
                                            This means refereed maxinum step `t` in feedforward.
                                            If `0`, this model will reference all series elements included 
                                            in observed data points.
                                            If not `0`, only first sequence will be observed by this model 
                                            and will be feedfowarded as feature points.
                                            This parameter enables you to build this class as `Decoder` in
                                            Sequence-to-Sequence(Seq2seq) scheme.

            hidden_n:                       `int` of the number of units in hidden layer.
            output_n:                       `int` of the nuber of units in output layer.
            dropout_rate:                   `float` of dropout rate.
            input_adjusted_flag:            `bool` of flag that means this class will adjusted observed data points by normalization.
            observed_activation:            `act_type` in `mxnet.ndarray.Activation` or `mxnet.symbol.Activation` 
                                            that activates observed data points.

            optimizer_name:                 `str` of name of optimizer.

            input_gate_activation:          `act_type` in `mxnet.ndarray.Activation` or `mxnet.symbol.Activation` in input gate.
            forget_gate_activation:         `act_type` in `mxnet.ndarray.Activation` or `mxnet.symbol.Activation` in forget gate.
            output_gate_activation:         `act_type` in `mxnet.ndarray.Activation` or `mxnet.symbol.Activation` in output gate.
            hidden_activation:              `act_type` in `mxnet.ndarray.Activation` or `mxnet.symbol.Activation` in hidden layer.
            output_activation:              `act_type` in `mxnet.ndarray.Activation` or `mxnet.symbol.Activation` in output layer.
                                            If this value is `identity`, the activation function equivalents to the identity function.

            output_layer_flag:              `bool` that means this class has output layer or not.
            output_no_bias_flag:            `bool` for using bias or not in output layer(last hidden layer).
            output_nn:               is-a `NNHybrid` as output layers.
                                            If not `None`, `output_layer_flag` and `output_no_bias_flag` will be ignored.

            ctx:                            `mx.cpu()` or `mx.gpu()`.
            hybridize_flag:                  Call `mxnet.gluon.HybridBlock.hybridize()` or not.
            regularizatable_data_list:           `list` of `Regularizatable`.
            scale:                          `float` of scaling factor for initial parameters.
        '''
        if isinstance(computable_loss, ComputableLoss) is False and isinstance(computable_loss, nn.modules.loss._Loss) is False:
            raise TypeError("The type of `computable_loss` must be `ComputableLoss` or `gluon.loss.Loss`.")

        super(LSTMNetworks, self).__init__()
        self.initializer_f = initializer_f
        self.optimizer_f = optimizer_f
        self.__not_init_flag = not_init_flag

        if dropout_rate > 0.0:
            self.dropout_forward = nn.Dropout(p=dropout_rate)
        else:
            self.dropout_forward = None

        self.__observed_activation = observed_activation
        self.__input_gate_activation = input_gate_activation
        self.__forget_gate_activation = forget_gate_activation
        self.__output_gate_activation = output_gate_activation
        self.__hidden_activation = hidden_activation
        self.__output_activation = output_activation
        self.__output_layer_flag = output_layer_flag

        self.__computable_loss = computable_loss
        self.__learning_rate = learning_rate
        self.__hidden_n = hidden_n
        self.__output_n = output_n
        self.__dropout_rate = dropout_rate
        self.__input_adjusted_flag = input_adjusted_flag

        for v in regularizatable_data_list:
            if isinstance(v, RegularizatableData) is False:
                raise TypeError("The type of values of `regularizatable_data_list` must be `Regularizatable`.")
        self.__regularizatable_data_list = regularizatable_data_list

        self.__ctx = ctx

        logger = getLogger("accelbrainbase")
        self.__logger = logger

        self.__input_dim = None
        self.__input_seq_len = None

        self.__output_layer_flag = output_layer_flag
        self.__output_no_bias_flag = output_no_bias_flag
        self.__output_nn = output_nn
        self.seq_len = seq_len

        self.epoch = 0
        self.__loss_list = []

    def initialize_params(self, input_dim, input_seq_len):
        '''
        Initialize params.

        Args:
            input_dim:      The number of units in input layer.
        '''
        if self.__input_dim is not None:
            return
        self.__input_dim = input_dim
        self.__input_seq_len = input_seq_len

        if self.__not_init_flag is False:
            if self.init_deferred_flag is False:
                self.observed_fc = nn.Linear(
                    input_dim,
                    self.__hidden_n * 4, 
                    bias=False, 
                )
                if self.initializer_f is None:
                    self.observed_fc.weight = torch.nn.init.xavier_normal_(
                        self.observed_fc.weight,
                        gain=1.0
                    )
                else:
                    self.observed_fc.weight = self.initializer_f(self.observed_fc.weight)

                self.hidden_fc = nn.Linear(
                    self.__hidden_n,
                    self.__hidden_n * 4, 
                )
                if self.initializer_f is None:
                    self.hidden_fc.weight = torch.nn.init.xavier_normal_(
                        self.hidden_fc.weight,
                        gain=1.0
                    )
                else:
                    self.hidden_fc.weight = self.initializer_f(self.observed_fc.weight)

                self.input_gate_fc = nn.Linear(
                    self.__hidden_n,
                    self.__hidden_n, 
                    bias=False, 
                )
                self.forget_gate_fc = nn.Linear(
                    self.__hidden_n,
                    self.__hidden_n, 
                    bias=False, 
                )
                self.output_gate_fc = nn.Linear(
                    self.__hidden_n,
                    self.__hidden_n, 
                    bias=False, 
                )

            self.output_fc = None
            self.output_nn = None
            if self.__output_layer_flag is True and self.__output_nn is None:
                if self.__output_no_bias_flag is True:
                    use_bias = False
                else:
                    use_bias = True

                # Different from mxnet version.
                self.output_fc = nn.Linear(
                    self.__hidden_n * self.__input_seq_len,
                    self.__output_n * self.__input_seq_len, 
                    bias=use_bias
                )
                self.__output_dim = self.__output_n
            elif self.__output_nn is not None:
                self.output_nn = self.__output_nn
                self.__output_dim = self.output_nn.units_list[-1]
            else:
                self.__output_dim = self.__hidden_n

        self.to(self.__ctx)
        if self.init_deferred_flag is False:
            if self.__not_init_flag is False:
                if self.optimizer_f is None:
                    self.optimizer = Adam(
                        self.parameters(), 
                        lr=self.__learning_rate,
                    )
                else:
                    self.optimizer = self.optimizer_f(
                        self.parameters(), 
                    )

    def learn(self, iteratable_data):
        '''
        Learn the observed data points
        for vector representation of the input time-series.

        Args:
            iteratable_data:     is-a `IteratableData`.

        '''
        if isinstance(iteratable_data, IteratableData) is False:
            raise TypeError("The type of `iteratable_data` must be `IteratableData`.")

        self.__loss_list = []
        learning_rate = self.__learning_rate
        try:
            epoch = self.epoch
            iter_n = 0
            for batch_observed_arr, batch_target_arr, test_batch_observed_arr, test_batch_target_arr in iteratable_data.generate_learned_samples():
                self.__batch_size = batch_observed_arr.shape[0]
                self.__seq_len = batch_observed_arr.shape[1]
                self.initialize_params(
                    input_dim=batch_observed_arr.reshape(
                        self.__batch_size * self.__seq_len,
                        -1
                    ).shape[-1],
                    input_seq_len=self.__seq_len
                )
                if self.output_nn is not None:
                    if hasattr(self.output_nn, "optimizer") is False:
                        _ = self.inference(batch_observed_arr)

                self.optimizer.zero_grad()
                if self.output_nn is not None:
                    self.output_nn.optimizer.zero_grad()

                # rank-3
                pred_arr = self.inference(batch_observed_arr)
                loss = self.compute_loss(
                    pred_arr,
                    batch_target_arr
                )
                loss.backward()

                if self.output_nn is not None:
                    self.output_nn.optimizer.step()
                self.optimizer.step()
                self.regularize()

                if (iter_n+1) % int(iteratable_data.iter_n / iteratable_data.epochs) == 0:
                    with torch.inference_mode():
                        # rank-3
                        test_pred_arr = self.inference(test_batch_observed_arr)

                        test_loss = self.compute_loss(
                            test_pred_arr,
                            test_batch_target_arr
                        )

                    _loss = loss.to('cpu').detach().numpy().copy()
                    _test_loss = test_loss.to('cpu').detach().numpy().copy()
                    self.__loss_list.append((_loss, _test_loss))
                    self.__logger.debug("Epochs: " + str(epoch + 1) + " Train loss: " + str(_loss) + " Test loss: " + str(_test_loss))
                    epoch += 1
                iter_n += 1

        except KeyboardInterrupt:
            self.__logger.debug("Interrupt.")

        self.epoch = epoch

        self.__logger.debug("end. ")

    def inference(self, observed_arr):
        '''
        Inference the feature points to reconstruct the time-series.

        Args:
            observed_arr:           rank-3 array like or sparse matrix as the observed data points.

        Returns:
            `mxnet.ndarray` of inferenced feature points.
        '''
        return self(observed_arr)

    def compute_loss(self, pred_arr, labeled_arr):
        '''
        Compute loss.

        Args:
            pred_arr:       `mxnet.ndarray` or `mxnet.symbol`.
            labeled_arr:    `mxnet.ndarray` or `mxnet.symbol`.

        Returns:
            loss.
        '''
        return self.__computable_loss(pred_arr, labeled_arr)

    def regularize(self):
        '''
        Regularization.
        '''
        if len(self.__regularizatable_data_list) > 0:
            params_dict = self.extract_learned_dict()
            for regularizatable in self.__regularizatable_data_list:
                params_dict = regularizatable.regularize(params_dict)

            for k, params in params_dict.items():
                self.load_state_dict({k: params}, strict=False)

    def extract_learned_dict(self):
        '''
        Extract (pre-) learned parameters.

        Returns:
            `dict` of the parameters.
        '''
        params_dict = {}
        for k in self.state_dict().keys():
            params_dict.setdefault(k, self.state_dict()[k])

        return params_dict

    def extract_feature_points(self):
        '''
        Extract the activities in hidden layer and reset it, 
        considering this method will be called per one cycle in instances of time-series.

        Returns:
            The `mxnet.ndarray` of array like or sparse matrix of feature points or virtual visible observed data points.
        '''
        return self.feature_points_arr

    def forward(self, x):
        '''
        Forward with Gluon API.

        Args:
            x:      `mxnet.ndarray` of observed data points.
        
        Returns:
            `mxnet.ndarray` or `mxnet.symbol` of inferenced feature points.
        '''
        self.__batch_size = x.shape[0]
        self.__seq_len = x.shape[1]
        x = x.reshape(
            self.__batch_size,
            self.__seq_len,
            -1
        )
        self.initialize_params(
            input_dim=x.shape[2],
            input_seq_len=self.__seq_len
        )

        hidden_activity_arr = self.hidden_forward_propagate(x)

        if self.__dropout_rate > 0:
            hidden_activity_arr = self.dropout_forward(hidden_activity_arr)
        self.feature_points_arr = hidden_activity_arr

        if self.output_nn is not None:
            pred_arr = self.output_nn(hidden_activity_arr)
            return pred_arr
        if self.__output_layer_flag is True:
            # rank-3
            pred_arr = self.output_forward_propagate(hidden_activity_arr)
            return pred_arr
        else:
            return hidden_activity_arr

    def hidden_forward_propagate(self, observed_arr):
        '''
        Forward propagation in LSTM gate.

        Args:
            F:                      `mxnet.ndarray` or `mxnet.symbol`.
            observed_arr:           rank-3 tensor of observed data points.
        
        Returns:
            Predicted data points.
        '''
        pred_arr = None

        hidden_activity_arr = torch.zeros((self.__batch_size, self.__hidden_n), dtype=torch.float32)
        hidden_activity_arr = hidden_activity_arr.to(self.__ctx)
        cec_activity_arr = torch.zeros((self.__batch_size, self.__hidden_n), dtype=torch.float32)
        cec_activity_arr = cec_activity_arr.to(self.__ctx)

        if self.seq_len is not None:
            cycle_n = self.seq_len
        else:
            cycle_n = self.__seq_len

        for cycle in range(cycle_n):
            if cycle == 0:
                if observed_arr[:, cycle:cycle+1].shape[1] != 0:
                    hidden_activity_arr, cec_activity_arr = self.__lstm_forward(
                        observed_arr[:, cycle:cycle+1],
                        hidden_activity_arr,
                        cec_activity_arr
                    )
                    skip_flag = False
                else:
                    skip_flag = True
            else:
                if observed_arr.shape[1] > 1:
                    x_arr = observed_arr[:, cycle:cycle+1]
                else:
                    x_arr = torch.unsqueeze(pred_arr[:, -1], axis=1)

                if x_arr.shape[1] != 0:
                    hidden_activity_arr, cec_activity_arr = self.__lstm_forward(
                        x_arr,
                        hidden_activity_arr,
                        cec_activity_arr
                    )
                    skip_flag = False
                else:
                    skip_flag = True

            if skip_flag is False:
                add_arr = torch.unsqueeze(hidden_activity_arr, axis=1)
                if pred_arr is None:
                    pred_arr = add_arr
                else:
                    pred_arr = torch.cat((pred_arr, add_arr), dim=1)

        return pred_arr

    def __lstm_forward(
        self,
        observed_arr,
        hidden_activity_arr,
        cec_activity_arr
    ):
        '''
        Forward propagate in LSTM gate.
        
        Args:
            F:                      `mxnet.ndarray` or `mxnet.symbol`.
            observed_arr:           rank-2 tensor of observed data points.
            hidden_activity_arr:    rank-2 tensor of activities in hidden layer.
            cec_activity_arr:       rank-2 tensor of activities in the constant error carousel.
        
        Returns:
            Tuple data.
            - rank-2 tensor of activities in hidden layer,
            - rank-2 tensor of activities in LSTM gate.
        '''
        if len(observed_arr.shape) == 3:
            observed_arr = observed_arr[:, 0]

        if self.__input_adjusted_flag is True:
            observed_arr = torch.div(
                observed_arr, 
                torch.sum(torch.ones_like(observed_arr))
            )

        observed_lstm_matrix = self.observed_fc(observed_arr)

        # using bias
        hidden_lstm_matrix = self.hidden_fc(hidden_activity_arr)
        lstm_matrix = observed_lstm_matrix + hidden_lstm_matrix

        given_activity_arr = lstm_matrix[:, :self.__hidden_n]
        input_gate_activity_arr = lstm_matrix[:, self.__hidden_n:self.__hidden_n * 2]
        forget_gate_activity_arr = lstm_matrix[:, self.__hidden_n * 2:self.__hidden_n * 3]
        output_gate_activity_arr = lstm_matrix[:, self.__hidden_n * 3:self.__hidden_n * 4]

        # no bias
        _input_gate_activity_arr = self.input_gate_fc(cec_activity_arr)
        input_gate_activity_arr = input_gate_activity_arr + _input_gate_activity_arr
        # no bias
        _forget_gate_activity_arr = self.forget_gate_fc(cec_activity_arr)
        forget_gate_activity_arr = forget_gate_activity_arr + _forget_gate_activity_arr
        given_activity_arr = self.__observed_activation(given_activity_arr)
        input_gate_activity_arr = self.__input_gate_activation(input_gate_activity_arr)
        forget_gate_activity_arr = self.__forget_gate_activation(forget_gate_activity_arr)

        # rank-2
        _cec_activity_arr = torch.mul(given_activity_arr, input_gate_activity_arr) + torch.mul(forget_gate_activity_arr, cec_activity_arr)

        # no bias
        _output_gate_activity_arr = self.output_gate_fc(_cec_activity_arr)

        output_gate_activity_arr = output_gate_activity_arr + _output_gate_activity_arr
        output_gate_activity_arr = self.__output_gate_activation(output_gate_activity_arr)

        # rank-2
        _hidden_activity_arr = torch.mul(
            output_gate_activity_arr, 
            self.__hidden_activation(_cec_activity_arr)
        )

        return (_hidden_activity_arr, _cec_activity_arr)

    def output_forward_propagate(self, pred_arr):
        '''
        Forward propagation in output layer.
        
        Args:
            F:                   `mxnet.ndarray` or `mxnet.symbol`.
            pred_arr:            rank-3 tensor of predicted data points.

        Returns:
            rank-3 tensor of propagated data points.
        '''
        if self.__output_layer_flag is False:
            return pred_arr

        batch_size = pred_arr.shape[0]
        seq_len = pred_arr.shape[1]
        # Different from mxnet version.
        pred_arr = self.output_fc(
            torch.reshape(
                pred_arr, 
                (batch_size, -1)
            )
        )
        if self.__output_activation == "identity_adjusted":
            pred_arr = torch.div(pred_arr, torch.sum(torch.ones_like(pred_arr)))
        elif self.__output_activation != "identity":
            pred_arr = self.__output_activation(pred_arr)
        pred_arr = torch.reshape(
            pred_arr, 
            (batch_size, seq_len, -1)
        )
        return pred_arr

    def save_parameters(self, filename):
        '''
        Save parameters to files.

        Args:
            filename:       File name.
        '''
        torch.save(
            {
                'epoch': self.epoch,
                'model_state_dict': self.state_dict(),
                'optimizer_state_dict': self.optimizer.state_dict(),
                'loss': self.loss_arr,
                'input_dim': self.__input_dim,
                'input_seq_len': self.__input_seq_len,
            }, 
            filename
        )

    def load_parameters(self, filename, ctx=None, strict=True):
        '''
        Load parameters to files.

        Args:
            filename:       File name.
            ctx:            Context-manager that changes the selected device.
            strict:         Whether to strictly enforce that the keys in state_dict match the keys returned by this module’s state_dict() function. Default: `True`.
        '''
        checkpoint = torch.load(filename)
        self.initialize_params(
            input_dim=checkpoint["input_dim"],
            input_seq_len=checkpoint["input_seq_len"],
        )
        self.load_state_dict(checkpoint['model_state_dict'], strict=strict)
        self.optimizer.load_state_dict(
            checkpoint['optimizer_state_dict']
        )
        self.epoch = checkpoint['epoch']
        self.__loss_list = checkpoint['loss'].tolist()
        if ctx is not None:
            self.to(ctx)
            self.__ctx = ctx

    def set_readonly(self, value):
        ''' setter for losses. '''
        raise TypeError("This property must be read-only.")

    def get_loss_arr(self):
        ''' getter for losses. '''
        return np.array(self.__loss_list)

    loss_arr = property(get_loss_arr, set_readonly)

    def get_output_dim(self):
        return self.__output_dim

    output_dim = property(get_output_dim, set_readonly)

    def get_init_deferred_flag(self):
        ''' getter for `bool` that means initialization in this class will be deferred or not.'''
        return self.__init_deferred_flag
    
    def set_init_deferred_flag(self, value):
        ''' setter for `bool` that means initialization in this class will be deferred or not. '''
        self.__init_deferred_flag = value

    init_deferred_flag = property(get_init_deferred_flag, set_init_deferred_flag)
