# -*- coding: utf-8 -*-
from accelbrainbase.observable_data import ObservableData
from accelbrainbase.iteratable_data import IteratableData
from accelbrainbase.regularizatable_data import RegularizatableData
from accelbrainbase.computable_loss import ComputableLoss
from mxnet.gluon.block import HybridBlock
from mxnet import gluon
from mxnet import autograd
import numpy as np
import mxnet as mx
from mxnet import MXNetError
from logging import getLogger


class LSTMNetworks(HybridBlock, ObservableData):
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
        batch_size,
        seq_len,
        computable_loss,
        initializer=None,
        learning_rate=1e-05,
        learning_attenuate_rate=0.1,
        attenuate_epoch=50,
        hidden_n=200,
        output_n=1,
        dropout_rate=0.5,
        optimizer_name="SGD",
        input_adjusted_flag=True,
        observed_activation="tanh",
        input_gate_activation="sigmoid",
        forget_gate_activation="sigmoid",
        output_gate_activation="sigmoid",
        hidden_activation="tanh",
        output_activation="tanh",
        output_layer_flag=True,
        output_no_bias_flag=False,
        output_nn=None,
        ctx=mx.gpu(),
        hybridize_flag=True,
        regularizatable_data_list=[],
        scale=1.0,
        **kwargs
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
        if isinstance(computable_loss, ComputableLoss) is False and isinstance(computable_loss, gluon.loss.Loss) is False:
            raise TypeError("The type of `computable_loss` must be `ComputableLoss` or `gluon.loss.Loss`.")

        super(LSTMNetworks, self).__init__(**kwargs)

        if initializer is None:
            self.initializer = mx.initializer.Xavier(
                rnd_type="gaussian", 
                factor_type="in", 
                magnitude=1
            )
        else:
            if isinstance(initializer, mx.initializer.Initializer) is False:
                raise TypeError("The type of `initializer` must be `mxnet.initializer.Initializer`.")

            self.initializer = initializer

        with self.name_scope():
            if self.init_deferred_flag is False:
                self.observed_fc = gluon.nn.Dense(hidden_n * 4, use_bias=False, prefix="observed_fc_")
                self.hidden_fc = gluon.nn.Dense(hidden_n * 4, prefix="hidden_fc_")
                self.input_gate_fc = gluon.nn.Dense(hidden_n, use_bias=False, prefix="input_gate_fc_")
                self.forget_gate_fc = gluon.nn.Dense(hidden_n, use_bias=False, prefix="forget_gate_fc_")
                self.output_gate_fc = gluon.nn.Dense(hidden_n, use_bias=False, prefix="output_gate_fc_")
                if dropout_rate > 0.0:
                    self.dropout_forward = gluon.nn.Dropout(rate=dropout_rate)

            self.output_fc = None
            self.output_nn = None
            if output_layer_flag is True and output_nn is None:
                if output_no_bias_flag is True:
                    use_bias = False
                else:
                    use_bias = True
                self.output_fc = gluon.nn.Dense(output_n, prefix="output_fc_", use_bias=use_bias)
            elif output_nn is not None:
                self.output_nn = output_nn

        if self.init_deferred_flag is False:
            self.collect_params().initialize(self.initializer, force_reinit=True, ctx=ctx)
            self.trainer = gluon.Trainer(self.collect_params(), optimizer_name, {"learning_rate": learning_rate})
            if hybridize_flag is True:
                self.hybridize()
                if self.output_nn is not None:
                    self.output_nn.hybridize()

        self.__observed_activation = observed_activation
        self.__input_gate_activation = input_gate_activation
        self.__forget_gate_activation = forget_gate_activation
        self.__output_gate_activation = output_gate_activation
        self.__hidden_activation = hidden_activation
        self.__output_activation = output_activation
        self.__output_layer_flag = output_layer_flag

        self.__computable_loss = computable_loss
        self.__batch_size = batch_size
        self.__learning_rate = learning_rate
        self.__learning_attenuate_rate = learning_attenuate_rate
        self.__attenuate_epoch = attenuate_epoch
        self.__seq_len = seq_len
        self.__hidden_n = hidden_n
        self.__output_n = output_n
        self.__dropout_rate = dropout_rate
        self.__input_adjusted_flag = input_adjusted_flag

        for v in regularizatable_data_list:
            if isinstance(v, RegularizatableData) is False:
                raise TypeError("The type of values of `regularizatable_data_list` must be `Regularizatable`.")
        self.__regularizatable_data_list = regularizatable_data_list

        self.__ctx = ctx
        self.__hybridize_flag = hybridize_flag

        logger = getLogger("accelbrainbase")
        self.__logger = logger

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
            epoch = 0
            iter_n = 0
            for batch_observed_arr, batch_target_arr, test_batch_observed_arr, test_batch_target_arr in iteratable_data.generate_learned_samples():
                self.epoch = epoch
                if ((epoch + 1) % self.__attenuate_epoch == 0):
                    learning_rate = learning_rate * self.__learning_attenuate_rate
                    self.trainer.set_learning_rate(learning_rate)

                with autograd.record():
                    # rank-3
                    pred_arr = self.inference(batch_observed_arr)
                    loss = self.compute_loss(
                        pred_arr,
                        batch_target_arr
                    )
                loss.backward()

                self.trainer.step(batch_observed_arr.shape[0])
                self.regularize()

                if (iter_n+1) % int(iteratable_data.iter_n / iteratable_data.epochs) == 0:
                    # rank-3
                    test_pred_arr = self.inference(test_batch_observed_arr)

                    test_loss = self.compute_loss(
                        test_pred_arr,
                        test_batch_target_arr
                    )

                    self.__loss_list.append((loss.asnumpy().mean(), test_loss.asnumpy().mean()))
                    self.__logger.debug("Epochs: " + str(epoch + 1) + " Train loss: " + str(loss.asnumpy().mean()) + " Test loss: " + str(test_loss.asnumpy().mean()))
                    epoch += 1
                iter_n += 1

        except KeyboardInterrupt:
            self.__logger.debug("Interrupt.")

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
        params_dict = self.extract_learned_dict()
        for regularizatable_data in self.__regularizatable_data_list:
            params_dict = regularizatable_data.regularize(params_dict)

        for k, params in self.collect_params().items():
            params.set_data(params_dict[k])

    def extract_feature_points(self):
        '''
        Extract the activities in hidden layer and reset it, 
        considering this method will be called per one cycle in instances of time-series.

        Returns:
            The `mxnet.ndarray` of array like or sparse matrix of feature points or virtual visible observed data points.
        '''
        return self.feature_points_arr

    def extract_learned_dict(self):
        '''
        Extract (pre-) learned parameters.

        Returns:
            `dict` of the parameters.
        '''
        params_dict = self.collect_params()
        if self.output_nn is not None:
            params_dict.update(self.output_nn.collect_params())

        params_arr_dict = {}
        for k in params_dict:
            params_arr_dict.setdefault(k, params_dict[k].data())

        return params_arr_dict

    def hybrid_forward(self, F, x):
        '''
        Hybrid forward with Gluon API.

        Args:
            F:      `mxnet.ndarray` or `mxnet.symbol`.
            x:      `mxnet.ndarray` of observed data points.
        
        Returns:
            `mxnet.ndarray` or `mxnet.symbol` of inferenced feature points.
        '''
        # rank-3
        return self.forward_propagation(F, x)

    def forward_propagation(self, F, x):
        '''
        Hybrid forward with Gluon API.

        Args:
            F:      `mxnet.ndarray` or `mxnet.symbol`.
            x:      `mxnet.ndarray` of observed data points.
        
        Returns:
            `mxnet.ndarray` or `mxnet.symbol` of inferenced feature points.
        '''
        hidden_activity_arr = self.hidden_forward_propagate(F, x)

        if self.__dropout_rate > 0:
            hidden_activity_arr = self.dropout_forward(hidden_activity_arr)
        self.feature_points_arr = hidden_activity_arr

        if self.output_nn is not None:
            pred_arr = self.output_nn.forward_propagation(F, hidden_activity_arr)
            return pred_arr
        if self.__output_layer_flag is True:
            # rank-3
            pred_arr = self.output_forward_propagate(F, hidden_activity_arr)
            return pred_arr
        else:
            return hidden_activity_arr

    def hidden_forward_propagate(self, F, observed_arr):
        '''
        Forward propagation in LSTM gate.

        Args:
            F:                      `mxnet.ndarray` or `mxnet.symbol`.
            observed_arr:           rank-3 tensor of observed data points.
        
        Returns:
            Predicted data points.
        '''
        pred_arr = None

        hidden_activity_arr = F.zeros((self.__batch_size, self.__hidden_n), dtype=np.float32)
        cec_activity_arr = F.zeros((self.__batch_size, self.__hidden_n), dtype=np.float32)

        for cycle in range(self.__seq_len):
            if cycle == 0:
                hidden_activity_arr, cec_activity_arr = self.__lstm_forward(
                    F,
                    F.slice(observed_arr, begin=(0, cycle), end=(self.__batch_size, cycle+1)),
                    hidden_activity_arr,
                    cec_activity_arr
                )
            else:
                try:
                    x_arr = F.slice(observed_arr, begin=(0, cycle), end=(self.__batch_size, cycle+1))
                except MXNetError:
                    x_arr = F.slice(pred_arr, begin=(0, cycle-1), end=(self.__batch_size, cycle))

                hidden_activity_arr, cec_activity_arr = self.__lstm_forward(
                    F,
                    x_arr,
                    hidden_activity_arr,
                    cec_activity_arr
                )

            add_arr = F.expand_dims(hidden_activity_arr, axis=1)
            if pred_arr is None:
                pred_arr = add_arr
            else:
                pred_arr = F.concat(pred_arr, add_arr, dim=1)

        return pred_arr

    def __lstm_forward(
        self,
        F,
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
        if self.__input_adjusted_flag is True:
            observed_arr = F.broadcast_div(observed_arr, F.sum(F.ones_like(observed_arr)))

        observed_lstm_matrix = self.observed_fc(observed_arr)

        # using bias
        hidden_lstm_matrix = self.hidden_fc(hidden_activity_arr)
        lstm_matrix = observed_lstm_matrix + hidden_lstm_matrix

        given_activity_arr = F.slice(lstm_matrix, begin=(0, 0), end=(self.__batch_size, self.__hidden_n))
        input_gate_activity_arr = F.slice(lstm_matrix, begin=(0, self.__hidden_n), end=(self.__batch_size, self.__hidden_n * 2))
        forget_gate_activity_arr = F.slice(lstm_matrix, begin=(0, self.__hidden_n * 2), end=(self.__batch_size, self.__hidden_n * 3))
        output_gate_activity_arr = F.slice(lstm_matrix, begin=(0, self.__hidden_n * 3), end=(self.__batch_size, self.__hidden_n * 4))

        # no bias
        _input_gate_activity_arr = self.input_gate_fc(cec_activity_arr)
        input_gate_activity_arr = input_gate_activity_arr + _input_gate_activity_arr
        # no bias
        _forget_gate_activity_arr = self.forget_gate_fc(cec_activity_arr)
        forget_gate_activity_arr = forget_gate_activity_arr + _forget_gate_activity_arr
        given_activity_arr = F.Activation(given_activity_arr, self.__observed_activation)
        input_gate_activity_arr = F.Activation(input_gate_activity_arr, self.__input_gate_activation)
        forget_gate_activity_arr = F.Activation(forget_gate_activity_arr, self.__forget_gate_activation)

        # rank-2
        _cec_activity_arr = F.broadcast_mul(given_activity_arr, input_gate_activity_arr) + F.broadcast_mul(forget_gate_activity_arr, cec_activity_arr)

        # no bias
        _output_gate_activity_arr = self.output_gate_fc(_cec_activity_arr)

        output_gate_activity_arr = output_gate_activity_arr + _output_gate_activity_arr
        output_gate_activity_arr = F.Activation(output_gate_activity_arr, self.__output_gate_activation)

        # rank-2
        _hidden_activity_arr = F.broadcast_mul(output_gate_activity_arr, F.Activation(_cec_activity_arr, self.__hidden_activation))

        return (_hidden_activity_arr, _cec_activity_arr)

    def output_forward_propagate(self, F, pred_arr):
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
        pred_arr = self.output_fc(F.reshape(pred_arr, shape=(self.__batch_size * self.__seq_len, -1)))
        if self.__output_activation == "identity_adjusted":
            pred_arr = F.broadcast_div(pred_arr, F.sum(F.ones_like(pred_arr)))
        elif self.__output_activation != "identity":
            pred_arr = F.Activation(pred_arr, self.__output_activation)
        pred_arr = F.reshape(pred_arr, shape=(self.__batch_size, self.__seq_len, -1))
        return pred_arr

    def set_readonly(self, value):
        ''' setter for losses. '''
        raise TypeError("This property must be read-only.")

    def get_loss_arr(self):
        ''' getter for losses. '''
        return np.array(self.__loss_list)

    loss_arr = property(get_loss_arr, set_readonly)

    def get_init_deferred_flag(self):
        ''' getter for `bool` that means initialization in this class will be deferred or not.'''
        return self.__init_deferred_flag
    
    def set_init_deferred_flag(self, value):
        ''' setter for `bool` that means initialization in this class will be deferred or not. '''
        self.__init_deferred_flag = value

    init_deferred_flag = property(get_init_deferred_flag, set_init_deferred_flag)

    # is-a `mxnet.initializer` for parameters of model. If `None`, it is drawing from the Xavier distribution.
    __initializer = None

    def get_initializer(self):
        ''' getter for `mxnet.initializer`. '''
        return self.__initializer
    
    def set_initializer(self, value):
        ''' setter for `mxnet.initializer`. '''
        self.__initializer = value
    
    initializer = property(get_initializer, set_initializer)
