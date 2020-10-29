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


class NeuralNetworks(HybridBlock, ObservableData):
    '''
    Neural Networks.

    References:
        - Kamyshanska, H., & Memisevic, R. (2014). The potential energy of an autoencoder. IEEE transactions on pattern analysis and machine intelligence, 37(6), 1261-1273.
    '''

    # `bool` that means initialization in this class will be deferred or not.
    __init_deferred_flag = False

    def __init__(
        self,
        computable_loss,
        initializer=None,
        learning_rate=1e-05,
        learning_attenuate_rate=1.0,
        attenuate_epoch=50,
        units_list=[100, 1],
        dropout_rate_list=[0.0, 0.5],
        optimizer_name="SGD",
        activation_list=["tanh", "sigmoid"],
        hidden_batch_norm_list=[None, None],
        ctx=mx.gpu(),
        hybridize_flag=True,
        regularizatable_data_list=[],
        scale=1.0,
        output_no_bias_flag=False,
        all_no_bias_flag=False,
        not_init_flag=False,
        **kwargs
    ):
        '''
        Init.

        Args:
            computable_loss:                is-a `ComputableLoss` or `gluon.loss.Loss`.
            initializer:                    is-a `mxnet.initializer.Initializer` for parameters of model. If `None`, it is drawing from the Xavier distribution.
            learning_rate:                  `float` of learning rate.
            learning_attenuate_rate:        `float` of attenuate the `learning_rate` by a factor of this value every `attenuate_epoch`.
            attenuate_epoch:                `int` of attenuate the `learning_rate` by a factor of `learning_attenuate_rate` every `attenuate_epoch`.

            units_list:                     `list` of int` of the number of units in hidden/output layers.
            dropout_rate_list:              `list` of `float` of dropout rate.
            optimizer_name:                 `str` of name of optimizer.
            activation_list:                `list` of act_type` in `mxnet.ndarray.Activation` or `mxnet.symbol.Activation` in input gate.
            hidden_batch_norm_list:         `list` of `mxnet.gluon.nn.BatchNorm`.
            ctx:                            `mx.cpu()` or `mx.gpu()`.
            hybridize_flag:                  Call `mxnet.gluon.HybridBlock.hybridize()` or not.
            regularizatable_data_list:           `list` of `RegularizatableData`.
            scale:                          `float` of scaling factor for initial parameters.
            output_no_bias_flag:            `bool` for using bias or not in output layer(last hidden layer).
            all_no_bias_flag:               `bool` for using bias or not in all layer.
            not_init_flag:                  `bool` of whether initialize parameters or not.
        '''
        if isinstance(computable_loss, ComputableLoss) is False and isinstance(computable_loss, gluon.loss.Loss) is False:
            raise TypeError("The type of `computable_loss` must be `ComputableLoss` or `gluon.loss.Loss`.")

        if len(units_list) != len(activation_list):
            raise ValueError("The length of `units_list` and `activation_list` must be equivalent.")
        self.__units_list = units_list

        if len(dropout_rate_list) != len(units_list):
            raise ValueError("The length of `dropout_rate_list` and `activation_list` must be equivalent.")

        super(NeuralNetworks, self).__init__(**kwargs)

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
            self.fc_list = [None] * len(units_list)
            for i in range(len(units_list)):
                if all_no_bias_flag is True:
                    use_bias = False
                elif output_no_bias_flag is True and i + 1 == len(units_list):
                    use_bias = False
                else:
                    use_bias = True
                self.fc_list[i] = gluon.nn.Dense(units_list[i], use_bias=use_bias)
                self.register_child(self.fc_list[i])

            self.dropout_forward_list = [None] * len(dropout_rate_list)
            for i in range(len(dropout_rate_list)):
                self.dropout_forward_list[i] = gluon.nn.Dropout(rate=dropout_rate_list[i])
                self.register_child(self.dropout_forward_list[i])

            self.hidden_batch_norm_list = hidden_batch_norm_list
            for i in range(len(hidden_batch_norm_list)):
                if self.hidden_batch_norm_list[i] is not None:
                    self.register_child(self.hidden_batch_norm_list[i])

        if self.init_deferred_flag is False:
            if not_init_flag is False:
                self.collect_params().initialize(self.initializer, force_reinit=True, ctx=ctx)
                self.trainer = gluon.Trainer(self.collect_params(), optimizer_name, {"learning_rate": learning_rate})
                if hybridize_flag is True:
                    self.hybridize()

        self.activation_list = activation_list

        self.__computable_loss = computable_loss
        self.__learning_rate = learning_rate
        self.__learning_attenuate_rate = learning_attenuate_rate
        self.__attenuate_epoch = attenuate_epoch

        for v in regularizatable_data_list:
            if isinstance(v, RegularizatableData) is False:
                raise TypeError("The type of values of `regularizatable_data_list` must be `RegularizatableData`.")
        self.__regularizatable_data_list = regularizatable_data_list

        self.__ctx = ctx

        logger = getLogger("accelbrainbase")
        self.__logger = logger

    def learn(self, iteratable_data):
        '''
        Learn samples drawn by `IteratableData.generate_learned_samples()`.

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
        Inference samples drawn by `IteratableData.generate_inferenced_samples()`.

        Args:
            observed_arr:   rank-2 Array like or sparse matrix as the observed data points.
                            The shape is: (batch size, feature points)

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
        for regularizatable in self.__regularizatable_data_list:
            params_dict = regularizatable.regularize(params_dict)

        for k, params in self.collect_params().items():
            params.set_data(params_dict[k])

    def extract_learned_dict(self):
        '''
        Extract (pre-) learned parameters.

        Returns:
            `dict` of the parameters.
        '''
        params_dict = self.collect_params()
        
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
        x = x.flatten()
        for i in range(len(self.activation_list)):
            x = self.fc_list[i](x)

            if self.activation_list[i] == "identity_adjusted":
                x = x / F.sum(F.ones_like(x))
            elif self.activation_list[i] == "softmax":
                x = F.softmax(x)
            elif self.activation_list[i] == "log_softmax":
                x = F.softmax(x)
            elif self.activation_list[i] != "identity":
                x = F.Activation(x, self.activation_list[i])
            if self.dropout_forward_list[i] is not None:
                x = self.dropout_forward_list[i](x)
            if self.hidden_batch_norm_list[i] is not None:
                x = self.hidden_batch_norm_list[i](x)

        return x

    def set_readonly(self, value):
        ''' setter '''
        raise TypeError("This property must be read-only.")

    def get_loss_arr(self):
        ''' getter for losses. '''
        return np.array(self.__loss_list)

    loss_arr = property(get_loss_arr, set_readonly)

    def get_init_deferred_flag(self):
        ''' getter for `bool` that means initialization in this class will be deferred or not. '''
        return self.__init_deferred_flag
    
    def set_init_deferred_flag(self, value):
        ''' setter for `bool` that means initialization in this class will be deferred or not. '''
        self.__init_deferred_flag = value

    init_deferred_flag = property(get_init_deferred_flag, set_init_deferred_flag)

    def get_units_list(self):
        ''' getter for `list` of units in each layer. '''
        return self.__units_list
    
    units_list = property(get_units_list, set_readonly)

    # is-a `mxnet.initializer` for parameters of model. If `None`, it is drawing from the Xavier distribution.
    __initializer = None

    def get_initializer(self):
        ''' getter for `mxnet.initializer`. '''
        return self.__initializer
    
    def set_initializer(self, value):
        ''' setter for `mxnet.initializer`.'''
        self.__initializer = value
    
    initializer = property(get_initializer, set_initializer)
