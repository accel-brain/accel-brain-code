# -*- coding: utf-8 -*-
from accelbrainbase.observable_data import ObservableData
from accelbrainbase.iteratable_data import IteratableData
from accelbrainbase.regularizatable_data import RegularizatableData
from accelbrainbase.computable_loss import ComputableLoss
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim.optimizer import Optimizer, required
from torch.optim.adam import Adam
from logging import getLogger


class NeuralNetworks(nn.Module, ObservableData):
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
        initializer_f=None,
        optimizer_f=None,
        learning_rate=1e-05,
        units_list=[100, 1],
        dropout_rate_list=[0.0, 0.5],
        activation_list=[
            torch.nn.functional.tanh, 
            torch.nn.functional.sigmoid
        ],
        hidden_batch_norm_list=[
            100, 
            None
        ],
        ctx="cpu",
        regularizatable_data_list=[],
        scale=1.0,
        output_no_bias_flag=False,
        all_no_bias_flag=False,
        not_init_flag=False,
    ):
        '''
        Init.

        Args:
            computable_loss:                is-a `ComputableLoss` or `nn.modules.loss._Loss`.
            initializer_f:                  A function that contains `torch.nn.init`.
                                            This function receive `tensor` as input and output initialized `tensor`. 
                                            If `None`, it is drawing from the Xavier distribution.

            optimizer_f:                    A function that contains `torch.optim.optimizer.Optimizer` for parameters of model.
                                            This function receive `self.parameters()` as input and output `torch.optim.optimizer.Optimizer`.

            learning_rate:                  `float` of learning rate.
            units_list:                     `list` of int` of the number of units in hidden/output layers.
            dropout_rate_list:              `list` of `float` of dropout rate.
            activation_list:                `list` of act_type` in `mxnet.ndarray.Activation` or `mxnet.symbol.Activation` in input gate.
            hidden_batch_norm_list:         `list` of `mxnet.gluon.nn.BatchNorm`.
            ctx:                            Context-manager that changes the selected device.
            regularizatable_data_list:           `list` of `RegularizatableData`.
            scale:                          `float` of scaling factor for initial parameters.
            output_no_bias_flag:            `bool` for using bias or not in output layer(last hidden layer).
            all_no_bias_flag:               `bool` for using bias or not in all layer.
            not_init_flag:                  `bool` of whether initialize parameters or not.
        '''
        super(NeuralNetworks, self).__init__()

        if isinstance(computable_loss, ComputableLoss) is False and isinstance(computable_loss, nn.modules.loss._Loss) is False:
            raise TypeError("The type of `computable_loss` must be `ComputableLoss` or `nn.modules.loss._Loss`.")

        if len(units_list) != len(activation_list):
            raise ValueError("The length of `units_list` and `activation_list` must be equivalent.")
        self.__units_list = units_list

        if len(dropout_rate_list) != len(units_list):
            raise ValueError("The length of `dropout_rate_list` and `activation_list` must be equivalent.")

        self.initializer_f = initializer_f
        self.optimizer_f = optimizer_f
        self.__units_list = units_list
        self.__all_no_bias_flag = all_no_bias_flag
        self.__output_no_bias_flag = output_no_bias_flag

        self.dropout_forward_list = [None] * len(dropout_rate_list)
        for i in range(len(dropout_rate_list)):
            self.dropout_forward_list[i] = nn.Dropout(p=dropout_rate_list[i])
        self.dropout_forward_list = nn.ModuleList(self.dropout_forward_list)

        self.hidden_batch_norm_list = [None] * len(hidden_batch_norm_list)
        for i in range(len(hidden_batch_norm_list)):
            if hidden_batch_norm_list[i] is not None:
                if isinstance(hidden_batch_norm_list[i], int) is True:
                    self.hidden_batch_norm_list[i] = nn.BatchNorm1d(
                        hidden_batch_norm_list[i]
                    )
                else:
                    self.hidden_batch_norm_list[i] = hidden_batch_norm_list[i]

        self.hidden_batch_norm_list = nn.ModuleList(self.hidden_batch_norm_list)

        self.__not_init_flag = not_init_flag
        self.activation_list = activation_list

        self.__computable_loss = computable_loss
        self.__learning_rate = learning_rate

        for v in regularizatable_data_list:
            if isinstance(v, RegularizatableData) is False:
                raise TypeError("The type of values of `regularizatable_data_list` must be `RegularizatableData`.")
        self.__regularizatable_data_list = regularizatable_data_list

        self.__ctx = ctx

        self.fc_list = []
        self.flatten = nn.Flatten()

        self.epoch = 0
        logger = getLogger("accelbrainbase")
        self.__logger = logger
        self.__input_dim = None

    def initialize_params(self, input_dim):
        '''
        Initialize params.

        Args:
            input_dim:      The number of units in input layer.
        '''
        if self.__input_dim is not None:
            return
        self.__input_dim = input_dim

        if len(self.fc_list) > 0:
            return

        if self.__all_no_bias_flag is True:
            use_bias = False
        else:
            use_bias = True

        fc = nn.Linear(
            input_dim, 
            self.__units_list[0], 
            bias=use_bias
        )
        if self.initializer_f is None:
            fc.weight = torch.nn.init.xavier_normal_(
                fc.weight,
                gain=1.0
            )
        else:
            fc.weight = self.initializer_f(fc.weight)

        fc_list = [fc]

        for i in range(1, len(self.__units_list)):
            if self.__all_no_bias_flag is True:
                use_bias = False
            elif self.__output_no_bias_flag is True and i + 1 == len(self.__units_list):
                use_bias = False
            else:
                use_bias = True

            fc = nn.Linear(
                self.__units_list[i-1], 
                self.__units_list[i], 
                bias=use_bias
            )

            if self.initializer_f is None:
                fc.weight = torch.nn.init.xavier_normal_(
                    fc.weight,
                    gain=1.0
                )
            else:
                fc.weight = self.initializer_f(fc.weight)

            fc_list.append(fc)

        self.fc_list = nn.ModuleList(fc_list)
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
        Learn samples drawn by `IteratableData.generate_learned_samples()`.

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
                self.initialize_params(input_dim=self.flatten(batch_observed_arr).shape[-1])
                self.optimizer.zero_grad()
                # rank-3
                pred_arr = self.inference(batch_observed_arr)
                loss = self.compute_loss(
                    pred_arr,
                    batch_target_arr
                )
                loss.backward()
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
        Inference samples drawn by `IteratableData.generate_inferenced_samples()`.

        Args:
            observed_arr:   rank-2 Array like or sparse matrix as the observed data points.
                            The shape is: (batch size, feature points)

        Returns:
            `tensor` of inferenced feature points.
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

    def forward(self, x):
        '''
        Forward with torch.

        Args:
            x:      `tensor` of observed data points.
        
        Returns:
            `tensor` of inferenced feature points.
        '''
        x = self.flatten(x)
        self.initialize_params(input_dim=x.shape[-1])
        for i in range(len(self.activation_list)):
            x = self.fc_list[i](x)

            if self.activation_list[i] == "identity_adjusted":
                x = x / torch.sum(torch.ones_like(x))
            elif self.activation_list[i] == "softmax":
                x = F.softmax(x)
            elif self.activation_list[i] == "log_softmax":
                x = F.log_softmax(x)
            elif self.activation_list[i] != "identity":
                x = self.activation_list[i](x)

            if self.dropout_forward_list[i] is not None:
                x = self.dropout_forward_list[i](x)
            if self.hidden_batch_norm_list[i] is not None:
                x = self.hidden_batch_norm_list[i](x)

        return x

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
        self.initialize_params(input_dim=checkpoint["input_dim"])
        self.load_state_dict(checkpoint['model_state_dict'], strict=strict)
        self.optimizer.load_state_dict(
            checkpoint['optimizer_state_dict']
        )
        self.epoch = checkpoint['epoch']
        self.loss_arr = checkpoint['loss']
        if ctx is not None:
            self.to(ctx)
            self.__ctx = ctx

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
