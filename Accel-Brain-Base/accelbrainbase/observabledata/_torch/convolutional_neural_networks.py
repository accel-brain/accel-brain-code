# -*- coding: utf-8 -*-
from accelbrainbase.iteratable_data import IteratableData
from accelbrainbase.regularizatable_data import RegularizatableData
from accelbrainbase.observabledata._torch.neural_networks import NeuralNetworks
from accelbrainbase.observable_data import ObservableData
from accelbrainbase.computable_loss import ComputableLoss

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim.optimizer import Optimizer, required
from torch.optim.adam import Adam

import numpy as np
from logging import getLogger


class ConvolutionalNeuralNetworks(nn.Module, ObservableData):
    '''
    Convolutional Neural Networks.

    References:
        - Dumoulin, V., & V,kisin, F. (2016). A guide to convolution arithmetic for deep learning. arXiv preprint arXiv:1603.07285.
        - Kamyshanska, H., & Memisevic, R. (2014). The potential energy of an autoencoder. IEEE transactions on pattern analysis and machine intelligence, 37(6), 1261-1273.
        - Masci, J., Meier, U., Cireşan, D., & Schmidhuber, J. (2011, June). Stacked convolutional auto-encoders for hierarchical feature extraction. In International Conference on Artificial Neural Networks (pp. 52-59). Springer, Berlin, Heidelberg.
    '''

    # `bool` that means initialization in this class will be deferred or not.
    __init_deferred_flag = False

    # is-a `NNHybrid`.
    __input_nn = None
    # is-a `NNHybrid`.
    __output_nn = None

    # `bool`, compute accuracy or not in learning.
    __compute_acc_flag = True

    def get_compute_acc_flag(self):
        ''' getter '''
        return self.__compute_acc_flag
    
    def set_compute_acc_flag(self, value):
        ''' setter '''
        self.__compute_acc_flag = value

    compute_acc_flag = property(get_compute_acc_flag, set_compute_acc_flag)

    def __init__(
        self,
        computable_loss,
        initializer_f=None,
        optimizer_f=None,
        learning_rate=1e-05,
        hidden_units_list=[
            torch.nn.Conv2d(
                in_channels=3,
                out_channels=16,
                kernel_size=3,
                stride=1,
                padding=1,
            ), 
            torch.nn.Conv2d(
                in_channels=16,
                out_channels=32,
                kernel_size=3,
                stride=1,
                padding=1,
            ),
        ],
        input_nn=None,
        input_result_height=None,
        input_result_width=None,
        input_result_channel=None,
        output_nn=None,
        hidden_dropout_rate_list=[0.5, 0.5],
        hidden_batch_norm_list=[
            torch.nn.BatchNorm2d(16), 
            torch.nn.BatchNorm2d(32),
        ],
        hidden_activation_list=[
            torch.nn.functional.tanh, 
            torch.nn.functional.sigmoid
        ],
        hidden_residual_flag=False,
        hidden_dense_flag=False,
        dense_axis=1,
        ctx="cpu",
        regularizatable_data_list=[],
        scale=1.0,
        not_init_flag=False,
    ):
        '''
        Init.

        Args:
            computable_loss:                is-a `ComputableLoss` or `mxnet.gluon.loss`.
            initializer:                    is-a `mxnet.initializer` for parameters of model. If `None`, it is drawing from the Xavier distribution.
            learning_rate:                  `float` of learning rate.
            hidden_units_list:              `list` of `mxnet.gluon.nn._conv` in hidden layers.
            input_nn:                       is-a `NeuralNetworks` as input layers.
                                            If `None`, first layer in `hidden_units_list` will be considered as as input layer.

            output_nn:                      is-a `NeuralNetworks` as output layers.
                                            If `None`, last layer in `hidden_units_list` will be considered as an output layer.

            hidden_dropout_rate_list:       `list` of `float` of dropout rate in hidden layers.
            hidden_batch_norm_list:         `list` of `mxnet.gluon.nn.BatchNorm` in hidden layers.

            optimizer_name:                 `str` of name of optimizer.

            hidden_activation_list:         `list` of act_type` in `mxnet.ndarray.Activation` or `mxnet.symbol.Activation` in input gate.

            hidden_residual_flag:           `bool` whether execute the residual learning or not in hidden layers.

            ctx:                            `mx.cpu()` or `mx.gpu()`.
            hybridize_flag:                  Call `mxnet.gluon.HybridBlock.hybridize()` or not.
            regularizatable_data_list:      `list` of `RegularizatableData`.
            scale:                          `float` of scaling factor for initial parameters.
            not_init_flag:                  `bool` of whether initialize parameters or not.
        '''
        if len(hidden_units_list) != len(hidden_activation_list):
            raise ValueError("The length of `hidden_units_list` and `hidden_activation_list` must be equivalent.")

        if len(hidden_dropout_rate_list) != len(hidden_units_list):
            raise ValueError("The length of `hidden_dropout_rate_list` and `hidden_units_list` must be equivalent.")

        if len(hidden_batch_norm_list) != len(hidden_units_list):
            raise ValueError("The length of `hidden_batch_norm_list` and `hidden_units_list` must be equivalent.")

        if isinstance(computable_loss, ComputableLoss) is False and isinstance(computable_loss, nn.modules.loss._Loss) is False:
            raise TypeError("The type of `computable_loss` must be `ComputableLoss` or `nn.modules.loss._Loss`.")

        super(ConvolutionalNeuralNetworks, self).__init__()

        self.input_nn = input_nn
        self.output_nn = output_nn

        self.hidden_units_list = nn.ModuleList(hidden_units_list)
        self.hidden_batch_norm_list = nn.ModuleList(hidden_batch_norm_list)

        dropout_list = [None] * len(hidden_dropout_rate_list)
        for i in range(len(hidden_dropout_rate_list)):
            dropout_list[i] = nn.Dropout(p=hidden_dropout_rate_list[i])
        self.hidden_dropout_rate_list = nn.ModuleList(dropout_list)

        self.__not_init_flag = not_init_flag

        self.__computable_loss = computable_loss
        self.__learning_rate = learning_rate

        for v in regularizatable_data_list:
            if isinstance(v, RegularizatableData) is False:
                raise TypeError("The type of values of `regularizatable_data_list` must be `RegularizatableData`.")
        self.__regularizatable_data_list = regularizatable_data_list

        for i in range(len(hidden_units_list)):
            if initializer_f is None:
                hidden_units_list[i].weight = torch.nn.init.xavier_normal_(
                    hidden_units_list[i].weight,
                    gain=1.0
                )
            else:
                hidden_units_list[i].weight = initializer_f(hidden_units_list[i].weight)

        self.hidden_units_list = nn.ModuleList(hidden_units_list)

        self.optimizer_f = optimizer_f
        self.__ctx = ctx

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

        self.flatten = nn.Flatten()

        self.hidden_activation_list = hidden_activation_list

        self.__computable_loss = computable_loss
        self.__learning_rate = learning_rate

        self.input_result_height = input_result_height
        self.input_result_width = input_result_width
        self.input_result_channel = input_result_channel

        self.__hidden_residual_flag = hidden_residual_flag
        self.__hidden_dense_flag = hidden_dense_flag
        self.__dense_axis = dense_axis

        self.__safe_params_dict = {}

        self.epoch = 0
        self.__loss_list = []
        logger = getLogger("accelbrainbase")
        self.__logger = logger

    def learn(self, iteratable_data):
        '''
        Learn the observed data points
        for vector representation of the input images.

        Args:
            iteratable_data:     is-a `IteratableData`.

        '''
        if isinstance(iteratable_data, IteratableData) is False:
            raise TypeError("The type of `iteratable_data` must be `IteratableData`.")

        self.__loss_list = []
        self.__acc_list = []
        learning_rate = self.__learning_rate
        try:
            epoch = self.epoch
            iter_n = 0
            for batch_observed_arr, batch_target_arr, test_batch_observed_arr, test_batch_target_arr in iteratable_data.generate_learned_samples():
                self.epoch = epoch
                self.batch_size = batch_observed_arr.shape[0]

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

                    self.__logger.debug("Epochs: " + str(epoch + 1) + " Train loss: " + str(loss.to('cpu').detach().numpy().mean()) + " Test loss: " + str(test_loss.to('cpu').detach().numpy().mean()))
                    self.__loss_list.append((_loss, _test_loss))

                if self.compute_acc_flag is True:
                    if (iter_n+1) % int(iteratable_data.iter_n / iteratable_data.epochs) == 0:
                        acc, inferenced_label_arr, answer_label_arr = self.compute_acc(pred_arr, batch_target_arr)
                        test_acc, test_inferenced_label_arr, test_answer_label_arr = self.compute_acc(test_pred_arr, test_batch_target_arr)

                        if ((epoch + 1) % 100 == 0):
                            self.__logger.debug("-" * 100)
                            self.__logger.debug("Train accuracy: " + str(acc) + " Test accuracy: " + str(test_acc))
                            self.__logger.debug("Train infenreced label(inferenced):")
                            self.__logger.debug(inferenced_label_arr.to('cpu').detach().numpy())
                            self.__logger.debug("Train infenreced label(answer):")
                            self.__logger.debug(answer_label_arr.to('cpu').detach().numpy())

                            self.__logger.debug("Test infenreced label(inferenced):")
                            self.__logger.debug(test_inferenced_label_arr.to('cpu').detach().numpy())
                            self.__logger.debug("Test infenreced label(answer):")
                            self.__logger.debug(test_answer_label_arr.to('cpu').detach().numpy())
                            self.__logger.debug("-" * 100)

                            if (test_answer_label_arr[0].to('cpu').detach().numpy() == test_answer_label_arr.to('cpu').detach().numpy()).astype(int).sum() != test_answer_label_arr.shape[0]:
                                if (test_inferenced_label_arr[0].to('cpu').detach().numpy() == test_inferenced_label_arr.to('cpu').detach().numpy()).astype(int).sum() == test_inferenced_label_arr.shape[0]:
                                    self.__logger.debug("It may be overfitting.")

                        self.__acc_list.append(
                            (
                                acc,
                                test_acc
                            )
                        )

                if (iter_n+1) % int(iteratable_data.iter_n / iteratable_data.epochs) == 0:
                    epoch += 1
                iter_n += 1

        except KeyboardInterrupt:
            self.__logger.debug("Interrupt.")

        self.__logger.debug("end. ")
        self.epoch = epoch

    def inference(self, observed_arr):
        '''
        Inference the labels.

        Args:
            observed_arr:   rank-4 Array like or sparse matrix as the observed data points.
                            The shape is: (batch size, channel, height, width)

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

    def forward(self, x):
        '''
        Hybrid forward with Gluon API.

        Args:
            x:      `tensor` of observed data points.
        
        Returns:
            `tensor` of inferenced feature points.
        '''
        if self.input_nn is not None:
            x = self.input_nn(x)
            x = torch.reshape(
                x, 
                shape=(
                    -1, 
                    self.input_result_channel, 
                    self.input_result_height, 
                    self.input_result_width
                )
            )

        for i in range(len(self.hidden_activation_list)):
            if i > 0 and i + 1 < len(self.hidden_activation_list):
                if self.__hidden_residual_flag is True or self.__hidden_dense_flag is True:
                    _x = x

            x = self.hidden_units_list[i](x)
            if self.hidden_activation_list[i] == "identity_adjusted":
                x = x / torch.sum(torch.ones_like(x))
            elif self.hidden_activation_list[i] != "identity":
                x = self.hidden_activation_list[i](x)

            if self.hidden_dropout_rate_list[i] is not None:
                x = self.hidden_dropout_rate_list[i](x)

            if self.hidden_batch_norm_list[i] is not None:
                x = self.hidden_batch_norm_list[i](x)

            if i > 0 and i + 1 < len(self.hidden_activation_list):
                if self.__hidden_residual_flag is True:
                    x = torch.add(_x, x)
                elif self.__hidden_dense_flag is True:
                    x = F.cat((_x, x), dim=self.__dense_axis)

                if self.__hidden_residual_flag is True or self.__hidden_dense_flag is True:
                    _x = x

        if self.output_nn is not None:
            x = self.output_nn(x)

        return x

    def compute_acc(self, prob_arr, batch_target_arr):
        '''
        Compute accuracy.

        Args:
            prob_arr:               Softmax probabilities.
            batch_target_arr:       t-hot vectors.
        
        Returns:
            Tuple data.
            - Accuracy.
            - inferenced label.
            - real label.
        '''
        inferenced_label_arr = prob_arr.argmax(axis=1)
        answer_label_arr = batch_target_arr.argmax(axis=1)
        acc = (inferenced_label_arr == answer_label_arr).sum() / self.batch_size
        return acc.to('cpu').detach().numpy(), inferenced_label_arr, answer_label_arr

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
        ''' setter '''
        raise TypeError("This property must be read-only.")

    def set_batch_size(self, value):
        ''' setter for batch size. '''
        self.__batch_size = value

    def get_batch_size(self):
        ''' getter for batch size. '''
        return self.__batch_size
    
    batch_size = property(get_batch_size, set_batch_size)

    def get_loss_arr(self):
        ''' getter for for `list` of accuracies. '''
        return np.array(self.__loss_list)

    loss_arr = property(get_loss_arr, set_readonly)

    def get_acc_list(self):
        ''' getter for `list` of accuracies. '''
        return np.array(self.__acc_list)
    
    acc_arr = property(get_acc_list, set_readonly)

    def get_init_deferred_flag(self):
        ''' getter for `bool` that means initialization in this class will be deferred or not.'''
        return self.__init_deferred_flag
    
    def set_init_deferred_flag(self, value):
        ''' setter for `bool` that means initialization in this class will be deferred or not. '''
        self.__init_deferred_flag = value

    init_deferred_flag = property(get_init_deferred_flag, set_init_deferred_flag)
