# -*- coding: utf-8 -*-
import numpy as np
import torch
from torch import nn
from torch.optim.sgd import SGD

import warnings
from accelbrainbase.observable_data import ObservableData
from accelbrainbase.iteratable_data import IteratableData
from accelbrainbase.regularizatable_data import RegularizatableData
from logging import getLogger


class RestrictedBoltzmannMachines(nn.Module, ObservableData):
    '''
    Restricted Boltzmann Machines(RBM).
    
    According to graph theory, the structure of RBM corresponds to 
    a complete bipartite graph which is a special kind of bipartite 
    graph where every node in the visible layer is connected to every 
    node in the hidden layer. Based on statistical mechanics and 
    thermodynamics(Ackley, D. H., Hinton, G. E., & Sejnowski, T. J. 1985), 
    the state of this structure can be reflected by the energy function.

    In relation to RBM, the Contrastive Divergence(CD) is a method for 
    approximation of the gradients of the log-likelihood(Hinton, G. E. 2002).
    This algorithm draws a distinction between a positive phase and a 
    negative phase. Conceptually, the positive phase is to the negative 
    phase what waking is to sleeping.

    The procedure of this method is similar to Markov Chain Monte Carlo method(MCMC).
    However, unlike MCMC, the visbile variables to be set first in visible layer is 
    not randomly initialized but the observed data points in training dataset are set 
    to the first visbile variables. And, like Gibbs sampler, drawing samples from hidden 
    variables and visible variables is repeated k times. Empirically (and surprisingly), 
    `k` is considered to be `1`.

    **Note** that this class does not support a *Hybrid* of imperative and symbolic 
    programming. Only `mxnet.ndarray` is supported.

    References:
        - Ackley, D. H., Hinton, G. E., & Sejnowski, T. J. (1985). A learning algorithm for Boltzmann machines. Cognitive science, 9(1), 147-169.
        - Hinton, G. E. (2002). Training products of experts by minimizing contrastive divergence. Neural computation, 14(8), 1771-1800.
        - Le Roux, N., & Bengio, Y. (2008). Representational power of restricted Boltzmann machines and deep belief networks. Neural computation, 20(6), 1631-1649.
    '''

    # The list of losses.
    __loss_arr = []
    # Learning rate.
    __learning_rate = 0.5
    # Batch size in learning.
    __batch_size = 0
    # Batch size in inference(recursive learning or not).
    __r_batch_size = 0

    def __init__(
        self,
        computable_loss,
        initializer_f=None,
        optimizer_f=None,
        visible_activation=torch.nn.Sigmoid(),
        hidden_activation=torch.nn.Sigmoid(),
        visible_dim=1000,
        hidden_dim=100,
        learning_rate=0.005,
        visible_dropout_rate=0.0,
        hidden_dropout_rate=0.0,
        visible_batch_norm=None,
        hidden_batch_norm=None,
        regularizatable_data_list=[],
        ctx="cpu",
    ):
        '''
        Init.
        
        Args:
            computable_loss:            is-a `ComputableLoss`.
            visible_activation:         `mxnet.ndarray.Activation` or `mxnet.symbol.Activation` in visible layer.
            hidden_activation:          `mxnet.ndarray.Activation` or `mxnet.symbol.Activation` in hidden layer.
            visible_dim:                `int` of dimension in visible layer.
            hidden_dim:                 `int` of dimension in hidden layer.
            initializer:                is-a `mxnet.initializer` for parameters of model. If `None`, it is drawing from the Xavier distribution.
            optimizer_name:             `str` of name of optimizer.
            learning_rate:              `float` of learning rate.
            learning_attenuate_rate:    `float` of attenuate the `learning_rate` by a factor of this value every `attenuate_epoch`.
            attenuate_epoch:            `int` of attenuate the `learning_rate` by a factor of `learning_attenuate_rate` every `attenuate_epoch`.
            visible_dropout_rate:       `float` of dropout rate in visible layer.
            hidden_dropout_rate:        `float` of dropout rate in hidden layer.
            visible_batch_norm:         `gluon.nn.BatchNorm` in visible layer.
            hidden_batch_norm:          `gluon.nn.BatchNorm` in hidden layer.
            regularizatable_data_list:  `list` of `RegularizatableData`s.
            ctx:                        `mx.gpu()` or `mx.cpu()`.
        '''
        super(RestrictedBoltzmannMachines, self).__init__()

        for v in regularizatable_data_list:
            if isinstance(v, RegularizatableData) is False:
                raise TypeError("The type of values of `regularizatable_data_list` must be `RegularizatableData`.")
        self.__regularizatable_data_list = regularizatable_data_list

        self.__computable_loss = computable_loss
        self.visible_activation = visible_activation
        self.hidden_activation = hidden_activation

        self.__visible_unit = nn.Linear(
            visible_dim,
            hidden_dim, 
            bias=True, 
        )

        self.visible_dropout_forward = None
        if visible_dropout_rate > 0:
            self.visible_dropout_forward = nn.Dropout(
                p=visible_dropout_rate
            )

        self.hidden_dropout_forward = None
        if hidden_dropout_rate > 0:
            self.hidden_dropout_forward = nn.Dropout(
                p=hidden_dropout_rate
            )

        self.visible_batch_norm = visible_batch_norm
        self.hidden_batch_norm = hidden_batch_norm

        self.__ctx = ctx
        self.to(self.__ctx)

        if initializer_f is not None:
            self.__visible_unit.weight = initializer_f(
                self.__visible_unit.weight
            )
        else:
            self.__visible_unit.weight = torch.nn.init.xavier_normal_(
                self.__visible_unit.weight,
                gain=1.0
            )

        if optimizer_f is not None:
            self.optimizer = optimizer_f(
                self.parameters()
            )
        else:
            self.optimizer = SGD(
                self.parameters(), 
                lr=self.__learning_rate,
            )

        self.__learning_rate = learning_rate

        self.__loss_arr = np.array([])

        logger = getLogger("accelbrainbase")
        self.__logger = logger

        self.__loss_list = []
        self.__test_loss_list = []

        self.epoch = 0

    def learn(self, iteratable_data):
        '''
        Learn samples drawn by `IteratableData.generate_learned_samples()`.

        Args:
            iteratable_data:     is-a `IteratableData`.

        '''
        if isinstance(iteratable_data, IteratableData) is False:
            raise TypeError("The type of `iteratable_data` must be `IteratableData`.")

        self.__loss_list = []
        self.__test_loss_list = []

        try:
            epoch = self.epoch
            iter_n = 0
            for observed_arr, label_arr, test_observed_arr, test_label_arr in iteratable_data.generate_learned_samples():
                self.batch_size = observed_arr.shape[0]
                observed_arr = observed_arr.reshape((self.batch_size, -1))
                test_observed_arr = test_observed_arr.reshape((self.batch_size, -1))

                self.optimizer.zero_grad()
                visible_activity_arr = self.inference(observed_arr)
                loss = self.compute_loss(
                    observed_arr,
                    visible_activity_arr
                )
                loss.backward()
                self.optimizer.step()
                self.regularize()

                if (iter_n+1) % int(iteratable_data.iter_n / iteratable_data.epochs) == 0:
                    with torch.inference_mode():
                        test_visible_activity_arr = self.inference(test_observed_arr)
                        test_loss = self.compute_loss(
                            test_observed_arr,
                            test_visible_activity_arr
                        )
                    _loss = loss.to('cpu').detach().numpy().copy()
                    _test_loss = test_loss.to('cpu').detach().numpy().copy()

                    self.__loss_list.append(_loss)
                    self.__test_loss_list.append(_test_loss)
                    self.__logger.debug("Epoch: " + str(epoch + 1) + " Train loss: " + str(self.__loss_list[-1]) + " Test loss: " + str(self.__test_loss_list[-1]))
                    epoch += 1
                iter_n += 1

        except KeyboardInterrupt:
            self.__logger.debug("Interrupt.")

        self.__logger.debug("end. ")

        self.__loss_arr = np.c_[
            np.array(self.__loss_list[:len(self.__test_loss_list)]),
            np.array(self.__test_loss_list)
        ]
        self.epoch = epoch

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
        Forward with Gluon API.

        Args:
            F:      `mxnet.ndarray` or `mxnet.symbol`.
            x:      `mxnet.ndarray` of observed data points.
        
        Returns:
            `mxnet.ndarray` or `mxnet.symbol` of inferenced feature points.
        '''
        self.batch_size = x.shape[0]
        x = x.reshape((self.batch_size, -1))

        self.__visible_activity_arr = x

        x = self.__visible_unit(x)

        if self.visible_activation == "identity_adjusted":
            x = x / torch.sum(torch.ones_like(x))
        elif self.visible_activation != "identity":
            x = self.visible_activation(x)

        if self.visible_dropout_forward is not None:
            x = self.visible_dropout_forward(x)
        if self.visible_batch_norm is not None:
            x = self.visible_batch_norm(x)

        self.__hidden_activity_arr = x

        self.__diff_weights_arr = torch.mm(
            self.__visible_activity_arr.T,
            self.__hidden_activity_arr,
        )

        #self.__visible_diff_bias_arr += nd.nansum(self.__visible_activity_arr, axis=0)
        #self.__hidden_diff_bias_arr += nd.nansum(self.__hidden_activity_arr, axis=0)

        params_dict = self.extract_learned_dict()
        weight_keys_list = [key for key in params_dict.keys() if "weight" in key]
        weights_arr = params_dict[weight_keys_list[0]]

        self.__visible_activity_arr = torch.mm(
            self.__hidden_activity_arr,
            weights_arr,
        )
        x = self.__visible_activity_arr
        if self.hidden_activation == "identity_adjusted":
            x = x / torch.sum(torch.ones_like(x))
        elif self.hidden_activation != "identity":
            x = self.hidden_activation(x)
        if self.hidden_dropout_forward is not None:
            x = self.hidden_dropout_forward(x)
        if self.hidden_batch_norm is not None:
            x = self.hidden_batch_norm(x)
        self.__visible_activity_arr = x

        self.__hidden_activity_arr = self.__visible_unit(
            self.__visible_activity_arr
        )
        x = self.__hidden_activity_arr
        if self.visible_activation == "identity_adjusted":
            x = x / torch.sum(torch.ones_like(x))
        elif self.visible_activation != "identity":
            x = self.visible_activation(x)
        if self.visible_dropout_forward is not None:
            x = self.visible_dropout_forward(x)
        if self.visible_batch_norm is not None:
            x = self.visible_batch_norm(x)
        self.__hidden_activity_arr = x

        self.__diff_weights_arr = self.__diff_weights_arr - torch.mm(
            self.__visible_activity_arr.T,
            self.__hidden_activity_arr,
        )
        #self.__visible_diff_bias_arr -= nd.nansum(self.__visible_activity_arr, axis=0)
        #self.__hidden_diff_bias_arr -= nd.nansum(self.__hidden_activity_arr, axis=0)

        return self.__visible_activity_arr

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
        raise TypeError("This is read-only.")

    def get_loss_list(self):
        ''' getter for `list` of losses in training. '''
        return self.__loss_list

    loss_list = property(get_loss_list, set_readonly)

    def get_test_loss_arr(self):
        ''' getter for `list` of losses in test. '''
        return self.__test_loss_list

    test_loss_list = property(get_test_loss_arr, set_readonly)

    def get_loss_arr(self):
        ''' getter for losses. '''
        return self.__loss_arr

    loss_arr = property(get_loss_arr, set_readonly)

    def get_feature_points_arr(self):
        ''' getter for `mxnet.narray` of feature points in middle hidden layer. '''
        return self.__hidden_activity_arr

    feature_points_arr = property(get_feature_points_arr, set_readonly)

    def get_weights_arr(self):
        ''' getter for `mxnet.ndarray` of weights matrics. '''
        return self.__weights_arr

    def set_weights_arr(self, value):
        ''' setter for `mxnet.ndarray` of weights matrics.'''
        self.__weights_arr = value
    
    weights_arr = property(get_weights_arr, set_weights_arr)

    def get_visible_bias_arr(self):
        ''' getter for `mxnet.ndarray` of biases in visible layer.'''
        return self.__visible_bias_arr
    
    def set_visible_bias_arr(self, value):
        ''' setter for `mxnet.ndarray` of biases in visible layer.'''
        self.__visible_bias_arr = value
    
    visible_bias_arr = property(get_visible_bias_arr, set_visible_bias_arr)

    def get_hidden_bias_arr(self):
        ''' getter for `mxnet.ndarray` of biases in hidden layer.'''
        return self.__hidden_bias_arr
    
    def set_hidden_bias_arr(self, value):
        ''' setter for `mxnet.ndarray` of biases in hidden layer.'''
        self.__hidden_bias_arr = value
    
    hidden_bias_arr = property(get_hidden_bias_arr, set_hidden_bias_arr)

    def get_visible_activity_arr(self):
        ''' getter for `mxnet.ndarray` of activities in visible layer.'''
        return self.__visible_activity_arr

    def set_visible_activity_arr(self, value):
        ''' setter for `mxnet.ndarray` of activities in visible layer.'''
        self.__visible_activity_arr = value

    visible_activity_arr = property(get_visible_activity_arr, set_visible_activity_arr)

    def get_hidden_activity_arr(self):
        ''' getter for `mxnet.ndarray` of activities in hidden layer.'''
        return self.__hidden_activity_arr

    def set_hidden_activity_arr(self, value):
        ''' setter for `mxnet.ndarray` of activities in hidden layer.'''
        self.__hidden_activity_arr = value

    hidden_activity_arr = property(get_hidden_activity_arr, set_hidden_activity_arr)
