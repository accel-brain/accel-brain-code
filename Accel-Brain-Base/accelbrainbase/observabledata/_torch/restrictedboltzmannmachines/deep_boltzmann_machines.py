# -*- coding: utf-8 -*-
import numpy as np
import torch
from torch import nn

import warnings
from accelbrainbase.observabledata._torch.restricted_boltzmann_machines import RestrictedBoltzmannMachines
from accelbrainbase.iteratable_data import IteratableData
from logging import getLogger


class DeepBoltzmannMachines(RestrictedBoltzmannMachines):
    '''
    Deep Boltzmann Machines(DBM).
    
    As is well known, DBM is composed of layers of RBMs 
    stacked on top of each other(Salakhutdinov, R., & Hinton, G. E. 2009). 
    This model is a structural expansion of Deep Belief Networks(DBN), 
    which is known as one of the earliest models of Deep Learning
    (Le Roux, N., & Bengio, Y. 2008). Like RBM, DBN places nodes in layers. 
    However, only the uppermost layer is composed of undirected edges, 
    and the other consists of directed edges.

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
        - Salakhutdinov, R., & Hinton, G. E. (2009). Deep boltzmann machines. InInternational conference on artificial intelligence and statistics (pp. 448-455).

    '''

    # `bool` that means initialization in this class will be deferred or not.
    __init_deferred_flag = False

    def __init__(
        self, 
        computable_loss,
        rbm_list,
        learning_rate=0.005,
        ctx="cpu",
    ):
        '''
        Init.
        
        Args:
            computable_loss:            is-a `ComputableLoss`.
            rbm_list:                   `list` of `RestrictedBoltzmannMachines`s.
            initializer:                is-a `mxnet.initializer` for parameters of model. If `None`, it is drawing from the Xavier distribution.
            optimizer_name:             `str` of name of optimizer.
            learning_rate:              `float` of learning rate.
            learning_attenuate_rate:    `float` of attenuate the `learning_rate` by a factor of this value every `attenuate_epoch`.
            attenuate_epoch:            `int` of attenuate the `learning_rate` by a factor of `learning_attenuate_rate` every `attenuate_epoch`.
            ctx:                        `mx.gpu()` or `mx.cpu()`.

        '''
        for rbm in rbm_list:
            if isinstance(rbm, RestrictedBoltzmannMachines) is False:
                raise TypeError("The type of values of `rbm_list` must be `RestrictedBoltzmannMachines`.")
        self.rbm_list = rbm_list

        init_deferred_flag = self.init_deferred_flag
        self.init_deferred_flag = True
        super().__init__(
            computable_loss=computable_loss,
            learning_rate=learning_rate,
            ctx=ctx,
        )
        self.init_deferred_flag = init_deferred_flag

        self.epoch = 0
        logger = getLogger("accelbrainbase")
        self.__logger = logger

    def parameters(self):
        '''
        '''
        params_dict_list = []
        for rbm in self.rbm_list:
            params_dict_list.append({"params": rbm.parameters()})

        return params_dict_list

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

                for rbm in self.rbm_list:
                    rbm.optimizer.zero_grad()

                visible_activity_arr = self.inference(observed_arr)
                loss = self.compute_loss(
                    observed_arr,
                    visible_activity_arr
                )
                loss.backward()
                for rbm in self.rbm_list:
                    rbm.optimizer.step()
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

    def forward(self, x):
        '''
        Inference samples drawn by `IteratableData.generate_inferenced_samples()`.

        Args:
            observed_arr:   rank-2 Array like or sparse matrix as the observed data points.
                            The shape is: (batch size, feature points)

        Returns:
            `mxnet.ndarray` of inferenced feature points.
        '''
        self.batch_size = x.shape[0]
        observed_arr = x.reshape((self.batch_size, -1))
        for i in range(len(self.rbm_list)):
            self.rbm_list[i].batch_size = self.batch_size
            _ = self.rbm_list[i].inference(observed_arr)
            observed_arr = self.rbm_list[i].feature_points_arr

        return observed_arr

    def regularize(self):
        '''
        Regularization.
        '''
        for i in range(len(self.rbm_list)):
            self.rbm_list[i].regularize()

    def __rename_file(self, filename):
        filename_list = filename.split(".")
        _format = filename_list[-1]

        rbm_filename_list = []
        for rbm_key in range(len(self.rbm_list)):
            rbm_filename = filename.replace("." + _format, "_" + str(rbm_key) + "." + _format)
            rbm_filename_list.append(rbm_filename)
        return rbm_filename_list

    def save_parameters(self, filename):
        '''
        Save parameters to files.

        Args:
            filename:       File name.
        '''
        torch.save(
            {
                'epoch': self.epoch,
                'loss': self.loss_arr,
            }, 
            filename
        )

        filename_list = self.__rename_file(filename)
        for i in range(len(filename_list)):
            torch.save(
                {
                    'model_state_dict': self.rbm_list[i].state_dict(),
                    'optimizer_state_dict': self.rbm_list[i].optimizer.state_dict(),
                }, 
                filename_list[i]
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
        self.epoch = checkpoint['epoch']
        self.__loss_list = checkpoint['loss'].tolist()

        filename_list = self.__rename_file(filename)
        for i in range(len(filename_list)):
            checkpoint = torch.load(filename_list[i])
            self.rbm_list[i].load_state_dict(
                checkpoint['model_state_dict'], 
                strict=strict
            )
            self.rbm_list[i].optimizer.load_state_dict(
                checkpoint['optimizer_state_dict']
            )
            if ctx is not None:
                self.rbm_list[i].to(ctx)

        if ctx is not None:
            self.to(ctx)
            self.__ctx = ctx

    def set_readonly(self, value):
        ''' setter '''
        raise TypeError("This is read-only.")

    def get_feature_points_arr(self):
        ''' getter for `mxnet.ndarray` of feature points.'''
        return self.__hidden_activity_arr

    feature_points_arr = property(get_feature_points_arr, set_readonly)

    def get_init_deferred_flag(self):
        ''' getter for `bool` that means initialization in this class will be deferred or not.'''
        return self.__init_deferred_flag
    
    def set_init_deferred_flag(self, value):
        ''' setter for `bool` that means initialization in this class will be deferred or not.'''
        self.__init_deferred_flag = value

    init_deferred_flag = property(get_init_deferred_flag, set_init_deferred_flag)
