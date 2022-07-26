# -*- coding: utf-8 -*-
from accelbrainbase.observabledata._torch.neuralnetworks.auto_encoder import AutoEncoder
from accelbrainbase.observabledata._torch.neural_networks import NeuralNetworks
from accelbrainbase.iteratable_data import IteratableData
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim.optimizer import Optimizer, required
from torch.optim.adam import Adam
from logging import getLogger


class VariationalAutoEncoder(AutoEncoder):
    """
    Variational Auto-Encoder.

    References:
        - Kingma, D. P., & Welling, M. (2014). Auto-encoding variational Bayes, May 2014. arXiv preprint arXiv:1312.6114.
    """

    def __init__(
        self,
        encoder,
        decoder,
        computable_loss,
        learning_rate=1e-05,
        ctx="cpu",
        tied_weights_flag=False,
        not_init_flag=False,
        hidden_dim=None,
    ):
        '''
        Init.

        Args:
            encoder:                        is-a `NeuralNetworks`.
            decoder:                        is-a `NeuralNetworks`.
            computable_loss:                is-a `ComputableLoss` or `mxnet.gluon.loss`.
            learning_rate:                  `float` of learning rate.
            observed_activation:            `act_type` in `mxnet.ndarray.Activation` or `mxnet.symbol.Activation` 
                                            that activates observed data points.

            ctx:                            Context-manager that changes the selected device.
            tied_weights_flag:              `bool` of flag to tied weights or not.
            not_init_flag:                  `bool` of whether initialize parameters or not.
        '''
        super().__init__(
            encoder=encoder,
            decoder=decoder,
            computable_loss=computable_loss,
            learning_rate=learning_rate,
            ctx=ctx,
            tied_weights_flag=tied_weights_flag,
            not_init_flag=not_init_flag,
        )
        self.__ctx = ctx
        self.__learning_rate = learning_rate
        self.__not_init_flag = not_init_flag
        self.__hidden_dim = hidden_dim
        self.__computable_loss = computable_loss
        logger = getLogger("accelbrainbase")
        self.__logger = logger

    def parameters(self):
        '''
        '''
        return [
            {
                "params": self.encoder.parameters(),
            },
            {
                "params": self.decoder.parameters(),
            },
            {
                "params": self.encoder_mu.parameters(),
            },
            {
                "params": self.encoder_var.parameters(),
            }
        ]

    def initialize_params(self, input_dim):
        '''
        Initialize params.
        '''
        if self.encoder_optimizer is not None and self.decoder_optimizer is not None:
            return

        self.__encoder_input_dim = input_dim
        self.__decoder_input_dim = self.__hidden_dim

        self.encoder_mu = nn.Linear(
            self.encoder.units_list[-1],
            self.__hidden_dim
        )
        self.encoder_var = nn.Linear(
            self.encoder.units_list[-1],
            self.__hidden_dim
        )
        self.to(self.__ctx)

        if self.initializer_f is None:
            self.encoder_mu.weight = torch.nn.init.xavier_normal_(
                self.encoder_mu.weight,
                gain=1.0
            )
            self.encoder_var.weight = torch.nn.init.xavier_normal_(
                self.encoder_var.weight,
                gain=1.0
            )
        else:
            self.encoder_mu.weight = self.initializer_f(self.encoder_mu.weight)
            self.encoder_var.weight = self.initializer_f(self.encoder_var.weight)

        if self.init_deferred_flag is False:
            if self.__not_init_flag is False:
                self.encoder.initialize_params(self.__encoder_input_dim)
                self.decoder.initialize_params(self.__decoder_input_dim)
                self.encoder_optimizer = self.encoder.optimizer
                self.decoder_optimizer = self.decoder.optimizer

                if self.optimizer_f is None:
                    self.encoder_mu_optimizer = Adam(
                        self.encoder_mu.parameters(), 
                        lr=self.__learning_rate,
                    )
                    self.encoder_var_optimizer = Adam(
                        self.encoder_var.parameters(), 
                        lr=self.__learning_rate,
                    )
                else:
                    self.encoder_mu_optimizer = self.optimizer_f(
                        self.encoder_mu_optimizer.parameters(), 
                    )
                    self.encoder_var_optimizer = self.optimizer_f(
                        self.encoder_var_optimizer.parameters(), 
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
        try:
            epoch = self.epoch
            iter_n = 0
            for batch_observed_arr, batch_target_arr, test_batch_observed_arr, test_batch_target_arr in iteratable_data.generate_learned_samples():
                batch_size = batch_observed_arr.shape[0]
                self.initialize_params(
                    input_dim=batch_observed_arr.reshape(batch_size, -1).shape[1]
                )
                self.encoder_optimizer.zero_grad()
                self.encoder_mu_optimizer.zero_grad()
                self.encoder_var_optimizer.zero_grad()
                self.decoder_optimizer.zero_grad()
                # rank-3
                pred_arr = self.inference(batch_observed_arr)
                loss = self.compute_loss(
                    pred_arr,
                    batch_target_arr
                )
                loss.backward()
                self.encoder_optimizer.step()
                self.encoder_mu_optimizer.step()
                self.encoder_var_optimizer.step()
                self.decoder_optimizer.step()
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
        Inference the feature points.

        Args:
            observed_arr:   rank-2 Array like or sparse matrix as the observed data points.
                            The shape is: (batch size, feature points)

        Returns:
            `tensor` of inferenced feature points.
        '''
        pred_arr = self.forward(observed_arr)
        return pred_arr

    def compute_loss(self, pred_arr, labeled_arr):
        '''
        Compute loss.

        Args:
            pred_arr:       `tensor`.
            labeled_arr:    `tensor`.

        Returns:
            loss.
        '''
        labeled_arr = pred_arr.reshape_as(pred_arr)
        reconstruction_loss = self.__computable_loss(pred_arr, labeled_arr)
        kl_loss = -0.5 * torch.sum(1 + self.__log_var_arr - self.__mu_arr.pow(2) - self.__log_var_arr.exp())
        loss = reconstruction_loss + kl_loss
        return loss

    def forward(self, x):
        '''
        Forward with torch.

        Args:
            x:      `tensor` of observed data points.
        
        Returns:
            `tensor` of inferenced feature points.
        '''
        batch_size = x.shape[0]
        self.initialize_params(
            input_dim=x.reshape(batch_size, -1).shape[1]
        )

        encoded_arr = self.encoder.inference(x)
        mu_arr = self.encoder_mu(encoded_arr)
        log_var_arr = self.encoder_var(encoded_arr)
        eps_arr = torch.randn_like(torch.exp(log_var_arr))
        Z_arr = mu_arr + torch.exp(log_var_arr / 2) * eps_arr

        self.feature_points_arr = Z_arr
        self.__mu_arr = mu_arr
        self.__log_var_arr = log_var_arr
        decoded_arr = self.decoder.inference(Z_arr)
        self.__pred_arr = decoded_arr
        return decoded_arr

    def __rename_file(self, filename):
        filename_list = filename.split(".")
        _format = filename_list[-1]
        encoder_filename = filename.replace("." + _format, "_encoder." + _format)
        decoder_filename = filename.replace("." + _format, "_decoder." + _format)
        return encoder_filename, decoder_filename

    def save_parameters(self, filename):
        '''
        Save parameters to files.

        Args:
            filename:       File name.
        '''
        encoder_filename, decoder_filename = self.__rename_file(filename)
        self.encoder.save_parameters(encoder_filename)
        self.decoder.save_parameters(decoder_filename)
        torch.save(
            {
                'epoch': self.epoch,
                'loss': self.loss_arr,
                'encoder_mu_state_dict': self.encoder_mu.state_dict(),
                'encoder_var_state_dict': self.encoder_var.state_dict(),
                'encoder_mu_optimizer_state_dict': self.encoder_mu_optimizer.state_dict(),
                'encoder_var_optimizer_state_dict': self.encoder_var_optimizer.state_dict(),
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
        encoder_filename, decoder_filename = self.__rename_file(filename)
        self.encoder.load_parameters(encoder_filename, ctx=ctx, strict=strict)
        self.decoder.load_parameters(decoder_filename, ctx=ctx, strict=strict)

        checkpoint = torch.load(filename)
        self.epoch = checkpoint['epoch']
        self.__loss_list = checkpoint['loss'].tolist()

        self.encoder_mu.load_state_dict(checkpoint['encoder_mu_state_dict'], strict=strict)
        self.encoder_var.load_state_dict(checkpoint['encoder_var_state_dict'], strict=strict)
        self.encoder_mu_optimizer.load_state_dict(
            checkpoint['encoder_mu_optimizer_state_dict']
        )
        self.encoder_var_optimizer.load_state_dict(
            checkpoint['encoder_var_optimizer_state_dict']
        )

    def set_readonly(self, value):
        ''' setter '''
        raise TypeError("This property must be read-only.")

    def get_loss_arr(self):
        ''' getter for losses. '''
        return np.array(self.__loss_list)

    loss_arr = property(get_loss_arr, set_readonly)

