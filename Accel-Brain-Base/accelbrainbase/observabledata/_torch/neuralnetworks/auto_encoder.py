# -*- coding: utf-8 -*-
from accelbrainbase.observabledata._torch.neural_networks import NeuralNetworks
from accelbrainbase.iteratable_data import IteratableData
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim.optimizer import Optimizer, required
from torch.optim.adam import Adam
from logging import getLogger


class AutoEncoder(NeuralNetworks):
    '''
    Auto-Encoder.

    References:
        - Kamyshanska, H., & Memisevic, R. (2014). The potential energy of an autoencoder. IEEE transactions on pattern analysis and machine intelligence, 37(6), 1261-1273.
    '''

    # `bool` that means initialization in this class will be deferred or not.
    __init_deferred_flag = False

    def __init__(
        self,
        encoder,
        decoder,
        computable_loss,
        learning_rate=1e-05,
        ctx="cpu",
        tied_weights_flag=False,
        not_init_flag=False,
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
        if isinstance(encoder, NeuralNetworks) is False:
            raise TypeError("The type of `encoder` must be `NeuralNetworks`.")
        if isinstance(decoder, NeuralNetworks) is False:
            raise TypeError("The type of `decoder` must be `NeuralNetworks`.")

        init_deferred_flag = self.init_deferred_flag
        self.init_deferred_flag = True
        super(AutoEncoder, self).__init__(computable_loss=computable_loss)
        self.init_deferred_flag = init_deferred_flag

        logger = getLogger("accelbrainbase")
        self.__logger = logger
        self.encoder = encoder
        self.decoder = decoder
        self.__tied_weights_flag = tied_weights_flag

        self.__computable_loss = computable_loss

        self.encoder_optimizer = None
        self.decoder_optimizer = None

        self.epoch = 0
        self.__learning_rate = learning_rate
        self.__not_init_flag = not_init_flag

        self.__ctx = ctx
        self.__encoder_input_dim = None
        self.__decoder_input_dim = None
        self.__loss_list = []

    def parameters(self):
        '''
        '''
        return [
            {
                "params": self.encoder.parameters(),
            },
            {
                "params": self.decoder.parameters(),
            }
        ]

    def initialize_params(self, input_dim):
        '''
        Initialize params.
        '''
        if self.encoder_optimizer is not None and self.decoder_optimizer is not None:
            return

        self.__encoder_input_dim = input_dim
        self.__decoder_input_dim = self.encoder.units_list[-1]
        if self.init_deferred_flag is False:
            if self.__not_init_flag is False:
                self.encoder.initialize_params(self.__encoder_input_dim)
                self.decoder.initialize_params(self.__decoder_input_dim)
                self.encoder_optimizer = self.encoder.optimizer
                self.decoder_optimizer = self.decoder.optimizer

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
                self.decoder_optimizer.zero_grad()
                # rank-3
                pred_arr = self.inference(batch_observed_arr)
                loss = self.compute_loss(
                    pred_arr,
                    batch_target_arr
                )
                loss.backward()
                self.encoder_optimizer.step()
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
        return self.__computable_loss(pred_arr, labeled_arr)

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
        params_arr_dict = {}

        params_dict = self.encoder.extract_learned_dict()
        for k in params_dict:
            params_arr_dict.setdefault(k, params_dict[k].data())

        params_dict = self.decoder.extract_learned_dict()
        for k in params_dict:
            params_arr_dict.setdefault(k, params_dict[k].data())

        return params_arr_dict

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
        self.feature_points_arr = encoded_arr
        decoded_arr = self.decoder.inference(encoded_arr)
        self.__pred_arr = decoded_arr
        return decoded_arr

    def regularize(self):
        '''
        Regularization.
        '''
        self.encoder.regularize()
        self.decoder.regularize()
        self.__tie_weights()

    def __tie_weights(self):
        if self.__tied_weights_flag is True:
            encoder_params_dict = self.encoder.extract_learned_dict()
            decoder_params_dict = self.decoder.extract_learned_dict()
            encoder_weight_keys_list = [key for key in encoder_params_dict.keys() if "fc_list" in key and "weight" in key]
            decoder_weight_keys_list = [key for key in decoder_params_dict.keys() if "fc_list" in key and "weight" in key]

            if len(encoder_weight_keys_list) != len(decoder_weight_keys_list):
                raise ValueError(
                    "The number of layers is invalid."
                )

            for i in range(len(self.encoder.units_list)):
                encoder_layer = i
                decoder_layer = len(self.encoder.units_list) - i - 1
                encoder_weight_keys, decoder_weight_keys = None, None
                for _encoder_weight_keys in encoder_weight_keys_list:
                    if "fc_list." + str(encoder_layer) + ".weight" in _encoder_weight_keys:
                        encoder_weight_keys = _encoder_weight_keys
                        break

                for _decoder_weight_keys in decoder_weight_keys_list:
                    if "fc_list." + str(decoder_layer) + ".weight" in _decoder_weight_keys:
                        decoder_weight_keys = _decoder_weight_keys
                        break

                if encoder_weight_keys is not None and decoder_weight_keys is not None:
                    try:
                        decoder_params_dict[decoder_weight_keys] = encoder_params_dict[encoder_weight_keys].T
                    except AssertionError:
                        raise ValueError(
                            "The shapes of weight matrixs must be equivalents in encoder layer " + str(encoder_layer) + " and decoder layer " + str(decoder_layer)
                        )

            for k, params in decoder_params_dict.items():
                if k in decoder_weight_keys_list:
                    self.decoder.load_state_dict({k: params}, strict=False)

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

    def set_readonly(self, value):
        ''' setter '''
        raise TypeError("This property must be read-only.")

    def get_loss_arr(self):
        ''' getter for losses. '''
        return np.array(self.__loss_list)

    loss_arr = property(get_loss_arr, set_readonly)

    def get_init_deferred_flag(self):
        ''' getter for `bool` that means initialization in this class will be deferred or not.'''
        return self.__init_deferred_flag
    
    def set_init_deferred_flag(self, value):
        ''' setter for `bool` that means initialization in this class will be deferred or not.'''
        self.__init_deferred_flag = value

    init_deferred_flag = property(get_init_deferred_flag, set_init_deferred_flag)
