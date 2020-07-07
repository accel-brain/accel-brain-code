# -*- coding: utf-8 -*-
from accelbrainbase.observabledata._mxnet.neural_networks import NeuralNetworks
from accelbrainbase._mxnet._exception.init_deferred_error import InitDeferredError

from mxnet.gluon.block import HybridBlock
from mxnet import gluon
from mxnet import autograd
import numpy as np
import mxnet as mx
from mxnet import MXNetError
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
        initializer=None,
        learning_rate=1e-05,
        learning_attenuate_rate=1.0,
        attenuate_epoch=50,
        units_list=[100, 1],
        dropout_rate_list=[0.0, 0.5],
        optimizer_name="SGD",
        activation_list=["tanh", "sigmoid"],
        ctx=mx.gpu(),
        hybridize_flag=True,
        regularizatable_data_list=[],
        scale=1.0,
        tied_weights_flag=False,
        **kwargs
    ):
        '''
        Init.

        Args:
            encoder:                        is-a `NeuralNetworks`.
            decoder:                        is-a `NeuralNetworks`.
            computable_loss:                is-a `ComputableLoss` or `mxnet.gluon.loss`.
            initializer:                    is-a `mxnet.initializer.Initializer` for parameters of model. If `None`, it is drawing from the Xavier distribution.
            learning_rate:                  `float` of learning rate.
            learning_attenuate_rate:        `float` of attenuate the `learning_rate` by a factor of this value every `attenuate_epoch`.
            attenuate_epoch:                `int` of attenuate the `learning_rate` by a factor of `learning_attenuate_rate` every `attenuate_epoch`.
                                            

            units_list:                     `list` of int` of the number of units in hidden/output layers.
            dropout_rate_list:              `list` of `float` of dropout rate.
            observed_activation:            `act_type` in `mxnet.ndarray.Activation` or `mxnet.symbol.Activation` 
                                            that activates observed data points.

            optimizer_name:                 `str` of name of optimizer.

            activation_list:                `list` of act_type` in `mxnet.ndarray.Activation` or `mxnet.symbol.Activation` in input gate.

            ctx:                            `mx.cpu()` or `mx.gpu()`.
            hybridize_flag:                  Call `mxnet.gluon.HybridBlock.hybridize()` or not.
            regularizatable_data_list:      `list` of `RegularizatableData`.
            scale:                          `float` of scaling factor for initial parameters.
            tied_weights_flag:              `bool` of flag to tied weights or not.
        '''
        if isinstance(encoder, NeuralNetworks) is False:
            raise TypeError("The type of `encoder` must be `NeuralNetworks`.")
        if isinstance(decoder, NeuralNetworks) is False:
            raise TypeError("The type of `decoder` must be `NeuralNetworks`.")

        logger = getLogger("accelbrainbase")
        self.__logger = logger
        init_deferred_flag = self.init_deferred_flag
        self.init_deferred_flag = True
        super().__init__(
            computable_loss=computable_loss,
            initializer=initializer,
            learning_rate=learning_rate,
            learning_attenuate_rate=learning_attenuate_rate,
            attenuate_epoch=attenuate_epoch,
            units_list=units_list,
            dropout_rate_list=dropout_rate_list,
            optimizer_name=optimizer_name,
            activation_list=activation_list,
            ctx=ctx,
            hybridize_flag=hybridize_flag,
            regularizatable_data_list=regularizatable_data_list,
            scale=scale,
            **kwargs
        )
        self.init_deferred_flag = init_deferred_flag
        self.encoder = encoder
        self.decoder = decoder
        self.__tied_weights_flag = tied_weights_flag
        if self.init_deferred_flag is False:
            try:
                self.collect_params().initialize(self.initializer, force_reinit=True, ctx=ctx)
                self.trainer = gluon.Trainer(self.collect_params(), optimizer_name, {"learning_rate": learning_rate})
                if hybridize_flag is True:
                    self.encoder.hybridize()
                    self.decoder.hybridize()
            except InitDeferredError:
                self.__logger.debug("The initialization should be deferred.")
        self.__computable_loss = computable_loss
        self.__units_list = units_list

    def collect_params(self, select=None):
        '''
        Overrided `collect_params` in `mxnet.gluon.HybridBlok`.
        '''
        params_dict = self.encoder.collect_params(select)
        params_dict.update(self.decoder.collect_params(select))
        return params_dict

    def inference(self, observed_arr):
        '''
        Inference the feature points.

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
        encoded_arr = self.encoder.forward_propagation(F, x)
        self.feature_points_arr = encoded_arr
        decoded_arr = self.decoder.forward_propagation(F, encoded_arr)
        self.__pred_arr = decoded_arr
        return decoded_arr

    def regularize(self):
        '''
        Regularization.
        '''
        self.__tie_weights()
        super().regularize()

    def __tie_weights(self):
        if self.__tied_weights_flag is True:
            encoder_params_dict = self.encoder.extract_learned_dict()
            decoder_params_dict = self.decoder.extract_learned_dict()
            encoder_weight_keys_list = [key for key in encoder_params_dict.keys() if "weight" in key]
            decoder_weight_keys_list = [key for key in decoder_params_dict.keys() if "weight" in key]

            for i in range(len(self.encoder.units_list)):
                encoder_layer = i
                decoder_layer = len(self.encoder.units_list) - i - 1
                encoder_weight_keys, decoder_weight_keys = None, None
                for _encoder_weight_keys in encoder_weight_keys_list:
                    if "_dense" + str(encoder_layer) + "_weight" in _encoder_weight_keys:
                        encoder_weight_keys = _encoder_weight_keys
                        break

                for _decoder_weight_keys in decoder_weight_keys_list:
                    if "_dense" + str(decoder_layer) + "_weight" in _decoder_weight_keys:
                        decoder_weight_keys = _decoder_weight_keys
                        break

                if encoder_weight_keys is not None and decoder_weight_keys is not None:
                    try:
                        decoder_params_dict[decoder_weight_keys] = encoder_params_dict[encoder_weight_keys].T
                    except AssertionError:
                        raise ValueError(
                            "The shapes of weight matrixs must be equivalents in encoder layer " + str(encoder_layer) + " and decoder layer " + str(decoder_layer)
                        )

            for k, params in self.decoder.collect_params().items():
                if k in decoder_weight_keys_list:
                    params.set_data(decoder_params_dict[k])

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

    def load_parameters(self, filename, ctx=None, allow_missing=False, ignore_extra=False):
        '''
        Load parameters to files.

        Args:
            filename:       File name.
            ctx:            `mx.cpu()` or `mx.gpu()`.
            allow_missing:  `bool` of whether to silently skip loading parameters not represents in the file.
            ignore_extra:   `bool` of whether to silently ignre parameters from the file that are not present in this `Block`.
        '''
        encoder_filename, decoder_filename = self.__rename_file(filename)
        self.encoder.load_parameters(encoder_filename, ctx=ctx, allow_missing=allow_missing, ignore_extra=ignore_extra)
        self.decoder.load_parameters(decoder_filename, ctx=ctx, allow_missing=allow_missing, ignore_extra=ignore_extra)

    def set_readonly(self, value):
        ''' setter '''
        raise TypeError("This property must be read-only.")

    def get_init_deferred_flag(self):
        ''' getter for `bool` that means initialization in this class will be deferred or not.'''
        return self.__init_deferred_flag
    
    def set_init_deferred_flag(self, value):
        ''' setter for `bool` that means initialization in this class will be deferred or not.'''
        self.__init_deferred_flag = value

    init_deferred_flag = property(get_init_deferred_flag, set_init_deferred_flag)
