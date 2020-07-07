# -*- coding: utf-8 -*-
from accelbrainbase.observabledata._mxnet.lstm_networks import LSTMNetworks
from accelbrainbase.iteratable_data import IteratableData
from accelbrainbase.regularizatable_data import RegularizatableData
from accelbrainbase.computable_loss import ComputableLoss
from accelbrainbase._mxnet._exception.init_deferred_error import InitDeferredError

import mxnet.ndarray as nd
import mxnet as mx
from mxnet import gluon
from logging import getLogger


class EncoderDecoder(LSTMNetworks):
    '''
    Encoder/Decoder based on LSTM networks.
    
    This library provides Encoder/Decoder based on LSTM, 
    which is a reconstruction model and makes it possible to extract 
    series features embedded in deeper layers. The LSTM encoder learns 
    a fixed length vector of time-series observed data points and the 
    LSTM decoder uses this representation to reconstruct the time-series 
    using the current hidden state and the value inferenced at the previous time-step.
    
    One interesting application example is the Encoder/Decoder for 
    Anomaly Detection (EncDec-AD) paradigm (Malhotra, P., et al. 2016).
    This reconstruction model learns to reconstruct normal time-series behavior, 
    and thereafter uses reconstruction error to detect anomalies. 
    Malhotra, P., et al. (2016) showed that EncDec-AD paradigm is robust 
    and can detect anomalies from predictable, unpredictable, periodic, aperiodic, 
    and quasi-periodic time-series. Further, they showed that the paradigm is able 
    to detect anomalies from short time-series (length as small as 30) as well as 
    long time-series (length as large as 500).

    References:
        - Cho, K., Van Merriënboer, B., Gulcehre, C., Bahdanau, D., Bougares, F., Schwenk, H., & Bengio, Y. (2014). Learning phrase representations using RNN encoder-decoder for statistical machine translation. arXiv preprint arXiv:1406.1078.
        - Malhotra, P., Ramakrishnan, A., Anand, G., Vig, L., Agarwal, P., & Shroff, G. (2016). LSTM-based encoder-decoder for multi-sensor anomaly detection. arXiv preprint arXiv:1607.00148.
    '''

    # is-a `LSTMNetworks`.
    __encoder = None

    # is-a `LSTMNetworks`.
    __decoder = None

    def __init__(
        self,
        encoder,
        decoder,
        computable_loss,
        initializer=None,
        batch_size=100,
        learning_rate=1e-05,
        learning_attenuate_rate=0.1,
        attenuate_epoch=50,
        seq_len=0,
        hidden_n=200,
        output_n=1,
        dropout_rate=0.5,
        optimizer_name="SGD",
        observed_activation="tanh",
        input_gate_activation="sigmoid",
        forget_gate_activation="sigmoid",
        output_gate_activation="sigmoid",
        hidden_activation="tanh",
        output_activation="tanh",
        output_layer_flag=True,
        ctx=mx.gpu(),
        hybridize_flag=True,
        regularizatable_data_list=[],
        scale=1.0,
        generating_flag=True,
        **kwargs
    ):
        '''
        Init.

        Override.

        Args:
            encoder:                        is-a `LSTMNetworks`.
            decoder:                        is-a `LSTMNetworks`.
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

            hidden_n:                       `int` of the number of units in hidden layers.
            output_n:                       `int` of the nuber of units in output layer.
            dropout_rate:                   `float` of dropout rate.
            observed_activation:            `act_type` in `mxnet.ndarray.Activation` or `mxnet.symbol.Activation` 
                                            that activates observed data points.

            optimizer_name:                 `str` of name of optimizer.

            input_gate_activation:          `act_type` in `mxnet.ndarray.Activation` or `mxnet.symbol.Activation` in input gate.
            forget_gate_activation:         `act_type` in `mxnet.ndarray.Activation` or `mxnet.symbol.Activation` in forget gate.
            output_gate_activation:         `act_type` in `mxnet.ndarray.Activation` or `mxnet.symbol.Activation` in output gate.
            hidden_activation:              `act_type` in `mxnet.ndarray.Activation` or `mxnet.symbol.Activation` in hidden layer.
            output_activation:              `act_type` in `mxnet.ndarray.Activation` or `mxnet.symbol.Activation` in output layer.
            output_layer_flag:              `bool` that means this class has output layer or not.

            ctx:                            `mx.cpu()` or `mx.gpu()`.
            hybridize_flag:                  Call `mxnet.gluon.HybridBlock.hybridize()` or not.
            regularizatable_data_list:           `list` of `Regularizatable`.
            scale:                          `float` of scaling factor for initial parameters.
            generating_flag:                `bool` of flag. 
                                            If `False`, the `decoder` will decode all encoded data.
                                            If `True`, the `decoder`'s decoding will start at last sequence of encoded data,
                                            generating other series.
        '''
        if isinstance(encoder, LSTMNetworks) is False:
            raise TypeError("The type of `lstm_networks` must be `LSTMNetworks`.")
        if isinstance(decoder, LSTMNetworks) is False:
            raise TypeError("The type of `lstm_networks` must be `LSTMNetworks`.")

        logger = getLogger("accelbrainbase")
        self.__logger = logger
        init_deferred_flag = self.init_deferred_flag
        self.init_deferred_flag = True

        self.__batch_size = batch_size
        self.__seq_len = seq_len

        super().__init__(
            computable_loss=computable_loss,
            initializer=initializer,
            batch_size=batch_size,
            learning_rate=learning_rate,
            learning_attenuate_rate=learning_attenuate_rate,
            attenuate_epoch=attenuate_epoch,
            seq_len=seq_len,
            hidden_n=hidden_n,
            output_n=output_n,
            dropout_rate=dropout_rate,
            optimizer_name=optimizer_name,
            observed_activation=observed_activation,
            input_gate_activation=input_gate_activation,
            forget_gate_activation=forget_gate_activation,
            output_gate_activation=output_gate_activation,
            hidden_activation=hidden_activation,
            output_activation=output_activation,
            output_layer_flag=output_layer_flag,
            ctx=ctx,
            hybridize_flag=hybridize_flag,
            regularizatable_data_list=regularizatable_data_list,
            scale=scale,
            **kwargs
        )
        self.init_deferred_flag = init_deferred_flag
        self.encoder = encoder
        self.decoder = decoder
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

        self.__generating_flag = generating_flag

    def collect_params(self, select=None):
        '''
        Overrided `collect_params` in `mxnet.gluon.HybridBlok`.
        '''
        params_dict = self.encoder.collect_params(select)
        params_dict.update(self.decoder.collect_params(select))
        return params_dict

    def forward_propagation(self, F, x):
        '''
        Hybrid forward with Gluon API.

        Args:
            F:      `mxnet.ndarray` or `mxnet.symbol`.
            x:      `mxnet.ndarray` of observed data points.
        
        Returns:
            `mxnet.ndarray` or `mxnet.symbol` of inferenced feature points.
        '''
        temp_arr = F.zeros_like(x)
        x = F.reshape(x, shape=(self.__batch_size, self.__seq_len, -1))

        encoded_arr = self.encoder.forward_propagation(F, x)
        self.feature_points_arr = encoded_arr

        if self.__generating_flag is True:
            arr = F.expand_dims(
                F.flip(
                    F.transpose(
                        encoded_arr,
                        (1, 0, 2)
                    ),
                    axis=0
                )[0],
                axis=1
            )
        else:
            arr = F.flip(encoded_arr, axis=1)

        decoded_arr = self.decoder.forward_propagation(F, arr)
        decoded_arr = F.flip(decoded_arr, axis=1)

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

    def get_encoder(self):
        ''' getter for `LSTMNetworks` of encoder.'''
        return self.__encoder
    
    def set_encoder(self, value):
        ''' setter for `LSTMNetworks` of encoder.'''
        if isinstance(value, LSTMNetworks) is False:
            raise TypeError("The type of `encoder` must be `LSTMNetworks`.")
        self.__encoder = value
    
    encoder = property(get_encoder, set_encoder)

    def get_decoder(self):
        ''' getter for `LSTMNetworks` of decoder.'''
        return self.__decoder
    
    def set_decoder(self, value):
        ''' setter for `LSTMNetworks` of decoder.'''
        if isinstance(value, LSTMNetworks) is False:
            raise TypeError("The type of `decoder` must be `LSTMNetworks`.")
        self.__decoder = value

    decoder = property(get_decoder, set_decoder)
