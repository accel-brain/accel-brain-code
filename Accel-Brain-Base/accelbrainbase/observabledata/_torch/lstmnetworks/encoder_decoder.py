# -*- coding: utf-8 -*-
from accelbrainbase.observabledata._torch.lstm_networks import LSTMNetworks
from accelbrainbase.iteratable_data import IteratableData
from accelbrainbase.regularizatable_data import RegularizatableData
from accelbrainbase.computable_loss import ComputableLoss
import numpy as np
import torch
from torch import nn
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
        learning_rate=1e-05,
        ctx="cpu",
        regularizatable_data_list=[],
        generating_flag=True,
        not_init_flag=False,
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
        super().__init__(
            computable_loss=computable_loss,
            learning_rate=learning_rate,
            ctx=ctx,
        )
        self.init_deferred_flag = init_deferred_flag
        self.encoder = encoder
        self.decoder = decoder
        self.encoder_optimizer = None
        self.decoder_optimizer = None
        self.__computable_loss = computable_loss
        self.__generating_flag = generating_flag
        self.__not_init_flag = not_init_flag
        self.epoch = 0
        self.__input_seq_len = None
        self.__encoder_input_dim = None
        self.__decoder_input_dim = None

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

    def initialize_params(self, input_dim, input_seq_len):
        '''
        Initialize params.
        '''
        if self.encoder_optimizer is not None and self.decoder_optimizer is not None:
            return

        self.__input_seq_len = input_seq_len
        self.__encoder_input_dim = input_dim
        if self.init_deferred_flag is False:
            if self.__not_init_flag is False:
                self.encoder.initialize_params(
                    self.__encoder_input_dim,
                    input_seq_len
                )
                self.__decoder_input_dim = self.encoder.output_dim
                self.decoder.initialize_params(
                    self.__decoder_input_dim,
                    input_seq_len
                )
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
                seq_len = batch_observed_arr.shape[1]
                self.initialize_params(
                    input_dim=batch_observed_arr.reshape((batch_size, seq_len, -1)).shape[2],
                    input_seq_len=seq_len
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

    def forward(self, x):
        '''
        Forward with torch.

        Args:
            x:      `tensor` of observed data points.
        
        Returns:
            `tensor` of inferenced feature points.
        '''
        batch_size = x.shape[0]
        seq_len = x.shape[1]
        x = torch.reshape(
            x, 
            shape=(
                batch_size, 
                seq_len, 
                -1
            )
        )
        self.initialize_params(
            input_dim=x.shape[2],
            input_seq_len=seq_len,
        )

        encoded_arr = self.encoder(x)
        self.feature_points_arr = encoded_arr

        if self.__generating_flag is True:
            arr = torch.unsqueeze(encoded_arr[:, -1, :], axis=1)
        else:
            arr = torch.flip(encoded_arr, dims=(1, ))

        decoded_arr = self.decoder(arr)
        decoded_arr = torch.flip(decoded_arr, dims=(1, ))

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
        self.loss_arr = checkpoint['loss']

    def set_readonly(self, value):
        ''' setter '''
        raise TypeError("This property must be read-only.")

    def get_loss_arr(self):
        ''' getter for losses. '''
        return np.array(self.__loss_list)

    loss_arr = property(get_loss_arr, set_readonly)
