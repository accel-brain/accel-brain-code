# -*- coding: utf-8 -*-
import numpy as np
from logging import getLogger, StreamHandler, NullHandler, DEBUG, ERROR

# is-a `GenerativeModel`.
from accelbrainbase.computableloss._mxnet.l2_norm_loss import L2NormLoss
from accelbrainbase.extractabledata.unlabeled_csv_extractor import UnlabeledCSVExtractor
from accelbrainbase.iteratabledata._mxnet.unlabeled_sequential_csv_iterator import UnlabeledSequentialCSVIterator
from accelbrainbase.noiseabledata._mxnet.gauss_noise import GaussNoise
from accelbrainbase.observabledata._mxnet.lstm_networks import LSTMNetworks
from accelbrainbase.observabledata._mxnet.lstmnetworks.encoder_decoder import EncoderDecoder

from accelbrainbase.observabledata._mxnet.adversarialmodel.generative_model import GenerativeModel
from accelbrainbase.computableloss._mxnet.l2_norm_loss import L2NormLoss
from accelbrainbase.extractabledata._mxnet.image_extractor import ImageExtractor
from accelbrainbase.iteratabledata._mxnet.unlabeled_image_iterator import UnlabeledImageIterator
from accelbrainbase.noiseabledata._mxnet.gauss_noise import GaussNoise
from accelbrainbase.observabledata._mxnet.convolutional_neural_networks import ConvolutionalNeuralNetworks
from accelbrainbase.observabledata._mxnet.neural_networks import NeuralNetworks
from accelbrainbase.observabledata._mxnet.convolutionalneuralnetworks.convolutionalautoencoder.convolutional_ladder_networks import ConvolutionalLadderNetworks
from accelbrainbase.observabledata._mxnet.adversarialmodel.discriminative_model import DiscriminativeModel
from accelbrainbase.observabledata._mxnet.adversarialmodel.generative_model import GenerativeModel
from accelbrainbase.computableloss._mxnet.generator_loss import GeneratorLoss
from accelbrainbase.computableloss._mxnet.discriminator_loss import DiscriminatorLoss
from accelbrainbase.samplabledata.true_sampler import TrueSampler
from accelbrainbase.samplabledata.condition_sampler import ConditionSampler
from accelbrainbase.samplabledata.noisesampler._mxnet.uniform_noise_sampler import UniformNoiseSampler
from accelbrainbase.controllablemodel._mxnet.gan_controller import GANController

import mxnet as mx
import mxnet.ndarray as nd
import numpy as np
import pandas as pd
from mxnet.gluon.nn import Conv2D
from mxnet.gluon.nn import Conv2DTranspose
from mxnet.gluon.nn import BatchNorm
from mxnet import gluon


class RecursiveSeq2SeqModel(GenerativeModel):
    '''
    Seq2Seq model which can do recursive and recurrent inference.
    '''

    __condition_sampler = None

    def get_condition_sampler(self):
        return self.__condition_sampler
    
    def set_condition_sampler(self, value):
        self.__condition_sampler = value

    condition_sampler = property(get_condition_sampler, set_condition_sampler)

    def __init__(
        self,
        batch_size,
        seq_len,
        output_n,
        noise_sampler, 
        model=None, 
        re_encoder_model=None,
        initializer=None,
        computable_loss=None,
        condition_sampler=None,
        conditonal_dim=2,
        learning_rate=1e-05,
        optimizer_name="SGD",
        hybridize_flag=True,
        scale=1.0, 
        ctx=mx.gpu(), 
        channel=1000,
        diff_mode=True,
        log_mode=True,
        hidden_n=200,
        expand_dims_flag=True,
        **kwargs
    ):
        '''
        Init.

        Args:
            batch_size:                     `int` of batch size.
            seq_len:                        `int` of the length of sequence.
            output_n:                       `int` of the dimension of outputs.
            noise_sampler:                  is-a `NoiseSampler`.
            model:                          model.
            re_encoder_model:               is-a `ReEncoderModel`.
            initializer:                    is-a `mxnet.initializer` for parameters of model. If `None`, it is drawing from the Xavier distribution.
            computable_loss:                is-a `ComputableLoss`.
            condition_sampler:              is-a `ConditionSampler`.
            conditonal_dim:                 `int` of the dimension of conditions.
            learning_rate:                  `float` of learning rate.
            optimizer_name:                 `str` of optimizer's name.
            hybridize_flag:                 Call `mxnet.gluon.HybridBlock.hybridize()` or not. 
            scale:                          `float` of scales.
            ctx:                            `mx.cpu()` or `mx.gpu()`.
            diff_mode:                      `bool`. If `True`, this class outputs difference sequences.
            log_mode:                       `bool`. If `True`, this class outputs logarithmic rates of change.
            hidden_n:                       `int` of the number of hidden units.
            expand_dims_flag:               `bool`. If `True`, this class expands dimensions of output data (axis=1).

        '''
        if computable_loss is None:
            computable_loss = L2NormLoss()

        if model is None:
            if log_mode is True:
                o_act = "tanh"
            else:
                o_act = "identity"

            encoder = LSTMNetworks(
                # is-a `ComputableLoss` or `mxnet.gluon.loss`.
                computable_loss=computable_loss,
                # `int` of batch size.
                batch_size=batch_size,
                # `int` of the length of series.
                seq_len=seq_len,
                # `int` of the number of units in hidden layer.
                hidden_n=hidden_n,
                # `int` of the number of units in output layer.
                output_n=output_n,
                # `float` of dropout rate.
                dropout_rate=0.0,
                # `act_type` in `mxnet.ndarray.Activation` or `mxnet.symbol.Activation` 
                # that activates observed data points.
                observed_activation="tanh",
                # `act_type` in `mxnet.ndarray.Activation` or `mxnet.symbol.Activation` in input gate.
                input_gate_activation="sigmoid",
                # `act_type` in `mxnet.ndarray.Activation` or `mxnet.symbol.Activation` in forget gate.
                forget_gate_activation="sigmoid",
                # `act_type` in `mxnet.ndarray.Activation` or `mxnet.symbol.Activation` in output gate.
                output_gate_activation="sigmoid",
                # `act_type` in `mxnet.ndarray.Activation` or `mxnet.symbol.Activation` in hidden layer.
                hidden_activation="tanh",
                # `act_type` in `mxnet.ndarray.Activation` or `mxnet.symbol.Activation` in output layer.
                output_activation=o_act,
                # `bool` that means this class has output layer or not.
                output_layer_flag=True,
                # `bool` for using bias or not in output layer(last hidden layer).
                output_no_bias_flag=False,
                # Call `mxnet.gluon.HybridBlock.hybridize()` or not.
                hybridize_flag=True,
                # `mx.cpu()` or `mx.gpu()`.
                ctx=ctx,
                input_adjusted_flag=False
            )

            decoder = LSTMNetworks(
                # is-a `ComputableLoss` or `mxnet.gluon.loss`.
                computable_loss=computable_loss,
                # `int` of batch size.
                batch_size=batch_size,
                # `int` of the length of series.
                seq_len=seq_len,
                # `int` of the number of units in hidden layer.
                hidden_n=hidden_n,
                # `int` of the number of units in output layer.
                output_n=output_n,
                # `float` of dropout rate.
                dropout_rate=0.0,
                # `act_type` in `mxnet.ndarray.Activation` or `mxnet.symbol.Activation` 
                # that activates observed data points.
                observed_activation="tanh",
                # `act_type` in `mxnet.ndarray.Activation` or `mxnet.symbol.Activation` in input gate.
                input_gate_activation="sigmoid",
                # `act_type` in `mxnet.ndarray.Activation` or `mxnet.symbol.Activation` in forget gate.
                forget_gate_activation="sigmoid",
                # `act_type` in `mxnet.ndarray.Activation` or `mxnet.symbol.Activation` in output gate.
                output_gate_activation="sigmoid",
                # `act_type` in `mxnet.ndarray.Activation` or `mxnet.symbol.Activation` in hidden layer.
                hidden_activation="tanh",
                # `act_type` in `mxnet.ndarray.Activation` or `mxnet.symbol.Activation` in output layer.
                output_activation=o_act,
                # `bool` that means this class has output layer or not.
                output_layer_flag=True,
                # `bool` for using bias or not in output layer(last hidden layer).
                output_no_bias_flag=False,
                # Call `mxnet.gluon.HybridBlock.hybridize()` or not.
                hybridize_flag=True,
                # `mx.cpu()` or `mx.gpu()`.
                ctx=ctx,
                input_adjusted_flag=False
            )

            model = EncoderDecoder(
                # is-a `LSTMNetworks`.
                encoder=encoder,
                # is-a `LSTMNetworks`.
                decoder=decoder,
                # `int` of batch size.
                batch_size=batch_size,
                # `int` of the length of series.
                seq_len=seq_len,
                # is-a `ComputableLoss` or `mxnet.gluon.loss`.
                computable_loss=computable_loss,
                # is-a `mxnet.initializer` for parameters of model. If `None`, it is drawing from the Xavier distribution.
                initializer=initializer,
                # `float` of learning rate.
                learning_rate=learning_rate,
                # `float` of attenuate the `learning_rate` by a factor of this value every `attenuate_epoch`.
                learning_attenuate_rate=1.0,
                # `int` of attenuate the `learning_rate` by a factor of `learning_attenuate_rate` every `attenuate_epoch`.
                attenuate_epoch=50,
                # `str` of name of optimizer.
                optimizer_name=optimizer_name,
                # Call `mxnet.gluon.HybridBlock.hybridize()` or not.
                hybridize_flag=True,
                # `mx.cpu()` or `mx.gpu()`.
                ctx=ctx,
                generating_flag=False
            )

        if re_encoder_model is None:
            re_encoder_model = LSTMNetworks(
                # is-a `ComputableLoss` or `mxnet.gluon.loss`.
                computable_loss=computable_loss,
                # `int` of batch size.
                batch_size=batch_size,
                # `int` of the length of series.
                seq_len=seq_len,
                # `int` of the number of units in hidden layer.
                hidden_n=hidden_n,
                # `int` of the number of units in output layer.
                output_n=output_n,
                # `float` of dropout rate.
                dropout_rate=0.0,
                # `act_type` in `mxnet.ndarray.Activation` or `mxnet.symbol.Activation` 
                # that activates observed data points.
                observed_activation="tanh",
                # `act_type` in `mxnet.ndarray.Activation` or `mxnet.symbol.Activation` in input gate.
                input_gate_activation="sigmoid",
                # `act_type` in `mxnet.ndarray.Activation` or `mxnet.symbol.Activation` in forget gate.
                forget_gate_activation="sigmoid",
                # `act_type` in `mxnet.ndarray.Activation` or `mxnet.symbol.Activation` in output gate.
                output_gate_activation="sigmoid",
                # `act_type` in `mxnet.ndarray.Activation` or `mxnet.symbol.Activation` in hidden layer.
                hidden_activation="tanh",
                # `act_type` in `mxnet.ndarray.Activation` or `mxnet.symbol.Activation` in output layer.
                output_activation=o_act,
                # `bool` that means this class has output layer or not.
                output_layer_flag=True,
                # `bool` for using bias or not in output layer(last hidden layer).
                output_no_bias_flag=False,
                # Call `mxnet.gluon.HybridBlock.hybridize()` or not.
                hybridize_flag=True,
                # `mx.cpu()` or `mx.gpu()`.
                ctx=ctx,
                input_adjusted_flag=False
            )

        init_deferred_flag = self.init_deferred_flag
        self.init_deferred_flag = True

        super().__init__(
            noise_sampler=noise_sampler, 
            model=model, 
            initializer=initializer,
            condition_sampler=condition_sampler,
            conditonal_dim=conditonal_dim,
            learning_rate=learning_rate,
            optimizer_name=optimizer_name,
            hybridize_flag=hybridize_flag,
            scale=1.0, 
            ctx=ctx, 
            **kwargs
        )
        self.init_deferred_flag = init_deferred_flag

        self.re_encoder_model = re_encoder_model

        if initializer is None:
            self.initializer = mx.initializer.Xavier(
                rnd_type="gaussian", 
                factor_type="in", 
                magnitude=2
            )
        else:
            if isinstance(initializer, mx.initializer.Initializer) is False:
                raise TypeError("The type of `initializer` must be `mxnet.initializer.Initializer`.")
            self.initializer = initializer

        with self.name_scope():
            self.register_child(self.re_encoder_model)

        if self.init_deferred_flag is False:
            try:
                self.collect_params().initialize(self.initializer, force_reinit=True, ctx=ctx)
                self.trainer = gluon.Trainer(
                    self.collect_params(),
                    optimizer_name,
                    {
                        "learning_rate": learning_rate
                    }
                )
                if hybridize_flag is True:
                    self.model.hybridize()
                    self.re_encoder_model.hybridize()
                    if self.condition_sampler is not None:
                        if self.condition_sampler.model is not None:
                            self.condition_sampler.model.hybridize()
            except InitDeferredError:
                self.__logger.debug("The initialization should be deferred.")

        logger = getLogger("accelbrainbase")
        self.__logger = logger

        self.__learning_rate = learning_rate

        self.__cnn = model
        self.__condition_sampler = condition_sampler
        self.__computable_loss = computable_loss

        self.__q_shape = None
        self.__loss_list = []
        self.__epoch_counter = 0

        self.conditonal_dim = conditonal_dim
        self.__expand_dims_flag = expand_dims_flag

    def collect_params(self, select=None):
        '''
        Overrided `collect_params` in `mxnet.gluon.HybridBlok`.
        '''
        params_dict = super().collect_params(select)
        params_dict.update(self.re_encoder_model.collect_params(select))
        return params_dict

    def inference_g(self, observed_arr):
        '''
        Inference with generator.

        Args:
            observed_arr:       `mxnet.ndarray` of observed data points.
        
        Returns:
            Tuple data.
            - re-parametric data.
            - encoded data points.
            - re-encoded data points.
        '''
        encoded_arr = self.model.encoder(observed_arr)
        decoded_arr = self.model.decoder(encoded_arr)
        re_encoded_arr = self.re_encoder_model(decoded_arr)

        anomaly_arr = nd.square(
            encoded_arr - re_encoded_arr
        )
        anomaly_arr = nd.expand_dims(nd.exp(anomaly_arr.mean(axis=1)), axis=1)
        mean_arr = nd.expand_dims(decoded_arr.mean(axis=1), axis=1)
        gauss_arr = nd.random.normal_like(data=observed_arr, loc=0, scale=3.0)

        re_param_arr = mean_arr + (gauss_arr * anomaly_arr)

        kl_arr = -0.5 * (1 + nd.log(anomaly_arr) - mean_arr + anomaly_arr)
        re_param_arr = re_param_arr + kl_arr

        return re_param_arr, encoded_arr, re_encoded_arr

    def draw(self):
        '''
        Draws samples from the `fake` distribution.

        Returns:
            `np.ndarray` of samples.
        '''
        if self.condition_sampler is None:
            observed_arr = self.noise_sampler.draw()
            generated_arr, encoded_arr, re_encoded_arr = self.inference_g(observed_arr)

            if self.__expand_dims_flag is True:
                observed_arr = nd.expand_dims(observed_arr, axis=1)
                generated_arr = nd.expand_dims(generated_arr, axis=1)
            
            return (
                nd.concat(observed_arr, generated_arr, dim=self.conditonal_dim),
                encoded_arr,
                re_encoded_arr
            )
        else:
            raise NotImplementedError()

            condition_arr, sampled_arr = self.condition_sampler.draw()
            if sampled_arr is not None:
                if self.noise_sampler is not None:
                    sampled_arr = sampled_arr + self.noise_sampler.draw()
                inferenced_arr = self.model(sampled_arr)

                if self.__expand_dims_flag is True:
                    inferenced_arr = nd.expand_dims(inferenced_arr, axis=1)
                    condition_arr = nd.expand_dims(condition_arr, axis=1)
                generated_arr = nd.concat(
                    inferenced_arr,
                    condition_arr,
                    dim=self.conditonal_dim
                )
                return generated_arr
            else:
                observed_arr = condition_arr
                if self.noise_sampler is not None:
                    condition_arr = condition_arr + self.noise_sampler.draw()
                generated_arr = self.model(condition_arr)
                
                if self.__expand_dims_flag is True:
                    observed_arr = nd.expand_dims(observed_arr, axis=1)
                    generated_arr = nd.expand_dims(generated_arr, axis=1)
                return nd.concat(observed_arr, generated_arr, dim=self.conditonal_dim)

    def recursive_draw(self, limit=1):
        '''
        Recursive draw.

        Args:
            limit:      `int` of the number of recursive inferences.

        Returns:
            `list` of `np.array`.
        '''
        generated_arr_list = [None] * limit

        if self.__expand_dims_flag is True:
            _observed_arr, _, _ = self.draw()
            observed_arr = _observed_arr[:, 0]
        else:
            observed_arr, _, _ = self.draw()

        if self.conditonal_dim == 1:
            channel = observed_arr.shape[1] // 2
            generated_arr = observed_arr[:, channel:]
            observed_arr = observed_arr[:, :channel]
        elif self.conditonal_dim == 2:
            width = observed_arr.shape[2] // 2
            generated_arr = observed_arr[:, :, width:]
            observed_arr = observed_arr[:, :, :width]
        elif self.conditonal_dim == 3:
            height = observed_arr.shape[3] // 2
            generated_arr = observed_arr[:, :, :, height:]
            observed_arr = observed_arr[:, :, :, :height]

        for i in range(limit):
            generated_arr, _, _ = self.inference_g(observed_arr)
            if i == 0:
                generated_arr = self.__re_scaling(generated_arr, observed_arr)
            else:
                generated_arr = self.__re_scaling(generated_arr)

            if self.__condition_sampler is not None:
                self.__condition_sampler.output_shape = generated_arr.shape
                noise_arr = self.__condition_sampler.generate()
                generated_arr += noise_arr

            generated_arr_list[i] = generated_arr

            observed_arr = generated_arr

        self.__observed_arr = observed_arr
        self.__generated_arr_list = generated_arr_list

        return generated_arr_list

    def __re_scaling(self, generated_arr, observed_arr=None):
        if observed_arr is None:
            if self.__expand_dims_flag is True:
                _observed_arr, _, _ = self.draw()
                observed_arr = _observed_arr[:, 0]
            else:
                observed_arr, _, _ = self.draw()

            if self.conditonal_dim == 1:
                channel = observed_arr.shape[1] // 2
                observed_arr = observed_arr[:, :channel]
            elif self.conditonal_dim == 2:
                width = observed_arr.shape[2] // 2
                observed_arr = observed_arr[:, :, :width]
            elif self.conditonal_dim == 3:
                height = observed_arr.shape[3] // 2
                observed_arr = observed_arr[:, :, :, :height]

        o_min_arr = nd.expand_dims(observed_arr.min(axis=1), axis=1)
        o_max_arr = nd.expand_dims(observed_arr.max(axis=1), axis=1)
        g_min_arr = nd.expand_dims(generated_arr.min(axis=1), axis=1)
        g_max_arr = nd.expand_dims(generated_arr.max(axis=1), axis=1)
        generated_arr = (generated_arr - g_min_arr) / (g_max_arr - g_min_arr)
        generated_arr = (o_max_arr - o_min_arr) * generated_arr
        generated_arr = o_min_arr + generated_arr

        return generated_arr

    def rest_recursive_draw(self, limit=12):
        '''
        Recursive draw for the rest of the period.

        Args:
            limit:      `int` of the number of recursive inferences.

        Returns:
            `list` of `np.array`.
        '''
        observed_arr = self.__observed_arr
        generated_arr_list = self.__generated_arr_list
        for i in range(limit):
            generated_arr, _, _ = self.inference_g(observed_arr)
            if i == 0:
                generated_arr = self.__re_scaling(generated_arr, observed_arr)
            else:
                generated_arr = self.__re_scaling(generated_arr)

            generated_arr_list.append(generated_arr)
            observed_arr = generated_arr

        self.__observed_arr = observed_arr
        self.__generated_arr_list = generated_arr_list

        return generated_arr_list

    def inference(self, observed_arr):
        '''
        Draws samples from the `true` distribution.

        Args:
            observed_arr:     `np.ndarray` of observed data points.
        
        Returns:
            `np.ndarray` of inferenced.
        '''
        
        self.__pred_arr = super().inference(observed_arr)
        return self.__pred_arr
