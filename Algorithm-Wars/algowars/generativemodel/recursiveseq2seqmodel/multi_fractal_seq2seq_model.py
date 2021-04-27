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

from algowars.generativemodel.recursive_seq2seq_model import RecursiveSeq2SeqModel

import mxnet as mx
import mxnet.ndarray as nd
import numpy as np
import pandas as pd
from mxnet.gluon.nn import Conv2D
from mxnet.gluon.nn import Conv2DTranspose
from mxnet.gluon.nn import BatchNorm
from mxnet import gluon
from mxnet import autograd


class MultiFractalSeq2SeqModel(RecursiveSeq2SeqModel):
    '''
    Multi-Fractal Seq2Seq model which can do recursive and recurrent inference.
    '''

    __long_term_seq_len = 30

    def get_long_term_seq_len(self):
        return self.__long_term_seq_len
    
    def set_long_term_seq_len(self, value):
        self.__long_term_seq_len = value

    long_term_seq_len = property(get_long_term_seq_len, set_long_term_seq_len)

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
        generated_arr, encoded_arr, re_encoded_arr = super().inference_g(observed_arr)

        if autograd.is_recording():
            limit = self.long_term_seq_len

            seq_len = self.noise_sampler.seq_len
            self.noise_sampler.seq_len = limit
            long_term_observed_arr = self.noise_sampler.draw()

            observed_mean_arr = nd.expand_dims(nd.mean(long_term_observed_arr, axis=1), axis=1)
            sum_arr = None
            for seq in range(2, long_term_observed_arr.shape[1]):
                add_arr = nd.sum(long_term_observed_arr[:, :seq] - observed_mean_arr, axis=1)
                if sum_arr is None:
                    sum_arr = nd.expand_dims(add_arr, axis=0)
                else:
                    sum_arr = nd.concat(
                        sum_arr,
                        nd.expand_dims(add_arr, axis=0),
                        dim=0
                    )
            max_arr = nd.max(sum_arr, axis=0)
            min_arr = nd.min(sum_arr, axis=0)

            diff_arr = long_term_observed_arr - observed_mean_arr
            std_arr = nd.power(nd.mean(nd.square(diff_arr), axis=1), 1/2)
            R_S_arr = (max_arr - min_arr) / std_arr
            len_arr = nd.ones_like(R_S_arr, ctx=R_S_arr.context) * np.log(long_term_observed_arr.shape[1] / 2)
            observed_H_arr = nd.log(R_S_arr) / len_arr

            self.noise_sampler.seq_len = seq_len

            g_min_arr = nd.expand_dims(generated_arr.min(axis=1), axis=1)
            g_max_arr = nd.expand_dims(generated_arr.max(axis=1), axis=1)
            o_min_arr = nd.expand_dims(observed_arr.min(axis=1), axis=1)
            o_max_arr = nd.expand_dims(observed_arr.max(axis=1), axis=1)

            _observed_arr = generated_arr

            long_term_generated_arr = None
            for i in range(limit):
                generated_arr, _, _ = super().inference_g(_observed_arr)

                g_min_arr = nd.expand_dims(generated_arr.min(axis=1), axis=1)
                g_max_arr = nd.expand_dims(generated_arr.max(axis=1), axis=1)
                o_min_arr = nd.expand_dims(_observed_arr.min(axis=1), axis=1)
                o_max_arr = nd.expand_dims(_observed_arr.max(axis=1), axis=1)
                generated_arr = (generated_arr - g_min_arr) / (g_max_arr - g_min_arr)
                generated_arr = (o_max_arr - o_min_arr) * generated_arr
                generated_arr = o_min_arr + generated_arr

                if self.condition_sampler is not None:
                    self.condition_sampler.output_shape = generated_arr.shape
                    noise_arr = self.condition_sampler.generate()
                    generated_arr += noise_arr

                if long_term_generated_arr is None:
                    long_term_generated_arr = generated_arr
                else:
                    long_term_generated_arr = nd.concat(
                        long_term_generated_arr,
                        generated_arr,
                        dim=1
                    )

                _observed_arr = generated_arr

            generated_mean_arr = nd.expand_dims(nd.mean(long_term_generated_arr, axis=1), axis=1)
            sum_arr = None
            for seq in range(2, long_term_generated_arr.shape[1]):
                add_arr = nd.sum(long_term_generated_arr[:, :seq] - generated_mean_arr, axis=1)
                if sum_arr is None:
                    sum_arr = nd.expand_dims(add_arr, axis=0)
                else:
                    sum_arr = nd.concat(
                        sum_arr,
                        nd.expand_dims(add_arr, axis=0),
                        dim=0
                    )
            max_arr = nd.max(sum_arr, axis=0)
            min_arr = nd.min(sum_arr, axis=0)

            diff_arr = long_term_generated_arr - generated_mean_arr
            std_arr = nd.power(nd.mean(nd.square(diff_arr), axis=1), 1/2)
            R_S_arr = (max_arr - min_arr) / std_arr
            len_arr = nd.ones_like(R_S_arr, ctx=R_S_arr.context) * np.log(long_term_generated_arr.shape[1] / 2)
            generated_H_arr = nd.log(R_S_arr) / len_arr

            multi_fractal_loss = nd.abs(generated_H_arr - observed_H_arr)
            multi_fractal_loss = nd.mean(multi_fractal_loss, axis=0, exclude=True)
            multi_fractal_loss = nd.expand_dims(multi_fractal_loss, axis=-1)
            multi_fractal_loss = nd.expand_dims(multi_fractal_loss, axis=-1)

            generated_arr = generated_arr + multi_fractal_loss

        return generated_arr, encoded_arr, re_encoded_arr

    """
    def __long_term_correlate(self, generated_arr, H, depth, wave_n):
        if generated_arr.shape[1] < wave_n:
            return generated_arr

        depth = depth - 1
        if depth <= 0:
            return generated_arr

        new_generated_arr = None
        for wave in range(wave_n):
            if wave == 0:
                start_seq_len = 0
                end_seq_len = generated_arr.shape[1]//wave_n
            else:
                start_seq_len = end_seq_len + 1
                end_seq_len = (generated_arr.shape[1]//5) * (wave + 1)

            if wave != wave_n - 1:
                add_arr = self.__H_walk(
                    generated_arr[:, start_seq_len:end_seq_len], 
                    H
                )
            else:
                add_arr = self.__H_walk(
                    generated_arr[:, start_seq_len:], 
                    H
                )
            add_arr = self.__long_term_correlate(
                add_arr, 
                H, 
                depth, 
                wave_n,
            )

            if new_generated_arr is None:
                new_generated_arr = add_arr
            else:
                new_generated_arr = nd.concat(
                    new_generated_arr,
                    add_arr,
                    dim=1
                )

        return new_generated_arr

    def __H_walk(self, generated_arr, H):
        if generated_arr.shape[1] <= 1:
            return generated_arr
        seq_arr = nd.arange(generated_arr.shape[1], ctx=generated_arr.context)
        H_weight_arr = nd.power(seq_arr, H)
        H_weight_arr = H_weight_arr / H_weight_arr.sum()
        H_weight_arr = nd.expand_dims(H_weight_arr, axis=0)
        H_weight_arr = nd.expand_dims(H_weight_arr, axis=-1)
        H_weight_arr = nd.expand_dims(H_weight_arr, axis=-1)

        arr = H_weight_arr.asnumpy()
        new_arr = generated_arr * H_weight_arr

        return new_arr
    """
