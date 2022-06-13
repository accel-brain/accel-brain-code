# -*- coding: utf-8 -*-
from logging import getLogger
from logging import getLogger, StreamHandler, NullHandler, DEBUG, ERROR
import numpy as np
import numpy as np
import pandas as pd

from accelbrainbase.computableloss._torch.l2_norm_loss import L2NormLoss
from accelbrainbase.extractabledata.unlabeled_csv_extractor import UnlabeledCSVExtractor
from accelbrainbase.iteratabledata._torch.unlabeled_sequential_csv_iterator import UnlabeledSequentialCSVIterator
from accelbrainbase.noiseabledata._torch.gauss_noise import GaussNoise
from accelbrainbase.observabledata._torch.lstm_networks import LSTMNetworks
from accelbrainbase.observabledata._torch.lstmnetworks.encoder_decoder import EncoderDecoder

from pysummarization.abstractable_semantics import AbstractableSemantics
from pysummarization.vectorizable_token import VectorizableToken

import torch


class EncDecAD(AbstractableSemantics):
    '''
    LSTM-based Encoder/Decoder scheme for Anomaly Detection (EncDec-AD).

    This library applies the Encoder-Decoder scheme for Anomaly Detection (EncDec-AD)
    to text summarizations by intuition. In this scheme, LSTM-based Encoder/Decoder 
    or so-called the sequence-to-sequence(Seq2Seq) model learns to reconstruct normal time-series behavior,
    and thereafter uses reconstruction error to detect anomalies.
    
    Malhotra, P., et al. (2016) showed that EncDecAD paradigm is robust and 
    can detect anomalies from predictable, unpredictable, periodic, aperiodic, 
    and quasi-periodic time-series. Further, they showed that the paradigm is able to 
    detect anomalies from short time-series (length as small as 30) as well as long 
    time-series (length as large as 500).

    This library refers to the intuitive insight in relation to the use case of 
    reconstruction error to detect anomalies above to apply the model to text summarization.
    As exemplified by Seq2Seq paradigm, document and sentence which contain tokens of text
    can be considered as time-series features. The anomalies data detected by EncDec-AD 
    should have to express something about the text.

    From the above analogy, this library introduces two conflicting intuitions. On the one hand,
    the anomalies data may catch observer's eye from the viewpoints of rarity or amount of information
    as the indicator of natural language processing like TF-IDF shows. On the other hand,
    the anomalies data may be ignorable noise as mere outlier.
    
    In any case, this library deduces the function and potential of EncDec-AD in text summarization
    is to draw the distinction of normal and anomaly texts and is to filter the one from the other.

    References:
        - Malhotra, P., Ramakrishnan, A., Anand, G., Vig, L., Agarwal, P., & Shroff, G. (2016). LSTM-based encoder-decoder for multi-sensor anomaly detection. arXiv preprint arXiv:1607.00148.
    '''

    # Logs of accuracy.
    __logs_tuple_list = []

    __ctx = "cpu"

    def get_ctx(self):
        ''' getter for `mx.gpu()` or `mx.cpu()`. '''
        return self.__ctx

    def set_ctx(self, value):
        ''' setter for `mx.gpu()` or `mx.cpu()`. '''
        self.__ctx = value

    ctx = property(get_ctx, set_ctx)

    def __init__(
        self, 
        computable_loss=None,
        normal_prior_flag=False,
        encoder_decoder_controller=None,
        hidden_neuron_count=20,
        output_neuron_count=20,
        dropout_rate=0.5,
        epochs=100,
        learning_rate=1e-05,
        seq_len=8,
        ctx="cpu",
        initializer_f=None,
        optimizer_f=None,
    ):
        '''
        Init.

        Args:
            computable_loss:                is-a `ComputableLoss`.
            normal_prior_flag:              If `True`, this class will select abstract sentence
                                            from sentences with low reconstruction error.

            encoder_decoder_controller:     is-a `EncoderDecoderController`.
            hidden_neuron_count:            The number of units in hidden layers.
            output_neuron_count:            The number of units in output layers.

            dropout_rate:                   Probability of dropout.
            epochs:                         The epochs in mini-batch training Encoder/Decoder and retrospective encoder.
            batch_size:                     Batch size.
            learning_rate:                  Learning rate.
            learning_attenuate_rate:        Attenuate the `learning_rate` by a factor of this value every `attenuate_epoch`.
            attenuate_epoch:                Attenuate the `learning_rate` by a factor of `learning_attenuate_rate` every `attenuate_epoch`.
                                            

            seq_len:                        The length of sequneces in Decoder with Attention model.

        '''
        if computable_loss is None:
            computable_loss = L2NormLoss()

        self.ctx = ctx
        self.__normal_prior_flag = normal_prior_flag
        if encoder_decoder_controller is None:
            encoder_decoder_controller = self.__build_encoder_decoder_controller(
                computable_loss=computable_loss,
                hidden_neuron_count=hidden_neuron_count,
                output_neuron_count=output_neuron_count,
                dropout_rate=dropout_rate,
                learning_rate=learning_rate,
                seq_len=seq_len,
                initializer_f=initializer_f,
                optimizer_f=optimizer_f,
            )
        else:
            if isinstance(encoder_decoder_controller, EncoderDecoderController) is False:
                raise TypeError()

        self.__encoder_decoder_controller = encoder_decoder_controller

        logger = getLogger("accelbrainbase")
        self.__logger = logger
        self.__logs_tuple_list = []
        self.__computable_loss = computable_loss

    def __build_encoder_decoder_controller(
        self,
        computable_loss=None,
        hidden_neuron_count=20,
        output_neuron_count=20,
        dropout_rate=0.5,
        epochs=1000,
        learning_rate=1e-05,
        seq_len=8,
        initializer_f=None,
        optimizer_f=None,
    ):
        encoder = LSTMNetworks(
            initializer_f=initializer_f,
            optimizer_f=optimizer_f,
            # is-a `ComputableLoss` or `mxnet.gluon.loss`.
            computable_loss=computable_loss,
            # `int` of the length of series.
            seq_len=seq_len,
            # `int` of the number of units in hidden layer.
            hidden_n=hidden_neuron_count,
            # `float` of dropout rate.
            dropout_rate=dropout_rate,
            # `bool` that means this class has output layer or not.
            output_layer_flag=False,
            # `mx.cpu()` or `mx.gpu()`.
            ctx=self.ctx,
        )

        decoder = LSTMNetworks(
            initializer_f=initializer_f,
            optimizer_f=optimizer_f,
            # is-a `ComputableLoss` or `mxnet.gluon.loss`.
            computable_loss=computable_loss,
            # `int` of the length of series.
            seq_len=seq_len,
            # `int` of the number of units in hidden layer.
            hidden_n=hidden_neuron_count,
            # `int` of the number of units in output layer.
            output_n=output_neuron_count,
            # `float` of dropout rate.
            dropout_rate=dropout_rate,
            # `bool` that means this class has output layer or not.
            output_layer_flag=True,
            # `bool` for using bias or not in output layer(last hidden layer).
            output_no_bias_flag=True,
            # `mx.cpu()` or `mx.gpu()`.
            ctx=self.ctx,
        )

        encoder_decoder_controller = EncoderDecoder(
            # is-a `LSTMNetworks`.
            encoder=encoder,
            # is-a `LSTMNetworks`.
            decoder=decoder,
            # is-a `ComputableLoss` or `mxnet.gluon.loss`.
            computable_loss=computable_loss,
            # `float` of learning rate.
            learning_rate=learning_rate,
            # `mx.cpu()` or `mx.gpu()`.
            ctx=self.ctx,
        )
        self.__computable_loss = computable_loss
        return encoder_decoder_controller

    def learn(self, iteratable_data):
        '''
        Learn the observed data points
        for vector representation of the input time-series.

        Args:
            iteratable_data:     is-a `IteratableData`.

        '''
        self.__encoder_decoder_controller.learn(iteratable_data)

    def inference(self, observed_arr):
        '''
        Infernece by the model.

        Args:
            observed_arr:       `np.ndarray` of observed data points.

        Returns:
            `np.ndarray` of inferenced feature points.
        '''

        if isinstance(observed_arr, torch.Tensor) is False:
            observed_arr = torch.from_numpy(observed_arr)
            observed_arr = observed_arr.to(self.ctx)
        return self.__encoder_decoder_controller.inference(observed_arr)

    def summarize(self, iteratable_data, vectorizable_token, sentence_list, limit=5):
        '''
        Summarize input document.

        Args:
            iteratable_data:        is-a `IteratableData`.
            vectorizable_token:     is-a `VectorizableToken`.
            sentence_list:          `list` of all sentences.
            limit:                  The number of selected abstract sentence.
        
        Returns:
            `np.ndarray` of scores.
        '''
        if isinstance(vectorizable_token, VectorizableToken) is False:
            raise TypeError()

        _score_arr = None
        _test_arr = None
        for _, _, test_arr, _ in iteratable_data.generate_inferenced_samples():
            reconstruced_arr = self.inference(test_arr)
            score_arr = self.__computable_loss(test_arr, reconstruced_arr)
            score_arr = score_arr.to('cpu').detach().numpy().copy()

            if _score_arr is None:
                _score_arr = score_arr
            else:
                _score_arr = np.r_[_score_arr, score_arr]
            if _test_arr is None:
                _test_arr = test_arr.to('cpu').detach().numpy().copy()
            else:
                _test_arr = np.r_[_test_arr, test_arr.to('cpu').detach().numpy().copy()]

        score_list = _score_arr.tolist()
        test_arr = _test_arr
        score_arr = _score_arr

        abstract_list = []
        for i in range(limit):
            if self.__normal_prior_flag is True:
                key = score_arr.argmin()
            else:
                key = score_arr.argmax()

            score = score_list.pop(key)
            score_arr = np.array(score_list)

            seq_arr = test_arr[key]
            token_arr = vectorizable_token.tokenize(seq_arr.tolist())
            s = " ".join(token_arr.tolist())
            _s = "".join(token_arr.tolist())

            for sentence in sentence_list:
                if s in sentence or _s in sentence:
                    abstract_list.append(sentence)
                    abstract_list = list(set(abstract_list))
                else:
                    hit_n = 0
                    for token in token_arr.tolist():
                        if token in sentence:
                            hit_n += 1
                    if hit_n == len(token_arr.tolist()):
                        abstract_list.append(sentence)
                        abstract_list = list(set(abstract_list))

            if len(abstract_list) >= limit:
                break

        return abstract_list

    def set_readonly(self, value):
        ''' setter '''
        raise TypeError()

    def get_encoder_decoder_controller(self):
        ''' getter '''
        return self.__encoder_decoder_controller

    encoder_decoder_controller = property(get_encoder_decoder_controller, set_readonly)
