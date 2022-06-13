# -*- coding: utf-8 -*-
from logging import getLogger
import numpy as np

from pysummarization.abstractable_semantics import AbstractableSemantics
from pysummarization.vectorizable_token import VectorizableToken

from logging import getLogger
from logging import getLogger, StreamHandler, NullHandler, DEBUG, ERROR
import numpy as np
import pandas as pd
import torch
from torch import nn

from accelbrainbase.computableloss._torch.l2_norm_loss import L2NormLoss
from accelbrainbase.extractabledata.unlabeled_csv_extractor import UnlabeledCSVExtractor
from accelbrainbase.iteratabledata._torch.unlabeled_sequential_csv_iterator import UnlabeledSequentialCSVIterator
from accelbrainbase.noiseabledata._torch.gauss_noise import GaussNoise
from accelbrainbase.observabledata._torch.lstm_networks import LSTMNetworks
from accelbrainbase.observabledata._torch.lstmnetworks.encoder_decoder import EncoderDecoder
from accelbrainbase.iteratable_data import IteratableData


class ReSeq2Seq(AbstractableSemantics):
    '''
    A retrospective sequence-to-sequence learning(re-seq2seq).

    The concept of the re-seq2seq(Zhang, K. et al., 2018) provided inspiration to this library.
    This model is a new sequence learning model mainly in the field of Video Summarizations.
    "The key idea behind re-seq2seq is to measure how well the machine-generated summary 
    is similar to the original video in an abstract semantic space" (Zhang, K. et al., 2018, p3).

    The encoder of a seq2seq model observes the original video and output feature points
    which represents the semantic meaning of the observed data points.
    Then the feature points is observed by the decoder of this model.
    Additionally, in the re-seq2seq model, the outputs of the decoder is propagated
    to a retrospective encoder, which infers feature points to represent the 
    semantic meaning of the summary. "If the summary preserves the important and 
    relevant information in the original video, then we should expect that the 
    two embeddings are similar (e.g. in Euclidean distance)" (Zhang, K. et al., 2018, p3).

    This library refers to this intuitive insight above to apply the model to text summarizations.
    Like videos, semantic feature representation based on representation learning of manifolds 
    is also possible in text summarizations.
    
    The intuition in the design of their loss function is also suggestive.
    "The intuition behind our modeling is that the outputs should convey 
    the same amount of information as the inputs. For summarization, 
    this is precisely the goal: a good summary should be such that after viewing 
    the summary, users would get about the same amount of information as if they 
    had viewed the original video" (Zhang, K. et al., 2018, p7).

    But the model in this library and Zhang, K. et al.(2018) are different in some respects
    from the relation with the specification of the Deep Learning library: [accel-brain-base](https://github.com/accel-brain/accel-brain-code/tree/master/Accel-Brain-Base).
    First, Encoder/Decoder based on LSTM is not designed as a hierarchical structure. 
    Second, it is possible to introduce regularization techniques which are not discussed in 
    Zhang, K. et al.(2018) such as the dropout, the gradient clipping, and limitation of weights.
    Third, the regression loss function for matching summaries is simplified in terms of 
    calculation efficiency in this library.

    **Note** that penalty terms that penalize mismatched pairs is not implemented 
    due to an implementation issue.

    References:
        - Zhang, K., Grauman, K., & Sha, F. (2018). Retrospective Encoders for Video Summarization. In Proceedings of the European Conference on Computer Vision (ECCV) (pp. 383-399).
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
        margin_param=0.01,
        retrospective_lambda=0.5,
        retrospective_eta=0.5,
        encoder_decoder_controller=None,
        retrospective_encoder=None,
        hidden_neuron_count=20,
        output_neuron_count=20,
        dropout_rate=0.5,
        batch_size=20,
        learning_rate=1e-05,
        optimizer_name="sgd",
        seq_len=8,
        ctx="cpu",
        initializer_f=None,
        optimizer_f=None,
    ):
        '''
        Init.

        Args:
            margin_param:                   A margin parameter for the mismatched pairs penalty.
            retrospective_lambda:           Tradeoff parameter for loss function.
            retrospective_eta:              Tradeoff parameter for loss function.
            encoder_decoder_controller:     is-a `EncoderDecoderController`.
            retrospective_encoder:          is-a `LSTMModel` as a retrospective encoder(or re-encoder).
            hidden_neuron_count:            The number of units in hidden layers.
            output_neuron_count:            The number of units in output layers.

            dropout_rate:                   Probability of dropout.

            batch_size:                     Batch size.
            learning_rate:                  Learning rate.
            learning_attenuate_rate:        Attenuate the `learning_rate` by a factor of this value every `attenuate_epoch`.
            attenuate_epoch:                Attenuate the `learning_rate` by a factor of `learning_attenuate_rate` every `attenuate_epoch`.
                                            Additionally, in relation to regularization,
                                            this class constrains weight matrixes every `attenuate_epoch`.

            grad_clip_threshold:            Threshold of the gradient clipping.
            seq_len:                        The length of sequneces in Decoder with Attention model.

        '''
        super(ReSeq2Seq, self).__init__()
        self.__ctx = ctx
        if computable_loss is None:
            computable_loss = L2NormLoss()

        if isinstance(margin_param, float) is False:
            raise TypeError("The type of `margin_param` must be `float`.")
        if margin_param <= 0:
            raise ValueError("The value of `margin_param` must be more than `0`.")

        self.__margin_param = margin_param

        if isinstance(retrospective_lambda, float) is False or isinstance(retrospective_eta, float) is False:
            raise TypeError("The type of `retrospective_lambda` and `retrospective_eta` must be `float`.")

        if retrospective_lambda < 0 or retrospective_eta < 0:
            raise ValueError("The values of `retrospective_lambda` and `retrospective_eta` must be more then `0`.")
        if retrospective_lambda + retrospective_eta != 1:
            raise ValueError("The sum of `retrospective_lambda` and `retrospective_eta` must be `1`.")

        self.__retrospective_lambda = retrospective_lambda
        self.__retrospective_eta = retrospective_eta

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
                ctx=ctx
            )
        else:
            if isinstance(encoder_decoder_controller, EncoderDecoderController) is False:
                raise TypeError()

        if retrospective_encoder is None:
            retrospective_encoder = self.__build_retrospective_encoder(
                computable_loss=computable_loss,
                hidden_neuron_count=hidden_neuron_count,
                output_neuron_count=output_neuron_count,
                dropout_rate=dropout_rate,
                learning_rate=learning_rate,
                seq_len=seq_len,
                initializer_f=initializer_f,
                optimizer_f=optimizer_f,
                ctx=ctx
            )
        else:
            if isinstance(retrospective_encoder, LSTMModel) is False:
                raise TypeError()

        self.__encoder_decoder_controller = encoder_decoder_controller
        self.__retrospective_encoder = retrospective_encoder

        self.__batch_size = batch_size
        self.__learning_rate = learning_rate

        self.__output_neuron_count = output_neuron_count
        self.__hidden_neuron_count = hidden_neuron_count

        logger = getLogger("accelbrainbase")
        self.__logger = logger
        self.__logs_tuple_list = []

        self.epoch = 0

    def __build_encoder_decoder_controller(
        self,
        computable_loss,
        hidden_neuron_count=20,
        output_neuron_count=20,
        dropout_rate=0.5,
        learning_rate=1e-05,
        seq_len=8,
        initializer_f=None,
        optimizer_f=None,
        ctx="cpu"
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
            ctx=ctx,
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
            output_layer_flag=False,
            # `bool` for using bias or not in output layer(last hidden layer).
            output_no_bias_flag=True,
            # `mx.cpu()` or `mx.gpu()`.
            ctx=ctx,
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
            ctx=ctx,
        )
        self.__computable_loss = computable_loss
        return encoder_decoder_controller

    def __build_retrospective_encoder(
        self,
        computable_loss,
        hidden_neuron_count=20,
        output_neuron_count=20,
        dropout_rate=0.5,
        learning_rate=1e-05,
        seq_len=8,
        initializer_f=None,
        optimizer_f=None,
        ctx="cpu"
    ):
        retrospective_encoder = LSTMNetworks(
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
            output_layer_flag=False,
            # `mx.cpu()` or `mx.gpu()`.
            ctx=ctx,
        )

        return retrospective_encoder

    def learn(self, iteratable_data):
        '''
        Learn the observed data points
        for vector representation of the input time-series.

        Args:
            iteratable_data:     is-a `IteratableData`.

        '''
        if isinstance(iteratable_data, IteratableData) is False:
            raise TypeError("The type of `iteratable_data` must be `IteratableData`.")

        learning_rate = self.__learning_rate

        encoder_best_params_list = []
        decoder_best_params_list = []
        re_encoder_best_params_list = []
        try:
            self.__memory_tuple_list = []
            eary_stop_flag = False
            loss_list = []
            min_loss = None
            epoch = self.epoch
            iter_n = 0
            for batch_observed_arr, batch_target_arr, test_batch_observed_arr, test_batch_target_arr in iteratable_data.generate_learned_samples():
                if epoch == 0 and iter_n == 0:
                    batch_size = batch_observed_arr.shape[0]
                    seq_len = batch_observed_arr.shape[1]
                    self.__encoder_decoder_controller.initialize_params(
                        input_dim=batch_observed_arr.reshape((batch_size, seq_len, -1)).shape[2],
                        input_seq_len=seq_len
                    )

                    torch.autograd.set_detect_anomaly(True)

                    decoded_arr = self.__encoder_decoder_controller(batch_observed_arr)
                    batch_size = decoded_arr.shape[0]
                    seq_len = decoded_arr.shape[1]
                    self.__retrospective_encoder.initialize_params(
                        input_dim=decoded_arr.reshape((batch_size, seq_len, -1)).shape[2],
                        input_seq_len=seq_len
                    )
                self.__encoder_decoder_controller.encoder.optimizer.zero_grad()
                self.__encoder_decoder_controller.decoder.optimizer.zero_grad()
                self.__retrospective_encoder.optimizer.zero_grad()

                observed_arr, encoded_arr, decoded_arr, re_encoded_arr = self.inference(batch_observed_arr)
                loss = self.compute_retrospective_loss(
                    observed_arr, 
                    encoded_arr, 
                    decoded_arr, 
                    re_encoded_arr
                )
                loss.backward()
                self.__retrospective_encoder.optimizer.step()
                self.__encoder_decoder_controller.decoder.optimizer.step()
                self.__encoder_decoder_controller.encoder.optimizer.step()

                if (iter_n+1) % int(iteratable_data.iter_n / iteratable_data.epochs) == 0:
                    with torch.inference_mode():
                        test_observed_arr, test_encoded_arr, test_decoded_arr, test_re_encoded_arr = self.inference(test_batch_observed_arr)
                        test_loss = self.compute_retrospective_loss(test_observed_arr, test_encoded_arr, test_decoded_arr, test_re_encoded_arr)
                    _loss = loss.to('cpu').detach().numpy().copy()
                    _test_loss = test_loss.to('cpu').detach().numpy().copy()

                    self.__verificate_retrospective_loss(_loss, _test_loss)

                    epoch += 1
                iter_n += 1

        except KeyboardInterrupt:
            self.__logger.debug("Interrupt.")
        
        self.__logger.debug("end. ")
        self.epoch = epoch

    def inference(self, observed_arr):
        '''
        Infernece by the model.

        Args:
            observed_arr:       `np.ndarray` of observed data points.

        Returns:
            `np.ndarray` of inferenced feature points.
        '''
        return self.forward(observed_arr)

    def forward(self, x):
        '''
        Hybrid forward with Gluon API.

        Args:
            F:      `mxnet.ndarray` or `mxnet.symbol`.
            x:      `mxnet.ndarray` of observed data points.
        
        Returns:
            `mxnet.ndarray` or `mxnet.symbol` of inferenced feature points.
        '''
        observed_arr = x
        decoded_arr = self.__encoder_decoder_controller(observed_arr)
        encoded_arr = self.__encoder_decoder_controller.feature_points_arr
        re_encoded_arr = self.__retrospective_encoder(decoded_arr)

        return observed_arr, encoded_arr, decoded_arr, re_encoded_arr

    def summarize(self, iteratable_data, vectorizable_token, sentence_list, limit=5):
        '''
        Summarize input document.

        Args:
            iteratable_data:        is-a `IteratableData`.
            vectorizable_token:     is-a `VectorizableToken`.
            sentence_list:          `list` of all sentences.
            limit:                  The number of selected abstract sentence.
        
        Returns:
            `list` of `str` of abstract sentences.
        '''
        if isinstance(vectorizable_token, VectorizableToken) is False:
            raise TypeError()

        _score_arr = None
        _test_arr = None
        for _, _, test_arr, _ in iteratable_data.generate_inferenced_samples():
            observed_arr, encoded_arr, decoded_arr, re_encoded_arr = self.inference(test_arr)
            delta_arr = self.compute_retrospective_delta(observed_arr, encoded_arr, decoded_arr, re_encoded_arr)
            score_arr = delta_arr.reshape(delta_arr.shape[0], -1).mean(dim=1).reshape(delta_arr.shape[0], 1).to('cpu').detach().numpy().copy()
            if _score_arr is None:
                _score_arr = score_arr
            else:
                _score_arr = np.r_[_score_arr, score_arr]
            if _test_arr is None:
                _test_arr = test_arr.to('cpu').detach().numpy().copy()
            else:
                _test_arr = np.r_[_test_arr, test_arr.to('cpu').detach().numpy().copy()]

        score_list = _score_arr.tolist()
        score_arr = _score_arr
        test_arr = _test_arr

        abstract_list = []
        for i in range(limit):
            key = score_arr.argmin()
            seq_arr = test_arr[key]

            score = score_list.pop(key)
            score_arr = np.array(score_list)

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

    def compute_retrospective_delta(
        self,
        observed_arr, 
        encoded_arr, 
        decoded_arr, 
        re_encoded_arr
    ):
        if self.__output_neuron_count == self.__hidden_neuron_count:
            target_arr =  torch.sub(
                encoded_arr, 
                torch.unsqueeze(observed_arr.mean(dim=2), dim=2)
            )
            summary_delta_arr = decoded_arr - target_arr
        else:
            # For each batch, draw a samples from the Uniform distribution.
            if self.__output_neuron_count > self.__hidden_neuron_count:
                all_dim_arr = np.arange(self.__output_neuron_count)
                np.random.shuffle(all_dim_arr)
                choiced_dim_arr = all_dim_arr[:self.__hidden_neuron_count]
                target_arr = torch.sub(
                    encoded_arr, 
                    torch.unsqueeze(
                        observed_arr[:, :, choiced_dim_arr].mean(
                            dim=2
                        ), 
                        dim=2
                    )
                )
                summary_delta_arr = decoded_arr[:, :, choiced_dim_arr] - target_arr
            else:
                all_dim_arr = np.arange(self.__hidden_neuron_count)
                np.random.shuffle(all_dim_arr)
                choiced_dim_arr = all_dim_arr[:self.__output_neuron_count]
                target_arr = torch.sub(
                    encoded_arr[:, :, choiced_dim_arr], 
                    torch.unsqueeze(
                        observed_arr.mean(
                            dim=2
                        ), 
                        dim=2
                    )
                )
                summary_delta_arr = decoded_arr - target_arr

        match_delta_arr = None

        for i in range(self.__batch_size):
            arr = encoded_arr[i, -1] - re_encoded_arr[i, -1]
            if match_delta_arr is None:
                match_delta_arr = torch.unsqueeze(arr, dim=0)
            else:
                match_delta_arr = torch.cat(
                    (
                        match_delta_arr,
                        torch.unsqueeze(arr, dim=0)
                    ),
                    dim=0
                )

        summary_delta_arr = torch.abs(summary_delta_arr)
        match_delta_arr = torch.abs(match_delta_arr)
        delta_arr = summary_delta_arr + torch.unsqueeze(
            self.__retrospective_lambda * match_delta_arr, dim=1
        )

        return delta_arr

    def compute_retrospective_loss(
        self,
        observed_arr, 
        encoded_arr, 
        decoded_arr, 
        re_encoded_arr
    ):
        '''
        Compute retrospective loss.

        Returns:
            The tuple data.
            - `np.ndarray` of delta.
            - `np.ndarray` of losses of each batch.
            - float of loss of all batch.

        '''
        delta_arr = self.compute_retrospective_delta(
            observed_arr, 
            encoded_arr, 
            decoded_arr, 
            re_encoded_arr
        )

        loss = torch.mean(
            delta_arr,
        )

        return loss

    def __verificate_retrospective_loss(self, train_loss, test_loss):
        self.__logger.debug("Epoch: " + str(len(self.__logs_tuple_list) + 1))

        try:
            _train_loss = train_loss.to('cpu').detach().numpy().copy()
        except:
            _train_loss = train_loss
        try:
            _test_loss = test_loss.to('cpu').detach().numpy().copy()
        except:
            _test_loss = test_loss

        self.__logger.debug("Loss: ")
        self.__logger.debug(
            "Training: " + str(_train_loss) + " Test: " + str(_test_loss)
        )
        self.__logs_tuple_list.append(
            (
                _train_loss,
                _test_loss
            )
        )

    def __rename_file(self, filename):
        filename_list = filename.split(".")
        _format = filename_list[-1]
        encoder_decoder_filename = filename.replace("." + _format, "_encoder_decoder." + _format)
        retrospective_encoder_filename = filename.replace("." + _format, "_retrospective_encoder." + _format)
        return encoder_decoder_filename, retrospective_encoder_filename

    def save_parameters(self, filename):
        '''
        Save parameters to files.

        Args:
            filename:       File name.
        '''
        encoder_decoder_filename, retrospective_encoder_filename = self.__rename_file(filename)
        self.__encoder_decoder_controller.save_parameters(encoder_decoder_filename)
        self.__retrospective_encoder.save_parameters(retrospective_encoder_filename)

    def load_parameters(self, filename, ctx=None, allow_missing=False, ignore_extra=False):
        '''
        Load parameters to files.

        Args:
            filename:       File name.
            ctx:            `mx.cpu()` or `mx.gpu()`.
            allow_missing:  `bool` of whether to silently skip loading parameters not represents in the file.
            ignore_extra:   `bool` of whether to silently ignre parameters from the file that are not present in this `Block`.
        '''
        encoder_decoder_filename, retrospective_encoder_filename = self.__rename_file(filename)
        self.__encoder_decoder_controller.save_parameters(encoder_decoder_filename, ctx=ctx, allow_missing=allow_missing, ignore_extra=ignore_extra)
        self.__retrospective_encoder.save_parameters(retrospective_encoder_filename, ctx=ctx, allow_missing=allow_missing, ignore_extra=ignore_extra)

    def get_logs_arr(self):
        ''' getter '''
        return np.array(
            self.__logs_tuple_list,
        )

    def set_readonly(self, value):
        ''' setter '''
        raise TypeError()
    
    logs_arr = property(get_logs_arr, set_readonly)

    def get_encoder_decoder_controller(self):
        ''' getter '''
        return self.__encoder_decoder_controller
    
    encoder_decoder_controller = property(get_encoder_decoder_controller, set_readonly)

    def get_retrospective_encoder(self):
        ''' getter '''
        return self.__retrospective_encoder
    
    retrospective_encoder = property(get_retrospective_encoder, set_readonly)
