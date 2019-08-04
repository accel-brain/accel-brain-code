# -*- coding: utf-8 -*-
import numpy as np
from pysummarization.vectorizable_token import VectorizableToken
from pydbm.nn.simple_auto_encoder import SimpleAutoEncoder
from pydbm.nn.neural_network import NeuralNetwork as Encoder
from pydbm.nn.neural_network import NeuralNetwork as Decoder
from pydbm.nn.nn_layer import NNLayer as EncoderLayer
from pydbm.nn.nn_layer import NNLayer as DecoderLayer
from pydbm.activation.softmax_function import SoftmaxFunction
from pydbm.loss.cross_entropy import CrossEntropy
from pydbm.activation.identity_function import IdentityFunction
from pydbm.activation.tanh_function import TanhFunction
from pydbm.activation.logistic_function import LogisticFunction

from pydbm.optimization.optparams.adam import Adam
from pydbm.synapse.nn_graph import NNGraph as EncoderGraph
from pydbm.synapse.nn_graph import NNGraph as DecoderGraph
from pydbm.verification.verificate_function_approximation import VerificateFunctionApproximation
from pydbm.optimization.batch_norm import BatchNorm

import numpy as np
import pandas as pd
from logging import getLogger


class SkipGramVectorizer(VectorizableToken):
    '''
    Vectorize token by skip-gram.
    '''

    def __init__(
        self,
        token_list,
        epochs=300,
        skip_n=1,
        batch_size=50,
        feature_dim=20,
        scale=1e-05,
        learning_rate=1e-05,
        auto_encoder=None
    ):
        '''
        Initialize.
        
        Args:
            token_list:         The list of all tokens in all sentences.
            skip_n:             N of n-gram.
            training_count:     The epochs.
            batch_size:         Batch size.
            learning_rate:      Learning rate.
            feature_dim:        The dimension of feature points.
        '''
        if auto_encoder is not None and isinstance(auto_encoder, SimpleAutoEncoder) is False:
            raise TypeError()

        self.__logger = getLogger("pydbm")
        self.__token_arr = np.array(token_list)
        self.__token_uniquie_arr = np.array(list(set(token_list)))

        if auto_encoder is None:
            activation_function = TanhFunction()

            encoder_graph = EncoderGraph(
                activation_function=activation_function,
                hidden_neuron_count=self.__token_uniquie_arr.shape[0],
                output_neuron_count=feature_dim,
                scale=scale,
            )

            encoder_layer = EncoderLayer(encoder_graph)

            opt_params = Adam()
            opt_params.dropout_rate = 0.5

            encoder = Encoder(
                nn_layer_list=[
                    encoder_layer, 
                ],
                epochs=epochs,
                batch_size=batch_size,
                learning_rate=learning_rate,
                learning_attenuate_rate=1.0,
                attenuate_epoch=50,
                computable_loss=CrossEntropy(),
                opt_params=opt_params,
                verificatable_result=VerificateFunctionApproximation(),
                test_size_rate=0.3,
                tol=1e-15
            )

            decoder_graph = DecoderGraph(
                activation_function=SoftmaxFunction(),
                hidden_neuron_count=feature_dim,
                output_neuron_count=self.__token_uniquie_arr.shape[0],
                scale=scale,
            )

            decoder_layer = DecoderLayer(decoder_graph)

            opt_params = Adam()
            opt_params.dropout_rate = 0.0

            decoder = Decoder(
                nn_layer_list=[
                    decoder_layer, 
                ],
                epochs=epochs,
                batch_size=batch_size,
                learning_rate=learning_rate,
                learning_attenuate_rate=1.0,
                attenuate_epoch=50,
                computable_loss=CrossEntropy(),
                opt_params=opt_params,
                verificatable_result=VerificateFunctionApproximation(),
                test_size_rate=0.3,
                tol=1e-15
            )

            auto_encoder = SimpleAutoEncoder(
                encoder=encoder,
                decoder=decoder,
                epochs=epochs,
                batch_size=batch_size,
                learning_rate=learning_rate,
                learning_attenuate_rate=1.0,
                attenuate_epoch=50,
                computable_loss=CrossEntropy(),
                verificatable_result=VerificateFunctionApproximation(),
                test_size_rate=0.3,
                tol=1e-15,
            )

        self.__auto_encoder = auto_encoder

        self.__epochs = epochs
        self.__batch_size = batch_size
        self.__skip_n = skip_n

    def learn(self):
        '''
        Learn.
        '''
        batch_key_arr = np.arange(self.__token_uniquie_arr.shape[0])
        np.random.shuffle(batch_key_arr)
        batch_observed_arr, batch_labeled_arr = None, None
        for batch_key in batch_key_arr:
            token_key_arr = np.where(self.__token_arr == self.__token_uniquie_arr[batch_key])[0]
            token_key = token_key_arr[np.random.randint(low=0, high=token_key_arr.shape[0])]

            token_observed_arr = np.zeros(self.__token_uniquie_arr.shape[0])
            uniquie_key = np.where(self.__token_uniquie_arr == self.__token_arr[token_key])[0]
            token_observed_arr[uniquie_key] = 1.0

            token_labeled_arr = np.zeros(self.__token_uniquie_arr.shape[0])
            for token_key in token_key_arr:
                for n in range(1, self.__skip_n+1):
                    try:
                        skip_key = np.where(
                            self.__token_uniquie_arr == self.__token_arr[token_key - n]
                        )[0]
                        token_labeled_arr[skip_key] += 1.0
                    except IndexError:
                        pass

                    try:
                        skip_key = np.where(
                            self.__token_uniquie_arr == self.__token_arr[token_key + n]
                        )[0]
                        token_labeled_arr[skip_key] += 1.0
                    except IndexError:
                        continue

            token_labeled_arr = token_labeled_arr / token_labeled_arr.sum()
            
            if batch_observed_arr is None:
                batch_observed_arr = np.expand_dims(token_observed_arr, axis=0)
            else:
                batch_observed_arr = np.r_[batch_observed_arr, np.expand_dims(token_observed_arr, axis=0)]
            if batch_labeled_arr is None:
                batch_labeled_arr = np.expand_dims(token_labeled_arr, axis=0)
            else:
                batch_labeled_arr = np.r_[batch_labeled_arr, np.expand_dims(token_labeled_arr, axis=0)]

        self.__auto_encoder.learn(batch_observed_arr, batch_labeled_arr)

    def vectorize(self, token_list):
        '''
        Tokenize token list.
        
        Args:
            token_list:   The list of tokens.
        
        Returns:
            [vector of token, vector of token, vector of token, ...]
        '''
        batch_observed_arr = None
        for token in token_list:
            token_observed_arr = np.zeros(self.__token_uniquie_arr.shape[0])
            uniquie_key = np.where(self.__token_uniquie_arr == token)[0]
            token_observed_arr[uniquie_key] = 1.0

            if batch_observed_arr is None:
                batch_observed_arr = np.expand_dims(token_observed_arr, axis=0)
            else:
                batch_observed_arr = np.r_[batch_observed_arr, np.expand_dims(token_observed_arr, axis=0)]

        return self.__auto_encoder.encoder.inference(batch_observed_arr).tolist()

    def convert_tokens_into_matrix(self, token_list):
        '''
        Create matrix of sentences.

        Args:
            token_list:     The list of tokens.
        
        Returns:
            2-D `np.ndarray` of sentences.
            Each row means one hot vectors of one sentence.
        '''
        return np.array(self.vectorize(token_list)).astype(np.float32)

    def tokenize(self, vector_list):
        '''
        Tokenize vector.

        Args:
            vector_list:    The list of vector of one token.
        
        Returns:
            token
        '''
        vector_arr = np.array(vector_list)
        if vector_arr.ndim == 2 and vector_arr.shape[0] > 1:
            vector_arr = np.nanmean(vector_arr, axis=0)

        vector_arr = vector_arr.reshape(1, -1)

        batch_observed_arr = np.zeros((
            self.__token_uniquie_arr.shape[0],
            self.__token_uniquie_arr.shape[0]
        ))
        for i in range(batch_observed_arr.shape[0]):
            batch_observed_arr[i, i] = 1.0
        
        feature_arr = self.__auto_encoder.encoder.inference(batch_observed_arr)
        diff_arr = np.nansum(np.square(vector_arr - feature_arr), axis=1)
        return np.array([self.__token_uniquie_arr[diff_arr.argmin(axis=0)]])

    def get_token_arr(self):
        ''' getter '''
        return self.__token_arr
    
    def set_token_arr(self, value):
        ''' setter '''
        raise TypeError("This property must be read-only.")
    
    token_arr = property(get_token_arr, set_token_arr)

    def set_readonly(self, value):
        ''' setter '''
        raise TypeError("This property must be read-only.")
    
    def get_token_list(self):
        ''' getter '''
        return self.__token_list
    
    token_list = property(get_token_list, set_readonly)

    def get_auto_encoder(self):
        ''' getter '''
        return self.__auto_encoder
    
    def set_auto_encoder(self, value):
        ''' setter '''
        if isinstance(value, SimpleAutoEncoder) is False:
            raise TypeError()
        self.__auto_encoder = value
    
    auto_encoder = property(get_auto_encoder, set_auto_encoder)
