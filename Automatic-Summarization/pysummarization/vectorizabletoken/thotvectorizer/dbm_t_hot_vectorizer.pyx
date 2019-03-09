# -*- coding: utf-8 -*-
import numpy as np
from pysummarization.vectorizabletoken.t_hot_vectorizer import THotVectorizer
from pysummarization.computabledistance.euclid_distance import EuclidDistance
# `StackedAutoEncoder` is-a `DeepBoltzmannMachine`.
from pydbm.dbm.deepboltzmannmachine.stacked_auto_encoder import StackedAutoEncoder
# The `Concrete Builder` in Builder Pattern.
from pydbm.dbm.builders.dbm_multi_layer_builder import DBMMultiLayerBuilder
# Contrastive Divergence for function approximation.
from pydbm.approximation.contrastive_divergence import ContrastiveDivergence
# Tanh Function as activation function.
from pydbm.activation.tanh_function import TanhFunction


class DBMTHotVectorizer(THotVectorizer):
    '''
    Vectorize token by t-hot Vectorizer.

    This class outputs the dimension reduced vectors with Deep Boltzmann Machines
    as a Stacked Auto Encoder.
    '''
    # is-a `StackedAutoEncoder`.
    __dbm = None
    # is-a `ComputableDistance`.
    __computable_distance = None

    def pre_learn(
        self,
        hidden_n=100,
        training_count=1000,
        batch_size=10,
        learning_rate=1e-05,
        dbm=None
    ):
        if dbm is not None and isinstance(dbm, StackedAutoEncoder) is False:
            raise TypeError("The type of `dbm` must be `StackedAutoEncoder`.")

        vector_arr = np.array(super().vectorize(self.token_arr.tolist()))
        vector_arr = (vector_arr - vector_arr.mean()) / (vector_arr.std() + 1e-08)

        if dbm is None:
            # Setting objects for activation function.
            activation_list = [
                TanhFunction(), 
                TanhFunction(), 
                TanhFunction()
            ]

            # Setting the object for function approximation.
            approximaion_list = [ContrastiveDivergence(), ContrastiveDivergence()]

            dbm = StackedAutoEncoder(
                DBMMultiLayerBuilder(),
                [vector_arr.shape[1], 100, vector_arr.shape[1]],
                activation_list,
                approximaion_list,
                learning_rate # Setting learning rate.
            )

            # Execute learning.
            dbm.learn(
                vector_arr,
                training_count=1000, # If approximation is the Contrastive Divergence, this parameter is `k` in CD method.
                batch_size=100,  # Batch size in mini-batch training.
                r_batch_size=-1,  # if `r_batch_size` > 0, the function of `dbm.learn` is a kind of reccursive learning.
                sgd_flag=True
            )

        dbm.learn(
            vector_arr,
            training_count=1,
            batch_size=vector_arr.shape[0],
            r_batch_size=-1,
            sgd_flag=True
        )
        self.__dbm = dbm

    def vectorize(self, token_list):
        '''
        Tokenize token list.
        
        Args:
            token_list:   The list of tokens.
        
        Returns:
            [vector of token, vector of token, vector of token, ...]
        '''
        return [self.__dbm_t_hot(token).tolist() for token in token_list]

    def tokenize(self, vector_list):
        '''
        Tokenize vector.

        Args:
            vector_list:    The list of vector of one token.
        
        Returns:
            token
        '''
        if self.computable_distance is None:
            self.computable_distance = EuclidDistance()
        vector_arr = np.array(vector_list)
        distance_arr = np.empty_like(vector_arr)
        for i in range(vector_arr):
            distance_arr[i] = self.computable_distance.compute(
                vector_arr[i],
                self.__dbm.get_feature_point(layer_number=1)
            )
        key_arr = distance_arr.argmin(axis=0)
        return self.__token_arr[key_arr]

    def __dbm_t_hot(self, token):
        arr = np.zeros(len(self.__token_arr))
        key = self.__token_arr.tolist().index(token)
        arr = self.__dbm.get_feature_point(layer_number=1)[key]
        return arr

    def get_dbm(self):
        ''' getter '''
        return self.__dbm
    
    def set_dbm(self, value):
        ''' setter '''
        raise TypeError("This property must be read-only.")
    
    dbm = property(get_dbm, set_dbm)

    def get_computable_distance(self):
        ''' getter '''
        return self.__computable_distance
    
    def set_computable_distance(self, value):
        ''' setter '''
        if isinstance(value, ComputableDistance) is False:
            raise TypeError()
        self.__computable_distance = value
    
    computable_distance = property(get_computable_distance, set_computable_distance)
