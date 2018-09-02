# -*- coding: utf-8 -*-
from logging import getLogger, StreamHandler, NullHandler, DEBUG, ERROR
import numpy as np
from pysummarization.nlp_base import NlpBase
from pysummarization.tokenizabledoc.mecab_tokenizer import MeCabTokenizer
from pysummarization.similarity_filter import SimilarityFilter
from pysummarization.vectorizablesentence.lstm_rtrbm import LSTMRTRBM


class LSTMRTRBMCosine(SimilarityFilter):
    '''
    Concrete class for filtering mutually similar sentences.
    '''
    
    def __init__(
        self,
        document,
        tokenizable_doc=None,
        hidden_neuron_count=1000,
        training_count=1,
        batch_size=10,
        learning_rate=1e-03,
        seq_len=5,
        debug_mode=False
    ):
        '''
        Init.
        
        Args:
            document:                       String of document.
            tokenizable_doc:                is-a `TokenizableDoc`.
            hidden_neuron_count:            The number of units in hidden layer.
            training_count:                 The number of training.
            bath_size:                      Batch size of Mini-batch.
            learning_rate:                  Learning rate.
            seq_len:                        The length of one sequence.
            debug_mode:                     Debug mode or not.
        '''
        if debug_mode is True:
            logger = getLogger("pydbm")
            handler = StreamHandler()
            handler.setLevel(DEBUG)
            logger.setLevel(DEBUG)
            logger.addHandler(handler)

            logger = getLogger("pysummarization")
            handler = StreamHandler()
            handler.setLevel(DEBUG)
            logger.setLevel(DEBUG)
            logger.addHandler(handler)

        # The object of NLP.
        nlp_base = NlpBase()
        if tokenizable_doc is None:
            # Set tokenizer. This is japanese tokenizer with MeCab.
            nlp_base.tokenizable_doc = MeCabTokenizer()
        else:
            nlp_base.tokenizable_doc = tokenizable_doc

        sentence_list = nlp_base.listup_sentence(document)
        if len(sentence_list) < batch_size:
            raise ValueError("The number of sentence is insufficient.")

        all_token_list = []
        for i in range(len(sentence_list)):
            nlp_base.tokenize(sentence_list[i])
            all_token_list.extend(nlp_base.token)
            sentence_list[i] = nlp_base.token

        token_master_list = list(set(all_token_list))
        vectorlizable_sentence = LSTMRTRBM()
        vectorlizable_sentence.learn(
            sentence_list=sentence_list, 
            token_master_list=token_master_list,
            hidden_neuron_count=hidden_neuron_count,
            training_count=training_count,
            batch_size=batch_size,
            learning_rate=learning_rate,
            seq_len=seq_len
        )
        self.__vectorlizable_sentence = vectorlizable_sentence
        self.__token_master_list = token_master_list
        self.__sentence_list = sentence_list
        self.__batch_size = batch_size

    def calculate(self, token_list_x, token_list_y):
        '''
        Calculate similarity with the so-called Cosine similarity of Tf-Idf vectors.
        
        Concrete method.
        
        Args:
            token_list_x:    [token, token, token, ...]
            token_list_y:    [token, token, token, ...]
        
        Returns:
            Similarity.
        '''
        if len(token_list_x) == 0 or len(token_list_y) == 0:
            return 0.0

        x_list = self.__sentence_list[:self.__batch_size-1]
        y_list = self.__sentence_list[:self.__batch_size-1]
        x_list.append(token_list_x)
        y_list.append(token_list_y)
        x_arr = self.__vectorlizable_sentence.vectorize(x_list)[-1]
        y_arr = self.__vectorlizable_sentence.vectorize(y_list)[-1]

        dot_prod = np.dot(x_arr, y_arr)
        norm_x = np.linalg.norm(x_arr)
        norm_y = np.linalg.norm(y_arr)
        try:
            result = dot_prod / (norm_x * norm_y)
            if np.isnan(result) is True:
                return 0.0
            else:
                return result
        except ZeroDivisionError:
            return 0.0
