# -*- coding: utf-8 -*-
from logging import getLogger, StreamHandler, NullHandler, DEBUG, ERROR
import numpy as np
from pysummarization.nlp_base import NlpBase
from pysummarization.tokenizabledoc.mecab_tokenizer import MeCabTokenizer
from pysummarization.similarity_filter import SimilarityFilter
from pysummarization.vectorizable_sentence import VectorizableSentence
from pysummarization.clusterable_doc import ClusterableDoc
from pysummarization.vectorizablesentence.lstm_rtrbm import LSTMRTRBM


class LSTMRTRBMClustering(SimilarityFilter):
    '''
    Concrete class for filtering mutually similar sentences.
    '''
    
    def __init__(
        self,
        document,
        clusterable_doc,
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
            clusterable_doc:                is-a `ClusterableDoc`.
            tokenizable_doc:                is-a `TokenizableDoc`.
            hidden_neuron_count:            The number of units in hidden layer.
            training_count:                 The number of training.
            bath_size:                      Batch size of Mini-batch.
            learning_rate:                  Learning rate.
            seq_len:                        The length of one sequence.
            debug_mode:                     Debug mode or not.
        '''
        if isinstance(clusterable_doc, ClusterableDoc) is False:
            raise TypeError()
        if isinstance(vectorizable_sentence, VectorizableSentence) is False:
            raise TypeError()

        if debug_mode is True:
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

        feature_arr = vectorlizable_sentence.vectorize(sentence_list)
        _ = clusterable_doc.learn(feature_arr)

        self.__vectorlizable_sentence = vectorlizable_sentence
        self.__token_master_list = token_master_list
        self.__sentence_list = sentence_list
        self.__batch_size = batch_size
        self.__clusterable_doc = clusterable_doc

    def calculate(self, token_list_x, token_list_y):
        '''
        Check whether `token_list_x` and `token_list_y` belonging to the same cluster, 
        and if so, this method returns `1.0`, if not, returns `0.0`.
        
        Concrete method.
        
        Args:
            token_list_x:    [token, token, token, ...]
            token_list_y:    [token, token, token, ...]
        
        Returns:
            `0.0` or `1.0`.
        '''
        if len(token_list_x) == 0 or len(token_list_y) == 0:
            return 0.0

        x_list = self.__sentence_list[:self.__batch_size-1]
        y_list = self.__sentence_list[:self.__batch_size-1]
        x_list.append(token_list_x)
        y_list.append(token_list_y)
        x_arr = self.__vectorlizable_sentence.vectorize(x_list)[-1]
        y_arr = self.__vectorlizable_sentence.vectorize(y_list)[-1]
        labeled_arr = self.__clusterable_doc.inference(np.r_[x_arr, y_arr])
        
        if labeled_arr[0] == labeled_arr[1]:
            return 1.0
        else:
            return 0.0
