# -*- coding: utf-8 -*-
from logging import getLogger, StreamHandler, NullHandler, DEBUG, ERROR
import numpy as np
from pysummarization.nlp_base import NlpBase
from pysummarization.tokenizabledoc.mecab_tokenizer import MeCabTokenizer
from pysummarization.similarity_filter import SimilarityFilter
from pysummarization.vectorizablesentence.encoder_decoder import EncoderDecoder


class EncoderDecoderCosine(SimilarityFilter):
    '''
    Concrete class for filtering mutually similar sentences.
    '''
    
    def __init__(
        self,
        document,
        tokenizable_doc=None,
        hidden_neuron_count=200,
        epochs=100,
        batch_size=100,
        learning_rate=1e-05,
        learning_attenuate_rate=0.1,
        attenuate_epoch=50,
        bptt_tau=8,
        weight_limit=0.5,
        dropout_rate=0.5,
        test_size_rate=0.3,
        debug_mode=False
    ):
        '''
        Init.
        
        Args:
            document:                       String of document.
            tokenizable_doc:                is-a `TokenizableDoc`.
            hidden_neuron_count:            The number of units in hidden layer.
            epochs:                         Epochs of Mini-batch.
            bath_size:                      Batch size of Mini-batch.
            learning_rate:                  Learning rate.
            learning_attenuate_rate:        Attenuate the `learning_rate` by a factor of this value every `attenuate_epoch`.
            attenuate_epoch:                Attenuate the `learning_rate` by a factor of `learning_attenuate_rate` every `attenuate_epoch`.
                                            Additionally, in relation to regularization,
                                            this class constrains weight matrixes every `attenuate_epoch`.

            bptt_tau:                       Refereed maxinum step `t` in Backpropagation Through Time(BPTT).
            weight_limit:                   Regularization for weights matrix
                                            to repeat multiplying the weights matrix and `0.9`
                                            until $\sum_{j=0}^{n}w_{ji}^2 < weight\_limit$.

            dropout_rate:                   The probability of dropout.
            test_size_rate:                 Size of Test data set. If this value is `0`, the 
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

        all_token_list = []
        for i in range(len(sentence_list)):
            nlp_base.tokenize(sentence_list[i])
            all_token_list.extend(nlp_base.token)
            sentence_list[i] = nlp_base.token

        token_master_list = list(set(all_token_list))
        vectorlizable_sentence = EncoderDecoder()
        vectorlizable_sentence.learn(
            sentence_list=sentence_list, 
            token_master_list=token_master_list,
            hidden_neuron_count=hidden_neuron_count,
            epochs=epochs,
            batch_size=batch_size,
            learning_rate=learning_rate,
            learning_attenuate_rate=learning_attenuate_rate,
            attenuate_epoch=attenuate_epoch,
            bptt_tau=bptt_tau,
            weight_limit=weight_limit,
            dropout_rate=dropout_rate,
            test_size_rate=test_size_rate
        )
        self.__vectorlizable_sentence = vectorlizable_sentence
        self.__token_master_list = token_master_list

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

        x_arr = self.__vectorlizable_sentence.vectorize([token_list_x])[0]
        y_arr = self.__vectorlizable_sentence.vectorize([token_list_y])[0]

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
