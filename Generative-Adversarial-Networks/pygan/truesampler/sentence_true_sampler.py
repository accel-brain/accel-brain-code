# -*- coding: utf-8 -*-
import numpy as np
from pygan.true_sampler import TrueSampler
from pysummarization.itertextgenerator.iter_sentence_generator import IterSentenceGenerator


class SentenceTrueSampler(TrueSampler):
    '''
    Sampler which draws samples from the `true` distribution of sentences.
    '''

    def __init__(
        self,
        token_list,
        nlp_base,
        tokenizable_doc,
        vectorizable_token,
        batch_size=20,
        seq_len=10,
        norm_mode="z_score"
    ):
        '''
        Init.

        Args:
            token_list:             `list` of all token.
            tokenizable_doc:        is-a `TokenizableDoc`.
            vectorizable_token:     is-a `VectorizableToken`.
            batch_size:             Batch size.
            seq_len:                The length of sequence.
            norm_mode:              How to normalize pixel values of images.
                                    - `z_score`: Z-Score normalization.
                                    - `min_max`: Min-max normalization.
                                    - `tanh`: Normalization by tanh function.
        '''
        self.__iter_text_generator = IterSentenceGenerator(
                token_list=token_list,
                nlp_base=nlp_base,
                tokenizable_doc=tokenizable_doc,
                vectorizable_token=vectorizable_token,
                epochs=1,
                batch_size=batch_size,
                seq_len=seq_len
        )
        self.__norm_mode = norm_mode

    def draw(self):
        '''
        Draws samples from the `true` distribution.
        
        Returns:
            `np.ndarray` of samples.
        '''
        observed_arr = None
        for result_tuple in self.__iter_text_generator.generate_real_token():
            observed_arr = result_tuple[0]
            break

        if self.__norm_mode == "z_score":
            if observed_arr.std() != 0:
                observed_arr = (observed_arr - observed_arr.mean()) / observed_arr.std()
        elif self.__norm_mode == "min_max":
            if observed_arr.max() > observed_arr.min():
                observed_arr = (observed_arr - observed_arr.min()) / (observed_arr.max() - observed_arr.min())
        elif self.__norm_mode == "tanh":
            observed_arr = np.tanh(observed_arr)
        return observed_arr

    def get_iter_text_generator(self):
        ''' getter '''
        return self.__iter_text_generator
    
    def set_iter_text_generator(self, value):
        ''' setter '''
        self.__iter_text_generator = value
    
    iter_text_generator = property(get_iter_text_generator, set_iter_text_generator)
