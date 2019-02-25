# -*- coding: utf-8 -*-
import numpy as np
from pygan.noise_sampler import NoiseSampler
from pygan.noisesampler.gauss_sampler import GaussSampler
from pysummarization.itertextgenerator.iter_sentence_generator import IterSentenceGenerator


class SentenceSampler(NoiseSampler):
    '''
    Sampler which draws samples from the noise prior of tokens.
    '''

    def __init__(
        self,
        document,
        nlp_base,
        tokenizable_doc,
        vectorizable_token,
        batch_size=20,
        seq_len=10,
        norm_mode="z_score",
        add_noise_sampler=None
    ):
        '''
        Init.

        Args:
            document:               `str` of all sentence.
            tokenizable_doc:        is-a `TokenizableDoc`.
            vectorizable_token:     is-a `VectorizableToken`.
            batch_size:             Batch size.
            seq_len:                The length of sequence.
            norm_mode:              How to normalize pixel values of images.
                                    - `z_score`: Z-Score normalization.
                                    - `min_max`: Min-max normalization.
                                    - `tanh`: Normalization by tanh function.

            add_noise_sampler:      is-a `NoiseSampler` to add noise to image feature.
        '''
        self.__iter_text_generator = IterSentenceGenerator(
                document=document,
                nlp_base=nlp_base,
                tokenizable_doc=tokenizable_doc,
                vectorizable_token=vectorizable_token,
                epochs=1,
                batch_size=batch_size,
                seq_len=seq_len
        )
        self.__add_noise_sampler = add_noise_sampler
        self.__norm_mode = norm_mode

    def generate(self):
        '''
        Draws samples from the `true` distribution.
        
        Returns:
            `np.ndarray` of samples.
        '''
        observed_arr = None
        token_arr = None
        for result_tuple in self.__iter_text_generator.generate_uniform():
            observed_arr = result_tuple[0]
            token_arr = result_tuple[1]
            break

        if self.__add_noise_sampler is not None:
            observed_arr = observed_arr + self.__add_noise_sampler.generate()

        if self.__norm_mode == "z_score":
            if observed_arr.std() != 0:
                observed_arr = (observed_arr - observed_arr.mean()) / observed_arr.std()
        elif self.__norm_mode == "min_max":
            if observed_arr.max() > observed_arr.min():
                observed_arr = (observed_arr - observed_arr.min()) / (observed_arr.max() - observed_arr.min())
        elif self.__norm_mode == "tanh":
            observed_arr = np.tanh(observed_arr)

        self.__token_arr = token_arr
        return observed_arr

    def get_iter_text_generator(self):
        ''' getter '''
        return self.__iter_text_generator
    
    def set_iter_text_generator(self, value):
        ''' setter '''
        self.__iter_text_generator = value
    
    iter_text_generator = property(get_iter_text_generator, set_iter_text_generator)

    def get_token_arr(self):
        ''' getter '''
        return self.__token_arr
    
    def set_readonly(self, value):
        ''' setter '''
        raise TypeError()
    
    token_arr = property(get_token_arr, set_readonly)
