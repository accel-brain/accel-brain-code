# -*- coding: utf-8 -*-
from abc import ABCMeta, abstractmethod, abstractproperty


class NoiseSampler(metaclass=ABCMeta):
    '''
    Generate samples based on the noise prior.
    '''

    # is-a `NoiseSampler`.
    __noise_sampler = None

    def get_noise_sampler(self):
        ''' getter for a `NoiseSampler`. '''
        return self.__noise_sampler
    
    def set_noise_sampler(self, value):
        ''' setter for a `NoiseSampler`. '''
        if isinstance(value, NoiseSampler) is False:
            raise TypeError()
        self.__noise_sampler = value
    
    noise_sampler = property(get_noise_sampler, set_noise_sampler)

    @abstractmethod
    def generate(self):
        '''
        Generate noise samples.
        
        Returns:
            `np.ndarray` of samples.
        '''
        raise NotImplementedError()
