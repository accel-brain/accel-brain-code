# -*- coding: utf-8 -*-
from pygan.generative_model import GenerativeModel
from abc import abstractmethod


class ConditionalGenerativeModel(GenerativeModel):
    '''
    Generate samples based on the conditional noise prior.

    `GenerativeModel` that has a `Conditioner`, where the function of 
    `Conditioner` is a conditional mechanism to use previous knowledge 
    to condition the generations, incorporating information from previous 
    observed data points to itermediate layers of the `Generator`. 
    In this method, this model can "look back" without a recurrent unit 
    as used in RNN or LSTM.

    This model observes not only random noises but also any other prior
    information as a previous knowledge and outputs feature points.
    Dut to the `Conditioner`, this model has the capacity to exploit
    whatever prior knowledge that is available and can be represented
    as a matrix or tensor.

    References:
        - Mirza, M., & Osindero, S. (2014). Conditional generative adversarial nets. arXiv preprint arXiv:1411.1784.

    '''

    @abstractmethod
    def extract_conditions(self):
        '''
        Extract samples of conditions.
        
        Returns:
            `np.ndarray` of samples.
        '''
        raise NotImplementedError()
