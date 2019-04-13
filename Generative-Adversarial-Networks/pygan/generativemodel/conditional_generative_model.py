# -*- coding: utf-8 -*-
from pygan.generative_model import GenerativeModel
from abc import abstractmethod


class ConditionalGenerativeModel(GenerativeModel):
    '''
    Generate samples based on the conditonal noise prior.

    `GenerativeModel` that has a `Conditioner`, where the function of 
    `Conditioner` is a conditional mechanism to use previous knowledge 
    to condition the generations, incorporating information from previous 
    observed data points to itermediate layers of the `Generator`. 
    In this method, this model can "look back" without a recurrent unit 
    as used in RNN or LSTM.

    This model observes not only random noises but also any other prior
    information as a previous knowledge and outputs feature points.
    Dut to the `Conditoner`, this model has the capacity to exploit
    whatever prior knowledge that is available and can be represented
    as a matrix or tensor.

    References:
        - Yang, L. C., Chou, S. Y., & Yang, Y. H. (2017). MidiNet: A convolutional generative adversarial network for symbolic-domain music generation. arXiv preprint arXiv:1703.10847.

    '''

    @abstractmethod
    def extract_conditions(self):
        '''
        Extract samples of conditions.
        
        Returns:
            `np.ndarray` of samples.
        '''
        raise NotImplementedError()
