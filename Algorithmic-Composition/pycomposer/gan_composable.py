# -*- coding: utf-8 -*-
import numpy as np
import pandas as pd
from abc import ABCMeta, abstractmethod


class GANComposable(metaclass=ABCMeta):
    '''
    The interface to build an Algorithmic Composer 
    based on Generative Adversarial Networks(GANs) or its variants such as
    Conditional Generative Adversarial Networks(Conditional GANs)(Yang, L. C., et al., 2017) and 
    Algorithmic Composer based on Adversarial Auto-Encoders(AAEs) (Makhzani, A., et al., 2015).

    In the general GAN framework, the composer learns observed data points 
    drawn from a true distribution of input MIDI files and generates feature 
    points drawn from a fake distribution that means such as Uniform distribution 
    or Normal distribution, imitating the true MIDI files data.

    The components included in this class are functionally differentiated into three models.

    1. `TrueSampler`.
    2. `Generator`.
    3. `Discriminator`.

    The function of `TrueSampler` is to draw samples from a true distribution of input MIDI files. 
    `Generator` has `NoiseSampler`s and draw fake samples from a Uniform distribution or Normal 
    distribution by use it. And `Discriminator` observes those input samples, trying discriminating 
    true and fake data. 

    While `Discriminator` observes `Generator`'s observation to discrimine the output from true samples, 
    `Generator` observes `Discriminator`'s observations to confuse `Discriminator`s judgments. 
    In GANs framework, the mini-max game can be configured by the observations of observations.

    After this game, the `Generator` will grow into a functional equivalent that enables to imitate 
    the `TrueSampler` and makes it possible to compose similar but slightly different music by the 
    imitation.

    References:
        - Fang, W., Zhang, F., Sheng, V. S., & Ding, Y. (2018). A method for improving CNN-based image recognition using DCGAN. Comput. Mater. Contin, 57, 167-178.
        - Gauthier, J. (2014). Conditional generative adversarial nets for convolutional face generation. Class Project for Stanford CS231N: Convolutional Neural Networks for Visual Recognition, Winter semester, 2014(5), 2.
        - Goodfellow, I., Pouget-Abadie, J., Mirza, M., Xu, B., Warde-Farley, D., Ozair, S., ... & Bengio, Y. (2014). Generative adversarial nets. In Advances in neural information processing systems (pp. 2672-2680).
        - Long, J., Shelhamer, E., & Darrell, T. (2015). Fully convolutional networks for semantic segmentation. In Proceedings of the IEEE conference on computer vision and pattern recognition (pp. 3431-3440).
        - Makhzani, A., Shlens, J., Jaitly, N., Goodfellow, I., & Frey, B. (2015). Adversarial autoencoders. arXiv preprint arXiv:1511.05644.
        - Yang, L. C., Chou, S. Y., & Yang, Y. H. (2017). MidiNet: A convolutional generative adversarial network for symbolic-domain music generation. arXiv preprint arXiv:1703.10847.

    '''

    @abstractmethod
    def learn(self, iter_n=500, k_step=10):
        '''
        Learning.

        Args:
            iter_n:     The number of training iterations.
            k_step:     The number of learning of the `discriminator`.

        '''
        raise NotImplementedError()
    
    @abstractmethod
    def extract_logs(self):
        '''
        Extract update logs data.

        Returns:
            Tuple data.
            The shape is:
            - `list` of probabilities inferenced by the `discriminator` (mean) in the `discriminator`'s update turn.
            - `list` of probabilities inferenced by the `discriminator` (mean) in the `generator`'s update turn.

        '''
        raise NotImplementedError()

    @abstractmethod
    def compose(self, file_path, velocity_mean=None, velocity_std=None):
        '''
        Compose by learned model.

        Args:
            file_path:      Path to generated MIDI file.
            velocity_mean:  Mean of velocity.
                            This class samples the velocity from a Gaussian distribution of 
                            `velocity_mean` and `velocity_std`.
                            If `None`, the average velocity in MIDI files set to this parameter.

            velocity_std:   Standard deviation(SD) of velocity.
                            This class samples the velocity from a Gaussian distribution of 
                            `velocity_mean` and `velocity_std`.
                            If `None`, the SD of velocity in MIDI files set to this parameter.
        '''
        raise NotImplementedError()
