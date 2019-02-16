# -*- coding: utf-8 -*-
import numpy as np
from pygan.noise_sampler import NoiseSampler
from pygan.noisesampler.gauss_sampler import GaussSampler
from pydbm.cnn.featuregenerator.image_generator import ImageGenerator


class ImageSampler(NoiseSampler):
    '''
    Sampler which draws samples from the noise prior of images.
    '''

    def __init__(
        self,
        batch_size,
        image_dir,
        seq_len=None,
        gray_scale_flag=True,
        wh_size_tuple=(100, 100),
        norm_mode="z_score",
        add_noise_sampler=None
    ):
        '''
        Init.

        Args:
            training_image_dir:             Dir path which stores image files for training.
            test_image_dir:                 Dir path which stores image files for test.
            seq_len:                        The length of one sequence.
            gray_scale_flag:                Gray scale or not(RGB).
            wh_size_tuple:                  Tuple(`width`, `height`).
            norm_mode:                      How to normalize pixel values of images.
                                            - `z_score`: Z-Score normalization.
                                            - `min_max`: Min-max normalization.
                                            - `tanh`: Normalization by tanh function.

            add_noise_sampler:              is-a `NoiseSampler` to add noise to image feature.
        '''
        self.__feature_generator = ImageGenerator(
            epochs=1,
            batch_size=batch_size,
            training_image_dir=image_dir,
            test_image_dir=image_dir,
            seq_len=seq_len,
            gray_scale_flag=gray_scale_flag,
            wh_size_tuple=wh_size_tuple,
            norm_mode=norm_mode
        )
        if add_noise_sampler is None:
            if seq_len is None:
                output_shape = (batch_size, wh_size_tuple[0], wh_size_tuple[1])
            else:
                output_shape = (batch_size, seq_len, wh_size_tuple[0], wh_size_tuple[1])

            self.__add_noise_sampler = gauss_sampler(
                mu=0.0, 
                sigma=1.0,
                output_shape=output_shape
            )
        else:
            self.__add_noise_sampler = add_noise_sampler

        self.__norm_mode = norm_mode

    def generate(self):
        '''
        Draws samples from the `true` distribution.
        
        Returns:
            `np.ndarray` of samples.
        '''
        observed_arr = None
        for result_tuple in self.__feature_generator.generate():
            observed_arr = result_tuple[0]
            break

        observed_arr = observed_arr + self.__add_noise_sampler.generate()

        if self.__norm_mode == "z_score":
            for i in range(observed_arr.shape[0]):
                observed_arr[i] = (observed_arr[i] - observed_arr[i].mean()) / observed_arr[i].std()
        elif self.__norm_mode == "min_max":
            for i in range(observed_arr.shape[0]):
                observed_arr[i] = (observed_arr[i] - observed_arr[i].min()) / (observed_arr[i].max() - observed_arr[i].min())
        elif self.__norm_mode == "tanh":
            observed_arr = np.tanh(observed_arr)
        
        return observed_arr
