# -*- coding: utf-8 -*-
import numpy as np
from pygan.noise_sampler import NoiseSampler
from pydbm.cnn.featuregenerator.image_generator import ImageGenerator


class ImageNoiseSampler(NoiseSampler):
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
        norm_mode="z_score"
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

        if self.noise_sampler is not None:
            self.noise_sampler.output_shape = observed_arr.shape
            observed_arr += self.noise_sampler.generate()

        observed_arr = observed_arr.astype(float)
        if self.__norm_mode == "z_score":
            if observed_arr.std() != 0:
                observed_arr = (observed_arr - observed_arr.mean()) / observed_arr.std()
        elif self.__norm_mode == "min_max":
            if (observed_arr.max() - observed_arr.min()) != 0:
                observed_arr = (observed_arr - observed_arr.min()) / (observed_arr.max() - observed_arr.min())
        elif self.__norm_mode == "tanh":
            observed_arr = np.tanh(observed_arr)
        
        return observed_arr
