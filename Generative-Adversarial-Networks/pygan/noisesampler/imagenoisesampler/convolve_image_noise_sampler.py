# -*- coding: utf-8 -*-
import numpy as np
from pygan.noisesampler.image_noise_sampler import ImageNoiseSampler
from pydbm.cnn.featuregenerator.image_generator import ImageGenerator
from pydbm.cnn.layerablecnn.convolution_layer import ConvolutionLayer
from pydbm.synapse.cnn_graph import CNNGraph
from pydbm.activation.tanh_function import TanhFunction


class ConvolveImageNoiseSampler(ImageNoiseSampler):
    '''
    Sampler which draws samples from the noise prior of images
    and has convolution operator to convolve sampled image data.

    This sampler will not learn as CNNs model
    but *condition* input noise.
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
        super().__init__(
            batch_size=batch_size,
            image_dir=image_dir,
            seq_len=seq_len,
            gray_scale_flag=gray_scale_flag,
            wh_size_tuple=wh_size_tuple,
            norm_mode=norm_mode
        )

        if gray_scale_flag is True:
            channel = 1
        else:
            channel = 3

        self.__conv_layer = ConvolutionLayer(
            CNNGraph(
                activation_function=TanhFunction(),
                filter_num=batch_size,
                channel=channel,
                kernel_size=3,
                scale=0.1,
                stride=1,
                pad=1
            )
        )

    def generate(self):
        '''
        Draws samples from the `true` distribution.
        
        Returns:
            `np.ndarray` of samples.
        '''
        observed_arr = super().generate()
        return self.__conv_layer.convolve(observed_arr)

    def get_conv_layer(self):
        ''' getter '''
        return self.__conv_layer
    
    def set_conv_layer(self, value):
        ''' setter '''
        self.__conv_layer = value
    
    conv_layer = property(get_conv_layer, set_conv_layer)
