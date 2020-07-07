# -*- coding: utf-8 -*-
from accelbrainbase.extractabledata.image_extractor import ImageExtractor as _ImageExtractor
from PIL import Image
import numpy as np
import mxnet as mx


class ImageExtractor(_ImageExtractor):
    '''
    Image Extractor.
    '''

    def __init__(
        self, 
        width,
        height,
        channel,
        ctx=mx.gpu()
    ):
        '''
        Init.

        Args:
            width:          `int` of image width.
            height:         `int` of image height.
            channel:        `int` of channel of image.
            ctx:            `mx.cpu()` or `mx.gpu()`.
        '''
        self.width = width
        self.height = height
        self.channel = channel
        self.__ctx = ctx

    def extract(
        self,
        path,
    ):
        '''
        Extract image file data.

        Args:
            path:     `str` of image files.
        
        Returns:
            `mxnet.ndarray` of data.
            The shape is (`channel`, `width`, `height`).
        '''
        img_arr = super().extract(path=path)
        img_arr = mx.ndarray.array(img_arr, ctx=self.__ctx)

        return img_arr
