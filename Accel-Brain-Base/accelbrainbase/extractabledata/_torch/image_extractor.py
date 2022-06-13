# -*- coding: utf-8 -*-
from accelbrainbase.extractabledata.image_extractor import ImageExtractor as _ImageExtractor
from PIL import Image
import numpy as np
import torch


class ImageExtractor(_ImageExtractor):
    '''
    Image Extractor.
    '''

    def __init__(
        self, 
        width,
        height,
        channel,
        ctx="cpu"
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
        img_arr = torch.from_numpy(img_arr.astype(np.float32)).clone()
        img_arr = img_arr.to(self.__ctx).float()

        return img_arr
