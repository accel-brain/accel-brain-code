# -*- coding: utf-8 -*-
from accelbrainbase.extractable_data import ExtractableData
from PIL import Image
import numpy as np


class ImageExtractor(ExtractableData):
    '''
    Image Extractor.
    '''

    # `int` of width.
    __width = 128
    # `int` of height.
    __height = 96
    # `int` of channel.
    __channel = 3

    def __init__(
        self, 
        width,
        height,
        channel,
    ):
        '''
        Init.

        Args:
            width:          `int` of image width.
            height:         `int` of image height.
            channel:        `int` of channel of image.
        '''
        self.width = width
        self.height = height
        self.channel = channel

    def extract(
        self,
        path,
    ):
        '''
        Extract image file data.

        Args:
            path:     `str` of image files.
        
        Returns:
            Observed data points.
        '''
        img = Image.open(path)
        img = img.resize((self.width, self.height))

        if self.channel == 1:
            img = img.convert("L")
        img_arr = np.asarray(img)

        if self.channel == 3 or self.channel == 4:
            img_arr = img_arr.transpose((2, 0, 1))
        elif self.channel == 1:
            img_arr = np.expand_dims(img_arr, axis=0)

        return img_arr

    def get_width(self):
        ''' getter of `int` of width. '''
        return self.__width

    def set_width(self, value):
        ''' setter of `int` of width. '''
        if isinstance(value, int):
            self.__width = value
        else:
            raise TypeError("The type of `width` must be `int`.")
    
    width = property(get_width, set_width)

    def get_height(self):
        ''' getter of `int` of height. '''
        return self.__height
    
    def set_height(self, value):
        ''' setter of `int` of height. '''
        if isinstance(value, int):
            self.__height = value
        else:
            raise TypeError("The type of `height` must be `int`.")

    height = property(get_height, set_height)

    def get_channel(self):
        ''' getter of `int` of channel. '''
        return self.__channel

    def set_channel(self, value):
        ''' setter of `int` of channel. '''
        if isinstance(value, int):
            self.__channel = value
        else:
            raise TypeError("The type of `channel` must be `int`.")

    channel = property(get_channel, set_channel)
