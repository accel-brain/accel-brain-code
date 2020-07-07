# -*- coding: utf-8 -*-
from mxnet.gluon.nn.conv_layers import _Pooling


class GlobalAvgPool2D(_Pooling):
    """Global average pooling operation for spatial data.

    Parameters
    ----------
    pool_size:    tuple, default (1, 1).

    layout : str, default 'NCHW'
        Dimension ordering of data and out ('NCHW' or 'NHWC').
        'N', 'C', 'H', 'W' stands for batch, channel, height, and width
        dimensions respectively.


    Inputs:
        - **data**: 4D input tensor with shape
            `(batch_size, in_channels, height, width)` when `layout` is `NCHW`.
            For other layouts shape is permuted accordingly.

    Outputs:
        - **out**: 4D output tensor with shape
            `(batch_size, channels, 1, 1)` when `layout` is `NCHW`.
    """
    def __init__(self, pool_size=(1, 1), layout='NCHW', **kwargs):
        assert layout in ('NCHW', 'NHWC'),\
            "Only NCHW and NHWC layouts are valid for 2D Pooling"
        super(GlobalAvgPool2D, self).__init__(
            pool_size, None, 0, True, True, 'avg', layout, **kwargs)
