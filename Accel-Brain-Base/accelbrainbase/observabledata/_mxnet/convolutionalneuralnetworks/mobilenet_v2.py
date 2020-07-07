# -*- coding: utf-8 -*-
from accelbrainbase._mxnet.relu_n import ReLuN
from accelbrainbase._mxnet.global_avg_pool_2d import GlobalAvgPool2D
from accelbrainbase.observabledata._mxnet.convolutional_neural_networks import ConvolutionalNeuralNetworks
from accelbrainbase._mxnet._exception.init_deferred_error import InitDeferredError

from mxnet import gluon
from mxnet import autograd
import numpy as np
import mxnet as mx
from mxnet import MXNetError
from logging import getLogger
from mxnet.gluon.nn import Conv2D
from mxnet.gluon.nn import Conv2DTranspose
from mxnet.gluon.nn import BatchNorm


class MobileNetV2(ConvolutionalNeuralNetworks):
    '''
    Mobilenet V2.

    References:
        - He, K., Zhang, X., Ren, S., & Sun, J. (2016). Deep residual learning for image recognition. In Proceedings of the IEEE conference on computer vision and pattern recognition (pp. 770-778).
        - Howard, A. G., Zhu, M., Chen, B., Kalenichenko, D., Wang, W., Weyand, T., ... & Adam, H. (2017). Mobilenets: Efficient convolutional neural networks for mobile vision applications. arXiv preprint arXiv:1704.04861.
        - Sandler, M., Howard, A., Zhu, M., Zhmoginov, A., & Chen, L. C. (2018). Mobilenetv2: Inverted residuals and linear bottlenecks. In Proceedings of the IEEE Conference on Computer Vision and Pattern Recognition (pp. 4510-4520).
    '''

    # `bool` that means initialization in this class will be deferred or not.
    __init_deferred_flag = False

    def __init__(
        self,
        computable_loss,
        initializer=None,
        learning_rate=1e-05,
        learning_attenuate_rate=1.0,
        attenuate_epoch=50,
        ctx=mx.gpu(),
        hybridize_flag=True,
        activation="relu6",
        filter_multiplier=1.0,
        input_filter_n=32,
        input_kernel_size=(3, 3),
        input_strides=(2, 2),
        input_padding=(1, 1),
        bottleneck_dict_list=[
            {
                "filter_rate": 1,
                "filter_n": 16,
                "block_n": 1,
                "stride": 1
            },
            {
                "filter_rate": 6,
                "filter_n": 24,
                "block_n": 2,
                "stride": 2
            },
            {
                "filter_rate": 6,
                "filter_n": 32,
                "block_n": 3,
                "stride": 2
            },
            {
                "filter_rate": 6,
                "filter_n": 64,
                "block_n": 4,
                "stride": 2
            },
            {
                "filter_rate": 6,
                "filter_n": 96,
                "block_n": 3,
                "stride": 1
            },
            {
                "filter_rate": 6,
                "filter_n": 160,
                "block_n": 3,
                "stride": 2
            },
            {
                "filter_rate": 6,
                "filter_n": 320,
                "block_n": 1,
                "stride": 1
            },
        ],
        hidden_filter_n=1280,
        pool_size=(7, 7),
        output_nn=None,
        optimizer_name="SGD",
        shortcut_flag=True,
        global_shortcut_flag=False,
        output_batch_norm_flag=True,
        scale=1.0,
        init_deferred_flag=None,
        **kwargs
    ):
        '''
        Init.

        Args:
            computable_loss:            is-a `ComputableLoss` or `mxnet.gluon.loss`.
            initializer:                is-a `mxnet.initializer.Initializer` for parameters of model. If `None`, it is drawing from the Xavier distribution.
            learning_rate:                  `float` of learning rate.
            learning_attenuate_rate:        `float` of attenuate the `learning_rate` by a factor of this value every `attenuate_epoch`.
            attenuate_epoch:                `int` of attenuate the `learning_rate` by a factor of `learning_attenuate_rate` every `attenuate_epoch`.

            ctx:                        `mx.cpu()` or `mx.gpu()`.
            hybridize_flag:             `bool` flag whether this class will hybridize models or not.
            activation:                 `str` of activtion function.
                                        - `relu`: ReLu function.
                                        - `relu6`: ReLu6 function.
                                        - `identity`: Identity function.
                                        - `identity_adjusted`: Identity function and normalization(divided by sum) function.

            filter_multiplier:          `float` of multiplier to compress size of model.
            input_filter_n:             `int` of the number of filters in input lauer.
            input_kernel:               `tuple` or `int` of kernel size in input layer.
            input_strides:              `tuple` or `int` of strides in input layer.
            input_padding:              `tuple` or `int` of zero-padding in input layer.
            bottleneck_dict_list:       `list` of information of bottleneck layers whose `dict` means ...
                                        - `filter_rate`: `float` of filter expfilter.
                                        - `filter_n`: `int` of the number of filters.
                                        - `block_n`: `int` of the number of blocks.
                                        - `stride`: `int` or `tuple` of strides.

            hidden_filter_n:            `int` of the number of filters in hidden layers.
            pool_size:                  `tuple` or `int` of pooling size in hidden layer.
                                        If `None`, the pooling layer will not attatched in hidden layer.

            optimizer_name:             `str` of name of optimizer.

            shortcut_flag:              `bool` flag that means shortcut will be added into residual blocks or not.
            global_shortcut_flag:       `bool` flag that means shortcut will be added into residual blocks or not.
                                        This shortcut will propagate input data into output layer.
            scale:                      `float` of scaling factor for initial parameters.
            init_deferred_flag:         `bool` that means initialization in this class will be deferred or not.

        '''

        #super(MobileNetV2, self).__init__(**kwargs)
        if init_deferred_flag is None:
            init_deferred_flag = self.init_deferred_flag
        elif isinstance(init_deferred_flag, bool) is False:
            raise TypeError("The type of `init_deferred_flag` must be `bool`.")

        self.init_deferred_flag = True

        super().__init__(
            computable_loss=computable_loss,
            initializer=initializer,
            learning_rate=learning_rate,
            learning_attenuate_rate=learning_attenuate_rate,
            attenuate_epoch=attenuate_epoch,
            output_nn=output_nn,
            optimizer_name=optimizer_name,
            ctx=ctx,
            hybridize_flag=hybridize_flag,
            scale=scale,
            not_init_flag=True,
            hidden_units_list=[],
            hidden_dropout_rate_list=[],
            hidden_batch_norm_list=[],
            hidden_activation_list=[],
            **kwargs
        )
        self.init_deferred_flag = init_deferred_flag

        input_filter_n = int(round(input_filter_n * filter_multiplier))
        hidden_filter_n = int(hidden_filter_n * filter_multiplier) if int(filter_multiplier) > 1 else hidden_filter_n

        if activation == "relu6" or activation == "identity_adjusted" or activation == "identity":
            batch_norm_scale = True
        else:
            batch_norm_scale = False

        self.__input_layers_list = [
            Conv2D(
                channels=input_filter_n,
                kernel_size=input_kernel_size,
                strides=input_strides,
                padding=input_padding,
                use_bias=False,
            ),
            BatchNorm(
                axis=1,
                epsilon=1e-05,
                center=True,
                scale=batch_norm_scale
            ),
        ]
        if activation == "relu6":
            self.__input_layers_list.append(
                ReLuN(min_n=0, max_n=6)
            )
        elif activation == "relu":
            self.__input_layers_list.append(
                ReLuN(min_n=0, max_n=-1)
            )

        inverted_residual_blocks_list = [None] * len(bottleneck_dict_list)
        in_filter_n = input_filter_n

        for i in range(len(bottleneck_dict_list)):
            inverted_residual_blocks_list[i] = []
            blocks_list = []

            filter_expfilter_n = int(round(in_filter_n * bottleneck_dict_list[i]["filter_rate"]))

            for j in range(bottleneck_dict_list[i]["block_n"]):
                channel_expand_conv = Conv2D(
                    channels=filter_expfilter_n,
                    kernel_size=(1, 1),
                    strides=(1, 1),
                    padding=(0, 0),
                    groups=1,
                    use_bias=False,
                )
                if j == 0:
                    strides = bottleneck_dict_list[i]["stride"]
                else:
                    strides = (1, 1)

                bottleneck_conv = Conv2D(
                    channels=filter_expfilter_n,
                    kernel_size=(3, 3),
                    strides=strides,
                    padding=(1, 1),
                    groups=1,
                    use_bias=False,
                )
                linear_conv = Conv2D(
                    channels=bottleneck_dict_list[i]["filter_n"],
                    kernel_size=(1, 1),
                    strides=(1, 1),
                    padding=(0, 0),
                    groups=1,
                    use_bias=False,
                )
                blocks_list.append(channel_expand_conv)
                blocks_list.append(
                    BatchNorm(
                        axis=1,
                        epsilon=1e-05,
                        center=True,
                        scale=batch_norm_scale
                    ),
                )
                if activation == "relu6":
                    blocks_list.append(ReLuN(min_n=0, max_n=6))
                elif activation == "relu":
                    blocks_list.append(ReLuN(min_n=0, max_n=-1))

                blocks_list.append(bottleneck_conv)
                blocks_list.append(
                    BatchNorm(
                        axis=1,
                        epsilon=1e-05,
                        center=True,
                        scale=batch_norm_scale
                    ),
                )
                if activation == "relu6":
                    blocks_list.append(ReLuN(min_n=0, max_n=6))
                elif activation == "relu":
                    blocks_list.append(ReLuN(min_n=0, max_n=-1))

                blocks_list.append(linear_conv)
                blocks_list.append(
                    BatchNorm(
                        axis=1,
                        epsilon=1e-05,
                        center=True,
                        scale=True
                    ),
                )

            inverted_residual_blocks_list[i].append(blocks_list)

            in_filter_n = int(round(bottleneck_dict_list[i]["filter_n"] * filter_multiplier))

        self.__inverted_residual_blocks_list = inverted_residual_blocks_list

        self.__output_layers_list = [
            Conv2D(
                channels=hidden_filter_n,
                kernel_size=(1, 1),
                strides=(1, 1),
                padding=(0, 0),
                use_bias=False,
            )
        ]

        if output_batch_norm_flag is True:
            self.__output_layers_list.append(
                BatchNorm(
                    axis=1,
                    epsilon=1e-05,
                    center=True,
                    scale=batch_norm_scale
                )
            )

        if activation == "relu6":
            self.__output_layers_list.append(ReLuN(min_n=0, max_n=6))
        elif activation == "relu":
            self.__output_layers_list.append(ReLuN(min_n=0, max_n=-1))

        if pool_size is not None:
            self.__output_layers_list.append(GlobalAvgPool2D(pool_size=pool_size))

        self.__shortcut_flag = shortcut_flag

        if initializer is None:
            if activation == "relu" or activation == "relu6":
                magnitude = 2
            else:
                magnitude = 1

            self.initializer = mx.initializer.Xavier(
                rnd_type="gaussian", 
                factor_type="in", 
                magnitude=magnitude
            )
        else:
            if isinstance(initializer, mx.initializer.Initializer) is False:
                raise TypeError("The type of `initializer` must be `mxnet.initializer.Initializer`.")
            self.initializer = initializer

        with self.name_scope():
            for i in range(len(self.__input_layers_list)):
                self.register_child(self.__input_layers_list[i])

            for i in range(len(self.__inverted_residual_blocks_list)):
                for j in range(len(self.__inverted_residual_blocks_list[i])):
                    for k in range(len(self.__inverted_residual_blocks_list[i][j])):
                        self.register_child(self.__inverted_residual_blocks_list[i][j][k])

            for i in range(len(self.__output_layers_list)):
                self.register_child(self.__output_layers_list[i])

            if output_nn is not None:
                self.output_nn = output_nn

        self.__global_shortcut_flag = global_shortcut_flag

        if self.init_deferred_flag is False:
            try:
                self.collect_params().initialize(self.initializer, force_reinit=True, ctx=ctx)
                params_dict = {
                    "learning_rate": learning_rate
                }

                self.trainer = gluon.Trainer(
                    self.collect_params(), 
                    optimizer_name, 
                    params_dict
                )

                if hybridize_flag is True:
                    self.hybridize()
                    if self.output_nn is not None:
                        self.output_nn.hybridize()

            except InitDeferredError:
                self.__logger.debug("The initialization should be deferred.")

        logger = getLogger("accelbrainbase")
        self.__logger = logger

        self.__activation = activation
        self.__hybridize_flag = hybridize_flag

    def inference(self, observed_arr):
        '''
        Inference the labels.

        Args:
            observed_arr:   rank-4 Array like or sparse matrix as the observed data points.
                            The shape is: (batch size, channel, height, width)

        Returns:
            `mxnet.ndarray` of inferenced feature points.
        '''
        return self(observed_arr)

    def hybrid_forward(self, F, x):
        '''
        Hybrid forward with Gluon API.

        Args:
            F:      `mxnet.ndarray` or `mxnet.symbol`.
            x:      `mxnet.ndarray` of observed data points.
        
        Returns:
            `mxnet.ndarray` or `mxnet.symbol` of inferenced feature points.
        '''
        # rank-3
        return self.forward_propagation(F, x)

    def forward_propagation(self, F, x):
        '''
        Hybrid forward with Gluon API.

        Args:
            F:      `mxnet.ndarray` or `mxnet.symbol`.
            x:      `mxnet.ndarray` of observed data points.
        
        Returns:
            `mxnet.ndarray` or `mxnet.symbol` of inferenced feature points.
        '''
        if self.__global_shortcut_flag is True:
            global_x = x

        for i in range(len(self.__input_layers_list)):
            x = self.__input_layers_list[i](x)

        if self.__activation == "identity_adjusted":
            x = x / F.sum(F.ones_like(x))

        for i in range(len(self.__inverted_residual_blocks_list)):
            for j in range(len(self.__inverted_residual_blocks_list[i])):
                if j >= 0 and self.__shortcut_flag is True:
                    _x = x
                for k in range(len(self.__inverted_residual_blocks_list[i][j])):
                    x = self.__inverted_residual_blocks_list[i][j][k](x)
                if j >= 0 and self.__shortcut_flag is True:
                    x = F.elemwise_add(_x, x)

        if self.__activation == "identity_adjusted":
            x = x / F.sum(F.ones_like(x))

        for i in range(len(self.__output_layers_list)):
            x = self.__output_layers_list[i](x)

        if self.__global_shortcut_flag is True:
            x = F.elemwise_add(global_x, x)

        self.feature_points_arr = x

        if self.output_nn is not None:
            x = self.output_nn.forward_propagation(F, x)

        return x

    # is-a `mxnet.initializer` for parameters of model. If `None`, it is drawing from the Xavier distribution.
    __initializer = None

    def get_initializer(self):
        ''' getter for `mxnet.initializer`. '''
        return self.__initializer
    
    def set_initializer(self, value):
        ''' setter for `mxnet.initializer`. '''
        self.__initializer = value
    
    initializer = property(get_initializer, set_initializer)
