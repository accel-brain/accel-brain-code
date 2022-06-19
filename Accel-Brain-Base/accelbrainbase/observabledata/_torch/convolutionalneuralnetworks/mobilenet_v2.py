# -*- coding: utf-8 -*-
from accelbrainbase.observabledata._torch.convolutional_neural_networks import ConvolutionalNeuralNetworks
import numpy as np
from logging import getLogger
import torch
from torch.optim.adam import Adam


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
        optimizer_f=None,
        learning_rate=1e-05,
        weight_decay=0.01,
        ctx="cpu",
        activation="relu6",
        filter_multiplier=1.0,
        input_in_channel=3,
        input_filter_n=32,
        input_kernel_size=3,
        input_strides=2,
        input_padding=1,
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
        pool_size=7,
        output_nn=None,
        shortcut_flag=True,
        global_shortcut_flag=False,
        output_batch_norm_flag=True,
        regularizatable_data_list=[],
        init_deferred_flag=None,
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
                                        - `relu6`: ReLU6 function.
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
            regularizatable_data_list:  `list` of `RegularizatableData`.

        '''
        if init_deferred_flag is None:
            init_deferred_flag = self.init_deferred_flag
        elif isinstance(init_deferred_flag, bool) is False:
            raise TypeError("The type of `init_deferred_flag` must be `bool`.")

        self.init_deferred_flag = True

        super().__init__(
            computable_loss=computable_loss,
            learning_rate=learning_rate,
            output_nn=output_nn,
            ctx=ctx,
            not_init_flag=True,
            regularizatable_data_list=regularizatable_data_list,
            hidden_units_list=[],
            hidden_dropout_rate_list=[],
            hidden_batch_norm_list=[],
            hidden_activation_list=[],
        )
        self.init_deferred_flag = init_deferred_flag

        input_filter_n = int(round(input_filter_n * filter_multiplier))
        hidden_filter_n = int(hidden_filter_n * filter_multiplier) if int(filter_multiplier) > 1 else hidden_filter_n

        self.__input_layers_list = [
            torch.nn.Conv2d(
                in_channels=input_in_channel,
                out_channels=input_filter_n,
                kernel_size=input_kernel_size,
                stride=input_strides,
                padding=input_padding,
                bias=False,
            ),
            torch.nn.BatchNorm2d(
                input_filter_n,
            ),
        ]
        if activation == "relu6":
            self.__input_layers_list.append(
                torch.nn.ReLU6()
            )
        elif activation == "relu":
            self.__input_layers_list.append(
                torch.nn.ReLu()
            )
        self.__input_layers_list = torch.nn.ModuleList(self.__input_layers_list)

        inverted_residual_blocks_list = [None] * len(bottleneck_dict_list)
        in_filter_n = input_filter_n

        for i in range(len(bottleneck_dict_list)):
            inverted_residual_blocks_list[i] = []
            blocks_list = []

            filter_expfilter_n = int(round(in_filter_n * bottleneck_dict_list[i]["filter_rate"]))

            for j in range(bottleneck_dict_list[i]["block_n"]):
                if j == 0:
                    in_channels = input_filter_n
                else:
                    in_channels = filter_expfilter_n

                channel_expand_conv = torch.nn.Conv2d(
                    in_channels=in_channels,
                    out_channels=filter_expfilter_n,
                    kernel_size=1,
                    stride=1,
                    padding=0,
                    groups=1,
                    bias=False,
                )
                if j == 0:
                    strides = bottleneck_dict_list[i]["stride"]
                else:
                    strides = 1

                bottleneck_conv = torch.nn.Conv2d(
                    in_channels=filter_expfilter_n,
                    out_channels=filter_expfilter_n,
                    kernel_size=3,
                    stride=strides,
                    padding=1,
                    groups=1,
                    bias=False,
                )
                linear_conv = torch.nn.Conv2d(
                    in_channels=filter_expfilter_n,
                    out_channels=bottleneck_dict_list[i]["filter_n"],
                    kernel_size=1,
                    stride=1,
                    padding=0,
                    groups=1,
                    bias=False,
                )
                blocks_list.append(channel_expand_conv)
                blocks_list.append(
                    torch.nn.BatchNorm2d(
                        bottleneck_dict_list[i]["filter_n"]
                    ),
                )
                if activation == "relu6":
                    blocks_list.append(torch.nn.ReLU6())
                elif activation == "relu":
                    blocks_list.append(torch.nn.ReLu())

                blocks_list.append(bottleneck_conv)
                blocks_list.append(
                    torch.nn.BatchNorm2d(
                        bottleneck_dict_list[i]["filter_n"]
                    ),
                )
                if activation == "relu6":
                    blocks_list.append(torch.nn.ReLU6())
                elif activation == "relu":
                    blocks_list.append(torch.nn.ReLu())

                blocks_list.append(linear_conv)
                blocks_list.append(
                    torch.nn.BatchNorm2d(bottleneck_dict_list[i]["filter_n"]),
                )
                blocks_list = torch.nn.ModuleList(blocks_list)

            inverted_residual_blocks_list[i].append(blocks_list)
            inverted_residual_blocks_list[i] = torch.nn.ModuleList(inverted_residual_blocks_list[i])

            in_filter_n = int(round(bottleneck_dict_list[i]["filter_n"] * filter_multiplier))

        # TODO:ModuleListの2d 版
        self.__inverted_residual_blocks_list = torch.nn.ModuleList(inverted_residual_blocks_list)

        self.__output_layers_list = [
            torch.nn.Conv2d(
                in_channels=bottleneck_dict_list[i]["filter_n"],
                out_channels=hidden_filter_n,
                kernel_size=1,
                stride=1,
                padding=0,
                bias=False,
            )
        ]

        if output_batch_norm_flag is True:
            self.__output_layers_list.append(
                torch.nn.BatchNorm2d(
                    hidden_filter_n
                )
            )

        if activation == "relu6":
            self.__output_layers_list.append(torch.nn.ReLU6())
        elif activation == "relu":
            self.__output_layers_list.append(torch.nn.ReLu())

        if pool_size is not None:
            self.__output_layers_list.append(torch.nn.AdaptiveAvgPool2d(pool_size))

        self.__output_layers_list = torch.nn.ModuleList(self.__output_layers_list)

        self.__shortcut_flag = shortcut_flag

        if output_nn is not None:
            self.output_nn = output_nn

        self.__global_shortcut_flag = global_shortcut_flag

        self.__ctx = ctx
        self.to(self.__ctx)

        if self.init_deferred_flag is False:
            if optimizer_f is not None:
                self.optimizer = optimizer_f(
                    self.parameters()
                )
            else:
                self.optimizer = Adam(
                    self.parameters(),
                    lr=learning_rate,
                    weight_decay=weight_decay,
                )

        logger = getLogger("accelbrainbase")
        self.__logger = logger

        self.__activation = activation
        self.epoch = 0

    def inference(self, observed_arr):
        '''
        Inference the labels.

        Args:
            observed_arr:   rank-4 Array like or sparse matrix as the observed data points.
                            The shape is: (batch size, channel, height, width)

        Returns:
            `mxnet.ndarray` of inferenced feature points.
        '''
        return self.forward(observed_arr)

    def forward(self, x):
        '''
        Forward with Gluon API.

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
            x = x / torch.sum(torch.ones_like(x))

        for i in range(len(self.__inverted_residual_blocks_list)):
            for j in range(len(self.__inverted_residual_blocks_list[i])):
                if j >= 0 and self.__shortcut_flag is True:
                    _x = x
                for k in range(len(self.__inverted_residual_blocks_list[i][j])):
                    x = self.__inverted_residual_blocks_list[i][j][k](x)
                if j >= 0 and self.__shortcut_flag is True:
                    x = torch.add(_x, x)

        if self.__activation == "identity_adjusted":
            x = x / torch.sum(torch.ones_like(x))

        for i in range(len(self.__output_layers_list)):
            x = self.__output_layers_list[i](x)

        if self.__global_shortcut_flag is True:
            x = torch.add(global_x, x)

        self.feature_points_arr = x

        if self.output_nn is not None:
            x = self.output_nn(x)

        return x
