# -*- coding: utf-8 -*-
from accelbrainbase.observabledata._torch.neuralnetworks.auto_encoder import AutoEncoder
from accelbrainbase.noiseabledata._torch.gauss_noise import GaussNoise
from accelbrainbase.noiseable_data import NoiseableData

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim.adam import Adam
import numpy as np
from logging import getLogger


class LadderNetworks(AutoEncoder):
    '''
    Ladder Networks.

    In most classification problems, finding and producing labels for the samples is hard. 
    In many cases plenty of unlabeled data existand it seems obvious that using them should 
    improve the results. For instance, there are plenty of unlabeled images available and 
    in most image classification tasks there are vastly more bits of information in the 
    statistical structure of input images than in their labels.

    It is argued here that the reason why unsupervised learning has not been able to improve 
    results is that most current versions are incompatible with supervised learning. 
    The problem is that many un-supervised learning methods try to represent as much 
    information about the original data as possible whereas supervised learning tries to 
    filter out all the information which is irrelevant for the task at hand.

    Ladder network is an Auto-Encoder which can discard information.
    Unsupervised learning needs to toleratediscarding information in order to 
    work well with supervised learning. Many unsupervised learning methods are 
    not good at this but one class of models stands out as an exception: 
    hierarchical latent variable models. Unfortunately their derivation can be 
    quite complicated and often involves approximations which compromise their per-formance.
    
    A simpler alternative is offered by Auto-Encoders which also have the benefit 
    of being compatible with standard supervised feedforward networks. They would be a 
    promising candidate for combining supervised and unsupervised learning but unfortunately 
    Auto-Encoders normally correspond to latent variable models with a single layer of 
    stochastic variables, that is, they do not tolerate discarding information.

    Ladder network makes it possible to solve that problem by settting recursive derivation 
    of the learning rule with a distributed cost function, building denoisng Auto-Encoder recursively.
    Normally denoising Auto-Encoders have a fixed input but the cost functions on the higher layers 
    can influence their input mappings and this creates a bias towards PCA-type solutions.

    References:
        - Bengio, Y., Lamblin, P., Popovici, D., & Larochelle, H. (2007). Greedy layer-wise training of deep networks. In Advances in neural information processing systems (pp. 153-160).
        - Dumoulin, V., & V,kisin, F. (2016). A guide to convolution arithmetic for deep learning. arXiv preprint arXiv:1603.07285.
        - Erhan, D., Bengio, Y., Courville, A., Manzagol, P. A., Vincent, P., & Bengio, S. (2010). Why does unsupervised pre-training help deep learning?. Journal of Machine Learning Research, 11(Feb), 625-660.
        - Erhan, D., Courville, A., & Bengio, Y. (2010). Understanding representations learned in deep architectures. Department dInformatique et Recherche Operationnelle, University of Montreal, QC, Canada, Tech. Rep, 1355, 1.
        - Goodfellow, I., Bengio, Y., & Courville, A. (2016). Deep learning (adaptive computation and machine learning series). Adaptive Computation and Machine Learning series, 800.
        - Masci, J., Meier, U., Cireşan, D., & Schmidhuber, J. (2011, June). Stacked convolutional auto-encoders for hierarchical feature extraction. In International Conference on Artificial Neural Networks (pp. 52-59). Springer, Berlin, Heidelberg.
        - Rasmus, A., Berglund, M., Honkala, M., Valpola, H., & Raiko, T. (2015). Semi-supervised learning with ladder networks. In Advances in neural information processing systems (pp. 3546-3554).
        - Valpola, H. (2015). From neural PCA to deep unsupervised learning. In Advances in Independent Component Analysis and Learning Machines (pp. 143-171). Academic Press.
    '''

    # is-a `NoiseableData`.
    __noiseable_data = GaussNoise(mu=0.0, sigma=1e-03)

    def get_noiseable_data(self):
        ''' getter '''
        return self.__noiseable_data

    def set_noiseable_data(self, value):
        ''' setter '''
        if isinstance(value, NoiseableData) is True:
            self.__noiseable_data = value
        else:
            raise TypeError("The type of `noiseable_data` must be `NoiseableData`.")

    noiseable_data = property(get_noiseable_data, set_noiseable_data)

    __recoding_ld_loss = False

    def get_recoding_ld_loss(self):
        ''' getter '''
        return self.__recoding_ld_loss
    
    def set_recoding_ld_loss(self, value):
        ''' setter '''
        self.__recoding_ld_loss = value

    recoding_ld_loss = property(get_recoding_ld_loss, set_recoding_ld_loss)

    # alpha weight.
    __alpha = 1e-05
    # sigma weight.
    __sigma = 0.7
    # mu weight.
    __mu = 0.7

    def forward(self, x):
        '''
        Hybrid forward with Gluon API.

        Args:
            x:      `tensor` of observed data points.
        
        Returns:
            `tensor` of inferenced feature points.
        '''
        self.batch_size = x.shape[0]

        noised_x = self.noiseable_data.noise(x)
        for i in range(len(self.encoder.fc_list)):
            x = self.encoder.fc_list[i](x)
            noised_x = self.encoder.fc_list[i](noised_x)
            if self.encoder.activation_list[i] == "identity_adjusted":
                x = x / torch.sum(torch.ones_like(x))
                noised_x = noised_x / torch.sum(torch.ones_like(noised_x))
            elif self.encoder.activation_list[i] != "identity":
                x = self.encoder.activation_list[i](x)
                noised_x = self.encoder.activation_list[i](noised_x)
            if self.encoder.dropout_forward_list[i] is not None:
                x = self.encoder.dropout_forward_list[i](x)
                noised_x = self.encoder.dropout_forward_list[i](noised_x)

            if self.encoder.hidden_batch_norm_list[i] is not None:
                x = self.encoder.hidden_batch_norm_list[i](x)
                noised_x = self.encoder.hidden_batch_norm_list[i](noised_x)

            noised_x = self.noiseable_data.noise(noised_x)
            alpha_arr, sigma_arr, mu_arr = self.forward_ladder_net(x, noised_x)
            x = torch.add(x, (alpha_arr * self.alpha))
            x = torch.add(x, (sigma_arr * self.sigma))
            x = torch.add(x, (mu_arr * self.mu))

        self.feature_points_arr = x

        if self.recoding_ld_loss is True:
            if x.requires_grad is True:
                self.alpha_encoder_loss_list.append(torch.mean(torch.square(alpha_arr)).to("cpu").detach().numpy())
                self.sigma_encoder_loss_list.append(torch.mean(torch.square(sigma_arr)).to("cpu").detach().numpy())
                self.mu_encoder_loss_list.append(torch.mean(torch.square(mu_arr)).to("cpu").detach().numpy())
            else:
                self.alpha_encoder_test_loss_list.append(torch.mean(torch.square(alpha_arr)).to("cpu").detach().numpy())
                self.sigma_encoder_test_loss_list.append(torch.mean(torch.square(sigma_arr)).to("cpu").detach().numpy())
                self.mu_encoder_test_loss_list.append(torch.mean(torch.square(mu_arr)).to("cpu").detach().numpy())

        noised_x = self.noiseable_data.noise(x)
        for i in range(len(self.decoder.fc_list)):
            x = self.decoder.fc_list[i](x)
            noised_x = self.decoder.fc_list[i](noised_x)
            if self.decoder.activation_list[i] == "identity_adjusted":
                x = x / torch.sum(torch.ones_like(x))
                noised_x = noised_x / torch.sum(torch.ones_like(noised_x))
            elif self.decoder.activation_list[i] != "identity":
                x = self.decoder.activation_list[i](x)
                noised_x = self.decoder.activation_list[i](noised_x)
            if self.decoder.dropout_forward_list[i] is not None:
                x = self.decoder.dropout_forward_list[i](x)
                noised_x = self.decoder.dropout_forward_list[i](noised_x)

            if self.decoder.hidden_batch_norm_list[i] is not None:
                x = self.decoder.hidden_batch_norm_list[i](x)
                noised_x = self.decoder.hidden_batch_norm_list[i](noised_x)

            noised_x = self.noiseable_data.noise(noised_x)
            alpha_arr, sigma_arr, mu_arr = self.forward_ladder_net(x, noised_x)

            x = torch.add(x, (alpha_arr * self.alpha))
            x = torch.add(x, (sigma_arr * self.sigma))
            x = torch.add(x, (mu_arr * self.mu))

        if self.recoding_ld_loss is True:
            if x.requires_grad is True:
                self.alpha_decoder_loss_list.append(torch.mean(torch.square(alpha_arr)).to("cpu").detach().numpy())
                self.sigma_decoder_loss_list.append(torch.mean(torch.square(sigma_arr)).to("cpu").detach().numpy())
                self.mu_decoder_loss_list.append(torch.mean(torch.square(mu_arr)).to("cpu").detach().numpy())
            else:
                self.alpha_decoder_test_loss_list.append(torch.mean(torch.square(alpha_arr)).to("cpu").detach().numpy())
                self.sigma_decoder_test_loss_list.append(torch.mean(torch.square(sigma_arr)).to("cpu").detach().numpy())
                self.mu_decoder_test_loss_list.append(torch.mean(torch.square(mu_arr)).to("cpu").detach().numpy())

        return x

    def forward_ladder_net(self, x, noised_x):
        pow_f = torch.pow

        hidden_arr = x.reshape(self.batch_size, -1)
        row_arr = torch.ones_like(hidden_arr)
        row = torch.mean(torch.sum(row_arr, axis=1))
        sigma_arr = torch.mm(
            hidden_arr,
            hidden_arr.T
        )
        sigma_arr = sigma_arr + 1e-08
        sigma_arr = torch.eye(self.batch_size, device=sigma_arr.device) - pow_f(sigma_arr, -1)
        sigma_arr = torch.mean(sigma_arr, axis=0)
        sigma_arr = torch.reshape(sigma_arr, shape=(
            self.batch_size,
            1,
        ))
        sigma_arr = torch.div(sigma_arr, row)
        mu_arr = torch.div(torch.square(x), row)
        alpha_arr = x - noised_x

        return alpha_arr, sigma_arr, mu_arr

    def set_readonly(self, value):
        ''' setter '''
        raise TypeError("This property must be read-only.")

    def get_alpha(self):
        ''' getter '''
        return self.__alpha
    
    def set_alpha(self, value):
        ''' setter '''
        self.__alpha = value
    
    alpha = property(get_alpha, set_alpha)

    def get_sigma(self):
        ''' getter '''
        return self.__sigma
    
    def set_sigma(self, value):
        ''' setter '''
        self.__sigma = value
    
    sigma = property(get_sigma, set_sigma)

    def get_mu(self):
        ''' getter '''
        return self.__mu
    
    def set_mu(self, value):
        ''' setter '''
        self.__mu = value
    
    mu = property(get_mu, set_mu)

    def get_alpha_loss_arr(self):
        '''
        getter
        
        Returns:
            Logs of alpha losses. The shape is ...
            - encoder's train losses of alpha.
            - encoder's test losses of alpha.
            - decoder's train losses of alpha.
            - decoder's test losses of alpha.
        '''
        return np.c_[
            np.array(self.__alpha_encoder_loss_list),
            np.array(self.__alpha_encoder_test_loss_list[:len(self.__alpha_encoder_loss_list)]),
            np.array(self.__alpha_decoder_loss_list),
            np.array(self.__alpha_decoder_test_loss_list[:len(self.__alpha_decoder_loss_list)]),
        ]

    def get_sigma_loss_arr(self):
        '''
        getter
        
        Returns:
            Logs of sigma losses. The shape is ...
            - encoder's train losses of sigma.
            - encoder's test losses of sigma.
            - decoder's train losses of sigma.
            - decoder's test losses of sigma.
        '''
        return np.c_[
            np.array(self.__sigma_encoder_loss_list),
            np.array(self.__sigma_encoder_test_loss_list[:len(self.__sigma_encoder_loss_list)]),
            np.array(self.__sigma_decoder_loss_list),
            np.array(self.__sigma_decoder_test_loss_list[:len(self.__sigma_decoder_loss_list)])
        ]

    def get_mu_loss_arr(self):
        '''
        getter
        
        Returns:
            Logs of mu losses. The shape is ...
            - encoder's train losses of mu.
            - encoder's test losses of mu.
            - decoder's train losses of mu.
            - decoder's test losses of mu.
        '''
        return np.c_[
            np.array(self.__mu_encoder_loss_list),
            np.array(self.__mu_encoder_test_loss_list[:len(self.__mu_encoder_loss_list)]),
            np.array(self.__mu_decoder_loss_list),
            np.array(self.__mu_decoder_test_loss_list[:len(self.__mu_decoder_loss_list)])
        ]

    alpha_loss_arr = property(get_alpha_loss_arr, set_readonly)
    sigma_loss_arr = property(get_sigma_loss_arr, set_readonly)
    mu_loss_arr = property(get_mu_loss_arr, set_readonly)

    # `list` of encoder's alpha loss.
    __alpha_encoder_loss_list = []

    def get_alpha_encoder_loss_list(self):
        ''' getter '''
        return self.__alpha_encoder_loss_list
    
    def set_alpha_encoder_loss_list(self, value):
        ''' setter '''
        self.__alpha_encoder_loss_list = value

    alpha_encoder_loss_list = property(get_alpha_encoder_loss_list, set_alpha_encoder_loss_list)

    # `list` of encoder's sigma loss.
    __sigma_encoder_loss_list = []

    def get_sigma_encoder_loss_list(self):
        ''' getter '''
        return self.__sigma_encoder_loss_list
    
    def set_sigma_encoder_loss_list(self, value):
        ''' setter '''
        self.__sigma_encoder_loss_list = value

    sigma_encoder_loss_list = property(get_sigma_encoder_loss_list, set_sigma_encoder_loss_list)

    # `list` of encoder's mu loss.
    __mu_encoder_loss_list = []

    def get_mu_encoder_loss_list(self):
        ''' getter '''
        return self.__mu_encoder_loss_list
    
    def set_mu_encoder_loss_list(self, value):
        ''' setter '''
        self.__mu_encoder_loss_list = value
    
    mu_encoder_loss_list = property(get_mu_encoder_loss_list, set_mu_encoder_loss_list)

    # `list` of decoder's alpha loss.
    __alpha_decoder_loss_list = []

    def get_alpha_decoder_loss_list(self):
        ''' getter '''
        return self.__alpha_decoder_loss_list
    
    def set_alpha_decoder_loss_list(self, value):
        ''' setter '''
        self.__alpha_decoder_loss_list = value
    
    alpha_decoder_loss_list = property(get_alpha_decoder_loss_list, set_alpha_decoder_loss_list)

    # `list` of decoder's sigma loss.
    __sigma_decoder_loss_list = []

    def get_sigma_decoder_loss_list(self):
        ''' getter '''
        return self.__sigma_decoder_loss_list
    
    def set_sigma_decoder_loss_list(self, value):
        ''' setter '''
        self.__sigma_decoder_loss_list = value
    
    sigma_decoder_loss_list = property(get_sigma_decoder_loss_list, set_sigma_decoder_loss_list)

    # `list` of decoder's mu loss.
    __mu_decoder_loss_list = []

    def get_mu_decoder_loss_list(self):
        ''' getter '''
        return self.__mu_decoder_loss_list

    def set_mu_decoder_loss_list(self, value):
        ''' setter '''
        self.__mu_decoder_loss_list = value

    mu_decoder_loss_list = property(get_mu_decoder_loss_list, set_mu_decoder_loss_list)

    # `list` of encoder's alpha test loss.
    __alpha_encoder_test_loss_list = []

    def get_alpha_encoder_test_loss_list(self):
        ''' getter '''
        return self.__alpha_encoder_test_loss_list

    def set_alpha_encoder_test_loss_list(self, value):
        ''' setter '''
        self.__alpha_encoder_test_loss_list = value

    alpha_encoder_test_loss_list = property(get_alpha_encoder_test_loss_list, set_alpha_encoder_test_loss_list)

    # `list` of encoder's sigma test loss.
    __sigma_encoder_test_loss_list = []

    def get_sigma_encoder_test_loss_list(self):
        ''' getter '''
        return self.__sigma_encoder_test_loss_list
    
    def set_sigma_encoder_test_loss_list(self, value):
        ''' setter '''
        self.__sigma_encoder_test_loss_list = value

    sigma_encoder_test_loss_list = property(get_sigma_encoder_test_loss_list, set_sigma_encoder_test_loss_list)

    # `list` of encoder's mu test loss.
    __mu_encoder_test_loss_list = []

    def get_mu_encoder_test_loss_list(self):
        ''' getter '''
        return self.__mu_encoder_test_loss_list

    def set_mu_encoder_test_loss_list(self, value):
        ''' setter '''
        self.__mu_encoder_test_loss_list = value

    mu_encoder_test_loss_list = property(get_mu_encoder_test_loss_list, set_mu_encoder_test_loss_list)

    # `list` of decoder's alpha test loss.
    __alpha_decoder_test_loss_list = []

    def get_alpha_decoder_test_loss_list(self):
        ''' getter '''
        return self.__alpha_decoder_test_loss_list

    def set_alpha_decoder_test_loss_list(self, value):
        ''' setter '''
        self.__alpha_decoder_test_loss_list = value

    alpha_decoder_test_loss_list = property(get_alpha_decoder_test_loss_list, set_alpha_decoder_test_loss_list)

    # `list` of decoder's sigma test loss.
    __sigma_decoder_test_loss_list = []

    def get_sigma_decoder_test_loss_list(self):
        ''' getter '''
        return self.__sigma_decoder_test_loss_list

    def set_sigma_decoder_test_loss_list(self, value):
        ''' setter '''
        self.__sigma_decoder_test_loss_list = value

    sigma_decoder_test_loss_list = property(get_sigma_decoder_test_loss_list, set_sigma_decoder_test_loss_list)

    # `list` of decoder's mu test loss.
    __mu_decoder_test_loss_list = []

    def get_mu_decoder_test_loss_list(self):
        ''' getter '''
        return self.__mu_decoder_test_loss_list
    
    def set_mu_decoder_test_loss_list(self, value):
        ''' setter '''
        self.__mu_decoder_test_loss_list = value
    
    mu_decoder_test_loss_list = property(get_mu_decoder_test_loss_list, set_mu_decoder_test_loss_list)
