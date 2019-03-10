# -*- coding: utf-8 -*-
from logging import getLogger
import numpy as np
cimport numpy as np
from pydbm.rnn.encoder_decoder_controller import EncoderDecoderController
from pydbm.rnn.lstm_model import LSTMModel
from pydbm.rnn.lstmmodel.attention_lstm_model import AttentionLSTMModel
from pydbm.cnn.feature_generator import FeatureGenerator
from pydbm.loss.interface.computable_loss import ComputableLoss
from pydbm.verification.interface.verificatable_result import VerificatableResult
ctypedef np.float64_t DOUBLE_t


class AttentionEncoderDecoder(EncoderDecoderController):
    '''
    Encoder/Decoder based on LSTM networks with attention model.
    
    This library provides Encoder/Decoder based on LSTM, 
    which is a reconstruction model and makes it possible to extract 
    series features embedded in deeper layers. The LSTM encoder learns 
    a fixed length vector of time-series observed data points and the 
    LSTM decoder uses this representation to reconstruct the time-series 
    using the current hidden state and the value inferenced at the previous time-step.
    
    One interesting application example is the Encoder/Decoder for 
    Anomaly Detection (EncDec-AD) paradigm (Malhotra, P., et al. 2016).
    This reconstruction model learns to reconstruct normal time-series behavior, 
    and thereafter uses reconstruction error to detect anomalies. 
    Malhotra, P., et al. (2016) showed that EncDec-AD paradigm is robust 
    and can detect anomalies from predictable, unpredictable, periodic, aperiodic, 
    and quasi-periodic time-series. Further, they showed that the paradigm is able 
    to detect anomalies from short time-series (length as small as 30) as well as 
    long time-series (length as large as 500).

    References:
        - https://github.com/chimera0/accel-brain-code/blob/master/Deep-Learning-by-means-of-Design-Pattern/demo/demo_sine_wave_prediction_by_LSTM_encoder_decoder.ipynb
        - https://github.com/chimera0/accel-brain-code/blob/master/Deep-Learning-by-means-of-Design-Pattern/demo/demo_anomaly_detection_by_enc_dec_ad.ipynb
        - Cho, K., Van Merriënboer, B., Gulcehre, C., Bahdanau, D., Bougares, F., Schwenk, H., & Bengio, Y. (2014). Learning phrase representations using RNN encoder-decoder for statistical machine translation. arXiv preprint arXiv:1406.1078.
        - Malhotra, P., Ramakrishnan, A., Anand, G., Vig, L., Agarwal, P., & Shroff, G. (2016). LSTM-based encoder-decoder for multi-sensor anomaly detection. arXiv preprint arXiv:1607.00148.
    '''
    
    def __init__(
        self,
        encoder,
        decoder,
        computable_loss,
        verificatable_result,
        int epochs=100,
        int batch_size=100,
        double learning_rate=1e-05,
        double learning_attenuate_rate=0.1,
        int attenuate_epoch=50,
        double test_size_rate=0.3,
        dropout_rate=0.5,
        tol=1e-04,
        tld=100.0
    ):
        '''
        Init.
        
        Args:
            encoder:                        is-a `LSTMModel`.
            decoder:                        is-a `AttentionLSTMModel`.
            computable_loss:                is-a `ComputableLoss`.
            verificatable_result:           is-a `VerificatableResult`.
            epochs:                         Epochs of mini-batch.
            bath_size:                      Batch size of mini-batch.
            learning_rate:                  Learning rate.
            learning_attenuate_rate:        Attenuate the `learning_rate` by a factor of this value every `attenuate_epoch`.
            attenuate_epoch:                Attenuate the `learning_rate` by a factor of `learning_attenuate_rate` every `attenuate_epoch`.
                                            Additionally, in relation to regularization,
                                            this class constrains weight matrixes every `attenuate_epoch`.

            test_size_rate:                 Size of Test data set. If this value is `0`, the validation will not be executed.
            dropout_rate:                   The propability of dropout.
            tol:                            Tolerance for the optimization.
                                            When the loss or score is not improving by at least tol 
                                            for two consecutive iterations, convergence is considered 
                                            to be reached and training stops.

            tld:                            Tolerance for deviation of loss.

        '''
        if isinstance(encoder, LSTMModel) is False:
            raise TypeError()
        if isinstance(decoder, AttentionLSTMModel) is False:
            raise TypeError()
        if isinstance(computable_loss, ComputableLoss) is False:
            raise TypeError()
        if isinstance(verificatable_result, VerificatableResult) is False:
            raise TypeError()

        super().__init__(
            encoder=encoder,
            decoder=decoder,
            computable_loss=computable_loss,
            verificatable_result=verificatable_result,
            epochs=epochs,
            batch_size=batch_size,
            learning_rate=learning_rate,
            learning_attenuate_rate=learning_attenuate_rate,
            attenuate_epoch=attenuate_epoch,
            test_size_rate=test_size_rate,
            dropout_rate=dropout_rate,
            tol=tol,
            tld=tld
        )

