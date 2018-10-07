# -*- coding: utf-8 -*-
from pydbm.cnn.convolutional_neural_network import ConvolutionalNeuralNetwork
from pydbm.cnn.convolutionalneuralnetwork.residual_learning import ResidualLearning
from pydbm.cnn.convolutionalneuralnetwork.convolutional_auto_encoder import ConvolutionalAutoEncoder
from pydbm.cnn.spatio_temporal_auto_encoder import SpatioTemporalAutoEncoder
from pydbm.cnn.layerablecnn.convolution_layer import ConvolutionLayer as ConvolutionLayer1
from pydbm.cnn.layerablecnn.convolution_layer import ConvolutionLayer as ConvolutionLayer2
from pydbm.cnn.featuregenerator.image_generator import ImageGenerator

from pydbm.activation.relu_function import ReLuFunction
from pydbm.activation.tanh_function import TanhFunction
from pydbm.activation.logistic_function import LogisticFunction
from pydbm.loss.mean_squared_error import MeanSquaredError
from pydbm.optimization.optparams.adam import Adam

from pydbm.synapse.cnn_graph import CNNGraph as ConvGraph1
from pydbm.synapse.cnn_graph import CNNGraph as ConvGraph2

from pydbm.rnn.lstm_model import LSTMModel as Encoder
from pydbm.rnn.lstm_model import LSTMModel as Decoder
from pydbm.optimization.optparams.adam import Adam as EncoderAdam
from pydbm.optimization.optparams.adam import Adam as DecoderAdam
from pydbm.synapse.recurrenttemporalgraph.lstm_graph import LSTMGraph as EncoderGraph
from pydbm.synapse.recurrenttemporalgraph.lstm_graph import LSTMGraph as DecoderGraph

from pydbm.verification.verificate_function_approximation import VerificateFunctionApproximation
from logging import getLogger, StreamHandler, NullHandler, DEBUG, ERROR
from PIL import Image


def Main(params_dict):

    logger = getLogger("pydbm")
    handler = StreamHandler()
    if params_dict["debug_mode"] is True:
        handler.setLevel(DEBUG)
        logger.setLevel(DEBUG)
    else:
        handler.setLevel(ERROR)
        logger.setLevel(ERROR)

    logger.addHandler(handler)

    epochs = params_dict["epochs"]
    batch_size = params_dict["batch_size"]
    seq_len = params_dict["seq_len"]
    channel = params_dict["channel"]
    height = params_dict["height"]
    width = params_dict["width"]
    scale = params_dict["scale"]
    training_image_dir = params_dict["training_image_dir"]
    test_image_dir = params_dict["test_image_dir"]

    enc_dim = batch_size * height * width
    dec_dim = batch_size * height * width

    feature_generator = ImageGenerator(
        epochs=epochs,
        batch_size=batch_size,
        training_image_dir=training_image_dir,
        test_image_dir=test_image_dir,
        seq_len=seq_len,
        gray_scale_flag=True,
        wh_size_tuple=(height, width),
        norm_mode="z_score"
    )

    # Init.
    encoder_graph = EncoderGraph()

    # Activation function in LSTM.
    encoder_graph.observed_activating_function = TanhFunction()
    encoder_graph.input_gate_activating_function = LogisticFunction()
    encoder_graph.forget_gate_activating_function = LogisticFunction()
    encoder_graph.output_gate_activating_function = LogisticFunction()
    encoder_graph.hidden_activating_function = TanhFunction()
    encoder_graph.output_activating_function = LogisticFunction()

    # Initialization strategy.
    # This method initialize each weight matrices and biases in Gaussian distribution: `np.random.normal(size=hoge) * 0.01`.
    encoder_graph.create_rnn_cells(
        input_neuron_count=enc_dim,
        hidden_neuron_count=200,
        output_neuron_count=enc_dim
    )

    # Optimizer for Encoder.
    encoder_opt_params = EncoderAdam()
    encoder_opt_params.weight_limit = 0.5
    encoder_opt_params.dropout_rate = 0.5

    encoder = Encoder(
        # Delegate `graph` to `LSTMModel`.
        graph=encoder_graph,
        # The number of epochs in mini-batch training.
        epochs=epochs,
        # The batch size.
        batch_size=batch_size,
        # Learning rate.
        learning_rate=1e-05,
        # Attenuate the `learning_rate` by a factor of this value every `attenuate_epoch`.
        learning_attenuate_rate=0.1,
        # Attenuate the `learning_rate` by a factor of `learning_attenuate_rate` every `attenuate_epoch`.
        attenuate_epoch=50,
        # Refereed maxinum step `t` in BPTT. If `0`, this class referes all past data in BPTT.
        bptt_tau=8,
        # Size of Test data set. If this value is `0`, the validation will not be executed.
        test_size_rate=0.3,
        # Loss function.
        computable_loss=MeanSquaredError(),
        # Optimizer.
        opt_params=encoder_opt_params,
        # Verification function.
        verificatable_result=VerificateFunctionApproximation(),
        # Tolerance for the optimization.
        # When the loss or score is not improving by at least tol 
        # for two consecutive iterations, convergence is considered 
        # to be reached and training stops.
        tol=0.0
    )


    # Init.
    decoder_graph = DecoderGraph()

    # Activation function in LSTM.
    decoder_graph.observed_activating_function = TanhFunction()
    decoder_graph.input_gate_activating_function = LogisticFunction()
    decoder_graph.forget_gate_activating_function = LogisticFunction()
    decoder_graph.output_gate_activating_function = LogisticFunction()
    decoder_graph.hidden_activating_function = TanhFunction()
    decoder_graph.output_activating_function = LogisticFunction()

    # Initialization strategy.
    # This method initialize each weight matrices and biases in Gaussian distribution: `np.random.normal(size=hoge) * 0.01`.
    decoder_graph.create_rnn_cells(
        input_neuron_count=200,
        hidden_neuron_count=dec_dim,
        output_neuron_count=200
    )

    # Optimizer for Decoder.
    decoder_opt_params = DecoderAdam()
    decoder_opt_params.weight_limit = 0.5
    decoder_opt_params.dropout_rate = 0.5

    decoder = Decoder(
        # Delegate `graph` to `LSTMModel`.
        graph=decoder_graph,
        # The number of epochs in mini-batch training.
        epochs=epochs,
        # The batch size.
        batch_size=batch_size,
        # Learning rate.
        learning_rate=1e-05,
        # Attenuate the `learning_rate` by a factor of this value every `attenuate_epoch`.
        learning_attenuate_rate=0.1,
        # Attenuate the `learning_rate` by a factor of `learning_attenuate_rate` every `attenuate_epoch`.
        attenuate_epoch=50,
        # Refereed maxinum step `t` in BPTT. If `0`, this class referes all past data in BPTT.
        bptt_tau=8,
        # Size of Test data set. If this value is `0`, the validation will not be executed.
        test_size_rate=0.3,
        # Loss function.
        computable_loss=MeanSquaredError(),
        # Optimizer.
        opt_params=decoder_opt_params,
        # Verification function.
        verificatable_result=VerificateFunctionApproximation(),
        # Tolerance for the optimization.
        # When the loss or score is not improving by at least tol 
        # for two consecutive iterations, convergence is considered 
        # to be reached and training stops.
        tol=0.0
    )


    conv1 = ConvolutionLayer1(
        ConvGraph1(
            activation_function=TanhFunction(),
            filter_num=batch_size,
            channel=channel,
            kernel_size=3,
            scale=scale,
            stride=1,
            pad=1
        )
    )

    conv2 = ConvolutionLayer2(
        ConvGraph2(
            activation_function=TanhFunction(),
            filter_num=batch_size,
            channel=batch_size,
            kernel_size=3,
            scale=scale,
            stride=1,
            pad=1
        )
    )

    cnn = SpatioTemporalAutoEncoder(
        layerable_cnn_list=[
            conv1, 
            conv2
        ],
        encoder=encoder,
        decoder=decoder,
        epochs=epochs,
        batch_size=batch_size,
        learning_rate=1e-05,
        learning_attenuate_rate=0.1,
        attenuate_epoch=25,
        computable_loss=MeanSquaredError(),
        opt_params=Adam(),
        verificatable_result=VerificateFunctionApproximation(),
        test_size_rate=0.3,
        tol=1e-15,
        save_flag=False
    )

    cnn.learn_generated(feature_generator)

    test_len = 0
    test_limit = 1

    test_arr_list = []
    rec_arr_list = []
    for batch_observed_arr, batch_target_arr, test_batch_observed_arr, test_batch_target_arr in feature_generator.generate():
        test_len += 1
        result_arr = cnn.inference(test_batch_observed_arr)
        for batch in range(test_batch_target_arr.shape[0]):
            for seq in range(test_batch_target_arr[batch].shape[0]):
                arr = test_batch_target_arr[batch][seq][0]
                arr = (arr - arr.min()) / (arr.max() - arr.min())
                arr *= 255
                img = Image.fromarray(np.uint8(arr))
                img.save("result/" + str(i) + "_" + str(seq) + "_observed.png")
            for seq in range(result_arr[batch].shape[0]):
                arr = result_arr[batch][seq][0]
                arr = (arr - arr.min()) / (arr.max() - arr.min())
                arr *= 255
                img = Image.fromarray(np.uint8(arr))
                img.save("result/" + str(i) + "_" + str(seq) + "_reconsturcted.png")

        if test_len >= test_limit:
            break

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(
        description='Spatio-Temporal Auto-Encoder.'
    )
    parser.add_argument(
        '-ep',
        '--epochs',
        type=int,
        default=100,
        help='Epochs in Mini-batch training.'
    )

    parser.add_argument(
        '-bs',
        '--batch_size',
        type=int,
        default=20,
        help='Batch size in Mini-batch training.'
    )
    parser.add_argument(
        '-sl',
        '--seq_len',
        type=int,
        default=10,
        help='The length of sequence.'
    )

    parser.add_argument(
        '-c',
        '--channel',
        type=int,
        default=1,
        help='Channel of images.'
    )

    parser.add_argument(
        '-ih',
        '--height',
        type=int,
        default=100,
        help='Image height.'
    )

    parser.add_argument(
        '-iw',
        '--width',
        type=int,
        default=100,
        help='Image width.'
    )
    parser.add_argument(
        '-s',
        '--scale',
        type=float,
        default=20,
        help='Scale of initial weight matirx.'
    )

    parser.add_argument(
        '-traindir',
        '--training_image_dir',
        type=str,
        default="img/training/",
        help='Dir path of training image dataset.'
    )
    parser.add_argument(
        '-testdir',
        '--test_image_dir',
        type=str,
        default="img/test/",
        help='Dir path of test image dataset.'
    )
    parser.add_argument(
        '-d',
        '--debug_mode',
        type=bool,
        default=False,
        help="Debug mode or not."
    )

    args = parser.parse_args()
    params = vars(args)
    Main(params)
