import os

import tensorflow as tf

from utils.ModelLoader import ModelLoader
from openvino.inference_engine import IECore, IENetwork

from utils.helper import convertToDefault

class vino_ANN:
    def __init__(self, input_shape, save_dir, load_dir):
        self.input_shape = input_shape
        self.save_dir = save_dir
        self.load_dir = load_dir

    def train(self,
                    work_dir='models/saved/'):
        """
        Loads a ANN model to be used for OpenVINO
        """

        # Convert ANN model to binary .pb file and save
        ml = ModelLoader("model", None)
        loaded_model = ml.load_keras_model(load_dir=self.load_dir)
        ml.save_keras_as_vino(save_dir=self.save_dir)

        # Run OpenVINO's Model Optimizer TensorFlow script (Have script [mo_tf.py] in main directory with DAAL scripts)
        generateCommand = "mo_tf.py --input_model %smodel.pb --input_shape [%d,%d] --output_dir %s" % (
            self.save_dir,
            self.input_shape[0], self.input_shape[1],
            self.save_dir)

        # Run vino model creation command
        print(generateCommand)
        os.system(generateCommand)

    def classify(self, data):
        # Load vino model
        ml = ModelLoader('model', None)
        net, execNet = ml.load_vino_model(load_dir=self.save_dir)

        # Get input and outputs of model
        input_blob = next(iter(net.inputs))
        out_blob = next(iter(net.outputs))

        # Reshape data for input
        data = data.reshape(data.shape[0], data.shape[1],)

        # Input data into model for predicting
        res = execNet.infer(inputs={input_blob: data})

        # Get prediction results
        res = res[list(res.keys())[0]]

        return convertToDefault(res)


class vino_CNN_1D:
    def __init__(self, input_shape, save_dir, load_dir):
        self.input_shape = input_shape
        self.save_dir = save_dir
        self.load_dir = load_dir

    def train(self,
                    work_dir='models/saved/'):
        """
        Loads a CNN1D model to be used for OpenVINO
        """

        # Convert ANN model to binary .pb file and save
        ml = ModelLoader("model", None)
        loaded_model = ml.load_keras_model(load_dir=self.load_dir)
        ml.save_keras_as_vino(save_dir=self.save_dir)

        # Run OpenVINO's Model Optimizer TensorFlow script (Have script [mo_tf.py] in main directory with DAAL scripts)
        generateCommand = "mo_tf.py --input_model %smodel.pb --input_shape [%d,%d,%d] --output_dir %s" % (
            self.save_dir,
            self.input_shape[0], self.input_shape[1], self.input_shape[2],
            self.save_dir)

        # Run vino model creation command
        print(generateCommand)
        os.system(generateCommand)

    def classify(self, data):
        # Load vino model
        ml = ModelLoader('model', None)
        net, execNet = ml.load_vino_model(load_dir=self.save_dir)

        # Get input and outputs of model
        input_blob = next(iter(net.inputs))
        out_blob = next(iter(net.outputs))

        # Reshape data for input
        data = data.reshape(data.shape[0], data.shape[1], 1)

        # Input data into model for predicting
        res = execNet.infer(inputs={input_blob: data})

        # Get prediction results
        res = res[list(res.keys())[0]]

        return convertToDefault(res)


class vino_CNN_2D:
    def __init__(self, input_shape, save_dir, load_dir):
        self.input_shape = input_shape
        self.save_dir = save_dir
        self.load_dir = load_dir

    def train(self):
        """
        Loads an CNN2D model to be used for OpenVINO
        """

        # Convert ANN model to binary .pb file and save
        ml = ModelLoader('model', None)
        loaded_model = ml.load_keras_model(self.load_dir)
        ml.save_keras_as_vino(self.save_dir)

        # Run OpenVINO's Model Optimizer TensorFlow script (Have script [mo_tf.py] in main directory with DAAL scripts)
        generateCommand = "mo_tf.py --input_model %smodel.pb --input_shape [%d,%d,%d,%d] --output_dir %s" % (
            self.save_dir,
            self.input_shape[0], self.input_shape[1], self.input_shape[2], self.input_shape[3],
            self.save_dir)
        # 22880, 5, 10, 1

        print(generateCommand)
        os.system(generateCommand)

    def classify(self, data):
        # Load vino model
        ml = ModelLoader('model', None)
        net, execNet = ml.load_vino_model(load_dir=self.save_dir)

        # Get input and outputs of model
        input_blob = next(iter(net.inputs))
        out_blob = next(iter(net.outputs))

        # Transform data to pass in
        data = data.reshape(self.input_shape[0], self.input_shape[3], self.input_shape[1], self.input_shape[2])
        print(data.shape)

        # Input data into model for predicting
        res = execNet.infer(inputs={input_blob: data})

        # Get prediction results
        res = res[list(res.keys())[0]]

        return convertToDefault(res)


class vino_LSTM:
    def __init__(self, input_shape, save_dir, load_dir):
        self.input_shape = input_shape
        self.save_dir = save_dir
        self.load_dir = load_dir

    def train(self):
        """
        Loads a ANN model to be used for OpenVINO
        """

        config = tf.ConfigProto()
        config.gpu_options.allow_growth = True
        session = tf.Session(config=config)


        # Convert ANN model to binary .pb file and save
        ml = ModelLoader('model', None)
        loaded_model = ml.load_keras_model(self.load_dir)
        ml.save_keras_as_vino(self.save_dir)

        # Run OpenVINO's Model Optimizer TensorFlow script (Have script [mo_tf.py] in main directory with DAAL scripts)
        generateCommand = "mo_tf.py --input_model %smodel.pb --input_shape [%d,%d,%d] --output_dir %s" % (
            self.save_dir, self.input_shape[0], self.input_shape[1], self.input_shape[2], self.save_dir)

        print(generateCommand)
        os.system(generateCommand)

    def classify(self, data):
        # Load vino model
        ml = ModelLoader('model', None)
        net, execNet = ml.load_vino_model(load_dir=self.save_dir)

        # Get input and outputs of model
        input_blob = next(iter(net.inputs))
        out_blob = next(iter(net.outputs))

        # Input data into model for predicting
        res = execNet.infer(inputs={input_blob: data})

        # Get prediction results
        res = res[list(res.keys())[0]]

        return convertToDefault(res)


class vino_CNN_LSTM:
    def __init__(self, input_shape, save_dir, load_dir):
        self.input_shape = input_shape
        self.save_dir = save_dir
        self.load_dir = load_dir

    def train(self):
        """
        Loads a ANN model to be used for OpenVINO
        """

        config = tf.ConfigProto()
        config.gpu_options.allow_growth = True
        session = tf.Session(config=config)


        # Convert ANN model to binary .pb file and save
        ml = ModelLoader('model', None)
        loaded_model = ml.load_keras_model(self.load_dir)
        ml.save_keras_as_vino(self.save_dir)

        # Run OpenVINO's Model Optimizer TensorFlow script (Have script [mo_tf.py] in main directory with DAAL scripts)
        generateCommand = "mo_tf.py --input_model %smodel.pb --input_shape [%d,%d,%d] --output_dir %s" % (
            self.save_dir, self.input_shape[0], self.input_shape[1], self.input_shape[2], self.save_dir)

        print(generateCommand)
        os.system(generateCommand)

    def classify(self, data):
        # Load vino model
        ml = ModelLoader('model', None)
        net, execNet = ml.load_vino_model(load_dir=self.save_dir)

        # Get input and outputs of model
        input_blob = next(iter(net.inputs))
        out_blob = next(iter(net.outputs))

        data = data.reshape(data.shape[0], data.shape[1], 1)

        # Input data into model for predicting
        res = execNet.infer(inputs={input_blob: data})

        # Get prediction results
        res = res[list(res.keys())[0]]

        return convertToDefault(res)
