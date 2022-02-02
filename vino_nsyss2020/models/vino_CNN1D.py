import os
import time

import numpy as np

from models.CNN1D import CNN1D
from models.ModelLoader import ModelLoader
from openvino.inference_engine import IECore, IENetwork

from utils.helper import convertToOneHot, collect_statistics, convertToDefault
from utils.helper2 import one_hot


class vino_CNN1D:
    def __init__(self, data, labels):
        self.data = data
        self.labels = labels
        self.X_train_1D = None
        self.X_test_1D = None
        self.y_train = None
        self.y_test = None

        self.top_class_names = None
        self.n_classes_top = None

        self.prep_data()

        # Must run `pycharm-community` command from terminal to initialize openvino in pycharm

    def prep_data(self):
        """
        Sets X_train_1D, X_test_1D, y_train, y_test, and  variables for training / testing.
        Run this method to reset values
        """

        # Create cnn1d model to pull data from
        cnn1d_data = CNN1D(self.data, self.labels)

        # Clone data from cnn1d model
        self.X_train_1D = cnn1d_data.X_train_1D
        self.X_test_1D = cnn1d_data.X_test_1D
        self.y_train = cnn1d_data.y_train
        self.y_test = cnn1d_data.y_test

        self.top_class_names = cnn1d_data.top_class_names
        self.n_classes_top = cnn1d_data.n_classes_top

    def train_model(self,
                    work_dir='models/saved/'):
        """
        Loads a CNN1D model to be used for OpenVINO
        """
        len = self.X_test_1D.shape

        # Start train timing
        startTime = time.time()

        # Convert ANN model to binary .pb file and save
        ml = ModelLoader('model_cnn1d', None)
        loaded_model = ml.load_keras_model()
        ml.save_keras_as_vino()

        # Run OpenVINO's Model Optimizer TensorFlow script (Have script [mo_tf.py] in main directory with DAAL scripts)
        generateCommand = "mo_tf.py --input_model %svino_cnn1d.pb --input_shape [%d,%d,%d] --output_dir %s" % (
        work_dir, len[0], len[1], len[2], work_dir)

        print(generateCommand)
        os.system(generateCommand)

        # End train timing
        endTime = time.time()

        print("Preparation (VINO Convolutional 2D Neural Network) elapsed in %.3f seconds" % (endTime - startTime))

        ml = ModelLoader('vino_cnn1d', None)
        net, execNet = ml.load_vino_model()
        return self.load_saved_model(net, execNet)

    def load_saved_model(self, net, execNet,
                         work_dir='models/saved/'):
        # Begin testing time
        startTime = time.time()

        # Get input and outputs of model
        input_blob = next(iter(net.inputs))
        out_blob = next(iter(net.outputs))

        # Input data into model for predicting
        res = execNet.infer(inputs={input_blob: self.X_test_1D})

        # Get prediction results
        res = res['top_level_output/Softmax']

        # End testing time
        endTime = time.time()

        # Collect statistics
        test_tpr, test_far, test_accu, test_report = collect_statistics(self.y_test, convertToDefault(res))

        print("Test (VINO Convolutional 1D Neural Network) elapsed in %.3f seconds" % (endTime - startTime))
        print("--- Testing Results  ---")
        print("Test accuracy: ", test_accu)
        print("TPR: ", test_tpr)
        print("FAR: ", test_far)
        print(test_report)
        print("------------------------")

        return test_accu, test_tpr, test_far, test_report
