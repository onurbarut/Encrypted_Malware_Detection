import os
import time

from models.ANN import ANN
from models.ModelLoader import ModelLoader
from openvino.inference_engine import IECore, IENetwork, IEPlugin

from utils.helper import collect_statistics, convertToDefault


class vino_ANN:
    def __init__(self, data, labels):
        self.data = data
        self.labels = labels
        self.X_train = None
        self.X_test = None
        self.y_train = None
        self.y_test = None
        self.prep_data()

    def prep_data(self):
        """
        Sets X_train, X_test, y_train, y_test, and  variables for training / testing.
        Run this method to reset values
        """
        # Create ann model to pull data values from
        ann_data = ANN(self.data, self.labels)

        # Clone data from ann model
        self.X_train = ann_data.X_train
        self.X_test = ann_data.X_test
        self.y_train = ann_data.y_train
        self.y_test = ann_data.y_test

    def train_model(self,
                    work_dir='models/saved/'):
        """
        Loads a ANN model to be used for OpenVINO
        """
        len = self.X_test.shape

        # Begin train timing
        startTime = time.time()

        # Convert ANN model to binary .pb file and save
        ml = ModelLoader('model_ann', None)
        loaded_model = ml.load_keras_model()
        ml.save_keras_as_vino()

        # Run OpenVINO's Model Optimizer TensorFlow script (Have script [mo_tf.py] in main directory with DAAL scripts)
        generateCommand = "mo_tf.py --input_model %svino_ann.pb --input_shape [%d,%d] --output_dir %s" % (
            work_dir, len[0], len[1], work_dir)

        print(generateCommand)
        os.system(generateCommand)

        # End train timing
        endTime = time.time()

        print("Preparation (VINO Artificial Neural Network) elapsed in %.3f seconds" % (endTime - startTime))

        ml = ModelLoader('vino_ann', None)
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
        res = execNet.infer(inputs={input_blob: self.X_test})

        # Get prediction results
        res = res['dense_3/Sigmoid']

        # End testing time
        endTime = time.time()

        # Collect statistics
        test_tpr, test_far, test_accu, test_report = collect_statistics(self.y_test.flatten(), convertToDefault(res))

        print("Test (VINO Artificial Neural Network) elapsed in %.3f seconds" % (endTime - startTime))
        print("--- Testing Results  ---")
        print("Test accuracy: ", test_accu)
        print("TPR: ", test_tpr)
        print("FAR: ", test_far)
        print(test_report)
        print("------------------------")

        return test_accu, test_tpr, test_far, test_report
