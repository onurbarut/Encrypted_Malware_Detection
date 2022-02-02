import time

import numpy as np
from _daal4py import svm_training, kernel_function_linear, svm_prediction

from utils.helper import collect_statistics
from models.ModelLoader import ModelLoader


class daal_SVM:
    def __init__(self, data, labels):
        self.data = data
        self.labels = labels
        self.X_train = None
        self.X_test = None
        self.y_train = None
        self.y_test = None

        # Prep data for use in training / testing
        self.prep_data()

    def prep_data(self):
        """
        Sets X_train, X_test, y_train, and y_test variables for training / testing.
        Run this method to reset values
        """
        # make 0 values -1
        self.labels = [-1 if i == 0 else 1 for i in self.labels]

        # Reshape labels
        self.labels = np.array(self.labels).reshape((len(self.labels), 1))

        # Setup train / test data
        dataLen = len(self.data)
        mark = 0.8
        upperBound = int(dataLen * mark)

        self.X_train = self.data[0:upperBound]
        self.y_train = self.labels[0:upperBound]
        self.X_test = self.data[upperBound:]
        self.y_test = self.labels[upperBound:]

    def train_model(self,
                    save_model=True):
        # Begin train timing
        startTime = time.time()

        # Support Vector Machine
        kern = kernel_function_linear(method='defaultDense')
        trainAlg = svm_training(nClasses=2, C=1e+6, maxIterations=1e+7, cacheSize=2000, kernel=kern,
                                accuracyThreshold=1e-3, doShrinking=True)
        # Train model
        trainResult = trainAlg.compute(self.X_train, self.y_train)

        # Create prediction classes
        predictAlgTrain = svm_prediction(nClasses=2, kernel=kern)
        predictAlgTest = svm_prediction(nClasses=2, kernel=kern)

        # Make train and test predictions
        predictResultTrain = predictAlgTrain.compute(self.X_train, trainResult.model)
        predictResultTest = predictAlgTest.compute(self.X_test, trainResult.model)

        # End train timing
        endTime = time.time()

        # Flatten y
        trainLabel = self.y_train.flatten()
        testLabel = self.y_test.flatten()

        # Compare train predictions
        predictionsTrain = predictResultTrain.prediction.flatten()

        correctTrain = np.sum(np.logical_or(np.logical_and(trainLabel > 0, predictionsTrain > 0),
                                            np.logical_and(trainLabel < 0, predictionsTrain < 0)))
        trainAccu = float(correctTrain) / len(trainLabel) * 100

        # Compare test predictions
        predictionsTest = predictResultTest.prediction.flatten()

        correctTest = np.sum(np.logical_or(np.logical_and(testLabel > 0, predictionsTest > 0),
                                           np.logical_and(testLabel < 0, predictionsTest < 0)))
        testAccu = float(correctTest) / len(testLabel) * 100

        # Collect statistics
        train_tpr, train_far, train_accu, train_report = collect_statistics(trainLabel, predictionsTrain)
        test_tpr, test_far, test_accu, test_report = collect_statistics(testLabel, predictionsTest)

        print("Training and test (Support Vector Machine) elapsed in %.3f seconds" % (endTime - startTime))
        print("--- Training Results ---")
        print("Train accuracy: ", trainAccu)
        print("TPR: ", train_tpr)
        print("FAR: ", train_far)
        print("--- Testing Results  ---")
        print("Test accuracy: ", testAccu)
        print("TPR: ", test_tpr)
        print("FAR: ", test_far)
        print(test_report)
        print("------------------------")

        if save_model:
            ml = ModelLoader('daal_svm', trainResult.model)
            ml.save_sk_daal_model()

        return test_accu, test_tpr, test_far, test_report

    def load_saved_model(self, loaded_model):
        # Flatten y
        testLabel = self.y_test.flatten()

        # Begin test timing
        startTime = time.time()

        # create prediction class
        kern = kernel_function_linear(method='defaultDense')
        predictAlg = svm_prediction(nClasses=2, kernel=kern)

        # make predictions
        predictResultTest = predictAlg.compute(self.X_test, loaded_model)

        # End test timing
        endTime = time.time()

        # assess accuracy
        predictions = predictResultTest.prediction.flatten()
        correctTest = np.sum(
            np.logical_or(np.logical_and(testLabel > 0, predictions > 0), np.logical_and(testLabel < 0, predictions < 0)))
        testAccu = float(correctTest) / len(testLabel) * 100

        # Collect statistics
        test_tpr, test_far, test_accu, test_report = collect_statistics(testLabel, predictResultTest.prediction.flatten())

        print("Test (Support Vector Machine) elapsed in %.3f seconds" % (endTime - startTime))
        print("--- Testing Results  ---")
        print("Test accuracy: ", testAccu)
        print("TPR: ", test_tpr)
        print("FAR: ", test_far)
        print(test_report)
        print("------------------------")

        return test_accu, test_tpr, test_far, test_report
