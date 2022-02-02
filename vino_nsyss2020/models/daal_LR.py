import time
import numpy as np

from _daal4py import logistic_regression_training, logistic_regression_prediction

from utils.helper import collect_statistics
from models.ModelLoader import ModelLoader


class daal_LR:
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
        # Reshape labels
        self.data = np.array(self.data)
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

        nClasses = 2

        # begin train timing
        startTime = time.time()

        # Create Logistic Regression Classifier
        trainAlg = logistic_regression_training(nClasses=nClasses, interceptFlag=True)

        # Train model
        trainResult = trainAlg.compute(self.X_train,
                                       self.y_train)
        # Create prediction classes 0.
        predictAlgTrain = logistic_regression_prediction(nClasses=nClasses)
        predictAlgTest = logistic_regression_prediction(nClasses=nClasses)
        # Make train and test predictions
        predictResultTrain = predictAlgTrain.compute(self.X_train, trainResult.model)
        predictResultTest = predictAlgTest.compute(self.X_test, trainResult.model)

        # End train timing
        endTime = time.time()

        # Flatten y values
        trainLabel = self.y_train.flatten()
        testLabel = self.y_test.flatten()

        # Collect statistics
        train_tpr, train_far, train_accu, train_report = collect_statistics(trainLabel, predictResultTrain.prediction.flatten())
        test_tpr, test_far, test_accu, test_report = collect_statistics(testLabel, predictResultTest.prediction.flatten())

        print("Training and test (Logistic Regression) elapsed in %.3f seconds" % (endTime - startTime))
        print("--- Training Results ---")
        print("Train accuracy: ", train_accu)
        print("TPR: ", train_tpr)
        print("FAR: ", train_far)
        print("--- Testing Results  ---")
        print("Test accuracy: ", test_accu)
        print("TPR: ", test_tpr)
        print("FAR: ", test_far)
        print(test_report)
        print("------------------------")

        if save_model:
            ml = ModelLoader('daal_lr', trainResult.model)
            ml.save_sk_daal_model()

        return test_accu, test_tpr, test_far, test_report

    def load_saved_model(self, loaded_model):
        # Flatten y
        testLabel = self.y_test.flatten()

        # Begin test timing
        startTime = time.time()

        # Create prediction class
        predictAlg = logistic_regression_prediction(nClasses=2)

        # Make predictions
        predictResultTest = predictAlg.compute(self.X_test, loaded_model)

        # End test timing
        endTime = time.time()

        # Collect statistics
        test_tpr, test_far, test_accu, test_report = collect_statistics(testLabel, predictResultTest.prediction.flatten())

        print("Test (DAAL Logistic Regression) elapsed in %.3f seconds" % (endTime - startTime))
        print("--- Testing Results  ---")
        print("Test accuracy: ", test_accu)
        print("TPR: ", test_tpr)
        print("FAR: ", test_far)
        print(test_report)
        print("------------------------")

        return test_accu, test_tpr, test_far, test_report
