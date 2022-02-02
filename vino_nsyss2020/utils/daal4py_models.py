import numpy as np
from _daal4py import decision_forest_classification_training, decision_forest_classification_prediction
from _daal4py import logistic_regression_training, logistic_regression_prediction
from _daal4py import kdtree_knn_classification_training, kdtree_knn_classification_prediction
from _daal4py import svm_training, kernel_function_linear, svm_prediction


class daal_LR: # DAAL Logistic Regression
    """docstring for Logistic Regression"""
    def __init__(self):
        # Model Definition
        self.nClasses = 2
        self.model = logistic_regression_training(nClasses=self.nClasses, interceptFlag=True)

    def train(self, X_train, y_train):
        # Train the model
        self.trainResult = self.model.compute(X_train, np.array(y_train).reshape((len(y_train), 1)))
        return self.trainResult.model

    def classify(self, data):
        self.predictAlgorithm = logistic_regression_prediction(nClasses=self.nClasses) 
        return self.predictAlgorithm.compute(data, self.trainResult.model).prediction.flatten()



class daal_kNN: # DAAL k Nearest Neighbor 
    """docstring for k Nearest Neighbor """
    def __init__(self, k):
        # Model Definition
        self.nClasses = 2
        self.k=k
        self.model = kdtree_knn_classification_training(nClasses=self.nClasses, k=self.k)

    def train(self, X_train, y_train):
        # Train the model
        self.trainResult = self.model.compute(X_train, np.array(y_train).reshape((len(y_train), 1)))
        return self.trainResult.model

    def classify(self, data):
        self.predictAlgorithm = kdtree_knn_classification_prediction(nClasses=self.nClasses, k=self.k) 
        return self.predictAlgorithm.compute(data, self.trainResult.model).prediction.flatten()


class daal_DF: # DAAL Decision Forest
    """docstring for Decision Forest """
    def __init__(self, n=100, m=10):
        # Model Definition
        self.nClasses = 2
        self.model = decision_forest_classification_training(nClasses=self.nClasses, nTrees=n, maxTreeDepth=m)

    def train(self, X_train, y_train):
        # Train the model
        self.trainResult = self.model.compute(X_train, np.array(y_train).reshape((len(y_train), 1)))
        return self.trainResult.model

    def classify(self, data):
        self.predictAlgorithm = decision_forest_classification_prediction(self.nClasses) 
        return self.predictAlgorithm.compute(data, self.trainResult.model).prediction.flatten()


class daal_SVM: # DAAL Support Vector Machine
    """docstring for Support Vector Machine """
    def __init__(self, C=1.0, kernel='rbf'):
        # Model Definition
        self.nClasses = 2
        self.kern = kernel_function_linear(method='defaultDense')
        self.model = svm_training(nClasses=self.nClasses, C=C, maxIterations=100000, cacheSize=200, kernel=self.kern,
                                accuracyThreshold=1e-2, doShrinking=True)

    def train(self, X_train, y_train):
        # Train the model
        self.trainResult = self.model.compute(X_train, np.array(y_train).reshape((len(y_train), 1)))
        return self.trainResult.model

    def classify(self, data):
        self.predictAlgorithm = svm_prediction(self.nClasses) 
        return self.predictAlgorithm.compute(data, self.trainResult.model).prediction.flatten()


#class MLP: # Multi Layer Perceptron
#    """docstring for Multi Layer Perceptron """
#    def __init__(self, solver='adam', hidden_units=128):
#        # Model Definition
#        model = MLPClassifier(solver='adam', alpha=1e-4, tol=1e-3, max_iter = 100, hidden_layer_sizes=(hidden_units,), early_stopping=True, random_state=42, verbose=True)
#        self.model = model

#    def train(self, X_train, y_train):
#        # Train the model
#        return self.model.fit(X_train, y_train)

#    def classify(self, data):
#        return self.model.predict(data)

