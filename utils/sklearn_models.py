import numpy as np
from sklearn.svm import SVC
from sklearn.neural_network import MLPClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression


class LR: # Logistic Regression
    """docstring for Logistic Regression"""
    def __init__(self):
        # Model Definition
        model = LogisticRegression(random_state=42)
        self.model = model

    def train(self, X_train, y_train):
        # Train the model
        return self.model.fit(X_train, y_train)

    def classify(self, data):
        return self.model.predict(data)


class kNN: # k Nearest Neighbor 
    """docstring for k Nearest Neighbor """
    def __init__(self, n):
        # Model Definition
        model = KNeighborsClassifier(n_neighbors=n)
        self.model = model

    def train(self, X_train, y_train):
        # Train the model
        return self.model.fit(X_train, y_train)

    def classify(self, data):
        return self.model.predict(data)


class RF: # Random Forest
    """docstring for Random Forest """
    def __init__(self, n=100, m=10):
        # Model Definition
        model = RandomForestClassifier(n_estimators=n, max_depth=m, random_state=42, n_jobs = -1, max_features="auto")
        self.model = model

    def train(self, X_train, y_train):
        # Train the model
        return self.model.fit(X_train, y_train)

    def classify(self, data):
        return self.model.predict(data)


class SVM: # Support Vector Machine
    """docstring for Support Vector Machine """
    def __init__(self, C=1.0, kernel='rbf'):
        # Model Definition
        model = SVC(C=C, kernel=kernel, decision_function_shape='ovr', tol=1e-2, max_iter = 100000, random_state=42, verbose=True)
        self.model = model

    def train(self, X_train, y_train):
        # Train the model
        return self.model.fit(X_train, y_train)

    def classify(self, data):
        return self.model.predict(data)


class MLP: # Multi Layer Perceptron
    """docstring for Multi Layer Perceptron """
    def __init__(self, solver='adam', hidden_units=128):
        # Model Definition
        model = MLPClassifier(solver='adam', alpha=1e-4, tol=1e-3, max_iter = 100, hidden_layer_sizes=(hidden_units,), early_stopping=True, random_state=42, verbose=True)
        self.model = model

    def train(self, X_train, y_train):
        # Train the model
        return self.model.fit(X_train, y_train)

    def classify(self, data):
        return self.model.predict(data)

